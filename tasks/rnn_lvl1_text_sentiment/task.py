"""
RNN / LSTM Text Sentiment Classification Task
=============================================

Trains a bidirectional LSTM on variable-length token sequences to classify
synthetic 3-class sentiment (negative / neutral / positive).

Dataset
-------
Synthetic: each sample is a sequence of integer token IDs drawn from
class-specific vocabulary buckets, mimicking word-index representations.

No external data required; everything is generated on the fly.

Architecture
------------
Embedding -> BiLSTM (2 layers) -> mean-pool over time -> FC head -> Softmax

Optimizer : Adam with ReduceLROnPlateau
Loss      : CrossEntropyLoss
Metrics   : Accuracy, macro F1, per-class F1
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -----------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

def get_task_metadata() -> dict:
    """Return pytorch_task_v1 metadata."""
    return {
        "task_id":     "rnn_lvl1_text_sentiment",
        "task_name":   "rnn_text_sentiment_classifier",
        "series":      "Recurrent Neural Networks",
        "level":       1,
        "algorithm":   "Bidirectional LSTM with mean-pooling",
        "task_type":   "multiclass_classification",
        "input_type":  "sequence",
        "output_type": "class_label",
        "num_classes": 3,
        "description": (
            "3-class sentiment classification on synthetic token sequences. "
            "Negative / Neutral / Positive labels. BiLSTM encoder, Adam + "
            "ReduceLROnPlateau, CrossEntropyLoss."
        ),
        "metrics":    ["accuracy", "macro_f1", "per_class_f1"],
        "thresholds": {"val_accuracy": 0.78, "val_macro_f1": 0.75},
    }


# -----------------------------------------------------------------------
# Synthetic text dataset
# -----------------------------------------------------------------------

VOCAB_SIZE   = 500
SEQ_LEN_MIN  = 10
SEQ_LEN_MAX  = 40
NUM_CLASSES  = 3
PAD_IDX      = 0

# Each class "owns" a slab of the vocabulary – signal tokens
_CLASS_VOCAB_SIZE  = 100
_CLASS_VOCAB_START = [1, 101, 201]          # offsets per class
_NOISE_VOCAB_START = 301                    # shared noise tokens


def _make_sequence(label: int, rng: np.random.Generator) -> list:
    """
    Generate a token-id sequence for `label`.
    ~60% tokens are class-specific; ~40% are shared noise.
    """
    length = rng.integers(SEQ_LEN_MIN, SEQ_LEN_MAX + 1)
    start  = _CLASS_VOCAB_START[label]
    tokens = []
    for _ in range(length):
        if rng.random() < 0.60:
            tok = int(rng.integers(start, start + _CLASS_VOCAB_SIZE))
        else:
            tok = int(rng.integers(_NOISE_VOCAB_START,
                                   _NOISE_VOCAB_START + (_VOCAB_SIZE - _NOISE_VOCAB_START)))
        tokens.append(tok)
    return tokens


class SentimentDataset(Dataset):
    """Variable-length integer sequence sentiment dataset."""

    def __init__(self, sequences: list, labels: list):
        self.sequences = [torch.tensor(s, dtype=torch.long) for s in sequences]
        self.labels    = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def _collate_fn(batch):
    """Pad sequences to max length in the batch; return (padded, lengths, labels)."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    labels  = torch.stack(labels)
    return padded, lengths, labels


def make_dataloaders(
    num_samples: int = 3000,
    val_split:   float = 0.20,
    batch_size:  int = 64,
    seed:        int = 42,
):
    """
    Generate a synthetic text-sentiment dataset and return DataLoaders.

    Returns:
        train_loader, val_loader, train_seqs, val_seqs, train_labels, val_labels
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    per_class = num_samples // NUM_CLASSES
    all_seqs, all_labels = [], []
    for cls in range(NUM_CLASSES):
        for _ in range(per_class):
            all_seqs.append(_make_sequence(cls, rng))
            all_labels.append(cls)

    # Shuffle
    idx = rng.permutation(len(all_seqs)).tolist()
    all_seqs   = [all_seqs[i]   for i in idx]
    all_labels = [all_labels[i] for i in idx]

    n_val   = int(len(all_seqs) * val_split)
    val_seqs,   val_labels   = all_seqs[:n_val],   all_labels[:n_val]
    train_seqs, train_labels = all_seqs[n_val:],   all_labels[n_val:]

    train_ds = SentimentDataset(train_seqs, train_labels)
    val_ds   = SentimentDataset(val_seqs,   val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=_collate_fn)

    return train_loader, val_loader, train_seqs, val_seqs, train_labels, val_labels


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class BiLSTMSentiment(nn.Module):
    """
    Bidirectional 2-layer LSTM with mean-pool sentence representation.

    Forward:
        tokens [B, T] -> Embedding [B, T, E]
        -> pack -> BiLSTM [B, T, 2H] -> unpack -> mean pool [B, 2H]
        -> Dropout -> FC -> logits [B, num_classes]
    """

    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE + 1,   # +1 for PAD_IDX=0
        embed_dim:   int = 64,
        hidden_size: int = 128,
        num_layers:  int = 2,
        num_classes: int = NUM_CLASSES,
        dropout:     float = 0.3,
        pad_idx:     int = PAD_IDX,
    ):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, tokens: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(tokens))   # [B, T, E]

        # Pack padded sequence for efficient RNN computation
        packed = pack_padded_sequence(embedded, lengths.cpu(),
                                      batch_first=True,
                                      enforce_sorted=False)
        output_packed, _ = self.lstm(packed)
        output, _        = pad_packed_sequence(output_packed, batch_first=True)
        # output: [B, T, 2H]

        # Mean-pool over valid (non-padded) time steps
        mask = (tokens != self.pad_idx).unsqueeze(-1).float()   # [B, T, 1]
        sum_out = (output * mask).sum(dim=1)                    # [B, 2H]
        count   = mask.sum(dim=1).clamp(min=1)                  # [B, 1]
        pooled  = sum_out / count                               # [B, 2H]

        return self.fc(self.dropout(pooled))                    # [B, C]


def build_model(
    vocab_size:  int = VOCAB_SIZE + 1,
    embed_dim:   int = 64,
    hidden_size: int = 128,
    num_layers:  int = 2,
    num_classes: int = NUM_CLASSES,
    dropout:     float = 0.3,
    device:      torch.device = None,
) -> BiLSTMSentiment:
    """Instantiate and return the BiLSTM model on `device`."""
    if device is None:
        device = get_device()
    model = BiLSTMSentiment(vocab_size=vocab_size, embed_dim=embed_dim,
                             hidden_size=hidden_size, num_layers=num_layers,
                             num_classes=num_classes, dropout=dropout)
    return model.to(device)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train(
    model:        BiLSTMSentiment,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int = 20,
    lr:           float = 1e-3,
    device:       torch.device = None,
) -> tuple:
    """
    Train with Adam and ReduceLROnPlateau scheduler.

    Returns:
        train_losses, val_losses, train_accs, val_accs
    """
    if device is None:
        device = get_device()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=False
    )

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for tokens, lengths, labels in train_loader:
            tokens, lengths, labels = (tokens.to(device),
                                       lengths.to(device),
                                       labels.to(device))
            optimizer.zero_grad()
            logits = model(tokens, lengths)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss    += loss.item() * len(labels)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += len(labels)

        train_losses.append(t_loss / t_total)
        train_accs.append(t_correct / t_total)

        # ---- validate ----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for tokens, lengths, labels in val_loader:
                tokens, lengths, labels = (tokens.to(device),
                                           lengths.to(device),
                                           labels.to(device))
                logits  = model(tokens, lengths)
                v_loss += criterion(logits, labels).item() * len(labels)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += len(labels)

        val_losses.append(v_loss / v_total)
        val_accs.append(v_correct / v_total)
        scheduler.step(val_losses[-1])

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:2d}/{epochs}] "
                  f"train loss={train_losses[-1]:.4f} acc={train_accs[-1]:.3f} | "
                  f"val loss={val_losses[-1]:.4f} acc={val_accs[-1]:.3f}")

    return train_losses, val_losses, train_accs, val_accs


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate(
    model:  BiLSTMSentiment,
    loader: DataLoader,
    device: torch.device = None,
) -> dict:
    """
    Return accuracy, macro F1, and per-class F1 on `loader`.
    All values are Python floats (JSON-serialisable).
    """
    if device is None:
        device = get_device()
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            tokens, lengths = tokens.to(device), lengths.to(device)
            preds = model(tokens, lengths).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc       = float(accuracy_score(all_labels, all_preds))
    macro_f1  = float(f1_score(all_labels, all_preds, average="macro",
                                zero_division=0))
    per_class = f1_score(all_labels, all_preds, average=None,
                          zero_division=0).tolist()
    report    = classification_report(all_labels, all_preds,
                                      target_names=["Negative","Neutral","Positive"],
                                      zero_division=0)
    return {
        "accuracy":      acc,
        "macro_f1":      macro_f1,
        "per_class_f1":  [float(v) for v in per_class],
        "report":        report,
    }


# -----------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------

def predict(
    model:     BiLSTMSentiment,
    sequences: list,
    device:    torch.device = None,
) -> np.ndarray:
    """
    Predict class labels for a list of raw token-id sequences.
    Returns numpy array of integer class labels.
    """
    if device is None:
        device = get_device()
    tensors = [torch.tensor(s, dtype=torch.long) for s in sequences]
    lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long)
    padded  = pad_sequence(tensors, batch_first=True, padding_value=PAD_IDX)
    model.eval()
    with torch.no_grad():
        logits = model(padded.to(device), lengths.to(device))
    return logits.argmax(1).cpu().numpy()


# -----------------------------------------------------------------------
# Save artifacts
# -----------------------------------------------------------------------

def save_artifacts(
    model:         BiLSTMSentiment,
    train_losses:  list,
    val_losses:    list,
    train_accs:    list,
    val_accs:      list,
    train_metrics: dict,
    val_metrics:   dict,
    output_dir:    str = OUTPUT_DIR,
) -> None:
    """Persist model, training history, and plots."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(),
               os.path.join(output_dir, "bilstm_sentiment.pth"))

    history = {
        "train_losses": [float(v) for v in train_losses],
        "val_losses":   [float(v) for v in val_losses],
        "train_accs":   [float(v) for v in train_accs],
        "val_accs":     [float(v) for v in val_accs],
    }
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    safe_train = {k: v for k, v in train_metrics.items() if k != "report"}
    safe_val   = {k: v for k, v in val_metrics.items()   if k != "report"}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"train": safe_train, "val": safe_val}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ep = range(1, len(train_losses) + 1)

        ax1.plot(ep, train_losses, label="Train"); ax1.plot(ep, val_losses, label="Val")
        ax1.set_title("CrossEntropy Loss"); ax1.set_xlabel("Epoch")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(ep, train_accs, label="Train"); ax2.plot(ep, val_accs, label="Val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylim(0, 1)
        ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.suptitle("BiLSTM Sentiment — Training Curves")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "rnn_training_curves.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    print(f"  Artifacts saved to {output_dir}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("BiLSTM Text Sentiment  (rnn_lvl1_text_sentiment)")
    print("=" * 60)

    CFG = dict(
        seed=42, num_samples=3000, val_split=0.20, batch_size=64,
        embed_dim=64, hidden_size=128, num_layers=2, dropout=0.3,
        epochs=20, lr=1e-3,
    )
    set_seed(CFG["seed"])
    device = get_device()
    print(f"Device : {device}")
    print(f"Config : {CFG}\n")

    # ---- Data ----
    print("Building dataloaders …")
    (train_loader, val_loader,
     train_seqs, val_seqs,
     train_labels, val_labels) = make_dataloaders(
        num_samples=CFG["num_samples"], val_split=CFG["val_split"],
        batch_size=CFG["batch_size"], seed=CFG["seed"],
    )
    print(f"  Train sequences : {len(train_seqs)}")
    print(f"  Val   sequences : {len(val_seqs)}")

    # ---- Model ----
    print("\nBuilding model …")
    model = build_model(
        embed_dim=CFG["embed_dim"], hidden_size=CFG["hidden_size"],
        num_layers=CFG["num_layers"], dropout=CFG["dropout"], device=device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # ---- Train ----
    print("\nTraining …")
    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader,
        epochs=CFG["epochs"], lr=CFG["lr"], device=device,
    )

    # ---- Evaluate ----
    print("\nEvaluating …")
    train_metrics = evaluate(model, train_loader, device)
    val_metrics   = evaluate(model, val_loader,   device)

    print("\n  Train  accuracy={:.4f}  macro_f1={:.4f}".format(
          train_metrics["accuracy"], train_metrics["macro_f1"]))
    print("  Val    accuracy={:.4f}  macro_f1={:.4f}".format(
          val_metrics["accuracy"], val_metrics["macro_f1"]))
    print("\n  Val Classification Report:")
    print(val_metrics["report"])

    # ---- Artifacts ----
    save_artifacts(model, train_losses, val_losses, train_accs, val_accs,
                   train_metrics, val_metrics, output_dir=OUTPUT_DIR)

    # ---- Quality checks ----
    print("=" * 60)
    print("Quality Checks:")
    print("=" * 60)

    checks = {
        "train_loss_decreased":  train_losses[-1] < train_losses[0],
        "val_accuracy_ge_0.78":  val_metrics["accuracy"] >= 0.78,
        "val_macro_f1_ge_0.75":  val_metrics["macro_f1"] >= 0.75,
        "no_severe_overfit":     (train_metrics["accuracy"] -
                                   val_metrics["accuracy"]) < 0.20,
    }

    all_pass = True
    for name, passed in checks.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {passed}")
        if not passed:
            all_pass = False

    print("=" * 60)
    print("PASS: All quality checks passed!" if all_pass
          else "FAIL: One or more quality checks failed.")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)
