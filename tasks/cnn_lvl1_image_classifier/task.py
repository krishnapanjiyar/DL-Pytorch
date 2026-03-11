"""
CNN Image Classifier Task
=========================

Implements a Convolutional Neural Network (CNN) for multi-class image
classification on a synthetic dataset of simple geometric shape images
(circles, squares, triangles, crosses).

Architecture:
    Input (1x32x32) -> Conv2d+BN+ReLU+MaxPool (x3) -> FC layers -> Softmax

Loss   : CrossEntropyLoss
Metrics: Accuracy, macro Precision, Recall, F1-score
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# -----------------------------------------------------------------------
# Output directory
# -----------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Seeds / device
# -----------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return GPU device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

def get_task_metadata() -> dict:
    """Return task metadata following pytorch_task_v1 protocol."""
    return {
        "task_id":    "cnn_lvl1_image_classifier",
        "task_name":  "cnn_image_classifier",
        "series":     "Convolutional Neural Networks",
        "level":      1,
        "algorithm":  "CNN (Conv2d + BatchNorm + MaxPool + FC)",
        "task_type":  "multiclass_classification",
        "input_type": "image",
        "output_type": "class_label",
        "num_classes": 4,
        "image_size": 32,
        "description": (
            "Multi-class image classification of synthetic grayscale geometric "
            "shapes (circle, square, triangle, cross) using a 3-block CNN with "
            "BatchNorm and MaxPooling, trained with SGD + momentum + cosine LR."
        ),
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "thresholds": {"val_accuracy": 0.80, "val_f1": 0.78},
    }


# -----------------------------------------------------------------------
# Synthetic image generation
# -----------------------------------------------------------------------

def _draw_circle(img: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    H, W = img.shape
    for y in range(H):
        for x in range(W):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                img[y, x] = 1.0
    return img


def _draw_square(img: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    H, W = img.shape
    x1, x2 = max(0, cx - half), min(W, cx + half)
    y1, y2 = max(0, cy - half), min(H, cy + half)
    img[y1:y2, x1:x2] = 1.0
    return img


def _draw_triangle(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    H, W = img.shape
    top    = cy - size
    bottom = cy + size
    for y in range(max(0, top), min(H, bottom + 1)):
        t = (y - top) / max(1, bottom - top)
        half_w = int(size * t)
        x1 = max(0, cx - half_w)
        x2 = min(W, cx + half_w + 1)
        img[y, x1:x2] = 1.0
    return img


def _draw_cross(img: np.ndarray, cx: int, cy: int, size: int, thick: int = 2) -> np.ndarray:
    H, W = img.shape
    # Horizontal bar
    y1, y2 = max(0, cy - thick), min(H, cy + thick + 1)
    x1, x2 = max(0, cx - size), min(W, cx + size + 1)
    img[y1:y2, x1:x2] = 1.0
    # Vertical bar
    y1, y2 = max(0, cy - size), min(H, cy + size + 1)
    x1, x2 = max(0, cx - thick), min(W, cx + thick + 1)
    img[y1:y2, x1:x2] = 1.0
    return img


def _make_shape_image(label: int, img_size: int = 32,
                      rng: np.random.Generator = None) -> np.ndarray:
    """Draw shape `label` on a blank image with random position/scale."""
    if rng is None:
        rng = np.random.default_rng(42)
    img = np.zeros((img_size, img_size), dtype=np.float32)
    cx  = rng.integers(10, img_size - 10)
    cy  = rng.integers(10, img_size - 10)
    sz  = rng.integers(5, 10)
    if label == 0:
        img = _draw_circle(img, cx, cy, sz)
    elif label == 1:
        img = _draw_square(img, cx, cy, sz)
    elif label == 2:
        img = _draw_triangle(img, cx, cy, sz)
    else:
        img = _draw_cross(img, cx, cy, sz, thick=2)
    # Add mild Gaussian noise
    img += rng.normal(0, 0.05, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def _generate_dataset(n_samples: int, img_size: int = 32,
                       num_classes: int = 4, seed: int = 42):
    """Return (X, y) arrays. X shape: [N, 1, H, W], y shape: [N]."""
    rng = np.random.default_rng(seed)
    per_class = n_samples // num_classes
    images, labels = [], []
    for cls in range(num_classes):
        for _ in range(per_class):
            img = _make_shape_image(cls, img_size, rng)
            images.append(img[np.newaxis, :, :])   # add channel dim
            labels.append(cls)
    X = np.stack(images, axis=0).astype(np.float32)   # [N, 1, H, W]
    y = np.array(labels, dtype=np.int64)
    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# -----------------------------------------------------------------------
# DataLoaders
# -----------------------------------------------------------------------

def make_dataloaders(
    batch_size: int = 64,
    val_split:  float = 0.2,
    num_samples: int = 2000,
    img_size:    int = 32,
    num_classes: int = 4,
    seed:        int = 42,
):
    """
    Generate synthetic shape images and return DataLoaders.

    Returns:
        train_loader, val_loader, X_train, X_val, y_train, y_val
    """
    set_seed(seed)
    X, y = _generate_dataset(num_samples, img_size, num_classes, seed=seed)

    n_val   = int(num_samples * val_split)
    X_train, X_val = X[n_val:], X[:n_val]
    y_train, y_val = y[n_val:], y[:n_val]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNClassifier(nn.Module):
    """
    3-block CNN for grayscale image classification.

    Input : [B, 1, 32, 32]
    Block1: Conv(1->32) + BN + ReLU + MaxPool  -> [B, 32, 16, 16]
    Block2: Conv(32->64) + BN + ReLU + MaxPool -> [B, 64,  8,  8]
    Block3: Conv(64->128) + BN + ReLU + MaxPool-> [B, 128, 4,  4]
    FC1   : 128*4*4 -> 256 + ReLU + Dropout(0.5)
    FC2   : 256 -> num_classes
    """
    def __init__(self, num_classes: int = 4, img_size: int = 32,
                 dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=True),
        )
        fc_in = 128 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int = 4, img_size: int = 32,
                dropout: float = 0.5, device: torch.device = None) -> CNNClassifier:
    """Build and return the CNN model, moved to `device`."""
    if device is None:
        device = get_device()
    model = CNNClassifier(num_classes=num_classes, img_size=img_size,
                          dropout=dropout)
    return model.to(device)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train(
    model:        CNNClassifier,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int = 30,
    lr:           float = 0.05,
    device:       torch.device = None,
) -> tuple:
    """
    Train with SGD + momentum and CosineAnnealingLR.

    Returns:
        train_losses (list), val_losses (list), train_accs (list), val_accs (list)
    """
    if device is None:
        device = get_device()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)
            correct      += (logits.argmax(1) == y_batch).sum().item()
            total        += len(y_batch)
        scheduler.step()
        train_losses.append(running_loss / total)
        train_accs.append(correct / total)

        # --- validate ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                v_loss    += criterion(logits, y_batch).item() * len(y_batch)
                v_correct += (logits.argmax(1) == y_batch).sum().item()
                v_total   += len(y_batch)
        val_losses.append(v_loss / v_total)
        val_accs.append(v_correct / v_total)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] "
                  f"train loss={train_losses[-1]:.4f} acc={train_accs[-1]:.3f} | "
                  f"val loss={val_losses[-1]:.4f} acc={val_accs[-1]:.3f}")

    return train_losses, val_losses, train_accs, val_accs


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate(
    model:  CNNClassifier,
    loader: DataLoader,
    device: torch.device = None,
) -> dict:
    """
    Compute accuracy, macro precision, recall, F1, and confusion matrix.

    Returns a metrics dictionary (all values JSON-serialisable).
    """
    if device is None:
        device = get_device()
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    acc  = float(accuracy_score(all_labels, all_preds))
    prec = float(precision_score(all_labels, all_preds, average="macro",
                                  zero_division=0))
    rec  = float(recall_score(all_labels, all_preds, average="macro",
                               zero_division=0))
    f1   = float(f1_score(all_labels, all_preds, average="macro",
                           zero_division=0))
    cm   = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "confusion_matrix": cm,
    }


# -----------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------

def predict(
    model:  CNNClassifier,
    X:      np.ndarray,
    device: torch.device = None,
) -> np.ndarray:
    """Return class predictions for raw numpy image array X [N,1,H,W]."""
    if device is None:
        device = get_device()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
    return logits.argmax(1).cpu().numpy()


# -----------------------------------------------------------------------
# Save artifacts
# -----------------------------------------------------------------------

def save_artifacts(
    model:        CNNClassifier,
    train_losses: list,
    val_losses:   list,
    train_accs:   list,
    val_accs:     list,
    train_metrics: dict,
    val_metrics:   dict,
    output_dir:    str = OUTPUT_DIR,
) -> None:
    """Save model weights, training history JSON, and loss/accuracy plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Model checkpoint
    torch.save(model.state_dict(),
               os.path.join(output_dir, "cnn_classifier.pth"))

    # Training history
    history = {
        "train_losses": [float(v) for v in train_losses],
        "val_losses":   [float(v) for v in val_losses],
        "train_accs":   [float(v) for v in train_accs],
        "val_accs":     [float(v) for v in val_accs],
    }
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Metrics summary
    metrics = {"train": train_metrics, "val": val_metrics}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Loss + accuracy curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, len(train_losses) + 1)

        ax1.plot(epochs_range, train_losses, label="Train")
        ax1.plot(epochs_range, val_losses,   label="Val")
        ax1.set_title("Loss (CrossEntropy)")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, train_accs, label="Train")
        ax2.plot(epochs_range, val_accs,   label="Val")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.suptitle("CNN Image Classifier — Training Curves")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "cnn_training_curves.png"), dpi=150)
        plt.close(fig)
        print(f"  Plots saved to {output_dir}")
    except ImportError:
        print("  matplotlib not available — skipping plots")

    print(f"  Artifacts saved to {output_dir}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CNN Image Classifier  (cnn_lvl1_image_classifier)")
    print("=" * 60)

    # ---- Config ----
    CFG = dict(
        seed=42, num_classes=4, img_size=32, num_samples=2400,
        val_split=0.20, batch_size=64, epochs=30, lr=0.05, dropout=0.5,
    )
    set_seed(CFG["seed"])
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {CFG}\n")

    # ---- Data ----
    print("Building dataloaders …")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        batch_size=CFG["batch_size"], val_split=CFG["val_split"],
        num_samples=CFG["num_samples"], img_size=CFG["img_size"],
        num_classes=CFG["num_classes"], seed=CFG["seed"],
    )
    print(f"  Train samples : {len(X_train)}")
    print(f"  Val   samples : {len(X_val)}")

    # ---- Model ----
    print("\nBuilding model …")
    model = build_model(num_classes=CFG["num_classes"],
                        img_size=CFG["img_size"],
                        dropout=CFG["dropout"], device=device)
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

    print("\n  Train metrics:")
    for k, v in train_metrics.items():
        if k != "confusion_matrix":
            print(f"    {k:12s}: {v:.4f}")

    print("\n  Val metrics:")
    for k, v in val_metrics.items():
        if k != "confusion_matrix":
            print(f"    {k:12s}: {v:.4f}")

    print("\n  Confusion matrix (val):")
    for row in val_metrics["confusion_matrix"]:
        print("   ", row)

    # ---- Artifacts ----
    print("\nSaving artifacts …")
    save_artifacts(model, train_losses, val_losses, train_accs, val_accs,
                   train_metrics, val_metrics, output_dir=OUTPUT_DIR)

    # ---- Quality assertions ----
    print("\n" + "=" * 60)
    print("Quality Checks:")
    print("=" * 60)

    checks = {}
    checks["train_loss_decreased"] = train_losses[-1] < train_losses[0]
    checks["val_accuracy_ge_0.80"] = val_metrics["accuracy"] >= 0.80
    checks["val_f1_ge_0.78"]       = val_metrics["f1"]       >= 0.78
    checks["no_severe_overfit"]    = (train_metrics["accuracy"] -
                                       val_metrics["accuracy"]) < 0.25

    all_pass = True
    for name, passed in checks.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {passed}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: One or more quality checks failed.")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)
