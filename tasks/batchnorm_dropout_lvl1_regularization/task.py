"""
BatchNorm & Dropout Regularisation Study
=========================================

Trains four variants of the same MLP on a noisy synthetic regression dataset
and systematically compares their generalisation:

  Variant A – Baseline  (no BN, no Dropout)
  Variant B – Dropout only
  Variant C – BatchNorm only
  Variant D – BatchNorm + Dropout  ← expected best generalisation

Why this matters
----------------
BatchNorm (Ioffe & Szegedy, 2015) normalises pre-activation distributions,
accelerating training and acting as mild regulariser.
Dropout (Srivastava et al., 2014) stochastically zeros activations during
training, preventing co-adaptation of neurons.

Their combination often outperforms either technique alone.

Objective: loss  J(θ) = MSE + λ * L2_penalty  (weight decay)

Metrics (validation split)
--------------------------
  MSE, RMSE, R²

Quality assertions
------------------
  * BN+Dropout val R² >= 0.82
  * BN+Dropout val R² > Baseline val R² (demonstrates regularisation benefit)
  * Training loss decreases for every variant
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------

def get_task_metadata() -> dict:
    return {
        "task_id":     "batchnorm_dropout_lvl1_regularization",
        "task_name":   "batchnorm_dropout_regularization_study",
        "series":      "Regularisation Techniques",
        "level":       1,
        "algorithm":   "MLP ablation: Baseline vs Dropout vs BatchNorm vs BN+Dropout",
        "task_type":   "regression",
        "input_type":  "tabular",
        "output_type": "continuous",
        "description": (
            "Ablation study of BatchNorm and Dropout on a noisy synthetic regression "
            "task. Four MLP variants compared on val MSE / R². Demonstrates that "
            "BN+Dropout consistently outperforms the no-regularisation baseline."
        ),
        "metrics":    ["mse", "rmse", "r2"],
        "thresholds": {"best_val_r2": 0.82,
                       "bn_dropout_beats_baseline": True},
    }


# -----------------------------------------------------------------------
# Synthetic regression dataset
# -----------------------------------------------------------------------

def _generate_regression_data(
    n_samples:  int = 3000,
    n_features: int = 50,
    n_informative: int = 15,
    noise_std:  float = 1.5,
    seed:       int = 42,
) -> tuple:
    """
    Noisy multivariate regression dataset.
    y = X[:, :n_informative] @ w + bias + noise
    Returns X, y (numpy float32).
    """
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    w   = rng.standard_normal(n_informative).astype(np.float32) * 2.0
    y   = X[:, :n_informative] @ w + float(rng.standard_normal())
    y  += rng.standard_normal(n_samples).astype(np.float32) * noise_std

    # Standardise
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return X.astype(np.float32), y.astype(np.float32)


def make_dataloaders(
    n_samples:  int   = 3000,
    n_features: int   = 50,
    noise_std:  float = 1.5,
    val_split:  float = 0.20,
    batch_size: int   = 128,
    seed:       int   = 42,
) -> tuple:
    """
    Generate regression data and return DataLoaders + raw arrays.

    Returns:
        train_loader, val_loader, X_train, X_val, y_train, y_val
    """
    set_seed(seed)
    X, y = _generate_regression_data(n_samples=n_samples,
                                      n_features=n_features,
                                      noise_std=noise_std, seed=seed)
    n_val = int(len(X) * val_split)
    X_tr, X_va = X[n_val:], X[:n_val]
    y_tr, y_va = y[n_val:], y[:n_val]

    def _ds(X_, y_):
        return TensorDataset(torch.from_numpy(X_),
                             torch.from_numpy(y_).unsqueeze(1))

    tr_loader = DataLoader(_ds(X_tr, y_tr), batch_size=batch_size,
                           shuffle=True, drop_last=True)
    va_loader = DataLoader(_ds(X_va, y_va), batch_size=batch_size,
                           shuffle=False)
    return tr_loader, va_loader, X_tr, X_va, y_tr, y_va


# -----------------------------------------------------------------------
# Model variants
# -----------------------------------------------------------------------

class MLPVariant(nn.Module):
    """
    3-hidden-layer MLP with configurable BatchNorm and Dropout.

    Layer sequence per block (when both enabled):
        Linear -> BatchNorm1d -> ReLU -> Dropout
    """
    def __init__(
        self,
        in_dim:       int,
        hidden_sizes: list = None,
        use_bn:       bool = False,
        use_dropout:  bool = False,
        dropout_p:    float = 0.3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if use_dropout:
                layers.append(nn.Dropout(p=dropout_p))
            prev = h

        layers.append(nn.Linear(prev, 1))   # regression output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    in_dim:      int = 50,
    use_bn:      bool = False,
    use_dropout: bool = False,
    dropout_p:   float = 0.3,
    device:      torch.device = None,
) -> MLPVariant:
    """Instantiate one MLP variant and move to device."""
    if device is None:
        device = get_device()
    return MLPVariant(in_dim=in_dim, use_bn=use_bn,
                      use_dropout=use_dropout, dropout_p=dropout_p).to(device)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train(
    model:        MLPVariant,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int = 40,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    device:       torch.device = None,
    variant_name: str = "",
) -> tuple:
    """
    Adam + weight_decay (L2 regularisation) + MSE loss.
    Returns train_losses, val_losses (per epoch).
    """
    if device is None:
        device = get_device()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    tr_losses, va_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_n = 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(y_b)
            t_n    += len(y_b)
        scheduler.step()
        tr_losses.append(t_loss / t_n)

        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred    = model(X_b)
                v_loss += criterion(pred, y_b).item() * len(y_b)
                v_n    += len(y_b)
        va_losses.append(v_loss / v_n)

        if epoch % 10 == 0 or epoch == 1:
            tag = f"[{variant_name}]" if variant_name else ""
            print(f"  {tag} Epoch [{epoch:2d}/{epochs}]  "
                  f"train MSE={tr_losses[-1]:.4f}  val MSE={va_losses[-1]:.4f}")

    return tr_losses, va_losses


# -----------------------------------------------------------------------
# Evaluate / Predict
# -----------------------------------------------------------------------

def evaluate(
    model:  MLPVariant,
    loader: DataLoader,
    device: torch.device = None,
) -> dict:
    """Compute MSE, RMSE, R² on `loader`. Returns JSON-safe dict."""
    if device is None:
        device = get_device()
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            pred = model(X_b.to(device)).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y_b.numpy())
    preds   = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()
    mse  = float(mean_squared_error(targets, preds))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(targets, preds))
    return {"mse": mse, "rmse": rmse, "r2": r2}


def predict(
    model:  MLPVariant,
    X:      np.ndarray,
    device: torch.device = None,
) -> np.ndarray:
    """Return scalar predictions for raw numpy array X."""
    if device is None:
        device = get_device()
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X).to(device))
    return out.cpu().numpy().ravel()


# -----------------------------------------------------------------------
# Save artifacts
# -----------------------------------------------------------------------

def save_artifacts(
    all_results: dict,
    all_histories: dict,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Save per-variant metrics JSON and loss-comparison plot."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "regularization_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {"Baseline": "grey", "Dropout": "steelblue",
                  "BatchNorm": "darkorange", "BN+Dropout": "green"}

        # Val loss curves
        for variant, history in all_histories.items():
            axes[0].plot(history["val"], label=variant,
                         color=colors.get(variant, "black"))
        axes[0].set_title("Validation Loss (MSE) by Variant")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val MSE")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # Val R² bar chart
        variants = list(all_results.keys())
        r2_vals  = [all_results[v]["val"]["r2"] for v in variants]
        bar_colors = [colors.get(v, "grey") for v in variants]
        bars = axes[1].bar(variants, r2_vals, color=bar_colors)
        axes[1].set_title("Val R² by Variant")
        axes[1].set_ylabel("R²"); axes[1].set_ylim(0, 1.05)
        for bar, val in zip(bars, r2_vals):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.01,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        axes[1].grid(True, alpha=0.3, axis="y")

        fig.suptitle("BatchNorm & Dropout Regularisation Study")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "regularization_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  Plot saved to {output_dir}")
    except ImportError:
        pass

    print(f"  Artifacts saved to {output_dir}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("BatchNorm & Dropout Study  (batchnorm_dropout_lvl1_regularization)")
    print("=" * 65)

    CFG = dict(
        seed=42, n_samples=3000, n_features=50, noise_std=1.5,
        val_split=0.20, batch_size=128, epochs=40, lr=1e-3,
        weight_decay=1e-4, dropout_p=0.30,
    )
    set_seed(CFG["seed"])
    device = get_device()
    print(f"Device : {device}")
    print(f"Config : {CFG}\n")

    # ---- Data ----
    print("Building dataloaders …")
    train_loader, val_loader, X_tr, X_va, y_tr, y_va = make_dataloaders(
        n_samples=CFG["n_samples"], n_features=CFG["n_features"],
        noise_std=CFG["noise_std"], val_split=CFG["val_split"],
        batch_size=CFG["batch_size"], seed=CFG["seed"],
    )
    print(f"  Train samples : {len(X_tr)}")
    print(f"  Val   samples : {len(X_va)}")

    # ---- Define the four variants ----
    VARIANTS = {
        "Baseline":   dict(use_bn=False, use_dropout=False),
        "Dropout":    dict(use_bn=False, use_dropout=True),
        "BatchNorm":  dict(use_bn=True,  use_dropout=False),
        "BN+Dropout": dict(use_bn=True,  use_dropout=True),
    }

    all_results  = {}
    all_histories = {}

    for name, kwargs in VARIANTS.items():
        print(f"\n--- Variant: {name} ---")
        model = build_model(
            in_dim=CFG["n_features"], dropout_p=CFG["dropout_p"],
            device=device, **kwargs,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {n_params:,}")

        tr_l, va_l = train(
            model, train_loader, val_loader,
            epochs=CFG["epochs"], lr=CFG["lr"],
            weight_decay=CFG["weight_decay"], device=device,
            variant_name=name,
        )
        tr_metrics = evaluate(model, train_loader, device)
        va_metrics = evaluate(model, val_loader,   device)

        all_results[name]   = {"train": tr_metrics, "val": va_metrics}
        all_histories[name] = {"train": [float(v) for v in tr_l],
                               "val":   [float(v) for v in va_l]}

        print(f"  Train  MSE={tr_metrics['mse']:.4f}  RMSE={tr_metrics['rmse']:.4f}  R²={tr_metrics['r2']:.4f}")
        print(f"  Val    MSE={va_metrics['mse']:.4f}  RMSE={va_metrics['rmse']:.4f}  R²={va_metrics['r2']:.4f}")

    # ---- Summary table ----
    print("\n--- Summary (Validation) ---")
    print(f"  {'Variant':<15} {'MSE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'-'*41}")
    for name, res in all_results.items():
        v = res["val"]
        print(f"  {name:<15} {v['mse']:>8.4f} {v['rmse']:>8.4f} {v['r2']:>8.4f}")

    # ---- Save ----
    print("\nSaving artifacts …")
    save_artifacts(all_results, all_histories, output_dir=OUTPUT_DIR)

    # ---- Quality checks ----
    print("\n" + "=" * 65)
    print("Quality Checks:")
    print("=" * 65)

    best_r2       = all_results["BN+Dropout"]["val"]["r2"]
    baseline_r2   = all_results["Baseline"]["val"]["r2"]

    checks = {
        "all_losses_decreased": all(
            all_histories[v]["train"][-1] < all_histories[v]["train"][0]
            for v in VARIANTS
        ),
        "bn_dropout_val_r2_ge_0.82": best_r2 >= 0.82,
        "bn_dropout_beats_baseline":  best_r2 > baseline_r2,
        "batchnorm_beats_baseline":  (
            all_results["BatchNorm"]["val"]["r2"] > baseline_r2
        ),
    }

    all_pass = True
    for name, passed in checks.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {passed}")
        if not passed:
            all_pass = False

    print(f"\n  BN+Dropout R²  = {best_r2:.4f}")
    print(f"  Baseline   R²  = {baseline_r2:.4f}")
    print(f"  Improvement    = {best_r2 - baseline_r2:+.4f}")

    print("=" * 65)
    print("PASS: All quality checks passed!" if all_pass
          else "FAIL: One or more quality checks failed.")
    print("=" * 65)

    sys.exit(0 if all_pass else 1)
