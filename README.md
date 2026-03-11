# DL PyTorch Tasks

Four new tasks added to the CoderGym MLtasks collection, following the `pytorch_task_v1` protocol. Each task is fully self-contained, requires no external dataset downloads, and is self-verifiable via `sys.exit(exit_code)`.

---

## Tasks

| Task ID | Description | Dataset | Architecture |
|---|---|---|---|
| `cnn_lvl1_image_classifier` | Multi-class image classification (4 shape classes) | Synthetic 32×32 grayscale images — circle, square, triangle, cross (2,400 samples) | Conv2d + BatchNorm2d + MaxPool2d × 3 → FC head |
| `rnn_lvl1_text_sentiment` | Sentiment classification — Negative / Neutral / Positive | Synthetic variable-length token sequences, class-specific vocabulary (3,000 samples) | Embedding → BiLSTM (2 layers) → Masked Mean Pool → Dropout → FC |
| `transfer_lvl1_feature_extraction` | Transfer learning: pre-train on source task, fine-tune on target task, compare vs. scratch | Synthetic Gaussian blobs — source: 512-D 10-class (4,000 samples); target: 512-D 4-class (800 samples) | 3-block MLP backbone (512→256→128→64) + swappable FC head |
| `batchnorm_dropout_lvl1_regularization` | Ablation study: compare 4 MLP variants to show regularisation benefit on regression | Synthetic noisy tabular data — 50 features, 15 informative, noise std = 1.5 (3,000 samples) | Linear (50→256→128→64→1) with 4 variants: Baseline / Dropout / BatchNorm / BN+Dropout |

### Training Details

| Task ID | Optimizer | Loss | Pass Criteria |
|---|---|---|---|
| `cnn_lvl1_image_classifier` | SGD + Nesterov momentum + CosineAnnealingLR | CrossEntropyLoss | Val accuracy ≥ 0.80, macro F1 ≥ 0.78 |
| `rnn_lvl1_text_sentiment` | Adam + ReduceLROnPlateau + grad clip | CrossEntropyLoss | Val accuracy ≥ 0.78, macro F1 ≥ 0.75 |
| `transfer_lvl1_feature_extraction` | Adam + CosineAnnealingLR (per phase) | CrossEntropyLoss | Transfer val accuracy ≥ 0.75, transfer ≥ scratch − 2% |
| `batchnorm_dropout_lvl1_regularization` | Adam + weight decay (L2) + StepLR | MSELoss | BN+Dropout val R² ≥ 0.82, BN+Dropout R² > Baseline R² |

---

## How to Run

Each task is a single self-contained script. Run from the project root:

```
python MLtasks/tasks/<task_id>/task.py
```

Each script will:
1. Generate the dataset in memory
2. Train the model and print metrics per epoch
3. Evaluate on train and validation splits
4. Save model weights, metrics JSON, and plots to `MLtasks/tasks/<task_id>/output/`
5. Assert quality thresholds and exit with code `0` (PASS) or `1` (FAIL)

---

## Dependencies

- Python 3.9+
- PyTorch
- NumPy
- scikit-learn
- Matplotlib

```
pip install torch scikit-learn numpy matplotlib
```
