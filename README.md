# DL PyTorch Tasks

Four new tasks added to the CoderGym MLtasks collection, following the `pytorch_task_v1` protocol. Each task is implemented in PyTorch, uses a new dataset or training feature, and exits with `sys.exit(0)` on pass or `sys.exit(1)` on fail.

---

## Tasks

| Task ID | Description | Dataset | Architecture |
|---|---|---|---|
| `cnn_lvl1_image_classifier` | Multi-class classification of geometric shape images | Synthetic 32×32 grayscale images — circle, square, triangle, cross | Conv2d + BatchNorm2d + MaxPool2d → FC head |
| `rnn_lvl1_text_sentiment` | 3-class sentiment classification on token sequences | Synthetic variable-length token sequences (Negative / Neutral / Positive) | Bidirectional LSTM → Mean Pool → FC |
| `transfer_lvl1_feature_extraction` | Pre-train backbone, freeze, fine-tune on target task | Synthetic Gaussian blobs — 10-class source, 4-class target | 3-block MLP backbone + swappable FC head |
| `batchnorm_dropout_lvl1_regularization` | Ablation study comparing 4 MLP regularisation variants | Synthetic noisy tabular data (50 features, 15 informative) | MLP — Baseline / Dropout / BatchNorm / BN+Dropout |

---

## How to Run

```
python MLtasks/tasks/<task_id>/task.py
```

---

## Dependencies

```
pip install torch scikit-learn numpy matplotlib
```
