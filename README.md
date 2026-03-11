# DL PyTorch Tasks

Four new tasks added to the CoderGym MLtasks collection, following the `pytorch_task_v1` protocol. Each task is implemented in PyTorch, uses a new dataset or training feature, and exits with `sys.exit(0)` on pass or `sys.exit(1)` on fail.

---

## Tasks

| Task ID | Description | Architecture |
|---|---|---|
| `cnn_lvl1_image_classifier` | Multi-class classification of synthetic geometric shape images | Conv2d + BatchNorm2d + MaxPool2d → FC head |
| `rnn_lvl1_text_sentiment` | 3-class sentiment classification on synthetic token sequences | Bidirectional LSTM → Mean Pool → FC |
| `transfer_lvl1_feature_extraction` | Transfer learning — pre-train backbone, freeze, fine-tune on target task | 3-block MLP backbone + swappable FC head |
| `batchnorm_dropout_lvl1_regularization` | Ablation study comparing 4 MLP regularisation variants on regression | MLP — Baseline / Dropout / BatchNorm / BN+Dropout |

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
