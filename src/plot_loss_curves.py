"""Epoch-wise training curves for the models that actually train iteratively (MLP, HistGradientBoosting).

Refits both on the same training subset and configuration as src/train_models.py.
Run from the repo root:  python src/plot_loss_curves.py   ->  results/plots/loss_curves.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train_models import PLOTS_DIR, TRAIN_PATH, create_models, prepare_data, training_subset, validate_data

train = training_subset(pd.read_parquet(TRAIN_PATH))
X, y, _ = prepare_data(train, validate_data(train, "Training data"))
models = create_models()

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))

# MLP: training log loss per epoch; early stopping holds out 10% of rows and tracks their accuracy
mlp = models["MLP"].fit(X, y).named_steps["classifier"]
ax = axs[0]
ax.plot(range(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, label="training loss (log loss)")
ax.set(xlabel="epoch", ylabel="training log loss", title=f"MLP: {mlp.n_iter_} epochs (early stopping)")
ax2 = ax.twinx()
ax2.plot(range(1, len(mlp.validation_scores_) + 1), mlp.validation_scores_, c="tab:orange", label="validation accuracy")
ax2.set_ylabel("validation accuracy (10% held-out rows)")
fig.legend(*[sum(z, []) for z in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())],
           loc="upper left", bbox_to_anchor=(0.06, 0.88), fontsize=8)

# HistGradientBoosting: early stopping is on by default for >10k rows; scores are negative log loss
hgb = models["HistGradientBoosting"].fit(X, y).named_steps["classifier"]
ax = axs[1]
ax.plot(-hgb.train_score_, label="training log loss")
ax.plot(-hgb.validation_score_, label="validation log loss")
ax.set(xlabel="boosting iteration", ylabel="log loss",
       title=f"HistGradientBoosting: {hgb.n_iter_} iterations (early stopping: {hgb.do_early_stopping_})")
ax.legend(fontsize=8)

fig.tight_layout()
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOTS_DIR / "loss_curves.png", dpi=130)
print(f"MLP epochs {mlp.n_iter_}, final train loss {mlp.loss_curve_[-1]:.4f}, best val acc {max(mlp.validation_scores_):.4f}")
print(f"HGB iterations {hgb.n_iter_}, final train/val log loss {-hgb.train_score_[-1]:.4f} / {-hgb.validation_score_[-1]:.4f}")
