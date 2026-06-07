import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; notebooks override to inline
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------------------------------------------------
# Style constants
# -----------------------------------------------------------------------

_PALETTE = {
    "primary": "#2563EB",     # blue
    "secondary": "#16A34A",   # green
    "accent": "#DC2626",      # red
    "muted": "#64748B",       # slate
    "bg": "#F8FAFC",
    "grid": "#E2E8F0",
}


def _fig_style(fig, ax):
    """Apply consistent style to a figure/axis."""
    fig.patch.set_facecolor(_PALETTE["bg"])
    ax.set_facecolor(_PALETTE["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_PALETTE["grid"])
    ax.spines["bottom"].set_color(_PALETTE["grid"])
    ax.tick_params(colors=_PALETTE["muted"])
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="major", color=_PALETTE["grid"], linewidth=0.8)
    ax.grid(True, which="minor", color=_PALETTE["grid"], linewidth=0.3, linestyle=":")


# -----------------------------------------------------------------------
# 1. Metric over time
# -----------------------------------------------------------------------

def plot_metric_over_time(
    metric_values,
    title="Metric Over Chunks",
    ylabel="Metric",
    xlabel="Chunk",
    save_path=None,
    figsize=(10, 4),
    color=None,
):
    values = np.array(metric_values, dtype=float)
    x = np.arange(len(values))
    color = color or _PALETTE["primary"]

    fig, ax = plt.subplots(figsize=figsize)
    _fig_style(fig, ax)

    ax.plot(x, values, color=color, linewidth=2, marker="o", markersize=5, label=ylabel)
    ax.fill_between(x, values, alpha=0.10, color=color)
    ax.set_title(title, fontsize=14, fontweight="bold", color="#1E293B", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=_PALETTE["muted"])
    ax.set_ylabel(ylabel, fontsize=11, color=_PALETTE["muted"])
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# -----------------------------------------------------------------------
# 2. Compare two models
# -----------------------------------------------------------------------

def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    title="Model Comparison",
    ylabel="Metric",
    xlabel="Chunk",
    save_path=None,
    figsize=(10, 4),
):

    v1 = np.array(metric1, dtype=float)
    v2 = np.array(metric2, dtype=float)
    x1 = np.arange(len(v1))
    x2 = np.arange(len(v2))

    fig, ax = plt.subplots(figsize=figsize)
    _fig_style(fig, ax)

    ax.plot(x1, v1, color=_PALETTE["primary"], lw=2, marker="o", ms=5, label=labels[0])
    ax.plot(x2, v2, color=_PALETTE["secondary"], lw=2, marker="s", ms=5, label=labels[1])
    ax.fill_between(x1, v1, alpha=0.08, color=_PALETTE["primary"])
    ax.fill_between(x2, v2, alpha=0.08, color=_PALETTE["secondary"])

    ax.set_title(title, fontsize=14, fontweight="bold", color="#1E293B", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=_PALETTE["muted"])
    ax.set_ylabel(ylabel, fontsize=11, color=_PALETTE["muted"])
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# -----------------------------------------------------------------------
# 3. Predictions vs Ground Truth
# -----------------------------------------------------------------------

def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title="Predictions vs Ground Truth (Latest Chunk)",
    save_path=None,
    figsize=(10, 4),
    max_samples=200,
):
    y_true = np.array(y_true).ravel()[:max_samples]
    y_pred = np.array(y_pred).ravel()[:max_samples]
    x = np.arange(len(y_true))
    correct = y_true == y_pred

    fig, ax = plt.subplots(figsize=figsize)
    _fig_style(fig, ax)

    ax.scatter(x[correct], y_true[correct], color=_PALETTE["secondary"],
               s=20, alpha=0.7, label="Correct", zorder=3)
    ax.scatter(x[~correct], y_true[~correct], color=_PALETTE["accent"],
               s=25, alpha=0.8, marker="x", label="Wrong (true)", zorder=4)
    ax.scatter(x[~correct], y_pred[~correct], color=_PALETTE["primary"],
               s=25, alpha=0.5, marker="^", label="Wrong (pred)", zorder=4)

    ax.set_title(title, fontsize=14, fontweight="bold", color="#1E293B", pad=12)
    ax.set_xlabel("Sample Index", fontsize=11, color=_PALETTE["muted"])
    ax.set_ylabel("Class Label", fontsize=11, color=_PALETTE["muted"])
    ax.legend(fontsize=9, frameon=False, ncol=3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# -----------------------------------------------------------------------
# 4. Confusion Matrix
# -----------------------------------------------------------------------

def plot_confusion_matrix(mat, classes, title="Confusion Matrix", save_path=None, figsize=(6, 5)):
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    thresh = mat.max() / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]),
                    ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black",
                    fontsize=11)

    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# -----------------------------------------------------------------------
# 5. ROC Curve
# -----------------------------------------------------------------------

def plot_roc_curve(fpr, tpr, auc=None, title="ROC Curve", save_path=None, figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    _fig_style(fig, ax)

    label = f"ROC (AUC = {auc:.3f})" if auc is not None else "ROC Curve"
    ax.plot(fpr, tpr, color=_PALETTE["primary"], lw=2, label=label)
    ax.plot([0, 1], [0, 1], color=_PALETTE["muted"], lw=1, linestyle="--", label="Chance")
    ax.fill_between(fpr, tpr, alpha=0.08, color=_PALETTE["primary"])

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11, color=_PALETTE["muted"])
    ax.set_ylabel("True Positive Rate", fontsize=11, color=_PALETTE["muted"])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax