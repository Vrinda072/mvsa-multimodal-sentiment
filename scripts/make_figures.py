"""Draw the figures in results/figures from the saved metrics in results/metrics.

Usage: python scripts/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "results", "metrics")
FIGURES = os.path.join(ROOT, "results", "figures")

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#eef4fb", "#8db8ea", BLUE, "#154a8a"])
DIVERGING = LinearSegmentedColormap.from_list("polarity", [ORANGE, "#efeeea", BLUE])
FAMILIES = {"logreg": "Logistic regression", "linear_svm": "Linear SVM", "random_forest": "Random forest", "mlp": "MLP"}
SETUPS = {"baseline_tfidf_hist": "Baseline", "text_only": "Text only", "image_only": "Image only", "early": "Early fusion",
          "early_meta": "Early + meta", "late": "Late fusion", "gated": "Gated fusion"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False,
})


def metric(name):
    return os.path.join(METRICS, name)


def save(fig, name):
    fig.savefig(os.path.join(FIGURES, name), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", name)


def bars(ax, groups, series, width=0.26, fmt="{:.2f}"):
    """Grouped bars with a gap between bars and a value above each one."""
    x = np.arange(len(groups))
    for i, (label, values, colour) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        drawn = ax.bar(x + offset, values, width - 0.025, color=colour, label=label)
        ax.bar_label(drawn, labels=[fmt.format(v) for v in values], padding=2, fontsize=8, color=MUTED)
    ax.set_xticks(x, groups)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def heatmap(ax, table, cmap, vmin, vmax, fmt, highlight=None):
    """A labelled grid, with text that stays readable on dark and light cells."""
    values = table.to_numpy(dtype=float)
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(.5, table.shape[1] - 1), minor=True)
    ax.set_yticks(np.arange(.5, table.shape[0] - 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for (r, c), v in np.ndenumerate(values):
        if np.isnan(v):
            continue
        red, green, blue, _ = cmap((v - vmin) / (vmax - vmin))
        dark = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
        ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=9, color="white" if dark else INK)
    if highlight:
        r, c = highlight
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, edgecolor=INK, linewidth=2.5, zorder=5))


def dataset_overview():
    summary = json.load(open(metric("preprocessing_summary.json")))
    panels = [("Sentiment labels", summary["label_counts"], ["positive", "negative", "neutral"], "Positive,Negative,Neutral"),
              ("Text and image agreement", summary["mismatch_counts"], ["agreement", "soft_mismatch", "hard_mismatch"], "Agreement,Soft mismatch,Hard mismatch"),
              ("Data splits", summary["split_sizes"], ["train", "val", "test", "hard"], "Train,Validation,Test,Hard (held out)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (title, counts, keys, names) in zip(axes, panels):
        drawn = ax.bar(names.split(","), [counts[k] for k in keys], 0.55, color=BLUE)
        ax.bar_label(drawn, labels=[f"{counts[k]:,}" for k in keys], padding=2, fontsize=9, color=MUTED)
        ax.set_title(title, loc="left", pad=10)
        ax.margins(y=0.1)
        ax.set_ylabel("Posts")
        ax.tick_params(length=0)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "01_dataset_overview.png")


def model_comparison():
    summary = pd.read_csv(metric("summary_mean_over_seeds.csv"))
    table = summary.pivot(index="setup", columns="family", values="test_macro_f1").reindex(index=list(SETUPS), columns=list(FAMILIES))
    table.index, table.columns = list(SETUPS.values()), list(FAMILIES.values())
    chosen = json.load(open(metric("chosen_model.json")))
    highlight = (list(SETUPS).index(chosen["chosen_setup"]), list(FAMILIES).index(chosen["chosen_family"]))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    heatmap(ax, table, SEQUENTIAL, 0.48, 0.68, "{:.3f}", highlight)
    ax.xaxis.tick_top()
    ax.set_title("Test macro-F1, mean over 5 seeds (outlined: chosen by validation score)", loc="left", pad=26)
    save(fig, "03_model_comparison.png")


def per_class():
    table = pd.read_csv(metric("per_class_metrics.csv"), index_col=0)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    series = [("Precision", table["precision"], BLUE), ("Recall", table["recall"], ORANGE), ("F1", table["f1"], AQUA)]
    bars(ax, [label.capitalize() for label in table.index], series)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Chosen model on the test set, per class", loc="left")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
    save(fig, "04_per_class_metrics.png")


def confusion():
    table = pd.read_csv(metric("confusion_matrix_seed0.csv"), index_col=0)
    table.index = [label.capitalize() for label in table.index]
    table.columns = [label.capitalize() for label in table.columns]
    fig, ax = plt.subplots(figsize=(5, 4))
    heatmap(ax, table, SEQUENTIAL, 0, table.to_numpy().max(), "{:.0f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Chosen model, test posts (seed 0)", loc="left")
    save(fig, "05_confusion_matrix.png")


def robustness():
    table = pd.read_csv(metric("robustness_missing_modality.csv"), index_col=0)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    series = [("Both modalities", table["f1_full"], BLUE), ("Image masked", table["f1_no_image"], ORANGE), ("Text masked", table["f1_no_text"], AQUA)]
    bars(ax, ["Early fusion", "Late fusion", "Gated fusion"], series)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Masking a modality with its training-set mean (logistic regression, seed 0)", loc="left")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
    save(fig, "06_missing_modality.png")


def gate_coefficients():
    table = pd.read_csv(metric("gate_coefficients.csv"), index_col=0)
    table.columns = [label.capitalize() for label in table.columns]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    limit = np.abs(table.to_numpy()).max()
    heatmap(ax, table, DIVERGING, -limit, limit, "{:+.2f}")
    ax.xaxis.tick_top()
    ax.set_title("Gate coefficients: effect on each class's log-odds (seed 0)", loc="left", pad=26)
    save(fig, "07_gate_coefficients.png")


def bias_variance():
    curve = pd.read_csv(metric("bias_variance_curve.csv"), index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, name, title in zip(axes, ["early", "late"], ["Early fusion", "Late fusion"]):
        ax.plot(curve.index, curve[f"{name}_train"], marker="o", markersize=6, linewidth=2, color=BLUE, label="Training")
        ax.plot(curve.index, curve[f"{name}_val"], marker="o", markersize=6, linewidth=2, color=ORANGE, label="Validation")
        ax.axvline(0.01, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(0.011, 0.47, "selected C", fontsize=8.5, color=MUTED)
        ax.set_xscale("log")
        ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("C (larger = weaker regularisation)")
        ax.set_title(f"Logistic regression, {title.lower()}", loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "08_bias_variance_curve.png")


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    dataset_overview()
    model_comparison()
    per_class()
    confusion()
    robustness()
    gate_coefficients()
    bias_variance()
