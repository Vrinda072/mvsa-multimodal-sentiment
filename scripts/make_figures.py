"""Draw the figures in results/figures from the saved metrics in results/metrics.

Usage: python scripts/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")   # draw to files only, so it also works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "results", "metrics")
FIGURES = os.path.join(ROOT, "results", "figures")

# One palette for every figure, so they look like a set in the report
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
        # Relative luminance of the cell colour decides white or black text
        dark = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
        ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=9, color="white" if dark else INK)
    if highlight:
        r, c = highlight
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, edgecolor=INK, linewidth=2.5, zorder=5))


def dataset_overview():
    """Figure 1: label counts, agreement groups and split sizes."""
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
    """Figure 3: test macro-F1 for every family and setup, with the chosen model outlined."""
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
    """Figure 4: precision, recall and F1 per class for the chosen model."""
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
    """Figure 5: confusion matrix of the chosen model on the test split."""
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
    """Figure 6: macro-F1 when the image or the text is replaced by its mean embedding."""
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
    """Figure 7: the gate's logistic regression coefficients."""
    table = pd.read_csv(metric("gate_coefficients.csv"), index_col=0)
    table.columns = [label.capitalize() for label in table.columns]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    limit = np.abs(table.to_numpy()).max()   # symmetric scale, so white means zero
    heatmap(ax, table, DIVERGING, -limit, limit, "{:+.2f}")
    ax.xaxis.tick_top()
    ax.set_title("Gate coefficients: effect on each class's log-odds (seed 0)", loc="left", pad=26)
    save(fig, "07_gate_coefficients.png")


def bias_variance():
    """Figure 8: training and validation macro-F1 as C changes."""
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


def permutation_importance():
    """Figure 9: drop in macro-F1 when a modality, or a group of gate inputs, is shuffled."""
    modality = pd.read_csv(metric("permutation_importance_modality.csv"))
    gate = pd.read_csv(metric("permutation_importance_gate.csv"), index_col=0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    setups = ["early", "late", "gated"]
    series = [(f"{name.capitalize()} shuffled", modality.set_index(["setup", "shuffled"]).loc[[(s, name) for s in setups], "drop_mean"].to_numpy(), colour)
              for name, colour in (("text", BLUE), ("image", ORANGE))]
    bars(left, ["Early fusion", "Late fusion", "Gated fusion"], series, width=0.34, fmt="{:.3f}")
    left.set_ylim(0, 0.2)
    left.set_ylabel("Drop in test macro-F1")
    left.set_title("Shuffling each modality", loc="left")
    left.legend(loc="upper left", ncol=2)
    names = ["Text probabilities", "Image probabilities", "Agreement signals", "Confidence"]
    drawn = right.barh(names[::-1], gate["mean"].to_numpy()[::-1], 0.55, color=BLUE)
    for y, v in enumerate(gate["mean"].to_numpy()[::-1]):   # values sit right of zero even when slightly negative
        right.text(max(v, 0) + 0.003, y, f"{v:.3f}", va="center", fontsize=8, color=MUTED)
    right.axvline(0, color=GRID, linewidth=1)
    right.set_xlim(-0.01, 0.2)
    right.set_xlabel("Drop in test macro-F1")
    right.set_title("Inside the gate (5 seeds)", loc="left")
    right.tick_params(length=0)
    right.xaxis.grid(True, color=GRID, linewidth=0.8)
    right.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "09_permutation_importance.png")


def tidy(ax, grid_axis="y"):
    """Light grid behind the data, no tick marks."""
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def learning_curve():
    """Figure 10: training and validation macro-F1 as the training set grows (notebook 05)."""
    curve = pd.read_csv(metric("learning_curve.csv"))
    stats = curve.groupby(["setup", "train_posts"])[["train_macro_f1", "val_macro_f1"]].agg(["mean", "std"]).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, name in zip(axes, ["late", "early"]):
        s = stats.loc[name]
        for col, colour, label in (("train_macro_f1", BLUE, "Training"), ("val_macro_f1", ORANGE, "Validation")):
            mean, std = s[(col, "mean")], s[(col, "std")]
            ax.plot(s.index, mean, marker="o", markersize=6, linewidth=2, color=colour, label=label)
            ax.fill_between(s.index, mean - std, mean + std, color=colour, alpha=0.15, linewidth=0)   # ± 1 std over seeds
        ax.set_xlabel("Training posts")
        ax.set_title(f"{'Late' if name == 'late' else 'Early'} fusion", loc="left")
        tidy(ax)
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "10_learning_curve.png")


def confidence_intervals():
    """Figure 11: test scores of each setup with 95% bootstrap intervals (notebook 05)."""
    table = pd.read_csv(metric("bootstrap_intervals.csv"), index_col=0).reindex(list(SETUPS)).dropna(how="all")
    names = [SETUPS[s] for s in table.index][::-1]   # first setup at the top
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, title in zip(axes, ["macro_f1", "accuracy"], ["Test macro-F1", "Test accuracy"]):
        mid, low, high = (table[c].to_numpy()[::-1] for c in (key, f"{key}_low", f"{key}_high"))
        y = np.arange(len(names))
        ax.hlines(y, low, high, color=BLUE, linewidth=2)
        ax.plot(mid, y, "o", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        for yy, m, h in zip(y, mid, high):
            ax.text(h + 0.006, yy, f"{m:.3f}", va="center", fontsize=8, color=MUTED)
        ax.set_yticks(y, names)
        ax.set_title(title, loc="left")
        ax.set_xlim(min(table[f"{key}_low"].min() - 0.03, 0.4), table[f"{key}_high"].max() + 0.06)
        tidy(ax, "x")
    fig.suptitle("Chosen family, seed 0: point estimate and 95% bootstrap interval over test posts",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "11_confidence_intervals.png")


def calibration():
    """Figure 12: reliability diagrams before and after temperature scaling (notebook 05)."""
    bins = pd.read_csv(metric("reliability_bins.csv"))
    summary = pd.read_csv(metric("calibration.csv")).set_index(["model", "version"])
    models = [m for m in ("chosen", "followup") if m in set(bins["model"])]
    titles = {"chosen": "Chosen model", "followup": "Follow-up model (demo)"}
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.4), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (3, 3)), label="Perfectly calibrated")
        # Scaled line dashed with square markers, so it stays visible when it lies on top of the raw one
        for version, colour, style, mark in (("raw", BLUE, "-", "o"), ("temperature scaled", ORANGE, "--", "s")):
            b = bins[(bins.model == model) & (bins.version == version) & (bins.posts > 0)]
            ece = summary.loc[(model, version), "ece"]
            label = f"{version.capitalize()} (ECE {ece:.3f})"
            if version != "raw":
                label = f"Temperature {summary.loc[(model, version), 'temperature']:.2f} (ECE {ece:.3f})"
            ax.plot(b["confidence"], b["accuracy"], marker=mark, markersize=6, linewidth=2, linestyle=style,
                    color=colour, label=label)
        ax.set_xlim(0.3, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Mean top probability in the bin")
        ax.set_title(titles[model], loc="left")
        ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.9)
        tidy(ax)
    axes[0][0].set_ylabel("Share right in the bin")
    fig.tight_layout()
    save(fig, "12_calibration.png")


def risk_coverage():
    """Figure 13: accuracy on the posts kept when the model abstains below a confidence threshold (notebook 05)."""
    table = pd.read_csv(metric("risk_coverage.csv"))
    fig, ax = plt.subplots(figsize=(7.6, 4))
    for model, colour, name in (("chosen", BLUE, "Chosen model"), ("followup", ORANGE, "Follow-up model (demo)")):
        t = table[(table.model == model) & (table.posts > 0)]
        if t.empty:
            continue
        ax.plot(t["coverage"], t["accuracy"], marker="o", markersize=5, linewidth=2, color=colour, label=name)
        for cut in (0.45, 0.70):   # the demo's low and high confidence cut-offs
            row = t[np.isclose(t["threshold"], cut)]
            if len(row):
                c, a = row["coverage"].iloc[0], row["accuracy"].iloc[0]
                ax.plot(c, a, "o", markersize=11, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.5)
                ax.annotate(f"{cut:.2f}", (c, a), xytext=(7, -13), textcoords="offset points", fontsize=8, color=MUTED)
    ax.invert_xaxis()   # read left to right as the model gets pickier
    ax.set_xlabel("Coverage (share of test posts the model answers)")
    ax.set_ylabel("Accuracy on those posts")
    ax.set_title("Answering only when sure (ringed: the demo's 0.45 and 0.70 cut-offs)", loc="left")
    ax.legend(loc="lower right")
    tidy(ax)
    save(fig, "13_risk_coverage.png")


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    dataset_overview()
    model_comparison()
    per_class()
    confusion()
    robustness()
    gate_coefficients()
    bias_variance()
    permutation_importance()
    # Figures 10 to 13 come from notebook 05; skip them until its results are in results/metrics
    for draw, needs in ((learning_curve, "learning_curve.csv"), (confidence_intervals, "bootstrap_intervals.csv"),
                        (calibration, "reliability_bins.csv"), (risk_coverage, "risk_coverage.csv")):
        if os.path.exists(metric(needs)):
            draw()
        else:
            print(f"Skipped {draw.__name__}: {needs} not in results/metrics yet")
