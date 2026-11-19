"""Draw the system architecture diagram in docs/images/architecture.png.

Usage: python scripts/make_architecture_diagram.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images", "architecture.png")

# Same palette as scripts/make_figures.py
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINT = {BLUE: "#eaf2fc", ORANGE: "#fdeee7", AQUA: "#e5f6ef", MUTED: "#f1f0ed"}

# Stages run left to right. Each box is (title, detail lines, colour); boxes in a stage stack top to bottom.
# Stages listed in PARALLEL hold side-by-side parts, so they get no arrows between their boxes.
STAGES = [
    ("1  Data", [
        ("MVSA-Single", ["4,869 Twitter posts", "separate text and image labels"], MUTED),
        ("Cleaning and labels", ["30 duplicates removed", "one label per post (MVSA rule)", "text cleaned, emojis to words"], MUTED),
        ("Grouped 8:1:1 split", ["train 3,591 / val 446 / test 447", "same caption stays in one split", "355 hard mismatches held out"], MUTED),
    ]),
    ("2  Features (fitted on train only)", [
        ("CLIP ViT-B/32, frozen", ["text embedding, 512-d", "image embedding, 512-d", "text-image cosine"], BLUE),
        ("TF-IDF", ["word 1-2-grams, 5,000 terms"], BLUE),
        ("Hand-made features", ["colour histogram, 24 bins", "text counts, VADER, image cues", "log + standardised"], BLUE),
    ]),
    ("3  Classifiers and fusion", [
        ("4 classifier families", ["logistic regression, linear SVM", "random forest, MLP", "balanced class weights"], ORANGE),
        ("6 fusion setups", ["text only, image only", "early, early + meta", "late: w p_text + (1-w) p_image"], ORANGE),
        ("Gated fusion (proposed)", ["stacking on out-of-fold", "probabilities + agreement", "signals (cosine, JS, confidence)"], ORANGE),
    ]),
    ("4  Selection and evaluation", [
        ("Tune on validation", ["small grids, seed 0", "C, alpha, leaf size, w, gate C"], AQUA),
        ("Choose by val macro-F1", ["logistic regression + late fusion", "test split never used to choose"], AQUA),
        ("Test over 5 seeds", ["accuracy, precision, recall, F1", "McNemar, missing modality,", "permutation importance, errors"], AQUA),
    ]),
    ("5  Prediction", [
        ("Follow-up model", ["3 x 5-fold grouped CV", "on train + val only", "CLIP text + image + TF-IDF"], MUTED),
        ("Saved weights", ["results/model/final_model.npz"], MUTED),
        ("predict.py and demo page", ["new caption and/or picture", "label, score, readings"], MUTED),
    ]),
]

PARALLEL = {1, 2}
COL_W, GAP, TOP = 3.3, 0.75, 9.2
BOX_H = {1: 0.95, 2: 1.25, 3: 1.55}   # box height by number of detail lines
V_GAP = 0.45


def box(ax, x, y, w, h, title, lines, colour):
    """One rounded box with a bold title and smaller detail lines. (x, y) is the top-left corner."""
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=TINT[colour], edgecolor=colour, linewidth=1.4))
    ax.text(x + 0.15, y - 0.2, title, fontsize=9.5, fontweight="bold", color=INK, va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.15, y - 0.52 - 0.3 * i, line, fontsize=8, color=MUTED, va="top")


def arrow(ax, start, end, colour=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=colour, linewidth=1.2,
                                 shrinkA=2, shrinkB=2))


def main():
    width = len(STAGES) * COL_W + (len(STAGES) - 1) * GAP
    fig, ax = plt.subplots(figsize=(width * 1.02, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(-0.1, width + 0.1)
    ax.set_ylim(3.3, TOP + 0.75)
    ax.axis("off")

    stage_mid = []   # (right edge, left edge, middle height) of each stage, for the arrows between stages
    for s, (header, boxes) in enumerate(STAGES):
        x = s * (COL_W + GAP)
        ax.text(x, TOP + 0.45, header, fontsize=10.5, fontweight="bold", color=INK, va="center")
        y, tops = TOP, []
        for title, lines, colour in boxes:
            h = BOX_H[len(lines)]
            box(ax, x, y, COL_W, h, title, lines, colour)
            tops.append((y, h))
            y -= h + V_GAP
        # Arrows down the stage where each step feeds the next
        for (y1, h1), (y2, _) in ([] if s in PARALLEL else zip(tops, tops[1:])):
            arrow(ax, (x + COL_W / 2, y1 - h1), (x + COL_W / 2, y2))
        bottom = tops[-1][0] - tops[-1][1]
        stage_mid.append((x + COL_W, x, (TOP + bottom) / 2))

    # Arrows across stages
    for (right, _, mid), (_, left, _) in zip(stage_mid, stage_mid[1:]):
        arrow(ax, (right, mid), (left, mid), INK)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print("Wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
