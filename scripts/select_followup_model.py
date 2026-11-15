"""Follow-up model selection, using the training and validation splits only.

The main results tune each setup on the validation split. This follow-up asks whether a few cheap changes help:
a smaller C, a TF-IDF word model as a third voter next to the CLIP text and image models, and a multiplier on the
neutral probability. To avoid fitting the choice to one small validation split, every configuration is scored by
3 x 5-fold grouped cross-validation on train + validation (posts with the same caption stay in one fold).
The rule was fixed before looking at any result: the highest mean macro-F1 wins. The test and hard splits are
never loaded here; scripts/train_final_model.py scores the chosen configuration on the test split once.

Usage: python scripts/select_followup_model.py PATH_TO_PREPROCESSED_FOLDER
Writes results/metrics/followup_selection.json.
"""
import argparse
import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["negative", "neutral", "positive"]
CS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
NEUTRAL = [0.75, 1.0, 1.25, 1.5]
WEIGHTS = [0.3, 0.5, 0.7]
NEUTRAL_INDEX = LABELS.index("neutral")


def clip_model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preprocessed", help="the preprocessed/ folder created by the notebooks")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    F = os.path.join(args.preprocessed, "features")
    posts = pd.read_csv(os.path.join(args.preprocessed, "posts_preprocessed.csv"), dtype={"id": str},
                        keep_default_na=False).set_index("id")

    def load(s):
        return (np.loadtxt(os.path.join(F, f"{s}_ids.txt"), dtype=str, ndmin=1),
                np.load(os.path.join(F, f"{s}_clip_text.npy")), np.load(os.path.join(F, f"{s}_clip_image.npy")),
                np.load(os.path.join(F, f"{s}_meta.npy")), sparse.load_npz(os.path.join(F, f"{s}_tfidf.npz")))

    a, b = load("train"), load("val")
    ids = np.r_[a[0], b[0]]
    T, I, M = np.vstack([a[1], b[1]]), np.vstack([a[2], b[2]]), np.vstack([a[3], b[3]])
    W = sparse.vstack([a[4], b[4]]).tocsr()
    y = posts.loc[ids, "label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy(int)
    groups = posts.loc[ids, "text_clean"].str.lower().to_numpy()
    X = {"text": T, "image": I, "early": np.hstack([T, I]), "early_meta": np.hstack([T, I, M])}

    folds = [f for r in range(3) for f in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(T, y, groups)]
    P = []
    for n, (tr, te) in enumerate(folds):
        p = {(k, C): clip_model(C).fit(x[tr], y[tr]).predict_proba(x[te]) for k, x in X.items() for C in CS}
        p["tfidf"] = LogisticRegression(class_weight="balanced", max_iter=5000).fit(W[tr], y[tr]).predict_proba(W[te])
        P.append(p)
        print(f"fold {n + 1} of {len(folds)}", flush=True)

    def score(fn, n):
        rows = []
        for (_, te), p in zip(folds, P):
            q = fn(p).copy()
            q[:, NEUTRAL_INDEX] *= n
            pred = q.argmax(1)
            rows.append((f1_score(y[te], pred, average="macro"), accuracy_score(y[te], pred),
                         f1_score(y[te], pred, average=None, labels=[NEUTRAL_INDEX])[0]))
        s = np.array(rows)
        return {"macro_f1": s[:, 0].mean(), "macro_f1_std": s[:, 0].std(), "accuracy": s[:, 1].mean(),
                "neutral_f1": s[:, 2].mean()}

    results = []
    for C, n in itertools.product(CS, NEUTRAL):
        for k in ("early", "early_meta"):
            results.append(({"setup": k, "C": C, "neutral": n}, score(lambda p: p[(k, C)], n)))
        for w in WEIGHTS:
            results.append(({"setup": "late", "C": C, "w": w, "neutral": n},
                            score(lambda p: w * p[("text", C)] + (1 - w) * p[("image", C)], n)))
        results.append(({"setup": "late_tfidf", "C": C, "neutral": n},
                        score(lambda p: (p[("text", C)] + p[("image", C)] + p["tfidf"]) / 3, n)))
    results.sort(key=lambda r: -r[1]["macro_f1"])

    print(f"\n{len(results)} configurations compared. Top 8:")
    for cfg, s in results[:8]:
        print(f"  macro-F1 {s['macro_f1']:.3f} +- {s['macro_f1_std']:.3f}  accuracy {s['accuracy']:.3f}  "
              f"neutral F1 {s['neutral_f1']:.3f}  {cfg}")
    main_model = next(s for c, s in results if c == {"setup": "late", "C": 0.01, "w": 0.5, "neutral": 1.0})
    print(f"main model (late fusion, C=0.01): macro-F1 {main_model['macro_f1']:.3f}  accuracy {main_model['accuracy']:.3f}")

    out = os.path.join(ROOT, "results", "metrics", "followup_selection.json")
    json.dump({"rule": "highest mean macro-F1 over 3 x 5-fold grouped cross-validation on train + validation",
               "configurations_compared": len(results), "chosen": results[0][0],
               "cv_chosen": {k: round(v, 4) for k, v in results[0][1].items()},
               "cv_main_model": {k: round(v, 4) for k, v in main_model.items()},
               "top_8": [{"config": c, **{k: round(v, 4) for k, v in s.items()}} for c, s in results[:8]]},
              open(out, "w"), indent=1)
    print("Chosen:", results[0][0], "-> saved", out)


if __name__ == "__main__":
    main()
