"""Train the model used for new predictions and save it to results/model/final_model.npz.

This is the follow-up model chosen by scripts/select_followup_model.py (results/metrics/followup_selection.json):
three logistic regressions, on CLIP text embeddings, CLIP image embeddings and TF-IDF word counts, whose class
probabilities are averaged, with the neutral probability multiplied by a fixed factor. Like the main results, it
is trained on the training split only. The script scores it on the validation and test splits, next to the main
model from results/metrics/chosen_model.json, and saves both comparisons to results/metrics/followup_results.json.

It also trains the separate caption and picture readings (each modality's model trained on that modality's own
labels) and works out, on the validation split, how often predictions at each confidence level were right.

Usage: python scripts/train_final_model.py PATH_TO_PREPROCESSED_FOLDER
"""
import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from mvsa.evaluation import bootstrap_scores, interval  # noqa: E402
from mvsa.predict import tfidf_vector  # noqa: E402

LABELS = ["negative", "neutral", "positive"]
NEUTRAL_INDEX = LABELS.index("neutral")
CLIP_MODEL = "openai/clip-vit-base-patch32"
# For the separate text and image readings. Chosen on the validation set from 0.0001, 0.001, 0.01 and 0.1:
# best macro-F1 against each modality's own labels, for both text and image.
OWN_LABEL_C = 0.001
# Confidence levels shown with every prediction (the same cut-offs as src/mvsa/predict.py)
HIGH_CONFIDENCE, LOW_CONFIDENCE = 0.70, 0.45


def load_split(folder, split):
    """Return one split's CLIP text and image embeddings, TF-IDF matrix, labels and per-modality labels."""
    features = os.path.join(folder, "features")
    posts = pd.read_csv(os.path.join(folder, "posts_preprocessed.csv"), dtype={"id": str},
                        keep_default_na=False).set_index("id")
    ids = np.loadtxt(os.path.join(features, f"{split}_ids.txt"), dtype=str, ndmin=1)
    code = {label: i for i, label in enumerate(LABELS)}
    y = posts.loc[ids, "label"].map(code).fillna(-1).to_numpy(dtype=int)   # -1: hard mismatch, no label
    own = {m: posts.loc[ids, f"{m}_label"].map(code).to_numpy(dtype=int) for m in ("text", "image")}
    return {"text": np.load(os.path.join(features, f"{split}_clip_text.npy")),
            "image": np.load(os.path.join(features, f"{split}_clip_image.npy")),
            "words": sparse.load_npz(os.path.join(features, f"{split}_tfidf.npz")),
            "text_clean": posts.loc[ids, "text_clean"].tolist(), "y": y, "own": own}


def fit(X, y, C):
    """Fit one CLIP modality's model, with the same settings as in src/mvsa/modeling.py."""
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(scaler.transform(X), y)
    return scaler, model


def proba(scaler, model, X):
    return model.predict_proba(scaler.transform(X))


def scores(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {"macro_f1": round(f.mean(), 4), "accuracy": round(accuracy_score(y, pred), 4),
            "macro_precision": round(p.mean(), 4), "macro_recall": round(r.mean(), 4),
            "per_class": {l: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4)}
                          for i, l in enumerate(LABELS)},
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preprocessed", help="the preprocessed/ folder created by the notebooks")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    chosen = json.load(open(os.path.join(ROOT, "results", "metrics", "chosen_model.json")))
    main_C, main_w = chosen["config"]["hp"]["text"]["C"], chosen["config"]["late_w"]
    cfg = json.load(open(os.path.join(ROOT, "results", "metrics", "followup_selection.json")))["chosen"]
    if cfg["setup"] != "late_tfidf":
        raise SystemExit("This script saves the late fusion of CLIP text, CLIP image and TF-IDF models only.")
    C, neutral = cfg["C"], cfg["neutral"]

    train = load_split(args.preprocessed, "train")
    y = train["y"]
    text_scaler, text_model = fit(train["text"], y, C)
    image_scaler, image_model = fit(train["image"], y, C)
    words_model = LogisticRegression(class_weight="balanced", max_iter=5000).fit(train["words"], y)
    main_text = fit(train["text"], y, main_C)
    main_image = fit(train["image"], y, chosen["config"]["hp"]["image"]["C"])
    own_text_scaler, own_text_model = fit(train["text"], train["own"]["text"], OWN_LABEL_C)
    own_image_scaler, own_image_model = fit(train["image"], train["own"]["image"], OWN_LABEL_C)

    def followup(d):
        p = (proba(text_scaler, text_model, d["text"]) + proba(image_scaler, image_model, d["image"])
             + words_model.predict_proba(d["words"])) / 3
        p[:, NEUTRAL_INDEX] *= neutral
        return p / p.sum(1, keepdims=True)

    def main_model(d):
        return main_w * proba(*main_text, d["text"]) + (1 - main_w) * proba(*main_image, d["image"])

    # The saved TF-IDF vocabulary, used by src/mvsa/predict.py without scikit-learn
    tfidf = joblib.load(os.path.join(args.preprocessed, "artifacts", "tfidf.joblib"))
    terms = np.array(sorted(tfidf.vocabulary_, key=tfidf.vocabulary_.get))

    report = {"followup_config": cfg, "trained_on": "training split", "splits": {}}
    for split in ("val", "test"):
        d = load_split(args.preprocessed, split)
        pf, pm = followup(d), main_model(d)
        f, m = pf.argmax(1), pm.argmax(1)
        report["splits"][split] = {"followup": scores(d["y"], f), "main_model": scores(d["y"], m)}
        print(f"{split:5s} follow-up  macro-F1 {f1_score(d['y'], f, average='macro'):.4f}  accuracy {accuracy_score(d['y'], f):.4f}")
        print(f"      main model macro-F1 {f1_score(d['y'], m, average='macro'):.4f}  accuracy {accuracy_score(d['y'], m):.4f}")
        for name, scaler, model in (("text", own_text_scaler, own_text_model), ("image", own_image_scaler, own_image_model)):
            p = model.predict(scaler.transform(d[name]))
            print(f"      {name} reading vs {name} labels: macro-F1 {f1_score(d['own'][name], p, average='macro'):.4f}  "
                  f"neutral F1 {f1_score(d['own'][name], p, average=None, labels=[1])[0]:.4f}")
        if split == "val":
            top, right = pf.max(1), f == d["y"]
            levels = {"high": top >= HIGH_CONFIDENCE, "medium": (top >= LOW_CONFIDENCE) & (top < HIGH_CONFIDENCE),
                      "low": top < LOW_CONFIDENCE}
            report["confidence_levels_val"] = {k: {"share_of_posts": round(v.mean(), 3), "right": round(right[v].mean(), 3)}
                                               for k, v in levels.items() if v.any()}
            print("      validation, right at each confidence level:",
                  {k: v["right"] for k, v in report["confidence_levels_val"].items()})
            # The hand-written TF-IDF in predict.py must give the same vectors as scikit-learn
            mine = np.vstack([tfidf_vector(t, {w: i for i, w in enumerate(terms)}, tfidf.idf_) for t in d["text_clean"]])
            assert np.allclose(mine, d["words"].toarray()), "TF-IDF in predict.py differs from the training features"
        else:
            b = int(((f == d["y"]) & (m != d["y"])).sum())
            c = int(((f != d["y"]) & (m == d["y"])).sum())
            report["mcnemar_test"] = {"followup_only_right": b, "main_only_right": c,
                                      "p_value": float(binomtest(b, b + c, 0.5).pvalue)}
            print(f"      McNemar: follow-up only right {b}, main model only right {c}, "
                  f"p = {report['mcnemar_test']['p_value']:.4f}")
            # 95% bootstrap intervals over test posts, for the follow-up and for its gain over the main model
            boot = bootstrap_scores(d["y"], {"followup": f, "main": m}, n_boot=5000, seed=0)
            lo, hi = interval(np.hstack([boot["followup"], boot["followup"] - boot["main"]]))
            report["bootstrap_95_test"] = {k: [round(lo[j], 3), round(hi[j], 3)] for j, k in enumerate(
                ("followup_macro_f1", "followup_accuracy", "gain_macro_f1", "gain_accuracy"))}
            print("      95% intervals:", report["bootstrap_95_test"])
    hard = load_split(args.preprocessed, "hard")
    posts = pd.read_csv(os.path.join(args.preprocessed, "posts_preprocessed.csv"), dtype={"id": str}).set_index("id")
    hard_ids = np.loadtxt(os.path.join(args.preprocessed, "features", "hard_ids.txt"), dtype=str, ndmin=1)
    report["hard_mismatch_agreement"] = {}
    for name, fn in (("followup", followup), ("main_model", main_model)):
        said = np.array(LABELS)[fn(hard).argmax(1)]
        report["hard_mismatch_agreement"][name] = {
            m: round(float((said == posts.loc[hard_ids, f"{m}_label"].to_numpy()).mean()), 3) for m in ("text", "image")}
    print("hard-mismatch posts, share matching each label:", report["hard_mismatch_agreement"])

    json.dump(report, open(os.path.join(ROOT, "results", "metrics", "followup_results.json"), "w"), indent=1)
    out = os.path.join(ROOT, "results", "model", "final_model.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, labels=np.array(LABELS), clip_model=CLIP_MODEL, neutral_multiplier=neutral,
                        text_mean=text_scaler.mean_, text_scale=text_scaler.scale_,
                        text_coef=text_model.coef_, text_intercept=text_model.intercept_,
                        image_mean=image_scaler.mean_, image_scale=image_scaler.scale_,
                        image_coef=image_model.coef_, image_intercept=image_model.intercept_,
                        words_terms=terms, words_idf=tfidf.idf_,
                        words_coef=words_model.coef_, words_intercept=words_model.intercept_,
                        own_text_mean=own_text_scaler.mean_, own_text_scale=own_text_scaler.scale_,
                        own_text_coef=own_text_model.coef_, own_text_intercept=own_text_model.intercept_,
                        own_image_mean=own_image_scaler.mean_, own_image_scale=own_image_scaler.scale_,
                        own_image_coef=own_image_model.coef_, own_image_intercept=own_image_model.intercept_)
    print("Saved", out)


if __name__ == "__main__":
    main()
