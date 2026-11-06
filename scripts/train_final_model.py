"""Train the selected model on the training split and save it to results/model/final_model.npz.

It uses the settings chosen during tuning (results/metrics/chosen_model.json) and prints its validation and test
scores, so they can be compared with the reported results.

Usage: python scripts/train_final_model.py PATH_TO_PREPROCESSED_FOLDER
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["negative", "neutral", "positive"]
CLIP_MODEL = "openai/clip-vit-base-patch32"


def load_split(folder, split):
    """Return one split's text embeddings, image embeddings and labels."""
    features = os.path.join(folder, "features")
    posts = pd.read_csv(os.path.join(folder, "posts_preprocessed.csv"), dtype={"id": str}).set_index("id")
    ids = np.loadtxt(os.path.join(features, f"{split}_ids.txt"), dtype=str, ndmin=1)
    y = posts.loc[ids, "label"].map({label: i for i, label in enumerate(LABELS)}).to_numpy(dtype=int)
    return (np.load(os.path.join(features, f"{split}_clip_text.npy")),
            np.load(os.path.join(features, f"{split}_clip_image.npy")), y)


def fit(X, y, C):
    """Fit one modality's model, with the same settings as in src/mvsa/modeling.py."""
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(scaler.transform(X), y)
    return scaler, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preprocessed", help="the preprocessed/ folder created by the notebooks")
    args = parser.parse_args()

    chosen = json.load(open(os.path.join(ROOT, "results", "metrics", "chosen_model.json")))
    if (chosen["chosen_family"], chosen["chosen_setup"]) != ("logreg", "late"):
        raise SystemExit("This script only saves logistic regression with late fusion.")
    hp, w = chosen["config"]["hp"], chosen["config"]["late_w"]

    text, image, y = load_split(args.preprocessed, "train")
    text_scaler, text_model = fit(text, y, hp["text"]["C"])
    image_scaler, image_model = fit(image, y, hp["image"]["C"])

    for split in ("val", "test"):
        text, image, y = load_split(args.preprocessed, split)
        proba = (w * text_model.predict_proba(text_scaler.transform(text))
                 + (1 - w) * image_model.predict_proba(image_scaler.transform(image)))
        pred = proba.argmax(1)
        print(f"{split:5s} macro-F1 {f1_score(y, pred, average='macro'):.4f}  accuracy {accuracy_score(y, pred):.4f}")

    out = os.path.join(ROOT, "results", "model", "final_model.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, labels=np.array(LABELS), text_weight=w, clip_model=CLIP_MODEL,
                        text_mean=text_scaler.mean_, text_scale=text_scaler.scale_,
                        text_coef=text_model.coef_, text_intercept=text_model.intercept_,
                        image_mean=image_scaler.mean_, image_scale=image_scaler.scale_,
                        image_coef=image_model.coef_, image_intercept=image_model.intercept_)
    print("Saved", out)


if __name__ == "__main__":
    main()
