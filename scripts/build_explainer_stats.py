"""Save the training-set statistics the demo page needs to explain a prediction.

Writes results/model/explainer_stats.npz with:
- the cosine similarity between each training post's CLIP caption and picture embeddings, sorted
- for each CLIP space, the first two principal components of the training embeddings and a random sample of
  training posts projected onto them, with each post's own label for that part
- one example post from the test split (its caption and its two CLIP embeddings), shown when the page opens

Usage: python scripts/build_explainer_stats.py PATH_TO_PREPROCESSED_FOLDER
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["negative", "neutral", "positive"]
MAP_POINTS = 900
# A test post the saved model gets right, with the same label for its text and its image. The dataset's pictures
# are not redistributed here, so the page loads this one from a public copy of the dataset.
EXAMPLE_ID = "1031"
EXAMPLE_IMAGE_URL = "https://huggingface.co/datasets/xwycyj/MVSA-Single/resolve/main/data/1031.jpg"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("preprocessed", help="folder written by the preprocessing notebook")
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "model", "explainer_stats.npz"))
    args = parser.parse_args()

    features = os.path.join(args.preprocessed, "features")
    posts = pd.read_csv(os.path.join(args.preprocessed, "posts_preprocessed.csv"), dtype={"id": str},
                        keep_default_na=False).set_index("id")
    ids = np.loadtxt(os.path.join(features, "train_ids.txt"), dtype=str)
    train = {m: np.load(os.path.join(features, f"train_clip_{m}.npy")) for m in ("text", "image")}

    similarity = (train["text"] * train["image"]).sum(axis=1) / (
        np.linalg.norm(train["text"], axis=1) * np.linalg.norm(train["image"], axis=1))
    arrays = {"similarity_sorted": np.sort(similarity).astype(np.float32)}

    sample = np.random.default_rng(0).choice(len(ids), min(MAP_POINTS, len(ids)), replace=False)
    code = {label: i for i, label in enumerate(LABELS)}
    for m, X in train.items():
        mean = X.mean(axis=0)
        _, singular, axes = np.linalg.svd(X - mean, full_matrices=False)
        variance = singular ** 2 / (singular ** 2).sum()
        arrays.update({f"map_{m}_mean": mean.astype(np.float32), f"map_{m}_axes": axes[:2].astype(np.float32),
                       f"map_{m}_variance": variance[:2].astype(np.float32),
                       f"map_{m}_points": ((X[sample] - mean) @ axes[:2].T).astype(np.float32),
                       f"map_{m}_labels": posts.loc[ids[sample], f"{m}_label"].map(code).to_numpy(dtype=np.int8)})

    test_ids = list(np.loadtxt(os.path.join(features, "test_ids.txt"), dtype=str))
    k = test_ids.index(EXAMPLE_ID)
    arrays.update(example_id=np.array(EXAMPLE_ID), example_caption=np.array(posts.loc[EXAMPLE_ID, "text_raw"]),
                  example_image_url=np.array(EXAMPLE_IMAGE_URL),
                  example_text=np.load(os.path.join(features, "test_clip_text.npy"))[k].astype(np.float32),
                  example_image=np.load(os.path.join(features, "test_clip_image.npy"))[k].astype(np.float32))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB): {len(ids)} training posts, "
          f"median caption-picture similarity {np.median(similarity):.3f}, example post {EXAMPLE_ID}")


if __name__ == "__main__":
    main()
