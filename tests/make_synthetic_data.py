"""Create a small, randomly generated dataset in the MVSA-Single format, for testing the pipeline.

The posts and images are synthetic and contain no real data, so scores from this data mean nothing.

Usage: python tests/make_synthetic_data.py OUTPUT_FOLDER
"""
import os
import random
import sys

import numpy as np
from PIL import Image

WORDS = {"positive": "great happy love wonderful", "neutral": "today report station update", "negative": "awful sad hate terrible"}
COLOURS = {"positive": (230, 180, 60), "neutral": (120, 120, 120), "negative": (40, 50, 110)}
LABELS = list(WORDS)
N_POSTS = 320


def main(folder):
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    data = os.path.join(folder, "data")
    os.makedirs(data, exist_ok=True)
    with open(os.path.join(folder, "labelResultAll.txt"), "w", encoding="utf-8") as labels:
        labels.write("ID\ttext,image\n")
        for post in range(1, N_POSTS + 1):
            text_label = rng.choices(LABELS, weights=[6, 2, 3])[0]
            image_label = text_label if rng.random() < 0.6 else rng.choice(LABELS)
            labels.write(f"{post}\t{text_label},{image_label}\n")
            words = " ".join(rng.choices(WORDS[text_label].split(), k=4))
            with open(os.path.join(data, f"{post}.txt"), "w", encoding="utf-8") as f:
                f.write(f"RT @someone: {words} number {post} #synthetic{post} https://t.co/abc !!")
            noise = np_rng.integers(-40, 40, (64, 64, 3))
            pixels = np.clip(np.array(COLOURS[image_label]) + noise, 0, 255).astype("uint8")
            Image.fromarray(pixels).save(os.path.join(data, f"{post}.jpg"))
    print(f"Wrote {N_POSTS} synthetic posts to {folder}")


if __name__ == "__main__":
    main(sys.argv[1])
