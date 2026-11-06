# Dataset

This project uses **MVSA-Single**, a set of 4,869 Twitter posts, each with a text, an image and separate sentiment labels for the text and the image.

The dataset is not stored in this repository. It is 213.1 MB, and its Kaggle page lists the licence as unknown, so please download it from the source.

## Download

- **Kaggle:** [vincemarcs/mvsasingle](https://www.kaggle.com/datasets/vincemarcs/mvsasingle)
- **Original source:** [MVSA: Sentiment Analysis on Multi-view Social Data](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)
- **Citation:** T. Niu, S. A. Zhu, L. Pang and A. El Saddik, "Sentiment Analysis on Multi-view Social Data", MultiMedia Modeling (MMM), pp. 15-27, Miami, 2016.

Unzip it into `data/raw/`, which Git ignores, so that the folder looks like this:

```text
data/raw/MVSA_Single/
├── labelResultAll.txt      one line per post: ID, then text label,image label
└── data/
    ├── 1.txt               the post's text
    ├── 1.jpg               the post's image
    └── ...
```

## What the pipeline finds

| Check | Count |
|---|---|
| Posts in the label file | 4,869 |
| Exact duplicates removed (same text and image) | 30 |
| Posts kept | 4,839 |
| Agreement / soft mismatch / hard mismatch | 2,576 / 1,908 / 355 |
| Positive / negative / neutral (hard mismatches excluded) | 2,670 / 1,346 / 468 |
| Train / validation / test | 3,591 / 446 / 447 |

Hard-mismatch posts have a positive label on one modality and a negative label on the other, so they have no single true label. They are never trained on and are evaluated separately.

## Test data

To try the pipeline without downloading anything, `tests/make_synthetic_data.py` generates a small, randomly generated dataset in the same format. It contains no real posts.
