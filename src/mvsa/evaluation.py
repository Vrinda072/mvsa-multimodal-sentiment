"""Extra evaluation tools: bootstrap intervals, McNemar's test, calibration and selective prediction.

Everything here works on plain arrays (true labels, predicted labels or class probabilities), so it doesn't
depend on how the model was built. Used by notebooks/02_modeling/05_additional_evaluation.ipynb and
scripts/train_final_model.py.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


def macro_f1(y, pred):
    return f1_score(y, pred, average="macro")


def bootstrap_scores(y, preds, n_boot=2000, seed=0):
    """Macro-F1 and accuracy of several models on the same bootstrap resamples of the posts.

    preds maps a model name to its predicted labels. Every model is scored on the same resampled posts, so the
    difference between two models can be read off resample by resample (a paired bootstrap).
    Returns {name: array of shape (n_boot, 2)}, with columns macro-F1 and accuracy.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    out = {name: np.empty((n_boot, 2)) for name in preds}
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))   # resample posts with replacement
        for name, pred in preds.items():
            out[name][b] = macro_f1(y[i], pred[i]), accuracy_score(y[i], pred[i])
    return out


def interval(values, level=0.95):
    """Percentile interval of bootstrap values, as (low, high)."""
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail], axis=0)
    return low, high


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test: counts of posts only A or only B got right, and the two-sided p-value."""
    a_ok, b_ok = pred_a == y, pred_b == y
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def reliability_table(y, P, bins=10):
    """Bin posts by the model's top probability and compare the mean confidence with the accuracy in each bin."""
    conf, right = P.max(1), (P.argmax(1) == y)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)   # bin k holds edges[k] <= conf < edges[k+1]
    rows = []
    for k in range(bins):
        m = which == k
        rows.append({"bin_low": edges[k], "bin_high": edges[k + 1], "posts": int(m.sum()),
                     "confidence": conf[m].mean() if m.any() else np.nan,
                     "accuracy": right[m].mean() if m.any() else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(y, P, bins=10):
    """Average gap between confidence and accuracy over the bins, weighted by how many posts each bin holds."""
    t = reliability_table(y, P, bins)
    t = t[t["posts"] > 0]
    return float((t["posts"] * (t["confidence"] - t["accuracy"]).abs()).sum() / t["posts"].sum())


def apply_temperature(P, T):
    """Soften (T > 1) or sharpen (T < 1) class probabilities without changing which class is on top."""
    logits = np.log(np.clip(P, 1e-12, 1)) / T
    logits -= logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def fit_temperature(y, P):
    """The temperature that minimises the negative log-likelihood of the true labels. Fit it on validation."""
    y = np.asarray(y)

    def nll(T):
        return -np.log(apply_temperature(P, T)[np.arange(len(y)), y] + 1e-12).mean()

    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def risk_coverage(y, P, thresholds):
    """Accuracy and macro-F1 on the posts the model keeps when it abstains below each confidence threshold."""
    conf, pred = P.max(1), P.argmax(1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append({"threshold": t, "coverage": keep.mean(), "posts": int(keep.sum()),
                     "accuracy": accuracy_score(y[keep], pred[keep]) if keep.any() else np.nan,
                     "macro_f1": macro_f1(y[keep], pred[keep]) if keep.any() else np.nan})
    return pd.DataFrame(rows)


def stratified_subsample(y, fraction, seed):
    """Indices of a random subset holding the same share of every class (for the learning curve)."""
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.flatnonzero(y == c), max(1, round(fraction * (y == c).sum())), replace=False)
           for c in np.unique(y)]
    return np.sort(np.concatenate(idx))
