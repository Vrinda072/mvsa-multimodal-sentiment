"""Explain one prediction: every intermediate number the demo page draws.

For each model this gives the standardised inputs, how much each input pushed each class score, the scores and
the probabilities. It also compares the caption with the picture (cosine similarity of their CLIP embeddings,
Jensen-Shannon divergence between the two separate readings, entropy) and places the post on two-dimensional maps
of the training posts. The training-set statistics it needs are saved by scripts/build_explainer_stats.py.
"""
import os

import numpy as np

from mvsa.predict import TOKEN_RE, clean_caption, softmax

DEFAULT_STATS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "model", "explainer_stats.npz")
TOP_DIMENSIONS = 8     # how many embedding dimensions are listed one by one; the rest are summed
HISTOGRAM_BINS = 30


def load_stats(path=DEFAULT_STATS):
    """Load the saved training-set statistics as a dictionary of arrays."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def rounded(values, digits=4):
    return [round(float(v), digits) for v in np.ravel(values)]


def entropy(p):
    """Shannon entropy in bits: 0 when one class takes everything, log2(3) when three classes are equal."""
    p = np.asarray(p, dtype=float)
    return float(-(p * np.log2(np.clip(p, 1e-12, 1))).sum())


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits: 0 when two distributions match, 1 when they share nothing."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = (p + q) / 2

    def kl(a, b):
        return float((a * np.log2(np.clip(a, 1e-12, 1) / np.clip(b, 1e-12, 1))).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def clip_branch(model, modality, embedding):
    """One CLIP model worked through: z = standardised input, push = z times weight, per class."""
    p = model.parts
    z = (np.atleast_2d(embedding) - p[f"{modality}_mean"]) / p[f"{modality}_scale"]
    logits = (z @ p[f"{modality}_coef"].T + p[f"{modality}_intercept"])[0]
    push = z[0][:, None] * p[f"{modality}_coef"].T            # dimensions x classes; columns sum to logit - intercept
    top = np.argsort(-np.abs(push[:, int(np.argmax(logits))]))[:TOP_DIMENSIONS]
    return {"z": rounded(z[0], 2), "logits": rounded(logits), "proba": rounded(softmax(logits)),
            "intercept": rounded(p[f"{modality}_intercept"]),
            "top": [{"dim": int(i), "z": round(float(z[0, i]), 3), "push": rounded(push[i])} for i in top],
            "rest": rounded(push.sum(axis=0) - push[top].sum(axis=0))}


def words_branch(model, words):
    """The word model worked through, with one entry per term of the caption that is in the vocabulary."""
    p = model.parts
    words = np.atleast_2d(words)
    logits = (words @ p["words_coef"].T + p["words_intercept"])[0]
    terms = [{"term": str(p["words_terms"][i]), "w": round(float(words[0, i]), 3),
              "push": rounded(words[0, i] * p["words_coef"][:, i], 3)} for i in np.nonzero(words[0])[0]]
    terms.sort(key=lambda t: -t["w"])
    return {"logits": rounded(logits), "proba": rounded(softmax(logits)), "intercept": rounded(p["words_intercept"])}, terms


def embedding_map(stats, modality, embedding):
    """Training posts on their first two principal components, and where this post falls among them."""
    points = [[round(float(x), 3), round(float(y), 3), int(label)]
              for (x, y), label in zip(stats[f"map_{modality}_points"], stats[f"map_{modality}_labels"])]
    here = None
    if embedding is not None:
        here = rounded((np.ravel(embedding) - stats[f"map_{modality}_mean"]) @ stats[f"map_{modality}_axes"].T, 3)
    return {"pts": points, "me": here, "var": rounded(stats[f"map_{modality}_variance"], 3)}


def explain_embeddings(model, stats, caption=None, text=None, image=None, words=None):
    """The prediction for one post plus everything behind it. text and image are CLIP embeddings, or None."""
    result = model.predict_embeddings(text, image, words)
    both = text is not None and image is not None
    branches, terms = {}, []
    if image is not None:
        branches["image"] = clip_branch(model, "image", image)
    if text is not None:
        branches["text"] = clip_branch(model, "text", text)
        if words is not None:
            branches["words"], terms = words_branch(model, words)

    out = {"label": result["label"], "score": round(result["score"], 4), "confidence": result["confidence"],
           "used": result["used"], "conflict": result["conflict"], "labels": model.labels,
           "caption": caption or "", "tokens": TOKEN_RE.findall(clean_caption(caption).lower()) if caption else [],
           "branches": branches, "terms": terms, "final": rounded(list(result["probabilities"].values())),
           "readings": {name: {"label": r["label"], "proba": rounded(list(r["probabilities"].values()))}
                        for name, r in result["readings"].items()},
           "neutral_multiplier": model.neutral_multiplier}
    if both and "words" in branches:
        average = np.mean([branches[k]["proba"] for k in ("text", "image", "words")], axis=0)
        scaled = average.copy()
        scaled[model.labels.index("neutral")] *= model.neutral_multiplier
        out.update(average=rounded(average), scaled=rounded(scaled))

    sorted_cos = stats["similarity_sorted"]
    hist, edges = np.histogram(sorted_cos, bins=HISTOGRAM_BINS)
    agreement = {"hist": hist.tolist(), "edges": rounded(edges), "median": round(float(np.median(sorted_cos)), 3),
                 "posts": int(len(sorted_cos)), "H_final": round(entropy(out["final"]), 3)}
    if both:
        cos = cosine(text, image)
        # cos_pct: the share of training posts whose caption and picture agree less than this one
        agreement.update(cos=round(cos, 4), cos_pct=round(float(np.searchsorted(sorted_cos, cos) / len(sorted_cos) * 100), 1))
    for name, reading in out["readings"].items():
        agreement[f"H_{name}"] = round(entropy(reading["proba"]), 3)
    if len(out["readings"]) == 2:
        agreement["js"] = round(js_divergence(out["readings"]["text"]["proba"], out["readings"]["image"]["proba"]), 4)
    out["agreement"] = agreement
    out["maps"] = {"text": embedding_map(stats, "text", text), "image": embedding_map(stats, "image", image)}
    return out


def explain_post(model, stats, text=None, image_path=None, example_image=False):
    """Embed a caption and a picture with CLIP, then explain the prediction.

    image_path can be a path or a file-like object. With example_image, the saved example post's picture
    embedding is used in place of a picture.
    """
    if text is None and image_path is None and not example_image:
        raise ValueError("Give a text, an image, or both.")
    text_embedding = model.embed_text(text) if text is not None else None
    words = model.words_vector(text) if text is not None else None
    if image_path is not None:
        image_embedding = model.embed_image(image_path)
    else:
        image_embedding = stats["example_image"][None, :] if example_image else None
    return explain_embeddings(model, stats, text, text_embedding, image_embedding, words)


def explain_example(model, stats):
    """The saved example post, explained from its stored embeddings, so it needs neither CLIP nor the dataset."""
    caption = str(stats["example_caption"])
    out = explain_embeddings(model, stats, caption, stats["example_text"][None, :], stats["example_image"][None, :],
                             model.words_vector(caption))
    out.update(example=True, example_id=str(stats["example_id"]), image_url=str(stats["example_image_url"]))
    return out
