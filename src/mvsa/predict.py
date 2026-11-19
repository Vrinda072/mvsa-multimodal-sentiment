"""Predict the sentiment of a new post.

The saved model (scripts/train_final_model.py) is the follow-up model: three logistic regressions, on CLIP text
embeddings, CLIP image embeddings and TF-IDF word counts of the caption, whose class probabilities are averaged,
with the neutral probability multiplied by a fixed factor. It is stored as plain arrays, so it loads without
scikit-learn.
"""
import math
import os
import re
from collections import Counter

import numpy as np

from mvsa.text import ftfy, clean_text

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "results", "model", "final_model.npz")
EMPTY_PLACEHOLDER = "no text"   # the same placeholder that training used for empty captions

# Confidence cut-offs. On the validation split, with both caption and picture given, how often predictions at
# each level were right is saved in results/metrics/followup_results.json.
HIGH_CONFIDENCE = 0.70
LOW_CONFIDENCE = 0.45


def confidence_level(score):
    """Describe how far to trust a prediction, from its score."""
    if score >= HIGH_CONFIDENCE:
        return "high"
    return "medium" if score >= LOW_CONFIDENCE else "low"


TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")   # scikit-learn's default token pattern


def tfidf_vector(text, vocabulary, idf):
    """TF-IDF features as in training: lowercase, words and word pairs, 1 + log(count), times idf, unit length."""
    words = TOKEN_RE.findall(text.lower())
    terms = words + [" ".join(pair) for pair in zip(words, words[1:])]   # single words plus neighbouring pairs
    vector = np.zeros(len(idf))
    for term, count in Counter(terms).items():
        if term in vocabulary:            # terms the model never saw in training are ignored
            # Sublinear term frequency (1 + log count) times how rare the term is across training captions
            vector[vocabulary[term]] = (1 + math.log(count)) * idf[vocabulary[term]]
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector   # a caption with no known terms stays all zeros


def clean_caption(text):
    """Fix broken characters and clean a caption, the same way the training captions were cleaned."""
    text = (ftfy.fix_text(text) if ftfy is not None else text).strip()
    return clean_text(text)


def softmax(logits):
    """Turn raw scores into probabilities that sum to 1."""
    # Subtracting the largest score first avoids overflow in exp() and does not change the result
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class SentimentModel:
    """The saved follow-up model, plus CLIP for turning new captions and pictures into embeddings."""

    def __init__(self, path=DEFAULT_MODEL):
        # allow_pickle=False: the file holds only plain arrays, so no code can be run while loading it
        with np.load(path, allow_pickle=False) as f:
            self.parts = {k: f[k] for k in f.files}   # weights, scalers and vocabulary, by name
        self.labels = [str(l) for l in self.parts["labels"]]
        self.clip_model = str(self.parts["clip_model"])                     # which CLIP was used in training
        self.neutral_multiplier = float(self.parts["neutral_multiplier"])   # chosen by cross-validation
        self.vocabulary = {str(t): i for i, t in enumerate(self.parts["words_terms"])}   # term -> column
        self._clip = None   # CLIP is loaded lazily, on first use

    def _proba(self, modality, embeddings, prefix=""):
        """Probabilities from one modality's logistic regression, using the training-set scaling."""
        p, key = self.parts, f"{prefix}{modality}"   # prefix "own_" selects the separate-reading models
        embeddings = np.atleast_2d(embeddings)
        expected = p[f"{key}_mean"].shape[0]
        if embeddings.shape[1] != expected:
            raise ValueError(f"Expected {expected}-dimensional {modality} embeddings, got {embeddings.shape[1]}.")
        scaled = (embeddings - p[f"{key}_mean"]) / p[f"{key}_scale"]   # standardise with training mean and std
        # Logistic regression: a weighted sum per class (the logits), then softmax to get probabilities
        return softmax(scaled @ p[f"{key}_coef"].T + p[f"{key}_intercept"])

    @property
    def has_readings(self):
        """True when the saved model includes the separate text and image readings."""
        return "own_text_coef" in self.parts

    def reading(self, modality, embeddings):
        """What one modality says on its own: its model was trained on that modality's own labels."""
        return self._proba(modality, embeddings, prefix="own_")

    def words_vector(self, text):
        """TF-IDF features of a caption."""
        return tfidf_vector(clean_caption(text), self.vocabulary, self.parts["words_idf"])[None, :]

    def _words_proba(self, words):
        p = self.parts
        return softmax(np.atleast_2d(words) @ p["words_coef"].T + p["words_intercept"])

    def proba_from_embeddings(self, text=None, image=None, words=None):
        """The fused probabilities for a caption and a picture.

        text and image are CLIP embeddings and words the caption's TF-IDF vector. The three models' probabilities
        are averaged, the neutral one is multiplied by the saved factor, and the result is scaled to sum to 1.
        If only one CLIP embedding is given, that modality's model is used alone.
        """
        if text is None and image is None:
            raise ValueError("Give a text, an image, or both.")
        if image is None:
            return self._proba("text", text)
        if text is None:
            return self._proba("image", image)
        if words is None:
            raise ValueError("Give the caption's TF-IDF vector (words_vector) along with the text embedding.")
        proba = (self._proba("text", text) + self._proba("image", image) + self._words_proba(words)) / 3
        proba[:, self.labels.index("neutral")] *= self.neutral_multiplier
        return proba / proba.sum(axis=1, keepdims=True)

    def _load_clip(self):
        """Load CLIP once, on first use, because it is slow to start."""
        if self._clip is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(self.clip_model).to(device).eval()
            self._clip = (torch, model, CLIPProcessor.from_pretrained(self.clip_model), device)
        return self._clip

    def embed_text(self, text):
        """Clean a caption and embed it, the same way the training data was processed."""
        torch, model, processor, device = self._load_clip()
        cleaned = clean_caption(text) or EMPTY_PLACEHOLDER
        # CLIP reads at most 77 tokens, so longer captions are cut off at that limit
        tokens = processor.tokenizer([cleaned], padding=True, truncation=True, return_tensors="pt",
                                     max_length=model.config.text_config.max_position_embeddings).to(device)
        with torch.no_grad():                       # inference only, so no gradients are needed
            out = model.get_text_features(**tokens)
        # Newer versions of transformers return an output object instead of a plain tensor
        out = out if torch.is_tensor(out) else out.pooler_output
        # Scale to unit length, as in training, so a dot product between embeddings is their cosine similarity
        return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()

    def embed_image(self, path):
        """Embed an image, the same way the training data was processed."""
        from PIL import Image
        torch, model, processor, device = self._load_clip()
        with Image.open(path) as im:
            im.draft("RGB", (224, 224))             # decode large JPEGs at reduced size, which is much faster
            image = im.convert("RGB")               # e.g. PNGs with transparency or greyscale images
        # CLIP's own preprocessing: resize, centre-crop to 224 x 224 and normalise the colours
        pixels = processor.image_processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**pixels)
        out = out if torch.is_tensor(out) else out.pooler_output
        return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()

    def _summary(self, proba):
        """Turn one row of class probabilities into the label, its score, a confidence level and all probabilities."""
        best = int(proba.argmax())
        return {"label": self.labels[best], "score": float(proba[best]), "confidence": confidence_level(float(proba[best])),
                "probabilities": {l: float(v) for l, v in zip(self.labels, proba)}}

    def predict_embeddings(self, text=None, image=None, words=None):
        """Overall sentiment plus, when the model has them, what the text and the image each say.

        With both inputs, the overall result comes from the fused model. With one input, it is that modality's
        own reading, because a caption or picture alone should be judged on what it says itself.
        A conflict means one reading is positive and the other negative, as in sarcasm or mixed messages.
        """
        readings = {}
        if self.has_readings:
            # Separate readings: what the caption alone and the picture alone say, from models trained on each
            # part's own label
            for name, emb in (("text", text), ("image", image)):
                if emb is not None:
                    readings[name] = self._summary(self.reading(name, emb)[0])
        if text is not None and image is not None or not readings:
            overall = self._summary(self.proba_from_embeddings(text, image, words)[0])   # the fused prediction
        else:
            overall = next(iter(readings.values()))   # only one part given: use its own reading
        # Flag a possible sarcasm or mixed message when one reading is positive and the other negative
        labels = {r["label"] for r in readings.values()}
        return {**overall, "used": [n for n, e in (("text", text), ("image", image)) if e is not None],
                "readings": readings, "conflict": labels == {"positive", "negative"}}

    def predict(self, text=None, image_path=None):
        """Return the predicted label, its score and the probability of every class for one post.

        image_path can be a path or a file-like object.
        """
        if text is None and image_path is None:
            raise ValueError("Give a text, an image, or both.")
        # Turn each given part into model inputs; a missing part stays None
        text_embedding = self.embed_text(text) if text is not None else None
        words = self.words_vector(text) if text is not None else None   # TF-IDF vector for the word model
        image_embedding = self.embed_image(image_path) if image_path is not None else None
        return self.predict_embeddings(text_embedding, image_embedding, words)
