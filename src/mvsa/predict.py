"""Predict the sentiment of a new post with the final model.

The final model is two logistic regressions, one on CLIP text embeddings and one on CLIP image embeddings,
whose class probabilities are averaged (late fusion). It is stored as plain arrays, so it loads with any
scikit-learn version.
"""
import os

import numpy as np

from mvsa.text import ftfy, clean_text

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "results", "model", "final_model.npz")
EMPTY_PLACEHOLDER = "no text"   # the same placeholder that training used for empty captions

# Confidence cut-offs, chosen from the validation set with both text and image given. There, predictions scoring
# 0.70 or more were right 84% of the time, and those below 0.45 only 41%, close to the 33% of a random guess.
HIGH_CONFIDENCE = 0.70
LOW_CONFIDENCE = 0.45


def confidence_level(score):
    """Describe how far to trust a prediction, from its score."""
    if score >= HIGH_CONFIDENCE:
        return "high"
    return "medium" if score >= LOW_CONFIDENCE else "low"


def softmax(logits):
    """Turn raw scores into probabilities that sum to 1."""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class SentimentModel:
    def __init__(self, path=DEFAULT_MODEL):
        with np.load(path, allow_pickle=False) as f:
            self.parts = {k: f[k] for k in f.files}
        self.labels = [str(l) for l in self.parts["labels"]]
        self.text_weight = float(self.parts["text_weight"])
        self.clip_model = str(self.parts["clip_model"])
        self._clip = None

    def _proba(self, modality, embeddings, prefix=""):
        """Probabilities from one modality's logistic regression, using the training-set scaling."""
        p, key = self.parts, f"{prefix}{modality}"
        embeddings = np.atleast_2d(embeddings)
        expected = p[f"{key}_mean"].shape[0]
        if embeddings.shape[1] != expected:
            raise ValueError(f"Expected {expected}-dimensional {modality} embeddings, got {embeddings.shape[1]}.")
        scaled = (embeddings - p[f"{key}_mean"]) / p[f"{key}_scale"]
        return softmax(scaled @ p[f"{key}_coef"].T + p[f"{key}_intercept"])

    @property
    def has_readings(self):
        """True when the saved model includes the separate text and image readings."""
        return "own_text_coef" in self.parts

    def reading(self, modality, embeddings):
        """What one modality says on its own: its model was trained on that modality's own labels."""
        return self._proba(modality, embeddings, prefix="own_")

    def proba_from_embeddings(self, text=None, image=None):
        """Late fusion of both modalities. If only one is given, that modality's model is used alone."""
        if text is None and image is None:
            raise ValueError("Give a text, an image, or both.")
        if image is None:
            return self._proba("text", text)
        if text is None:
            return self._proba("image", image)
        w = self.text_weight
        return w * self._proba("text", text) + (1 - w) * self._proba("image", image)

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
        text = (ftfy.fix_text(text) if ftfy is not None else text).strip()
        cleaned = clean_text(text) or EMPTY_PLACEHOLDER
        tokens = processor.tokenizer([cleaned], padding=True, truncation=True, return_tensors="pt",
                                     max_length=model.config.text_config.max_position_embeddings).to(device)
        with torch.no_grad():
            out = model.get_text_features(**tokens)
        out = out if torch.is_tensor(out) else out.pooler_output
        return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()

    def embed_image(self, path):
        """Embed an image, the same way the training data was processed."""
        from PIL import Image
        torch, model, processor, device = self._load_clip()
        with Image.open(path) as im:
            im.draft("RGB", (224, 224))
            image = im.convert("RGB")
        pixels = processor.image_processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**pixels)
        out = out if torch.is_tensor(out) else out.pooler_output
        return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()

    def _summary(self, proba):
        best = int(proba.argmax())
        return {"label": self.labels[best], "score": float(proba[best]), "confidence": confidence_level(float(proba[best])),
                "probabilities": {l: float(v) for l, v in zip(self.labels, proba)}}

    def predict_embeddings(self, text=None, image=None):
        """Overall sentiment plus, when the model has them, what the text and the image each say.

        With both inputs, the overall result comes from the evaluated late-fusion model. With one input, it is that
        modality's own reading, because a caption or picture alone should be judged on what it says itself.
        A conflict means one reading is positive and the other negative, as in sarcasm or mixed messages.
        """
        readings = {}
        if self.has_readings:
            for name, emb in (("text", text), ("image", image)):
                if emb is not None:
                    readings[name] = self._summary(self.reading(name, emb)[0])
        if text is not None and image is not None or not readings:
            overall = self._summary(self.proba_from_embeddings(text, image)[0])
        else:
            overall = next(iter(readings.values()))
        labels = {r["label"] for r in readings.values()}
        return {**overall, "used": [n for n, e in (("text", text), ("image", image)) if e is not None],
                "readings": readings, "conflict": labels == {"positive", "negative"}}

    def predict(self, text=None, image_path=None):
        """Return the predicted label, its score and the probability of every class for one post.

        image_path can be a path or a file-like object.
        """
        text_embedding = self.embed_text(text) if text is not None else None
        image_embedding = self.embed_image(image_path) if image_path is not None else None
        if text_embedding is None and image_embedding is None:
            raise ValueError("Give a text, an image, or both.")
        return self.predict_embeddings(text_embedding, image_embedding)
