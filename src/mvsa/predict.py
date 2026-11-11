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

    def _proba(self, modality, embeddings):
        """Probabilities from one modality's logistic regression, using the training-set scaling."""
        p = self.parts
        embeddings = np.atleast_2d(embeddings)
        expected = p[f"{modality}_mean"].shape[0]
        if embeddings.shape[1] != expected:
            raise ValueError(f"Expected {expected}-dimensional {modality} embeddings, got {embeddings.shape[1]}.")
        scaled = (embeddings - p[f"{modality}_mean"]) / p[f"{modality}_scale"]
        return softmax(scaled @ p[f"{modality}_coef"].T + p[f"{modality}_intercept"])

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

    def predict(self, text=None, image_path=None):
        """Return the predicted label, its score and the probability of every class for one post.

        image_path can be a path or a file-like object.
        """
        text_embedding = self.embed_text(text) if text is not None else None
        image_embedding = self.embed_image(image_path) if image_path is not None else None
        proba = self.proba_from_embeddings(text_embedding, image_embedding)[0]
        best = int(proba.argmax())
        return {"label": self.labels[best], "score": float(proba[best]), "confidence": confidence_level(float(proba[best])),
                "probabilities": {l: float(v) for l, v in zip(self.labels, proba)},
                "used": [name for name, given in (("text", text), ("image", image_path)) if given is not None]}
