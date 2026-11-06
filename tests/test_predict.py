"""Unit tests for the prediction code. These use small random data, so they don't need CLIP or the dataset."""
import os
import sys
import tempfile
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mvsa.predict import SentimentModel  # noqa: E402


def fit(X, y):
    scaler = StandardScaler().fit(X)
    return scaler, LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(X), y)


class PredictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.y = rng.integers(0, 3, 90)
        cls.text = rng.normal(size=(90, 8)) + cls.y[:, None]
        cls.image = rng.normal(size=(90, 8)) - cls.y[:, None]
        cls.text_fit, cls.image_fit = fit(cls.text, cls.y), fit(cls.image, cls.y)
        cls.folder = tempfile.mkdtemp()
        path = os.path.join(cls.folder, "model.npz")
        arrays = {}
        for name, (scaler, model) in (("text", cls.text_fit), ("image", cls.image_fit)):
            arrays.update({f"{name}_mean": scaler.mean_, f"{name}_scale": scaler.scale_,
                           f"{name}_coef": model.coef_, f"{name}_intercept": model.intercept_})
        np.savez(path, labels=np.array(["negative", "neutral", "positive"]), text_weight=0.25,
                 clip_model="unused", **arrays)
        cls.model = SentimentModel(path)

    def sklearn_proba(self, fitted, X):
        scaler, model = fitted
        return model.predict_proba(scaler.transform(X))

    def test_text_only_matches_scikit_learn(self):
        np.testing.assert_allclose(self.model.proba_from_embeddings(text=self.text), self.sklearn_proba(self.text_fit, self.text))

    def test_image_only_matches_scikit_learn(self):
        np.testing.assert_allclose(self.model.proba_from_embeddings(image=self.image), self.sklearn_proba(self.image_fit, self.image))

    def test_late_fusion_uses_the_weight(self):
        expected = 0.25 * self.sklearn_proba(self.text_fit, self.text) + 0.75 * self.sklearn_proba(self.image_fit, self.image)
        np.testing.assert_allclose(self.model.proba_from_embeddings(self.text, self.image), expected)

    def test_probabilities_sum_to_one(self):
        np.testing.assert_allclose(self.model.proba_from_embeddings(self.text, self.image).sum(axis=1), 1)

    def test_rejects_wrong_embedding_size(self):
        with self.assertRaises(ValueError):
            self.model.proba_from_embeddings(text=np.zeros((1, 4)))

    def test_rejects_missing_input(self):
        with self.assertRaises(ValueError):
            self.model.proba_from_embeddings()


if __name__ == "__main__":
    unittest.main()
