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

    def test_confidence_levels(self):
        from mvsa.predict import confidence_level
        self.assertEqual([confidence_level(s) for s in (0.9, 0.70, 0.6, 0.45, 0.44)], ["high", "high", "medium", "medium", "low"])

    def model_with_readings(self, text_says, image_says):
        """A model whose separate readings always say the given labels, to test how results are combined."""
        path = os.path.join(self.folder, f"readings_{text_says}_{image_says}.npz")
        arrays = dict(np.load(os.path.join(self.folder, "model.npz")))
        for name, says in (("text", text_says), ("image", image_says)):
            arrays.update({f"own_{name}_mean": np.zeros(8), f"own_{name}_scale": np.ones(8), f"own_{name}_coef": np.zeros((3, 8)),
                           f"own_{name}_intercept": np.eye(3)[["negative", "neutral", "positive"].index(says)] * 5})
        np.savez(path, **arrays)
        return SentimentModel(path)

    def test_without_readings_nothing_changes(self):
        result = self.model.predict_embeddings(self.text[:1], self.image[:1])
        self.assertEqual((result["readings"], result["conflict"]), ({}, False))

    def test_both_inputs_use_late_fusion_and_report_readings(self):
        m = self.model_with_readings("neutral", "positive")
        result = m.predict_embeddings(self.text[:1], self.image[:1])
        expected = m.proba_from_embeddings(self.text[:1], self.image[:1])[0]
        self.assertAlmostEqual(result["score"], expected.max())
        self.assertEqual({k: v["label"] for k, v in result["readings"].items()}, {"text": "neutral", "image": "positive"})
        self.assertFalse(result["conflict"])

    def test_one_input_uses_its_own_reading(self):
        result = self.model_with_readings("neutral", "positive").predict_embeddings(text=self.text[:1])
        self.assertEqual((result["label"], result["used"]), ("neutral", ["text"]))

    def test_opposite_readings_are_a_conflict(self):
        result = self.model_with_readings("negative", "positive").predict_embeddings(self.text[:1], self.image[:1])
        self.assertTrue(result["conflict"])

    def test_rejects_missing_input(self):
        with self.assertRaises(ValueError):
            self.model.proba_from_embeddings()


if __name__ == "__main__":
    unittest.main()
