"""Unit tests for the prediction code. These use small random data, so they don't need CLIP or the dataset."""
import os
import sys
import tempfile
import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mvsa.predict import SentimentModel, tfidf_vector  # noqa: E402

CAPTIONS = ["best day ever", "worst day ever", "meeting at noon", "I love this love this", "so sad today",
            "Reading the newspaper", "happy happy day", "terrible news today", "just a photo of lunch"]


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
        cls.tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True).fit(CAPTIONS)
        cls.words = cls.tfidf.transform([CAPTIONS[i % len(CAPTIONS)] for i in range(90)])
        cls.words_fit = LogisticRegression(class_weight="balanced", max_iter=1000).fit(cls.words, cls.y)
        cls.folder = tempfile.mkdtemp()
        path = os.path.join(cls.folder, "model.npz")
        arrays = {}
        for name, (scaler, model) in (("text", cls.text_fit), ("image", cls.image_fit)):
            arrays.update({f"{name}_mean": scaler.mean_, f"{name}_scale": scaler.scale_,
                           f"{name}_coef": model.coef_, f"{name}_intercept": model.intercept_})
        terms = np.array(sorted(cls.tfidf.vocabulary_, key=cls.tfidf.vocabulary_.get))
        np.savez(path, labels=np.array(["negative", "neutral", "positive"]), neutral_multiplier=0.75,
                 clip_model="unused", words_terms=terms, words_idf=cls.tfidf.idf_, words_coef=cls.words_fit.coef_,
                 words_intercept=cls.words_fit.intercept_, **arrays)
        cls.model = SentimentModel(path)

    def sklearn_proba(self, fitted, X):
        scaler, model = fitted
        return model.predict_proba(scaler.transform(X))

    def test_text_only_matches_scikit_learn(self):
        np.testing.assert_allclose(self.model.proba_from_embeddings(text=self.text), self.sklearn_proba(self.text_fit, self.text))

    def test_image_only_matches_scikit_learn(self):
        np.testing.assert_allclose(self.model.proba_from_embeddings(image=self.image), self.sklearn_proba(self.image_fit, self.image))

    def test_tfidf_matches_scikit_learn(self):
        vocabulary = {t: i for i, t in enumerate(sorted(self.tfidf.vocabulary_, key=self.tfidf.vocabulary_.get))}
        for caption in CAPTIONS + ["love love LOVE this day!", "words never seen", ""]:
            np.testing.assert_allclose(tfidf_vector(caption, vocabulary, self.tfidf.idf_),
                                       self.tfidf.transform([caption]).toarray()[0])

    def test_fusion_averages_three_models_and_scales_neutral(self):
        expected = (self.sklearn_proba(self.text_fit, self.text) + self.sklearn_proba(self.image_fit, self.image)
                    + self.words_fit.predict_proba(self.words)) / 3
        expected[:, 1] *= 0.75
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(self.model.proba_from_embeddings(self.text, self.image, self.words.toarray()), expected)

    def test_probabilities_sum_to_one(self):
        np.testing.assert_allclose(self.model.proba_from_embeddings(self.text, self.image, self.words.toarray()).sum(axis=1), 1)

    def test_both_inputs_need_the_word_features(self):
        with self.assertRaises(ValueError):
            self.model.proba_from_embeddings(self.text, self.image)

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
        result = self.model.predict_embeddings(self.text[:1], self.image[:1], self.words[:1].toarray())
        self.assertEqual((result["readings"], result["conflict"]), ({}, False))

    def test_both_inputs_use_fusion_and_report_readings(self):
        m = self.model_with_readings("neutral", "positive")
        result = m.predict_embeddings(self.text[:1], self.image[:1], self.words[:1].toarray())
        expected = m.proba_from_embeddings(self.text[:1], self.image[:1], self.words[:1].toarray())[0]
        self.assertAlmostEqual(result["score"], expected.max())
        self.assertEqual({k: v["label"] for k, v in result["readings"].items()}, {"text": "neutral", "image": "positive"})
        self.assertFalse(result["conflict"])

    def test_one_input_uses_its_own_reading(self):
        result = self.model_with_readings("neutral", "positive").predict_embeddings(text=self.text[:1])
        self.assertEqual((result["label"], result["used"]), ("neutral", ["text"]))

    def test_opposite_readings_are_a_conflict(self):
        result = self.model_with_readings("negative", "positive").predict_embeddings(self.text[:1], self.image[:1],
                                                                                       self.words[:1].toarray())
        self.assertTrue(result["conflict"])

    def test_rejects_missing_input(self):
        with self.assertRaises(ValueError):
            self.model.proba_from_embeddings()


if __name__ == "__main__":
    unittest.main()
