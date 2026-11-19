"""Unit tests for the numbers behind a prediction. These use small random data, so they don't need CLIP or the dataset."""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mvsa.explain import entropy, explain_embeddings, explain_example, js_divergence  # noqa: E402
from mvsa.predict import SentimentModel  # noqa: E402

# Small stand-ins for the real sizes (512-d embeddings, 5,000 terms), so the tests run instantly
DIMENSIONS = 12
TERMS = ["bad", "day", "good", "good day", "sad"]


def synthetic_model_and_stats():
    """A saved model and its statistics with random weights, in the same layout as the real files."""
    rng = np.random.default_rng(0)
    arrays = {"labels": np.array(["negative", "neutral", "positive"]), "neutral_multiplier": 0.75, "clip_model": "unused",
              "words_terms": np.array(TERMS), "words_idf": np.ones(len(TERMS)),
              "words_coef": rng.normal(size=(3, len(TERMS))), "words_intercept": rng.normal(size=3)}
    # Random weights and scalers for the two fused models and the two separate readings
    for name in ("text", "image", "own_text", "own_image"):
        arrays.update({f"{name}_mean": rng.normal(size=DIMENSIONS), f"{name}_scale": rng.uniform(0.5, 2, DIMENSIONS),
                       f"{name}_coef": rng.normal(size=(3, DIMENSIONS)), f"{name}_intercept": rng.normal(size=3)})
    path = os.path.join(tempfile.mkdtemp(), "model.npz")
    np.savez(path, **arrays)
    # Statistics in the same layout as explainer_stats.npz; the map axes are the first two dimensions
    stats = {"similarity_sorted": np.linspace(-0.5, 0.5, 101), "example_id": np.array("7"),
             "example_caption": np.array("good day"), "example_image_url": np.array("https://example.org/7.jpg"),
             "example_text": rng.normal(size=DIMENSIONS), "example_image": rng.normal(size=DIMENSIONS)}
    for name in ("text", "image"):
        stats.update({f"map_{name}_mean": np.zeros(DIMENSIONS), f"map_{name}_axes": np.eye(DIMENSIONS)[:2],
                      f"map_{name}_variance": np.array([0.6, 0.2]), f"map_{name}_points": rng.normal(size=(5, 2)),
                      f"map_{name}_labels": np.array([0, 1, 2, 0, 1])})
    return SentimentModel(path), stats, rng


class ExplainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.stats, rng = synthetic_model_and_stats()
        cls.text, cls.image = rng.normal(size=(1, DIMENSIONS)), rng.normal(size=(1, DIMENSIONS))
        cls.caption = "good day, not a bad day"
        cls.words = cls.model.words_vector(cls.caption)
        # One explanation with every input given, shared by most tests below
        cls.full = explain_embeddings(cls.model, cls.stats, cls.caption, cls.text, cls.image, cls.words)

    def test_pushes_add_up_to_the_class_scores(self):
        for name in ("text", "image"):
            branch = self.full["branches"][name]
            # The listed dimensions, the rest and the intercept must add up to each class score
            total = np.sum([t["push"] for t in branch["top"]], axis=0) + branch["rest"] + branch["intercept"]
            np.testing.assert_allclose(total, branch["logits"], atol=2e-3)
        words = self.full["branches"]["words"]
        total = np.sum([t["push"] for t in self.full["terms"]], axis=0) + words["intercept"]
        np.testing.assert_allclose(total, words["logits"], atol=5e-3)

    def test_listed_dimensions_are_the_strongest_for_the_winning_class(self):
        branch = self.full["branches"]["image"]
        winner = int(np.argmax(branch["logits"]))
        strengths = [abs(t["push"][winner]) for t in branch["top"]]
        self.assertEqual(strengths, sorted(strengths, reverse=True))
        self.assertEqual(len(branch["z"]), DIMENSIONS)

    def test_steps_match_the_model(self):
        expected = self.model.proba_from_embeddings(self.text, self.image, self.words)[0]
        np.testing.assert_allclose(self.full["final"], expected, atol=1e-4)
        np.testing.assert_allclose(np.sum(self.full["average"]), 1, atol=1e-3)
        # Repeat the combination by hand: neutral x 0.75, then renormalise
        scaled = np.array(self.full["average"]) * [1, 0.75, 1]
        np.testing.assert_allclose(self.full["scaled"], scaled, atol=1e-4)
        np.testing.assert_allclose(self.full["final"], scaled / scaled.sum(), atol=1e-3)
        self.assertEqual(self.full["label"], self.model.labels[int(np.argmax(expected))])

    def test_agreement_measures(self):
        agreement = self.full["agreement"]
        cos = float(self.text[0] @ self.image[0] / (np.linalg.norm(self.text) * np.linalg.norm(self.image)))
        self.assertAlmostEqual(agreement["cos"], cos, places=3)
        # cos_pct should be the share of training posts with a lower similarity
        share_below = float((self.stats["similarity_sorted"] < cos).mean() * 100)
        self.assertAlmostEqual(agreement["cos_pct"], share_below, delta=1.0)
        self.assertEqual(sum(agreement["hist"]), 101)
        self.assertEqual(agreement["posts"], 101)
        self.assertTrue(0 <= agreement["js"] <= 1)

    def test_entropy_and_divergence_limits(self):
        self.assertAlmostEqual(entropy([1, 0, 0]), 0)
        self.assertAlmostEqual(entropy([1 / 3] * 3), math.log2(3))
        self.assertAlmostEqual(js_divergence([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0)
        self.assertAlmostEqual(js_divergence([1, 0, 0], [0, 0, 1]), 1)

    def test_caption_alone_has_no_fusion_steps(self):
        alone = explain_embeddings(self.model, self.stats, self.caption, self.text, None, self.words)
        self.assertNotIn("average", alone)
        self.assertNotIn("image", alone["branches"])
        self.assertNotIn("cos", alone["agreement"])
        self.assertIsNone(alone["maps"]["image"]["me"])
        self.assertEqual(alone["final"], alone["readings"]["text"]["proba"])

    def test_picture_alone_has_no_caption_parts(self):
        alone = explain_embeddings(self.model, self.stats, None, None, self.image, None)
        self.assertEqual(list(alone["branches"]), ["image"])
        self.assertEqual(alone["terms"], [])
        self.assertEqual(alone["tokens"], [])
        self.assertEqual(alone["used"], ["image"])

    def test_map_places_the_post(self):
        here = self.full["maps"]["text"]["me"]
        np.testing.assert_allclose(here, self.text[0, :2], atol=1e-3)   # the test axes are the first two dimensions
        self.assertEqual(len(self.full["maps"]["text"]["pts"]), 5)

    def test_example_needs_no_clip(self):
        example = explain_example(self.model, self.stats)
        self.assertTrue(example["example"])
        self.assertEqual(example["caption"], "good day")
        self.assertEqual(example["image_url"], "https://example.org/7.jpg")
        self.assertIn("average", example)


if __name__ == "__main__":
    unittest.main()
