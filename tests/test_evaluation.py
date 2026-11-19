"""Unit tests for src/mvsa/evaluation.py, on small hand-made arrays."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mvsa.evaluation import (apply_temperature, bootstrap_scores, expected_calibration_error,  # noqa: E402
                             fit_temperature, interval, mcnemar, reliability_table, risk_coverage,
                             stratified_subsample)


class BootstrapTests(unittest.TestCase):
    def test_perfect_model_scores_one_on_every_resample(self):
        y = np.array([0, 1, 2] * 20)
        boot = bootstrap_scores(y, {"perfect": y.copy()}, n_boot=50)
        self.assertEqual(boot["perfect"].shape, (50, 2))
        self.assertTrue(np.allclose(boot["perfect"], 1.0))

    def test_resamples_are_paired_and_repeatable(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, 100)
        a, b = rng.integers(0, 3, 100), rng.integers(0, 3, 100)
        one = bootstrap_scores(y, {"a": a, "b": b}, n_boot=30, seed=4)
        two = bootstrap_scores(y, {"a": a}, n_boot=30, seed=4)
        np.testing.assert_allclose(one["a"], two["a"])   # same seed, same resamples, whatever else is scored

    def test_interval_brackets_the_middle(self):
        low, high = interval(np.arange(1001))
        self.assertAlmostEqual(low, 25)
        self.assertAlmostEqual(high, 975)


class McNemarTests(unittest.TestCase):
    def test_counts_and_p_value(self):
        y = np.zeros(20, dtype=int)
        a = np.zeros(20, dtype=int)                 # always right
        b = np.r_[np.ones(10, dtype=int), np.zeros(10, dtype=int)]   # wrong on the first 10
        only_a, only_b, p = mcnemar(y, a, b)
        self.assertEqual((only_a, only_b), (10, 0))
        self.assertAlmostEqual(p, 2 * 0.5 ** 10)

    def test_identical_models_give_p_of_one(self):
        y = np.array([0, 1, 2, 1])
        self.assertEqual(mcnemar(y, y, y), (0, 0, 1.0))


class CalibrationTests(unittest.TestCase):
    def test_confident_and_right_is_perfectly_calibrated(self):
        y = np.array([0, 1, 2, 0])
        P = np.eye(3)[y]
        self.assertAlmostEqual(expected_calibration_error(y, P), 0.0)

    def test_overconfident_model_has_large_error(self):
        y = np.array([0, 0, 0, 0])
        P = np.tile([0.05, 0.9, 0.05], (4, 1))     # 90% sure of the wrong class every time
        self.assertAlmostEqual(expected_calibration_error(y, P), 0.9)

    def test_reliability_table_counts_every_post_once(self):
        rng = np.random.default_rng(0)
        P = rng.dirichlet(np.ones(3), 200)
        y = rng.integers(0, 3, 200)
        self.assertEqual(reliability_table(y, P)["posts"].sum(), 200)

    def test_temperature_keeps_the_top_class(self):
        P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
        for T in (0.5, 1.0, 3.0):
            Q = apply_temperature(P, T)
            np.testing.assert_array_equal(Q.argmax(1), P.argmax(1))
            np.testing.assert_allclose(Q.sum(1), 1)
        np.testing.assert_allclose(apply_temperature(P, 1.0), P)

    def test_fitted_temperature_softens_an_overconfident_model(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 500)
        pred = np.where(rng.random(500) < 0.6, y, rng.integers(0, 3, 500))   # right about 70% of the time
        P = np.full((500, 3), 0.01)
        P[np.arange(500), pred] = 0.98                                        # but always 98% sure
        T = fit_temperature(y, P)
        self.assertGreater(T, 1)
        self.assertLess(expected_calibration_error(y, apply_temperature(P, T)), expected_calibration_error(y, P))


class SelectiveAndSubsampleTests(unittest.TestCase):
    def test_coverage_falls_as_the_threshold_rises(self):
        rng = np.random.default_rng(0)
        P = rng.dirichlet(np.ones(3), 300)
        y = rng.integers(0, 3, 300)
        table = risk_coverage(y, P, [0.0, 0.4, 0.6, 0.8])
        self.assertEqual(table["coverage"].iloc[0], 1.0)
        self.assertTrue((np.diff(table["coverage"]) <= 0).all())

    def test_subsample_keeps_class_shares(self):
        y = np.r_[np.zeros(600), np.ones(100), np.full(300, 2)].astype(int)
        idx = stratified_subsample(y, 0.1, seed=0)
        self.assertEqual(np.bincount(y[idx]).tolist(), [60, 10, 30])
        self.assertEqual(len(set(idx)), len(idx))


if __name__ == "__main__":
    unittest.main()
