"""Regression coverage for model-search completeness and honest bootstrap support."""

from itertools import combinations
import unittest
from unittest.mock import patch

import numpy as np

from amide_analysis.config import default_config
from amide_analysis.fitting import STRUCTURES, _bootstrap, _select_model, gaussian


class BootstrapComparisonTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.config["fitting"]["bootstrap_iterations"] = 4
        self.x = np.linspace(1600, 1700, 101)
        self.best = self.model(1, 10)
        self.processed = {"x": self.x, "corrected": self.best["fit"].copy()}
        self.fractions = {name: 100.0 if name == "alpha_helix" else 0.0
                          for name in STRUCTURES}
        self.percentage_result = (np.array([1.0]), np.array([100.0]),
                                  ["alpha_helix"], self.fractions)

    def model(self, count, criterion):
        centers = np.linspace(1630, 1674, count) if count > 1 else np.array([1655.0])
        params = np.column_stack((np.full(count, .5), centers, np.full(count, 10.0)))
        return {"params": params, "fit": sum(gaussian(self.x, *row) for row in params),
                "initial_centers": centers, "count": count, "bound_hits": False,
                "aicc": criterion, "bic": criterion, "sse": 1.0,
                "condition_number": 100.0, "nfev": 10}

    def execute(self, alternatives, side_effect):
        with patch("amide_analysis.fitting._bounded_fit", side_effect=side_effect) as fits:
            with patch("amide_analysis.fitting._percentages", return_value=self.percentage_result):
                result = _bootstrap(self.processed, self.best, alternatives,
                                    self.config, np.random.default_rng(1))
        return result, fits

    def test_failed_all_alternatives_are_indeterminate_not_support(self):
        alternative = self.model(2, 20)
        result, fits = self.execute([alternative], [self.best, None] * 4)
        self.assertEqual(result["success_fraction"], 1)
        self.assertEqual(result["success_count"], 4)
        self.assertEqual(result["count_support"], 0)
        self.assertEqual(result["count_comparison_fraction"], 0)
        self.assertEqual(fits.call_count, 8)

    def test_no_alternatives_support_is_primary_success_fraction(self):
        result, fits = self.execute([], [self.best, None, self.best, None])
        self.assertEqual(result["success_fraction"], .5)
        self.assertEqual(result["count_support"], .5)
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(fits.call_count, 4)

    def test_every_available_competing_count_is_compared(self):
        second, third = self.model(2, 20), self.model(3, 30)
        two_fit, three_fit = self.model(2, 11), self.model(3, 9)
        # K=2 does not defeat the selected K=1; K=3 does on every repetition.
        result, fits = self.execute([second, third], [self.best, two_fit, three_fit] * 4)
        self.assertEqual(result["success_fraction"], 1)
        self.assertEqual(result["count_comparison_fraction"], 1)
        self.assertEqual(result["count_support"], 0)
        self.assertEqual(fits.call_count, 12)

    def test_one_unavailable_competing_count_makes_comparison_indeterminate(self):
        second, third = self.model(2, 20), self.model(3, 30)
        result, _ = self.execute([second, third], [self.best, self.model(2, 11), None] * 4)
        self.assertEqual(result["success_fraction"], 1)
        self.assertEqual(result["count_comparison_fraction"], 0)
        self.assertEqual(result["count_support"], 0)

    def test_genuine_support_uses_all_iterations_not_only_successes(self):
        second = self.model(2, 20)
        losing_alternative = self.model(2, 11)
        # One primary fit fails: three successful wins out of four attempts.
        result, fits = self.execute([second], [self.best, losing_alternative,
                                               None, self.best, losing_alternative,
                                               self.best, losing_alternative])
        self.assertEqual(result["success_fraction"], .75)
        self.assertEqual(result["count_comparison_fraction"], .75)
        self.assertEqual(result["count_support"], .75)
        self.assertEqual(fits.call_count, 7)


class CandidateSubsetSearchTests(unittest.TestCase):
    def test_all_five_candidate_subsets_are_tested_and_weak_candidate_can_win(self):
        config = default_config()
        centers = [1625.0, 1635.0, 1655.0, 1670.0, 1690.0]
        candidates = [dict(center=center, stable=True, support=1.0, prominence=5-i)
                      for i, center in enumerate(centers)]
        x = np.linspace(1600, 1700, 101)
        processed = {"x": x, "corrected": np.ones_like(x)}
        visited = []

        def fitted(_x, _y, initial_centers, _config, _rng):
            key = tuple(sorted(initial_centers))
            visited.append(key)
            criterion = -10.0 if key == (1690.0,) else float(len(key))
            return dict(count=len(key), initial_centers=np.array(key),
                        aicc=criterion, bic=criterion, sse=1.0)

        with patch("amide_analysis.fitting.detect_candidates", return_value=candidates):
            with patch("amide_analysis.fitting._bounded_fit", side_effect=fitted) as fitter:
                best, models, returned_candidates = _select_model(processed, config,
                                                                  np.random.default_rng(2))
        expected = {subset for size in range(1, 6) for subset in combinations(centers, size)}
        self.assertEqual(fitter.call_count, 31)
        self.assertEqual(len(models), 31)
        self.assertEqual(set(visited), expected)
        self.assertEqual(len(visited), len(set(visited)))
        self.assertIs(returned_candidates, candidates)
        self.assertIs(best, models[0])
        np.testing.assert_array_equal(best["initial_centers"], [1690.0])
        self.assertEqual(best["aicc"], -10)


if __name__ == "__main__":
    unittest.main()
