"""Regression coverage for model-search completeness and honest bootstrap support."""

from itertools import combinations
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from amide_analysis.config import default_config
from amide_analysis.fitting import STRUCTURES, _bootstrap, _select_model, gaussian


class BootstrapComparisonTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.config["fitting"]["bootstrap_iterations"] = 4
        self.config["fitting"]["bootstrap_retry_multistarts"] = 1
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
                "condition_number": 100.0, "nfev": 10,
                "admissible": True, "admissibility_reasons": []}

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

    def test_failed_warm_start_is_retried_with_independent_initializations(self):
        self.config["fitting"]["bootstrap_retry_multistarts"] = 3
        result, fits = self.execute([], [None, self.best] * 4)
        self.assertEqual(result["success_fraction"], 1)
        self.assertEqual(result["count_support"], 1)
        self.assertEqual(fits.call_count, 8)
        for call in fits.call_args_list[1::2]:
            self.assertEqual(call.kwargs["starts"], 3)
            self.assertIsNone(call.kwargs.get("initial"))

    def test_competing_numerical_failure_still_never_votes_after_retry(self):
        self.config["fitting"]["bootstrap_retry_multistarts"] = 3
        alternative = self.model(2, 20)
        result, fits = self.execute([alternative], [self.best, None, None] * 4)
        self.assertEqual(result["success_fraction"], 1)
        self.assertEqual(result["count_comparison_fraction"], 0)
        self.assertEqual(result["count_support"], 0)
        self.assertEqual(fits.call_count, 12)
        self.assertTrue(all(row["comparison_status"] == "indeterminate"
                            for row in result["diagnostics"]))
        self.assertTrue(all(str(row["unavailable_counts"]) == "2"
                            for row in result["diagnostics"]))

    def test_physically_invalid_primary_cannot_count_as_bootstrap_success(self):
        invalid = deepcopy(self.best)
        invalid["admissible"] = False
        invalid["admissibility_reasons"] = ["WIDTH_LOWER_BOUND"]
        invalid["bound_hits"] = True
        result, _ = self.execute([], [invalid] * 4)
        self.assertEqual(result["success_fraction"], 0)
        self.assertEqual(result["count_support"], 0)
        self.assertEqual(result["success_count"], 0)
        self.assertTrue(all(row["primary_fit_status"] == "inadmissible"
                            for row in result["diagnostics"]))
        self.assertTrue(all("WIDTH_LOWER_BOUND" in row["primary_fit_reasons"]
                            for row in result["diagnostics"]))

    def test_numerical_primary_failure_has_distinct_diagnostic_reason(self):
        result, _ = self.execute([], [None] * 4)
        self.assertEqual(result["success_count"], 0)
        self.assertEqual(len(result["diagnostics"]), 4)
        self.assertTrue(all(row["primary_fit_status"] == "optimizer_failed"
                            for row in result["diagnostics"]))
        self.assertTrue(all("OPTIMIZER_FAILURE" in row["primary_fit_reasons"]
                            for row in result["diagnostics"]))


class CandidateSubsetSearchTests(unittest.TestCase):
    def test_all_five_candidate_subsets_are_tested_and_weak_candidate_can_win(self):
        config = default_config()
        config["fitting"]["multistarts"] = 1
        centers = [1625.0, 1635.0, 1655.0, 1670.0, 1690.0]
        candidates = [dict(center=center, stable=True, support=1.0, prominence=5-i)
                      for i, center in enumerate(centers)]
        x = np.linspace(1600, 1700, 101)
        processed = {"x": x, "corrected": np.ones_like(x)}
        visited = []

        def fitted(_x, _y, initial_centers, _config, _rng, **kwargs):
            key = tuple(sorted(initial_centers))
            visited.append(key)
            criterion = -10.0 if key == (1690.0,) else float(len(key))
            return dict(count=len(key), initial_centers=np.array(key),
                        aicc=criterion, bic=criterion, sse=1.0,
                        admissible=True, admissibility_reasons=[])

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

    def test_remaining_start_budget_refines_once_per_count(self):
        config = default_config()
        # Every K has fewer than eight candidate subsets here. Only the
        # remaining per-K budget needs a refinement call.
        config["fitting"]["multistarts"] = 8
        centers = [1625.0, 1640.0, 1655.0, 1680.0]
        candidates = [dict(center=center, stable=True, support=1.0, prominence=4-index)
                      for index, center in enumerate(centers)]
        x = np.linspace(1600, 1700, 101)
        processed = {"x": x, "corrected": np.ones_like(x)}

        def fitted(_x, _y, initial_centers, _config, _rng, **kwargs):
            key = tuple(sorted(initial_centers))
            return dict(count=len(key), initial_centers=np.array(key),
                        params=np.column_stack((np.full(len(key), 0.5), key,
                                                np.full(len(key), 12.0))),
                        aicc=float(len(key)), bic=float(len(key)), sse=1.0,
                        admissible=True, admissibility_reasons=[])

        with patch("amide_analysis.fitting.detect_candidates", return_value=candidates):
            with patch("amide_analysis.fitting._bounded_fit", side_effect=fitted) as fitter:
                _select_model(processed, config, np.random.default_rng(3))
        first_pass = [call for call in fitter.call_args_list if call.kwargs.get("starts") == 1]
        refinements = [call for call in fitter.call_args_list if call.kwargs.get("starts", 1) > 1]
        expected = {subset for size in range(1, 5) for subset in combinations(centers, size)}
        self.assertEqual({tuple(sorted(call.args[2])) for call in first_pass}, expected)
        self.assertEqual(len(first_pass), len(expected))
        self.assertEqual(len(refinements), 4)
        self.assertEqual({len(call.args[2]) for call in refinements}, {1, 2, 3, 4})

    def test_failed_first_subset_still_uses_remaining_search_budget(self):
        config = default_config()
        config["fitting"]["multistarts"] = 4
        x = np.linspace(1600, 1700, 101)
        processed = {"x": x, "corrected": np.ones_like(x)}
        candidates = [dict(center=1655.0, stable=True, support=1.0, prominence=1.0)]
        recovered = {"count": 1, "initial_centers": np.array([1655.0]),
                     "params": np.array([[0.5, 1655.0, 12.0]]),
                     "aicc": -100.0, "bic": -90.0, "sse": 0.1,
                     "admissible": True, "admissibility_reasons": []}
        with patch("amide_analysis.fitting.detect_candidates", return_value=candidates):
            with patch("amide_analysis.fitting._bounded_fit", side_effect=[None, recovered]) as fitter:
                best, models, _ = _select_model(processed, config, np.random.default_rng(4))
        self.assertIs(best, recovered)
        self.assertIn(recovered, models)
        self.assertEqual(fitter.call_count, 2)
        self.assertEqual(fitter.call_args_list[1].kwargs["starts"], 3)


if __name__ == "__main__":
    unittest.main()
