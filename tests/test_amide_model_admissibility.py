"""Regression checks for genuine center constraints and admissible model selection.

The numerical cases use known Gaussian spectra, so shifts and ordering can be
checked against independent truths rather than an optimizer's implementation.
"""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from amide_analysis.config import default_config
from amide_analysis import fitting


class PhysicalCenterFitTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.config["resolution_confirmed"] = True
        self.x = np.linspace(1600.0, 1700.0, 101)

    def test_derivative_anchor_does_not_clip_a_legitimate_center_shift(self):
        truth = np.array([[0.8, 1632.0, 14.0]])
        signal = fitting.gaussian(self.x, *truth[0])
        fit = fitting._bounded_fit(self.x, signal, [1645.0], self.config,
                                   np.random.default_rng(40), starts=3)
        self.assertIsNotNone(fit)
        self.assertTrue(fit["admissible"], fit["admissibility_reasons"])
        self.assertFalse(fit["bound_hits"])
        np.testing.assert_allclose(fit["params"], truth, atol=0.005)
        self.assertGreater(abs(fit["params"][0, 1] - 1645.0), 6.0)

    def test_reversed_initial_component_rows_keep_centers_and_areas_aligned(self):
        truth = np.array([[0.7, 1628.0, 14.0], [0.4, 1655.0, 12.0],
                          [0.25, 1680.0, 10.0]])
        signal = sum(fitting.gaussian(self.x, *row) for row in truth)
        reversed_truth = truth[::-1].copy()
        fit = fitting._bounded_fit(self.x, signal, reversed_truth[:, 1],
                                   self.config, np.random.default_rng(41),
                                   starts=1, initial=reversed_truth)
        self.assertIsNotNone(fit)
        self.assertTrue(fit["admissible"], fit["admissibility_reasons"])
        np.testing.assert_allclose(fit["params"], truth, atol=0.005)
        self.assertTrue(np.all(np.diff(fit["params"][:, 1]) > 0))

    def test_close_initial_candidates_cannot_bypass_configured_center_separation(self):
        truth = np.array([[0.7, 1640.0, 14.0], [0.4, 1660.0, 12.0]])
        signal = sum(fitting.gaussian(self.x, *row) for row in truth)
        fit = fitting._bounded_fit(self.x, signal, [1648.0, 1650.0],
                                   self.config, np.random.default_rng(42), starts=4)
        self.assertIsNotNone(fit)
        required_separation = max(self.config["peaks"]["min_separation_cm1"],
                                  self.config["spectral_resolution_cm1"],
                                  2 * np.median(np.diff(self.x)))
        self.assertTrue(np.all(np.diff(fit["params"][:, 1]) >= required_separation - 1e-8))


class AdmissibleModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.x = np.linspace(1600.0, 1700.0, 101)
        self.processed = {"x": self.x, "corrected": np.ones_like(self.x)}
        self.candidates = [dict(center=center, stable=True, support=1.0,
                                prominence=prominence)
                           for center, prominence in ((1630.0, 2.0), (1660.0, 1.0))]

    def model(self, count, criterion, admissible):
        centers = np.array([1655.0]) if count == 1 else np.array([1630.0, 1660.0])
        params = np.column_stack((np.full(count, 0.5), centers, np.full(count, 12.0)))
        return {"count": count, "params": params, "initial_centers": centers,
                "fit": sum(fitting.gaussian(self.x, *row) for row in params),
                "aicc": criterion, "bic": criterion, "sse": 0.1,
                "condition_number": 100.0, "bound_hits": not admissible,
                "admissible": admissible,
                "admissibility_reasons": [] if admissible else ["WIDTH_LOWER_BOUND"],
                "nfev": 10}

    def test_physically_invalid_lowest_aicc_cannot_defeat_admissible_model(self):
        invalid = self.model(1, -1000.0, False)
        valid = self.model(2, -100.0, True)

        def fit_by_count(_x, _y, centers, _config, _rng, **_kwargs):
            return deepcopy(valid if len(centers) == 2 else invalid)

        with patch("amide_analysis.fitting.detect_candidates", return_value=self.candidates):
            with patch("amide_analysis.fitting._bounded_fit", side_effect=fit_by_count):
                best, models, _ = fitting._select_model(self.processed, self.config,
                                                        np.random.default_rng(43))
        self.assertIsNotNone(best)
        self.assertTrue(best["admissible"])
        self.assertEqual(best["count"], 2)
        self.assertEqual(best["aicc"], -100.0)
        self.assertTrue(any(not model["admissible"] for model in models))

    def test_no_admissible_model_is_retained_only_as_diagnostic(self):
        def invalid_fit(_x, _y, centers, _config, _rng, **_kwargs):
            return self.model(len(centers), -1000.0, False)

        with patch("amide_analysis.fitting.detect_candidates", return_value=self.candidates):
            with patch("amide_analysis.fitting._bounded_fit", side_effect=invalid_fit):
                best, models, _ = fitting._select_model(self.processed, self.config,
                                                        np.random.default_rng(44))
        self.assertIsNotNone(best)
        self.assertFalse(best["admissible"])
        self.assertIn("WIDTH_LOWER_BOUND", best["admissibility_reasons"])
        self.assertTrue(models)
        self.assertTrue(all(not model["admissible"] for model in models))


class PhysicalAdmissibilityReasonsTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.x = np.linspace(1600.0, 1700.0, 101)

    def model(self, params, condition=100.0):
        params = np.asarray(params, dtype=float)
        return {"params": params, "count": len(params),
                "initial_centers": params[:, 1].copy(),
                "fit": sum(fitting.gaussian(self.x, *row) for row in params),
                "condition_number": condition, "bound_hits": False,
                "aicc": -100.0, "bic": -90.0, "sse": 0.001, "nfev": 20}

    def reasons(self, model):
        return fitting._model_admissibility(model, self.x, self.config)

    def test_interior_known_component_has_no_physical_rejection(self):
        self.assertEqual(self.reasons(self.model([[0.7, 1655.0, 12.0]])), [])

    def test_width_floor_and_ceiling_remain_genuine_rejection_reasons(self):
        lower = self.config["peaks"]["min_fwhm_cm1"]
        upper = self.config["peaks"]["max_fwhm_cm1"]
        self.assertIn("WIDTH_LOWER_BOUND", self.reasons(self.model([[0.7, 1655.0, lower]])))
        self.assertIn("WIDTH_UPPER_BOUND", self.reasons(self.model([[0.7, 1655.0, upper]])))

    def test_band_edge_center_is_distinguished_from_old_derivative_window(self):
        self.assertIn("BAND_EDGE_CENTER", self.reasons(self.model([[0.7, 1600.0, 12.0]])))

    def test_negligible_area_component_is_rejected_before_aicc_competition(self):
        model = self.model([[0.7, 1630.0, 12.0], [1e-5, 1660.0, 12.0]])
        self.assertIn("WEAK_COMPONENT", self.reasons(model))

    def test_unresolved_centers_remain_rejected_despite_good_total_curve(self):
        model = self.model([[0.7, 1653.0, 12.0], [0.4, 1655.0, 12.0]])
        self.assertIn("UNRESOLVED_CENTER_SEPARATION", self.reasons(model))

    def test_condition_number_limit_is_not_weakened(self):
        model = self.model([[0.7, 1630.0, 12.0], [0.4, 1660.0, 12.0]],
                           condition=2 * self.config["fitting"]["max_condition_number"])
        self.assertIn("ILL_CONDITIONED", self.reasons(model))


if __name__ == "__main__":
    unittest.main()
