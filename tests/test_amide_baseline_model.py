"""Broad bands retain Gaussian tails after an arbitrary affine baseline removal.

Known physical components are generated before preprocessing. Recovering those
components verifies that the model accounts for the line removed by the input
baseline procedure, including real signal present at the band endpoints.
"""

from copy import deepcopy
import math
import unittest

import numpy as np
from scipy.integrate import quad

from amide_analysis.config import default_config
from amide_analysis import fitting
from amide_analysis.preprocess import preprocess_spectrum


class JointBaselineModelTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.config["resolution_confirmed"] = True
        self.config["fitting"]["multistarts"] = 3
        self.x = np.arange(1500.0, 1800.5, 0.5)
        self.truth = np.array([[0.6, 1633.0, 28.0], [0.8, 1667.0, 32.0]])
        self.physical_signal = sum(fitting.gaussian(self.x, *row) for row in self.truth)
        self.background = 0.12 + 0.0004 * (self.x - 1650.0)

    def preprocess(self, edge_points=3):
        config = deepcopy(self.config)
        config["baseline"]["edge_points"] = edge_points
        return preprocess_spectrum(self.x, self.physical_signal + self.background, config)

    def fit(self, x, signal):
        model = fitting._bounded_fit(x, signal, [1630.0, 1670.0], self.config,
                                     np.random.default_rng(83), starts=3)
        self.assertIsNotNone(model)
        self.assertTrue(model["admissible"], model["admissibility_reasons"])
        return model

    def test_broad_gaussian_parameters_are_recovered_with_nonzero_endpoint_tails(self):
        processed = self.preprocess()
        physical_band = sum(fitting.gaussian(processed["x"], *row) for row in self.truth)
        # Endpoint signal is large enough that assuming it to be zero biases
        # the fitted widths and areas in a pure-Gaussian corrected-signal fit.
        self.assertGreater(np.max(physical_band[[0, -1]]), 0.02)
        model = self.fit(processed["x"], processed["corrected"])
        np.testing.assert_allclose(model["params"], self.truth, atol=0.002)
        areas = fitting._areas(model, self.config["amide_bounds"])
        truth_areas = [quad(lambda value: float(fitting.gaussian(value, *row)),
                            1600.0, 1700.0)[0] for row in self.truth]
        np.testing.assert_allclose(areas, truth_areas, rtol=1e-4)

    def test_changing_endpoint_count_does_not_change_noiseless_physical_components(self):
        three = self.preprocess(edge_points=3)
        five = self.preprocess(edge_points=5)
        self.assertGreater(np.max(np.abs(three["corrected"] - five["corrected"])), 1e-4)
        first = self.fit(three["x"], three["corrected"])
        second = self.fit(five["x"], five["corrected"])
        np.testing.assert_allclose(first["params"], self.truth, atol=0.002)
        np.testing.assert_allclose(second["params"], self.truth, atol=0.002)
        np.testing.assert_allclose(first["params"], second["params"], atol=0.002)

    def test_returned_curve_includes_an_affine_adjustment_but_components_remain_gaussians(self):
        processed = self.preprocess()
        model = self.fit(processed["x"], processed["corrected"])
        adjustment = np.asarray(model["baseline_adjustment"])
        self.assertEqual(adjustment.shape, processed["x"].shape)
        self.assertGreater(np.max(np.abs(adjustment)), 0.02)
        np.testing.assert_allclose(np.diff(adjustment, n=2), 0.0, atol=1e-10)
        gaussian_sum = fitting._model(processed["x"], model["params"])
        np.testing.assert_allclose(model["fit"], gaussian_sum + adjustment, atol=1e-10)
        np.testing.assert_allclose(model["fit"], processed["corrected"], atol=1e-6)
        self.assertGreater(np.max(np.abs(gaussian_sum - model["fit"])), 0.02)

    def test_information_criteria_include_two_fitted_baseline_parameters(self):
        processed = self.preprocess()
        noise = np.random.default_rng(42).normal(0.0, 0.0002, len(processed["x"]))
        signal = processed["corrected"] + noise
        model = self.fit(processed["x"], signal)
        residual_sse = float(np.sum((model["fit"] - signal) ** 2))
        self.assertGreater(residual_sse, 0.0)
        self.assertAlmostEqual(model["sse"], residual_sse, places=12)
        # Six Gaussian parameters, two nuisance-line parameters, and variance.
        k, n = 3 * len(self.truth) + 3, len(signal)
        likelihood = n * math.log(residual_sse / n)
        expected_aicc = likelihood + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        expected_bic = likelihood + k * math.log(n)
        self.assertAlmostEqual(model["aicc"], expected_aicc, places=6)
        self.assertAlmostEqual(model["bic"], expected_bic, places=6)

    def test_offset_and_slope_of_input_do_not_change_recovered_components(self):
        x = self.preprocess()["x"]
        gaussian_sum = sum(fitting.gaussian(x, *row) for row in self.truth)
        added_background = 0.35 + 0.001 * (x - 1650.0)
        plain = self.fit(x, gaussian_sum)
        shifted = self.fit(x, gaussian_sum + added_background)
        np.testing.assert_allclose(plain["params"], self.truth, atol=0.002)
        np.testing.assert_allclose(shifted["params"], self.truth, atol=0.002)
        np.testing.assert_allclose(plain["params"], shifted["params"], atol=0.002)
        np.testing.assert_allclose(shifted["fit"] - plain["fit"], added_background, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
