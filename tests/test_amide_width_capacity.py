"""Independent broad-band truths distinguish search limits from fit evidence.

The generating curves and numerical area integrals do not use the fitting
module's Gaussian helpers. These checks cover widths beyond the former
40 cm^-1 search ceiling without changing identifiability or uncertainty QC.
"""

from copy import deepcopy
import unittest

import numpy as np
from scipy.integrate import quad

from amide_analysis.config import default_config
from amide_analysis import fitting


def independent_curve(x, parameters):
    return sum(amplitude * np.exp(-4 * np.log(2) * ((x - center) / width) ** 2)
               for amplitude, center, width in parameters)


def independent_percentages(parameters):
    areas = np.array([
        quad(lambda x: amplitude * np.exp(
            -4 * np.log(2) * ((x - center) / width) ** 2), 1600, 1700)[0]
        for amplitude, center, width in parameters
    ])
    return 100 * areas / areas.sum()


class BroadGaussianSearchCapacityTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(1600.0, 1700.0, 1.92905)
        self.config = default_config()
        self.config["resolution_confirmed"] = True

    def signal(self, parameters, noise=0.0):
        baseline = 0.15 + 0.00005 * (self.x - 1650)
        return independent_curve(self.x, parameters) + baseline + (
            np.random.default_rng(9191).normal(0, noise, len(self.x)))

    def fit(self, parameters, maximum=None, noise=0.0):
        config = deepcopy(self.config)
        if maximum is not None:
            config["peaks"]["max_fwhm_cm1"] = maximum
        signal = self.signal(parameters, noise)
        result = fitting._bounded_fit(
            self.x, signal, np.asarray(parameters)[:, 1], config,
            np.random.default_rng(91), starts=6)
        self.assertIsNotNone(result)
        return result, signal, config

    def test_default_search_recovers_broad_known_components_with_noise(self):
        # The default search must represent these ordinary broad Gaussian
        # truths; treating a 40 cm^-1 numerical ceiling as physics cannot.
        for width in (45.0, 55.0):
            truth = np.array([[0.08, 1654.0, width]])
            with self.subTest(true_fwhm=width):
                model, _, _ = self.fit(truth, noise=0.0003)
                self.assertTrue(model["admissible"], model["admissibility_reasons"])
                np.testing.assert_allclose(model["params"][:, 0], truth[:, 0], atol=0.001)
                np.testing.assert_allclose(model["params"][:, 1], truth[:, 1], atol=0.25)
                np.testing.assert_allclose(model["params"][:, 2], truth[:, 2], atol=0.5)
                self.assertLess(np.sqrt(model["sse"] / len(self.x)), 0.0004)

    def test_narrow_ceiling_rejects_even_exact_broad_gaussian_truth(self):
        for width in (45.0, 55.0):
            truth = np.array([[0.08, 1654.0, width]])
            with self.subTest(true_fwhm=width):
                narrow, _, _ = self.fit(truth, maximum=40.0)
                adequate, _, _ = self.fit(truth, maximum=80.0)
                self.assertFalse(narrow["admissible"])
                self.assertIn("WIDTH_UPPER_BOUND", narrow["admissibility_reasons"])
                self.assertTrue(adequate["admissible"], adequate["admissibility_reasons"])
                np.testing.assert_allclose(adequate["params"], truth, atol=0.001)
                self.assertLess(adequate["sse"], narrow["sse"] * 1e-8)

    def test_wider_search_ceilings_agree_on_two_component_truth_and_areas(self):
        truth = np.array([[0.06, 1634.0, 45.0], [0.04, 1674.0, 30.0]])
        models = []
        for ceiling in (60.0, 80.0):
            model, _, config = self.fit(truth, maximum=ceiling, noise=0.0003)
            self.assertTrue(model["admissible"], model["admissibility_reasons"])
            np.testing.assert_allclose(model["params"][:, 1], truth[:, 1], atol=0.25)
            np.testing.assert_allclose(model["params"][:, 2], truth[:, 2], atol=0.5)
            _, area_pct, _, _ = fitting._percentages(model, config)
            np.testing.assert_allclose(area_pct, independent_percentages(truth), atol=1.0)
            models.append(model)
        np.testing.assert_allclose(models[0]["params"], models[1]["params"],
                                   atol=0.001, rtol=0)

    def test_broad_resolved_truth_has_stable_bootstrap_areas(self):
        truth = np.array([[0.06, 1634.0, 45.0], [0.04, 1674.0, 30.0]])
        model, signal, config = self.fit(truth, maximum=80.0, noise=0.0003)
        config["fitting"]["bootstrap_iterations"] = 30
        bootstrap = fitting._bootstrap(
            {"x": self.x, "corrected": signal}, model, [], config,
            np.random.default_rng(91))
        self.assertGreaterEqual(bootstrap["success_fraction"], 0.9)
        np.testing.assert_allclose(
            np.mean(bootstrap["percentages"], axis=0),
            independent_percentages(truth), atol=1.0)
        self.assertTrue(np.all(np.std(bootstrap["percentages"], axis=0, ddof=1) < 2.0))

    def test_ceiling_hit_remains_a_rejection_with_a_larger_search(self):
        # Expanding the search does not silently accept an unresolved tail
        # whose optimum still lies outside the configured model capacity.
        truth = np.array([[0.08, 1654.0, 100.0]])
        model, _, _ = self.fit(truth, maximum=80.0)
        self.assertFalse(model["admissible"])
        self.assertIn("WIDTH_UPPER_BOUND", model["admissibility_reasons"])


if __name__ == "__main__":
    unittest.main()
