"""Truth recovery and honest uncertainty on independently generated FTIR bands."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from amide_analysis import fitting
from amide_analysis.config import default_config
from examples.validate_overlap import (
    SAMPLING_STEP_CM1, analyze_case, summarize, synthetic_spectrum, validation_cases,
)


class OverlapTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use all default QC thresholds, multistarts and bootstrap iterations.
        # Only resolution confirmation differs between these declared synthetic
        # experiments; it must remain false for real files without metadata.
        cls.config = default_config()
        cls.cases = {case["name"]: case for case in validation_cases()}
        cls.results = {name: analyze_case(case, cls.config)
                       for name, case in cls.cases.items()}

    def assert_quantitative_estimates_withheld(self, fit):
        self.assertEqual(fit["status"], "Unreliable fit", fit["warnings"])
        self.assertTrue(all(np.isnan(value) for value in fit["percentages"].values()))
        self.assertTrue(all(np.isnan(row["area_pct"]) and not row["accepted"]
                            for row in fit["components"]))

    def test_full_spectrum_sampling_matches_export_without_claiming_resolution(self):
        raw, _ = synthetic_spectrum(self.cases["moderate_overlap"]["parameters"])
        self.assertEqual(raw["x"][0], 450.0)
        self.assertGreater(raw["x"][-1], 3998.0)
        np.testing.assert_allclose(np.diff(raw["x"]), SAMPLING_STEP_CM1,
                                   rtol=0, atol=1e-10)
        self.assertFalse(self.config["resolution_confirmed"])

    def test_separated_and_moderate_overlap_recover_true_components_and_areas(self):
        for name in ("well_separated", "moderate_overlap"):
            with self.subTest(case=name):
                fit, truth, _ = self.results[name]
                self.assertEqual(fit["status"], "Reliable fit", fit["warnings"])
                self.assertEqual(len(fit["components"]), 3)
                np.testing.assert_allclose(
                    [row["center"] for row in fit["components"]],
                    [1630.0, 1655.0, 1674.0], atol=0.5, rtol=0)
                np.testing.assert_allclose(
                    [row["area_pct"] for row in fit["components"]],
                    truth, atol=2.0, rtol=0)
                self.assertTrue(all(row["accepted"] for row in fit["components"]))
                self.assertGreaterEqual(fit["metrics"]["bootstrap_success_fraction"],
                                        self.config["fitting"]["bootstrap_success_fraction"])
                self.assertTrue(summarize(self.cases[name], fit, truth)["validation_passed"])

    def test_broad_four_component_overlap_needs_truth_recovery_or_rejection(self):
        # A high R-squared may conceal an incorrect number or area of peaks.
        # Accepting all broad bands is explicitly not a validation criterion.
        name = "broad_overlap"
        fit, truth, _ = self.results[name]
        if fit["status"] == "Reliable fit":
            self.assertEqual(len(fit["components"]), 4)
            np.testing.assert_allclose(
                [row["center"] for row in fit["components"]],
                [1632.0, 1654.0, 1671.0, 1683.0], atol=0.5, rtol=0)
            np.testing.assert_allclose(
                [row["area_pct"] for row in fit["components"]], truth,
                atol=2.0, rtol=0)
        else:
            self.assert_quantitative_estimates_withheld(fit)
            self.assertTrue(fit["warnings"])
        self.assertTrue(summarize(self.cases[name], fit, truth)["validation_passed"])

    def test_spurious_noise_peak_cannot_displace_an_admissible_single_peak_model(self):
        # At this declared noise seed the derivatives also detect a weak peak
        # near 1616 cm^-1. The original unconstrained information-score ranking
        # preferred an invalid two-component model: its noise component had
        # FWHM at the lower bound and only about 0.25% area. It rejected the
        # whole spectrum despite a well-identified single generating component.
        fit, truth, _ = self.results["single_peak_with_noise"]
        self.assertEqual(fit["status"], "Reliable fit", fit["warnings"])
        self.assertEqual(len(fit["components"]), 1)
        component = fit["components"][0]
        self.assertAlmostEqual(component["center"], 1655.0, delta=0.5)
        self.assertAlmostEqual(component["fwhm"], 20.0, delta=0.5)
        self.assertAlmostEqual(component["area_pct"], truth[0], delta=1e-8)
        self.assertTrue(component["accepted"])
        self.assertFalse(fit["metrics"]["bound_hits"])
        self.assertEqual(component["structure"], "alpha_helix")
        self.assertGreaterEqual(fit["metrics"]["bootstrap_success_fraction"],
                                self.config["fitting"]["bootstrap_success_fraction"])

    def test_assignment_boundary_does_not_force_a_structural_category(self):
        fit, _, _ = self.results["assignment_boundary"]
        if fit["status"] == "Reliable fit":
            self.assertTrue(fit["components"])
            self.assertTrue(all(row["structure"] == "unassigned"
                                for row in fit["components"]))
            self.assertAlmostEqual(fit["percentages"]["unassigned"], 100.0)
            self.assertTrue(all(value == 0.0 for name, value in fit["percentages"].items()
                                if name != "unassigned"))
        else:
            self.assert_quantitative_estimates_withheld(fit)

    def test_noise_only_is_never_accepted_as_secondary_structure(self):
        fit, _, _ = self.results["noise_only"]
        self.assert_quantitative_estimates_withheld(fit)

    def test_unknown_resolution_withholds_estimates_for_an_identifiable_band(self):
        fit, _, _ = self.results["unknown_resolution"]
        self.assert_quantitative_estimates_withheld(fit)
        self.assertTrue(any("разрешение" in warning for warning in fit["warnings"]))
        self.assertTrue(fit["components"])
        self.assertTrue(all(np.isfinite(row["diagnostic_area_pct"])
                            for row in fit["components"]))

    def test_large_area_uncertainty_cannot_be_hidden_by_good_fit_and_bootstrap_success(self):
        case = self.cases["moderate_overlap"]
        original = fitting._bootstrap

        def wide_area_samples(*args, **kwargs):
            boot = deepcopy(original(*args, **kwargs))
            # This isolates uncertainty acceptance from numerical convergence:
            # every successful simulation is retained, sums stay at 100%, and
            # centers/model-count support are unchanged. Area SD exceeds the
            # unchanged 8 percentage-point limit even with an excellent curve.
            count = len(boot["percentages"])
            delta = np.where(np.arange(count) % 2 == 0, 15.0, -15.0)
            boot["percentages"][:, 0] += delta
            boot["percentages"][:, 1] -= delta
            return boot

        with patch("amide_analysis.fitting._bootstrap", side_effect=wide_area_samples):
            fit, _, _ = analyze_case(case, self.config)
        self.assertGreater(fit["metrics"]["r_squared"], 0.99)
        self.assertGreaterEqual(fit["metrics"]["bootstrap_success_fraction"], 0.8)
        self.assertGreater(fit["metrics"]["max_area_pct_sd"],
                           self.config["fitting"]["max_percentage_sd"])
        self.assertTrue(any("неустойчивы в bootstrap" in warning
                            for warning in fit["warnings"]))
        self.assert_quantitative_estimates_withheld(fit)


if __name__ == "__main__":
    unittest.main()
