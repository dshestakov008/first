"""Regression checks for selection eligibility and the reported estimand."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from amide_analysis.config import default_config
from amide_analysis.fitting import STRUCTURES, analyze_band, gaussian


class AnalyzeStatusTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.config["resolution_confirmed"] = True
        self.config["fitting"]["bootstrap_iterations"] = 12
        self.x = np.linspace(1600, 1700, 101)

    def _fixture(self, center=1655.0):
        params = np.array([[.8, center, 12.0]])
        curve = gaussian(self.x, *params[0])
        residual = np.random.default_rng(141).normal(0, .0001, len(self.x))
        processed = {"x": self.x, "step": 1.0, "source_step": 1.0,
                     "corrected": curve + residual, "noise_sd": .0001,
                     "derivatives": {9: np.zeros(len(self.x)), 13: np.zeros(len(self.x))}}
        processed["sensitivity_variants"] = [{"label": "checked_variant", "processed": processed.copy()}]
        best = {"params": params, "fit": curve, "sse": float(residual @ residual),
                "aicc": 10.0, "bic": 11.0, "count": 1, "condition_number": 100.0,
                "bound_hits": False, "initial_centers": np.array([center]),
                "admissible": True, "admissibility_reasons": [], "nfev": 10}
        boot = {"parameters": np.repeat(params[None, :, :], 12, axis=0),
                "percentages": np.full((12, 1), 100.0),
                "structure_samples": np.zeros((12, len(STRUCTURES))),
                "success_count": 12, "success_fraction": 1.0, "count_support": 1.0,
                "count_comparison_fraction": 1.0, "block_length": 3}
        return processed, best, boot

    def test_invalid_low_criterion_model_cannot_defeat_admissible_model(self):
        processed, best, boot = self._fixture()
        invalid = deepcopy(best)
        invalid.update(aicc=-100., bic=-100., count=2, admissible=False,
                       bound_hits=True, admissibility_reasons=["WIDTH_LOWER_BOUND"])
        competitor = deepcopy(best)
        competitor.update(aicc=20., bic=21., count=2)
        candidates = [{"stable": True}]
        with patch("amide_analysis.fitting._select_model", side_effect=[
                (invalid, [invalid, best, competitor], candidates),
                (best, [best], candidates)]):
            with patch("amide_analysis.fitting._bootstrap", return_value=boot) as bootstrap:
                fit = analyze_band(processed, self.config)
        self.assertTrue(fit["numerical_fit_reliable"], fit["warnings"])
        self.assertTrue(fit["quantitative_accepted"])
        self.assertEqual(fit["metrics"]["delta_criterion_next_count"], 10.)
        self.assertEqual(bootstrap.call_args.args[2], [competitor])
        rejected_row = fit["model_comparison"][0]
        self.assertFalse(rejected_row["eligible_for_model_selection"])
        self.assertFalse(rejected_row["selected"])
        self.assertTrue(fit["model_comparison"][1]["selected"])
        self.assertEqual(fit["bootstrap_diagnostics"], [])

    def test_no_admissible_model_skips_bootstrap_and_preserves_curve(self):
        processed, best, _ = self._fixture()
        best.update(admissible=False, bound_hits=True,
                    admissibility_reasons=["WIDTH_LOWER_BOUND"])
        with patch("amide_analysis.fitting._select_model", return_value=(best, [best], [{"stable": True}])):
            with patch("amide_analysis.fitting._bootstrap") as bootstrap:
                fit = analyze_band(processed, self.config)
        bootstrap.assert_not_called()
        self.assertFalse(fit["numerical_fit_reliable"])
        self.assertFalse(fit["quantitative_accepted"])
        self.assertIn("NO_MODEL_WITHIN_CONFIGURED_LIMITS", fit["fit_rejection_codes"])
        np.testing.assert_array_equal(fit["fit"], best["fit"])
        self.assertAlmostEqual(sum(fit["diagnostic_percentages"].values()), 100.)
        self.assertTrue(all(np.isnan(value) for value in fit["percentages"].values()))
        self.assertTrue(np.isnan(fit["components"][0]["center_ci_low"]))
        self.assertIn("Skipped", fit["metrics"]["bootstrap_status"])

    def test_structure_sd_uses_final_ci_aware_categories(self):
        processed, best, boot = self._fixture(center=1640.)
        # Final CI spans the assignment boundary, so this component remains
        # Unassigned. Its area is stable even though nominal labels alternate.
        boot["parameters"][:6, 0, 1] = 1638.
        boot["parameters"][6:, 0, 1] = 1642.
        boot["structure_samples"][:6, STRUCTURES.index("beta_sheet")] = 100.
        boot["structure_samples"][6:, STRUCTURES.index("random_coil")] = 100.
        with patch("amide_analysis.fitting._select_model", return_value=(best, [best], [{"stable": True}])):
            with patch("amide_analysis.fitting._bootstrap", return_value=boot):
                fit = analyze_band(processed, self.config)
        self.assertTrue(fit["numerical_fit_reliable"], fit["warnings"])
        self.assertEqual(fit["components"][0]["structure"], "unassigned")
        self.assertEqual(fit["metrics"]["max_structure_pct_sd"], 0.)
        self.assertEqual(fit["percentages"]["unassigned"], 100.)

    def test_unknown_resolution_is_protocol_blocker_not_numerical_failure(self):
        processed, best, boot = self._fixture()
        self.config["resolution_confirmed"] = False
        messages = []
        boot["diagnostics"] = [{"iteration": 1, "primary_fit_status": "accepted"}]
        with patch("amide_analysis.fitting._select_model", return_value=(best, [best], [{"stable": True}])):
            with patch("amide_analysis.fitting._bootstrap", return_value=boot) as bootstrap:
                fit = analyze_band(processed, self.config, progress=messages.append)
        self.assertTrue(fit["numerical_fit_reliable"], fit["warnings"])
        self.assertFalse(fit["quantitative_accepted"])
        self.assertEqual(fit["fit_rejection_codes"], [])
        self.assertEqual(fit["protocol_blockers"], ["INSTRUMENT_RESOLUTION_UNCONFIRMED"])
        self.assertEqual(fit["diagnostic_percentages"]["alpha_helix"], 100.)
        self.assertEqual(fit["bootstrap_diagnostics"], boot["diagnostics"])
        self.assertEqual(bootstrap.call_args.kwargs["progress"], messages.append)
        self.assertTrue(any("Bootstrap" in message for message in messages))

    def test_low_snr_skips_costly_component_search(self):
        processed, _, _ = self._fixture()
        processed["noise_sd"] = 1.
        with patch("amide_analysis.fitting._select_model") as selection:
            fit = analyze_band(processed, self.config)
        selection.assert_not_called()
        self.assertEqual(fit["fit_rejection_codes"], ["LOW_SIGNAL_TO_NOISE"])
        self.assertFalse(fit["quantitative_accepted"])

    def test_actual_source_spacing_restricts_width_and_separation_without_mutating_config(self):
        processed, best, boot = self._fixture()
        processed["source_step"] = 4.
        with patch("amide_analysis.fitting._select_model", return_value=(best, [best], [{"stable": True}])) as selection:
            with patch("amide_analysis.fitting._bootstrap", return_value=boot):
                analyze_band(processed, self.config)
        passed_config = selection.call_args_list[0].args[1]
        self.assertEqual(passed_config["peaks"]["min_fwhm_cm1"], 8.)
        self.assertEqual(passed_config["peaks"]["min_separation_cm1"], 8.)
        self.assertEqual(self.config["peaks"]["min_separation_cm1"], 6.)


if __name__ == "__main__":
    unittest.main()
