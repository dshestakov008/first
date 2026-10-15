"""Known spectral areas and rejection checks, without invented biological data."""

from copy import deepcopy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import quad

from amide_analysis.config import default_config
from amide_analysis.fitting import STRUCTURES, _classify, analyze_band, component_area, gaussian
from amide_analysis import fitting
from amide_analysis.pipeline import _processing_variants, run_analysis
from amide_analysis.preprocess import preprocess_spectrum


def known_spectrum():
    x = np.arange(450.0, 4000.0, 1.93)
    components = [(0.7, 1630.0, 12.0), (0.6, 1655.0, 10.0), (0.3, 1674.0, 12.0)]
    signal = sum(gaussian(x, *row) for row in components)
    y = signal + 0.09 + 0.00001 * (x - 450)
    y += np.random.default_rng(91).normal(0, 0.0007, len(x))
    areas = np.array([component_area(*row, 1600, 1700) for row in components])
    return {"x": x, "y": y}, areas / areas.sum() * 100


def analyze_known(config):
    raw, truth = known_spectrum()
    processed = preprocess_spectrum(raw["x"], raw["y"], config)
    variants, failures = _processing_variants(raw, config)
    if failures:
        raise AssertionError(failures)
    processed["sensitivity_variants"] = variants
    return analyze_band(processed, config, seed=912), truth


class FittingTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.config["resolution_confirmed"] = True
        self.config["fitting"]["multistarts"] = 3
        self.config["fitting"]["bootstrap_iterations"] = 12

    def test_truncated_gaussian_area_matches_independent_quadrature(self):
        for center, width in ((1630, 12), (1601, 18), (1699, 9)):
            numerical, _ = quad(lambda x: gaussian(x, 0.7, center, width), 1600, 1700)
            self.assertAlmostEqual(component_area(0.7, center, width, 1600, 1700), numerical, places=10)

    def test_three_components_are_selected_without_forcing_four_structures(self):
        fit, truth = analyze_known(self.config)
        self.assertEqual(fit["status"], "Reliable fit", fit["warnings"])
        self.assertEqual(len(fit["components"]), 3)
        np.testing.assert_allclose([row["center"] for row in fit["components"]], [1630, 1655, 1674], atol=0.2)
        np.testing.assert_allclose([row["area_pct"] for row in fit["components"]], truth, atol=1.0)
        self.assertAlmostEqual(sum(fit["percentages"].values()), 100, places=10)
        self.assertEqual(fit["percentages"]["random_coil"], 0)
        self.assertEqual(fit["percentages"]["unassigned"], 0)
        self.assertGreater(len(fit["model_comparison"]), 1)
        self.assertGreaterEqual(fit["metrics"]["bootstrap_count_support"], 0.8)
        self.assertEqual(len(fit["sensitivity"]), 3)

    def test_unconfirmed_resolution_retains_diagnostics_but_withholds_percentages(self):
        self.config["resolution_confirmed"] = False
        fit, _ = analyze_known(self.config)
        self.assertEqual(fit["status"], "Unreliable fit")
        self.assertTrue(all(np.isnan(value) for value in fit["percentages"].values()))
        self.assertTrue(all(np.isnan(row["area_pct"]) and not row["accepted"] for row in fit["components"]))
        self.assertTrue(all(np.isfinite(row["diagnostic_area_pct"]) for row in fit["components"]))

    def test_flat_corrected_band_does_not_create_components(self):
        x = np.arange(450., 4000., 1.93)
        processed = preprocess_spectrum(x, 0.05 + 0.00002 * x, self.config)
        fit = analyze_band(processed, self.config)
        self.assertEqual(fit["status"], "Unreliable fit")
        self.assertEqual(fit["components"], [])
        self.assertTrue(all(np.isnan(value) for value in fit["percentages"].values()))

    def test_missing_sensitivity_is_not_a_reliable_quantitative_fit(self):
        raw, _ = known_spectrum()
        processed = preprocess_spectrum(raw["x"], raw["y"], self.config)
        fit = analyze_band(processed, self.config)
        self.assertEqual(fit["status"], "Unreliable fit")
        self.assertTrue(any("чувствительности" in message for message in fit["warnings"]))

    def test_uncertain_category_is_unassigned_instead_of_nearest_structure(self):
        self.assertEqual(_classify(1640, 1638, 1642, self.config), "unassigned")
        self.assertEqual(_classify(1655, 1652, 1658, self.config), "alpha_helix")
        self.assertEqual(_classify(1690, 1688, 1692, self.config), "beta_sheet")
        self.assertEqual(_classify(1610, 1608, 1612, self.config), "unassigned")

    def test_same_seed_reproduces_bootstrap_and_parameters(self):
        one, _ = analyze_known(self.config)
        two, _ = analyze_known(self.config)
        self.assertEqual(one["percentages"], two["percentages"])
        self.assertEqual(one["metrics"], two["metrics"])
        self.assertEqual(one["components"], two["components"])

    def test_assignment_uncertainty_is_not_reported_as_preprocessing_area_change(self):
        original_bootstrap = fitting._bootstrap

        def broaden_first_center(*args, **kwargs):
            boot = original_bootstrap(*args, **kwargs)
            # Isolate classification uncertainty: area samples remain identical,
            # but a center interval crosses the 1640 assignment boundary.
            boot["parameters"][::2, 0, 1] = 1643.0
            return boot

        with patch("amide_analysis.fitting._bootstrap", side_effect=broaden_first_center):
            fit, _ = analyze_known(self.config)
        self.assertEqual(fit["components"][0]["structure"], "unassigned")
        self.assertTrue(all(row["max_delta_pct"] < 1.0 for row in fit["sensitivity"]))
        self.assertFalse(any("чувствительны к обработке" in warning for warning in fit["warnings"]))


class PipelineTests(unittest.TestCase):
    def test_real_file_read_export_and_rejection_are_auditable(self):
        raw, _ = known_spectrum()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            # XLSX as in the requested real workflow; original user file unavailable.
            pd.DataFrame({"Wavenumber": raw["x"][::-1], "Absorbance": raw["y"][::-1]}).to_excel(source / "synthetic.xlsx", index=False)
            (source / "invalid.csv").write_text("Wavenumber,Absorbance\n1600,wrong\n1700,0.1\n", encoding="utf-8")
            cfg = default_config()
            cfg["fitting"]["multistarts"] = 2
            cfg["fitting"]["bootstrap_iterations"] = 10
            result = run_analysis(source, root / "results", cfg, progress=lambda _: None)
            self.assertTrue(result["excel"].is_file())
            self.assertTrue((result["output_dir"] / "Unreliable_Spectra.csv").is_file())
            tables = result["tables"]
            self.assertEqual(len(tables["Spectrum_Results"]), 2)
            self.assertEqual(len(tables["Unreliable_Spectra"]), 2)
            self.assertTrue(tables["Secondary_Structure"][list(STRUCTURES)].isna().all().all())
            self.assertEqual(tables["Fit_Quality"]["status"].tolist(), ["Failed", "Unreliable fit"])
            quality = tables["Fit_Quality"].iloc[1]
            coefficients = json.loads(quality["baseline_coefficients"])
            self.assertEqual(len(coefficients), 2)
            processed = tables["Processed_Spectra"]
            coordinate = ((processed["wavenumber"] - quality["baseline_basis_center_cm1"])
                          / quality["baseline_basis_halfspan_cm1"])
            np.testing.assert_allclose(processed["baseline_adjustment"],
                                       coefficients[0] + coefficients[1] * coordinate)
            self.assertEqual(len(tables["Raw_Spectra"]), len(raw["x"]))
            self.assertTrue(tables["Metadata"]["sha256"].notna().all())
            self.assertFalse(np.isfinite(tables["Statistical_Tests"]["p_value"]).any())
            self.assertEqual(sum(path.suffix == ".png" for path in result["figures"]), 6)
            with self.assertRaises(ValueError):
                run_analysis(source, root / "results", cfg, progress=lambda _: None)


if __name__ == "__main__":
    unittest.main()
