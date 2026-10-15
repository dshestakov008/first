"""Independent numerical and inference checks for integrated regional signals."""

from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from amide_analysis.regions import (SPECTRAL_REGIONS, SUMMARY_COLUMNS, analyze_regions,
                                    mean_signal_in_region)


def source(spectrum_id, amplitude=1., *, step=2., lower=800., upper=3700.):
    x = np.arange(lower, upper + .01, step)
    y = 1. + amplitude * np.exp(-((x - 1654.) / 14.) ** 2)
    return dict(spectrum_id=spectrum_id, x=x, y=y)


def dataset():
    raw, metadata = [], []
    for group, amplitudes in (("Control", (1., 2., 3.)), ("Cancer", (4., 5., 6.))):
        for index, amplitude in enumerate(amplitudes):
            identity = f"{group}-{index}"
            raw.append(source(identity, amplitude))
            metadata.append(dict(spectrum_id=identity, animal_id=identity, group=group))
    return raw, pd.DataFrame(metadata)


class RegionalAnalysisTests(unittest.TestCase):
    def test_exact_bounds_integral_of_linear_signal_and_no_extrapolation(self):
        x = np.array([0., 2., 4.])
        self.assertAlmostEqual(mean_signal_in_region(x, 2 * x + 1, .5, 3.5), 5.)
        with self.assertRaisesRegex(ValueError, "no extrapolation"):
            mean_signal_in_region(x, 2 * x + 1, -.1, 3.)

    def test_requested_nine_ranges_and_seven_numeric_summary_columns(self):
        raw, metadata = dataset()
        result = analyze_regions(raw, metadata, {})
        self.assertEqual([(lo, hi) for _, lo, hi in SPECTRAL_REGIONS],
                         [(1600., 1700.), (1500., 1600.), (1200., 1350.),
                          (1350., 1500.), (900., 1200.), (1700., 1750.),
                          (2800., 3000.), (3000., 3600.), (900., 1800.)])
        summary = result["Region_Summary"]
        self.assertEqual(list(summary), SUMMARY_COLUMNS)
        self.assertEqual(len(summary), 9)
        self.assertTrue(np.issubdtype(summary["p-MWU"].dtype, np.number))
        self.assertTrue(np.isfinite(summary["p-BH"]).all())

    def test_known_exact_p_rank_effect_and_single_bh_family(self):
        raw, metadata = dataset()
        result = analyze_regions(raw, metadata, {})
        detailed = result["Region_Statistics"]
        amide = detailed.iloc[0]
        # All 3!3! label allocations out of 6! give two extreme allocations.
        self.assertAlmostEqual(amide["p_value"], 2 / 20)
        self.assertAlmostEqual(amide["Mann_Whitney_U"], 9.)
        self.assertAlmostEqual(amide["Effect_size"], 1.)
        expected_q = stats.false_discovery_control(detailed["p_value"].to_numpy(), method="bh")
        np.testing.assert_allclose(detailed["q_value_BH_FDR"], expected_q)
        self.assertTrue(detailed["Tested"].all())

    def test_group_reversal_preserves_p_and_reverses_signed_effect(self):
        raw, metadata = dataset()
        forward = analyze_regions(raw, metadata, {})["Region_Statistics"]
        reverse_meta = metadata.copy()
        reverse_meta["group"] = reverse_meta["group"].map({"Control": "Cancer", "Cancer": "Control"})
        reverse = analyze_regions(raw, reverse_meta, {})["Region_Statistics"]
        np.testing.assert_allclose(reverse["p_value"], forward["p_value"])
        np.testing.assert_allclose(reverse["Effect_size"], -forward["Effect_size"])
        # A presentation setting cannot reverse the specified Control/Cancer r.
        setting = {"plots": {"control_group": "Cancer"}}
        np.testing.assert_allclose(analyze_regions(raw, metadata, setting)["Region_Summary"]["r"],
                                   forward["Effect_size"])

    def test_discrete_l2_normalization_and_technical_repeat_mean(self):
        raw = [source("a-1", .5), source("a-2", 5.)]
        metadata = pd.DataFrame([dict(spectrum_id=entry["spectrum_id"], animal_id="a", group="Control")
                                 for entry in raw])
        result = analyze_regions(raw, metadata, {})
        values = result["Region_Spectrum_Values"].query("region_id == 'R1'")["value"].to_numpy()
        expected = []
        for entry in raw:
            x, signal = entry["x"], entry["y"] / np.linalg.norm(entry["y"])
            band = (x >= 1600.) & (x <= 1700.)
            expected.append(float(np.trapezoid(signal[band], x=x[band]) / 100.))
        np.testing.assert_allclose(values, expected)
        animal = result["Region_Animal_Values"].query("region_id == 'R1'").iloc[0]
        self.assertEqual(animal["expected_repeats"], 2)
        self.assertEqual(animal["available_repeats"], 2)
        self.assertAlmostEqual(animal["value"], np.mean(expected))
        self.assertEqual(result["Region_Statistics"].iloc[0]["Control_N"], 1)

    def test_missing_expected_repeat_excludes_whole_animal_endpoint(self):
        raw, metadata = dataset()
        raw.append(source("extra", 2.))
        metadata = pd.concat([metadata, pd.DataFrame([dict(spectrum_id="missing", animal_id="Control-0", group="Control")])],
                             ignore_index=True)
        # An extra input absent from metadata is never silently included.
        with self.assertRaisesRegex(ValueError, "absent from metadata"):
            analyze_regions(raw, metadata, {})
        result = analyze_regions(raw[:-1], metadata, {})
        mouse = result["Region_Animal_Values"].query("animal_id == 'Control-0'")
        self.assertFalse(mouse["accepted"].any())
        self.assertTrue(mouse["value"].isna().all())
        self.assertTrue(result["Region_Statistics"]["Control_N"].eq(2).all())
        self.assertIn("INCOMPLETE_REPEATS", set(result["Region_QC"]["code"]))

    def test_fit_rejection_resolution_and_secondary_method_do_not_change_regions(self):
        raw, metadata = dataset()
        baseline = analyze_regions(raw, metadata, {})
        for entry in raw:
            entry.update(status="Unreliable fit", quantitative_accepted=False)
        metadata["status"] = "Unreliable fit"
        settings = {"resolution_confirmed": False, "statistics": {"method": "welch"}}
        result = analyze_regions(raw, metadata, settings)
        pd.testing.assert_frame_equal(result["Region_Summary"], baseline["Region_Summary"])
        self.assertTrue(result["Region_Statistics"]["method"].str.startswith("Mann-Whitney").all())

    def test_coarsest_shared_grid_partial_coverage_and_no_extrapolation(self):
        raw, metadata = dataset()
        for entry in raw:
            keep = entry["x"] <= 3000.
            entry["x"], entry["y"] = entry["x"][keep], entry["y"][keep]
        # One coarser grid sets the common spacing, rather than inventing samples.
        raw[-1]["x"], raw[-1]["y"] = raw[-1]["x"][::2], raw[-1]["y"][::2]
        result = analyze_regions(raw, metadata, {})
        qc = result["Region_QC"].query("code == 'COMMON_GRID'").iloc[0]
        self.assertGreaterEqual(qc["common_grid_step_cm1"], 4.)
        self.assertLessEqual(qc["common_grid_upper_cm1"], 3000.)
        inaccessible = result["Region_Statistics"].query("region_id == 'R8'").iloc[0]
        self.assertFalse(inaccessible["Tested"])
        self.assertTrue(np.isnan(inaccessible["p_value"]))
        self.assertTrue(result["Region_Spectrum_Values"].query("region_id == 'R8'")["value"].isna().all())

    def test_tied_exact_permutation_is_finite_and_order_invariant(self):
        raw, metadata = dataset()
        for index, entry in enumerate(raw):
            entry["y"] = source("unused", [1., 1., 2., 2., 3., 3.][index])["y"]
        forward = analyze_regions(raw, metadata, {})
        reverse = analyze_regions(list(reversed(raw)), metadata.iloc[::-1], {})
        first = forward["Region_Statistics"].iloc[0]
        self.assertIn("permutation, ties", first["method"])
        # Enumerate the 20 assignments independently: U>=8 or U<=1 is 4/20.
        self.assertAlmostEqual(first["p_value"], .2)
        pd.testing.assert_frame_equal(forward["Region_Summary"], reverse["Region_Summary"])

    def test_three_groups_use_omnibus_and_do_not_put_kw_p_in_mwu_column(self):
        raw, metadata = dataset()
        for index in range(3):
            identity = f"Other-{index}"
            raw.append(source(identity, 7. + index))
            metadata.loc[len(metadata)] = dict(spectrum_id=identity, animal_id=identity, group="Other")
        result = analyze_regions(raw, metadata, {})
        details = result["Region_Statistics"]
        self.assertTrue(details["method"].str.startswith("Kruskal-Wallis").all())
        self.assertTrue(details["Tested"].all())
        self.assertTrue(result["Region_Summary"]["p-MWU"].isna().all())
        self.assertTrue(result["Region_Summary"]["p-BH"].isna().all())

    def test_empty_and_invalid_source_values_remain_missing_with_auditable_qc(self):
        raw, metadata = dataset()
        empty = analyze_regions([], metadata, {})
        self.assertTrue(empty["Region_Spectrum_Values"]["value"].isna().all())
        self.assertFalse(empty["Region_Statistics"]["Tested"].any())
        self.assertIn("NO_READABLE_SPECTRA", set(empty["Region_QC"]["code"]))
        invalid = deepcopy(raw)
        invalid[0]["y"][:] = 0.
        result = analyze_regions(invalid, metadata, {})
        self.assertTrue(result["Region_Spectrum_Values"].query("spectrum_id == 'Control-0'")["value"].isna().all())
        self.assertIn("SOURCE_UNAVAILABLE", set(result["Region_QC"]["code"]))


if __name__ == "__main__":
    unittest.main()
