"""Regression checks for predefined spectral-region comparisons.

Run with ``python -m unittest -v test_ir_region_analysis``.
"""

import unittest
import re

import numpy as np

import ir_region_analysis as analysis


def quiet(_message):
    pass


class RegionIntegrationTests(unittest.TestCase):
    def test_constant_signals_keep_their_value_in_regions_of_different_width(self):
        grid = np.array([900.0, 970.0, 1030.0, 1250.0, 1800.0])
        matrix = np.tile(np.array([-2.5, 0.0, 4.0])[:, None], (1, len(grid)))
        for lower, upper in ((900.0, 1800.0), (995.0, 1200.0), (1000.0, 1010.0)):
            with self.subTest(lower=lower, upper=upper):
                values = analysis.mean_signal_in_region(grid, matrix, lower, upper)
                np.testing.assert_allclose(values, [-2.5, 0.0, 4.0], atol=1e-13)

    def test_linear_signal_uses_exact_interpolated_boundaries(self):
        # Neither boundary is a grid point; the mean of a linear signal is
        # its value at the interval midpoint, even on an uneven source grid.
        grid = np.array([900.0, 975.0, 1080.0, 1260.0, 1500.0])
        slopes = np.array([0.02, -0.03, 0.0])
        intercepts = np.array([-15.0, 40.0, 2.0])
        matrix = slopes[:, None] * grid + intercepts[:, None]
        values = analysis.mean_signal_in_region(grid, matrix, 1000.0, 1200.0)
        np.testing.assert_allclose(values, slopes * 1100.0 + intercepts, atol=1e-12)

    def test_partial_coverage_never_extrapolates(self):
        grid = np.array([1000.0, 1100.0, 1200.0])
        matrix = np.ones((2, len(grid)))
        for lower, upper in ((900.0, 1150.0), (1050.0, 1250.0), (900.0, 1250.0)):
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(ValueError):
                    analysis.mean_signal_in_region(grid, matrix, lower, upper)

    def test_integration_rejects_invalid_grid_matrix_or_bounds(self):
        valid_grid = np.array([1000.0, 1100.0, 1200.0])
        valid_matrix = np.ones((2, 3))
        cases = [
            (valid_grid[::-1], valid_matrix, 1000.0, 1200.0),
            (np.array([1000.0, 1000.0, 1200.0]), valid_matrix, 1000.0, 1200.0),
            (np.array([1000.0, np.nan, 1200.0]), valid_matrix, 1000.0, 1200.0),
            (np.array([1000.0]), np.ones((2, 1)), 1000.0, 1200.0),
            (valid_grid, np.ones(3), 1000.0, 1200.0),
            (valid_grid, np.ones((2, 2)), 1000.0, 1200.0),
            (valid_grid, np.array([[1.0, np.nan, 1.0]]), 1000.0, 1200.0),
            (valid_grid, valid_matrix, 1100.0, 1100.0),
            (valid_grid, valid_matrix, 1200.0, 1000.0),
            (valid_grid, valid_matrix, np.nan, 1200.0),
        ]
        for index, args in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    analysis.mean_signal_in_region(*args)


class RegionalStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.arange(800.0, 3701.0, 25.0)
        self.control = np.tile(np.arange(4.0)[:, None], (1, len(self.grid)))
        self.cancer = np.tile(np.arange(10.0, 14.0)[:, None], (1, len(self.grid)))

    def calculate(self, grid=None, cancer=None, control=None):
        return analysis.regional_statistics(
            self.grid if grid is None else grid,
            self.cancer if cancer is None else cancer,
            self.control if control is None else control,
            say=quiet,
        )

    def test_requested_nine_regions_are_preserved_in_order(self):
        expected = (
            ("Амид I", 1600, 1700),
            ("Амид II", 1500, 1600),
            ("Амид III и фосфатные группы", 1200, 1350),
            ("Липиды, белки и карбоксильные группы", 1350, 1500),
            ("Углеводы, гликопротеины и фосфаты", 900, 1200),
            ("Карбонильные группы липидов", 1700, 1750),
            ("C–H-колебания (липиды и органические компоненты)", 2800, 3000),
            ("O–H и N–H-колебания (вода, белки)", 3000, 3600),
            ("Область молекулярных отпечатков", 900, 1800),
        )
        self.assertEqual(tuple(analysis.SPECTRAL_REGIONS), expected)
        result = self.calculate()
        table = result["statistics"]
        self.assertEqual(len(table), 9)
        self.assertEqual(table["Region"].tolist(), [item[0] for item in expected])
        self.assertEqual(table["Lower_cm-1"].tolist(), [item[1] for item in expected])
        self.assertEqual(table["Upper_cm-1"].tolist(), [item[2] for item in expected])
        self.assertTrue(table["Region_ID"].is_unique)
        self.assertTrue(table["Tested"].all())
        self.assertEqual(result["cancer_values"].shape, (4, 9))
        self.assertEqual(result["control_values"].shape, (4, 9))
        # The fingerprint region is an additional row, although it overlaps
        # the other low-frequency regions.
        np.testing.assert_allclose(result["cancer_values"][:, -1], np.arange(10.0, 14.0))

    def test_fully_separated_groups_have_known_exact_p_and_positive_effect(self):
        table = self.calculate()["statistics"]
        # Four versus four, with one extreme assignment in each tail:
        # p = 2 / choose(8, 4) = 1 / 35.
        np.testing.assert_allclose(table["p_value"], 1.0 / 35.0, atol=1e-13)
        np.testing.assert_allclose(table["q_value_BH_FDR"], 1.0 / 35.0, atol=1e-13)
        np.testing.assert_allclose(table["Effect_size"], 1.0)
        self.assertTrue(table["Significant_q_0.05"].all())
        self.assertTrue((table["Cancer_N"] == 4).all())
        self.assertTrue((table["Control_N"] == 4).all())
        np.testing.assert_allclose(table["Cancer_mean_region_signal"], 11.5)
        np.testing.assert_allclose(table["Control_mean_region_signal"], 1.5)

    def test_swapping_groups_reverses_effect_and_preserves_two_sided_p(self):
        original = self.calculate()["statistics"]
        reversed_groups = self.calculate(cancer=self.control, control=self.cancer)["statistics"]
        np.testing.assert_allclose(reversed_groups["Effect_size"], -original["Effect_size"])
        np.testing.assert_allclose(reversed_groups["p_value"], original["p_value"])
        np.testing.assert_allclose(reversed_groups["q_value_BH_FDR"], original["q_value_BH_FDR"])

    def test_all_ties_give_p_one_zero_effect_and_no_significance(self):
        matrix = np.ones((3, len(self.grid)))
        table = self.calculate(cancer=matrix, control=matrix)["statistics"]
        np.testing.assert_allclose(table["p_value"], 1.0)
        np.testing.assert_allclose(table["q_value_BH_FDR"], 1.0)
        np.testing.assert_allclose(table["Effect_size"], 0.0)
        self.assertFalse(table["Significant_q_0.05"].any())
        self.assertTrue(table["Conclusion"].notna().all())

    def test_changes_outside_all_requested_regions_do_not_enter_the_tests(self):
        # Sign changes outside the union preserve L2 norms and leave the
        # normalized measured signals identical inside every requested region.
        cancer = np.stack([1.0 + 0.001 * (index + 1) * self.grid for index in range(4)])
        control = cancer.copy()
        outside = ((self.grid < 900) | (self.grid > 3600)
                   | ((self.grid > 1800) & (self.grid < 2800)))
        control[:, outside] *= -1
        cancer /= np.linalg.norm(cancer, axis=1, keepdims=True)
        control /= np.linalg.norm(control, axis=1, keepdims=True)
        self.assertFalse(np.array_equal(cancer, control))
        result = self.calculate(cancer=cancer, control=control)
        np.testing.assert_allclose(result["cancer_values"], result["control_values"], atol=1e-14)
        np.testing.assert_allclose(result["statistics"]["p_value"], 1.0)
        np.testing.assert_allclose(result["statistics"]["Effect_size"], 0.0)

    def test_only_fully_covered_regions_enter_bh_and_missing_regions_stay_visible(self):
        grid = np.arange(1200.0, 1801.0, 25.0)
        control = np.tile(np.arange(4.0)[:, None], (1, len(grid)))
        cancer = np.tile(np.arange(10.0, 14.0)[:, None], (1, len(grid)))
        result = self.calculate(grid=grid, cancer=cancer, control=control)
        table = result["statistics"]
        expected_tested = np.array([True, True, True, True, False, True, False, False, False])
        np.testing.assert_array_equal(table["Tested"], expected_tested)
        self.assertEqual(len(table), 9)
        # Five identical tested p-values retain their value after BH. Treating
        # the four missing regions as extra tests would wrongly inflate q.
        np.testing.assert_allclose(table.loc[expected_tested, "p_value"], 1.0 / 35.0)
        np.testing.assert_allclose(table.loc[expected_tested, "q_value_BH_FDR"], 1.0 / 35.0)
        for column in ("p_value", "q_value_BH_FDR", "Effect_size"):
            self.assertTrue(table.loc[~expected_tested, column].isna().all())
        self.assertFalse(table.loc[~expected_tested, "Significant_q_0.05"].any())
        self.assertTrue(table.loc[~expected_tested, "Coverage_status"].notna().all())
        self.assertTrue(table.loc[~expected_tested, "Conclusion"].notna().all())
        self.assertTrue(np.isnan(result["cancer_values"][:, ~expected_tested]).all())
        self.assertTrue(np.isnan(result["control_values"][:, ~expected_tested]).all())

    def test_losing_one_endpoint_makes_entire_region_untested(self):
        grid = np.arange(925.0, 3601.0, 25.0)
        cancer = np.ones((2, len(grid)))
        control = np.zeros((2, len(grid)))
        table = self.calculate(grid=grid, cancer=cancer, control=control)["statistics"]
        # Both 900–1200 and 900–1800 lack their lower endpoint; coverage
        # of most of those intervals must not permit a partial-area test.
        np.testing.assert_array_equal(table["Tested"], [True, True, True, True, False,
                                                       True, True, True, False])

    def test_bh_corrects_distinct_region_p_values_as_one_family(self):
        grid = np.arange(1200.0, 1801.0, 25.0)
        # The cancer profiles rise linearly, producing five different local
        # contrasts. Their Mann–Whitney U values are 16, 15, 6, 13, and 16.
        cancer = np.arange(4.0)[:, None] + (grid - 1200.0) / 100.0
        control = np.tile((np.arange(4.0) + 1.0)[:, None], (1, len(grid)))
        table = self.calculate(grid=grid, cancer=cancer, control=control)["statistics"]
        tested = table.loc[table["Tested"]]
        expected_p = [1.0 / 35.0, 2.0 / 35.0, 24.0 / 35.0, 1.0 / 5.0, 1.0 / 35.0]
        # Sorted BH ranks: p = [1/35, 1/35, 2/35, 1/5, 24/35].
        expected_q = [1.0 / 14.0, 2.0 / 21.0, 24.0 / 35.0, 1.0 / 4.0, 1.0 / 14.0]
        np.testing.assert_allclose(tested["p_value"], expected_p, atol=1e-13)
        np.testing.assert_allclose(tested["q_value_BH_FDR"], expected_q, atol=1e-13)
        self.assertTrue((tested["p_value"] < 0.05).any())
        self.assertFalse(tested["Significant_q_0.05"].any())

    def test_no_fully_covered_region_is_an_explicit_error(self):
        grid = np.arange(1900.0, 2101.0, 25.0)
        with self.assertRaises(ValueError):
            self.calculate(grid=grid, cancer=np.ones((2, len(grid))),
                           control=np.zeros((2, len(grid))))

    def test_statistics_reject_invalid_inputs(self):
        cases = [
            (self.grid, self.cancer[:1], self.control),
            (self.grid, self.cancer, self.control[:1]),
            (self.grid, self.cancer[:, :-1], self.control),
            (self.grid, self.cancer[0], self.control),
            (self.grid[::-1], self.cancer, self.control),
        ]
        duplicate_grid = self.grid.copy()
        duplicate_grid[1] = duplicate_grid[0]
        cases.append((duplicate_grid, self.cancer, self.control))
        nonfinite_grid = self.grid.copy()
        nonfinite_grid[1] = np.nan
        cases.append((nonfinite_grid, self.cancer, self.control))
        for value in (np.nan, np.inf):
            invalid_cancer = self.cancer.copy()
            invalid_cancer[0, 2] = value
            cases.append((self.grid, invalid_cancer, self.control))
        for index, (grid, cancer, control) in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    self.calculate(grid=grid, cancer=cancer, control=control)


class CompactSummaryTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.arange(900.0, 3601.0, 25.0)
        self.control = np.tile(np.array([0.0, 1.0, 2.0, 20.0])[:, None], (1, len(self.grid)))
        self.cancer = np.tile(np.array([5.0, 6.0, 7.0, 60.0])[:, None], (1, len(self.grid)))

    def statistics(self):
        return analysis.regional_statistics(
            self.grid, self.cancer, self.control, say=quiet
        )["statistics"]

    def test_summary_has_seven_columns_and_all_original_requested_ranges(self):
        table = self.statistics()
        summary = analysis.make_summary_table(table)
        self.assertEqual(summary.columns.tolist(), ["Region", "Range, cm⁻¹", "Median control",
                         "Median cancer", "p-MWU", "p-BH", "r"])
        self.assertEqual(summary["Region"].tolist(), table["Region"].tolist())
        expected_ranges = [(1700, 1600), (1600, 1500), (1350, 1200), (1500, 1350),
                           (1200, 900), (1750, 1700), (3000, 2800), (3600, 3000),
                           (1800, 900)]
        actual_ranges = [tuple(int(value) for value in re.findall(r"\d+", str(text)))
                         for text in summary["Range, cm⁻¹"]]
        self.assertEqual(actual_ranges, expected_ranges)

    def test_summary_uses_medians_of_individual_region_values_and_rank_biserial_r(self):
        table = self.statistics()
        summary = analysis.make_summary_table(table)
        # Skewed samples make a mistaken mean visibly different from a median.
        self.assertNotEqual(self.control[:, 0].mean(), 1.5)
        self.assertNotEqual(self.cancer[:, 0].mean(), 6.5)
        np.testing.assert_allclose(table["Control_median_region_signal"], 1.5)
        np.testing.assert_allclose(table["Cancer_median_region_signal"], 6.5)
        np.testing.assert_allclose(summary["Median control"], 1.5)
        np.testing.assert_allclose(summary["Median cancer"], 6.5)
        np.testing.assert_allclose(summary["r"], table["Effect_size"])

    def test_small_p_values_display_threshold_without_rounding_larger_values_to_zero(self):
        table = self.statistics()
        table.loc[0, ["p_value", "q_value_BH_FDR"]] = [0.00009, 0.000099]
        table.loc[1, ["p_value", "q_value_BH_FDR"]] = [0.0001, 0.000123456]
        summary = analysis.make_summary_table(table)
        self.assertEqual(summary.loc[0, "p-MWU"], "< 0.0001")
        self.assertEqual(summary.loc[0, "p-BH"], "< 0.0001")
        self.assertAlmostEqual(summary.loc[1, "p-MWU"], 0.0001)
        self.assertAlmostEqual(summary.loc[1, "p-BH"], 0.000123456)
        # Formatting must leave the full numeric statistics usable for analysis.
        self.assertAlmostEqual(table.loc[0, "p_value"], 0.00009)

    def test_summary_distinguishes_untested_regions_from_nonsignificant_ones(self):
        grid = np.arange(1200.0, 1801.0, 25.0)
        matrix = np.ones((3, len(grid)))
        table = analysis.regional_statistics(grid, matrix, matrix, say=quiet)["statistics"]
        summary = analysis.make_summary_table(table)
        tested = table["Tested"].to_numpy()
        np.testing.assert_allclose(summary.loc[tested, "p-MWU"].to_numpy(dtype=float), 1.0)
        np.testing.assert_allclose(summary.loc[tested, "p-BH"].to_numpy(dtype=float), 1.0)
        self.assertTrue((summary.loc[~tested, "p-MWU"] == "Не проверена").all())
        self.assertTrue((summary.loc[~tested, "p-BH"] == "Не проверена").all())
        for column in ("Median control", "Median cancer", "r"):
            self.assertTrue(summary.loc[~tested, column].isna().all())


if __name__ == "__main__":
    unittest.main()
