"""Statistical regression tests using biological units, not technical repeats."""

import math
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from amide_analysis.statistics import (ANIMAL_COLUMNS, STRUCTURES, aggregate_animals,
                                       benjamini_hochberg, perform_statistics)


def config(method="mannwhitney", **settings):
    return {"seed": 43, "replicates": {"max_repeat_delta_pct": 10.0},
            "statistics": {"method": method, "bootstrap_iterations": 120,
                           "permutation_resamples": 1999, "min_n_ci": 3,
                           "alpha": .05, "posthoc_only_if_omnibus_significant": True,
                           **settings}, "plots": {"merge_beta_turn": False}}


def percentages(alpha=20.0, beta=20.0):
    return dict(random_coil=10.0, alpha_helix=float(alpha), beta_sheet=float(beta),
                beta_turn=10.0, unassigned=80.0-float(alpha)-float(beta))


def animal_rows(groups):
    rows = []
    for group, values in groups.items():
        for i, alpha in enumerate(values):
            rows.append(dict(animal_id=f"{group}-{i}", group=group,
                             status="Reliable fit", n_repeats=1,
                             **percentages(alpha), alpha_beta_ratio=alpha/20,
                             log_alpha_beta=math.log(alpha/20) if alpha > 0 else np.nan))
    return pd.DataFrame(rows, columns=ANIMAL_COLUMNS)


class AnimalAggregationTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame([
            {"spectrum_id": "a-1", "animal_id": "mouse-a", "group": "Control"},
            {"spectrum_id": "a-2", "animal_id": "mouse-a", "group": "Control"},
            {"spectrum_id": "b-1", "animal_id": "mouse-b", "group": "Treatment"},
        ])
        self.spectra = pd.DataFrame([
            {"spectrum_id": "a-1", "status": "Reliable fit", **percentages(20)},
            {"spectrum_id": "a-2", "status": "Reliable fit", **percentages(24)},
            {"spectrum_id": "b-1", "status": "Reliable fit", **percentages(30)},
        ])

    def test_repeats_become_one_biological_unit(self):
        animals, diagnostics = aggregate_animals(self.spectra, self.metadata, config())
        self.assertEqual(len(animals), 2)
        first = animals.set_index("animal_id").loc["mouse-a"]
        self.assertEqual(first["n_repeats"], 2)
        self.assertAlmostEqual(first["alpha_helix"], 22)
        self.assertAlmostEqual(first["alpha_beta_ratio"], 1.1)
        self.assertAlmostEqual(first["log_alpha_beta"], math.log(1.1))
        self.assertTrue((diagnostics["status"] == "Reliable fit").all())
        desc, tests = perform_statistics(animals, config())
        self.assertEqual(desc.query("metric == 'alpha_helix' and group == 'Control'")["n"].iloc[0], 1)
        self.assertTrue(tests["p_value"].isna().all())
        self.assertTrue(tests["status"].str.contains("Insufficient n").all())

    def test_one_unreliable_repeat_excludes_whole_animal(self):
        self.spectra.loc[1, "status"] = "Unreliable fit"
        # Even diagnostic fitted percentages must not sneak into accepted means.
        animals, diagnostics = aggregate_animals(self.spectra, self.metadata, config())
        first = animals.set_index("animal_id").loc["mouse-a"]
        self.assertEqual(first["status"], "Unreliable fit")
        self.assertTrue(first[list(STRUCTURES)].isna().all())
        self.assertTrue(np.isnan(first["alpha_beta_ratio"]))
        self.assertTrue(diagnostics["message"].str.contains("not reliable").any())
        self.assertEqual(animals.loc[animals["animal_id"] == "mouse-b", "alpha_helix"].iloc[0], 30)

    def test_missing_expected_repeat_excludes_whole_animal(self):
        animals, diagnostics = aggregate_animals(self.spectra.iloc[[0, 2]], self.metadata, config())
        first = animals.set_index("animal_id").loc["mouse-a"]
        self.assertEqual(first["status"], "Unreliable fit")
        self.assertEqual(first["n_repeats"], 2)
        self.assertTrue(diagnostics["message"].str.contains("Missing spectrum result").any())

    def test_excessive_repeat_delta_is_not_imputed(self):
        self.spectra.loc[1, list(STRUCTURES)] = list(percentages(40).values())
        animals, diagnostics = aggregate_animals(self.spectra, self.metadata, config())
        first = animals.set_index("animal_id").loc["mouse-a"]
        self.assertEqual(first["status"], "Unreliable fit")
        self.assertTrue(first[list(STRUCTURES)].isna().all())
        self.assertTrue(diagnostics["message"].str.contains("percentage points").any())

    def test_actual_zero_is_preserved_and_log_ratio_has_no_pseudocount(self):
        for i in [0, 1]:
            self.spectra.loc[i, list(STRUCTURES)] = list(percentages(0).values())
        animals, _ = aggregate_animals(self.spectra, self.metadata, config())
        first = animals.set_index("animal_id").loc["mouse-a"]
        self.assertEqual(first["status"], "Reliable fit")
        self.assertEqual(first["alpha_helix"], 0)
        self.assertEqual(first["alpha_beta_ratio"], 0)
        self.assertTrue(np.isnan(first["log_alpha_beta"]))

    def test_zero_beta_makes_ratios_missing(self):
        for i in [0, 1]:
            self.spectra.loc[i, list(STRUCTURES)] = list(percentages(20, 0).values())
        animals, _ = aggregate_animals(self.spectra, self.metadata, config())
        first = animals.set_index("animal_id").loc["mouse-a"]
        self.assertEqual(first["beta_sheet"], 0)
        self.assertTrue(np.isnan(first["alpha_beta_ratio"]))
        self.assertTrue(np.isnan(first["log_alpha_beta"]))

    def test_nonfinite_percentages_and_invalid_compositions_excluded(self):
        self.spectra.loc[0, "alpha_helix"] = np.nan
        self.spectra.loc[2, "unassigned"] = 0
        animals, diagnostics = aggregate_animals(self.spectra, self.metadata, config())
        self.assertTrue((animals["status"] == "Unreliable fit").all())
        self.assertTrue(animals[list(STRUCTURES)].isna().all().all())
        self.assertEqual(len(diagnostics), 2)

    def test_duplicate_or_cross_group_identifiers_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_animals(pd.concat([self.spectra, self.spectra.iloc[[0]]]), self.metadata, config())
        self.metadata.loc[1, "group"] = "Treatment"
        with self.assertRaisesRegex(ValueError, "multiple groups"):
            aggregate_animals(self.spectra, self.metadata, config())

    def test_beta_merge_is_explicit_and_preserves_four_original_structures(self):
        settings = config()
        settings["plots"]["merge_beta_turn"] = True
        animals, _ = aggregate_animals(self.spectra, self.metadata, settings)
        self.assertTrue(set(STRUCTURES).issubset(animals.columns))
        np.testing.assert_allclose(animals["beta_associated"], 30)


class StatisticalInferenceTests(unittest.TestCase):
    def test_bh_known_values_and_missing_entries(self):
        q = benjamini_hochberg([.01, .04, .03, np.nan])
        np.testing.assert_allclose(q[:3], [.03, .04, .04])
        self.assertTrue(np.isnan(q[3]))

    def test_known_exact_mwu_p_and_shared_family(self):
        animals = animal_rows({"Control": [10, 12, 14], "Treatment": [30, 32, 34]})
        _, tests = perform_statistics(animals, config())
        alpha = tests.set_index("metric").loc["alpha_helix"]
        self.assertEqual(alpha["method"], "Mann-Whitney U (exact)")
        self.assertAlmostEqual(alpha["p_value"], .1)
        # Four endpoints have p=.1, three are identical p=1: common m=7.
        self.assertAlmostEqual(alpha["q_value"], .175)
        self.assertEqual(alpha["effect"], 1)
        self.assertEqual(alpha["effect_ci_low"], 1)
        self.assertEqual(alpha["effect_ci_high"], 1)
        self.assertEqual(alpha["n_total"], 6)
        self.assertEqual(tests["p_value"].notna().sum(), 7)

    def test_welch_matches_reference_and_hedges_direction(self):
        first, second = np.array([10, 13, 14, 19]), np.array([25, 29, 31, 40])
        _, tests = perform_statistics(animal_rows({"A": first, "B": second}), config("welch"))
        result = tests.set_index("metric").loc["alpha_helix"]
        self.assertEqual(result["method"], "Welch t-test")
        self.assertAlmostEqual(result["p_value"], stats.ttest_ind(first, second, equal_var=False).pvalue)
        self.assertGreater(result["effect"], 0)
        self.assertTrue(np.isfinite(result["effect_ci_low"]))
        self.assertIn("Hedges", result["effect_name"])

    def test_declared_method_is_respected_and_bad_method_rejected(self):
        animals = animal_rows({"A": [10, 11, 12], "B": [20, 25, 29]})
        _, mw = perform_statistics(animals, config())
        _, welch = perform_statistics(animals, config("welch"))
        self.assertIn("Mann-Whitney", mw.set_index("metric").loc["alpha_helix", "method"])
        self.assertEqual(welch.set_index("metric").loc["alpha_helix", "method"], "Welch t-test")
        with self.assertRaisesRegex(ValueError, "Predeclared"):
            perform_statistics(animals, config("shapiro_auto"))

    def test_tied_small_mwu_uses_permutation(self):
        animals = animal_rows({"A": [10, 10, 11], "B": [20, 20, 21]})
        _, tests = perform_statistics(animals, config())
        alpha = tests.set_index("metric").loc["alpha_helix"]
        self.assertIn("permutation, ties", alpha["method"])
        self.assertAlmostEqual(alpha["p_value"], .1)

    def test_three_group_exact_permutation_and_gated_posthoc(self):
        animals = animal_rows({"A": [10, 11, 12], "B": [30, 31, 32], "C": [50, 51, 52]})
        _, tests = perform_statistics(animals, config(bootstrap_iterations=40))
        alpha = tests.query("metric == 'alpha_helix'")
        omnibus = alpha.query("hypothesis == 'omnibus'").iloc[0]
        self.assertEqual(omnibus["method"], "Kruskal-Wallis (permutation)")
        self.assertAlmostEqual(omnibus["p_value"], 6 / 1680)
        self.assertEqual(alpha.query("hypothesis == 'posthoc'")["p_value"].notna().sum(), 3)
        null_endpoint = tests.query("metric == 'random_coil' and hypothesis == 'posthoc'")
        self.assertTrue((null_endpoint["p_value"] == 1).all())
        self.assertTrue((null_endpoint["q_value"] == 1).all())
        self.assertFalse(null_endpoint["interpretation_allowed"].any())
        self.assertTrue(null_endpoint["status"].str.contains("gate").all())
        self.assertTrue(alpha.query("hypothesis == 'posthoc'")["interpretation_allowed"].all())
        self.assertEqual(tests["p_value"].notna().sum(), 28)
        finite = tests["p_value"].dropna()
        order = np.argsort(finite.to_numpy())
        expected = np.minimum.accumulate((finite.to_numpy()[order]*len(finite)/np.arange(1, len(finite)+1))[::-1])[::-1]
        np.testing.assert_allclose(tests.loc[finite.index, "q_value"].to_numpy()[order], expected)

    def test_welch_anova_matches_scipy_and_degenerate_endpoint_explicit(self):
        samples = [[10, 11, 14, 15], [25, 26, 29, 31], [40, 42, 43, 45]]
        _, tests = perform_statistics(animal_rows(dict(zip(["A", "B", "C"], samples))),
                                      config("welch", bootstrap_iterations=50))
        omnibus = tests.query("metric == 'alpha_helix' and hypothesis == 'omnibus'").iloc[0]
        self.assertEqual(omnibus["method"], "Welch ANOVA")
        # Independent trusted reference from SciPy 1.17 f_oneway(equal_var=False).
        # Store its value so regression coverage also runs on supported SciPy 1.11.
        np.testing.assert_allclose(omnibus["p_value"], 6.692962416635452e-06, rtol=1e-10)
        null_endpoint = tests.query("metric == 'random_coil' and hypothesis == 'omnibus'").iloc[0]
        self.assertTrue(np.isnan(null_endpoint["p_value"]))
        self.assertIn("Zero variance", null_endpoint["status"])

    def test_unreliable_animals_do_not_enter_sample_sizes_or_means(self):
        animals = animal_rows({"A": [10, 11, 12], "B": [20, 21, 22]})
        animals.loc[0, "status"] = "Unreliable fit"
        animals.loc[0, "alpha_helix"] = 99
        desc, _ = perform_statistics(animals, config())
        row = desc.query("metric == 'alpha_helix' and group == 'A'").iloc[0]
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["mean"], 11.5)

    def test_missing_log_values_are_not_zero_and_endpoint_counts_differ(self):
        animals = animal_rows({"A": [0, 10, 12], "B": [20, 21, 22]})
        desc, tests = perform_statistics(animals, config())
        self.assertEqual(desc.query("metric == 'alpha_helix' and group == 'A'")["n"].iloc[0], 3)
        self.assertEqual(desc.query("metric == 'log_alpha_beta' and group == 'A'")["n"].iloc[0], 2)
        self.assertEqual(tests.set_index("metric").loc["log_alpha_beta", "n1"], 2)

    def test_ci_is_reproducible_under_input_order_changes(self):
        animals = animal_rows({"A": [10, 13, 14, 19], "B": [25, 29, 31, 40]})
        _, first = perform_statistics(animals, config("welch"))
        _, second = perform_statistics(animals.sample(frac=1, random_state=3), config("welch"))
        # Samples must also have canonical animal ordering, not only fixed RNG.
        np.testing.assert_allclose(first["effect_ci_low"], second["effect_ci_low"], equal_nan=True)
        np.testing.assert_allclose(first["effect_ci_high"], second["effect_ci_high"], equal_nan=True)

    def test_empty_and_one_group_report_explicit_unavailable_status(self):
        empty = pd.DataFrame(columns=ANIMAL_COLUMNS)
        desc, tests = perform_statistics(empty, config())
        self.assertTrue(desc.empty)
        self.assertEqual(len(tests), 7)
        self.assertTrue(tests["status"].str.contains("Insufficient groups").all())
        _, tests = perform_statistics(animal_rows({"A": [10, 11, 12]}), config())
        self.assertTrue(tests["p_value"].isna().all())

    def test_duplicate_animals_rejected_before_pseudoreplication(self):
        animals = animal_rows({"A": [10, 11], "B": [20, 21]})
        with self.assertRaisesRegex(ValueError, "unique"):
            perform_statistics(pd.concat([animals, animals.iloc[[0]]]), config())

    def test_mean_ci_assumption_is_exported_and_beta_merge_is_additive(self):
        settings = config()
        settings["plots"]["merge_beta_turn"] = True
        desc, tests = perform_statistics(animal_rows({"A": [10, 11, 12], "B": [20, 21, 22]}), settings)
        self.assertEqual(tests["metric"].nunique(), 8)
        self.assertIn("beta_associated", tests["metric"].tolist())
        self.assertTrue(set(STRUCTURES).issubset(set(tests["metric"])))
        self.assertTrue(desc["mean_ci_method"].str.contains("assumes independent").all())

    def test_iqr_matches_known_animal_distribution_and_is_missing_without_data(self):
        animals = animal_rows({"A": [10, 12, 14, 18], "B": [20, 21, 22]})
        animals.loc[animals["group"] == "B", "status"] = "Unreliable fit"
        desc, _ = perform_statistics(animals, config())
        alpha = desc.query("metric == 'alpha_helix'").set_index("group")
        # Linear sample quartiles: Q1=11.5, Q3=15, so IQR=3.5.
        self.assertEqual(alpha.loc["A", "q25"], 11.5)
        self.assertEqual(alpha.loc["A", "q75"], 15)
        self.assertEqual(alpha.loc["A", "iqr"], 3.5)
        self.assertTrue(np.isnan(alpha.loc["B", "iqr"]))


if __name__ == "__main__":
    unittest.main()
