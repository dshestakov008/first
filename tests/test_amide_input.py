"""Regression checks for independent data import and preprocessing invariants."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter

from amide_analysis.config import default_config, load_config
from amide_analysis.io import discover_inputs, read_spectrum
from amide_analysis.preprocess import preprocess_spectrum


class ConfigurationTests(unittest.TestCase):
    def test_deep_merge_defaults_do_not_mutate(self):
        config = load_config({"baseline": {"method": "als"}})
        self.assertEqual(config["baseline"]["method"], "als")
        self.assertEqual(config["baseline"]["edge_points"], 3)
        config["smoothing"]["derivative_windows_cm1"].append(19)
        self.assertEqual(default_config()["smoothing"]["derivative_windows_cm1"], [9, 13, 17])
        self.assertFalse(load_config()["resolution_confirmed"])

    def test_unsafe_settings_fail_instead_of_coercion(self):
        cases = (
            {"basline": {}},
            {"baseline": {"method": "unknown"}},
            {"baseline": {"als_p": 0}},
            {"resolution_confirmed": "true"},
            {"peaks": {"min_fwhm_cm1": 40, "max_fwhm_cm1": 6}},
            {"smoothing": {"window_cm1": 50}},
            {"smoothing": {"polyorder": 1}},
            {"statistics": {"method": "auto"}},
            {"seed": True},
            {"plots": {"colors": {"Control": "not a color"}}},
        )
        for update in cases:
            with self.subTest(update=update), self.assertRaises(ValueError):
                load_config(update)

    def test_json_rejects_duplicates_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for content in ('{"seed":1,"seed":2}', '{"spectral_resolution_cm1":NaN}'):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_project_config_differs_only_by_confirmed_instrument_resolution(self):
        path = Path(__file__).resolve().parents[1] / "analysis_config.json"
        expected = default_config()
        expected["resolution_confirmed"] = True
        self.assertEqual(load_config(path), expected)

    def test_custom_labels_sizes_and_block_length(self):
        config = load_config({"plots": {"group_labels": {"Control": "Контроль"},
                                          "metric_labels": {"alpha_helix": "α-спирали"},
                                          "point_size": 30, "bar_width": .6},
                              "fitting": {"bootstrap_block_length": 4}})
        self.assertEqual(config["plots"]["group_labels"]["Control"], "Контроль")
        self.assertEqual(config["fitting"]["bootstrap_block_length"], 4)
        for bad in ({"plots": {"group_labels": {"Control": 1}}},
                    {"plots": {"bar_width": 0}},
                    {"fitting": {"bootstrap_block_length": 1.5}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_config(bad)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def spectrum_file(self, name="spectrum.csv", text="x;y\n1600;0\n1650;1\n1700;0\n"):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_csv_decimal_comma_empty_duplicate_sort_original_rows(self):
        path = self.spectrum_file(text="Wavenumber;Absorbance\n1700;0,2\n1650;1,1\n;\n1650;0,9\n1600;0,1\n1620;\n")
        result = read_spectrum(path)
        np.testing.assert_array_equal(result["x"], [1600, 1650, 1700])
        np.testing.assert_allclose(result["y"], [.1, 1, .2])
        self.assertEqual(result["raw_rows"], 6)
        self.assertEqual(result["cleaned_rows"], 3)
        self.assertEqual(len(result["raw_table"]), 6)
        self.assertEqual(result["raw_table"]["source_row"].tolist(), list(range(2, 8)))
        self.assertGreaterEqual(len(result["warnings"]), 3)

    def test_csv_no_header_preserves_first_point(self):
        result = read_spectrum(self.spectrum_file(text="1600,0\n1650,1\n1700,0\n"))
        np.testing.assert_array_equal(result["x"], [1600, 1650, 1700])

    def test_excel_reversed_headers_extra_column_and_empty_sheet(self):
        path = self.folder / "spectrum.xlsx"
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame([["Absorbance", "Wavenumber", "Note"], [.1, 1600, "a"],
                          [.9, 1650, "b"], [.2, 1700, "c"]]).to_excel(
                writer, sheet_name="Data", header=False, index=False)
            pd.DataFrame().to_excel(writer, sheet_name="Empty", header=False, index=False)
        result = read_spectrum(path)
        np.testing.assert_array_equal(result["x"], [1600, 1650, 1700])
        self.assertTrue(any("Столбцы" in warning for warning in result["warnings"]))

    def test_multiple_occupied_excel_sheets_fail(self):
        path = self.folder / "two.xlsx"
        with pd.ExcelWriter(path) as writer:
            for name in ("A", "B"):
                pd.DataFrame([[1600, 0], [1650, 1], [1700, 0]]).to_excel(
                    writer, sheet_name=name, header=False, index=False)
        with self.assertRaisesRegex(ValueError, "один непустой|ровно один"):
            read_spectrum(path)

    def test_bad_values_even_with_missing_partner_fail(self):
        for bad in ("bad", "inf", "NaN", "-inf"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                read_spectrum(self.spectrum_file(text=f"x;y\n1600;0\n1650;1\n;{bad}\n1700;0\n"))

    def test_transmittance_and_intensity_are_not_treated_as_absorbance(self):
        for label in ("Transmittance", "пропускание", "T", "%T", "Intensity", "Интенсивность", "I"):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "только Absorbance"):
                read_spectrum(self.spectrum_file(text=f"Wavenumber;{label}\n1600;30\n1650;10\n1700;30\n"))
        result = read_spectrum(self.spectrum_file(text="x;y\n1600;0\n1650;1\n1700;0\n"))
        self.assertTrue(any("принят за Absorbance" in warning for warning in result["warnings"]))

    def test_corrupt_excel_error_has_filename_and_valueerror_type(self):
        path = self.folder / "broken.xlsx"
        path.write_bytes(b"corrupt or mislabeled Excel export")
        with self.assertRaisesRegex(ValueError, "broken.xlsx"):
            read_spectrum(path)

    def test_default_metadata_never_infers_repeats(self):
        self.spectrum_file("Control/mouse1_rep1.csv")
        self.spectrum_file("Control/mouse1_rep2.csv")
        self.spectrum_file("root.csv")
        self.spectrum_file("outputs/ignored.csv")
        self.spectrum_file("Amide_results_20261014/ignored.csv")
        self.spectrum_file("Control/~$temporary.xlsx")
        result = discover_inputs(self.folder)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["animal_id"].nunique(), 3)
        self.assertEqual(set(result["group"]), {"Control", "Ungrouped"})

    def test_metadata_explicit_repeats_and_group_conflict(self):
        self.spectrum_file("Control/a.csv")
        self.spectrum_file("Control/b.csv")
        path = self.folder / "metadata.csv"
        path.write_text("file;animal_id;group;technical_repeat\nControl/a.csv;mouse1;Control;1\nControl/b.csv;mouse1;Control;2\n", encoding="utf-8")
        result = discover_inputs(self.folder)
        self.assertEqual(result["animal_id"].nunique(), 1)
        self.assertEqual(result["technical_repeat"].tolist(), ["1", "2"])
        path.write_text("file;animal_id;group\nControl/a.csv;mouse1;Control\nControl/b.csv;mouse1;Cancer\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "нескольких группах"):
            discover_inputs(self.folder)

    def test_metadata_missing_extra_and_duplicate_mapping_fail(self):
        self.spectrum_file("a.csv")
        self.spectrum_file("b.csv")
        path = self.folder / "metadata.csv"
        for lines in ("a.csv;m1;Control\n", "a.csv;m1;Control\nb.csv;m2;Control\nc.csv;m3;Cancer\n",
                      "a.csv;m1;Control\na.csv;m1;Control\nb.csv;m2;Control\n"):
            path.write_text("file;animal_id;group\n" + lines, encoding="utf-8")
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                discover_inputs(self.folder)

    def test_explicit_metadata_path_is_relative_to_cwd(self):
        self.spectrum_file("a.csv")
        metadata = self.folder / "subjects.csv"
        metadata.write_text("file;animal_id;group\na.csv;M1;Control\n", encoding="utf-8")
        result = discover_inputs(self.folder, os.path.relpath(metadata, Path.cwd()))
        self.assertEqual(result["animal_id"].tolist(), ["M1"])

    def test_custom_output_directory_is_pruned_by_markers(self):
        self.spectrum_file("Control/a.csv")
        output = self.folder / "custom_destination"
        output.mkdir()
        (output / "processing_config.json").write_text("{}", encoding="utf-8")
        (output / "Amide_I_Results.xlsx").write_bytes(b"not an input file")
        self.spectrum_file("custom_destination/diagnostics/a.csv")
        result = discover_inputs(self.folder)
        self.assertEqual(result["spectrum_id"].tolist(), ["Control/a.csv"])


class PreprocessingTests(unittest.TestCase):
    @staticmethod
    def data(step=1):
        x = np.arange(1550, 1750 + step / 2, step, dtype=float)
        y = .15 + .0008 * (x - 1600) + np.exp(-.5 * ((x - 1650) / 9)**2)
        return x, y

    def test_exact_band_and_baseline_corrected_not_smoothed(self):
        x, y = self.data()
        result = preprocess_spectrum(x, y)
        self.assertEqual(result["x"][[0, -1]].tolist(), [1600, 1700])
        np.testing.assert_allclose(result["raw"] - result["baseline"], result["corrected"])
        self.assertFalse(np.array_equal(result["corrected"], result["smoothed"]))
        self.assertGreater(result["noise_sd"], 0)
        self.assertTrue(any("не подтверждено" in item for item in result["warnings"]))

    def test_fractional_boundaries_use_interpolation_not_extrapolation(self):
        x, y = self.data(step=1.93)
        result = preprocess_spectrum(x, y)
        np.testing.assert_allclose(result["raw"][[0, -1]], np.interp([1600, 1700], x, y))
        np.testing.assert_allclose(np.diff(result["x"]), result["step"])
        self.assertAlmostEqual(result["source_step"], 1.93)
        self.assertNotAlmostEqual(result["source_step"], result["step"])
        self.assertTrue(any("интерполяц" in item for item in result["warnings"]))
        with self.assertRaisesRegex(ValueError, "экстраполяция запрещена"):
            preprocess_spectrum(x[x > 1600], y[x > 1600])

    def test_independent_derivatives_use_delta_and_are_not_resmoothed(self):
        x, y = self.data()
        rng = np.random.default_rng(1)
        y += rng.normal(0, .005, len(y))
        result = preprocess_spectrum(x, y)
        alternative = preprocess_spectrum(x, y, {"smoothing": {"window_cm1": 17}})
        for count, derivative in result["derivatives"].items():
            self.assertEqual(count % 2, 1)
            expected = savgol_filter(result["corrected"], count, 3, deriv=2, delta=result["step"], mode="interp")
            np.testing.assert_array_equal(derivative, expected)
            np.testing.assert_array_equal(derivative, alternative["derivatives"][count])

    def test_als_and_both_normalizations(self):
        x, y = self.data()
        als = preprocess_spectrum(x, y, {"baseline": {"method": "als"}})
        self.assertTrue(np.all(np.isfinite(als["baseline"])))
        true_baseline = .15 + .0008 * (als["x"] - 1600)
        self.assertLess(np.max(np.abs(als["baseline"] - true_baseline)), .08)
        area = preprocess_spectrum(x, y, {"normalization": "area"})
        l2 = preprocess_spectrum(x, y, {"normalization": "l2"})
        self.assertAlmostEqual(trapezoid(area["corrected"], area["x"]), 1)
        self.assertAlmostEqual(trapezoid(l2["corrected"]**2, l2["x"]), 1)

    def test_unsafe_gap_unsorted_or_sparse_data_fail(self):
        x, y = self.data()
        with self.assertRaisesRegex(ValueError, "строго возрастать"):
            preprocess_spectrum(x[::-1], y[::-1])
        mask = (x < 1620) | (x > 1640)
        with self.assertRaisesRegex(ValueError, "пропуск"):
            preprocess_spectrum(x[mask], y[mask])
        x, y = self.data(step=10)
        with self.assertRaisesRegex(ValueError, "Недостаточно точек"):
            preprocess_spectrum(x, y)


if __name__ == "__main__":
    unittest.main()
