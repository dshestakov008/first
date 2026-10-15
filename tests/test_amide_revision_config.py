"""Compatibility and validity of bootstrap retry configuration."""

import json
from pathlib import Path
import tempfile
import unittest

from amide_analysis.config import default_config, load_config


class BootstrapRetryConfigurationTests(unittest.TestCase):
    def test_old_saved_configuration_retains_new_retry_default_and_existing_gates(self):
        original = default_config()
        original["fitting"].pop("bootstrap_retry_multistarts")
        original["fitting"]["bootstrap_iterations"] = 12
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_config.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            loaded = load_config(path)
        self.assertEqual(loaded["fitting"]["bootstrap_retry_multistarts"], 3)
        self.assertEqual(loaded["fitting"]["bootstrap_iterations"], 12)
        self.assertFalse(loaded["resolution_confirmed"])
        self.assertEqual(loaded["fitting"]["bootstrap_success_fraction"], 0.8)
        self.assertEqual(loaded["fitting"]["max_percentage_sd"], 8.0)
        self.assertEqual(loaded["fitting"]["min_model_delta"], 6.0)

    def test_partial_fitting_configuration_retains_original_settings(self):
        loaded = load_config({"fitting": {"bootstrap_retry_multistarts": 5}})
        self.assertEqual(loaded["fitting"]["bootstrap_retry_multistarts"], 5)
        self.assertEqual(loaded["fitting"]["multistarts"], 6)
        self.assertEqual(loaded["fitting"]["bootstrap_iterations"], 30)
        self.assertFalse(loaded["resolution_confirmed"])

    def test_retry_budget_rejects_invalid_values_instead_of_coercing(self):
        for value in (0, -1, 1.0, 1.5, True, False, "3", None, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "fitting.bootstrap_retry_multistarts"):
                load_config({"fitting": {"bootstrap_retry_multistarts": value}})

    def test_single_start_budget_is_supported(self):
        loaded = load_config({"fitting": {"bootstrap_retry_multistarts": 1}})
        self.assertEqual(loaded["fitting"]["bootstrap_retry_multistarts"], 1)

    def test_project_config_confirms_user_instrument_without_changing_qc_gates(self):
        path = Path(__file__).resolve().parents[1] / "analysis_config.json"
        expected = default_config()
        expected["resolution_confirmed"] = True
        self.assertEqual(load_config(path), expected)


if __name__ == "__main__":
    unittest.main()
