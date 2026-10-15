"""Configured search bounds must be reported with their actual values."""

import unittest
import numpy as np

from amide_analysis.config import default_config
from amide_analysis.fitting import _model_constraint_details, gaussian


class ConstraintReportingTests(unittest.TestCase):
    def test_width_ceiling_is_explained_as_a_configured_limit(self):
        config = default_config()
        x = np.linspace(1600., 1700., 101)
        upper = config["peaks"]["max_fwhm_cm1"]
        model = {"params": np.array([[.7, 1655., upper]]),
                 "condition_number": 100.}
        details = _model_constraint_details(model, x, config)
        hit = next(row for row in details if row["reason"] == "WIDTH_UPPER_BOUND")
        self.assertEqual(hit["component"], 1)
        self.assertEqual(hit["value"], upper)
        self.assertEqual(hit["upper"], upper)
        self.assertIn("configured numerical", hit["source"])
        self.assertEqual(config["spectral_resolution_cm1"], 4.)
        self.assertNotEqual(hit["upper"], config["spectral_resolution_cm1"])

    def test_widening_the_ceiling_does_not_move_the_lower_limit_tolerance(self):
        config = default_config()
        x = np.linspace(1600., 1700., 101)
        model = {"params": np.array([[.7, 1655., 6.02]]),
                 "condition_number": 100.}
        for upper in (40., 80.):
            config["peaks"]["max_fwhm_cm1"] = upper
            reasons = [row["reason"] for row in _model_constraint_details(model, x, config)]
            self.assertNotIn("WIDTH_LOWER_BOUND", reasons)


if __name__ == "__main__":
    unittest.main()
