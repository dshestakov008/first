"""Plot each Gaussian on the same baseline and units as the observed curve."""

from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from amide_analysis import plotting


class RefinedBaselineFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_joint_baseline_aligns_signal_gaussians_and_residuals(self):
        x = np.linspace(1600., 1700., 101)
        gaussian = .7 * np.exp(-4 * np.log(2) * ((x - 1655.) / 16.) ** 2)
        adjustment = -.02 + .0005 * (x - 1600.)
        initial_baseline = .1 + .0001 * (x - 1600.)
        residual = .0001 * np.sin(x)
        factor = 2.5
        corrected = gaussian + adjustment + residual
        original = initial_baseline + factor * corrected
        processed = {"x": x, "raw": original, "baseline": initial_baseline,
                     "corrected": corrected, "smoothed": gaussian + adjustment,
                     "derivatives": {}, "normalization_factor": factor}
        fit = {"status": "Unreliable fit", "warnings": [],
               "components": [{"center": 1655., "amplitude": .7, "fwhm": 16.,
                               "structure": "alpha_helix", "area_pct": np.nan}],
               "fit": gaussian + adjustment, "residuals": residual,
               "baseline_adjustment": adjustment}
        captured = {}

        def capture(fig, output_dir, stem, config):
            captured[stem] = fig
            return []

        with patch.object(plotting, "_save", side_effect=capture):
            plotting.plot_spectrum({"x": x, "y": original}, processed, fit,
                                   Path("unused"), "spectrum", {"plots": {}})
        baseline_axes = captured["spectrum_baseline"].axes
        original_lines = {line.get_label(): line for line in baseline_axes[0].lines}
        np.testing.assert_allclose(original_lines["Amide I before correction"].get_ydata(), original)
        np.testing.assert_allclose(original_lines["Initial estimated baseline"].get_ydata(), initial_baseline)
        np.testing.assert_allclose(original_lines["Jointly refined linear baseline"].get_ydata(),
                                   initial_baseline + adjustment * factor)
        self.assertEqual(baseline_axes[0].get_ylabel(), "Absorbance")
        self.assertEqual(baseline_axes[1].get_ylabel(), "Normalised signal (a.u.)")
        lines = {line.get_label(): line for line in captured["spectrum_decomposition"].axes[0].lines}
        np.testing.assert_allclose(lines["Signal after jointly refined baseline"].get_ydata(),
                                   gaussian + residual)
        np.testing.assert_allclose(lines["Gaussian sum"].get_ydata(), gaussian, atol=1e-14)
        component = next(line for label, line in lines.items() if label.startswith("1:"))
        np.testing.assert_allclose(component.get_ydata(), gaussian)
        residual_line = captured["spectrum_decomposition"].axes[1].lines[0]
        np.testing.assert_allclose(residual_line.get_ydata(), residual)
        self.assertIn("jointly refined baseline", captured["spectrum_decomposition"].axes[0].get_title())

    def test_residual_fallback_uses_refined_observed_signal(self):
        x = np.linspace(1600., 1700., 51)
        gaussian = np.exp(-4 * np.log(2) * ((x - 1655.) / 16.) ** 2)
        adjustment = np.linspace(-.1, .1, len(x))
        residual = np.full(len(x), .02)
        corrected = gaussian + adjustment + residual
        processed = {"x": x, "raw": corrected, "baseline": np.zeros_like(x),
                     "corrected": corrected, "smoothed": corrected, "derivatives": {}}
        fit = {"status": "Reliable fit", "warnings": [],
               "components": [{"center": 1655., "amplitude": 1., "fwhm": 16.,
                               "structure": "alpha_helix", "area_pct": 100.}],
               "fit": gaussian + adjustment, "baseline_adjustment": adjustment}
        captured = {}

        def capture(fig, output_dir, stem, config):
            captured[stem] = fig
            return []

        with patch.object(plotting, "_save", side_effect=capture):
            plotting.plot_spectrum({"x": x, "y": corrected}, processed, fit,
                                   Path("unused"), "spectrum", {"plots": {}})
        residual_line = captured["spectrum_decomposition"].axes[1].lines[0]
        np.testing.assert_allclose(residual_line.get_ydata(), residual)


if __name__ == "__main__":
    unittest.main()
