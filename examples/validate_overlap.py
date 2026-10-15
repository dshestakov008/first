#!/usr/bin/env python3
"""Independent truth checks for sampled, overlapping synthetic Amide I bands.

These mathematical examples are not animal measurements. Their declared
4 cm^-1 resolution must not be reused as metadata for real spectra. The broad
four-component case is allowed to remain diagnostic: a high R-squared alone
does not establish recovery of its generating components.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amide_analysis import __version__
from amide_analysis.config import default_config, load_config
from amide_analysis.fitting import analyze_band
from amide_analysis.pipeline import _processing_variants
from amide_analysis.preprocess import preprocess_spectrum


NOTICE = "SYNTHETIC VALIDATION ONLY — NOT EXPERIMENTAL OR ANIMAL DATA"
SAMPLING_STEP_CM1 = 1.92905
NOISE_SD = 0.000375
NOISE_SEED = 91
FIT_SEED = 912
_LOG4 = 4.0 * np.log(2.0)


def validation_cases():
    """Return fixed cases chosen by identifiable versus unresolved overlap."""
    separated = [(0.07, 1630.0, 12.0), (0.06, 1655.0, 10.0),
                 (0.03, 1674.0, 12.0)]
    moderate = [(0.07, 1630.0, 20.0), (0.06, 1655.0, 18.0),
                (0.03, 1674.0, 16.0)]
    broad = [(0.06, 1632.0, 28.0), (0.05, 1654.0, 30.0),
             (0.025, 1671.0, 28.0), (0.015, 1683.0, 30.0)]
    return [
        {"name": "well_separated", "parameters": separated,
         "resolution_confirmed": True, "require_recovery": True},
        {"name": "moderate_overlap", "parameters": moderate,
         "resolution_confirmed": True, "require_recovery": True},
        {"name": "single_peak_with_noise", "parameters": [(0.08, 1655.0, 20.0)],
         "resolution_confirmed": True, "require_recovery": True},
        {"name": "broad_overlap", "parameters": broad,
         "resolution_confirmed": True, "require_recovery": False},
        {"name": "assignment_boundary", "parameters": [(0.08, 1650.0, 20.0)],
         "resolution_confirmed": True, "require_recovery": False},
        {"name": "noise_only", "parameters": [],
         "resolution_confirmed": True, "require_recovery": False},
        {"name": "unknown_resolution", "parameters": moderate,
         "resolution_confirmed": False, "require_recovery": False},
    ]


def synthetic_spectrum(parameters):
    """Generate data independently of the fitter's Gaussian/area functions."""
    x = np.arange(450.0, 4000.0, SAMPLING_STEP_CM1)
    y = 0.09 + 0.00001 * (x - 450.0)
    for amplitude, center, width in parameters:
        y += amplitude * np.exp(-_LOG4 * ((x - center) / width) ** 2)
    y += np.random.default_rng(NOISE_SEED).normal(0.0, NOISE_SD, len(x))
    # Use numerical quadrature, not the implementation's analytic Gaussian
    # integral, to establish independent areas within the analyzed band.
    areas = np.array([
        quad(lambda value: amplitude * np.exp(-_LOG4 * ((value - center) / width) ** 2),
             1600.0, 1700.0, epsabs=1e-12)[0]
        for amplitude, center, width in parameters
    ])
    truth = 100.0 * areas / areas.sum() if len(areas) else areas
    return {"x": x, "y": y}, truth


def analyze_case(case, config=None):
    """Run the actual preprocessing, sensitivity checks, fit, and bootstrap."""
    config = deepcopy(default_config() if config is None else config)
    config["resolution_confirmed"] = case["resolution_confirmed"]
    config["spectral_resolution_cm1"] = 4.0
    config = load_config(config)
    raw, truth = synthetic_spectrum(case["parameters"])
    processed = preprocess_spectrum(raw["x"], raw["y"], config)
    variants, failures = _processing_variants(raw, config)
    if failures:
        raise RuntimeError("; ".join(failures))
    processed["sensitivity_variants"] = variants
    fit = analyze_band(processed, config, seed=FIT_SEED)
    return fit, truth, processed


def summarize(case, fit, truth):
    """An accepted decomposition must recover truth to claim validation."""
    components = fit["components"]
    count_matches = len(components) == len(case["parameters"])
    center_error = area_error = None
    if components and count_matches:
        center_error = float(np.max(np.abs(
            np.array([row["center"] for row in components])
            - np.array([row[1] for row in case["parameters"]]))))
        area_error = float(np.max(np.abs(
            np.array([row["diagnostic_area_pct"] for row in components]) - truth)))
    reliable = fit["status"] == "Reliable fit"
    recovery = (count_matches and center_error is not None
                and center_error <= 0.5 and area_error <= 2.0)
    withheld = (all(np.isnan(value) for value in fit["percentages"].values())
                and all(np.isnan(row["area_pct"]) and not row["accepted"]
                        for row in components))
    if case["require_recovery"]:
        passed = reliable and recovery
        criterion = "Accepted fit recovers centers within 0.5 cm^-1 and areas within 2 percentage points."
    elif case["name"] == "broad_overlap":
        passed = (reliable and recovery) or (not reliable and withheld)
        criterion = "Recover known components if accepted; otherwise withhold uncertain quantitative estimates."
    elif case["name"] == "assignment_boundary":
        passed = ((reliable and all(row["structure"] == "unassigned" for row in components))
                  or (not reliable and withheld))
        criterion = "Do not force a component on a structural assignment boundary into a protein category."
    else:
        passed = not reliable and withheld
        criterion = "Withhold accepted estimates for noise-only data or unconfirmed instrument resolution."
    return {
        "case": case["name"], "notice": NOTICE, "validation_passed": bool(passed),
        "validation_criterion": criterion,
        "status": fit["status"], "warnings": fit["warnings"],
        "fit_rejection_codes": fit["fit_rejection_codes"],
        "protocol_blockers": fit["protocol_blockers"],
        "numerical_fit_reliable": fit["numerical_fit_reliable"],
        "quantitative_accepted": fit["quantitative_accepted"],
        "resolution_confirmed": case["resolution_confirmed"],
        "known_component_count": len(case["parameters"]),
        "selected_component_count": len(components),
        "known_band_area_pct": truth.tolist(),
        "max_center_error_cm1": center_error,
        "max_area_error_percentage_points": area_error,
        "accepted_percentages_withheld": bool(withheld),
        "metrics": fit["metrics"], "components": components,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing files are preserved.")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory to preserve existing files.")
    output.mkdir(parents=True, exist_ok=True)
    config = default_config()
    results = []
    for case in validation_cases():
        fit, truth, _ = analyze_case(case, config)
        result = summarize(case, fit, truth)
        results.append(result)
        print(f"{case['name']}: {fit['status']}; validation={result['validation_passed']}",
              flush=True)
    all_checks_passed = all(row["validation_passed"] for row in results)
    manifest = {
        "notice": NOTICE, "sampling_step_cm1": SAMPLING_STEP_CM1,
        "software_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "all_checks_passed": all_checks_passed,
        "declared_synthetic_resolution_cm1": 4.0,
        "noise_sd": NOISE_SD, "noise_seed": NOISE_SEED, "fit_seed": FIT_SEED,
        "analysis_config": config, "generating_cases": validation_cases(),
        "results": results,
        "limitations": [
            "Validation covers these Gaussian examples, not arbitrary real FTIR bands.",
            "Detected derivative minima need not reveal all generating components.",
            "Component labels and areas do not measure absolute protein abundance.",
            "Real instrument resolution remains unconfirmed until supplied by the experimenter.",
        ],
    }
    (output / "Overlap_Validation.json").write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(output / "Overlap_Validation.json")
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
