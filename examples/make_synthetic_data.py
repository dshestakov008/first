#!/usr/bin/env python3
"""Create explicitly synthetic FTIR spectra with known generating parameters.

No real samples or observed biological effects are represented here. There are
three generating Amide I peaks, not four imposed structural categories. Truth
areas are Gaussian integrals truncated to 1600--1700 cm^-1, matching the analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import erf

# Permit direct execution from the repository without package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amide_analysis.config import default_config


SYNTHETIC_NOTICE = (
    "SYNTHETIC DATA ONLY. Simulated spectra and simulated animal IDs; "
    "not measurements from BALB/c mice or a 4T1 experiment."
)
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian_band_area(center: float, fwhm: float, amplitude: float,
                       lower: float = 1600.0, upper: float = 1700.0) -> float:
    """Analytic area within the stated band, without infinite-domain tails."""
    sigma = fwhm / FWHM_FACTOR
    fraction = 0.5 * (
        erf((upper - center) / (np.sqrt(2.0) * sigma))
        - erf((lower - center) / (np.sqrt(2.0) * sigma))
    )
    return float(amplitude * sigma * np.sqrt(2.0 * np.pi) * fraction)


def make_dataset(output: Path, seed: int) -> dict:
    """Write nine spectra for eight simulated animals and a truth manifest."""
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Output folder is not empty: {output}. Choose a new --output folder "
            "to preserve existing files."
        )
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = np.arange(450.0, 4000.0 + 1e-8, 1.93)
    nominal_centers = np.array([1630.0, 1655.0, 1674.0])
    fwhms = np.array([12.0, 10.0, 12.0])
    structures = ["beta_sheet", "alpha_helix", "beta_turn"]
    metadata = []
    truths = []
    animal_truth = []

    for group, nominal_areas in (
        ("Control", np.array([0.40, 0.40, 0.20])),
        ("Cancer", np.array([0.48, 0.29, 0.23])),
    ):
        for animal_index in range(1, 5):
            animal_id = f"SYNTHETIC_{group}_{animal_index:02d}"
            area_fractions = nominal_areas + rng.normal(0.0, 0.008, 3)
            area_fractions = area_fractions / area_fractions.sum()
            centers = nominal_centers + rng.uniform(-0.15, 0.15, 3)
            total_amide_area = float(rng.uniform(2.0, 2.3))
            known_areas = area_fractions * total_amide_area
            amplitudes = np.array([
                area / gaussian_band_area(center, width, 1.0)
                for area, center, width in zip(known_areas, centers, fwhms)
            ])
            amide = sum(
                amplitude * np.exp(-0.5 * ((x - center) / (width / FWHM_FACTOR)) ** 2)
                for amplitude, center, width in zip(amplitudes, centers, fwhms)
            )
            # Other separated bands make this a full-spectrum input. They carry
            # no claims about structural identities and do not overlap Amide I.
            extra_bands = sum(
                height * np.exp(-0.5 * ((x - center) / sigma) ** 2)
                for center, sigma, height in (
                    (1080.0, 30.0, 0.045), (1450.0, 18.0, 0.025),
                    (2850.0, 18.0, 0.028), (2930.0, 22.0, 0.035),
                    (3300.0, 65.0, 0.04),
                )
            )
            baseline = 0.07 + 0.000015 * (x - 450.0)
            noise_sd = float(amide.max() / 200.0)
            percentages = dict.fromkeys(
                ["random_coil", "alpha_helix", "beta_sheet", "beta_turn", "unassigned"],
                0.0,
            )
            percentages.update({key: float(100.0 * frac)
                                for key, frac in zip(structures, area_fractions)})
            animal_truth.append({
                "animal_id": animal_id, "group": group,
                "known_generating_area_pct": percentages,
                "notice": SYNTHETIC_NOTICE,
            })
            repeats = 2 if group == "Control" and animal_index == 1 else 1
            for repeat in range(1, repeats + 1):
                y = baseline + extra_bands + amide + rng.normal(0.0, noise_sd, x.size)
                relative_file = Path(group) / f"{animal_id}_repeat_{repeat}.csv"
                file_path = output / "spectra" / relative_file
                file_path.parent.mkdir(parents=True, exist_ok=True)
                # Alternate exported axis direction to exercise input sorting.
                order = slice(None, None, -1) if animal_index % 2 == 0 else slice(None)
                pd.DataFrame({
                    "wavenumber_cm1": x[order], "absorbance": y[order],
                }).to_csv(file_path, index=False, float_format="%.12g")
                metadata.append({
                    "file": relative_file.as_posix(), "animal_id": animal_id,
                    "group": group, "technical_repeat": repeat,
                })
                truths.append({
                    "file": relative_file.as_posix(), "animal_id": animal_id,
                    "group": group, "technical_repeat": repeat,
                    "notice": SYNTHETIC_NOTICE,
                    "nominal_peak_snr": 200.0, "noise_sd": noise_sd,
                    "generating_components": [
                        {
                            "structure": structure, "center_cm1": float(center),
                            "fwhm_cm1": float(width), "amplitude": float(amplitude),
                            "band_area_1600_1700": float(area),
                            "band_area_pct": float(fraction * 100.0),
                        }
                        for structure, center, width, amplitude, area, fraction
                        in zip(structures, centers, fwhms, amplitudes,
                               known_areas, area_fractions)
                    ],
                    "known_generating_area_pct": percentages,
                })

    pd.DataFrame(metadata).to_excel(output / "metadata.xlsx", index=False)
    config = default_config()
    # This resolution is a declared property of this mathematical example;
    # it must never be copied as a confirmed resolution for real measurements.
    # Other analysis settings, including circular residual bootstrap blocks,
    # inherit the program defaults rather than being tuned to the known truth.
    config["spectral_resolution_cm1"] = 4.0
    config["resolution_confirmed"] = True
    config["dataset_label"] = "SYNTHETIC VALIDATION — NOT EXPERIMENTAL DATA"
    config["seed"] = seed
    (output / "synthetic_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    truth = {
        "notice": SYNTHETIC_NOTICE, "dataset_label": config["dataset_label"], "seed": seed,
        "spectral_resolution_cm1": 4.0,
        "resolution_meaning": "Declared synthetic example parameter; not real instrument metadata.",
        "sampling_step_cm1": 1.93, "amide_bounds_cm1": [1600.0, 1700.0],
        "full_axis_bounds_cm1": [float(x.min()), float(x.max())],
        "baseline": "0.07 + 0.000015 * (wavenumber_cm1 - 450.0)",
        "noise": "Independent Gaussian additive noise, nominal peak SNR 200.",
        "generating_peak_count": 3, "simulated_animal_count": 8,
        "spectrum_count": len(truths), "animals": animal_truth, "spectra": truths,
    }
    (output / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "SYNTHETIC_DATA_README.txt").write_text(
        SYNTHETIC_NOTICE + "\n\n"
        "These files test fitting and animal-level repeat aggregation.\n"
        "Known areas refer to generating Gaussians within 1600--1700 cm^-1.\n"
        "There is no generating random-coil component; the fitter must not force one.\n"
        "Use the supplied metadata.xlsx; one simulated animal has two technical repeats.\n"
        "Never report these outputs as experimental measurements.\n",
        encoding="utf-8",
    )
    return truth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "generated_synthetic")
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    try:
        truth = make_dataset(args.output, args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Synthetic data generation failed: {exc}\n")
    print(SYNTHETIC_NOTICE)
    print(f"Created {truth['spectrum_count']} spectra for "
          f"{truth['simulated_animal_count']} simulated animals in {args.output.resolve()}")
    print("Known generating parameters: truth.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
