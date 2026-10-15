"""Comparison of the nine requested FTIR regions, independent of peak fitting.

The endpoint is the exact-bound trapezoidal integral of an individually
L2-normalized absorbance spectrum divided by the region width. It follows the
regional endpoint in ``ir_region_analysis.py``; it is not a secondary-structure
percentage. Spectra share one non-extrapolated grid over their common measured
range, with spacing no finer than the coarsest median source spacing. Explicit
technical repeats are averaged within each animal before any group test.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import integrate, stats

from .statistics import _omnibus_p, _pair_p, _seed


SPECTRAL_REGIONS = (
    ("Амид I", 1600.0, 1700.0),
    ("Амид II", 1500.0, 1600.0),
    ("Амид III и фосфатные группы", 1200.0, 1350.0),
    ("Липиды, белки и карбоксильные группы", 1350.0, 1500.0),
    ("Углеводы, гликопротеины и фосфаты", 900.0, 1200.0),
    ("Карбонильные группы липидов", 1700.0, 1750.0),
    ("C–H-колебания (липиды и органические компоненты)", 2800.0, 3000.0),
    ("O–H и N–H-колебания (вода, белки)", 3000.0, 3600.0),
    ("Область молекулярных отпечатков", 900.0, 1800.0),
)
METRIC = "Mean individually L2-normalized absorbance in region"
NORMALIZATION = "Discrete L2 norm on the full common measured grid"
SUMMARY_COLUMNS = ["Region", "Range, cm⁻¹", "Median control", "Median cancer",
                   "p-MWU", "p-BH", "r"]
SPECTRUM_COLUMNS = ["spectrum_id", "animal_id", "group", "region_id", "region",
                    "lower_cm1", "upper_cm1", "value", "status", "l2_factor",
                    "metric", "normalization"]
ANIMAL_COLUMNS = ["animal_id", "group", "region_id", "region", "lower_cm1",
                  "upper_cm1", "expected_repeats", "available_repeats", "accepted",
                  "value", "status", "metric"]
QC_COLUMNS = ["scope", "spectrum_id", "animal_id", "group", "region_id", "status",
              "code", "message", "common_grid_lower_cm1", "common_grid_upper_cm1",
              "common_grid_points", "common_grid_step_cm1"]


def mean_signal_in_region(x, y, lower, upper):
    """Integrate a linearly interpolated spectrum at the exact requested limits."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(np.diff(x) <= 0)):
        raise ValueError("Need finite one-dimensional spectra on an increasing grid")
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Region bounds must be finite and increasing")
    if x[0] > lower or x[-1] < upper:
        raise ValueError("The measured grid does not cover the whole region; no extrapolation")
    interior = (x > lower) & (x < upper)
    band_x = np.r_[lower, x[interior], upper]
    band_y = np.r_[np.interp(lower, x, y), y[interior], np.interp(upper, x, y)]
    return float(integrate.trapezoid(band_y, x=band_x) / (upper - lower))


def _metadata(metadata):
    required = ["spectrum_id", "animal_id", "group"]
    if not set(required).issubset(metadata.columns):
        raise ValueError("Regional metadata must contain spectrum_id, animal_id and group")
    result = metadata[required].copy()
    if result.isna().any().any():
        raise ValueError("Regional metadata contains missing identifiers")
    for column in required:
        result[column] = result[column].astype(str)
        if result[column].str.strip().eq("").any():
            raise ValueError(f"Regional metadata contains an empty {column}")
    if result["spectrum_id"].duplicated().any():
        raise ValueError("Duplicate regional spectrum identifiers")
    if result.groupby("animal_id")["group"].nunique().gt(1).any():
        raise ValueError("One regional animal belongs to multiple groups")
    return result


def _group_order(metadata, config):
    groups = sorted(metadata["group"].unique())
    control = ("Control" if "Control" in groups
               else str(config.get("plots", {}).get("control_group", "Control")))
    if control in groups:
        groups.remove(control)
        groups.insert(0, control)
    return groups


def analyze_regions(raw_spectra, metadata: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Return regional values, animal-level tests, a seven-column table and QC.

    ``raw_spectra`` contains dictionaries with ``spectrum_id``, ``x`` and ``y``
    from readable source files. ``metadata`` includes every expected spectrum,
    including read failures. All expected technical repeats must have a finite
    value for an animal to enter that endpoint. Peak-fitting status and the
    instrument-resolution confirmation are deliberately not consulted.

    One predeclared two-sided MW test (or KW omnibus for >2 groups) is performed
    per covered region; all finite p values share one BH correction. The rank
    effect is positive when the second group exceeds the control/first group.
    For arbitrary two-group labels the detailed table records their order; the
    familiar seven-column summary is populated only for Control/Cancer labels.
    """
    meta = _metadata(metadata)
    lookup, failures, quality = {}, {}, []

    def qc(scope, code, message, *, status="Unavailable", spectrum_id="",
           animal_id="", group="", region_id="", **extra):
        quality.append(dict(scope=scope, spectrum_id=spectrum_id, animal_id=animal_id,
                            group=group, region_id=region_id, status=status,
                            code=code, message=message, **extra))

    expected = set(meta["spectrum_id"])
    seen = set()
    for item in raw_spectra:
        spectrum_id = str(item.get("spectrum_id", ""))
        if spectrum_id in seen:
            raise ValueError(f"Duplicate raw regional spectrum: {spectrum_id}")
        if spectrum_id not in expected:
            raise ValueError(f"Raw regional spectrum absent from metadata: {spectrum_id}")
        seen.add(spectrum_id)
        try:
            x = np.asarray(item["x"], dtype=float)
            y = np.asarray(item["y"], dtype=float)
            if (x.ndim != 1 or y.shape != x.shape or len(x) < 3
                    or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or np.any(np.diff(x) <= 0)):
                raise ValueError("Need at least three finite points on an increasing source grid")
            lookup[spectrum_id] = dict(x=x, y=y, step=float(np.median(np.diff(x))))
        except (KeyError, TypeError, ValueError) as exc:
            failures[spectrum_id] = f"Invalid source spectrum: {exc}"

    grid = np.empty(0)
    normalized, factors = {}, {}
    if lookup:
        lower = max(value["x"][0] for value in lookup.values())
        upper = min(value["x"][-1] for value in lookup.values())
        step = max(value["step"] for value in lookup.values())
        intervals = int(math.floor((upper - lower) / step + 1e-9))
        if intervals < 2:
            for spectrum_id in lookup:
                failures[spectrum_id] = "Source spectra have no common grid with at least three points"
            qc("dataset", "NO_COMMON_GRID", "The source spectra have insufficient common measured coverage")
        else:
            grid = np.linspace(lower, upper, intervals + 1)
            qc("dataset", "COMMON_GRID", "One shared grid; linear interpolation without extrapolation. "
               "Individual normalization precedes repeat aggregation. Regional values are not protein percentages.",
               status="OK", common_grid_lower_cm1=float(grid[0]),
               common_grid_upper_cm1=float(grid[-1]), common_grid_points=len(grid),
               common_grid_step_cm1=float(grid[1] - grid[0]))
            for spectrum_id, source in lookup.items():
                signal = np.interp(grid, source["x"], source["y"], left=np.nan, right=np.nan)
                factor = float(np.linalg.norm(signal))
                if not np.isfinite(signal).all() or not np.isfinite(factor) or factor <= 0:
                    failures[spectrum_id] = "Invalid or zero L2 norm on the common measured grid"
                else:
                    normalized[spectrum_id], factors[spectrum_id] = signal / factor, factor
    else:
        qc("dataset", "NO_READABLE_SPECTRA", "No readable source spectra; regional tests were not performed")

    spectra = []
    for record in meta.to_dict("records"):
        spectrum_id = record["spectrum_id"]
        missing_reason = failures.get(spectrum_id, "Expected source spectrum was not read")
        if spectrum_id not in normalized:
            qc("spectrum", "SOURCE_UNAVAILABLE", missing_reason, **record)
        for index, (name, lower, upper) in enumerate(SPECTRAL_REGIONS, 1):
            value = np.nan
            if spectrum_id not in normalized:
                status = missing_reason
            elif grid[0] > lower or grid[-1] < upper:
                status = "Region not fully covered by the common measured grid"
                qc("spectrum_region", "REGION_NOT_COVERED", status,
                   **record, region_id=f"R{index}")
            else:
                value = mean_signal_in_region(grid, normalized[spectrum_id], lower, upper)
                status = "OK"
            spectra.append(dict(**record, region_id=f"R{index}", region=name,
                                lower_cm1=lower, upper_cm1=upper, value=value, status=status,
                                l2_factor=factors.get(spectrum_id, np.nan),
                                metric=METRIC, normalization=NORMALIZATION))
    spectrum_frame = pd.DataFrame(spectra, columns=SPECTRUM_COLUMNS)

    animals = []
    for (animal_id, group), animal_meta in meta.groupby(["animal_id", "group"], sort=True):
        values = spectrum_frame[spectrum_frame["spectrum_id"].isin(animal_meta["spectrum_id"])]
        for index, (name, lower, upper) in enumerate(SPECTRAL_REGIONS, 1):
            endpoint = values[values["region_id"] == f"R{index}"]
            valid = endpoint["status"].eq("OK") & np.isfinite(endpoint["value"])
            count, expected_count = int(valid.sum()), len(animal_meta)
            accepted = count == expected_count and expected_count > 0
            status = "OK" if accepted else "At least one expected repeat has no valid regional value"
            value = float(endpoint["value"].mean()) if accepted else np.nan
            animals.append(dict(animal_id=animal_id, group=group, region_id=f"R{index}",
                                region=name, lower_cm1=lower, upper_cm1=upper,
                                expected_repeats=expected_count, available_repeats=count,
                                accepted=accepted, value=value, status=status, metric=METRIC))
            if not accepted:
                qc("animal_region", "INCOMPLETE_REPEATS", status, animal_id=animal_id,
                   group=group, region_id=f"R{index}")
    animal_frame = pd.DataFrame(animals, columns=ANIMAL_COLUMNS)

    groups = _group_order(meta, config)
    settings = config.get("statistics", {})
    minimum = max(2, int(settings.get("min_n_test", 2)))
    resamples = int(settings.get("permutation_resamples", 9999))
    if resamples < 1:
        raise ValueError("Regional permutation_resamples must be positive")
    alpha = float(settings.get("alpha", .05))
    statistics = []
    for index, (name, lower, upper) in enumerate(SPECTRAL_REGIONS, 1):
        endpoint = animal_frame[(animal_frame["region_id"] == f"R{index}") & animal_frame["accepted"].astype(bool)]
        samples = [endpoint.loc[endpoint["group"] == group, "value"].to_numpy(float) for group in groups]
        counts = dict(zip(groups, map(len, samples)))
        row = dict(region_id=f"R{index}", Region=name, Lower_cm1=lower, Upper_cm1=upper,
                   Tested=False, status="Insufficient groups: at least two required", metric=METRIC,
                   groups=json.dumps(groups, ensure_ascii=False),
                   n_by_group=json.dumps(counts, ensure_ascii=False),
                   group1=groups[0] if groups else "", group2=groups[1] if len(groups) == 2 else "",
                   method="Not tested", p_value=np.nan, q_value_BH_FDR=np.nan,
                   Mann_Whitney_U=np.nan, Effect_size=np.nan,
                   effect_direction="Second group minus first/control group by ranks",
                   Control_N=counts.get("Control", 0), Cancer_N=counts.get("Cancer", 0),
                   Control_median_region_signal=np.nan, Cancer_median_region_signal=np.nan,
                   group_medians=json.dumps({g: float(np.median(s)) if len(s) else None
                                              for g, s in zip(groups, samples)}, ensure_ascii=False),
                   Significant_q_alpha=False, alpha=alpha,
                   family="Nine predeclared regional endpoints; all finite regional p values",
                   interpretation="Regional normalized signal; not a protein secondary-structure percentage")
        for label in ("Control", "Cancer"):
            if label in groups:
                sample = samples[groups.index(label)]
                row[f"{label}_median_region_signal"] = float(np.median(sample)) if len(sample) else np.nan
        if len(groups) >= 2:
            if any(len(sample) < minimum for sample in samples):
                row["status"] = f"Insufficient n: at least {minimum} animals per group required"
            else:
                seed = _seed(config, "regional", f"R{index}", *groups)
                if len(groups) == 2:
                    first, second = samples
                    p, method, status = _pair_p(second, first, "mannwhitney", seed, resamples)
                    u = float(stats.mannwhitneyu(second, first, method="asymptotic").statistic)
                    row.update(Mann_Whitney_U=u, Effect_size=2 * u / (len(first) * len(second)) - 1)
                else:
                    p, method, status = _omnibus_p(samples, "mannwhitney", seed, resamples)
                    row["effect_direction"] = "Omnibus comparison; no signed pairwise effect"
                row.update(p_value=float(p), method=method, status=status, Tested=bool(np.isfinite(p)))
        statistics.append(row)
    stat_frame = pd.DataFrame(statistics)
    finite = np.isfinite(stat_frame["p_value"])
    if finite.any():
        stat_frame.loc[finite, "q_value_BH_FDR"] = stats.false_discovery_control(
            stat_frame.loc[finite, "p_value"].to_numpy(float), method="bh")
        stat_frame.loc[finite, "Significant_q_alpha"] = stat_frame.loc[finite, "q_value_BH_FDR"] < alpha
    familiar_pair = len(groups) == 2 and set(groups) == {"Control", "Cancer"}
    summary = pd.DataFrame([
        {"Region": row["Region"], "Range, cm⁻¹": f"{row['Upper_cm1']:g}–{row['Lower_cm1']:g}",
         "Median control": row["Control_median_region_signal"],
         "Median cancer": row["Cancer_median_region_signal"],
         "p-MWU": row["p_value"] if familiar_pair else np.nan,
         "p-BH": row["q_value_BH_FDR"] if familiar_pair else np.nan,
         "r": row["Effect_size"] if familiar_pair else np.nan}
        for row in stat_frame.to_dict("records")], columns=SUMMARY_COLUMNS)
    return {"Region_Spectrum_Values": spectrum_frame, "Region_Animal_Values": animal_frame,
            "Region_Statistics": stat_frame, "Region_Summary": summary,
            "Region_QC": pd.DataFrame(quality, columns=QC_COLUMNS)}
