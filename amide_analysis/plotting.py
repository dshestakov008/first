"""Publication plots; only accepted animal estimates enter group figures.

The component curves of a rejected fit remain useful diagnostic information,
but their relative areas are deliberately never printed as accepted estimates.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "amide-analysis-matplotlib"))

import matplotlib

# A caller can explicitly select another backend before importing this module.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


STRUCTURE_LABELS = {
    "random_coil": "Random coil",
    "alpha_helix": "α-helix",
    "beta_sheet": "β-sheet",
    "beta_turn": "β-turn",
    "unassigned": "Unassigned",
    "beta_associated": "β-associated structures\n(β-sheet + β-turn)",
}


def _options(config: dict) -> dict:
    return config.get("plots", {})


def _context(config: dict):
    options = _options(config)
    return plt.rc_context({
        "font.family": options.get("font_family", "DejaVu Sans"),
        "font.size": options.get("font_size", 10),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def _filename(stem: str) -> str:
    value = re.sub(r"[\\/\x00-\x1f<>:*?\"|]", "_", str(stem))
    return value.strip(" .") or "spectrum"


def _save(fig, output_dir: Path, stem: str, config: dict) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_label = config.get("dataset_label", "")
    if dataset_label:
        # Above the existing titles: preserved by bbox_inches="tight" in every
        # format, without covering data, residuals, or significance brackets.
        fig.text(.5, 1.015, str(dataset_label), ha="center", va="bottom",
                 fontsize=8, color="#666666")
    files = []
    try:
        for extension in ("png", "pdf", "svg"):
            path = output_dir / f"{_filename(stem)}.{extension}"
            fig.savefig(
                path, dpi=max(300, int(_options(config).get("dpi", 300))),
                bbox_inches="tight", facecolor="white",
            )
            files.append(path)
    finally:
        plt.close(fig)
    return files


def _array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _xy(data: dict, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = _array(data.get("x", [])), _array(data.get(y_key, []))
    if len(x) != len(y):
        return np.array([]), np.array([])
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _reverse_axis(axis, x):
    if len(x) > 1 and np.min(x) < np.max(x):
        axis.set_xlim(float(np.max(x)), float(np.min(x)))
    axis.set_xlabel("Wavenumber (cm⁻¹)")


def _note(axis, text: str):
    axis.text(.5, .5, text, transform=axis.transAxes, ha="center", va="center",
              wrap=True)


def _warning_text(fit: dict) -> str:
    warnings = fit.get("warnings", [])
    if isinstance(warnings, str):
        warnings = [warnings]
    # Full diagnostics belong to Fit_Quality; retain a readable figure caption.
    text = "; ".join(str(item) for item in warnings[:2])
    return text[:240] + ("…" if len(text) > 240 else "")


def plot_spectrum(raw: dict, processed: dict | None, fit: dict | None,
                  output_dir: Path, stem: str, config: dict) -> list[Path]:
    """Save the full measured spectrum and all available Amide-I diagnostics.

    All panels use decreasing wavenumber, as is conventional for FTIR. A failed
    preprocessing step still permits the original full spectrum to be exported.
    Gaussian curves use center/amplitude/FWHM, so rejected percentage estimates
    are neither needed nor silently replaced with zero.
    """
    files: list[Path] = []
    size = _options(config).get("figure_size", [9, 5])
    with _context(config):
        fig, axis = plt.subplots(figsize=size)
        x, y = _xy(raw, "y")
        if len(x):
            axis.plot(x, y, color="#253746", linewidth=1)
            _reverse_axis(axis, x)
        else:
            _note(axis, "No readable original spectrum")
        axis.set_ylabel("Absorbance")
        axis.set_title(f"{stem} — original full spectrum")
        files += _save(fig, output_dir, f"{stem}_full_spectrum", config)
        if processed is None:
            return files

        x = _array(processed.get("x", []))
        if len(x) < 2 or not np.all(np.isfinite(x)):
            return files
        fig, axes = plt.subplots(2, 1, figsize=(size[0], size[1] * 1.3), sharex=True)
        original = _array(processed.get("raw", []))
        baseline = _array(processed.get("baseline", []))
        corrected = _array(processed.get("corrected", []))
        smoothed = _array(processed.get("smoothed", []))
        adjustment = _array(fit.get("baseline_adjustment", [])) if fit is not None else np.array([])
        refined = (len(adjustment) == len(x) and np.all(np.isfinite(adjustment))
                   and bool(fit.get("components", []))) if fit is not None else False
        if not refined:
            adjustment = np.zeros_like(x)
        factor = float(processed.get("normalization_factor", 1))
        corrected_refined = corrected - adjustment if len(corrected) == len(x) else corrected
        if len(original) == len(x):
            axes[0].plot(x, original, label="Amide I before correction", color="#253746")
        if len(baseline) == len(x):
            axes[0].plot(x, baseline, "--", label="Initial estimated baseline" if refined else "Estimated baseline",
                         color="#d97732")
            if refined:
                axes[0].plot(x, baseline + adjustment * factor, ":",
                             label="Jointly refined linear baseline", color="#8655a1")
        if len(corrected) == len(x):
            axes[1].plot(x, corrected,
                         label="After initial baseline (fit input)" if refined else "Corrected signal (fit input)",
                         color="#253746", alpha=.6 if refined else 1)
            if refined:
                axes[1].plot(x, corrected_refined, label="After jointly refined baseline", color="#8655a1")
        if len(smoothed) == len(x):
            axes[1].plot(x, smoothed - adjustment,
                         label="SG-smoothed after refined baseline (visualisation)" if refined else "SG-smoothed (visualisation)",
                         color="#008577", alpha=.8)
        for axis in axes:
            _reverse_axis(axis, x)
            if axis.lines:
                axis.legend(frameon=False, fontsize="small")
        axes[0].set_ylabel("Absorbance")
        axes[1].set_ylabel("Absorbance" if factor == 1 else "Normalised signal (a.u.)")
        axes[0].set_title(f"{stem} — baseline correction")
        files += _save(fig, output_dir, f"{stem}_baseline", config)

        fig, axis = plt.subplots(figsize=size)
        derivatives = processed.get("derivatives", {})
        if not derivatives:
            derivatives = {"configured": processed.get("second_derivative", [])}
        drawn = False
        for window, derivative in derivatives.items():
            derivative = _array(derivative)
            if len(derivative) != len(x) or not np.all(np.isfinite(derivative)):
                continue
            line, = axis.plot(x, derivative, linewidth=1, label=f"SG window: {window} points")
            minima, _ = find_peaks(-derivative)
            axis.scatter(x[minima], derivative[minima], marker="o", s=18,
                         facecolors="none", edgecolors=line.get_color(), zorder=3)
            drawn = True
        if fit:
            for candidate in fit.get("candidates", []):
                center = candidate.get("center", candidate.get("position", np.nan))
                try:
                    if np.isfinite(float(center)):
                        axis.axvline(float(center), color="#666666", linestyle=":", alpha=.5)
                except (TypeError, ValueError):
                    continue
        if drawn:
            axis.axhline(0, color="gray", linewidth=.6)
            axis.legend(frameon=False, fontsize="small")
        else:
            _note(axis, "Second derivative unavailable")
        _reverse_axis(axis, x)
        axis.set_ylabel("d²(signal) / dν²")
        axis.set_title(f"{stem} — second derivative\nOpen markers: local minima; dotted lines: model candidates")
        files += _save(fig, output_dir, f"{stem}_second_derivative", config)

        if fit is not None:
            fig, axes = plt.subplots(2, 1, figsize=(size[0], size[1] * 1.4),
                                     sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            if len(corrected_refined) == len(x):
                axes[0].plot(x, corrected_refined, color="black", linewidth=1,
                             label="Signal after jointly refined baseline" if refined else "Corrected signal")
            reliable = fit.get("status") == "Reliable fit"
            curves = []
            for number, component in enumerate(fit.get("components", []), start=1):
                try:
                    center, amplitude, width = (float(component[key])
                                                for key in ("center", "amplitude", "fwhm"))
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.all(np.isfinite([center, amplitude, width])) or width <= 0 or amplitude < 0:
                    continue
                curve = amplitude * np.exp(-4 * np.log(2) * ((x - center) / width) ** 2)
                curves.append(curve)
                structure = STRUCTURE_LABELS.get(component.get("structure"), "Unassigned")
                label = f"{number}: {structure}, {center:.1f} cm⁻¹"
                percentage = component.get("area_pct", np.nan)
                try:
                    if reliable and np.isfinite(float(percentage)):
                        label += f", {float(percentage):.1f}%"
                except (ValueError, TypeError):
                    pass
                axes[0].plot(x, curve, linewidth=1, label=label)
            total = _array(fit.get("fit", []))
            if len(total) != len(x) or not np.all(np.isfinite(total)):
                total = np.sum(curves, axis=0) if curves else np.array([])
            elif refined:
                total = total - adjustment
            if len(total) == len(x):
                axes[0].plot(x, total, "--", color="#d83b3b", linewidth=1.6, label="Gaussian sum")
                residuals = _array(fit.get("residuals", []))
                if len(residuals) != len(x) and len(corrected_refined) == len(x):
                    residuals = corrected_refined - total
                if len(residuals) == len(x):
                    axes[1].plot(x, residuals, color="#253746", linewidth=1)
            else:
                _note(axes[0], "No usable Gaussian model")
                _note(axes[1], "Residuals unavailable")
            axes[1].axhline(0, color="gray", linewidth=.7)
            axes[1].set_ylabel("Residual")
            axes[0].set_ylabel("Signal after refined baseline (a.u.)" if refined else "Corrected signal (a.u.)")
            for axis in axes:
                _reverse_axis(axis, x)
            if axes[0].lines:
                axes[0].legend(frameon=False, fontsize="small", loc="best")
            label = "Reliable fit" if reliable else "Unreliable fit — diagnostic parameters only"
            baseline_caption = "\nComponents and signal relative to jointly refined baseline" if refined else ""
            axes[0].set_title(f"{stem} — {label}{baseline_caption}")
            warning_text = _warning_text(fit)
            if warning_text:
                fig.text(.5, .01, warning_text, ha="center", fontsize="small", wrap=True)
                fig.subplots_adjust(bottom=.14)
            files += _save(fig, output_dir, f"{stem}_decomposition", config)
    return files


def _accepted(animals: pd.DataFrame) -> pd.DataFrame:
    if "status" not in animals or "group" not in animals:
        return pd.DataFrame(columns=animals.columns)
    return animals.loc[animals["status"].eq("Reliable fit")].copy()


def _metric_labels(config: dict) -> tuple[list[str], dict[str, str]]:
    metrics = ["random_coil", "alpha_helix", "beta_sheet", "beta_turn"]
    if _options(config).get("merge_beta_turn", False):
        metrics = ["random_coil", "alpha_helix", "beta_associated"]
    labels = dict(STRUCTURE_LABELS)
    labels.update(_options(config).get("metric_labels", {}))
    # The merged category must remain explicit, even with a custom label.
    if "beta_associated" in metrics:
        labels["beta_associated"] = STRUCTURE_LABELS["beta_associated"]
    return metrics, labels


def _group_options(animals: pd.DataFrame, config: dict):
    values = animals.get("group", pd.Series(dtype=str)).dropna().astype(str)
    groups = sorted(values.unique().tolist())
    control = str(_options(config).get("control_group", "Control"))
    if control in groups:
        groups.remove(control)
        groups.insert(0, control)
    palette = plt.get_cmap("tab10")
    colors = {group: _options(config).get("colors", {}).get(group, palette(i % 10))
              for i, group in enumerate(groups)}
    return groups, control, colors


def _stars(q: float) -> str:
    return "****" if q < .0001 else "***" if q < .001 else "**" if q < .01 else "*"


def _annotations(axis, tests: pd.DataFrame, metrics: list[str], positions: dict,
                 top: float, config: dict) -> float:
    if not _options(config).get("significance_annotations", True) or tests.empty:
        return top
    required = {"metric", "group1", "group2", "q_value"}
    if not required.issubset(tests.columns):
        return top
    alpha = config.get("statistics", {}).get("alpha", .05)
    height = max(2.5, top * .04)
    highest = top
    for metric in metrics:
        level = 0
        for _, row in tests.loc[tests["metric"].eq(metric)].iterrows():
            allowed = row.get("interpretation_allowed", True)
            if pd.isna(allowed) or not bool(allowed):
                continue
            try:
                q = float(row["q_value"])
            except (ValueError, TypeError):
                continue
            group1, group2 = str(row["group1"]), str(row["group2"])
            if (not np.isfinite(q) or q >= alpha or q < 0 or group1 == group2
                    or (metric, group1) not in positions or (metric, group2) not in positions):
                continue
            x1, x2 = positions[(metric, group1)], positions[(metric, group2)]
            y = top + height * (1 + 2 * level)
            axis.plot([x1, x1, x2, x2], [y, y + height * .3, y + height * .3, y],
                      color="black", linewidth=.8)
            axis.text((x1 + x2) / 2, y + height * .4, _stars(q), ha="center", va="bottom")
            highest = max(highest, y + height * 1.8)
            level += 1
    return highest


def plot_groups(animals: pd.DataFrame, tests: pd.DataFrame, output_dir: Path,
                config: dict) -> list[Path]:
    """Mean ± sample SD and individual animals, plus group distributions.

    A group with no accepted animals keeps its label but has no zero-valued bar.
    SD for a singleton is undefined: draw the animal/mean without an error bar.
    Significance uses the existing BH-adjusted q-values, never a second test.
    """
    files: list[Path] = []
    metrics, labels = _metric_labels(config)
    accepted = _accepted(animals)
    if "beta_associated" in metrics and "beta_associated" not in accepted:
        if {"beta_sheet", "beta_turn"}.issubset(accepted):
            accepted["beta_associated"] = (pd.to_numeric(accepted["beta_sheet"], errors="coerce")
                                           + pd.to_numeric(accepted["beta_turn"], errors="coerce"))
    # Keep labels for excluded-only groups, so exclusions cannot disappear.
    groups, control, colors = _group_options(animals, config)
    options = _options(config)
    size = options.get("figure_size", [9, 5])
    group_labels = options.get("group_labels", {})
    with _context(config):
        fig, axis = plt.subplots(figsize=(max(float(size[0]), len(groups) * 1.1), size[1]))
        positions: dict[tuple[str, str], float] = {}
        maxima = []
        width = min(float(options.get("bar_width", .8)) / max(len(groups), 1), .8)
        rng = np.random.default_rng(config.get("seed", 20261014))
        legend_groups = set()
        for index, metric in enumerate(metrics):
            for group_index, group in enumerate(groups):
                center = index + (group_index - (len(groups) - 1) / 2) * width
                values = (pd.to_numeric(accepted.loc[accepted["group"].astype(str).eq(group), metric],
                                        errors="coerce").dropna().to_numpy()
                          if metric in accepted else np.array([]))
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                positions[(metric, group)] = center
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
                axis.bar(center, mean, width=width * .88,
                         facecolor="none" if group == control else colors[group],
                         edgecolor=colors[group], linewidth=1.2,
                         label=group_labels.get(group, group) if group not in legend_groups else None)
                legend_groups.add(group)
                if np.isfinite(sd):
                    axis.errorbar(center, mean, yerr=sd, color=colors[group],
                                  capsize=3, linewidth=1, fmt="none", zorder=4)
                jitter = rng.uniform(-width * .22, width * .22, len(values))
                axis.scatter(center + jitter, values, s=options.get("point_size", 25),
                             facecolors="white" if group == control else colors[group],
                             edgecolors="black", linewidths=.6, zorder=5)
                maxima.append(max(float(values.max()), mean + (sd if np.isfinite(sd) else 0)))
        if not maxima:
            _note(axis, "No accepted animal-level quantitative estimates\nSee Fit_Quality and Unreliable_Spectra")
        else:
            top = max(maxima + [1])
            upper = _annotations(axis, tests, metrics, positions, top, config)
            axis.set_ylim(0, max(upper * 1.06, top * 1.12))
        if groups:
            handles = []
            for group in groups:
                n = int(accepted.get("group", pd.Series(dtype=str)).astype(str).eq(group).sum())
                handles.append(Patch(facecolor="none" if group == control else colors[group],
                                     edgecolor=colors[group],
                                     label=f"{group_labels.get(group, group)} (n={n})"))
            axis.legend(handles=handles, frameon=False, fontsize="small", loc="upper left",
                        bbox_to_anchor=(1, 1))
        axis.set_xticks(range(len(metrics)), [labels[metric] for metric in metrics])
        axis.set_ylabel("Relative spectral component area (%)")
        axis.set_title("Amide I — accepted animals; mean ± SD")
        fig.text(.5, .01, "Each point is one animal; * denotes eligible BH-adjusted q < α. Exact q-values: Statistical_Tests.\n"
                 "The denominator includes Unassigned components; the displayed categories may sum to less than 100%.",
                 ha="center", fontsize="small")
        fig.subplots_adjust(bottom=.2)
        files += _save(fig, output_dir, "Secondary_Structure_Groups", config)

        fig, axes = plt.subplots(1, len(metrics), figsize=(max(float(size[0]), len(metrics) * 3), size[1]),
                                 sharey=True, squeeze=False)
        rng = np.random.default_rng(config.get("seed", 20261014))
        for metric, axis in zip(metrics, axes[0]):
            any_values = False
            for index, group in enumerate(groups):
                values = (pd.to_numeric(accepted.loc[accepted["group"].astype(str).eq(group), metric],
                                        errors="coerce").dropna().to_numpy()
                          if metric in accepted else np.array([]))
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                any_values = True
                # A box needs >=2 observations; no pretend distribution for n=1.
                if len(values) >= 2:
                    box = axis.boxplot([values], positions=[index], widths=.5, showfliers=False,
                                       patch_artist=True, manage_ticks=False)
                    box["boxes"][0].set(facecolor="none" if group == control else colors[group],
                                        edgecolor=colors[group], alpha=.4)
                    box["medians"][0].set(color="black")
                axis.scatter(index + rng.uniform(-.12, .12, len(values)), values,
                             s=options.get("point_size", 25),
                             facecolors="white" if group == control else colors[group],
                             edgecolors="black", linewidths=.6, zorder=5)
            axis.set_xticks(range(len(groups)), [group_labels.get(group, group) for group in groups],
                            rotation=60, ha="right")
            axis.set_title(labels[metric])
            if not any_values:
                _note(axis, "No accepted estimates")
            axis.set_ylim(bottom=0)
        axes[0][0].set_ylabel("Relative spectral component area (%)")
        fig.suptitle("Animal-level distributions — boxes show median and IQR")
        fig.subplots_adjust(bottom=.28, wspace=.15)
        files += _save(fig, output_dir, "Secondary_Structure_Distributions", config)
    return files
