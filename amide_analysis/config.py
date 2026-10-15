"""Validated, reproducible configuration for Amide I analysis."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from collections.abc import Mapping


_DEFAULTS = {
    "dataset_label": "",
    "amide_bounds": [1600.0, 1700.0],
    "spectral_resolution_cm1": 4.0,
    "resolution_confirmed": False,
    "irregular_step_tolerance": 0.05,
    "baseline": {
        "method": "linear", "edge_points": 3,
        "als_lambda": 100000.0, "als_p": 0.001, "als_iterations": 15,
    },
    "smoothing": {
        "window_cm1": 9.0, "polyorder": 3,
        "derivative_windows_cm1": [9.0, 13.0, 17.0],
        "max_window_fraction": 0.2,
    },
    "normalization": "none",
    "peaks": {
        "min_fwhm_cm1": 6.0, "max_fwhm_cm1": 80.0,
        "max_components": 6, "min_separation_cm1": 6.0,
        "min_derivative_support": 0.66,
        "derivative_prominence_noise_factor": 2.5,
    },
    "fitting": {
        "criterion": "aicc", "multistarts": 6, "max_nfev": 1500,
        "bootstrap_iterations": 30, "bootstrap_block_length": 3,
        "bootstrap_retry_multistarts": 3,
        "bootstrap_success_fraction": 0.8,
        "min_fit_snr": 10.0, "max_nrmse": 0.12,
        "max_residual_lag1": 0.8, "max_condition_number": 1e10,
        "min_component_area_pct": 1.0, "max_percentage_sd": 8.0,
        "max_preprocessing_delta_pct": 10.0, "min_model_delta": 6.0,
    },
    "assignment_ranges": [
        {"structure": "beta_sheet", "lower": 1620.0, "upper": 1640.0},
        {"structure": "random_coil", "lower": 1640.0, "upper": 1650.0},
        {"structure": "alpha_helix", "lower": 1650.0, "upper": 1660.0},
        {"structure": "beta_turn", "lower": 1660.0, "upper": 1680.0},
        {"structure": "beta_sheet", "lower": 1680.0, "upper": 1695.0},
    ],
    "assignment_boundary_tolerance_cm1": 1.0,
    "replicates": {"max_repeat_delta_pct": 10.0},
    "statistics": {
        "method": "mannwhitney", "bootstrap_iterations": 2000,
        "min_n_ci": 3, "min_n_test": 2, "alpha": 0.05,
        "posthoc_only_if_omnibus_significant": True,
        "include_log_ratios": True, "median_ci_bootstrap": False,
    },
    "plots": {
        "merge_beta_turn": False, "control_group": "Control", "colors": {},
        "metric_labels": {}, "group_labels": {}, "point_size": 24.0, "bar_width": 0.8,
        "font_family": "DejaVu Sans", "font_size": 10,
        "dpi": 300, "figure_size": [9, 5], "significance_annotations": True,
    },
    "seed": 20261014,
}


def default_config() -> dict:
    """Return a fresh configuration; never share nested mutable defaults."""
    return deepcopy(_DEFAULTS)


def _merge(base: dict, update: Mapping, prefix: str = "") -> dict:
    if not isinstance(update, Mapping):
        raise ValueError(f"{prefix or 'Конфигурация'}: требуется объект JSON")
    for key, value in update.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if key not in base:
            raise ValueError(f"Неизвестный параметр конфигурации: {dotted}")
        if dotted in {"plots.colors", "plots.metric_labels", "plots.group_labels"}:
            if not isinstance(value, Mapping):
                raise ValueError(f"{dotted}: требуется объект JSON")
            base[key].update(deepcopy(dict(value)))
        elif isinstance(base[key], dict):
            _merge(base[key], value, dotted)
        else:
            base[key] = deepcopy(value)
    return base


def _number(value, name, minimum=None, maximum=None, *, integer=False,
            open_min=False, open_max=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: требуется {'целое число' if integer else 'число'}")
    if not math.isfinite(value) or (integer and not isinstance(value, int)):
        raise ValueError(f"{name}: требуется конечное {'целое ' if integer else ''}число")
    if minimum is not None and (value <= minimum if open_min else value < minimum):
        raise ValueError(f"{name}: должно быть {'>' if open_min else '>='} {minimum}")
    if maximum is not None and (value >= maximum if open_max else value > maximum):
        raise ValueError(f"{name}: должно быть {'<' if open_max else '<='} {maximum}")


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name}: требуется true или false")


def _choice(value, name, choices):
    if value not in choices:
        raise ValueError(f"{name}: допустимы {', '.join(choices)}")


def validate_config(config: Mapping) -> dict:
    """Deep-merge a partial mapping, reject unknown/unsafe settings, return a copy."""
    c = _merge(default_config(), config)
    if not isinstance(c["dataset_label"], str):
        raise ValueError("dataset_label: требуется строка (пустая строка допустима)")
    bounds = c["amide_bounds"]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("amide_bounds: требуется [нижняя граница, верхняя граница]")
    for n, value in enumerate(bounds):
        _number(value, f"amide_bounds[{n}]", 0, open_min=True)
    if bounds[0] >= bounds[1]:
        raise ValueError("amide_bounds: нижняя граница должна быть меньше верхней")
    band_width = bounds[1] - bounds[0]
    _number(c["spectral_resolution_cm1"], "spectral_resolution_cm1", 0, open_min=True)
    _boolean(c["resolution_confirmed"], "resolution_confirmed")
    _number(c["irregular_step_tolerance"], "irregular_step_tolerance", 0, 0.5)
    b = c["baseline"]
    _choice(b["method"], "baseline.method", ("linear", "als"))
    _number(b["edge_points"], "baseline.edge_points", 1, integer=True)
    _number(b["als_lambda"], "baseline.als_lambda", 0, open_min=True)
    _number(b["als_p"], "baseline.als_p", 0, 0.5, open_min=True, open_max=True)
    _number(b["als_iterations"], "baseline.als_iterations", 1, 1000, integer=True)
    s = c["smoothing"]
    _number(s["polyorder"], "smoothing.polyorder", 2, 7, integer=True)
    _number(s["max_window_fraction"], "smoothing.max_window_fraction", 0, 0.5, open_min=True)
    _number(s["window_cm1"], "smoothing.window_cm1", 0, open_min=True)
    windows = s["derivative_windows_cm1"]
    if not isinstance(windows, list) or len(windows) < 2:
        raise ValueError("smoothing.derivative_windows_cm1: нужны хотя бы два окна")
    for n, window in enumerate(windows):
        _number(window, f"smoothing.derivative_windows_cm1[{n}]", 0, open_min=True)
    if len(set(windows)) != len(windows):
        raise ValueError("smoothing.derivative_windows_cm1: окна должны отличаться")
    if max(windows + [s["window_cm1"]]) > band_width * s["max_window_fraction"]:
        raise ValueError("Окно Savitzky–Golay превышает smoothing.max_window_fraction от полосы")
    _choice(c["normalization"], "normalization", ("none", "area", "l2"))
    p = c["peaks"]
    for key in ("min_fwhm_cm1", "max_fwhm_cm1", "min_separation_cm1",
                "derivative_prominence_noise_factor"):
        _number(p[key], f"peaks.{key}", 0, open_min=True)
    if p["min_fwhm_cm1"] >= p["max_fwhm_cm1"] or p["max_fwhm_cm1"] > band_width:
        raise ValueError("peaks: нужно 0 < min_fwhm_cm1 < max_fwhm_cm1 <= ширины полосы")
    if p["min_separation_cm1"] >= band_width:
        raise ValueError("peaks.min_separation_cm1 должно быть меньше ширины полосы")
    _number(p["max_components"], "peaks.max_components", 1, 12, integer=True)
    _number(p["min_derivative_support"], "peaks.min_derivative_support", 0, 1, open_min=True)
    f = c["fitting"]
    _choice(f["criterion"], "fitting.criterion", ("aicc", "bic"))
    for key, lower in (("multistarts", 1), ("max_nfev", 1), ("bootstrap_iterations", 2),
                       ("bootstrap_retry_multistarts", 1)):
        _number(f[key], f"fitting.{key}", lower, integer=True)
    _number(f["bootstrap_block_length"], "fitting.bootstrap_block_length", 1, 50, integer=True)
    _number(f["bootstrap_success_fraction"], "fitting.bootstrap_success_fraction", 0, 1, open_min=True)
    _number(f["max_residual_lag1"], "fitting.max_residual_lag1", 0, 1, open_max=True)
    for key in ("min_fit_snr", "max_nrmse", "max_condition_number"):
        _number(f[key], f"fitting.{key}", 0, open_min=True)
    for key in ("min_component_area_pct", "max_percentage_sd", "max_preprocessing_delta_pct"):
        _number(f[key], f"fitting.{key}", 0, 100, open_min=True)
    _number(f["min_model_delta"], "fitting.min_model_delta", 0)
    _number(c["assignment_boundary_tolerance_cm1"], "assignment_boundary_tolerance_cm1", 0)
    ranges = c["assignment_ranges"]
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("assignment_ranges: нужен непустой список диапазонов")
    checked = []
    for n, item in enumerate(ranges):
        if not isinstance(item, Mapping) or set(item) != {"structure", "lower", "upper"}:
            raise ValueError(f"assignment_ranges[{n}]: нужны только structure, lower, upper")
        _choice(item["structure"], f"assignment_ranges[{n}].structure",
                ("random_coil", "alpha_helix", "beta_sheet", "beta_turn"))
        for key in ("lower", "upper"):
            _number(item[key], f"assignment_ranges[{n}].{key}", bounds[0], bounds[1])
        if item["lower"] >= item["upper"]:
            raise ValueError(f"assignment_ranges[{n}]: нижняя граница не меньше верхней")
        if c["assignment_boundary_tolerance_cm1"] * 2 >= item["upper"] - item["lower"]:
            raise ValueError("assignment_boundary_tolerance_cm1 слишком велик для диапазона")
        checked.append((item["lower"], item["upper"]))
    checked.sort()
    if any(second[0] < first[1] for first, second in zip(checked, checked[1:])):
        raise ValueError("assignment_ranges: диапазоны не должны перекрываться внутри")
    _number(c["replicates"]["max_repeat_delta_pct"], "replicates.max_repeat_delta_pct", 0, 100)
    st = c["statistics"]
    _choice(st["method"], "statistics.method", ("mannwhitney", "welch"))
    _number(st["bootstrap_iterations"], "statistics.bootstrap_iterations", 2, integer=True)
    for key in ("min_n_ci", "min_n_test"):
        _number(st[key], f"statistics.{key}", 2, integer=True)
    _number(st["alpha"], "statistics.alpha", 0, 1, open_min=True, open_max=True)
    for key in ("posthoc_only_if_omnibus_significant", "include_log_ratios", "median_ci_bootstrap"):
        _boolean(st[key], f"statistics.{key}")
    plots = c["plots"]
    for key in ("merge_beta_turn", "significance_annotations"):
        _boolean(plots[key], f"plots.{key}")
    for key in ("control_group", "font_family"):
        if not isinstance(plots[key], str) or not plots[key].strip():
            raise ValueError(f"plots.{key}: требуется непустая строка")
    _number(plots["font_size"], "plots.font_size", 1, 100)
    _number(plots["point_size"], "plots.point_size", 0, open_min=True)
    _number(plots["bar_width"], "plots.bar_width", 0, 1, open_min=True)
    _number(plots["dpi"], "plots.dpi", 50, 1200, integer=True)
    if not isinstance(plots["figure_size"], list) or len(plots["figure_size"]) != 2:
        raise ValueError("plots.figure_size: нужны ширина и высота")
    for n, value in enumerate(plots["figure_size"]):
        _number(value, f"plots.figure_size[{n}]", 0, 50, open_min=True)
    from matplotlib.colors import is_color_like
    for group, color in plots["colors"].items():
        if not isinstance(group, str) or not group.strip() or not isinstance(color, str) or not is_color_like(color):
            raise ValueError(f"plots.colors: недопустимый цвет для группы {group!r}")
    for key in ("metric_labels", "group_labels"):
        for name, label in plots[key].items():
            if not isinstance(name, str) or not name.strip() or not isinstance(label, str) or not label.strip():
                raise ValueError(f"plots.{key}: названия и подписи должны быть непустыми строками")
    _number(c["seed"], "seed", 0, 2**32 - 1, integer=True)
    return c


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Повторяющийся ключ JSON: {key}")
        result[key] = value
    return result


def load_config(path=None) -> dict:
    """Load JSON or a mapping; omitted fields retain defaults, unknown keys fail."""
    if path is None:
        return validate_config({})
    if isinstance(path, Mapping):
        return validate_config(path)
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
        update = json.loads(text, object_pairs_hook=_json_pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                ValueError(f"Недопустимое число JSON: {value}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать конфигурацию {path}: {exc}") from exc
    return validate_config(update)
