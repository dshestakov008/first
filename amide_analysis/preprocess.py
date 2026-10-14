"""Band-limited baseline correction and independent SG derivatives."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .config import load_config


def _linear_baseline(x, y, edge_points):
    if 2 * edge_points >= len(x):
        raise ValueError("baseline.edge_points: крайние участки перекрываются; уменьшите число точек")
    indices = np.r_[np.arange(edge_points), np.arange(len(x) - edge_points, len(x))]
    centered = x[indices] - np.mean(x[indices])
    slope = np.dot(centered, y[indices] - np.mean(y[indices])) / np.dot(centered, centered)
    return np.mean(y[indices]) + slope * (x - np.mean(x[indices]))


def _als_baseline(y, stiffness, asymmetry, iterations):
    size = len(y)
    difference = diags([np.ones(size - 2), -2 * np.ones(size - 2), np.ones(size - 2)],
                       [0, 1, 2], shape=(size - 2, size), format="csc")
    penalty = stiffness * (difference.T @ difference)
    weights = np.ones(size, dtype=float)
    for _ in range(iterations):
        baseline = spsolve(diags(weights, 0, shape=(size, size), format="csc") + penalty,
                          weights * y)
        weights = np.where(y > baseline, asymmetry, 1.0 - asymmetry)
    if not np.all(np.isfinite(baseline)):
        raise ValueError("ALS не вычислил конечную базовую линию")
    return baseline


def _sg_window(width, step, size, polyorder, max_fraction, warnings):
    minimum = polyorder + 1
    if minimum % 2 == 0:
        minimum += 1
    maximum = min(size, int(np.floor(size * max_fraction)))
    if maximum % 2 == 0:
        maximum -= 1
    if maximum < minimum:
        raise ValueError(
            f"Недостаточно точек для безопасного Savitzky–Golay: {size} в полосе; "
            f"polyorder={polyorder}, max_window_fraction={max_fraction}. "
            "Нужен более плотный исходный спектр; интерполяция не повышает приборное разрешение."
        )
    # Select the nearest odd count; physical window width is count * grid step.
    target = width / step
    lower = max(1, int(np.floor(target)))
    if lower % 2 == 0:
        lower -= 1
    candidates = (lower, lower + 2)
    count = min(candidates, key=lambda value: (abs(value - target), value))
    original = count
    count = min(max(count, minimum), maximum)
    if count != original:
        warnings.append(
            f"Окно SG {width:g} см⁻¹ ограничено до {count} точек "
            f"({count * step:g} см⁻¹) для polyorder и защиты от избыточного сглаживания."
        )
    return count


def preprocess_spectrum(x, y, config=None) -> dict:
    """Preprocess the complete covered band without extrapolating endpoints.

    Fitting uses ``corrected``, which is never SG-smoothed. Each second
    derivative is calculated independently from that same corrected signal.
    ``raw`` and ``baseline`` retain original units; ``normalization_factor``
    is the divisor applied to the corrected signal and derivative arrays.
    """
    config = load_config(config)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        raise ValueError("x и y: нужны одномерные массивы одинаковой длины, минимум три точки")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x и y должны содержать только конечные числа")
    if np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("Волновые числа должны быть положительными и строго возрастать")
    lower, upper = map(float, config["amide_bounds"])
    if x[0] > lower or x[-1] < upper:
        raise ValueError(
            f"Спектр {x[0]:g}–{x[-1]:g} см⁻¹ не покрывает всю полосу "
            f"{lower:g}–{upper:g} см⁻¹; экстраполяция запрещена"
        )
    inside = (x >= lower) & (x <= upper)
    # Include bracketing increments so fractional endpoints do not falsely
    # imply an unusually tiny grid step.
    left = max(0, int(np.searchsorted(x, lower, side="right")) - 1)
    right = min(len(x), int(np.searchsorted(x, upper, side="left")) + 1)
    band_steps = np.diff(x[left:right])
    if len(band_steps) < 2 or inside.sum() < 3:
        raise ValueError("В полосе Амид I недостаточно исходных точек")
    nominal_step = float(np.median(band_steps))
    irregularity = float(np.max(np.abs(band_steps - nominal_step)) / nominal_step)
    if band_steps.max() > 4.0 * nominal_step:
        raise ValueError("В полосе есть пропуск более четырёх шагов; интерполяция такого разрыва запрещена")
    warnings = []
    if irregularity > config["irregular_step_tolerance"]:
        warnings.append(
            f"Неравномерность шага {irregularity:.1%} превышает "
            f"irregular_step_tolerance={config['irregular_step_tolerance']:.1%}; "
            "использована регулярная сетка и линейная интерполяция."
        )
    interior_x = x[(x > lower) & (x < upper)]
    band_x = np.r_[lower, interior_x, upper]
    band_y = np.interp(band_x, x, y)
    candidate_steps = np.diff(band_x)
    regular = np.allclose(candidate_steps, candidate_steps[0], rtol=1e-7, atol=1e-10)
    if regular:
        grid = band_x
        raw = band_y
    else:
        intervals = max(2, int(np.ceil((upper - lower) / nominal_step - 1e-9)))
        grid = np.linspace(lower, upper, intervals + 1)
        raw = np.interp(grid, x, y)
        warnings.append(
            f"Сигнал линейно перенесён без экстраполяции на регулярную сетку "
            f"{len(grid)} точек с точными границами {lower:g} и {upper:g} см⁻¹. "
            "Эта сетка не повышает приборное разрешение."
        )
    if lower not in x or upper not in x:
        warnings.append("Сигнал на точных границах полосы получен интерполяцией между исходными точками.")
    step = float(grid[1] - grid[0])
    if nominal_step > config["peaks"]["min_fwhm_cm1"] / 2:
        warnings.append("Шаг исходных данных даёт менее двух интервалов на минимальную FWHM; узкие компоненты плохо разрешимы.")
    if not config["resolution_confirmed"]:
        warnings.append(
            "Приборное спектральное разрешение не подтверждено: "
            "spectral_resolution_cm1 — предварительный параметр, шаг экспорта не заменяет разрешение."
        )
    b = config["baseline"]
    if b["method"] == "linear":
        baseline = _linear_baseline(grid, raw, b["edge_points"])
    else:
        baseline = _als_baseline(raw, b["als_lambda"], b["als_p"], b["als_iterations"])
    corrected = raw - baseline
    factor = 1.0
    if config["normalization"] == "area":
        factor = float(trapezoid(corrected, grid))
        if not np.isfinite(factor) or factor <= np.finfo(float).eps * max(1, np.max(np.abs(raw))):
            raise ValueError("Нормализация по площади невозможна: площадь скорректированной полосы неположительна/нулевая")
    elif config["normalization"] == "l2":
        factor = float(np.sqrt(trapezoid(corrected**2, grid)))
        if not np.isfinite(factor) or factor <= np.finfo(float).eps * max(1, np.max(np.abs(raw))):
            raise ValueError("L2-нормализация невозможна: нулевая скорректированная полоса")
    corrected = corrected / factor
    smoothing = config["smoothing"]
    window = _sg_window(smoothing["window_cm1"], step, len(grid), smoothing["polyorder"],
                        smoothing["max_window_fraction"], warnings)
    smoothed = savgol_filter(corrected, window, smoothing["polyorder"], mode="interp")
    derivatives = {}
    for width in smoothing["derivative_windows_cm1"]:
        count = _sg_window(width, step, len(grid), smoothing["polyorder"],
                           smoothing["max_window_fraction"], warnings)
        if count in derivatives:
            warnings.append(f"Окно производной {width:g} см⁻¹ совпало на этой сетке с уже использованным окном {count} точек.")
            continue
        derivatives[count] = savgol_filter(corrected, count, smoothing["polyorder"],
                                           deriv=2, delta=step, mode="interp")
    if len(derivatives) < 2:
        warnings.append("На исходной сетке доступно менее двух независимых размеров окна производной; устойчивость кандидатов ограничена.")
    # Robust independent-noise estimate from second differences; smooth
    # curvature can inflate it, while a numerical floor avoids artificial inf SNR.
    differences = np.diff(corrected, n=2)
    median = np.median(differences)
    noise = float(np.median(np.abs(differences - median)) / (0.6744897501960817 * np.sqrt(6.0)))
    floor = np.finfo(float).eps * max(float(np.max(np.abs(corrected))), 1.0)
    noise = max(noise, floor)
    if np.ptp(corrected) <= floor * 10:
        warnings.append("Скорректированная полоса практически постоянна; количественное разложение недостоверно.")
    second = savgol_filter(corrected, window, smoothing["polyorder"],
                           deriv=2, delta=step, mode="interp")
    return {
        "x": grid, "raw": raw, "baseline": baseline, "corrected": corrected,
        "smoothed": smoothed, "second_derivative": second,
        "derivatives": derivatives, "warnings": list(dict.fromkeys(warnings)),
        "noise_sd": noise, "step": step, "source_step": nominal_step,
        "normalization_factor": factor,
    }
