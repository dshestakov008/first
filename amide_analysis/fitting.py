"""Derivative-guided, bounded Gaussian fits with explicit rejection diagnostics."""

from copy import deepcopy
from itertools import combinations
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_coeffs
from scipy.special import erf


STRUCTURES = ("random_coil", "alpha_helix", "beta_sheet", "beta_turn", "unassigned")
_GAUSSIAN = 4.0 * math.log(2.0)


def gaussian(x, amplitude, center, fwhm):
    return amplitude * np.exp(-_GAUSSIAN * ((np.asarray(x) - center) / fwhm) ** 2)


def component_area(amplitude, center, fwhm, lower, upper):
    """Gaussian integral inside the analyzed band, without extrapolated tails."""
    sigma = fwhm / math.sqrt(8.0 * math.log(2.0))
    return float(amplitude * sigma * math.sqrt(math.pi / 2.0)
                 * (erf((upper - center) / (math.sqrt(2.0) * sigma))
                    - erf((lower - center) / (math.sqrt(2.0) * sigma))))


def detect_candidates(processed, config):
    """Cluster reproducible negative derivative minima; do not equate noise with peaks."""
    x, step = processed["x"], float(processed["step"])
    pcfg = config["peaks"]
    separation = max(pcfg["min_separation_cm1"], config["spectral_resolution_cm1"], 2 * step)
    detections = []
    derivatives = processed["derivatives"]
    for window, derivative in derivatives.items():
        window = int(window)
        derivative = np.asarray(derivative)
        coeff = savgol_coeffs(window, config["smoothing"]["polyorder"], deriv=2, delta=step)
        derivative_noise = float(processed["noise_sd"]) * np.linalg.norm(coeff)
        prominence = max(pcfg["derivative_prominence_noise_factor"] * derivative_noise,
                         np.max(np.abs(derivative)) * 0.03, np.finfo(float).eps)
        indices, info = find_peaks(-derivative, prominence=prominence,
                                  distance=max(1, int(math.ceil(separation / step))))
        margin = max(config["spectral_resolution_cm1"], step * window / 2)
        for index, prom in zip(indices, info["prominences"]):
            if (derivative[index] < -derivative_noise
                    and x[0] + margin <= x[index] <= x[-1] - margin):
                detections.append({"center": float(x[index]), "window": window,
                                   "prominence": float(prom)})
    clusters = []
    # Cluster strongest detections first; one smoothing window gives one vote.
    for item in sorted(detections, key=lambda item: -item["prominence"]):
        eligible = [(abs(np.mean([v["center"] for v in cluster]) - item["center"]), cluster)
                    for cluster in clusters
                    if item["window"] not in [v["window"] for v in cluster]]
        nearest = min(eligible, key=lambda pair: pair[0]) if eligible else None
        if nearest is not None and nearest[0] <= separation / 2:
            nearest[1].append(item)
        else:
            clusters.append([item])
    rows = []
    for cluster in clusters:
        centers = np.array([item["center"] for item in cluster])
        support = len({item["window"] for item in cluster}) / len(derivatives)
        center = float(np.average(centers, weights=[item["prominence"] for item in cluster]))
        rows.append({"center": center, "support": support,
            "center_spread_cm1": float(np.ptp(centers)),
            "windows": ",".join(str(item["window"]) for item in cluster),
            "prominence": float(np.mean([item["prominence"] for item in cluster])),
            "stable": bool(support >= pcfg["min_derivative_support"]
                           and np.ptp(centers) <= separation)})
    # Nearby clusters may arise from alternate minima in the same window. Retain
    # the stronger candidate so a numerical duplicate cannot add a component.
    retained = []
    for row in sorted(rows, key=lambda row: (-row["support"], -row["prominence"])):
        if all(abs(row["center"] - previous["center"]) >= separation for previous in retained):
            retained.append(row)
    return sorted(retained, key=lambda row: row["center"])


def _model(x, params):
    return np.sum([gaussian(x, *row) for row in np.asarray(params).reshape(-1, 3)], axis=0)


def _jacobian(x, params):
    columns = []
    for amplitude, center, width in np.asarray(params).reshape(-1, 3):
        dx = x - center
        shape = np.exp(-_GAUSSIAN * (dx / width) ** 2)
        columns.extend((shape, amplitude * shape * 2 * _GAUSSIAN * dx / width ** 2,
                        amplitude * shape * 2 * _GAUSSIAN * dx ** 2 / width ** 3))
    return np.column_stack(columns)


def _bounded_fit(x, y, centers, config, rng, starts=None, initial=None):
    scale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    target = y / scale
    centers = np.sort(np.asarray(centers, float))
    count = len(centers)
    min_width = max(config["peaks"]["min_fwhm_cm1"], config["spectral_resolution_cm1"],
                    2 * float(np.median(np.diff(x))))
    max_width = config["peaks"]["max_fwhm_cm1"]
    if min_width >= max_width:
        return None
    radius = max(config["spectral_resolution_cm1"], config["peaks"]["min_separation_cm1"])
    lower, upper = [], []
    for index, center in enumerate(centers):
        lo = max(x[0], center - radius,
                 (centers[index - 1] + center) / 2 if index else x[0])
        hi = min(x[-1], center + radius,
                 (centers[index + 1] + center) / 2 if index + 1 < count else x[-1])
        lower.extend((0.0, lo, min_width))
        upper.extend((3.0, hi, max_width))
    lower, upper = np.array(lower), np.array(upper)
    best = None
    starts = config["fitting"]["multistarts"] if starts is None else starts
    for start in range(starts):
        if start == 0 and initial is not None:
            guess = np.array(initial, float).copy().reshape(-1, 3)
            guess[:, 0] /= scale
            guess = guess.ravel()
        else:
            widths = np.full(count, min(max_width * 0.45, max(min_width * 1.7, 12.0)))
            candidate_centers = centers.copy()
            if start:
                widths *= rng.uniform(0.65, 1.4, count)
                candidate_centers += rng.normal(0, radius / 4, count)
            amplitudes = np.maximum(np.interp(candidate_centers, x, target), 0.05)
            guess = np.column_stack((amplitudes, candidate_centers, widths)).ravel()
        guess = np.clip(guess, lower + 1e-7, upper - 1e-7)
        try:
            result = least_squares(lambda p: _model(x, p) - target, guess,
                jac=lambda p: _jacobian(x, p), bounds=(lower, upper), x_scale="jac",
                max_nfev=config["fitting"]["max_nfev"], ftol=1e-9, xtol=1e-9, gtol=1e-9)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if not result.success or not np.isfinite(result.x).all():
            continue
        sse = float(np.sum(result.fun ** 2) * scale ** 2)
        if best is None or sse < best["sse"]:
            params = result.x.reshape(-1, 3).copy()
            params[:, 0] *= scale
            # Count the Gaussian parameters and the fitted residual variance.
            k, n = 3 * count + 1, len(x)
            likelihood = n * math.log(max(sse / n, np.finfo(float).tiny))
            aic = likelihood + 2 * k
            aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
            condition = float(np.linalg.cond(result.jac.T @ result.jac))
            bound_hits = np.any((result.x - lower < 1e-3 * (upper - lower))
                                | (upper - result.x < 1e-3 * (upper - lower)))
            best = {"params": params, "fit": _model(x, params), "sse": sse,
                "aicc": float(aicc), "bic": float(likelihood + k * math.log(n)),
                "count": count, "condition_number": condition, "bound_hits": bool(bound_hits),
                "initial_centers": centers, "nfev": int(result.nfev)}
    return best


def _select_model(processed, config, rng):
    candidates = detect_candidates(processed, config)
    stable = [row for row in candidates if row["stable"]]
    if not stable:
        return None, [], candidates
    stable = sorted(stable, key=lambda row: -row["support"] * row["prominence"])
    stable = stable[:config["peaks"]["max_components"]]
    fits = []
    for count in range(1, len(stable) + 1):
        # Evaluate every candidate subset: <= 63 for the default six candidates.
        subsets = sorted(combinations(stable, count),
                         key=lambda subset: -sum(v["support"] * v["prominence"] for v in subset))
        for subset in subsets:
            model = _bounded_fit(processed["x"], processed["corrected"],
                                 [row["center"] for row in subset], config, rng)
            if model is not None and np.isfinite(model[config["fitting"]["criterion"]]):
                fits.append(model)
    if not fits:
        return None, [], candidates
    fits.sort(key=lambda fit: fit[config["fitting"]["criterion"]])
    return fits[0], fits, candidates


def _areas(model, bounds):
    return np.array([component_area(*params, *bounds) for params in model["params"]])


def _classify(center, low, high, config):
    tolerance = config["assignment_boundary_tolerance_cm1"]
    low, high = min(low, center - tolerance), max(high, center + tolerance)
    possible = {band["structure"] for band in config["assignment_ranges"]
                if band["lower"] <= low and high <= band["upper"]}
    return possible.pop() if len(possible) == 1 else "unassigned"


def _percentages(model, config, intervals=None):
    areas = _areas(model, config["amide_bounds"])
    total = float(areas.sum())
    relative = 100 * areas / total if total > 0 else np.full(len(areas), np.nan)
    categories = []
    for index, params in enumerate(model["params"]):
        center = params[1]
        low, high = intervals[index] if intervals is not None else (center, center)
        categories.append(_classify(center, low, high, config))
    percentages = {name: float(np.sum(relative[np.array(categories) == name])) for name in STRUCTURES}
    return areas, relative, categories, percentages


def _bootstrap(processed, best, other_models, config, rng):
    iterations = config["fitting"]["bootstrap_iterations"]
    residual = processed["corrected"] - best["fit"]
    residual = residual - residual.mean()
    parameters, percentages, structure_samples = [], [], []
    retained_count = 0
    assessed_count = 0
    # Circular residual blocks preserve short correlations caused by spectral
    # sampling/interpolation. Length 1 explicitly requests an iid bootstrap.
    block_length = min(config["fitting"].get("bootstrap_block_length", 3), max(1, len(residual) // 4))
    for _ in range(iterations):
        starts = rng.integers(0, len(residual), size=int(math.ceil(len(residual) / block_length)))
        indices = (starts[:, None] + np.arange(block_length)) % len(residual)
        simulated = best["fit"] + residual[indices.ravel()[:len(residual)]]
        model = _bounded_fit(processed["x"], simulated, best["initial_centers"], config, rng,
                             starts=1, initial=best["params"])
        if model is None or model["bound_hits"]:
            continue
        parameters.append(model["params"])
        _, relative, _, structure = _percentages(model, config)
        percentages.append(relative)
        structure_samples.append([structure[name] for name in STRUCTURES])
        comparisons = [_bounded_fit(processed["x"], simulated, alternative["initial_centers"],
            config, rng, starts=1, initial=alternative["params"]) for alternative in other_models]
        # An unsuccessful competing fit is indeterminate, not a vote for K.
        if any(alternative is None for alternative in comparisons):
            continue
        assessed_count += 1
        if all(alternative[config["fitting"]["criterion"]] >= model[config["fitting"]["criterion"]]
               for alternative in comparisons):
            retained_count += 1
    count = len(parameters)
    return {"parameters": np.asarray(parameters), "percentages": np.asarray(percentages),
            "structure_samples": np.asarray(structure_samples), "success_count": count,
            "success_fraction": count / iterations if iterations else 0.0,
            "count_support": retained_count / iterations if iterations else 0.0,
            "count_comparison_fraction": assessed_count / iterations if iterations else 0.0,
            "block_length": block_length}


def analyze_band(processed, config, seed=None):
    """Return accepted estimates only if all configured reliability checks pass."""
    rng = np.random.default_rng(config["seed"] if seed is None else seed)
    config = deepcopy(config)
    config["peaks"]["min_fwhm_cm1"] = max(config["peaks"]["min_fwhm_cm1"],
        2 * processed.get("source_step", processed["step"]))
    x, y = processed["x"], processed["corrected"]
    empty = {"status": "Unreliable fit", "warnings": [], "components": [], "candidates": [],
        "model_comparison": [], "fit": np.full(len(x), np.nan), "residuals": np.full(len(x), np.nan),
        "metrics": {}, "percentages": {name: np.nan for name in STRUCTURES}, "sensitivity": []}
    noise = max(float(processed["noise_sd"]), np.finfo(float).eps)
    snr = float(max(np.max(y), 0) / noise)
    empty["metrics"]["snr"] = snr
    if np.ptp(y) <= np.finfo(float).eps or np.max(y) <= 0:
        empty["warnings"].append("Нет выраженной положительной полосы Amide I после коррекции.")
        return empty
    best, fits, candidates = _select_model(processed, config, rng)
    empty["candidates"] = candidates
    if best is None:
        empty["warnings"].append("Нет воспроизводимых минимумов второй производной или допустимой модели.")
        return empty
    warnings = []
    fc = config["fitting"]
    criterion = fc["criterion"]
    alternatives = [model for model in fits if model["count"] != best["count"]]
    other_count = alternatives[0] if alternatives else None
    best_per_count = {}
    for model in alternatives:
        best_per_count.setdefault(model["count"], model)
    delta = float(other_count[criterion] - best[criterion]) if other_count is not None else np.nan
    if not config["resolution_confirmed"]:
        warnings.append("Спектральное разрешение прибора не подтверждено; проценты только диагностические.")
    if len(processed["derivatives"]) < 2:
        warnings.append("Недостаточно различных окон производной для проверки устойчивости минимумов.")
    if sum(candidate["stable"] for candidate in candidates) > config["peaks"]["max_components"]:
        warnings.append("Число устойчивых кандидатов превышает предел моделей; разложение не проверено полностью.")
    if snr < fc["min_fit_snr"]:
        warnings.append("Недостаточное отношение сигнала к шуму.")
    if best["bound_hits"]:
        warnings.append("Параметр компонента достиг границы ограничений.")
    if best["condition_number"] > fc["max_condition_number"]:
        warnings.append("Плохо обусловленная модель: параметры компонентов неоднозначны.")
    if np.isfinite(delta) and delta < fc["min_model_delta"]:
        warnings.append("Количество компонентов неоднозначно по разнице AICc/BIC.")
    residual = y - best["fit"]
    nrmse = float(np.sqrt(np.mean(residual ** 2)) / max(np.ptp(y), np.finfo(float).eps))
    lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if np.std(residual) > 1e-14 else 0.0
    dw = float(np.sum(np.diff(residual) ** 2) / np.sum(residual ** 2)) if np.any(residual) else np.nan
    r2 = float(1 - best["sse"] / np.sum((y - y.mean()) ** 2))
    if nrmse > fc["max_nrmse"]:
        warnings.append("Остаточная ошибка аппроксимации слишком велика.")
    if not np.isfinite(lag1) or abs(lag1) > fc["max_residual_lag1"]:
        warnings.append("Остатки имеют выраженную последовательную структуру.")
    boot = _bootstrap(processed, best, list(best_per_count.values()), config, rng)
    if fc["bootstrap_iterations"] < 10:
        warnings.append("Менее 10 bootstrap-повторов: устойчивость количественных оценок не проверена.")
    if boot["success_fraction"] < fc["bootstrap_success_fraction"]:
        warnings.append("Слишком много неудачных или граничных bootstrap-разложений.")
    if boot["count_support"] < fc["bootstrap_success_fraction"]:
        warnings.append("Число компонентов неустойчиво в bootstrap-разложениях.")
    if boot["count_comparison_fraction"] < fc["bootstrap_success_fraction"]:
        warnings.append("Недостаточно успешных сравнений конкурирующих bootstrap-моделей.")
    count = best["count"]
    intervals = np.column_stack((best["params"][:, 1], best["params"][:, 1]))
    area_sd = np.full(count, np.nan)
    if boot["success_count"] >= 2:
        intervals = np.percentile(boot["parameters"][:, :, 1], [2.5, 97.5], axis=0).T
        area_sd = np.std(boot["percentages"], axis=0, ddof=1)
        structure_sd = np.std(boot["structure_samples"], axis=0, ddof=1)
        if np.max(area_sd) > fc["max_percentage_sd"] or np.max(structure_sd) > fc["max_percentage_sd"]:
            warnings.append("Относительные площади/классификация неустойчивы в bootstrap.")
    areas, relative, categories, fractions = _percentages(best, config, intervals)
    # Compare preprocessing variants using the same nominal-center assignment
    # on both sides. CI-aware Unassigned labels reflect classification uncertainty
    # and must not masquerade as changes in physical areas caused by preprocessing.
    _, _, _, sensitivity_reference = _percentages(best, config)
    if np.any(relative < fc["min_component_area_pct"]):
        warnings.append("Модель включает компонент с пренебрежимо малой площадью.")
    if np.any(np.diff(best["params"][:, 1]) < max(config["spectral_resolution_cm1"],
                                                2 * processed.get("source_step", processed["step"]))):
        warnings.append("Компоненты разделены хуже разрешения/шага исходного измерения.")
    sensitivity = []
    for variant in processed.get("sensitivity_variants", []):
        alternate, _, _ = _select_model(variant["processed"], config, rng)
        if alternate is None:
            sensitivity.append({"variant": variant["label"], "status": "Unresolved",
                                "max_delta_pct": np.nan, "components": np.nan})
            warnings.append(f"Неустойчивость к обработке: {variant['label']} не дала допустимой модели.")
            continue
        alternate_processed = variant["processed"]
        alt_y = alternate_processed["corrected"]
        alt_residual = alt_y - alternate["fit"]
        alt_nrmse = float(np.sqrt(np.mean(alt_residual ** 2)) / max(np.ptp(alt_y), np.finfo(float).eps))
        alt_lag = float(np.corrcoef(alt_residual[:-1], alt_residual[1:])[0, 1]) if np.std(alt_residual) > 1e-14 else 0.0
        alt_snr = float(max(np.max(alt_y), 0) / max(alternate_processed["noise_sd"], np.finfo(float).eps))
        alt_valid = (not alternate["bound_hits"] and alternate["condition_number"] <= fc["max_condition_number"]
                     and alt_nrmse <= fc["max_nrmse"] and np.isfinite(alt_lag)
                     and abs(alt_lag) <= fc["max_residual_lag1"] and alt_snr >= fc["min_fit_snr"])
        _, _, _, alt_fractions = _percentages(alternate, config)
        shift = max(abs(alt_fractions[key] - sensitivity_reference[key]) for key in STRUCTURES)
        sensitivity.append({"variant": variant["label"], "status": "Evaluated" if alt_valid else "QC failed",
            "percentage_basis": "Diagnostic nominal-center assignment for both primary and variant",
            "diagnostic_only": True,
            "max_delta_pct": shift, "components": alternate["count"], "bound_hits": alternate["bound_hits"],
            "condition_number": alternate["condition_number"], "nrmse": alt_nrmse,
            "residual_lag1": alt_lag, "snr": alt_snr, **alt_fractions})
        if not alt_valid:
            warnings.append(f"Контроль качества альтернативной обработки не пройден: {variant['label']}.")
        if shift > fc["max_preprocessing_delta_pct"] or alternate["count"] != count:
            warnings.append(f"Относительные площади/число компонентов чувствительны к обработке: {variant['label']}.")
    if not processed.get("sensitivity_variants"):
        warnings.append("Не выполнена проверка чувствительности к предварительной обработке.")
    reliable = not warnings
    components = []
    for index, (amplitude, center, width) in enumerate(best["params"]):
        components.append({"component": index + 1, "center": float(center),
            "amplitude": float(amplitude), "fwhm": float(width), "area": float(areas[index]),
            "area_pct": float(relative[index]) if reliable else np.nan,
            "diagnostic_area_pct": float(relative[index]), "structure": categories[index],
            "center_ci_low": float(intervals[index, 0]), "center_ci_high": float(intervals[index, 1]),
            "area_pct_sd": float(area_sd[index]), "accepted": reliable})
    comparison = [{"components": model["count"], "initial_centers": ",".join(f"{c:g}" for c in model["initial_centers"]),
        "aicc": model["aicc"], "bic": model["bic"], "sse": model["sse"],
        "condition_number": model["condition_number"], "bound_hits": model["bound_hits"],
        "selected": model is best} for model in fits]
    return {"status": "Reliable fit" if reliable else "Unreliable fit", "warnings": list(dict.fromkeys(warnings)),
        "components": components, "candidates": candidates, "model_comparison": comparison,
        "fit": best["fit"], "residuals": residual,
        "metrics": {"snr": snr, "components": count, "r_squared": r2, "nrmse": nrmse,
            "residual_lag1": lag1, "durbin_watson": dw, "sse": best["sse"],
            "aicc": best["aicc"], "bic": best["bic"], "delta_criterion_next_count": delta,
            "condition_number": best["condition_number"], "bound_hits": best["bound_hits"],
            "bootstrap_success_fraction": boot["success_fraction"], "bootstrap_count_support": boot["count_support"],
            "bootstrap_count_comparison_fraction": boot["count_comparison_fraction"],
            "bootstrap_scope": "Conditional on detected candidates; best original candidate subset per available K; single-start refits",
            "evaluated_candidate_subsets": len(fits),
            "bootstrap_success_count": boot["success_count"], "bootstrap_block_length": boot["block_length"],
            "max_area_pct_sd": float(np.nanmax(area_sd))
            if np.isfinite(area_sd).any() else np.nan},
        "percentages": fractions if reliable else {name: np.nan for name in STRUCTURES}, "sensitivity": sensitivity}
