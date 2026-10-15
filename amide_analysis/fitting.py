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


def _fit_geometry(x, config, count):
    step = float(np.median(np.diff(x)))
    separation = max(config["peaks"]["min_separation_cm1"],
                     config["spectral_resolution_cm1"], 2 * step)
    minimum_width = max(config["peaks"]["min_fwhm_cm1"],
                        config["spectral_resolution_cm1"], 2 * step)
    maximum_width = config["peaks"]["max_fwhm_cm1"]
    free_span = float(x[-1] - x[0] - (count - 1) * separation)
    return separation, minimum_width, maximum_width, free_span


def _model_admissibility(model, x, config):
    """Physical identifiability gates; derivative seeds are never constraints."""
    params = np.asarray(model["params"], dtype=float).reshape(-1, 3)
    if not np.isfinite(params).all():
        return ["NONFINITE_PARAMETERS"]
    separation, minimum, maximum, _ = _fit_geometry(x, config, len(params))
    reasons = []
    tolerance = 1e-3 * (maximum - minimum)
    if np.any(params[:, 2] <= minimum + tolerance):
        reasons.append("WIDTH_LOWER_BOUND")
    if np.any(params[:, 2] >= maximum - tolerance):
        reasons.append("WIDTH_UPPER_BOUND")
    edge_tolerance = 1e-3 * float(x[-1] - x[0])
    if np.any((params[:, 1] <= x[0] + edge_tolerance)
              | (params[:, 1] >= x[-1] - edge_tolerance)):
        reasons.append("BAND_EDGE_CENTER")
    if np.any(np.diff(params[:, 1]) <= separation + 1e-3 * separation):
        reasons.append("UNRESOLVED_CENTER_SEPARATION")
    areas = np.asarray([component_area(*row, *config["amide_bounds"]) for row in params])
    if not np.isfinite(areas).all() or areas.sum() <= 0 or np.any(areas <= 0):
        reasons.append("NONPOSITIVE_COMPONENT_AREA")
    elif np.any(100 * areas / areas.sum() < config["fitting"]["min_component_area_pct"]):
        reasons.append("WEAK_COMPONENT")
    scale = model.get("amplitude_scale")
    if scale is not None and np.any(params[:, 0] >= 3 * scale * (1 - 1e-3)):
        reasons.append("AMPLITUDE_UPPER_BOUND")
    condition = float(model["condition_number"])
    if not np.isfinite(condition) or condition > config["fitting"]["max_condition_number"]:
        reasons.append("ILL_CONDITIONED")
    return reasons


def _is_admissible(model):
    return model is not None and model.get("admissible", not model.get("bound_hits", False))


def _bounded_fit(x, y, centers, config, rng, starts=None, initial=None):
    """Fit ordered centers across the band, using softmax gaps and physical bounds.

    Derivative minima initialize the optimizer only. K free logits parameterize
    K+1 positive gaps whose sum fills the available span after minimum center
    separations. A fixed final logit removes the softmax translation ambiguity.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    centers = np.sort(np.asarray(centers, float))
    count = len(centers)
    separation, minimum, maximum, span = _fit_geometry(x, config, count)
    if not count or minimum >= maximum or span <= 0:
        return None
    scale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    target = y / scale
    lower = np.r_[np.tile([0.0, minimum], count), np.full(count, -20.0)]
    upper = np.r_[np.tile([3.0, maximum], count), np.full(count, 20.0)]

    def decode(vector, with_derivative=False):
        logits = np.r_[vector[2 * count:], 0.0]
        weights = np.exp(logits - np.max(logits))
        weights /= weights.sum()
        cumulative = np.cumsum(weights)[:count]
        positions = x[0] + np.arange(count) * separation + span * cumulative
        amplitude_width = vector[:2 * count].reshape(count, 2)
        params = np.column_stack((amplitude_width[:, 0], positions, amplitude_width[:, 1]))
        if not with_derivative:
            return params
        indicators = np.arange(count)[None, :] <= np.arange(count)[:, None]
        center_derivative = span * weights[:count][None, :] * (indicators - cumulative[:, None])
        return params, center_derivative

    def encode(params):
        params = np.asarray(params, float).reshape(count, 3)
        params = params[np.argsort(params[:, 1])].copy()
        free_positions = np.sort(params[:, 1] - x[0] - np.arange(count) * separation)
        free_positions = np.clip(free_positions, 1e-6 * span, (1 - 1e-6) * span)
        gaps = np.maximum(np.diff(np.r_[0.0, free_positions, span]), 1e-6 * span)
        logits = np.log(gaps[:-1] / gaps[-1])
        vector = np.r_[params[:, [0, 2]].ravel(), logits]
        return np.clip(vector, lower + 1e-8, upper - 1e-8)

    def jacobian(vector):
        params, center_derivative = decode(vector, with_derivative=True)
        physical = _jacobian(x, params)
        amplitude_width_columns = np.array([[3 * j, 3 * j + 2] for j in range(count)]).ravel()
        return np.column_stack((physical[:, amplitude_width_columns],
                                physical[:, 1::3] @ center_derivative))

    starts = config["fitting"]["multistarts"] if starts is None else starts
    best = None
    total_evaluations = 0
    for start in range(starts):
        if start == 0 and initial is not None:
            guess = np.array(initial, float).copy().reshape(count, 3)
            guess[:, 0] /= scale
        else:
            widths = np.full(count, min(maximum * 0.45, max(minimum * 1.7, 12.0)))
            candidate_centers = centers.copy()
            if start:
                widths *= rng.uniform(0.65, 1.4, count)
                candidate_centers += rng.normal(0, max(separation, 6.0) / 2, count)
            amplitudes = np.maximum(np.interp(candidate_centers, x, target), 0.05)
            guess = np.column_stack((amplitudes, candidate_centers, widths))
        try:
            result = least_squares(lambda vector: _model(x, decode(vector)) - target,
                encode(guess), jac=jacobian, bounds=(lower, upper), x_scale="jac",
                max_nfev=config["fitting"]["max_nfev"], ftol=1e-9, xtol=1e-9, gtol=1e-9)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        total_evaluations += int(result.nfev)
        if not result.success or not np.isfinite(result.x).all():
            continue
        normalized = decode(result.x)
        physical_jacobian = _jacobian(x, normalized)
        condition = float(np.linalg.cond(physical_jacobian.T @ physical_jacobian))
        params = normalized.copy()
        params[:, 0] *= scale
        sse = float(np.sum(result.fun ** 2) * scale ** 2)
        k, n = 3 * count + 1, len(x)  # Include fitted residual variance.
        likelihood = n * math.log(max(sse / n, np.finfo(float).tiny))
        aic = likelihood + 2 * k
        aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
        model = {"params": params, "fit": _model(x, params), "sse": sse,
            "aicc": float(aicc), "bic": float(likelihood + k * math.log(n)),
            "count": count, "condition_number": condition, "amplitude_scale": scale,
            "initial_centers": centers.copy(), "nfev": int(result.nfev)}
        reasons = _model_admissibility(model, x, config)
        model["admissibility_reasons"] = reasons
        model["admissible"] = not reasons
        model["bound_hits"] = any(code in reasons for code in (
            "WIDTH_LOWER_BOUND", "WIDTH_UPPER_BOUND", "BAND_EDGE_CENTER",
            "UNRESOLVED_CENTER_SEPARATION", "AMPLITUDE_UPPER_BOUND"))
        if (best is None or (model["admissible"] and not best["admissible"])
                or (model["admissible"] == best["admissible"] and sse < best["sse"])):
            best = model
    if best is not None:
        best["solver_starts"] = int(starts)
        best["solver_total_nfev"] = total_evaluations
    return best


def _select_model(processed, config, rng):
    candidates = detect_candidates(processed, config)
    stable = [row for row in candidates if row["stable"]]
    if not stable:
        return None, [], candidates
    stable = sorted(stable, key=lambda row: -row["support"] * row["prominence"])
    stable = stable[:config["peaks"]["max_components"]]
    fits = []
    criterion = config["fitting"]["criterion"]
    for count in range(1, len(stable) + 1):
        # Full-band centers make subsets alternative starts of the same K model.
        # Cover every derivative subset once, then fill the configured start
        # budget for K. This avoids repeating six solves for every subset.
        subsets = sorted(combinations(stable, count),
                         key=lambda subset: -sum(v["support"] * v["prominence"] for v in subset))
        count_fits = []
        for subset in subsets:
            model = _bounded_fit(processed["x"], processed["corrected"],
                [row["center"] for row in subset], config, rng, starts=1)
            if model is not None and np.isfinite(model[criterion]):
                model["search_phase"] = "candidate_subset"
                count_fits.append(model)
                fits.append(model)
        extra = config["fitting"]["multistarts"] - len(subsets)
        if extra > 0:
            if count_fits:
                pool = [model for model in count_fits if _is_admissible(model)] or count_fits
                provisional = min(pool, key=lambda model: model[criterion])
                retry_centers = provisional["params"][:, 1]
            else:
                # A failed first initialization must not silently remove K.
                retry_centers = [row["center"] for row in subsets[0]]
            refined = _bounded_fit(processed["x"], processed["corrected"],
                retry_centers, config, rng, starts=extra)
            if refined is not None and np.isfinite(refined[criterion]):
                refined["search_phase"] = "additional_starts"
                fits.append(refined)
    if not fits:
        return None, [], candidates
    fits.sort(key=lambda fit: fit[criterion])
    admissible = [model for model in fits if _is_admissible(model)]
    return (admissible[0] if admissible else fits[0]), fits, candidates


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


def _bootstrap(processed, best, other_models, config, rng, progress=None):
    iterations = config["fitting"]["bootstrap_iterations"]
    retry_starts = config["fitting"].get("bootstrap_retry_multistarts", 3)
    residual = processed["corrected"] - best["fit"]
    residual = residual - residual.mean()
    parameters, percentages, structure_samples, diagnostics = [], [], [], []
    retained_count = assessed_count = 0
    block_length = min(config["fitting"].get("bootstrap_block_length", 3), max(1, len(residual) // 4))
    # Draw all residual blocks first so retries do not change subsequent samples.
    block_starts = rng.integers(0, len(residual),
        size=(iterations, int(math.ceil(len(residual) / block_length))))

    def refit(reference, simulated):
        model = _bounded_fit(processed["x"], simulated, reference["params"][:, 1],
                             config, rng, starts=1, initial=reference["params"])
        attempts = 1
        if not _is_admissible(model) and retry_starts > 1:
            retried = _bounded_fit(processed["x"], simulated, reference["params"][:, 1],
                                  config, rng, starts=retry_starts, initial=None)
            attempts += retry_starts
            if retried is not None:
                model = retried
        if model is None:
            return None, "optimizer_failed", "OPTIMIZER_FAILURE", attempts
        if not _is_admissible(model):
            reasons = model.get("admissibility_reasons", ["PHYSICAL_BOUNDARY"])
            return model, "inadmissible", ",".join(reasons), attempts
        return model, "accepted", "", attempts

    for index in range(iterations):
        indices = (block_starts[index, :, None] + np.arange(block_length)) % len(residual)
        simulated = best["fit"] + residual[indices.ravel()[:len(residual)]]
        model, status, reasons, attempts = refit(best, simulated)
        row = {"iteration": index + 1, "primary_fit_status": status,
            "primary_fit_reasons": reasons, "primary_attempts": attempts,
            "comparison_status": "primary_unavailable", "compared_counts": "",
            "unavailable_counts": "", "winning_components": np.nan,
            "selected_count_retained": False}
        if status == "accepted":
            parameters.append(model["params"])
            _, relative, _, structure = _percentages(model, config)
            percentages.append(relative)
            structure_samples.append([structure[name] for name in STRUCTURES])
            comparisons, unavailable, comparison_reasons, compared = [], [], [], []
            comparison_attempts = 0
            for reference in other_models:
                alternative, alt_status, alt_reasons, alt_attempts = refit(reference, simulated)
                comparison_attempts += alt_attempts
                compared.append(reference["count"])
                if alt_status != "accepted":
                    unavailable.append(reference["count"])
                    comparison_reasons.append(f"K={reference['count']}:{alt_status}:{alt_reasons}")
                else:
                    comparisons.append(alternative)
            row.update({"compared_counts": ",".join(str(value) for value in compared),
                "unavailable_counts": ",".join(str(value) for value in unavailable),
                "comparison_failure_reasons": ";".join(comparison_reasons),
                "comparison_attempts": comparison_attempts,
                "comparison_status": "indeterminate" if unavailable else "complete"})
            # Failed or physically inadmissible competitors never count as
            # evidence for K; the comparison remains explicitly indeterminate.
            if not unavailable:
                assessed_count += 1
                winner = min([model] + comparisons,
                             key=lambda value: value[config["fitting"]["criterion"]])
                retained = winner["count"] == best["count"]
                retained_count += int(retained)
                row["winning_components"] = int(winner["count"])
                row["selected_count_retained"] = retained
        diagnostics.append(row)
        if progress is not None and (index == 0 or (index + 1) % 10 == 0 or index + 1 == iterations):
            progress(f"  Bootstrap: {index + 1}/{iterations}; допустимых разложений: {len(parameters)}.")
    count = len(parameters)
    return {"parameters": np.asarray(parameters), "percentages": np.asarray(percentages),
            "structure_samples": np.asarray(structure_samples), "success_count": count,
            "success_fraction": count / iterations if iterations else 0.0,
            "count_support": retained_count / iterations if iterations else 0.0,
            "count_comparison_fraction": assessed_count / iterations if iterations else 0.0,
            "block_length": block_length, "diagnostics": diagnostics}


def analyze_band(processed, config, seed=None, progress=None):
    """Audit model stability separately from quantitative protocol eligibility.

    Invalid models and diagnostic areas remain visible. Only physically
    admissible models enter information-criterion comparisons or bootstrap.
    Unknown instrument resolution is a protocol blocker, not evidence of an
    unstable numerical fit.
    """
    rng = np.random.default_rng(config["seed"] if seed is None else seed)
    config = deepcopy(config)
    config["peaks"]["min_fwhm_cm1"] = max(config["peaks"]["min_fwhm_cm1"],
        2 * processed.get("source_step", processed["step"]))
    config["peaks"]["min_separation_cm1"] = max(config["peaks"]["min_separation_cm1"],
        2 * processed.get("source_step", processed["step"]))
    x, y = np.asarray(processed["x"]), np.asarray(processed["corrected"])
    fc = config["fitting"]
    criterion = fc["criterion"]
    warnings, rejection_codes, protocol_blockers = [], [], []

    def report(message):
        if progress is not None:
            progress(message)

    def reject(code, message):
        rejection_codes.append(code)
        warnings.append(message)

    def admissible(model):
        if "admissible" in model:
            return bool(model["admissible"])
        # Backward compatibility for callers supplying pre-1.1 model fixtures.
        condition = model.get("condition_number", np.inf)
        return bool(not model.get("bound_hits", False) and np.isfinite(condition)
                    and condition <= fc["max_condition_number"])

    if not config["resolution_confirmed"]:
        protocol_blockers.append("INSTRUMENT_RESOLUTION_UNCONFIRMED")
        warnings.append("Спектральное разрешение прибора не подтверждено; проценты только диагностические.")
    empty = {"status": "Unreliable fit", "warnings": warnings, "components": [], "candidates": [],
        "model_comparison": [], "fit": np.full(len(x), np.nan), "residuals": np.full(len(x), np.nan),
        "metrics": {"numerical_fit_reliable": False, "quantitative_accepted": False},
        "percentages": {name: np.nan for name in STRUCTURES}, "sensitivity": [],
        "numerical_fit_reliable": False, "quantitative_accepted": False,
        "fit_rejection_codes": rejection_codes, "protocol_blockers": protocol_blockers,
        "diagnostic_percentages": {name: np.nan for name in STRUCTURES}, "bootstrap_diagnostics": []}
    noise = max(float(processed["noise_sd"]), np.finfo(float).eps)
    snr = float(max(np.max(y), 0) / noise)
    empty["metrics"]["snr"] = snr
    if np.ptp(y) <= np.finfo(float).eps or np.max(y) <= 0:
        reject("FLAT_OR_NONPOSITIVE_BAND", "Нет выраженной положительной полосы Amide I после коррекции.")
        return empty
    if snr < fc["min_fit_snr"]:
        reject("LOW_SIGNAL_TO_NOISE", "Недостаточное отношение сигнала к шуму; подбор компонентов пропущен.")
        report("Подбор компонентов пропущен: недостаточное отношение сигнала к шуму.")
        return empty
    report("Поиск моделей по воспроизводимым кандидатам второй производной.")
    best, fits, candidates = _select_model(processed, config, rng)
    empty["candidates"] = candidates
    if best is None:
        reject("NO_REPRODUCIBLE_CANDIDATE_OR_MODEL",
               "Нет воспроизводимых минимумов второй производной или сошедшейся модели.")
        return empty

    # Keep every converged candidate for audit, while comparing admissible
    # alternatives only. A rejected low-AICc solution cannot defeat valid fits.
    valid_models = sorted((model for model in fits
                           if admissible(model) and np.isfinite(model[criterion])),
                          key=lambda model: model[criterion])
    if valid_models:
        best = valid_models[0]
    primary_admissible = bool(valid_models) and admissible(best)
    if not primary_admissible:
        reject("NO_PHYSICALLY_ADMISSIBLE_MODEL",
               "Ни одна сошедшаяся модель не прошла физические ограничения; "
               "кривые сохранены диагностически, bootstrap пропущен.")
    alternatives = [model for model in valid_models if model["count"] != best["count"]]
    best_per_count = {}
    for model in alternatives:
        best_per_count.setdefault(model["count"], model)
    other_count = alternatives[0] if alternatives else None
    delta = float(other_count[criterion] - best[criterion]) if other_count is not None else np.nan
    if len(processed["derivatives"]) < 2:
        reject("INSUFFICIENT_DERIVATIVE_WINDOWS",
               "Недостаточно различных окон производной для проверки устойчивости минимумов.")
    if sum(candidate["stable"] for candidate in candidates) > config["peaks"]["max_components"]:
        reject("CANDIDATE_LIMIT_EXCEEDED",
               "Число устойчивых кандидатов превышает предел моделей; разложение не проверено полностью.")
    if best["bound_hits"]:
        reject("PHYSICAL_PARAMETER_BOUND", "Параметр компонента достиг физической границы ограничений.")
    if not np.isfinite(best["condition_number"]) or best["condition_number"] > fc["max_condition_number"]:
        reject("ILL_CONDITIONED_MODEL", "Плохо обусловленная модель: параметры компонентов неоднозначны.")
    if np.isfinite(delta) and delta < fc["min_model_delta"]:
        reject("AMBIGUOUS_COMPONENT_COUNT", "Количество компонентов неоднозначно по разнице AICc/BIC.")
    residual = y - best["fit"]
    nrmse = float(np.sqrt(np.mean(residual ** 2)) / max(np.ptp(y), np.finfo(float).eps))
    lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if np.std(residual) > 1e-14 else 0.0
    dw = float(np.sum(np.diff(residual) ** 2) / np.sum(residual ** 2)) if np.any(residual) else np.nan
    r2 = float(1 - best["sse"] / np.sum((y - y.mean()) ** 2))
    if not np.isfinite(nrmse) or nrmse > fc["max_nrmse"]:
        reject("HIGH_RESIDUAL_ERROR", "Остаточная ошибка аппроксимации слишком велика.")
    if not np.isfinite(lag1) or abs(lag1) > fc["max_residual_lag1"]:
        reject("STRUCTURED_RESIDUALS", "Остатки имеют выраженную последовательную структуру.")

    count = best["count"]
    intervals = np.column_stack((best["params"][:, 1], best["params"][:, 1]))
    area_sd = np.full(count, np.nan)
    structure_sd = np.full(len(STRUCTURES), np.nan)
    if primary_admissible:
        report(f"Bootstrap: {fc['bootstrap_iterations']} повторов для модели из {count} компонентов.")
        boot = _bootstrap(processed, best, list(best_per_count.values()), config, rng, progress=progress)
        bootstrap_status = "Evaluated"
        if fc["bootstrap_iterations"] < 10:
            reject("INSUFFICIENT_BOOTSTRAP_ITERATIONS",
                   "Менее 10 bootstrap-повторов: устойчивость количественных оценок не проверена.")
        if boot["success_fraction"] < fc["bootstrap_success_fraction"]:
            reject("LOW_BOOTSTRAP_SUCCESS", "Слишком много неудачных или недопустимых bootstrap-разложений.")
        if boot["count_support"] < fc["bootstrap_success_fraction"]:
            reject("BOOTSTRAP_COMPONENT_COUNT_INSTABILITY", "Число компонентов неустойчиво в bootstrap-разложениях.")
        if boot["count_comparison_fraction"] < fc["bootstrap_success_fraction"]:
            reject("INCOMPLETE_BOOTSTRAP_COUNT_COMPARISON",
                   "Недостаточно успешных сравнений конкурирующих bootstrap-моделей.")
    else:
        report("Bootstrap пропущен: физически допустимой модели нет.")
        bootstrap_status = "Skipped: no physically admissible model"
        boot = {"parameters": np.empty((0, count, 3)), "percentages": np.empty((0, count)),
                "success_count": 0, "success_fraction": np.nan, "count_support": np.nan,
                "count_comparison_fraction": np.nan, "block_length": np.nan, "diagnostics": []}
    if boot["success_count"] >= 2:
        intervals = np.percentile(boot["parameters"][:, :, 1], [2.5, 97.5], axis=0).T
        area_sd = np.std(boot["percentages"], axis=0, ddof=1)
    areas, relative, categories, fractions = _percentages(best, config, intervals)
    if boot["success_count"] >= 2:
        # Classify once using the final confidence intervals. Reclassifying
        # every bootstrap center against nominal boundaries would test a
        # different estimator and falsely report label switching as area SD.
        category_array = np.asarray(categories)
        samples = np.asarray(boot["percentages"])
        final_structure_samples = np.column_stack([
            samples[:, category_array == name].sum(axis=1) for name in STRUCTURES])
        structure_sd = np.std(final_structure_samples, axis=0, ddof=1)
        if (not np.isfinite(area_sd).all() or not np.isfinite(structure_sd).all()
                or np.max(area_sd) > fc["max_percentage_sd"]
                or np.max(structure_sd) > fc["max_percentage_sd"]):
            reject("BOOTSTRAP_AREA_INSTABILITY", "Относительные площади/классификация неустойчивы в bootstrap.")

    # Baseline/derivative sensitivity compares nominal-center assignments on
    # both sides, independently of the final CI-aware Unassigned labels.
    _, _, _, sensitivity_reference = _percentages(best, config)
    if not np.isfinite(relative).all() or np.any(relative < fc["min_component_area_pct"]):
        reject("NEGLIGIBLE_OR_INVALID_COMPONENT_AREA", "Модель включает компонент с пренебрежимо малой или недопустимой площадью.")
    if np.any(np.diff(best["params"][:, 1]) < max(config["spectral_resolution_cm1"],
                                                2 * processed.get("source_step", processed["step"]))):
        reject("UNRESOLVED_COMPONENT_SEPARATION", "Компоненты разделены хуже разрешения/шага исходного измерения.")
    sensitivity = []
    if primary_admissible:
        report("Проверка чувствительности к базовой линии и окнам производной.")
        for variant in processed.get("sensitivity_variants", []):
            alternate, _, _ = _select_model(variant["processed"], config, rng)
            if alternate is None:
                sensitivity.append({"variant": variant["label"], "status": "Unresolved",
                                    "max_delta_pct": np.nan, "components": np.nan})
                reject("UNRESOLVED_PREPROCESSING_VARIANT",
                       f"Неустойчивость к обработке: {variant['label']} не дала допустимой модели.")
                continue
            alternate_processed = variant["processed"]
            alt_y = alternate_processed["corrected"]
            alt_residual = alt_y - alternate["fit"]
            alt_nrmse = float(np.sqrt(np.mean(alt_residual ** 2)) / max(np.ptp(alt_y), np.finfo(float).eps))
            alt_lag = float(np.corrcoef(alt_residual[:-1], alt_residual[1:])[0, 1]) if np.std(alt_residual) > 1e-14 else 0.0
            alt_snr = float(max(np.max(alt_y), 0) / max(alternate_processed["noise_sd"], np.finfo(float).eps))
            alt_valid = (admissible(alternate) and np.isfinite(alt_nrmse)
                         and alt_nrmse <= fc["max_nrmse"] and np.isfinite(alt_lag)
                         and abs(alt_lag) <= fc["max_residual_lag1"] and alt_snr >= fc["min_fit_snr"])
            _, _, _, alt_fractions = _percentages(alternate, config)
            shift = max(abs(alt_fractions[key] - sensitivity_reference[key]) for key in STRUCTURES)
            sensitivity.append({"variant": variant["label"], "status": "Evaluated" if alt_valid else "QC failed",
                "percentage_basis": "Diagnostic nominal-center assignment for both primary and variant",
                "diagnostic_only": True, "admissible": admissible(alternate),
                "admissibility_reasons": alternate.get("admissibility_reasons", []),
                "max_delta_pct": shift, "components": alternate["count"], "bound_hits": alternate["bound_hits"],
                "condition_number": alternate["condition_number"], "nrmse": alt_nrmse,
                "residual_lag1": alt_lag, "snr": alt_snr, **alt_fractions})
            if not alt_valid:
                reject("PREPROCESSING_VARIANT_QC_FAILED",
                       f"Контроль качества альтернативной обработки не пройден: {variant['label']}.")
            if shift > fc["max_preprocessing_delta_pct"] or alternate["count"] != count:
                reject("PREPROCESSING_AREA_OR_COUNT_INSTABILITY",
                       f"Относительные площади/число компонентов чувствительны к обработке: {variant['label']}.")
    else:
        for variant in processed.get("sensitivity_variants", []):
            sensitivity.append({"variant": variant["label"], "status": "Skipped: primary model inadmissible",
                                "max_delta_pct": np.nan, "components": np.nan})
    if not processed.get("sensitivity_variants"):
        reject("MISSING_PREPROCESSING_SENSITIVITY", "Не выполнена проверка чувствительности к предварительной обработке.")

    rejection_codes = list(dict.fromkeys(rejection_codes))
    numeric_reliable = primary_admissible and not rejection_codes
    quantitative_accepted = numeric_reliable and not protocol_blockers
    components = []
    for index, (amplitude, center, width) in enumerate(best["params"]):
        components.append({"component": index + 1, "center": float(center),
            "amplitude": float(amplitude), "fwhm": float(width), "area": float(areas[index]),
            "area_pct": float(relative[index]) if quantitative_accepted else np.nan,
            "diagnostic_area_pct": float(relative[index]), "structure": categories[index],
            "center_ci_low": float(intervals[index, 0]) if boot["success_count"] >= 2 else np.nan,
            "center_ci_high": float(intervals[index, 1]) if boot["success_count"] >= 2 else np.nan,
            "area_pct_sd": float(area_sd[index]), "accepted": quantitative_accepted})
    comparison = [{"components": model["count"],
        "initial_centers": ",".join(f"{c:g}" for c in model["initial_centers"]),
        "aicc": model["aicc"], "bic": model["bic"], "sse": model["sse"],
        "condition_number": model["condition_number"], "bound_hits": model["bound_hits"],
        "search_phase": model.get("search_phase", "candidate_subset"),
        "solver_starts": model.get("solver_starts", np.nan),
        "solver_total_nfev": model.get("solver_total_nfev", model.get("nfev", np.nan)),
        "admissible": admissible(model), "admissibility_reasons": model.get("admissibility_reasons", []),
        "eligible_for_model_selection": admissible(model) and np.isfinite(model[criterion]),
        "selected": model is best, "selected_diagnostic_fallback": model is best and not primary_admissible}
        for model in fits]
    report("Разложение и проверки завершены; подготовка диагностических таблиц.")
    return {"status": "Reliable fit" if quantitative_accepted else "Unreliable fit",
        "warnings": list(dict.fromkeys(warnings)), "components": components, "candidates": candidates,
        "model_comparison": comparison, "fit": best["fit"], "residuals": residual,
        "numerical_fit_reliable": numeric_reliable, "quantitative_accepted": quantitative_accepted,
        "fit_rejection_codes": rejection_codes, "protocol_blockers": protocol_blockers,
        "diagnostic_percentages": fractions, "bootstrap_diagnostics": boot.get("diagnostics", []),
        "metrics": {"snr": snr, "components": count, "r_squared": r2, "nrmse": nrmse,
            "residual_lag1": lag1, "durbin_watson": dw, "sse": best["sse"],
            "aicc": best["aicc"], "bic": best["bic"], "delta_criterion_next_count": delta,
            "condition_number": best["condition_number"], "bound_hits": best["bound_hits"],
            "primary_model_admissible": primary_admissible,
            "numerical_fit_reliable": numeric_reliable, "quantitative_accepted": quantitative_accepted,
            "bootstrap_success_fraction": boot["success_fraction"], "bootstrap_count_support": boot["count_support"],
            "bootstrap_count_comparison_fraction": boot["count_comparison_fraction"],
            "bootstrap_scope": "Conditional on detected candidates; best admissible original subset per available K; staged refits",
            "bootstrap_status": bootstrap_status,
            "evaluated_candidate_subsets": sum(model.get("search_phase", "candidate_subset") == "candidate_subset"
                                               for model in fits),
            "admissible_candidate_subsets": sum(model.get("search_phase", "candidate_subset") == "candidate_subset"
                                                for model in valid_models),
            "evaluated_fit_records": len(fits),
            "model_search_total_nfev": sum(model.get("solver_total_nfev", model.get("nfev", 0)) for model in fits),
            "bootstrap_success_count": boot["success_count"], "bootstrap_block_length": boot["block_length"],
            "max_area_pct_sd": float(np.nanmax(area_sd)) if np.isfinite(area_sd).any() else np.nan,
            "max_structure_pct_sd": float(np.nanmax(structure_sd)) if np.isfinite(structure_sd).any() else np.nan,
            "bootstrap_structure_sd_basis": "Final CI-aware fixed categories; bootstrap component areas summed by final category"},
        "percentages": fractions if quantitative_accepted else {name: np.nan for name in STRUCTURES},
        "sensitivity": sensitivity}
