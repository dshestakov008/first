"""Inference on biological animals, with strict technical-repeat quality checks.

The statistical family comprises all finite p values emitted in one call: the
five compositional fractions, alpha/beta ratio, positive alpha/beta log ratio,
and an explicitly requested beta-sheet + beta-turn endpoint; omnibus and tested
pairwise hypotheses share one Benjamini--Hochberg correction. All planned
eligible pairwise tests are computed before correction; adjusted omnibus
significance gates interpretation, without selecting the correction family.
These endpoints
overlap and are exploratory, not independent estimates of secondary structure.
Methods are declared in configuration; normality tests never select a method.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats


STRUCTURES = ("random_coil", "alpha_helix", "beta_sheet", "beta_turn", "unassigned")
ANIMAL_COLUMNS = ["animal_id", "group", "status", "n_repeats", *STRUCTURES,
                  "alpha_beta_ratio", "log_alpha_beta"]
DIAGNOSTIC_COLUMNS = ["animal_id", "group", "spectrum_id", "status", "message"]
DESCRIPTIVE_COLUMNS = ["metric", "group", "n", "mean", "sd", "median", "q25",
                       "q75", "iqr", "mean_ci_low", "mean_ci_high", "mean_ci_method",
                       "median_ci_low", "median_ci_high", "median_ci_method", "status"]
TEST_COLUMNS = ["metric", "group1", "group2", "groups", "method", "n", "n1",
                "n2", "n_total", "p_value", "q_value", "effect", "effect_name",
                "effect_ci_low", "effect_ci_high", "effect_ci_method", "status",
                "hypothesis", "family", "posthoc_gate", "interpretation_allowed"]


def _merged_beta(config: dict) -> bool:
    return bool(config.get("plots", {}).get("merge_beta_turn", False)
                or config.get("statistics", {}).get("merge_beta_turn", False))


def _seed(config: dict, *parts: str) -> int:
    """Stable endpoint seeds make results independent of row/group order."""
    identity = "\x00".join([str(config.get("seed", 20261014)), *parts])
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")


def aggregate_animals(spectrum_results: pd.DataFrame, metadata: pd.DataFrame,
                      config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average repeats equally within animals; exclude an animal if any fails.

    Expected repeats are taken from metadata, including missing result rows.
    Zero percentages remain zero. Missing/failed fractions remain NaN and are
    never imputed. Diagnostics retain the exclusion reason for each animal.
    """
    required = {"spectrum_id", "animal_id", "group"}
    if not required.issubset(metadata.columns):
        raise ValueError("Metadata must contain spectrum_id, animal_id and group")
    meta = metadata.copy()
    if meta[list(required)].isna().any().any():
        raise ValueError("Spectrum, animal and group identifiers must be present")
    for column in required:
        meta[column] = meta[column].astype(str)
        if meta[column].str.strip().eq("").any():
            raise ValueError(f"Empty metadata identifier: {column}")
    if meta["spectrum_id"].duplicated().any():
        raise ValueError("Duplicate spectrum identifiers in metadata")
    if meta.groupby("animal_id")["group"].nunique().gt(1).any():
        raise ValueError("One animal cannot belong to multiple groups")

    results = spectrum_results.copy()
    if results.empty and "spectrum_id" not in results:
        results["spectrum_id"] = pd.Series(dtype=str)
    if "spectrum_id" not in results:
        raise ValueError("Spectrum results must contain spectrum_id")
    if results["spectrum_id"].isna().any():
        raise ValueError("Missing spectrum identifier in results")
    results["spectrum_id"] = results["spectrum_id"].astype(str)
    if results["spectrum_id"].duplicated().any():
        raise ValueError("Duplicate spectrum identifiers in results")
    if not set(results["spectrum_id"]).issubset(set(meta["spectrum_id"])):
        raise ValueError("Spectrum results contain identifiers absent from metadata")
    for column in ["status", *STRUCTURES]:
        if column not in results:
            results[column] = np.nan
    lookup = results.set_index("spectrum_id")
    limit = float(config.get("replicates", {}).get("max_repeat_delta_pct", 10.0))
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("max_repeat_delta_pct must be finite and nonnegative")

    rows, diagnostics = [], []
    for animal, animal_meta in meta.groupby("animal_id", sort=True):
        group = animal_meta["group"].iloc[0]
        values, reasons = [], []
        for spectrum_id in animal_meta["spectrum_id"]:
            if spectrum_id not in lookup.index:
                reason = "Missing spectrum result"
            else:
                spectrum = lookup.loc[spectrum_id]
                if not isinstance(spectrum["status"], str) or spectrum["status"] != "Reliable fit":
                    reason = f"Spectrum is not reliable: {spectrum['status']}"
                else:
                    try:
                        fractions = np.asarray([spectrum[s] for s in STRUCTURES], dtype=float)
                    except (TypeError, ValueError):
                        fractions = np.full(len(STRUCTURES), np.nan)
                    if not np.isfinite(fractions).all():
                        reason = "Missing or nonfinite accepted percentages"
                    elif ((fractions < 0) | (fractions > 100)).any():
                        reason = "Accepted percentages are outside [0, 100]"
                    elif not np.isclose(fractions.sum(), 100.0, atol=0.05, rtol=0):
                        reason = "Accepted percentages do not sum to 100%"
                    else:
                        values.append(fractions)
                        reason = ""
            if reason:
                reasons.append(reason)
                diagnostics.append(dict(animal_id=animal, group=group,
                                        spectrum_id=spectrum_id, status="Unreliable fit",
                                        message=reason))
        if not reasons and values:
            matrix = np.asarray(values)
            excessive = np.ptp(matrix, axis=0) > limit + 1e-10
            if excessive.any():
                names = ", ".join(np.asarray(STRUCTURES)[excessive])
                reason = f"Technical-repeat range exceeds {limit:g} percentage points: {names}"
                reasons.append(reason)
                diagnostics.append(dict(animal_id=animal, group=group,
                                        spectrum_id="", status="Unreliable fit", message=reason))
        accepted = bool(values) and not reasons
        fractions = np.mean(values, axis=0) if accepted else np.full(len(STRUCTURES), np.nan)
        row = dict(animal_id=animal, group=group,
                   status="Reliable fit" if accepted else "Unreliable fit",
                   n_repeats=len(animal_meta), **dict(zip(STRUCTURES, fractions)))
        alpha, beta = row["alpha_helix"], row["beta_sheet"]
        row["alpha_beta_ratio"] = alpha / beta if accepted and beta > 0 else np.nan
        row["log_alpha_beta"] = math.log(alpha / beta) if accepted and alpha > 0 and beta > 0 else np.nan
        if _merged_beta(config):
            row["beta_associated"] = row["beta_sheet"] + row["beta_turn"]
        if accepted:
            diagnostics.append(dict(animal_id=animal, group=group, spectrum_id="",
                                    status="Reliable fit",
                                    message="Arithmetic mean of accepted technical repeats; one biological unit"))
        rows.append(row)
    columns = ANIMAL_COLUMNS + (["beta_associated"] if _merged_beta(config) else [])
    return (pd.DataFrame(rows, columns=columns),
            pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS))


def benjamini_hochberg(p_values) -> np.ndarray:
    """Correct finite valid p values together; preserve missing positions."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p) & (p >= 0) & (p <= 1)
    indexes = np.flatnonzero(valid)
    if len(indexes):
        ordered = indexes[np.argsort(p[indexes], kind="stable")]
        adjusted = p[ordered] * len(ordered) / np.arange(1, len(ordered) + 1)
        q[ordered] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return q


def _kruskal_h(*samples, axis=-1):
    """Vectorized, tie-corrected H; identical samples have H=0."""
    samples = [np.moveaxis(np.asarray(a), axis, -1) for a in samples]
    sizes = [a.shape[-1] for a in samples]
    combined = np.concatenate(samples, axis=-1)
    total = sum(sizes)
    ranks = stats.rankdata(combined, axis=-1, method="average")
    term, offset = np.zeros(combined.shape[:-1]), 0
    for size in sizes:
        rank_sum = ranks[..., offset:offset + size].sum(axis=-1)
        term += rank_sum ** 2 / size
        offset += size
    h = 12 * term / (total * (total + 1)) - 3 * (total + 1)
    untied_squares = total * (total + 1) * (2 * total + 1) / 6
    correction = 1 - 12 * (untied_squares - np.sum(ranks ** 2, axis=-1)) / (total ** 3 - total)
    return np.divide(np.maximum(h, 0), correction,
                     out=np.zeros_like(h, dtype=float), where=correction > 1e-14)


def _welch_anova(*samples, axis=-1):
    """Welch's heteroscedastic ANOVA, returning F, denominator df and p."""
    arrays = [np.moveaxis(np.asarray(a), axis, -1) for a in samples]
    n = np.asarray([a.shape[-1] for a in arrays], dtype=float)
    means = np.stack([a.mean(axis=-1) for a in arrays], axis=-1)
    variances = np.stack([a.var(axis=-1, ddof=1) for a in arrays], axis=-1)
    k = len(arrays)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        weights = n / variances
        weight_sum = weights.sum(axis=-1, keepdims=True)
        weighted_mean = (weights * means).sum(axis=-1, keepdims=True) / weight_sum
        adjustment = np.sum((1 - weights / weight_sum) ** 2 / (n - 1), axis=-1)
        numerator = np.sum(weights * (means - weighted_mean) ** 2, axis=-1) / (k - 1)
        f = numerator / (1 + 2 * (k - 2) * adjustment / (k ** 2 - 1))
        df = (k ** 2 - 1) / (3 * adjustment)
        p = stats.f.sf(f, k - 1, df)
    return f, df, p


def _hedges_g(first, second, axis=-1):
    n1, n2 = first.shape[axis], second.shape[axis]
    df = n1 + n2 - 2
    pooled = ((n1 - 1) * first.var(axis=axis, ddof=1)
              + (n2 - 1) * second.var(axis=axis, ddof=1)) / df
    with np.errstate(divide="ignore", invalid="ignore"):
        effect = (second.mean(axis=axis) - first.mean(axis=axis)) / np.sqrt(pooled)
    # Exact normal-theory small-sample bias correction, including small n.
    correction = math.exp(math.lgamma(df / 2) - .5 * math.log(df / 2)
                          - math.lgamma((df - 1) / 2))
    return effect * correction


def _rank_biserial(first, second):
    """Positive effect means the second named group has larger values."""
    differences = second[..., :, None] - first[..., None, :]
    return np.mean(np.sign(differences), axis=(-2, -1))


def _effect(samples, method: str):
    if len(samples) == 2:
        if method == "mannwhitney":
            return _rank_biserial(*samples), "rank_biserial (group2 > group1)"
        return _hedges_g(*samples), "Hedges g (group2 - group1)"
    if method == "mannwhitney":
        n, k = sum(a.shape[-1] for a in samples), len(samples)
        return np.clip((_kruskal_h(*samples) - k + 1) / (n - k), 0, 1), "epsilon_squared"
    f, df, _ = _welch_anova(*samples)
    return f * (len(samples) - 1) / (f * (len(samples) - 1) + df), "Welch partial_eta_squared"


def _effect_ci(samples, method, iterations, seed):
    """Stratified animal-level percentile bootstrap, with bounded memory."""
    if iterations <= 0:
        return np.nan, np.nan, "Disabled"
    rng = np.random.default_rng(seed)
    effects = []
    # Pairwise sign matrices are the largest intermediate arrays.
    size = np.prod([len(a) for a in samples]) if len(samples) == 2 and method == "mannwhitney" else sum(map(len, samples))
    batch = max(1, min(256, int(2_000_000 / max(size, 1))))
    for start in range(0, iterations, batch):
        count = min(batch, iterations - start)
        resampled = [a[rng.integers(len(a), size=(count, len(a)))] for a in samples]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            effects.append(np.atleast_1d(_effect(resampled, method)[0]))
    estimates = np.concatenate(effects)
    finite = estimates[np.isfinite(estimates)]
    if len(finite) < max(2, math.ceil(iterations * 0.8)):
        return np.nan, np.nan, f"Unavailable: only {len(finite)}/{iterations} finite bootstrap effects"
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), f"Animal bootstrap percentile 95%; {len(finite)}/{iterations} valid"


def _pair_p(first, second, method, seed, resamples):
    if method == "welch":
        v1, v2 = np.var(first, ddof=1) / len(first), np.var(second, ddof=1) / len(second)
        if v1 == 0 and v2 == 0:
            return np.nan, "Welch t-test", "Zero variance prevents Welch inference"
        statistic = (first.mean() - second.mean()) / math.sqrt(v1 + v2)
        df = (v1 + v2) ** 2 / (v1 ** 2 / (len(first)-1) + v2 ** 2 / (len(second)-1))
        return float(2 * stats.t.sf(abs(statistic), df)), "Welch t-test", "OK"
    combined = np.concatenate([first, second])
    ties = len(np.unique(combined)) < len(combined)
    if ties:
        permutation = stats.PermutationMethod(n_resamples=resamples, batch=256,
                                             random_state=np.random.default_rng(seed))
        result = stats.mannwhitneyu(first, second, alternative="two-sided", method=permutation)
        label = "Mann-Whitney U (permutation, ties)"
    else:
        selected = "exact" if min(len(first), len(second)) <= 8 else "asymptotic"
        result = stats.mannwhitneyu(first, second, alternative="two-sided", method=selected)
        label = f"Mann-Whitney U ({selected})"
    return float(result.pvalue), label, "OK"


def _omnibus_p(samples, method, seed, resamples):
    if method == "welch":
        if any(np.var(a, ddof=1) == 0 for a in samples):
            return np.nan, "Welch ANOVA", "Zero variance prevents Welch ANOVA"
        _, _, p = _welch_anova(*samples)
        return float(p), "Welch ANOVA", "OK"
    combined = np.concatenate(samples)
    if len(np.unique(combined)) == 1:
        return 1.0, "Kruskal-Wallis (identical values)", "OK: all values identical"
    ties = len(np.unique(combined)) < len(combined)
    if ties or min(map(len, samples)) < 5:
        result = stats.permutation_test(samples, _kruskal_h, vectorized=True,
                                        permutation_type="independent", alternative="greater",
                                        n_resamples=resamples, batch=256,
                                        random_state=np.random.default_rng(seed))
        return float(result.pvalue), "Kruskal-Wallis (permutation)", "OK"
    return float(stats.kruskal(*samples).pvalue), "Kruskal-Wallis (asymptotic)", "OK"


def _test_row(metric, groups, samples, method, config, hypothesis):
    settings = config.get("statistics", {})
    counts = dict(zip(groups, map(len, samples)))
    minimum = max(2, int(settings.get("min_n_test", 2)))
    row = dict(metric=metric, group1=groups[0] if groups else "",
               group2=groups[1] if len(groups) == 2 else "All groups" if len(groups) > 2 else "",
               groups=json.dumps(groups, ensure_ascii=False), n=json.dumps(counts, ensure_ascii=False),
               n1=len(samples[0]) if samples else 0,
               n2=len(samples[1]) if len(samples) == 2 else np.nan,
               n_total=sum(map(len, samples)), method="", p_value=np.nan, q_value=np.nan,
               effect=np.nan, effect_name="", effect_ci_low=np.nan, effect_ci_high=np.nan,
               effect_ci_method="Not estimated", status="OK", hypothesis=hypothesis,
               family="All reported inferential hypotheses",
               posthoc_gate="BH-adjusted omnibus q < alpha" if hypothesis == "posthoc" and settings.get("posthoc_only_if_omnibus_significant", True) else "",
               interpretation_allowed=False)
    if len(groups) < 2:
        row["status"] = "Insufficient groups: at least two required"
        return row
    if any(len(a) < minimum for a in samples):
        row["status"] = f"Insufficient n: at least {minimum} accepted animals per group required"
        return row
    iterations = int(settings.get("bootstrap_iterations", 2000))
    resamples = int(settings.get("permutation_resamples", 9999))
    endpoint_seed = _seed(config, metric, *groups, hypothesis)
    operation = _pair_p if len(samples) == 2 else _omnibus_p
    if len(samples) == 2:
        p, name, status = operation(*samples, method, endpoint_seed, resamples)
    else:
        p, name, status = operation(samples, method, endpoint_seed, resamples)
    row.update(p_value=p, method=name, status=status)
    if np.isfinite(p):
        effect, effect_name = _effect(samples, method)
        row["effect_name"] = effect_name
        if np.isfinite(effect):
            low, high, ci_method = _effect_ci(samples, method, iterations, endpoint_seed)
            row.update(effect=float(effect), effect_ci_low=low,
                       effect_ci_high=high, effect_ci_method=ci_method)
        else:
            row["effect_ci_method"] = "Unavailable: zero variance or undefined effect"
    return row


def perform_statistics(animals: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe and compare biological units using one predeclared method.

    All planned eligible omnibus and pairwise hypotheses share a single BH
    family. Afterwards, adjusted omnibus significance gates interpretation of
    pairwise tests; their p/q values remain available for audit. The gate and
    family are exported explicitly. Significant plotting uses q and
    interpretation_allowed, never the raw p. The method uses permutations for
    ties, exact untied small two-group tests and tie-corrected asymptotic tests
    for larger untied samples. Log ratios require strictly positive fractions;
    no pseudocounts are introduced.
    """
    settings = config.get("statistics", {})
    method = settings.get("method", "mannwhitney")
    if method not in {"mannwhitney", "welch"}:
        raise ValueError("Predeclared statistics.method must be mannwhitney or welch")
    iterations = int(settings.get("bootstrap_iterations", 2000))
    resamples = int(settings.get("permutation_resamples", 9999))
    alpha = float(settings.get("alpha", 0.05))
    if iterations < 0 or resamples < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid statistics iteration count or significance threshold")
    required = {"animal_id", "group", "status"}
    if not required.issubset(animals.columns):
        raise ValueError("Animals must contain animal_id, group and status")
    if animals["animal_id"].duplicated().any():
        raise ValueError("Animal rows must be unique; technical repeats are not biological units")
    if animals[["animal_id", "group"]].isna().any().any():
        raise ValueError("Animal identifier and group cannot be missing")
    frame = animals.copy()
    frame["group"] = frame["group"].astype(str)
    frame["animal_id"] = frame["animal_id"].astype(str)
    frame = frame.sort_values(["group", "animal_id"], kind="stable")
    groups = sorted(frame["group"].unique())
    accepted = frame.loc[frame["status"] == "Reliable fit"].copy()
    metrics = [*STRUCTURES, "alpha_beta_ratio"]
    if settings.get("include_log_ratios", True):
        metrics.append("log_alpha_beta")
    if _merged_beta(config):
        metrics.append("beta_associated")
    for metric in metrics:
        if metric not in accepted:
            accepted[metric] = np.nan
        accepted[metric] = pd.to_numeric(accepted[metric], errors="coerce")
    # Recompute ratios from fractions to prevent injected finite ratios at zero.
    alpha_values, beta_values = accepted["alpha_helix"], accepted["beta_sheet"]
    accepted["alpha_beta_ratio"] = (alpha_values / beta_values).where(beta_values > 0)
    if "log_alpha_beta" in metrics:
        with np.errstate(divide="ignore", invalid="ignore"):
            accepted["log_alpha_beta"] = np.log(alpha_values / beta_values).where((alpha_values > 0) & (beta_values > 0))
    if "beta_associated" in metrics:
        accepted["beta_associated"] = accepted["beta_sheet"] + accepted["beta_turn"]

    descriptive_rows, test_rows = [], []
    min_ci = max(2, int(settings.get("min_n_ci", 3)))
    for metric in metrics:
        samples = []
        for group in groups:
            values = accepted.loc[accepted["group"] == group, metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            samples.append(values)
            n = len(values)
            row = dict(metric=metric, group=group, n=n, mean=np.nan, sd=np.nan,
                       median=np.nan, q25=np.nan, q75=np.nan, iqr=np.nan, mean_ci_low=np.nan,
                       mean_ci_high=np.nan, mean_ci_method="Unavailable",
                       median_ci_low=np.nan, median_ci_high=np.nan,
                       median_ci_method="Not requested", status="No accepted finite values")
            if n:
                row.update(mean=float(values.mean()), median=float(np.median(values)),
                           q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                           sd=float(values.std(ddof=1)) if n > 1 else np.nan,
                           status="OK" if n >= min_ci else "Insufficient n for mean CI")
                row["iqr"] = row["q75"] - row["q25"]
                if n >= min_ci:
                    half = float(stats.t.ppf(.975, n - 1) * values.std(ddof=1) / math.sqrt(n))
                    row.update(mean_ci_low=row["mean"] - half, mean_ci_high=row["mean"] + half,
                               mean_ci_method="Student t 95%; assumes independent approximately normal animal values")
                if settings.get("median_ci_bootstrap", False) and n >= min_ci and iterations > 0:
                    rng = np.random.default_rng(_seed(config, metric, group, "median"))
                    medians = []
                    for start in range(0, iterations, 256):
                        draw = values[rng.integers(n, size=(min(256, iterations-start), n))]
                        medians.extend(np.median(draw, axis=1))
                    low, high = np.quantile(medians, [.025, .975])
                    row.update(median_ci_low=float(low), median_ci_high=float(high),
                               median_ci_method="Animal bootstrap percentile 95%")
            descriptive_rows.append(row)
        hypothesis = "omnibus" if len(groups) > 2 else "two_group"
        omnibus = _test_row(metric, groups, samples, method, config, hypothesis)
        test_rows.append(omnibus)
        if len(groups) > 2:
            for first, second in itertools.combinations(range(len(groups)), 2):
                test_rows.append(_test_row(metric, [groups[first], groups[second]],
                                           [samples[first], samples[second]], method,
                                           config, "posthoc"))
    tests = pd.DataFrame(test_rows, columns=TEST_COLUMNS)
    tests["q_value"] = benjamini_hochberg(tests["p_value"].to_numpy(dtype=float))
    tests["interpretation_allowed"] = tests["p_value"].notna()
    if settings.get("posthoc_only_if_omnibus_significant", True):
        for metric in metrics:
            omnibus = tests.loc[(tests["metric"] == metric) & (tests["hypothesis"] == "omnibus")]
            if omnibus.empty:
                continue
            q_value = omnibus["q_value"].iloc[0]
            allowed = np.isfinite(q_value) and q_value < alpha
            blocked = (tests["metric"] == metric) & (tests["hypothesis"] == "posthoc")
            tests.loc[blocked, "interpretation_allowed"] &= allowed
            if not allowed:
                inferable = blocked & tests["p_value"].notna()
                tests.loc[inferable, "status"] = "gated: omnibus not significant"
    return pd.DataFrame(descriptive_rows, columns=DESCRIPTIVE_COLUMNS), tests
