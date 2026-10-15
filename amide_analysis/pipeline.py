"""Folder-to-report workflow with auditable per-spectrum and per-animal outcomes."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import traceback

import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import scipy

from . import __version__
from .config import load_config
from .fitting import STRUCTURES, analyze_band
from .io import discover_inputs, read_spectrum
from .plotting import plot_groups, plot_spectrum
from .preprocess import preprocess_spectrum
from .reports import export_results
from .regions import analyze_regions
from .statistics import aggregate_animals, perform_statistics


def _processing_variants(raw, config):
    """Vary baseline assumptions and derivative windows, without forcing peak count."""
    configurations = []
    alternate = deepcopy(config)
    alternate["baseline"]["method"] = "als" if config["baseline"]["method"] == "linear" else "linear"
    configurations.append(("alternate_baseline", alternate))
    if config["baseline"]["method"] == "linear":
        endpoints = deepcopy(config)
        endpoints["baseline"]["edge_points"] += 2
        configurations.append(("linear_additional_endpoint_points", endpoints))
    # The smaller window is omitted; derivatives are independently recalculated.
    windows = deepcopy(config)
    source_windows = config["smoothing"]["derivative_windows_cm1"]
    if len(source_windows) > 2:
        windows["smoothing"]["derivative_windows_cm1"] = sorted(source_windows)[1:]
    else:
        max_width = (config["amide_bounds"][1] - config["amide_bounds"][0]) * config["smoothing"]["max_window_fraction"]
        windows["smoothing"]["derivative_windows_cm1"] = sorted(set(
            min(max_width, value * 1.15) for value in source_windows))
    configurations.append(("alternative_derivative_windows", windows))
    variants, failures = [], []
    for label, settings in configurations:
        try:
            processed = preprocess_spectrum(raw["x"], raw["y"], settings)
        except ValueError as exc:
            failures.append(f"Проверка чувствительности {label} не выполнена: {exc}")
        else:
            variants.append({"label": label, "processed": processed})
    return variants, failures


def _seed(config, spectrum_id):
    payload = f"{config['seed']}\0{spectrum_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _empty_frame(rows, columns):
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def run_analysis(input_dir, output_dir=None, config=None, metadata_path=None, progress=print):
    """Analyze real files; rejected fits remain missing in biological statistics.

    This function never fabricates spectra or infers repeated animals from file
    names. Unknown instrument resolution keeps all percentages diagnostic.
    """
    config = load_config(config)
    input_dir = Path(input_dir).resolve()
    metadata = discover_inputs(input_dir, metadata_path)
    if output_dir is None:
        output_dir = input_dir.parent / ("Amide_results_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output_dir = Path(output_dir).resolve()
    if output_dir == input_dir:
        raise ValueError("Выходная папка должна отличаться от папки исходных спектров")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Выходная папка уже содержит файлы: {output_dir}. Выберите новую папку.")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    (output_dir / "processing_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log = []

    def say(text):
        log.append(text)
        progress(text)

    say(f"FTIR-анализ: версия {__version__}")
    say(f"Папка результатов: {output_dir}")
    say(f"Итоговый Excel будет сохранён: {output_dir / 'Amide_I_Results.xlsx'}")
    say(f"Найдено спектров: {len(metadata)}; животных: {metadata['animal_id'].nunique()}; "
        f"групп: {metadata['group'].nunique()}.")
    if not config["resolution_confirmed"]:
        say("Приборное разрешение не подтверждено. Это ограничение протокола, "
            "а не оценка качества спектра. Диагностические площади будут сохранены "
            "в Diagnostic_Structure; они не войдут в групповой статистический анализ.")
    else:
        say(f"Подтверждённое приборное разрешение: {config['spectral_resolution_cm1']:g} см⁻¹. "
            "Шаг экспортированных точек проверяется отдельно.")
    rows = {name: [] for name in ("Raw_Spectra", "Processed_Spectra", "Second_Derivative",
        "Peak_Parameters", "Fit_Quality", "Spectrum_Results", "Derivative_Candidates",
        "Model_Comparison", "Sensitivity", "Unreliable_Spectra", "Diagnostic_Structure",
        "Bootstrap_Diagnostics", "Model_Estimates")}
    figures = []
    regional_sources = []
    readable_spectra = calculated_fits = numerically_stable_spectra = 0
    metadata = metadata.copy()
    for number, record in enumerate(metadata.to_dict("records"), start=1):
        spectrum_started = perf_counter()
        spectrum_id, path = record["spectrum_id"], Path(record["file"])
        say(f"[{number}/{len(metadata)}] {path.name}")
        raw = processed = fit = None
        quality = {"spectrum_id": spectrum_id, "file": str(path), "animal_id": record["animal_id"],
                   "group": record["group"], "status": "Failed", "warnings": "",
                   "numerical_fit_reliable": False, "quantitative_accepted": False,
                   "fit_rejection_codes": "[]", "protocol_blockers": "[]"}
        result = {"spectrum_id": spectrum_id, "status": "Failed", **{key: np.nan for key in STRUCTURES}}
        messages = []
        try:
            metadata.loc[metadata["spectrum_id"] == spectrum_id, "sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            raw = read_spectrum(path)
            readable_spectra += 1
            regional_sources.append({"spectrum_id": spectrum_id, "x": raw["x"], "y": raw["y"]})
            messages.extend(raw["warnings"])
            original = raw.get("raw_table", pd.DataFrame({"wavenumber": raw["x"], "absorbance": raw["y"]}))
            for item in original.to_dict("records"):
                rows["Raw_Spectra"].append({"spectrum_id": spectrum_id, **item})
            processed = preprocess_spectrum(raw["x"], raw["y"], config)
            messages.extend(processed["warnings"])
            variants, failures = _processing_variants(raw, config)
            processed["sensitivity_variants"] = variants
            say("  Данные прочитаны; выполняются подгонки моделей и проверки устойчивости.")
            fit = analyze_band(processed, config, _seed(config, spectrum_id),
                               progress=lambda message: say(f"  {message}"))
            baseline_adjustment = np.asarray(
                fit.get("baseline_adjustment", np.zeros_like(processed["x"])), dtype=float)
            if baseline_adjustment.shape != np.asarray(processed["x"]).shape:
                raise ValueError("Поправка совместно уточнённой базовой линии не совпадает с сеткой спектра")
            refined_baseline = (np.asarray(processed["baseline"])
                                + baseline_adjustment * processed["normalization_factor"])
            corrected_refined = np.asarray(processed["corrected"]) - baseline_adjustment
            gaussian_fit = np.asarray(fit["fit"]) - baseline_adjustment
            if failures:
                fit["status"] = "Unreliable fit"
                fit["warnings"].extend(failures)
                fit["numerical_fit_reliable"] = False
                fit["quantitative_accepted"] = False
                fit.setdefault("fit_rejection_codes", []).append("SENSITIVITY_VARIANT_UNAVAILABLE")
                fit["percentages"] = {key: np.nan for key in STRUCTURES}
                for component in fit["components"]:
                    component["accepted"] = False
                    component["area_pct"] = np.nan
            numeric_reliable = bool(fit.get("numerical_fit_reliable", fit["status"] == "Reliable fit"))
            quantitative_accepted = bool(fit.get("quantitative_accepted", fit["status"] == "Reliable fit")
                                         and numeric_reliable and config["resolution_confirmed"])
            rejection_codes = list(dict.fromkeys(fit.get("fit_rejection_codes", [])))
            protocol_blockers = list(dict.fromkeys(fit.get("protocol_blockers", [])))
            if not config["resolution_confirmed"] and not protocol_blockers:
                protocol_blockers.append("INSTRUMENT_RESOLUTION_UNCONFIRMED")
            # Only scientifically accepted percentages can enter animal aggregation.
            if not quantitative_accepted:
                if fit["status"] == "Reliable fit":
                    fit["status"] = "Diagnostic fit" if numeric_reliable else "Unreliable fit"
                fit["percentages"] = {key: np.nan for key in STRUCTURES}
            else:
                fit["status"] = "Reliable fit"
            numerically_stable_spectra += int(numeric_reliable)
            if fit["metrics"].get("components", 0) and np.isfinite(fit["fit"]).all():
                calculated_fits += 1
            messages.extend(fit["warnings"])
            quality.update({"status": fit["status"], "raw_rows": raw["raw_rows"],
                "cleaned_rows": raw["cleaned_rows"], "band_points": len(processed["x"]),
                "grid_step_cm1": processed["step"], "noise_sd": processed["noise_sd"],
                "source_step_cm1": processed["source_step"],
                "normalization_factor": processed["normalization_factor"], **fit["metrics"],
                "baseline_coefficients": json.dumps(
                    np.asarray(fit.get("baseline_coefficients", []), dtype=float).tolist()),
                "model_constraint_details": json.dumps(fit.get("model_constraint_details", []), ensure_ascii=False),
                "numerical_fit_reliable": numeric_reliable,
                "quantitative_accepted": quantitative_accepted,
                "fit_rejection_codes": json.dumps(rejection_codes, ensure_ascii=False),
                "protocol_blockers": json.dumps(protocol_blockers, ensure_ascii=False)})
            result.update({"status": fit["status"], **fit["percentages"]})
            rows["Model_Estimates"].append({
                "spectrum_id": spectrum_id, "animal_id": record["animal_id"], "group": record["group"],
                "status": fit["status"], "quantitative_accepted": quantitative_accepted,
                "estimate_basis": "Diagnostic fitted spectral areas with nominal center assignment; these are not validated protein-structure percentages",
                **{key: fit.get("nominal_percentages", {}).get(key, np.nan) for key in STRUCTURES},
                "bootstrap_success_fraction": fit["metrics"].get("bootstrap_success_fraction", np.nan),
                "fit_rejection_codes": json.dumps(rejection_codes, ensure_ascii=False)})
            rows["Diagnostic_Structure"].append({
                "spectrum_id": spectrum_id, "animal_id": record["animal_id"], "group": record["group"],
                "percentage_basis": "Diagnostic only — not accepted for biological inference; relative fitted spectral areas",
                **{key: fit.get("diagnostic_percentages", {}).get(key, np.nan) for key in STRUCTURES},
                "numerical_fit_reliable": numeric_reliable, "quantitative_accepted": quantitative_accepted,
                "fit_rejection_codes": json.dumps(rejection_codes, ensure_ascii=False),
                "protocol_blockers": json.dumps(protocol_blockers, ensure_ascii=False)})
            for item in fit.get("bootstrap_diagnostics", []):
                rows["Bootstrap_Diagnostics"].append({**item, "spectrum_id": spectrum_id,
                    "animal_id": record["animal_id"], "group": record["group"]})
            for index, x in enumerate(processed["x"]):
                rows["Processed_Spectra"].append({"spectrum_id": spectrum_id, "wavenumber": x,
                    **{key: processed[key][index] for key in ("raw", "baseline", "corrected", "smoothed")},
                    "baseline_adjustment": baseline_adjustment[index],
                    "refined_baseline": refined_baseline[index],
                    "corrected_refined": corrected_refined[index],
                    "gaussian_fit": gaussian_fit[index],
                    "fit": fit["fit"][index], "residual": fit["residuals"][index]})
                for window, derivative in processed["derivatives"].items():
                    rows["Second_Derivative"].append({"spectrum_id": spectrum_id, "wavenumber": x,
                        "window_points": window, "window_cm1": window * processed["step"],
                        "second_derivative": derivative[index]})
            for key in ("components", "candidates", "model_comparison", "sensitivity"):
                target = {"components": "Peak_Parameters", "candidates": "Derivative_Candidates",
                          "model_comparison": "Model_Comparison", "sensitivity": "Sensitivity"}[key]
                for item in fit[key]:
                    rows[target].append({"spectrum_id": spectrum_id, **item})
            say(f"  {fit['status']}; компонентов: {fit['metrics'].get('components', 0)}; "
                f"численно устойчиво: {'да' if numeric_reliable else 'нет'}; "
                f"допущено к количественной статистике: {'да' if quantitative_accepted else 'нет'}.")
            if rejection_codes:
                say("  Причины отклонения модели: " + ", ".join(rejection_codes))
            for detail in fit.get("model_constraint_details", []):
                value = detail.get("value")
                if isinstance(value, (int, float)):
                    say(f"    Компонент {detail.get('component', '—')}: {detail.get('parameter')} = {value:g}; "
                        f"допустимый диапазон [{detail.get('lower')}, {detail.get('upper')}].")
            if protocol_blockers:
                say("  Ограничения протокола: " + ", ".join(protocol_blockers))
        except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
            messages.append(str(exc))
            quality["fit_rejection_codes"] = json.dumps(["SPECTRUM_PROCESSING_FAILED"])
            if not config["resolution_confirmed"]:
                quality["protocol_blockers"] = json.dumps(["INSTRUMENT_RESOLUTION_UNCONFIRMED"])
            say(f"  Failed: {exc}")
            (output_dir / f"error_{number:04d}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        quality["warnings"] = " | ".join(dict.fromkeys(messages))
        quality["analysis_seconds"] = float(perf_counter() - spectrum_started)
        say(f"  Анализ данных спектра: {quality['analysis_seconds']:.1f} с "
            "(без сохранения графиков).")
        result["warnings"] = quality["warnings"]
        if result["status"] == "Failed":
            rows["Model_Estimates"].append({
                "spectrum_id": spectrum_id, "animal_id": record["animal_id"], "group": record["group"],
                "status": "Failed", "quantitative_accepted": False,
                "estimate_basis": "No model estimate: spectrum processing failed",
                **{key: np.nan for key in STRUCTURES}, "fit_rejection_codes": quality["fit_rejection_codes"]})
            rows["Diagnostic_Structure"].append({
                "spectrum_id": spectrum_id, "animal_id": record["animal_id"], "group": record["group"],
                "percentage_basis": "No diagnostic percentages — spectrum processing failed",
                **{key: np.nan for key in STRUCTURES}, "numerical_fit_reliable": False,
                "quantitative_accepted": False, "fit_rejection_codes": quality["fit_rejection_codes"],
                "protocol_blockers": quality["protocol_blockers"]})
        rows["Fit_Quality"].append(quality)
        rows["Spectrum_Results"].append(result)
        if result["status"] != "Reliable fit":
            rows["Unreliable_Spectra"].append({"spectrum_id": spectrum_id, "file": str(path),
                "animal_id": record["animal_id"], "group": record["group"],
                "status": result["status"], "reason": quality["warnings"],
                "fit_rejection_codes": quality["fit_rejection_codes"],
                "protocol_blockers": quality["protocol_blockers"]})
        if raw is not None:
            try:
                figures.extend(plot_spectrum(raw, processed, fit, figures_dir, f"spectrum_{number:04d}", config))
            except (ValueError, OSError, RuntimeError) as exc:
                # A plotting failure must be explicit; it cannot alter biological values.
                say(f"  Ошибка сохранения графика: {exc}")
                quality["figure_warning"] = str(exc)
    spectrum_results = _empty_frame(rows["Spectrum_Results"], ["spectrum_id", "status", *STRUCTURES])
    animals, diagnostics = aggregate_animals(spectrum_results, metadata, config)
    descriptives, tests = perform_statistics(animals, config)
    say("Сравнение девяти заданных спектральных областей на уровне животных.")
    regional_tables = analyze_regions(regional_sources, metadata, config)
    tables = {"Metadata": metadata,
        "Raw_Spectra": _empty_frame(rows["Raw_Spectra"], ["spectrum_id", "source_row", "wavenumber", "absorbance"]),
        "Processed_Spectra": _empty_frame(rows["Processed_Spectra"],
            ["spectrum_id", "wavenumber", "raw", "baseline", "corrected", "smoothed",
             "baseline_adjustment", "refined_baseline", "corrected_refined", "gaussian_fit", "fit", "residual"]),
        "Second_Derivative": _empty_frame(rows["Second_Derivative"],
            ["spectrum_id", "wavenumber", "window_points", "window_cm1", "second_derivative"]),
        "Peak_Parameters": _empty_frame(rows["Peak_Parameters"],
            ["spectrum_id", "component", "center", "amplitude", "fwhm", "area", "area_pct", "structure", "accepted"]),
        "Secondary_Structure": animals, "Group_Statistics": descriptives, "Statistical_Tests": tests,
        "Fit_Quality": _empty_frame(rows["Fit_Quality"], ["spectrum_id", "status", "warnings"]),
        "Processing_Settings": pd.DataFrame([
            {"key": "run_utc", "value": datetime.now(timezone.utc).isoformat()},
            {"key": "input_folder", "value": str(input_dir)},
            {"key": "python", "value": platform.python_version()},
            {"key": "numpy", "value": np.__version__}, {"key": "pandas", "value": pd.__version__},
            {"key": "scipy", "value": scipy.__version__}, {"key": "matplotlib", "value": matplotlib.__version__},
            {"key": "openpyxl", "value": openpyxl.__version__},
            {"key": "software_version", "value": __version__},
            {"key": "interpretation", "value": "Relative spectral component areas, not absolute secondary-structure content"},
            {"key": "fit_model", "value": "Gaussian components plus a jointly profiled linear baseline; two baseline coefficients are nuisance parameters counted in AICc/BIC"},
            {"key": "processed_spectra_units", "value": "raw, baseline and refined_baseline are in original absorbance units; corrected, smoothed, baseline_adjustment, corrected_refined, gaussian_fit, fit and residual are divided by normalization_factor"},
            {"key": "processed_spectra_reconstruction", "value": "refined_baseline = baseline + baseline_adjustment * normalization_factor; corrected_refined = corrected - baseline_adjustment; gaussian_fit = fit - baseline_adjustment; raw = refined_baseline + normalization_factor * (gaussian_fit + residual)"},
            {"key": "regional_endpoint", "value": "Mean absorbance in each of nine predefined regions after discrete L2 normalization on the full common measured grid; independent of Amide I fit acceptance"},
            {"key": "regional_test_family", "value": "One two-sided Mann–Whitney test per eligible region (Kruskal–Wallis for more than two groups), BH correction across the nine predefined regional hypotheses; technical repeats averaged per animal"},
        ])}
    for key in ("Spectrum_Results", "Derivative_Candidates", "Model_Comparison", "Sensitivity", "Unreliable_Spectra"):
        tables[key] = _empty_frame(rows[key], ["spectrum_id", "status"])
    tables["Diagnostic_Structure"] = _empty_frame(rows["Diagnostic_Structure"],
        ["spectrum_id", "animal_id", "group", "percentage_basis", *STRUCTURES,
         "numerical_fit_reliable", "quantitative_accepted", "fit_rejection_codes", "protocol_blockers"])
    tables["Bootstrap_Diagnostics"] = _empty_frame(rows["Bootstrap_Diagnostics"],
        ["spectrum_id", "animal_id", "group", "iteration", "status"])
    tables["Animal_Diagnostics"] = diagnostics
    tables["Model_Estimates"] = _empty_frame(rows["Model_Estimates"],
        ["spectrum_id", "animal_id", "group", "status", "quantitative_accepted", "estimate_basis", *STRUCTURES])
    tables.update(regional_tables)
    reliable = int(animals["status"].eq("Reliable fit").sum())
    tested_regions = int(tables["Region_Statistics"]["Tested"].sum())
    significant_regions = int((tables["Region_Statistics"]["q_value_BH_FDR"] < config["statistics"]["alpha"]).sum())
    tables["Analysis_Summary"] = pd.DataFrame([
        {"Раздел": "Исходные спектры", "Результат": f"Прочитано {readable_spectra} из {len(metadata)}", "Таблица": "Raw_Spectra", "Пояснение": "Успех чтения оценивается отдельно от разложения на пики."},
        {"Раздел": "Сравнение девяти областей", "Результат": f"Проверено {tested_regions} из 9 областей", "Таблица": "Region_Summary", "Пояснение": "Медианы, p Манна–Уитни, p с поправкой BH и размер эффекта. Это показатели спектральных областей."},
        {"Раздел": "Различия областей после BH", "Результат": f"Значимы {significant_regions} из {tested_regions} проверенных областей", "Таблица": "Region_Statistics", "Пояснение": f"Единая поправка BH по заранее заданным областям; уровень значимости {config['statistics']['alpha']:g}."},
        {"Раздел": "Разложение Amide I", "Результат": f"Прошли проверки {reliable} из {len(animals)} животных", "Таблица": "Secondary_Structure", "Пояснение": "Только принятые модельные доли участвуют в сравнении структурных категорий."},
        {"Раздел": "Промежуточные оценки Amide I", "Результат": "Сохранены отдельно", "Таблица": "Model_Estimates", "Пояснение": "Диагностические модельные площади; статус и неопределённость приведены рядом. Не являются подтверждённым содержанием белковых структур."},
        {"Раздел": "Причины ограничений модели", "Результат": "Параметр, значение и предел", "Таблица": "Fit_Quality", "Пояснение": "Достижение настроенного предела не означает физическую невозможность спектра."},
        {"Раздел": "Единица статистического анализа", "Результат": f"По разметке: {metadata['animal_id'].nunique()} животных", "Таблица": "Metadata", "Пояснение": "Повторные измерения одной мыши должны иметь одинаковый animal_id. p-значения соответствуют этой разметке; без метаданных каждый файл считается отдельным животным."},
    ])
    say("Сохраняются Excel и CSV-таблицы, затем групповые графики.")
    excel = export_results(output_dir, tables, config)
    figures.extend(plot_groups(animals, tests, figures_dir, config))
    say(f"Прочитано спектров: {readable_spectra}/{len(metadata)}; "
        f"рассчитано разложений: {calculated_fits}/{len(metadata)}; "
        f"численно устойчивых разложений: {numerically_stable_spectra}/{len(metadata)}.")
    say(f"Сравнение спектральных областей: готово {tested_regions}/9 тестов; таблица Region_Summary.")
    say(f"Значимых различий областей после поправки BH: {significant_regions}/{tested_regions}.")
    say(f"Проверки декомпозиции Amide I прошли: {reliable}/{len(animals)} животных.")
    say(f"Excel сохранён: {excel}")
    say(f"Папка всех результатов: {output_dir}")
    say(f"Причины отклонения и ограничения протокола: {output_dir / 'Fit_Quality.csv'}")
    say(f"Диагностические площади: {output_dir / 'Diagnostic_Structure.csv'}")
    say(f"Подробности bootstrap: {output_dir / 'Bootstrap_Diagnostics.csv'}")
    say("Проценты характеризуют относительные площади спектральных компонентов; "
        "это не измерение абсолютного содержания структур конкретных белков.")
    (output_dir / "analysis_log.txt").write_text("\n".join(log), encoding="utf-8")
    return {"output_dir": output_dir, "excel": excel, "tables": tables, "figures": figures}
