"""Folder-to-report workflow with auditable per-spectrum and per-animal outcomes."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import traceback

import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import scipy

from .config import load_config
from .fitting import STRUCTURES, analyze_band
from .io import discover_inputs, read_spectrum
from .plotting import plot_groups, plot_spectrum
from .preprocess import preprocess_spectrum
from .reports import export_results
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

    say(f"Найдено спектров: {len(metadata)}; животных: {metadata['animal_id'].nunique()}; "
        f"групп: {metadata['group'].nunique()}.")
    if not config["resolution_confirmed"]:
        say("Приборное разрешение неизвестно. Разложения будут диагностическими; "
            "их проценты не войдут в групповой статистический анализ.")
    rows = {name: [] for name in ("Raw_Spectra", "Processed_Spectra", "Second_Derivative",
        "Peak_Parameters", "Fit_Quality", "Spectrum_Results", "Derivative_Candidates",
        "Model_Comparison", "Sensitivity", "Unreliable_Spectra")}
    figures = []
    metadata = metadata.copy()
    for number, record in enumerate(metadata.to_dict("records"), start=1):
        spectrum_id, path = record["spectrum_id"], Path(record["file"])
        say(f"[{number}/{len(metadata)}] {path.name}")
        raw = processed = fit = None
        quality = {"spectrum_id": spectrum_id, "file": str(path), "animal_id": record["animal_id"],
                   "group": record["group"], "status": "Failed", "warnings": ""}
        result = {"spectrum_id": spectrum_id, "status": "Failed", **{key: np.nan for key in STRUCTURES}}
        messages = []
        try:
            metadata.loc[metadata["spectrum_id"] == spectrum_id, "sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            raw = read_spectrum(path)
            messages.extend(raw["warnings"])
            original = raw.get("raw_table", pd.DataFrame({"wavenumber": raw["x"], "absorbance": raw["y"]}))
            for item in original.to_dict("records"):
                rows["Raw_Spectra"].append({"spectrum_id": spectrum_id, **item})
            processed = preprocess_spectrum(raw["x"], raw["y"], config)
            messages.extend(processed["warnings"])
            variants, failures = _processing_variants(raw, config)
            processed["sensitivity_variants"] = variants
            fit = analyze_band(processed, config, _seed(config, spectrum_id))
            if failures:
                fit["status"] = "Unreliable fit"
                fit["warnings"].extend(failures)
                fit["percentages"] = {key: np.nan for key in STRUCTURES}
                for component in fit["components"]:
                    component["accepted"] = False
                    component["area_pct"] = np.nan
            messages.extend(fit["warnings"])
            quality.update({"status": fit["status"], "raw_rows": raw["raw_rows"],
                "cleaned_rows": raw["cleaned_rows"], "band_points": len(processed["x"]),
                "grid_step_cm1": processed["step"], "noise_sd": processed["noise_sd"],
                "source_step_cm1": processed["source_step"],
                "normalization_factor": processed["normalization_factor"], **fit["metrics"]})
            result.update({"status": fit["status"], **fit["percentages"]})
            for index, x in enumerate(processed["x"]):
                rows["Processed_Spectra"].append({"spectrum_id": spectrum_id, "wavenumber": x,
                    **{key: processed[key][index] for key in ("raw", "baseline", "corrected", "smoothed")},
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
            say(f"  {fit['status']}; компонентов: {fit['metrics'].get('components', 0)}.")
        except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
            messages.append(str(exc))
            say(f"  Failed: {exc}")
            (output_dir / f"error_{number:04d}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        quality["warnings"] = " | ".join(dict.fromkeys(messages))
        result["warnings"] = quality["warnings"]
        rows["Fit_Quality"].append(quality)
        rows["Spectrum_Results"].append(result)
        if result["status"] != "Reliable fit":
            rows["Unreliable_Spectra"].append({"spectrum_id": spectrum_id, "file": str(path),
                "animal_id": record["animal_id"], "group": record["group"],
                "status": result["status"], "reason": quality["warnings"]})
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
    tables = {"Metadata": metadata,
        "Raw_Spectra": _empty_frame(rows["Raw_Spectra"], ["spectrum_id", "source_row", "wavenumber", "absorbance"]),
        "Processed_Spectra": _empty_frame(rows["Processed_Spectra"],
            ["spectrum_id", "wavenumber", "raw", "baseline", "corrected", "smoothed", "fit", "residual"]),
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
            {"key": "software_version", "value": "1.0.0"},
            {"key": "interpretation", "value": "Relative spectral component areas, not absolute secondary-structure content"},
        ])}
    for key in ("Spectrum_Results", "Derivative_Candidates", "Model_Comparison", "Sensitivity", "Unreliable_Spectra"):
        tables[key] = _empty_frame(rows[key], ["spectrum_id", "status"])
    tables["Animal_Diagnostics"] = diagnostics
    excel = export_results(output_dir, tables, config)
    figures.extend(plot_groups(animals, tests, figures_dir, config))
    reliable = int(animals["status"].eq("Reliable fit").sum())
    say(f"Пригодные животные: {reliable}/{len(animals)}. Excel: {excel}")
    say("Проценты характеризуют относительные площади спектральных компонентов; "
        "это не измерение абсолютного содержания структур конкретных белков.")
    (output_dir / "analysis_log.txt").write_text("\n".join(log), encoding="utf-8")
    return {"output_dir": output_dir, "excel": excel, "tables": tables, "figures": figures}
