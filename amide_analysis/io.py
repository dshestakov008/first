"""Strict FTIR import and explicit animal/technical-repeat metadata."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


_EXTENSIONS = {".xlsx", ".xls", ".csv"}
_OUTPUT_DIRS = {"output", "outputs", "results", "amide_results", "analysis_results"}
_METADATA_NAMES = {"metadata.csv", "metadata.xlsx", "metadata.xls"}
_COLUMNS = ["spectrum_id", "file", "animal_id", "group", "technical_repeat"]


def _blank(value) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _number(value) -> float:
    if _blank(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError("пустая ячейка или логическое значение")
    if isinstance(value, str):
        value = value.strip().replace("\u00a0", "").replace("\u202f", "")
        value = value.replace(" ", "").replace("−", "-").replace(",", ".")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("NaN или бесконечность")
    return result


def _x_header(value) -> bool:
    text = str(value).strip().lower()
    return text == "x" or any(word in text for word in
        ("wavenumber", "wave number", "волнов", "cm-1", "cm⁻¹", "см-1", "см⁻¹"))


def _y_header(value) -> bool:
    text = str(value).strip().lower()
    return text in {"y", "a", "abs"} or any(word in text for word in
        ("absorbance", "absorption", "signal", "a.u.", "поглощ", "оптическая плотность", "сигнал"))


def _unsupported_signal_header(value) -> str | None:
    text = str(value).strip().lower()
    if (re.fullmatch(r"%?t[\s,()%]*", text)
            or any(word in text for word in ("transmittance", "transmission", "пропуск"))):
        return "пропускание (Transmittance)"
    if text == "i" or any(word in text for word in ("intensity", "интенсив")):
        return "интенсивность (Intensity)"
    return None


def _explicit_absorbance_header(value) -> bool:
    text = str(value).strip().lower()
    return text in {"a", "abs"} or any(word in text for word in
        ("absorbance", "absorption", "поглощ", "оптическая плотность"))


def _read_csv_table(path: Path) -> pd.DataFrame:
    content = None
    for encoding in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            content = path.read_text(encoding=encoding)
            break
        except UnicodeError:
            continue
    if content is None:
        raise ValueError("не удалось определить кодировку CSV")
    sample = "\n".join(line for line in content.splitlines() if line.strip())[:65536]
    if not sample:
        return pd.DataFrame()
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        # Empty/missing fields can make Sniffer reject a valid delimited file.
        header = next(line for line in content.splitlines() if line.strip())
        counts = {character: header.count(character) for character in ";,\t|"}
        delimiter = max(counts, key=counts.get)
        if counts[delimiter] == 0:
            raise ValueError("CSV: не найден разделитель столбцов (; , табуляция или |)")
    rows = list(csv.reader(content.splitlines(), delimiter=delimiter, strict=True))
    nonempty_lengths = [len(row) for row in rows if row]
    if not nonempty_lengths:
        return pd.DataFrame()
    width = max(nonempty_lengths)
    # Missing trailing fields are blanks, not fabricated measurements.
    return pd.DataFrame([row + [""] * (width - len(row)) for row in rows], dtype=object)


def _read_table(path: Path) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if path.name.startswith("~$"):
        raise ValueError("временный файл Excel; выберите исходный файл")
    if suffix == ".csv":
        return _read_csv_table(path), "CSV"
    if suffix not in {".xlsx", ".xls"}:
        raise ValueError("поддерживаются только .xlsx, .xls и .csv")
    engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
    sheets = []
    with pd.ExcelFile(path, engine=engine) as workbook:
        for name in workbook.sheet_names:
            table = pd.read_excel(workbook, sheet_name=name, header=None,
                                  dtype=object, keep_default_na=False, na_filter=False)
            if not table.empty and table.map(lambda value: not _blank(value)).to_numpy().any():
                sheets.append((table, name))
    if len(sheets) != 1:
        raise ValueError("требуется ровно один непустой лист Excel")
    return sheets[0]


def read_spectrum(path) -> dict:
    """Read the first two occupied columns without silently accepting bad data.

    A recognized header may identify reversed x/y columns. Empty rows or missing
    values are removed with warnings; other nonnumeric values stop the import.
    Duplicate wavenumbers are averaged before sorting.
    """
    path = Path(path)
    try:
        table, sheet = _read_table(path)
        if table.empty:
            raise ValueError("нет данных")
        occupied = table.map(lambda value: not _blank(value))
        columns = list(table.columns[occupied.any(axis=0)])
        rows = list(table.index[occupied.any(axis=1)])
        if len(columns) < 2:
            raise ValueError("нужны два заполненных столбца: волновое число и сигнал")
        x_column, y_column = columns[:2]
        first_occupied = rows[0]
        first, last = 0, len(table) - 1
        a, b = table.loc[first_occupied, [x_column, y_column]]
        warnings = []
        for value in (a, b):
            signal_type = _unsupported_signal_header(value)
            if signal_type is not None:
                raise ValueError(
                    f"Этот анализ принимает только Absorbance, а заголовок обозначает {signal_type}. "
                    "Подайте экспорт поглощения (A); автоматическое преобразование сигнала не выполняется."
                )
        if len(columns) > 2:
            warnings.append(f"Прочитаны первые два заполненных столбца; остальные {len(columns) - 2} проигнорированы.")
        if _x_header(a) and _y_header(b):
            first = first_occupied + 1
            if not _explicit_absorbance_header(b):
                warnings.append("Тип сигнала в заголовке не указан: согласно требованиям входной сигнал принят за Absorbance (A).")
        elif _y_header(a) and _x_header(b):
            x_column, y_column = y_column, x_column
            first = first_occupied + 1
            warnings.append("Столбцы x/y определены по заголовкам и переставлены.")
            if not _explicit_absorbance_header(a):
                warnings.append("Тип сигнала в заголовке не указан: согласно требованиям входной сигнал принят за Absorbance (A).")
        else:
            warnings.append("Заголовок типа сигнала отсутствует: согласно требованиям входной сигнал принят за Absorbance (A).")
        raw_rows = max(0, last - first + 1)
        raw_table = pd.DataFrame({
            "source_row": np.arange(first + 1, last + 2, dtype=int),
            "wavenumber": list(table.loc[first:last, x_column]),
            "absorbance": list(table.loc[first:last, y_column]),
        })
        xs, ys, removed = [], [], []
        for row in range(first, last + 1):
            a, b = table.loc[row, [x_column, y_column]]
            if _blank(a) or _blank(b):
                # A nonblank bad value remains an error even if its partner is missing.
                for value in (a, b):
                    if not _blank(value):
                        try:
                            _number(value)
                        except (ValueError, TypeError, OverflowError) as exc:
                            raise ValueError(f"{sheet}, строка {row + 1}: {exc}") from exc
                removed.append(row + 1)
                continue
            try:
                xs.append(_number(a))
                ys.append(_number(b))
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"{sheet}, строка {row + 1}: нечисловое/неконечное значение ({exc})") from exc
        if removed:
            preview = ", ".join(map(str, removed[:10]))
            warnings.append(f"Удалены строки с пустыми/отсутствующими значениями: {len(removed)} (строки {preview}{'…' if len(removed) > 10 else ''}).")
        x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        if len(x) < 3:
            raise ValueError("после очистки осталось менее трёх точек")
        if np.any(x <= 0):
            raise ValueError("волновые числа должны быть положительными")
        order = np.argsort(x, kind="stable")
        if not np.array_equal(order, np.arange(len(x))):
            warnings.append("Точки упорядочены по возрастанию волнового числа.")
        x, y = x[order], y[order]
        unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        duplicates = int(np.sum(counts - 1))
        if duplicates:
            y = np.bincount(inverse, weights=y) / counts
            x = unique
            warnings.append(f"Повторяющиеся волновые числа объединены средним сигналом: {duplicates} дополнительных строк.")
        if len(x) < 3:
            raise ValueError("после объединения повторов осталось менее трёх волновых чисел")
        if np.any(y < 0):
            warnings.append("Отрицательные значения сигнала сохранены без изменений.")
        return {"x": x, "y": y, "warnings": warnings,
                "raw_rows": raw_rows, "cleaned_rows": len(x), "raw_table": raw_table}
    except Exception as exc:
        raise ValueError(f"Файл '{path.name}': {exc}") from exc


def _read_metadata(path: Path) -> pd.DataFrame:
    table, _ = _read_table(path)
    if table.empty:
        raise ValueError("метаданные пусты")
    occupied = table.map(lambda value: not _blank(value))
    rows = list(table.index[occupied.any(axis=1)])
    columns = list(table.columns[occupied.any(axis=0)])
    header = [str(table.at[rows[0], column]).strip().lower() for column in columns]
    if len(set(header)) != len(header):
        raise ValueError("метаданные: повторяющиеся названия столбцов")
    missing = {"file", "animal_id", "group"} - set(header)
    if missing:
        raise ValueError(f"метаданные: нет столбцов {', '.join(sorted(missing))}")
    data = table.loc[rows[1:], columns].copy()
    data.columns = header
    return data.reset_index(drop=True)


def discover_inputs(folder, metadata_path=None) -> pd.DataFrame:
    """Find spectra and associate repeats only through explicit metadata.

    Files without metadata are separate animals, even when names look similar.
    Spectrum IDs are POSIX paths relative to the input folder, including suffix.
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Каталог входных данных не найден: {folder}")
    metadata = None
    if metadata_path is not None:
        # Explicit CLI paths follow normal shell conventions (relative to cwd).
        metadata = Path(metadata_path).expanduser().resolve()
        if not metadata.is_file():
            raise ValueError(f"Файл метаданных не найден: {metadata}")
    else:
        choices = [path for path in folder.iterdir()
                   if path.is_file() and path.name.lower() in _METADATA_NAMES]
        if len(choices) > 1:
            raise ValueError("Найдено несколько metadata-файлов; задайте --metadata явно")
        if choices:
            metadata = choices[0].resolve()
    files = []
    for directory, subdirectories, names in os.walk(folder, followlinks=False):
        directory = Path(directory)
        # Prune recognized outputs before walking into their exported CSVs.
        subdirectories[:] = [name for name in subdirectories
                             if name.lower() not in _OUTPUT_DIRS
                             and not name.lower().startswith("amide_results_")
                             and not ((directory / name / "processing_config.json").is_file()
                                      and (directory / name / "Amide_I_Results.xlsx").is_file())]
        for name in names:
            path = directory / name
            if not path.is_file() or path.suffix.lower() not in _EXTENSIONS:
                continue
            relative = path.relative_to(folder)
            if path.name.startswith("~$") or path.name.lower() in _METADATA_NAMES:
                continue
            if metadata is not None and path.resolve() == metadata:
                continue
            if not path.resolve().is_relative_to(folder):
                raise ValueError(f"Спектр через символическую ссылку находится вне входного каталога: {relative}")
            files.append(path)
    files.sort(key=lambda path: path.relative_to(folder).as_posix())
    if not files:
        raise ValueError("Входной каталог не содержит спектров .xlsx, .xls или .csv")
    resolved = {path.resolve(): path for path in files}
    if len(resolved) != len(files):
        raise ValueError("Один исходный файл доступен по нескольким путям/ссылкам")
    mappings = {}
    if metadata is not None:
        data = _read_metadata(metadata)
        for n, row in data.iterrows():
            required = {key: row[key] for key in ("file", "animal_id", "group")}
            if any(_blank(value) for value in required.values()):
                raise ValueError(f"Метаданные, строка {n + 2}: пустой file, animal_id или group")
            value = str(row["file"]).strip().replace("\\", "/")
            candidate = Path(value)
            candidate = (candidate if candidate.is_absolute() else folder / candidate).resolve()
            if not candidate.is_relative_to(folder):
                raise ValueError(f"Метаданные, строка {n + 2}: путь вне входного каталога")
            if candidate in mappings:
                raise ValueError(f"Метаданные: повторяющийся путь {value}")
            repeat = row.get("technical_repeat", "")
            mappings[candidate] = {
                "animal_id": str(row["animal_id"]).strip(),
                "group": str(row["group"]).strip(),
                "technical_repeat": "" if _blank(repeat) else str(repeat).strip(),
            }
        missing = set(resolved) - set(mappings)
        extra = set(mappings) - set(resolved)
        if missing or extra:
            details = []
            if missing:
                details.append("нет соответствий для: " + ", ".join(sorted(path.relative_to(folder).as_posix() for path in missing)))
            if extra:
                details.append("лишние/несуществующие файлы: " + ", ".join(sorted(path.relative_to(folder).as_posix() for path in extra)))
            raise ValueError("Метаданные должны покрывать ровно все спектры; " + "; ".join(details))
    records = []
    for path in files:
        relative = path.relative_to(folder)
        identity = relative.as_posix()
        item = mappings.get(path.resolve(), {
            "animal_id": relative.with_suffix("").as_posix(),
            "group": relative.parts[0] if len(relative.parts) > 1 else "Ungrouped",
            "technical_repeat": "1",
        })
        records.append({"spectrum_id": identity, "file": str(path.resolve()), **item})
    result = pd.DataFrame.from_records(records, columns=_COLUMNS)
    groups = result.groupby("animal_id", sort=False)["group"].nunique()
    bad = groups[groups > 1]
    if not bad.empty:
        raise ValueError("Одна мышь указана в нескольких группах: " + ", ".join(map(str, bad.index)))
    # Optional repeat labels must be unique for each animal when supplied.
    supplied = result[result["technical_repeat"] != ""]
    duplicates = supplied.duplicated(["animal_id", "technical_repeat"], keep=False)
    if duplicates.any():
        raise ValueError("Повторяющиеся technical_repeat для одной мыши; исправьте метаданные")
    # Different extensions with identical stems must not silently create repeats.
    if metadata is None and result["animal_id"].duplicated().any():
        raise ValueError("Совпали animal_id по относительным именам файлов; нужны явные метаданные")
    return result
