#!/usr/bin/env python3
"""CLI and optional folder-picker entry point for Amide I analysis."""

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    if sys.version_info < (3, 11):
        print("Требуется Python 3.11 или новее.", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description="Анализ Amide I и относительных площадей компонентов FTIR")
    parser.add_argument("--input", type=Path, help="Папка исходных Excel/CSV спектров и подпапок групп")
    parser.add_argument("--output", type=Path, help="Новая или пустая выходная папка")
    parser.add_argument("--config", type=Path, help="JSON с параметрами обработки")
    parser.add_argument("--metadata", type=Path, help="metadata.xlsx/.xls/.csv: file, animal_id, group")
    parser.add_argument("--write-default-config", type=Path, metavar="FILE",
                        help="Сохранить настройки по умолчанию и завершить работу")
    args = parser.parse_args(argv)
    try:
        from amide_analysis.config import default_config, load_config
        if args.write_default_config:
            if args.write_default_config.exists():
                raise ValueError(f"Файл уже существует: {args.write_default_config}")
            args.write_default_config.parent.mkdir(parents=True, exist_ok=True)
            args.write_default_config.write_text(json.dumps(default_config(), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Настройки сохранены: {args.write_default_config}")
            return 0
        if args.input is None:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                try:
                    selected = filedialog.askdirectory(title="Выберите папку FTIR-спектров")
                finally:
                    root.destroy()
            except (ImportError, RuntimeError) as exc:
                raise ValueError("Диалог выбора недоступен. Запустите с --input ПАПКА_СПЕКТРОВ") from exc
            except tk.TclError as exc:
                raise ValueError("Диалог выбора недоступен. Запустите с --input ПАПКА_СПЕКТРОВ") from exc
            if not selected:
                print("Папка не выбрана.")
                return 0
            args.input = Path(selected)
        config_path = args.config
        if config_path is None:
            bundled = Path(__file__).with_name("analysis_config.json")
            config_path = bundled if bundled.exists() else None
        config = load_config(config_path)
        from amide_analysis.pipeline import run_analysis
        run_analysis(args.input, args.output, config, args.metadata)
        return 0
    except ImportError as exc:
        print(f"Не хватает зависимости: {exc.name}. Выполните: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
