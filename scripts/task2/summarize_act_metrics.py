#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ACT evaluation JSON files into a CSV table.")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten(f"{prefix}{key}.", nested, out)
    elif isinstance(value, (str, int, float)) or value is None:
        out[prefix[:-1]] = value


def main() -> None:
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise SystemExit("--inputs and --labels must have the same length.")

    rows: list[dict[str, Any]] = []
    fieldnames = {"label"}
    for label, path in zip(args.labels, args.inputs):
        data = json.loads(path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {"label": label}
        flatten("", data, row)
        rows.append(row)
        fieldnames.update(row.keys())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ordered = ["label"] + sorted(name for name in fieldnames if name != "label")
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
