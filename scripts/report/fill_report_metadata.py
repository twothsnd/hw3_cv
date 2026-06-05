#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill report author, GitHub, and weights metadata.")
    parser.add_argument("--metadata", type=Path, default=Path("report/metadata.json"))
    parser.add_argument("--report", type=Path, default=Path("report/main.tex"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Metadata file not found: {path}. Copy report/metadata.example.json first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("members"):
        raise SystemExit("metadata.members must contain at least one member.")
    for field in ["github_url", "weights_url"]:
        if not data.get(field):
            raise SystemExit(f"metadata.{field} is required.")
    return data


def build_author_block(members: list[dict[str, str]]) -> str:
    author_lines = []
    split_parts = []
    for index, member in enumerate(members, start=1):
        name = latex_escape(str(member.get("name", "")).strip())
        student_id = latex_escape(str(member.get("id", "")).strip())
        contribution = latex_escape(str(member.get("contribution", "")).strip())
        if not name or not student_id:
            raise SystemExit(f"Member {index} must have nonempty name and id.")
        author_lines.append(f"{name}, {student_id} \\\\")
        if contribution:
            split_parts.append(f"{name}: {contribution}")
    if split_parts:
        author_lines.append(r"\textit{Work split: " + "; ".join(split_parts) + "}")
    else:
        author_lines.append(r"\textit{Work split: all members contributed to implementation, experiments, and report writing}")
    return "\\author{\n" + "\n".join(author_lines) + "\n}"


def replace_single(pattern: str, repl: str, text: str, label: str) -> str:
    new_text, count = re.subn(pattern, lambda _match: repl, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise SystemExit(f"Could not replace {label}; pattern matched {count} times.")
    return new_text


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    text = args.report.read_text(encoding="utf-8")
    text = replace_single(r"\\author\{.*?\}\n\\date", build_author_block(metadata["members"]) + "\n\\date", text, "author block")
    text = replace_single(
        r"\\textbf\{GitHub:\} \\url\{.*?\}",
        r"\textbf{GitHub:} \url{" + str(metadata["github_url"]).strip() + "}",
        text,
        "GitHub URL",
    )
    text = replace_single(
        r"\\textbf\{Model weights:\} \\url\{.*?\}",
        r"\textbf{Model weights:} \url{" + str(metadata["weights_url"]).strip() + "}",
        text,
        "weights URL",
    )
    if args.dry_run:
        print(text[:1000])
        return
    args.report.write_text(text, encoding="utf-8")
    print(f"Updated {args.report}")


if __name__ == "__main__":
    main()
