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
    parser.add_argument("--notebook", type=Path, default=Path("report/HW3_report.ipynb"))
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


def build_notebook_member_line(members: list[dict[str, str]]) -> str:
    parts = []
    for member in members:
        name = str(member.get("name", "")).strip()
        student_id = str(member.get("id", "")).strip()
        if not name or not student_id:
            raise SystemExit("Every notebook member must have nonempty name and id.")
        parts.append(f"{name}（{student_id}）")
    return "、".join(parts)


def build_notebook_split_line(members: list[dict[str, str]]) -> str:
    parts = []
    for member in members:
        name = str(member.get("name", "")).strip()
        contribution = str(member.get("contribution", "")).strip()
        if contribution:
            parts.append(f"{name}: {contribution}")
    return "；".join(parts) if parts else "全体成员共同完成实现、实验、报告与整理"


def replace_single(pattern: str, repl: str, text: str, label: str) -> str:
    new_text, count = re.subn(pattern, lambda _match: repl, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise SystemExit(f"Could not replace {label}; pattern matched {count} times.")
    return new_text


def fill_latex_report(path: Path, metadata: dict[str, Any]) -> str:
    text = path.read_text(encoding="utf-8")
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
    return text


def fill_notebook_report(path: Path, metadata: dict[str, Any]) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    first = notebook["cells"][0]
    if first.get("cell_type") != "markdown":
        raise SystemExit("The first notebook cell must be markdown metadata.")
    source = first["source"]
    if isinstance(source, list):
        text = "".join(source)
    else:
        text = source
    text = replace_single(r"\*\*姓名 / 学号：\*\* .*\n", f"**姓名 / 学号：** {build_notebook_member_line(metadata['members'])}\n", text, "notebook members")
    text = replace_single(r"\*\*成员分工：\*\* .*\n", f"**成员分工：** {build_notebook_split_line(metadata['members'])}\n", text, "notebook work split")
    text = replace_single(r"\*\*GitHub 仓库：\*\* .*\n", f"**GitHub 仓库：** {str(metadata['github_url']).strip()}\n", text, "notebook GitHub URL")
    text = replace_single(r"\*\*模型权重云盘链接：\*\* .*\n", f"**模型权重云盘链接：** {str(metadata['weights_url']).strip()}\n", text, "notebook weights URL")
    first["source"] = text
    return json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    outputs: list[tuple[Path, str]] = []
    if args.report.exists():
        outputs.append((args.report, fill_latex_report(args.report, metadata)))
    if args.notebook.exists():
        outputs.append((args.notebook, fill_notebook_report(args.notebook, metadata)))
    if args.dry_run:
        for path, text in outputs:
            print(f"--- {path} ---")
            print(text[:1000])
        return
    for path, text in outputs:
        path.write_text(text, encoding="utf-8")
        print(f"Updated {path}")


if __name__ == "__main__":
    main()
