from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | os.PathLike[str], root: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or project_root()) / p


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(p)


def which_or_raise(executable: str) -> str:
    found = shutil.which(executable)
    if not found:
        raise SystemExit(
            f"Required executable '{executable}' was not found in PATH. "
            "Install it or pass an explicit executable path."
        )
    return found


def run_cmd(
    cmd: list[str],
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> float:
    printable = " ".join(cmd)
    print(f"[run] {printable}")
    if dry_run:
        return 0.0
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, env=merged_env, check=True)
    return time.perf_counter() - start


def latest_child(path: str | os.PathLike[str], glob_pattern: str = "*") -> Path:
    candidates = sorted(Path(path).glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No files matched {Path(path) / glob_pattern}")
    return candidates[-1]


def parse_key_value_items(items: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Expected KEY=VALUE item, got: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"Empty key in item: {item}")
        parsed[key] = value
    return parsed


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve()))
    except ValueError:
        return str(path.resolve())
