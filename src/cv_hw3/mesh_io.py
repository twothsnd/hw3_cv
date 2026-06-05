from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from plyfile import PlyData
except Exception:  # pragma: no cover - optional dependency fallback
    PlyData = None


def _empty_bbox() -> dict[str, list[float] | None]:
    return {"min": None, "max": None}


def _update_bbox(bbox: dict[str, list[float] | None], xyz: tuple[float, float, float]) -> None:
    if bbox["min"] is None:
        bbox["min"] = [xyz[0], xyz[1], xyz[2]]
        bbox["max"] = [xyz[0], xyz[1], xyz[2]]
        return
    mins = bbox["min"]
    maxs = bbox["max"]
    assert mins is not None and maxs is not None
    for i, value in enumerate(xyz):
        mins[i] = min(mins[i], value)
        maxs[i] = max(maxs[i], value)


def _bbox_size(bbox: dict[str, list[float] | None]) -> list[float] | None:
    mins = bbox["min"]
    maxs = bbox["max"]
    if mins is None or maxs is None:
        return None
    return [maxs[i] - mins[i] for i in range(3)]


def read_obj_stats(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "format": "obj",
        "vertices": 0,
        "texture_vertices": 0,
        "normals": 0,
        "faces": 0,
        "bbox": _empty_bbox(),
    }
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
                    _update_bbox(stats["bbox"], xyz)
                stats["vertices"] += 1
            elif line.startswith("vt "):
                stats["texture_vertices"] += 1
            elif line.startswith("vn "):
                stats["normals"] += 1
            elif line.startswith("f "):
                stats["faces"] += 1
    stats["bbox_size"] = _bbox_size(stats["bbox"])
    return stats


def read_ply_stats(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "format": "ply",
        "vertices": None,
        "faces": None,
        "bbox": _empty_bbox(),
        "encoding": None,
    }
    header: list[str] = []
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY without end_header: {path}")
            text = line.decode("utf-8", errors="replace").strip()
            header.append(text)
            if text == "end_header":
                break
        for text in header:
            if text.startswith("format "):
                stats["encoding"] = text.split()[1]
            elif text.startswith("element vertex "):
                stats["vertices"] = int(text.split()[-1])
            elif text.startswith("element face "):
                stats["faces"] = int(text.split()[-1])
        if stats["encoding"] == "ascii" and stats["vertices"]:
            for _ in range(int(stats["vertices"])):
                line = f.readline().decode("utf-8", errors="ignore").split()
                if len(line) >= 3:
                    _update_bbox(stats["bbox"], (float(line[0]), float(line[1]), float(line[2])))
    if stats["bbox"]["min"] is None and PlyData is not None:
        ply = PlyData.read(path)
        vertices = ply["vertex"]
        xs = vertices["x"]
        ys = vertices["y"]
        zs = vertices["z"]
        stats["bbox"] = {
            "min": [float(xs.min()), float(ys.min()), float(zs.min())],
            "max": [float(xs.max()), float(ys.max()), float(zs.max())],
        }
    stats["bbox_size"] = _bbox_size(stats["bbox"])
    return stats


def mesh_stats(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    ext = p.suffix.lower()
    if ext == ".obj":
        stats = read_obj_stats(p)
    elif ext == ".ply":
        stats = read_ply_stats(p)
    else:
        stats = {"format": ext.lstrip(".") or "unknown"}
    stats["path"] = str(p)
    stats["exists"] = True
    stats["file_size_bytes"] = p.stat().st_size
    return stats
