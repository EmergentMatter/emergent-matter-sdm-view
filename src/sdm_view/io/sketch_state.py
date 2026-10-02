"""Pure-Python I/O for principle-sketch JSON state.

Three sketches (XY / XZ / YZ planes), each carrying named curves with control
points. Both the Blender side and any JAX/CLI consumer use this module to
read/write the shared state file (default `/tmp/sdm-sketches.json`).

See project memory `sdm_view_drawings.md` for design intent.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = "v1"

Plane = Literal["xy", "xz", "yz"]
CurveType = Literal["polyline", "bezier", "circle", "arc"]


@dataclass
class Point:
    """A 2D point in a sketch's plane-local coordinates.

    `label` is optional: when set, it names the control point so it can be
    referred to parametrically (e.g. "the bar_origin point on plan_xy").
    """

    x: float
    y: float
    label: str = ""


@dataclass
class Curve:
    """A named curve inside a sketch.

    `type` determines which fields are meaningful:
      - polyline / bezier: uses `points`
      - circle:            uses `center` + `radius`
      - arc:               uses `center` + `radius` + `start_angle_deg` + `sweep_angle_deg`
    """

    id: str
    type: CurveType
    label: str = ""
    color: str = "orange"
    points: list[Point] = field(default_factory=list)
    center: Point | None = None
    radius: float | None = None
    start_angle_deg: float | None = None
    sweep_angle_deg: float | None = None


@dataclass
class Sketch:
    id: str
    plane: Plane
    label: str = ""
    curves: list[Curve] = field(default_factory=list)


def write_sketches(path: str | Path, sketches: Sequence[Sketch]) -> Path:
    """Write `sketches` into the project file at `path`, preserving any
    other top-level keys (e.g. dots / vectors written by annotation_state).
    """
    out = Path(path)
    existing: dict = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text())
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing["schema"] = SCHEMA_VERSION
    existing["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    existing["sketches"] = [_sketch_to_dict(s) for s in sketches]
    out.write_text(json.dumps(existing, indent=2))
    return out


def read_sketches(path: str | Path) -> list[Sketch]:
    raw = json.loads(Path(path).read_text())
    schema = raw.get("schema", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported sketch schema {schema!r} (expected {SCHEMA_VERSION!r})")
    return [_sketch_from_dict(d) for d in raw.get("sketches", [])]


def _sketch_to_dict(s: Sketch) -> dict:
    return {
        "id": s.id,
        "plane": s.plane,
        "label": s.label,
        "curves": [_curve_to_dict(c) for c in s.curves],
    }


def _curve_to_dict(c: Curve) -> dict:
    d: dict = {"id": c.id, "type": c.type, "label": c.label, "color": c.color}
    if c.points:
        d["points"] = [{"x": p.x, "y": p.y, "label": p.label} for p in c.points]
    if c.center is not None:
        d["center"] = {"x": c.center.x, "y": c.center.y, "label": c.center.label}
    if c.radius is not None:
        d["radius"] = c.radius
    if c.start_angle_deg is not None:
        d["start_angle_deg"] = c.start_angle_deg
    if c.sweep_angle_deg is not None:
        d["sweep_angle_deg"] = c.sweep_angle_deg
    return d


def _sketch_from_dict(d: dict) -> Sketch:
    return Sketch(
        id=d["id"],
        plane=d["plane"],
        label=d.get("label", ""),
        curves=[_curve_from_dict(c) for c in d.get("curves", [])],
    )


def _curve_from_dict(d: dict) -> Curve:
    points = [Point(**p) for p in d.get("points", [])]
    center_dict = d.get("center")
    center = Point(**center_dict) if center_dict else None
    return Curve(
        id=d["id"],
        type=d["type"],
        label=d.get("label", ""),
        color=d.get("color", "orange"),
        points=points,
        center=center,
        radius=d.get("radius"),
        start_angle_deg=d.get("start_angle_deg"),
        sweep_angle_deg=d.get("sweep_angle_deg"),
    )
