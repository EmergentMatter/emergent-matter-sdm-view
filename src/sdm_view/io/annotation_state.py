"""Pure-Python I/O for 3D scene annotations: dots and vectors.

Both are bidirectional channels between the user (in Blender) and any
JAX/CLI consumer that wants to read placed geometric intent.

- **Dots**: labeled 3D points. Proven in a consuming CEM first; ported to
  sdm-view as the dots feature lands here.
- **Vectors**: tail + direction + magnitude. Useful for surface normals,
  push-pull rod axes, force / motion directions, rolling-contact axes.

Both share one JSON file (default `/tmp/sdm-annotations.json`) with a
versioned schema so the writer/reader can evolve.

Schema v1:

    {
      "schema": "v1",
      "updated_at": "...",
      "dots":    [{"index", "label", "color", "x", "y", "z"}, ...],
      "vectors": [{"index", "label", "color",
                    "tail":  {"x", "y", "z"},
                    "head":  {"x", "y", "z"},
                    "magnitude"}, ...]
    }
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path

SCHEMA_VERSION = "v1"


@dataclass
class Dot:
    index: int
    label: str
    color: str
    x: float
    y: float
    z: float


@dataclass
class Vector3:
    """3D vector annotation: tail point + direction + magnitude.

    Stored as both endpoints (tail + head) for unambiguous round-trip.
    `magnitude` = ‖head − tail‖, recomputed on write for safety.
    """

    index: int
    label: str
    color: str
    tail_x: float
    tail_y: float
    tail_z: float
    head_x: float
    head_y: float
    head_z: float

    @property
    def magnitude(self) -> float:
        dx = self.head_x - self.tail_x
        dy = self.head_y - self.tail_y
        dz = self.head_z - self.tail_z
        return sqrt(dx * dx + dy * dy + dz * dz)


def write_annotations(
    path: str | Path,
    *,
    dots: Sequence[Dot] = (),
    vectors: Sequence[Vector3] = (),
) -> Path:
    """Write annotations into the project file at `path`, preserving any
    other top-level keys (e.g. sketches written by sketch_state)."""
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
    existing["dots"] = [_dot_to_dict(d) for d in dots]
    existing["vectors"] = [_vector_to_dict(v) for v in vectors]
    out.write_text(json.dumps(existing, indent=2))
    return out


def read_annotations(
    path: str | Path,
) -> tuple[list[Dot], list[Vector3]]:
    raw = json.loads(Path(path).read_text())
    schema = raw.get("schema", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported annotation schema {schema!r} (expected {SCHEMA_VERSION!r})")
    dots = [_dot_from_dict(d) for d in raw.get("dots", [])]
    vectors = [_vector_from_dict(v) for v in raw.get("vectors", [])]
    return dots, vectors


def _dot_to_dict(d: Dot) -> dict:
    return {
        "index": d.index,
        "label": d.label,
        "color": d.color,
        "x": round(d.x, 4),
        "y": round(d.y, 4),
        "z": round(d.z, 4),
    }


def _vector_to_dict(v: Vector3) -> dict:
    return {
        "index": v.index,
        "label": v.label,
        "color": v.color,
        "tail": {"x": round(v.tail_x, 4), "y": round(v.tail_y, 4), "z": round(v.tail_z, 4)},
        "head": {"x": round(v.head_x, 4), "y": round(v.head_y, 4), "z": round(v.head_z, 4)},
        "magnitude": round(v.magnitude, 4),
    }


def _dot_from_dict(d: dict) -> Dot:
    return Dot(
        index=int(d["index"]),
        label=str(d.get("label", "")),
        color=str(d.get("color", "orange")),
        x=float(d["x"]),
        y=float(d["y"]),
        z=float(d["z"]),
    )


def _vector_from_dict(d: dict) -> Vector3:
    tail = d["tail"]
    head = d["head"]
    return Vector3(
        index=int(d["index"]),
        label=str(d.get("label", "")),
        color=str(d.get("color", "orange")),
        tail_x=float(tail["x"]),
        tail_y=float(tail["y"]),
        tail_z=float(tail["z"]),
        head_x=float(head["x"]),
        head_y=float(head["y"]),
        head_z=float(head["z"]),
    )
