"""IO smoke tests for principle-sketch JSON state."""

import json
from pathlib import Path

import pytest

from sdm_view.io.sketch_state import (
    Curve,
    Point,
    Sketch,
    read_sketches,
    write_sketches,
)


def test_roundtrip(tmp_path: Path) -> None:
    src = [
        Sketch(
            id="plan_xy",
            plane="xy",
            label="Top view",
            curves=[
                Curve(
                    id="bar_outline",
                    type="polyline",
                    label="bar outline",
                    color="orange",
                    points=[
                        Point(x=0.0, y=-3.0, label="bar_corner_a"),
                        Point(x=0.0, y=+3.0, label="bar_corner_b"),
                    ],
                ),
                Curve(
                    id="cam_a_od",
                    type="circle",
                    label="cam A OD",
                    color="cyan",
                    center=Point(x=-12.0, y=0.0),
                    radius=8.5,
                ),
            ],
        ),
        Sketch(id="plan_xz", plane="xz", label="side", curves=[]),
        Sketch(id="plan_yz", plane="yz", label="front", curves=[]),
    ]

    path = tmp_path / "sketches.json"
    write_sketches(path, src)

    raw = json.loads(path.read_text())
    assert raw["schema"] == "v1"
    # "count" is schema-optional; kept as an explicit branch rather than
    # dict.get's default-value form, since a future schema may add it.
    assert raw["count"] if "count" in raw else True  # noqa: SIM401
    assert len(raw["sketches"]) == 3

    back = read_sketches(path)
    assert len(back) == 3
    assert back[0].id == "plan_xy"
    assert back[0].curves[0].points[0].label == "bar_corner_a"
    assert back[0].curves[1].radius == pytest.approx(8.5)
    assert back[0].curves[1].center.x == pytest.approx(-12.0)


def test_unknown_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "v999", "sketches": []}))
    with pytest.raises(ValueError, match="Unsupported sketch schema"):
        read_sketches(path)
