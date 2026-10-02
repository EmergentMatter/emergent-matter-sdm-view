"""IO smoke tests for annotation JSON state."""

import json
from pathlib import Path

import pytest

from sdm_view.io.annotation_state import (
    Dot,
    Vector3,
    read_annotations,
    write_annotations,
)


def test_vector_magnitude() -> None:
    v = Vector3(
        index=1,
        label="x_axis",
        color="orange",
        tail_x=0.0,
        tail_y=0.0,
        tail_z=0.0,
        head_x=3.0,
        head_y=4.0,
        head_z=0.0,
    )
    assert v.magnitude == pytest.approx(5.0)


def test_vector_roundtrip(tmp_path: Path) -> None:
    src = [
        Vector3(
            index=1,
            label="bridge_normal",
            color="cyan",
            tail_x=0.0,
            tail_y=0.0,
            tail_z=0.0,
            head_x=0.0,
            head_y=0.0,
            head_z=5.0,
        ),
        Vector3(
            index=2,
            label="cam_a_motion",
            color="orange",
            tail_x=-12.0,
            tail_y=0.0,
            tail_z=0.0,
            head_x=-12.0,
            head_y=3.0,
            head_z=0.0,
        ),
    ]
    path = tmp_path / "ann.json"
    write_annotations(path, vectors=src)

    raw = json.loads(path.read_text())
    assert raw["schema"] == "v1"
    assert len(raw["vectors"]) == 2
    assert raw["vectors"][0]["tail"]["x"] == 0.0
    assert raw["vectors"][0]["head"]["z"] == 5.0
    assert raw["vectors"][0]["magnitude"] == pytest.approx(5.0)

    _dots, vectors = read_annotations(path)
    assert len(vectors) == 2
    assert vectors[1].label == "cam_a_motion"
    assert vectors[1].tail_x == pytest.approx(-12.0)


def test_mixed_roundtrip(tmp_path: Path) -> None:
    dots = [Dot(index=1, label="origin", color="white", x=0.0, y=0.0, z=0.0)]
    vectors = [
        Vector3(
            index=1,
            label="x_hat",
            color="orange",
            tail_x=0.0,
            tail_y=0.0,
            tail_z=0.0,
            head_x=1.0,
            head_y=0.0,
            head_z=0.0,
        )
    ]
    path = tmp_path / "ann_mixed.json"
    write_annotations(path, dots=dots, vectors=vectors)
    back_dots, back_vectors = read_annotations(path)
    assert len(back_dots) == 1 and len(back_vectors) == 1
    assert back_dots[0].label == "origin"
    assert back_vectors[0].label == "x_hat"


def test_unknown_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "v999"}))
    with pytest.raises(ValueError, match="Unsupported annotation schema"):
        read_annotations(path)
