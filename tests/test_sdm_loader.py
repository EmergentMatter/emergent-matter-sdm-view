"""Test the .sdm loader.

`test_load_synthetic_fixture` pins the loader's behavior against a minimal,
self-contained schema v0.1 fixture built in this file, so the suite covers
`load_sdm_part` on every machine without depending on a sibling checkout.
`test_load_published_example` is an additional integration check against a
real sdm-core-generated file when one is available; it's a stronger check
(catches schema drift the synthetic fixture can't, since the synthetic
fixture only encodes what this module reads), not the only coverage.
"""

import json
import os
from pathlib import Path

import pytest

from sdm_view.io.sdm_loader import load_sdm_part

_ENV_VAR = "SDM_CORE_EXAMPLE_SDM"
_EXAMPLE_ENV = os.environ.get(_ENV_VAR)
SDM_CORE_EXAMPLE = Path(_EXAMPLE_ENV) if _EXAMPLE_ENV else None


def _write_sdm(path: Path) -> Path:
    """Write a minimal, schema-valid `.sdm` fixture exercising every field
    `load_sdm_part` reads."""
    doc = {
        "schema_version": "0.1",
        "name": "synthetic_test_part",
        "params": {
            "outer_radius": {
                "value": 10.0,
                "free": True,
                "bounds": [5.0, 20.0],
                "unit": "mm",
            },
            "height": {
                "value": 42.0,
                "free": False,
            },
        },
        "materials": [{"name": "PA12"}],
        "couplings": [],
        "objectives": [{"kind": "minimize_mass"}],
        "constraints": [],
        "metadata": {"generator": "test fixture"},
    }
    path.write_text(json.dumps(doc))
    return path


def test_load_synthetic_fixture(tmp_path: Path) -> None:
    part = load_sdm_part(_write_sdm(tmp_path / "synthetic.sdm"))

    assert part.name == "synthetic_test_part"
    by_name = {p.name: p for p in part.params}

    outer = by_name["outer_radius"]
    assert outer.value == pytest.approx(10.0)
    assert outer.free is True
    assert outer.bounds == (5.0, 20.0)
    assert outer.unit == "mm"

    height = by_name["height"]
    assert height.value == pytest.approx(42.0)
    assert height.free is False
    assert height.bounds is None
    assert height.unit == ""

    assert part.materials_raw == [{"name": "PA12"}]
    assert part.couplings_raw == []
    assert part.objectives_raw == [{"kind": "minimize_mass"}]
    assert part.constraints_raw == []
    assert part.metadata == {"generator": "test fixture"}


def test_load_synthetic_fixture_defaults_missing_collections(tmp_path: Path) -> None:
    """`materials`/`couplings`/`objectives`/`constraints`/`metadata` are all
    optional in the schema; a part with none of them present still loads,
    with the raw collections defaulting to empty rather than raising."""
    minimal = tmp_path / "minimal.sdm"
    minimal.write_text(json.dumps({"schema_version": "0.1", "name": "bare", "params": {}}))

    part = load_sdm_part(minimal)

    assert part.name == "bare"
    assert part.params == []
    assert part.materials_raw == []
    assert part.couplings_raw == []
    assert part.objectives_raw == []
    assert part.constraints_raw == []
    assert part.metadata == {}


@pytest.mark.skipif(
    SDM_CORE_EXAMPLE is None or not SDM_CORE_EXAMPLE.exists(),
    reason=(
        f"sdm-core's published example .sdm is not available; set {_ENV_VAR} to "
        "the path of hollow_cylinder_with_hinge.sdm in a sdm-core checkout "
        "(examples/hollow_cylinder_with_hinge.sdm) to run this test"
    ),
)
def test_load_published_example() -> None:
    # Pins shape and types, not exact values: this file is sdm-core's own
    # example and moves on its own schedule (found via this test, once the
    # fixture path was actually reachable locally: outer_radius drifted from
    # 10.0/(5, 20) to 20.0/(10, 40) at some point while this test had been
    # silently skipping on every CI run, since the hardcoded path never
    # existed on a runner -- the exact class of gap this file exists to
    # close). test_load_synthetic_fixture is what pins exact values.
    part = load_sdm_part(SDM_CORE_EXAMPLE)
    assert part.name == "hollow_cylinder_with_hinge"
    assert len(part.params) >= 4
    by_name = {p.name: p for p in part.params}
    assert "outer_radius" in by_name
    outer = by_name["outer_radius"]
    assert isinstance(outer.value, float)
    assert outer.free is True
    assert outer.bounds is not None
    assert outer.bounds[0] < outer.value < outer.bounds[1]
    assert outer.unit == "mm"

    height = by_name["height"]
    assert height.free is False
    assert height.bounds is None


def test_unknown_schema_warns_but_reads(tmp_path: Path) -> None:
    """A version this loader has not been checked against is a warning, not a
    refusal.

    `schema_version` is the minimum a reader must support, so refusing an
    unrecognised one locks the panel out of every file a newer sdm-core
    writes. This loader reads a stable subset, so best-effort is the honest
    behavior and the warning is what says so.
    """
    newer = tmp_path / "newer.sdm"
    newer.write_text('{"schema_version": "99.9", "name": "x"}')
    with pytest.warns(UserWarning, match="best-effort"):
        part = load_sdm_part(newer)
    assert part.name == "x"


def test_missing_schema_version_is_read_as_oldest(tmp_path: Path) -> None:
    """No declaration means the oldest version, which promises the least."""
    undeclared = tmp_path / "undeclared.sdm"
    undeclared.write_text('{"name": "x"}')
    part = load_sdm_part(undeclared)
    assert part.name == "x"


def test_malformed_document_still_raises(tmp_path: Path) -> None:
    """Leniency is about versions, not about accepting broken documents."""
    broken = tmp_path / "broken.sdm"
    broken.write_text('{"schema_version": "0.1"}')
    with pytest.raises(KeyError):
        load_sdm_part(broken)
