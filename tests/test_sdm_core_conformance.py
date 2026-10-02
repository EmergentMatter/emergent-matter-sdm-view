"""Hold the viewer against sdm-core's own consumer conformance corpus.

sdm-core ships `schema/conformance/` for exactly this: hand-authored `.sdm`
documents plus `expected/*.json` fingerprints, so a downstream host tests
against the emitter's fixtures instead of guessing at the wire format. The
corpus ships inside the built wheel, so it is found next to the installed
package rather than by walking to a sibling checkout.

These tests exist because every contract sdm-core changed went unnoticed here
until the viewer was opened by hand: a CLI flag that no longer parsed, two
table payloads nothing bound, and a schema floor three versions behind. Each
one is cheap to catch from the corpus and expensive to find in a viewport.

Skipped, not failed, when sdm-core is unavailable. It is a dev-only
dependency, and `io/` must keep working without it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sdm_core = pytest.importorskip(
    "software_defined_matter",
    reason=(
        "sdm-core is not importable, so the emitter contract cannot be checked. "
        "This repo does not declare it (see the note in pyproject.toml). To run "
        "these: uv pip install emergent-matter-sdm-core "
        "--index https://get.softwaredefinedmatter.com/simple"
    ),
)

CONSUMERS = Path(sdm_core.__file__).parent / "schema" / "conformance" / "consumers"
VALID = Path(sdm_core.__file__).parent / "schema" / "conformance" / "valid"


def _load_sidecar():
    """Import `viewer.sidecar` without its bpy-bound package `__init__`.

    `sidecar` is deliberately stdlib-only, so it is testable outside Blender,
    but `viewer/__init__.py` pulls in the operators and the viewport and those
    need `bpy`. Loading the file directly is what keeps that true: if someone
    adds a `bpy` import to `sidecar`, this stops working and says so.
    """
    path = Path(__file__).resolve().parents[1] / "src/sdm_view/blender/viewer/sidecar.py"
    spec = importlib.util.spec_from_file_location("sdm_view_sidecar_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sidecar = _load_sidecar()


def _fingerprints() -> list[tuple[str, dict]]:
    """Every `expected/*.json` in the consumer corpus, by fixture name."""
    return sorted((f.stem, json.loads(f.read_text())) for f in CONSUMERS.rglob("expected/*.json"))


def _emit(sdm_name: str):
    """Emit one corpus fixture through the real sidecar path.

    Uses this interpreter, which is the one that has sdm-core, and a
    temporary output directory so the corpus stays clean.
    """
    return sidecar.emit_artifacts(
        CONSUMERS / sdm_name,
        sys.executable,
        keep_artifacts=False,
    )


# ---------------------------------------------------------------------------
# The emitter's macro contract
# ---------------------------------------------------------------------------


def test_the_corpus_is_actually_there() -> None:
    """Guard against a silent pass if sdm-core stops shipping the corpus."""
    assert CONSUMERS.is_dir(), f"no consumer corpus at {CONSUMERS}"
    assert _fingerprints(), f"no expected/*.json fingerprints under {CONSUMERS}"


def test_every_emitted_table_macro_is_one_the_viewer_binds() -> None:
    """No fixture may require a table payload the viewport cannot carry.

    This is the check that was missing. `u_sdm_sweep` and `u_sdm_grid` were
    emitted for a long time with nothing here to bind them, and the failure
    mode is a part that renders wrong rather than a part that errors.
    """
    unsupported: dict[str, list[str]] = {}
    for name, fp in _fingerprints():
        required = fp.get("emission", {}).get("macros_required") or []
        missing = [
            m for m in required if m.endswith("_TABLE") and m not in sidecar.SUPPORTED_TABLE_MACROS
        ]
        if missing:
            unsupported[name] = missing
    assert not unsupported, (
        f"sdm-core emits table payloads this viewer never binds: {unsupported}. "
        f"Add the macro to sidecar.SUPPORTED_TABLE_MACROS, a field to "
        f"EmissionArtifacts, and a row to viewport._TABLE_BINDINGS."
    )


# ---------------------------------------------------------------------------
# The emitter CLI contract
# ---------------------------------------------------------------------------


def test_sidecar_invocation_is_accepted_by_the_cli() -> None:
    """The command `emit_artifacts` builds must still parse.

    argparse exits 2 on an unknown argument, which fails the whole emission
    and leaves whatever stale artifacts were on disk in place. That reads as
    "the viewer ignored my edit", not as an error, which is why it survived
    two removed flags.
    """
    artifacts = _emit("polygon_inline.sdm")
    assert artifacts.scene_source, "emitter returned an empty scene"
    assert artifacts.lib_source, "emitter returned an empty lib"


# ---------------------------------------------------------------------------
# The table payloads
# ---------------------------------------------------------------------------


def test_polygon_table_is_parsed() -> None:
    """A tabled polygon fixture round-trips into bindable artifacts."""
    expected = json.loads((CONSUMERS / "expected" / "polygon_table.json").read_text())
    artifacts = _emit("polygon_table.sdm")
    assert len(artifacts.poly_table) == expected["emission"]["poly_table_len"]
    assert artifacts.poly_tex_width == expected["emission"]["poly_tex_width"]


def test_sweep_table_is_parsed() -> None:
    """The sweep payload reaches the artifacts with a width to read it by."""
    expected = json.loads((CONSUMERS / "sweep" / "expected.json").read_text())["tabled"]
    artifacts = _emit("sweep/tabled.sdm")
    assert artifacts.sweep_table, "tabled sweep fixture produced no sweep_table"
    assert artifacts.sweep_tex_width == expected["sweep_tex_width"]
    # RGBA texels: a partial row would mean the frame layout is misread.
    assert len(artifacts.sweep_table) % 4 == 0


def test_raster_grid_is_parsed() -> None:
    """The raster payload reaches the artifacts, unpacked from base64."""
    artifacts = _emit("raster_field.sdm")
    assert artifacts.grid_table, "raster fixture produced no grid_table"
    assert artifacts.grid_tex_width > 0, "grid payload with no row width is unreadable"


# ---------------------------------------------------------------------------
# The .sdm schema floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", sorted(p.name for p in VALID.glob("minimal_0.*.sdm")))
def test_loader_reads_every_known_schema_version(fixture: str) -> None:
    """The panel's loader must not fall behind sdm-core's schema.

    It sat at an exact-match check on 0.1 while sdm-core shipped 0.4, so the
    slider panel refused every current file.
    """
    from sdm_view.io.sdm_loader import load_sdm_part

    part = load_sdm_part(VALID / fixture)
    assert part.name
