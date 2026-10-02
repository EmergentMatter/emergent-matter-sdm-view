"""Subprocess wrapper around ``python -m software_defined_matter.glsl``.

Blender 4.2 ships its own Python interpreter without our heavy deps (jax,
jsonschema, ...), so we don't try to import ``software_defined_matter``
inside Blender. Instead we shell out to a user-supplied Python that has the
package installed and read back the three artifacts:

  - ``sdf_lib.glsl``    static GLSL helper library
  - ``sdf_scene.glsl``  generated ``sdf_scene(vec3 p)`` + per-node functions
  - ``meta.json``       uniform schema + bbox + smooth-CSG flag

This file is the only Blender-side place that knows about sdm-core's CLI.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UniformSpec:
    name: str
    glsl_type: str
    initial: float
    bounds: tuple[float, float] | None
    unit: str
    source_param: str


@dataclass(frozen=True)
class ControlSpec:
    """One entry of the emitter's per-param control manifest.

    ``cls`` tells the panel which widget to build:
    - ``"live"``     : backed by a ``u_p_*`` uniform; scrubs with zero recompile
    - ``"re-emit"``  : takes effect on Rebuild (re-run generator + re-emit)
    - ``"topology"`` : same, and the tree's node count changes
    """

    param: str
    cls: str
    value: float
    free: bool
    unit: str
    bounds: tuple[float, float] | None
    uniform: str | None  # GLSL uniform name when cls == "live"
    ui: dict  # step / explore_bounds / group / order / role


@dataclass(frozen=True)
class EmissionArtifacts:
    """All output from one CLI invocation, materialised in memory."""

    lib_source: str
    scene_source: str
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]]
    smooth_csg: bool
    smooth_k: float
    entry_point: str
    uniforms: list[UniformSpec]
    controls: list[ControlSpec]  # empty when emitted by a pre-manifest sdm-core
    animations: list[dict]  # .sdm metadata.animations (authored/auto-built)
    artifact_dir: Path  # where the files were written (may be a tmp)
    sdm_path: Path  # source .sdm path
    # Segmentation table [{"id", "machine", "label"}]; empty = one component.
    components: tuple[dict, ...] = ()
    # Table-mode payloads the emitter hands off for the viewport to bind as
    # textures. Each is flat, in fetch order, and empty means the emitter
    # inlined that payload in the shader source instead, so there is nothing
    # to bind. The row widths come from the emission because sdm-core bakes
    # the matching SDM_*_TEX_W into sdf_lib.glsl: a width that disagrees with
    # the texture makes every fetch read the wrong row.
    #
    # poly:  (x, y) per texel,          RG32F,   u_sdm_poly
    # sweep: one frame row per texel,   RGBA32F, u_sdm_sweep
    # grid:  one raster sample per texel, R32F,  u_sdm_grid
    #
    # Omitting any of these is not a harmless default. The viewport binds no
    # texture for an empty table, so the part loses that geometry and renders
    # as whatever is left. It fails by drawing the wrong thing, not by
    # erroring. See sdm-core lib.glsl, the SDM_*_TABLE blocks.
    poly_table: tuple[float, ...] = ()
    poly_tex_width: int = 1024
    sweep_table: tuple[float, ...] = ()
    sweep_tex_width: int = 0
    grid_table: tuple[float, ...] = ()
    grid_tex_width: int = 0


# The table-mode payloads this viewer can carry from an emission to the GPU,
# named by the macro sdm-core defines in sdf_lib.glsl when it emits one.
#
# A macro that shows up in an emission and not here is geometry the viewport
# will quietly fail to draw: the shader fetches a sampler nothing ever bound.
# That is how sweep and raster support went missing for as long as it did, so
# tests/test_sdm_core_conformance.py holds this set against sdm-core's own
# fixtures and fails when the emitter learns a payload this has not.
#
# Only the *_TABLE macros are listed. The matching *_TEX_W, *_LEN and
# *_MAX_N are emitted into sdf_lib.glsl by sdm-core and ask nothing of a host.
SUPPORTED_TABLE_MACROS = frozenset(
    {
        "SDM_POLY_TABLE",
        "SDM_SWEEP_TABLE",
        "SDM_GRID_TABLE",
    }
)


def _read_tables(meta: dict) -> dict:
    """Pull every table-mode payload out of a parsed meta.json.

    Shared by both meta.json readers. They drifted once already: the emit
    path grew fields the reload path never learned, so a part reloaded from
    an artifact dir rendered differently from the same part freshly emitted.

    The grid arrives twice, as a JSON float list and as base64-packed f32.
    Prefer the packed form. A raster field is easily millions of samples, and
    parsing that as JSON numbers costs seconds and a lot of memory.
    """
    grid_b64 = meta.get("grid_table_b64")
    if grid_b64:
        raw = base64.b64decode(grid_b64)
        grid = struct.unpack(f"<{len(raw) // 4}f", raw)
    else:
        grid = tuple(float(x) for x in meta.get("grid_table") or ())
    return {
        "poly_table": tuple(float(x) for x in meta.get("poly_table") or ()),
        "poly_tex_width": int(meta.get("poly_tex_width", 1024)),
        "sweep_table": tuple(float(x) for x in meta.get("sweep_table") or ()),
        "sweep_tex_width": int(meta.get("sweep_tex_width", 0)),
        "grid_table": grid,
        "grid_tex_width": int(meta.get("grid_tex_width", 0)),
    }


def load_artifacts_from_dir(glsl_dir, sdm_path) -> EmissionArtifacts:
    """Read a previously emitted `<part>_glsl/` directory back into artifacts.

    Shared rather than copied. Two scripts had hand-rolled this (the shader-cost
    probe and the still renderer), and a hand-rolled second copy got the field
    names wrong on its first run, because `EmissionArtifacts` says `lib_source`
    while the file on disk is `sdf_lib.glsl`. One loader beside the dataclass
    cannot drift from it.

    Does NOT re-run the emitter; the directory must already hold
    `sdf_lib.glsl`, `sdf_scene.glsl` and `meta.json`.
    """
    import json
    from pathlib import Path

    glsl_dir, sdm_path = Path(glsl_dir), Path(sdm_path)
    meta = json.loads((glsl_dir / "meta.json").read_text())
    uniforms = [
        UniformSpec(
            name=u["name"],
            glsl_type=u["glsl_type"],
            initial=float(u["initial"]),
            bounds=tuple(u["bounds"]) if u.get("bounds") else None,
            unit=u.get("unit", ""),
            source_param=u["source_param"],
        )
        for u in meta.get("uniforms", [])
    ]
    bb = meta["bbox"]
    bb_lo, bb_hi = bb[0], bb[1]
    return EmissionArtifacts(
        lib_source=(glsl_dir / "sdf_lib.glsl").read_text(),
        scene_source=(glsl_dir / "sdf_scene.glsl").read_text(),
        # Built as fixed 3-tuples, not tuple(map(float, ...)) -- the latter
        # types as tuple[float, ...] (unknown length), which doesn't match
        # EmissionArtifacts' tuple[float, float, float] bbox corners.
        bbox=(
            (float(bb_lo[0]), float(bb_lo[1]), float(bb_lo[2])),
            (float(bb_hi[0]), float(bb_hi[1]), float(bb_hi[2])),
        ),
        smooth_csg=bool(meta.get("smooth_csg", False)),
        smooth_k=float(meta.get("smooth_k", 0.25)),
        entry_point=str(meta.get("entry_point", "sdf_scene")),
        uniforms=uniforms,
        controls=[],
        animations=[],
        artifact_dir=glsl_dir,
        sdm_path=sdm_path,
        **_read_tables(meta),
    )


class SidecarError(RuntimeError):
    """Wraps anything that goes wrong invoking or parsing the emitter."""


# Bump when the emitter's artifact contract changes in a way the viewer
# depends on; cached artifacts with an older (or absent) version re-emit.
# v3: rotate-bbox mirror fix; v2 emissions can carry mirrored component
# AABBs that prune real geometry.
# v4: material-space cutaway (sdf_scene_rcut) + rest-point lookup
# (sdm_rest_point); the template references both.
# v5: 3-tap canonical_sector_fold; v4 emissions render bulged
# pseudo-surfaces where folded geometry spills its wedge (blade hubs).
# v6: motion chains cover $ref-driven transforms (panel pivots, say);
# v5 rest points miss them, so pattern/cut don't ride transform poses.
# v7: unified field entry point (sdf_scene routes through sdf_scene_rcut,
# halving PSO inline mass) + poly-lod default 128 (full airfoil OML).
# v8: table-mode polygons, vertices bound as u_sdm_poly.
# v9: sweep and grid tables (u_sdm_sweep, u_sdm_grid) are parsed and bound.
# The stamp hashes only the .sdm, so it cannot see an sdm-core upgrade on
# its own: every artifact dir emitted before this stays a cache hit forever
# unless the version moves, and the viewer would keep serving a shader whose
# tables it never uploads. Bumping is cheap, at worst one re-emission.
_ARTIFACT_VERSION = 9


def emit_artifacts(
    sdm_path: Path,
    python_executable: str,
    *,
    keep_artifacts: bool = True,
) -> EmissionArtifacts:
    """Run the emitter on ``sdm_path`` and return parsed artifacts.

    Parameters
    ----------
    sdm_path
        Absolute path to the ``.sdm`` file.
    python_executable
        Path to a Python interpreter that has ``software_defined_matter``
        installed.
    keep_artifacts
        If True, write the files next to the .sdm (``<stem>_glsl/``); if
        False, write to a temporary directory that is cleaned up on
        unregister. The viewport draw handler only needs the in-memory
        ``EmissionArtifacts``, so the on-disk files are purely for debug.
    """
    sdm_path = Path(sdm_path).resolve()
    if not sdm_path.is_file():
        raise SidecarError(f"No such file: {sdm_path}")

    if keep_artifacts:
        out_dir = sdm_path.parent / f"{sdm_path.stem}_glsl"
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="sdm_glsl_"))

    # Content-hash cache: the CLI runs the full JAX emission (minutes of
    # cold-start on a big part), which is pure waste when the on-disk
    # artifacts were emitted from EXACTLY this .sdm. A stamp holding the
    # .sdm's sha256 next to the artifacts lets a re-import skip straight to
    # parsing. Any change to the file (including the rebuild path, which
    # rewrites the .sdm before emitting) changes the hash and re-emits.
    # Delete the ``<stem>_glsl`` dir to force a fresh emission.
    stamp = out_dir / ".source_sha256"
    # The stamp also carries an artifact-format version: bumping it
    # invalidates every cache after an emitter upgrade that changes what the
    # viewer needs from the artifacts (e.g. v2 = table-mode polygon storage
    # + AABB pruning), which the .sdm hash alone can't see.
    stamp_value = f"{hashlib.sha256(sdm_path.read_bytes()).hexdigest()}:v{_ARTIFACT_VERSION}"
    cache_hit = (
        keep_artifacts
        and stamp.is_file()
        and stamp.read_text().strip() == stamp_value
        and all((out_dir / f).is_file() for f in ("meta.json", "sdf_lib.glsl", "sdf_scene.glsl"))
    )

    if not cache_hit:
        cmd = [
            python_executable,
            "-m",
            "software_defined_matter.glsl",
            str(sdm_path),
            "--out",
            str(out_dir),
            # Nothing else. sdm-core dropped both flags this used to pass,
            # and argparse rejects an unknown one with exit 2, which fails
            # the whole emission and leaves stale artifacts on disk.
            #
            # `--poly-table` is gone because the choice is no longer ours.
            # Inline storage sizes every polygon's array to the scene's
            # largest and pads the rest, which once left Metal's fragment
            # compiler unfinished after 11 minutes on a gear part. The
            # emitter now applies its own whole-tree vertex budget and
            # reports what it picked in meta.json, so the viewport binds
            # whatever it is handed (see _build_poly_texture).
            #
            # `--no-validate` is gone with no replacement. The CLI always
            # validates on load now, so jsonschema's recursive oneOf is back
            # in the emit path for every part. If import time regresses on a
            # big tree, that is where it went, and the fix belongs upstream.
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SidecarError(
                f"Could not run {python_executable!r}. "
                f"Check the addon preference 'sdm-core Python'."
            ) from exc

        if proc.returncode != 0:
            raise SidecarError(
                f"sdm-core CLI failed (exit {proc.returncode}):\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"  stderr: {proc.stderr.strip()}"
            )
        if keep_artifacts:
            try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
                stamp.write_text(stamp_value + "\n")
            except OSError:
                pass  # cache is best-effort; never fail the import over it

    try:
        meta = json.loads((out_dir / "meta.json").read_text())
        lib_src = (out_dir / "sdf_lib.glsl").read_text()
        scene_src = (out_dir / "sdf_scene.glsl").read_text()
    except (OSError, json.JSONDecodeError) as exc:
        raise SidecarError(f"Could not read emitted artifacts from {out_dir}: {exc}") from exc

    uniforms = [
        UniformSpec(
            name=u["name"],
            glsl_type=u["glsl_type"],
            initial=float(u["initial"]),
            bounds=tuple(u["bounds"]) if u.get("bounds") else None,
            unit=u.get("unit", ""),
            source_param=u["source_param"],
        )
        for u in meta.get("uniforms", [])
    ]

    controls = [
        ControlSpec(
            param=c["param"],
            cls=c["class"],
            value=float(c["value"]),
            free=bool(c.get("free", False)),
            unit=c.get("unit", ""),
            bounds=tuple(c["bounds"]) if c.get("bounds") else None,
            uniform=c.get("uniform"),
            ui=dict(c.get("ui", {})),
        )
        for c in meta.get("controls", [])
    ]

    bbox_raw = meta["bbox"]
    bbox = (
        (float(bbox_raw[0][0]), float(bbox_raw[0][1]), float(bbox_raw[0][2])),
        (float(bbox_raw[1][0]), float(bbox_raw[1][1]), float(bbox_raw[1][2])),
    )

    artifacts = EmissionArtifacts(
        lib_source=lib_src,
        scene_source=scene_src,
        bbox=bbox,
        smooth_csg=bool(meta.get("smooth_csg", False)),
        smooth_k=float(meta.get("smooth_k", 0.25)),
        entry_point=str(meta.get("entry_point", "sdf_scene")),
        uniforms=uniforms,
        controls=controls,
        animations=_read_animations(sdm_path),
        artifact_dir=out_dir,
        sdm_path=sdm_path,
        components=tuple(meta.get("components") or ()),
        **_read_tables(meta),
    )

    # Clean up tmpdir lazily: caller may want to inspect during the session,
    # so we only remove on unregister via cleanup_tmp_dirs().
    if not keep_artifacts:
        _TMP_DIRS_TO_CLEAN.append(out_dir)

    return artifacts


def _read_animations(sdm_path: Path) -> list[dict]:
    """Animation specs live in the .sdm's metadata (not meta.json):
    animation is part data, not an emission artifact."""
    try:
        doc = json.loads(Path(sdm_path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return list((doc.get("metadata") or {}).get("animations", []))


_TMP_DIRS_TO_CLEAN: list[Path] = []


def cleanup_tmp_dirs() -> None:
    """Remove tmp artifact dirs created by ``emit_artifacts(keep_artifacts=False)``.

    Safe to call multiple times. Called on addon unregister.
    """
    while _TMP_DIRS_TO_CLEAN:
        d = _TMP_DIRS_TO_CLEAN.pop()
        shutil.rmtree(d, ignore_errors=True)
