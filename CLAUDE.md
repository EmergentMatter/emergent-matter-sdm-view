# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project Overview

`emergent-matter-sdm-view` is a **Layer 3** (host-app) shared library:
the Blender authoring + visualization layer that every Software Defined Matter CEM
plugs into. Consumers pass in their frozen `Parameters` dataclass + a
`make_components` callable, and sdm-view provides everything else: parameter
sliders, auto-regen, animation, cutaway, voxel-size knob, hot-reload,
annotation dots (bidirectional → JSON), parametric-spread showcase renders, and
the VDB-first geometry pipeline with live iso-tuning.

Extracted from patterns that had evolved independently across sibling CEM
repos and an earlier internal prototype addon. The goal: every new CEM gets
this stack for free, instead of re-implementing it per repo.

## Build & Run

```bash
uv sync
uv run pytest
```

## Code standard

Follow STYLE.md at the repo root for all code, comments, tests, and
docs. Pull request descriptions follow .github/PULL_REQUEST_TEMPLATE.md.

VDB writing uses Blender's **bundled** `openvdb` module, no separate pip
install. (PyPI `pyopenvdb` has no cp313 wheels and requires system OpenVDB
to source-build; not worth the friction when Blender ships it.) The `io/`
package's `vdb_export.py` is therefore designed to run inside Blender's
Python or in a subprocess invoked by `blender --background --python …`.

## Architecture

Package layout, the `io`/`blender` split, and the geometry pipeline:
[`docs/architecture.md`](docs/architecture.md).

## Technical Context

- **`viewer/` is the mesh-free path for `.sdm` parts**: sdm-core's GLSL
  emitter (`python -m software_defined_matter.glsl`) runs in a subprocess
  (the ONLY place Blender-side that knows sdm-core's CLI is
  `viewer/sidecar.py`); the SDF ray-marches as a fullscreen pass. It came in
  as a subtree merge of an earlier Emergent Matter internal Blender addon,
  heavily extended here on `feat/controls-panel`, so its history is in this
  repo. That original repo is now a tombstone; do not develop there.
- **Viewer prefs fall back when not addon-installed**: `viewer/prefs.py`
  serves AddonPreferences when sdm_view is an enabled add-on, else a
  module-level `runtime` object seeded from `$SDM_PYTHON` or a sibling
  `emergent-matter-sdm-core/.venv`. Launcher scripts rely on the fallback.
- **Bidirectional authoring** is the design intent: Blender is not just an
  output sink. Annotation dots are the seed pattern: data placed in the
  viewport flows back to the SDF model via JSON in `/tmp/`. Future patterns
  (draggable control points, in-viewport constraints) follow the same shape.
- **Multi-worktree reload**: the current reload path is single-rooted; this
  library should let multiple worktrees of the same project drive separate
  Blender sessions concurrently. Not yet built.
- **sdm-core's conformance corpus is the drift guard**: sdm-core ships
  `schema/conformance/` inside its wheel specifically for downstream hosts,
  with `consumers/expected/*.json` fingerprints naming the GLSL macros an
  emission requires. `tests/test_sdm_core_conformance.py` holds this repo
  against it. Before hand-testing a sdm-core upgrade in Blender, run that
  file: CLI flags, table payloads, and the `.sdm` schema floor all drifted
  unnoticed once because nothing checked them outside a viewport. It SKIPS
  unless sdm-core is importable, and CI has no sibling checkout, so it is a
  local gate rather than a CI one until CI installs sdm-core.
  A green pull request does not mean these ran.
- **A new emitter table payload needs three edits**: the macro in
  `sidecar.SUPPORTED_TABLE_MACROS`, a field pair on `EmissionArtifacts` read
  by `_read_tables`, and a row in `viewport._TABLE_BINDINGS`. Miss the last
  and the part renders wrong rather than erroring, because the shader fetches
  a sampler nothing bound.
- **Grid sampling is public in sdm-core now**: `software_defined_matter.
  grid_sampling` (`bind_sdf`, `make_grid`, `eval_chunked`, `BBox3`) replaced
  the private `_meshing.bind` / `_meshing.grid`. Reach for it rather than
  anything under `_meshing`, which is mesh extraction only.
- **bpy dependency is implicit**: pyproject does NOT declare `bpy` because
  Blender bundles its own Python. The `blender/` subpackage is only meant to
  load inside Blender. Tests for `blender/` use mocks or run inside `bpy`-pip
  if available; `io/` tests run in plain Python.

## Naming Conventions

Typed-prefix convention for engineering-quantity scalars (see STYLE.md):

| Prefix | Type | Example |
|---|---|---|
| `d_` | float | `d_voxel_size_mm` |
| `n_` | int | `n_animation_frames` |
| `b_` | bool | `b_show_cutaway` |
| `s_` | str | `s_dot_label` |

Blender operator/panel naming: `SDMVIEW_OT_*`, `SDMVIEW_PT_*`, `SDMVIEW_MT_*`.
(Consumer CEMs may register additional operators under their own prefix.)
Exception: the merged `viewer/` module kept its original `SDM_OT_*` /
`SDM_PT_*` names and `sdm.*` operator idnames, since renaming would break
saved keymaps/scripts; unify opportunistically if the operators are ever
reworked.

## Consumed dependencies

None at runtime (pure Python io layer + Blender's own bundled modules).
Optional: `pyopenvdb` for headless `.vdb` writing outside Blender.

## Consumers

See [README.md](README.md#consumers).

## Current Work

Tracked in GitHub issues and milestones, not in this file.
