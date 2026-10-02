# 0001. VDB-first geometry pipeline

## Status

Accepted (2026-05-21)

## Context

Bringing a CEM's SDF into Blender needs an iso-surface: the boundary
between inside and outside. A marching-cubes mesh fixes that boundary at
extraction time. Every "what if the wall is thinner?" or "what if the
neck is wider?" question then means re-evaluating the JAX SDF on the host
side, extracting a new mesh, and re-importing it, with the viewport
frozen while that round-trip runs.

An earlier internal Blender addon had already worked around this by
writing the full signed-distance field to disk instead of a fixed mesh,
and letting Blender extract the surface. That pattern is the one sdm-view
adopts.

## Decision

sdm-view writes the full SDF as an OpenVDB `.vdb` grid
(`io/vdb_export.py`), imports it as `bpy.data.volumes`, and attaches a
`VOLUME_TO_MESH` modifier (`blender/vdb_import.py`). The modifier's
`threshold` parameter is the iso level, so it becomes a live slider in
the Blender viewport: moving it re-extracts the surface from the grid
already on disk, with no re-evaluation of the JAX SDF and no host
round-trip.

Re-evaluating marching cubes on every threshold change, the alternative
this replaces, was rejected on exactly that cost: it makes iso-threshold
exploration slow enough that a user doesn't do it, which defeats the
point of an interactive viewer.

## Consequences

- Iso-threshold tuning is instant and lives entirely in the Blender
  viewport; no JAX re-eval is needed to answer "what if this wall were
  thinner."
- `scikit-image` marching cubes is kept, but only as a fallback path: for
  headless rendering and for STL export, where a fixed mesh is the actual
  deliverable rather than an exploration aid.
- Blender's bundled `openvdb` module is a hard dependency of the VDB
  write path. `io/vdb_export.py` targets that bundled module rather than
  the PyPI `pyopenvdb` package, which has no `cp313` wheels and would
  need a source-built system OpenVDB.
- The GLSL ray-march viewer (`blender/viewer/`) is a separate, later
  path that skips voxelization entirely: it ray-marches the SDF directly
  in a fragment shader, so it isn't governed by this decision the way
  `vdb_import.py` is. See
  [`src/sdm_view/blender/viewer/README.md`](../../src/sdm_view/blender/viewer/README.md).
