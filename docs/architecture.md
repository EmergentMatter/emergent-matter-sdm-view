# Architecture

## Package layout

`src/sdm_view/` splits into two layers, and the split is about what may
import `bpy`, not about feature area:

- **`io/`** is pure Python: no `bpy` import anywhere in the package. Safe
  to import from system Python, including a CEM's Layer-1 SDF code that
  wants a headless `.vdb` writer with no Blender process involved.
- **`blender/`** is `bpy`-using: operators, panels, and property groups.
  Only loaded inside Blender.

Within `blender/`, each submodule covers one feature in the catalog and
opens with a docstring stating what it provides and how it fits the rest
of the package. That docstring is the reference for what a module does;
this page doesn't restate it, and neither does the root `README.md`'s
directory tree. Not every feature in the catalog has a module yet; the
gap is tracked as a GitHub issue rather than shipped as an empty
placeholder file.

A consuming CEM defines its own frozen `Parameters` dataclass and a
`make_components(params)` callable, then composes the `sdm_view.blender`
feature modules into its addon's `register()`. There is no single
aggregator yet that enumerates and registers every feature module for a
consumer.

## The GLSL ray-march viewer

`blender/viewer/` is large enough to carry its own architecture doc,
kept next to its code rather than duplicated here: see
[`src/sdm_view/blender/viewer/README.md`](../src/sdm_view/blender/viewer/README.md).

## Geometry pipeline

sdm-view's primary path from an SDF to Blender geometry is OpenVDB, not
marching cubes. See [ADR 0001](adr/0001-vdb-first-geometry-pipeline.md)
for the decision, what it replaced, and where marching cubes still
applies.
