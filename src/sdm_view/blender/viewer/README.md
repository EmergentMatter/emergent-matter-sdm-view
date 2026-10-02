# sdm_view.blender.viewer: GLSL ray-march `.sdm` viewer

Ray-marched viewport preview of [Software Defined Matter](https://github.com/EmergentMatter/emergent-matter-sdm-core) `.sdm` parts inside Blender.

The module imports a `.sdm`, asks `sdm-core` to emit GLSL via its CLI, and renders the SDF as a fullscreen ray-march pass in the 3D viewport. Free `Param`s become sliders in the N-panel so you can scrub the design at perfectly smooth resolution: no meshing, no voxel grid.

```
.sdm  ─►  sdm-core CLI  ─►  sdf_lib.glsl + sdf_scene.glsl + meta.json  ─►  viewport ray-march + sliders
```

Started as a standalone EmergentMatter Blender addon; merged into sdm-view 2026-07-07 with full history. It now registers like every other sdm-view feature module.

## Quick start

```bash
cd emergent-matter-sdm-view
blender --factory-startup -P scripts/open_viewer.py -- /path/to/part.sdm
```

The launcher registers the module from `src/`, auto-discovers the sdm-core venv from a sibling `emergent-matter-sdm-core` checkout (override with `SDM_PYTHON=`), imports the part, and leaves Blender open. `SDM_AUTOPLAY=1` starts the first authored animation on open.

## Prerequisites

- **Blender 4.2+**.
- A working install of **`emergent-matter-sdm-core`** that ships the `software_defined_matter.glsl` module, i.e. a Python env where `python -m software_defined_matter.glsl <file.sdm> --out <dir>` works. With `uv`, that's `<sdm-core>/.venv/bin/python`.

## Use

1. `File > Import > Software Defined Matter (.sdm)` (or the quick-start launcher) → pick a `.sdm` (e.g. `examples/hollow_cylinder_with_hinge.sdm` in `sdm-core`).
2. A wireframe bbox placeholder appears; the SDF ray-marches in the viewport.
3. `N`-panel → **SDM** tab. The emitter's control manifest drives the panel:
   - **live** controls scrub a GLSL uniform: instant, keyframable;
   - **re-emit** / **topology** controls stage values; *Rebuild Part* writes them into the `.sdm`, re-runs the part's declared generator, re-emits GLSL, and swaps the shader (auto-triggered, debounced, threaded);
   - authored **animations** play as keyframed pose sweeps, with per-row stop-to-neutral;
   - **components** get stable distinct tints (SDF segmentation), plus a fast-viewport toggle;
   - **cutaway** clips in material (rest) space along an X/Y/Z plane, uncapped by default so you see straight into the solid; enabling caps shades the cut face as a binary section-material indicator.
4. **Rest pose** zeroes pose DOFs; **Reset to .sdm values** restores every control to the source file's declared value.

## How it works

```
File > Import operator
    │
    ▼
sidecar.emit_artifacts ── subprocess ─►  python -m software_defined_matter.glsl
    │                                              │
    │                                              ▼
    │                                    sdf_lib.glsl + sdf_scene.glsl + meta.json
    ▼
viewport.compile_part_shader
    │   stitches shader_template.glsl + lib + scene → single fragment shader
    │   compiles via GPUShaderCreateInfo (Metal-safe)
    ▼
viewport.register_renderer(obj_name, ...)
    │
    ▼
SpaceView3D.draw_handler ── every frame ─►  fullscreen tri + ray-march
                                            (reads u_p_* from obj custom props)
```

- All SDF math evaluates in **object-local space**, then the placeholder's world matrix transforms it. Moving / rotating the placeholder moves the part with it.
- Hits write `gl_FragDepth` so the SDF z-composes with the rest of the Blender scene (grid, other meshes, etc.).
- Misses `discard`, leaving whatever was rendered behind intact.
- Progressive refinement with circuit breakers keeps the UI responsive on heavy scenes; playback drops to quarter-res draft resolution.
- All ray-march uniforms live in a single std140 UBO, so a parameter scrub is one buffer upload rather than dozens of per-uniform calls.
- `flightrec.py` is an always-on flight recorder: one JSON line per slow/notable draw to `$SDM_FLIGHT_LOG` (default `/tmp/sdm_flight.jsonl`). It exists because the choke failure mode (seconds-long back-to-back draws starving WindowServer) can kill the very session watching for it; evidence has to hit disk immediately. Cost when nothing is wrong is one `perf_counter` pair per draw.

## Known limitations

- **One material per part.** Uses `Part.computed_envelope()` (the smooth union across all material regions). Component tinting exists; true per-material colouring does not.
- **No live reload on `.sdm` save.** Re-import (or Rebuild) after editing the file by hand.
- **Uniform-scale assumption for normals.** Inverse-transpose handles moderate non-uniform scale; extreme cases may shade oddly.
- **Expression-tree leaves in DSL.** `sdm-core`'s GLSL emitter raises `NotImplementedError` for `{"type": ...}` leaves other than `$ref`; such parts fail to import with the emitter's message.

## Layout

```
src/sdm_view/blender/viewer/
  __init__.py
  prefs.py
  sidecar.py
  shader_template.glsl
  viewport.py
  import_operator.py
  panel.py
  rebuild.py
  flightrec.py

scripts/  (repo root)
  open_viewer.py
  probe_shader_cost.py
```

Each Python module opens with a docstring stating what it provides;
`shader_template.glsl` carries the same at the top of the file as a
comment.
