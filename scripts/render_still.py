"""Render one still of an emitted SDM GLSL scene: the ray-march, not a mesh.

    /Applications/Blender.app/Contents/MacOS/Blender --factory-startup \\
        --window-geometry 60 60 480 320 \\
        --python scripts/render_still.py -- <part_glsl_dir> <out.png> [<part.sdm>]

Draws through exactly the shader the interactive viewer uses
(`viewport.assemble_fragment_source` and `viewport._build_create_info`), so what
comes out is what `open_viewer.py` would show, at a fixed camera, without a
human to drive it. That matters for a preview: a meshed render answers "what did
marching cubes make of the field", and this answers "what IS the field", which
is the question the viewer exists for.

**Windowed, not `--background`.** Blender's `gpu` module refuses to draw with no
window, so this follows `probe_shader_cost.py`: open a tiny window, defer the
work to a timer once the GL context is live, render, quit. A 480x320 window is
opened and closed; the image is whatever resolution you ask for.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import bpy  # noqa: F401  (establishes we're inside Blender)
import gpu
from mathutils import Matrix, Vector

SDM_VIEW_SRC = Path(__file__).resolve().parents[1] / "src"
if str(SDM_VIEW_SRC) not in sys.path:
    sys.path.insert(0, str(SDM_VIEW_SRC))

from sdm_view.blender.viewer import (  # noqa: E402  # must follow the sys.path patch above it
    sidecar,
    viewport,
)


def _camera(bb_min: Vector, bb_max: Vector, d_azimuth: float, d_elevation: float, d_dolly: float):
    """Three-quarter view, framed on the bbox: the angle a user would pick."""
    d_centre = (bb_min + bb_max) * 0.5
    d_radius = max((bb_max - bb_min).length * 0.5, 1e-3)
    d_az, d_el = math.radians(d_azimuth), math.radians(d_elevation)
    d_eye_dir = Vector(
        (math.cos(d_el) * math.cos(d_az), math.cos(d_el) * math.sin(d_az), math.sin(d_el))
    ).normalized()
    d_eye = d_centre + d_eye_dir * d_radius * d_dolly

    d_fwd = (d_centre - d_eye).normalized()
    d_right = d_fwd.cross(Vector((0, 0, 1))).normalized()
    d_up = d_right.cross(d_fwd)
    view = Matrix(
        (
            (d_right.x, d_right.y, d_right.z, -d_right.dot(d_eye)),
            (d_up.x, d_up.y, d_up.z, -d_up.dot(d_eye)),
            (-d_fwd.x, -d_fwd.y, -d_fwd.z, d_fwd.dot(d_eye)),
            (0, 0, 0, 1),
        )
    )
    d_near, d_far = d_radius * 0.05, d_radius * 10.0
    f = 1.0 / math.tan(math.radians(45.0) / 2.0)
    proj = Matrix(
        (
            (f, 0, 0, 0),
            (0, f, 0, 0),
            (0, 0, (d_far + d_near) / (d_near - d_far), 2 * d_far * d_near / (d_near - d_far)),
            (0, 0, -1, 0),
        )
    )
    return proj @ view


def _render() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        print("usage: ... -- <part_glsl_dir> <out.png> [<part.sdm>]")
        bpy.ops.wm.quit_blender()
        return
    glsl_dir = Path(argv[0]).resolve()
    d_out = Path(argv[1]).resolve()
    sdm_path = Path(argv[2]).resolve() if len(argv) > 2 else glsl_dir / "part.sdm"

    import os

    n_w = int(os.environ.get("SDM_STILL_W", "1200"))
    n_h = int(os.environ.get("SDM_STILL_H", "900"))
    d_az = float(os.environ.get("SDM_STILL_AZ", "-55"))
    d_el = float(os.environ.get("SDM_STILL_EL", "22"))
    d_dolly = float(os.environ.get("SDM_STILL_DOLLY", "2.5"))
    d_cut = float(os.environ.get("SDM_STILL_CUT", "0"))

    art = sidecar.load_artifacts_from_dir(glsl_dir, sdm_path)
    print(f"[still] {glsl_dir.name}: {len(art.poly_table) // 2} table vertices")

    # Use the viewer's own compile + upload path rather than a private copy.
    # Reimplementing it is how this script first rendered a blank frame twice:
    # every static uniform lives in the `sdm_params` UBO, not in individual
    # `uniform_float` calls, so hand-rolled `shader.uniform_float("u_persp", ...)`
    # raised ValueError on every single one. The local helper swallowed it.
    # The camera was never uploaded and the march ran against an identity
    # matrix. `probe_shader_cost.py` has the same bug, which is why its timings
    # were of an empty scene.
    renderer = viewport.compile_part_shader(art)
    print(f"[still] fragment {len(viewport.assemble_fragment_source(art)) / 1024:.0f} KB")

    bb_min, bb_max = Vector(art.bbox[0]), Vector(art.bbox[1])
    persp = _camera(bb_min, bb_max, d_az, d_el, d_dolly)
    binds = [
        ("u_persp", persp),
        ("u_persp_inv", persp.inverted()),
        ("u_object_inv", Matrix.Identity(4)),
        ("u_bbox_min", bb_min),
        ("u_bbox_max", bb_max),
        ("u_cut_normal", Vector((0.0, 1.0, 0.0))),
        ("u_cut_offset", 0.0),
        ("u_cut_on", d_cut),
    ]
    binds += [(s.name, float(s.initial)) for s in art.uniforms]

    renderer.shader.bind()
    viewport._bind_poly_tex(renderer)
    viewport._upload_uniforms(renderer, binds, {"u_fast": 0.0, "u_pack_depth": 0.0})

    offscreen = gpu.types.GPUOffScreen(n_w, n_h)
    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.92, 0.93, 0.95, 1.0), depth=1.0)
        renderer.batch.draw(renderer.shader)
        buf = fb.read_color(0, 0, n_w, n_h, 4, 0, "FLOAT")
    offscreen.free()

    buf.dimensions = n_w * n_h * 4
    img = bpy.data.images.new("sdm_still", n_w, n_h, alpha=True, float_buffer=True)
    img.pixels.foreach_set(list(buf))
    img.filepath_raw = str(d_out)
    img.file_format = "PNG"
    img.save()

    # A uniform frame means the march hit nothing: a broken camera, an
    # unbound table, a uniform that never uploaded. It writes a perfectly
    # valid PNG of the clear colour, so nothing downstream notices. Say so.
    d_px = list(buf)
    d_lo, d_hi = min(d_px), max(d_px)
    print(f"[still] wrote {d_out} ({n_w}x{n_h})")
    if d_hi - d_lo < 1e-4:
        print("[still] WARNING: frame is uniform. The ray-march hit nothing")


def _deferred() -> None:
    try:
        _render()
    except Exception:  # noqa: BLE001  # must still quit
        traceback.print_exc()
    bpy.ops.wm.quit_blender()


# The GL context is not live during a startup --python; defer as the probe does.
bpy.app.timers.register(_deferred, first_interval=0.5)
