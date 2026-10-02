"""Headless shader-cost probe for emitted SDM GLSL scenes.

Measures, WITHOUT opening a viewport (no GPU-watchdog / driver-wedge risk):
  1. fragment shader compile time (the pre-dedup failure mode), and
  2. per-pixel ray-march cost, by timing single offscreen draws at tiny
     resolutions and escalating only while each draw stays under a budget.

Run (windowed: Blender's `gpu` module refuses to draw in --background; the
probe defers to a timer after startup, prints results, and quits Blender):
  /Applications/Blender.app/Contents/MacOS/Blender --factory-startup \
      --window-geometry 60 60 480 320 \
      --python scripts/probe_shader_cost.py -- <part_glsl_dir> [<part.sdm>]

The <part_glsl_dir> must contain sdf_lib.glsl, sdf_scene.glsl, meta.json
(a prior emission; the probe does not re-run the emitter).
"""

from __future__ import annotations

import math
import sys
import time
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


def perspective_matrices(bbox_min: Vector, bbox_max: Vector):
    """Camera looking at the bbox from a 3/4 view, like a user would."""
    center = (bbox_min + bbox_max) * 0.5
    radius = max((bbox_max - bbox_min).length * 0.5, 1e-3)
    eye_dir = Vector((1.0, -1.0, 0.6)).normalized()
    eye = center + eye_dir * radius * 2.5

    fwd = (center - eye).normalized()
    right = fwd.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(fwd)
    view = Matrix(
        (
            (right.x, right.y, right.z, -right.dot(eye)),
            (up.x, up.y, up.z, -up.dot(eye)),
            (-fwd.x, -fwd.y, -fwd.z, fwd.dot(eye)),
            (0, 0, 0, 1),
        )
    )
    near, far = radius * 0.05, radius * 10.0
    f = 1.0 / math.tan(math.radians(45.0) / 2.0)
    proj = Matrix(
        (
            (f, 0, 0, 0),
            (0, f, 0, 0),
            (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
            (0, 0, -1, 0),
        )
    )
    return proj @ view


def timed_draw(shader, batch, offscreen, size: int):
    """Returns (milliseconds, hit_anything).

    The hit flag matters more than it looks: a march that reaches nothing is
    the fastest possible shader, so an empty scene reports superb numbers. This
    probe did exactly that for a whole session. Reading the full frame rather
    than one pixel costs nothing at these sizes and tells us whether the timing
    is of anything at all.
    """
    t0 = time.perf_counter()
    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
        batch.draw(shader)
        # Reading back forces the GPU to finish the command buffer, so the
        # wall-clock below includes actual shader execution.
        px = fb.read_color(0, 0, size, size, 4, 0, "FLOAT")
    d_ms = (time.perf_counter() - t0) * 1000.0
    # read_color hands back a (h, w, 4) Buffer; iterating it yields ROWS, not
    # floats. Reshape before flattening or the comparison is Buffer > Buffer.
    px.dimensions = size * size * 4
    flat = list(px)
    return d_ms, (max(flat) - min(flat)) > 1e-6


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not argv:
        print("usage: ... -- <part_glsl_dir> [<part.sdm>]")
        sys.exit(2)
    glsl_dir = Path(argv[0]).resolve()
    sdm_path = Path(argv[1]).resolve() if len(argv) > 1 else glsl_dir / "unknown.sdm"

    art = sidecar.load_artifacts_from_dir(glsl_dir, sdm_path)
    frag = viewport.assemble_fragment_source(art)
    print(f"[probe] part={glsl_dir.name} fragment={len(frag) / 1024:.0f} KB")

    # Compile and upload through the viewer's own path. Hand-rolling it is
    # how this probe reported 196 ms compile and sub-millisecond warm draws for
    # a scene where NOTHING uploaded: every static uniform lives in the
    # `sdm_params` UBO, so `shader.uniform_float("u_persp", ...)` raised
    # ValueError on each one and the local `_set` swallowed it. The march ran
    # against an identity camera and hit nothing, which is very fast.
    t0 = time.perf_counter()
    renderer = viewport.compile_part_shader(art)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[probe] compile: {compile_ms:.0f} ms")

    shader, batch = renderer.shader, renderer.batch

    bb_min, bb_max = Vector(art.bbox[0]), Vector(art.bbox[1])
    persp = perspective_matrices(bb_min, bb_max)

    import os as _os

    binds = [
        ("u_persp", persp),
        ("u_persp_inv", persp.inverted()),
        ("u_object_inv", Matrix.Identity(4)),
        ("u_bbox_min", bb_min),
        ("u_bbox_max", bb_max),
        ("u_cut_normal", Vector((0.0, 1.0, 0.0))),
        ("u_cut_offset", 0.0),
        ("u_cut_on", 0.0),
    ]
    binds += [(s.name, float(s.initial)) for s in art.uniforms]

    shader.bind()
    # Without this the sampler reads zeros: every polygon degenerates and the
    # cost measured is the cost of an EMPTY scene.
    viewport._bind_poly_tex(renderer)
    if art.poly_table:
        print(f"[probe] bound u_sdm_poly: {len(art.poly_table) // 2} vertices")
    # PROBE_FAST=1 measures the fast-viewport march (halved steps, relaxed
    # epsilon) instead of the accurate one.
    viewport._upload_uniforms(
        renderer,
        binds,
        {
            "u_fast": float(_os.environ.get("PROBE_FAST", "0")),
            "u_pack_depth": 0.0,
        },
    )

    # Escalate resolution only while the previous draw was fast. A wedged-GPU
    # frame at viewport size is ~seconds; we never get anywhere near that.
    BUDGET_MS = 400.0
    size = 4
    results = []
    while size <= 512:
        offscreen = gpu.types.GPUOffScreen(size, size)
        try:
            ms, _ = timed_draw(shader, batch, offscreen, size)
            # Second draw at same size: first includes pipeline warm-up.
            ms2, b_hit = timed_draw(shader, batch, offscreen, size)
        finally:
            offscreen.free()
        per_mpx = ms2 / (size * size) * 1e6
        results.append((size, ms, ms2, per_mpx))
        print(
            f"[probe] {size:>4}x{size:<4} first={ms:8.1f} ms  warm={ms2:8.1f} ms"
            f"  -> {per_mpx:10.0f} ms/Mpx"
            f"{'' if b_hit else '   <-- EMPTY FRAME, timing is meaningless'}"
        )
        if ms2 > BUDGET_MS:
            print(f"[probe] stopping: draw exceeded {BUDGET_MS:.0f} ms budget")
            break
        size *= 2

    if results:
        _, _, warm, per_mpx = results[-1]
        est_frame = per_mpx * 2.0  # ~2 Mpx typical Blender viewport
        print(f"[probe] extrapolated full-viewport (~2 Mpx) frame: {est_frame / 1000.0:.1f} s")
    print("[probe] done")


def _deferred() -> None:
    try:
        main()
    except Exception as exc:  # print full cause; a bare timer swallows it
        import traceback

        traceback.print_exc()
        print(f"[probe] FAILED: {exc}")
    finally:
        # quit_blender hard-exits without flushing Python's block-buffered
        # stdout (lost output when redirected to a file/pipe).
        sys.stdout.flush()
        sys.stderr.flush()
        bpy.ops.wm.quit_blender()


# GPU drawing needs the window's context to be fully alive; a startup --python
# script runs too early on some builds. One timer tick later is safe.
bpy.app.timers.register(_deferred, first_interval=0.5)
