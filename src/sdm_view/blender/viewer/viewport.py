"""GPU shader compilation + viewport draw handler.

The flow each time a part is imported:

  1. ``sidecar.emit_artifacts`` returns the GLSL strings and uniform schema.
  2. ``compile_part_shader`` stitches them into a single fragment shader and
     compiles via ``gpu.types.GPUShader``.
  3. The handle is stashed on the placeholder Blender object via a custom
     pointer (``_RENDERER_REGISTRY``); the draw handler walks the registry
     each frame and renders every active part.

There is exactly one draw handler installed for the addon's lifetime; it
iterates the registry rather than installing one handler per part.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from . import flightrec
from .sidecar import EmissionArtifacts

_TEMPLATE_PATH = Path(__file__).parent / "shader_template.glsl"
_LIB_MARKER = "// @@SDF_LIB@@"
_SCENE_MARKER = "// @@SDF_SCENE@@"


def _dump_failed_shader(artifacts: EmissionArtifacts, fragment: str, vertex: str) -> Path:
    """Write the assembled vertex + fragment + a header to a sibling debug file.

    Lives next to the .sdm's emitted artifacts (``<sdm>_glsl/``) so it's easy
    to find. Returns the path to the written file.
    """
    import tempfile

    target_dir = (
        artifacts.artifact_dir if artifacts.artifact_dir.exists() else Path(tempfile.gettempdir())
    )
    dump = target_dir / "assembled_shader_failed.glsl"
    header = (
        "// sdm-view: shader compile failed.\n"
        "// This file contains the assembled vertex + fragment shader exactly\n"
        "// as we handed it to gpu.types.GPUShader(...). Diagnose by:\n"
        "//   1. Open a terminal and run `blender` from there. Re-import the\n"
        "//      .sdm. Blender's GPU compile log will print to that terminal.\n"
        "//   2. Or feed this file through a GLSL validator (glslangValidator,\n"
        "//      moderngl, etc.) to spot syntactic issues.\n"
        f"// source .sdm: {artifacts.sdm_path}\n"
        "// =========================================================\n"
        "// VERTEX SHADER:\n"
        "// =========================================================\n"
    )
    dump.write_text(
        header
        + vertex
        + "\n// =========================================================\n// FRAGMENT SHADER:\n// =========================================================\n"  # noqa: E501  # line kept as one string literal; wrapping it would fragment the readable text
        + fragment
    )
    return dump


# Fullscreen triangle: a single oversized tri covering NDC [-1, 1] in both
# axes. Cheaper than a quad (no diagonal hit twice).
_FULLSCREEN_TRI_POSITIONS = (
    (-1.0, -1.0),
    (3.0, -1.0),
    (-1.0, 3.0),
)

# Vertex body only. With the ``GPUShaderCreateInfo`` API the stage I/O
# (``pos`` in, ``v_ndc`` out) is declared through ``vertex_in`` / the stage
# interface, NOT with in-source ``in``/``out`` qualifiers: Metal's shader
# translation rejects the legacy bare-qualifier form. So this string is just
# the function body Blender wraps into its generated ``main``.
_VERTEX_SOURCE = """\
void main() {
    v_ndc = pos;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""


# Static (per-frame) uniforms declared in shader_template.glsl. Together
# with the emitted per-Param ``u_p_*`` floats they live in ONE std140 UBO:
# as push constants the three mat4s alone blow past the 128-byte minimum
# Vulkan guarantees, and Blender 5.2 warns on EVERY draw (and a part with
# enough live params would eventually fail to bind outright).
# (No camera-position uniform: rays originate at the unprojected near-plane
# point per pixel, which is correct in both perspective and orthographic
# views: an eye origin breaks the axis-gizmo/numpad ortho snaps.)
#
# Member order is the std140 layout contract shared by _ubo_layout()'s GLSL
# struct and its Python float offsets: mat4s first, then vec3s (each padded
# to a vec4 slot), then scalars: sequential offsets, no interior padding.
_STATIC_UBO_MEMBERS = (
    ("u_persp", "mat4"),
    ("u_persp_inv", "mat4"),
    ("u_object_inv", "mat4"),
    ("u_bbox_min", "vec3"),
    ("u_bbox_max", "vec3"),
    ("u_cut_normal", "vec3"),
    ("u_cut_offset", "float"),
    ("u_cut_on", "float"),
    ("u_cut_caps", "float"),
    ("u_fast", "float"),
    ("u_pack_depth", "float"),
    ("u_comp_tint", "float"),
    ("u_pattern", "float"),
)


def _ubo_layout(artifacts: EmissionArtifacts):
    """(glsl_struct_source, defines_source, layout, total_floats).

    ``layout`` maps uniform name -> (float_offset, kind). The struct members
    drop the ``u_`` prefix so the ``#define u_x sdm_params.x`` aliases (which
    keep the template + emitted scene sources untouched) can't self-refer.
    Every emitted ``u_p_*`` param is a float appended after the statics.
    """
    members = list(_STATIC_UBO_MEMBERS) + [(u.name, "float") for u in artifacts.uniforms]
    struct_lines = ["struct SDMParams {"]
    defines = []
    layout: dict[str, tuple] = {}
    ofs = 0
    for name, kind in members:
        member = name[2:] if name.startswith("u_") else f"m_{name}"
        if kind == "mat4":
            struct_lines.append(f"  mat4 {member};")
            defines.append(f"#define {name} sdm_params.{member}")
            layout[name] = (ofs, "mat4")
            ofs += 16
        elif kind == "vec3":
            # vec4 slot: std140 vec3 alignment is 16 bytes anyway, and a
            # full vec4 keeps the Python offsets trivially sequential.
            struct_lines.append(f"  vec4 {member};")
            defines.append(f"#define {name} (sdm_params.{member}.xyz)")
            layout[name] = (ofs, "vec3")
            ofs += 4
        else:
            struct_lines.append(f"  float {member};")
            defines.append(f"#define {name} sdm_params.{member}")
            layout[name] = (ofs, "float")
            ofs += 1
    # std140 struct size rounds up to a vec4 multiple; pad the buffer too.
    pad = (-ofs) % 4
    for i in range(pad):
        struct_lines.append(f"  float _pad{i};")
    ofs += pad
    struct_lines.append("};")
    return "\n".join(struct_lines), "\n".join(defines), layout, ofs


# Custom-property names on the placeholder that drive the cutaway uniforms.
CUT_PROPS = ("sdm_cut_on", "sdm_cut_axis", "sdm_cut_flip", "sdm_cut_offset")


# Lines the fragment source declares in-source that must be removed because
# the equivalent is now declared through the CreateInfo API:
#   - the six static ``uniform ...;`` (from shader_template.glsl)
#   - ``in vec2 v_ndc;`` / ``out vec4 fragColor;`` (stage I/O)
#   - the emitted scene's per-param ``uniform float u_p_*;`` (push_constants)
# Match a leading ``uniform``/``in``/``out`` storage qualifier at statement
# start, tolerating leading whitespace and a trailing ``// ...`` comment.
_UNIFORM_DECL_RE = re.compile(r"^[ \t]*uniform[ \t]+\w+[ \t]+\w+[ \t]*;.*$", re.MULTILINE)
_STAGE_IO_RE = re.compile(
    r"^[ \t]*(?:in[ \t]+vec2[ \t]+v_ndc|out[ \t]+vec4[ \t]+fragColor)[ \t]*;.*$",
    re.MULTILINE,
)


def _strip_inline_declarations(src: str) -> str:
    """Remove bare ``uniform``/stage-``in``/``out`` declarations.

    Everything those names refer to is now declared via the CreateInfo API
    (push_constants + stage interface), so leaving the in-source declarations
    in place produces duplicate-symbol / illegal-qualifier errors on Metal.
    References to the names elsewhere in the body are left untouched.
    """
    src = _UNIFORM_DECL_RE.sub("", src)
    src = _STAGE_IO_RE.sub("", src)
    return src  # noqa: RET504  # named for readability at the return site


# ---------------------------------------------------------------------------
# Shader assembly
# ---------------------------------------------------------------------------


def _load_template() -> str:
    return _TEMPLATE_PATH.read_text()


def assemble_fragment_source(artifacts: EmissionArtifacts) -> str:
    """Stitch the template, the SDM lib, and the per-part scene into one
    fragment shader **body** ready for ``GPUShaderCreateInfo.fragment_source``.

    The static uniforms, the ``in vec2 v_ndc`` / ``out vec4 fragColor`` stage
    I/O, and the emitted scene's per-param ``uniform float u_p_*;`` lines are
    all stripped: their equivalents are declared through the CreateInfo API
    (push_constants + stage interface). Every *reference* to those names in
    the code is left intact.
    """
    tpl = _load_template()
    if _LIB_MARKER not in tpl or _SCENE_MARKER not in tpl:
        raise RuntimeError(
            f"shader_template.glsl is missing required marker(s): "
            f"{_LIB_MARKER!r}, {_SCENE_MARKER!r}"
        )
    scene = artifacts.scene_source
    if "sdf_scene_comp" not in scene:
        # Artifact emitted before component segmentation existed: stub the
        # id function so the template compiles (everything is component 0).
        scene += "\n\nint sdf_scene_comp(vec3 p) { return 0; }\n"
    if "sdf_scene_rcut" not in scene:
        # Pre-material-space-cutaway artifact: fall back to the old
        # world-space per-frame clip, and rest coordinates == query.
        scene += (
            "\n\nfloat sdf_scene_rcut(vec3 p, vec3 cn, float co) {"
            " return max(sdf_scene(p), dot(p, cn) - co); }\n"
            "vec3 sdm_rest_point(vec3 p, int cid) { return p; }\n"
        )
    assembled = tpl.replace(_LIB_MARKER, artifacts.lib_source).replace(_SCENE_MARKER, scene)
    return _strip_inline_declarations(assembled)


# ---------------------------------------------------------------------------
# Renderer (one per imported part)
# ---------------------------------------------------------------------------


@dataclass
class PartRenderer:
    """Everything needed to render one imported part each frame."""

    object_name: str
    artifacts: EmissionArtifacts
    shader: gpu.types.GPUShader
    # GPUBatch. gpu.types.* resolves as Any under ignore_missing_imports (no
    # typed bpy stubs are declared as a dependency), so Any here is the
    # honest type, not object -- object has no attributes at all, which is
    # what turned every renderer.batch.draw(...) call into a false positive.
    batch: Any
    uniform_names: list[str] = field(default_factory=list)
    # Half-res fast path: cached offscreen keyed by (w, h); freed on resize.
    offscreen: object = None
    offscreen_size: tuple = (0, 0)
    # Progressive refinement: a FULL-resolution, full-quality frame rendered
    # by a debounced timer once the inputs stop changing; composited in
    # place of the live half-res image while it still matches.
    refined: object = None
    refined_size: tuple = (0, 0)
    refined_key: object = None
    # Table-mode payload textures, bound on every draw. None means the
    # emitter inlined that payload in the shader source instead, so there is
    # nothing to bind. See _TABLE_BINDINGS for the formats.
    poly_tex: object = None
    sweep_tex: object = None
    grid_tex: object = None
    # std140 UBO backing every uniform (statics + u_p_*): the GPUUniformBuf,
    # the name -> (float_offset, kind) layout, and a reusable float scratch
    # list the packer fills before each ubo.update(). Any, not object, for
    # the same reason as `batch` above -- always a real GPUUniformBuf by the
    # time _upload_uniforms runs; the None default is a pre-construction
    # placeholder, not a state any draw call has to handle.
    ubo: Any = None
    ubo_layout: dict[str, tuple] = field(default_factory=dict)
    ubo_scratch: list = field(default_factory=list)


def _build_create_info(
    artifacts: EmissionArtifacts, fragment: str
) -> gpu.types.GPUShaderCreateInfo:
    """Assemble a ``GPUShaderCreateInfo`` for the ray-march shader.

    Static uniforms + per-Param uniforms live in one std140 UBO
    (``sdm_params``): see _STATIC_UBO_MEMBERS. The original ``u_*`` names
    are #define-aliased onto the struct members so the template and the
    emitted scene compile unmodified. Stage I/O is declared through a smooth
    ``VEC2 v_ndc`` interface; ``gl_FragDepth`` writes are enabled via
    ``depth_write('ANY')``.
    """
    for spec in artifacts.uniforms:
        if spec.glsl_type != "float":
            raise RuntimeError(
                f"Unsupported uniform glsl_type {spec.glsl_type!r} for "
                f"{spec.name!r}; the sdm_params UBO packs scalar floats only."
            )

    iface = gpu.types.GPUStageInterfaceInfo("sdm_vp_iface")
    iface.smooth("VEC2", "v_ndc")

    info = gpu.types.GPUShaderCreateInfo()

    # Stage interface + attributes.
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "fragColor")

    struct_src, defines_src, _, _ = _ubo_layout(artifacts)
    info.typedef_source(struct_src)
    info.uniform_buf(0, "SDMParams", "sdm_params")
    fragment = defines_src + "\n" + fragment

    # Table-mode payloads: the emitted lib references these samplers, and its
    # in-source `uniform sampler2D` declarations are stripped with the rest,
    # so declare them here and bind the textures on every draw. Only declare
    # the ones this emission actually carries; a sampler the shader never
    # fetches still costs a binding slot.
    for slot, binding in enumerate(_TABLE_BINDINGS):
        name, table_attr = binding[0], binding[3]
        if getattr(artifacts, table_attr):
            info.sampler(slot, "FLOAT_2D", name)

    # We write gl_FragDepth to z-compose with the rest of the viewport. On
    # Metal this MUST be declared or the write is a no-op / compile error.
    info.depth_write("ANY")

    info.vertex_source(_VERTEX_SOURCE)
    info.fragment_source(fragment)
    return info


# One row per table-mode payload sdm-core can emit, in sampler-slot order:
# (uniform name, texture format, floats per texel, artifacts table field,
# artifacts row-width field). The stride is what the matching fetch helper in
# sdm-core's lib.glsl reads per texel, so it is part of the wire contract and
# not a free choice: sdm_poly_fetch reads RG, sdm_sweep_fetch reads RGBA, and
# sdm_grid_fetch reads R.
_TABLE_BINDINGS = (
    ("u_sdm_poly", "RG32F", 2, "poly_table", "poly_tex_width", "poly_tex"),
    ("u_sdm_sweep", "RGBA32F", 4, "sweep_table", "sweep_tex_width", "sweep_tex"),
    ("u_sdm_grid", "R32F", 1, "grid_table", "grid_tex_width", "grid_tex"),
)


def _build_table_texture(artifacts: EmissionArtifacts, binding: tuple):
    """Texture holding one of the emitter's table-mode payloads.

    Row width is fixed by the emitter, which bakes the matching SDM_*_TEX_W
    into sdf_lib.glsl. It must equal the actual texture width or every fetch
    indexes the wrong row. Built at import time: never in a draw callback.
    """
    _name, fmt, stride, table_attr, width_attr, _tex_attr = binding
    table = getattr(artifacts, table_attr)
    if not table:
        return None
    w = getattr(artifacts, width_attr)
    if w <= 0:
        # The emitter reported a payload but no row width to read it by.
        # Guessing one would silently render scrambled geometry.
        raise RuntimeError(
            f"Emission carries {table_attr} but {width_attr} is {w!r}; "
            f"cannot size the {fmt} texture for {_name}."
        )
    n_texels = -(-len(table) // stride)  # ceil-div: a short final texel pads
    h = max(1, -(-n_texels // w))  # ceil-div
    data = list(table) + [0.0] * (w * h * stride - len(table))
    buf = gpu.types.Buffer("FLOAT", w * h * stride, data)
    return gpu.types.GPUTexture((w, h), format=fmt, data=buf)


def compile_part_shader(artifacts: EmissionArtifacts) -> PartRenderer:
    import time

    # Composite shader for the fast path: built here (import time), never
    # inside a draw callback.
    try:
        _ensure_composite()
    except Exception as exc:
        _FAST_PATH_BROKEN.append(True)
        flightrec.log("fast_path_broken", where="ensure_composite", exc=repr(exc))

    t0 = time.perf_counter()
    fragment = assemble_fragment_source(artifacts)
    try:
        info = _build_create_info(artifacts, fragment)
        shader = gpu.shader.create_from_info(info)
    except Exception as exc:
        # Blender's compile log goes to stdout (terminal-launched Blender on
        # Linux/macOS, Window > Toggle System Console on Windows). The
        # exception message itself only says "see console for more details",
        # so dump the assembled fragment to a known path for offline diagnosis.
        dump_path = _dump_failed_shader(artifacts, fragment, _VERTEX_SOURCE)
        raise RuntimeError(
            f"{exc}\n"
            f"Assembled fragment shader dumped to: {dump_path}\n"
            f"(Open a terminal and re-launch Blender from there to see the "
            f"GPU driver's compile log.)"
        ) from exc
    batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": _FULLSCREEN_TRI_POSITIONS},
    )
    _, _, layout, total_floats = _ubo_layout(artifacts)
    scratch = [0.0] * total_floats
    ubo = gpu.types.GPUUniformBuf(gpu.types.Buffer("FLOAT", total_floats, scratch))
    flightrec.log(
        "shader_create",
        part=Path(artifacts.sdm_path).stem,
        ms=round((time.perf_counter() - t0) * 1000, 1),
        poly_tex_w=artifacts.poly_tex_width if artifacts.poly_table else None,
        poly_len=len(artifacts.poly_table or ()),
        sweep_len=len(artifacts.sweep_table or ()),
        grid_len=len(artifacts.grid_table or ()),
        n_uniforms=len(artifacts.uniforms),
    )
    return PartRenderer(
        object_name="",  # filled in by the import operator
        artifacts=artifacts,
        shader=shader,
        batch=batch,
        uniform_names=[u.name for u in artifacts.uniforms],
        poly_tex=_build_table_texture(artifacts, _TABLE_BINDINGS[0]),
        sweep_tex=_build_table_texture(artifacts, _TABLE_BINDINGS[1]),
        grid_tex=_build_table_texture(artifacts, _TABLE_BINDINGS[2]),
        ubo=ubo,
        ubo_layout=layout,
        ubo_scratch=scratch,
    )


# ---------------------------------------------------------------------------
# Half-res composite pass (fast viewport)
# ---------------------------------------------------------------------------
# The raymarch renders into a half-size RGBA16F offscreen with window depth
# packed into alpha (alpha < 0 = miss sentinel), then this pass upscales it
# onto the viewport, restoring depth via gl_FragDepth so z-compositing with
# the rest of the scene still works. 4x fewer marches per redraw.

_COMPOSITE_FRAG = """
void main() {
    vec2 uv = v_ndc * 0.5 + 0.5;
    vec4 c = texture(img, uv);
    if (c.a < 0.0) {
        discard;               // miss sentinel: keep the scene's pixels
    }
    fragColor = vec4(c.rgb, 1.0);
    gl_FragDepth = c.a;
}
"""

_COMPOSITE: list[Any] = []  # built at import time: [shader, batch]

# Hard kill-switch: if the fast path ever raises during a draw, it stays off
# for the session (the accurate path is always correct, just slower).
_FAST_PATH_BROKEN: list[bool] = []


def _ensure_composite() -> None:
    """Build the composite shader OUTSIDE any draw callback.

    Creating GPU objects (PSO compile, batches) inside an active render pass
    is exactly the kind of thing Metal kills the process over: everything
    the fast path needs is created here, at import time.
    """
    if _COMPOSITE:
        return
    iface = gpu.types.GPUStageInterfaceInfo("sdm_vp_comp_iface")
    iface.smooth("VEC2", "v_ndc")
    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "fragColor")
    info.sampler(0, "FLOAT_2D", "img")
    info.depth_write("ANY")
    info.vertex_source(_VERTEX_SOURCE)
    info.fragment_source(_COMPOSITE_FRAG)
    shader = gpu.shader.create_from_info(info)
    batch = batch_for_shader(shader, "TRIS", {"pos": _FULLSCREEN_TRI_POSITIONS})
    _COMPOSITE.extend([shader, batch])


def _request_offscreen(renderer: PartRenderer, w: int, h: int):
    """Return a matching offscreen, or None while one is (re)built.

    GPUOffScreen allocation is deferred to a main-thread timer so it never
    happens inside a draw callback; the frame that observes a size change
    just renders via the accurate path once.
    """
    if renderer.offscreen_size == (w, h):
        # Matching size: either ready, or a build is already scheduled
        # (offscreen still None): never schedule twice for the same size.
        return renderer.offscreen

    # Mark pending FIRST so subsequent frames don't re-schedule.
    renderer.offscreen_size = (w, h)
    old = renderer.offscreen
    renderer.offscreen = None
    name = renderer.object_name

    def _build():
        if old is not None:
            try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
                old.free()
            except Exception:
                pass
        r = _RENDERER_REGISTRY.get(name)
        if r is None or r.offscreen_size != (w, h):
            return  # renderer gone or resized again since scheduling
        try:
            r.offscreen = gpu.types.GPUOffScreen(w, h, format="RGBA16F")
        except Exception:
            _FAST_PATH_BROKEN.append(True)
        return  # one-shot timer

    bpy.app.timers.register(_build, first_interval=0.0)
    return None


def _bind_table_textures(renderer: PartRenderer) -> None:
    """Bind every table-mode payload texture this emission carries.

    Sampler bindings don't persist across ``shader.bind()`` calls the way
    push_constants do, so every draw site re-binds after bind().
    """
    for binding in _TABLE_BINDINGS:
        name, tex_attr = binding[0], binding[5]
        tex = getattr(renderer, tex_attr)
        if tex is None:
            continue
        try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
            renderer.shader.uniform_sampler(name, tex)
        except ValueError:
            pass  # linker dropped every fetch (e.g. all-pruned debug scene)


def _upload_uniforms(renderer: PartRenderer, binds, overrides=None) -> None:
    """Pack (name, value) pairs into the sdm_params UBO and bind it.

    Matrices upload column-major (std140 mat4 = four column vec4s; matching
    what uniform_float("MAT4") used to do internally). The scratch list
    persists between calls, so a partial update (e.g. flipping only
    u_pack_depth for the accurate path) keeps every other member's last
    value. Names absent from the layout are ignored: a stale bind list must
    not be able to crash a redraw.
    """
    vals = dict(binds)
    if overrides:
        vals.update(overrides)
    buf = renderer.ubo_scratch
    layout = renderer.ubo_layout
    for name, v in vals.items():
        ent = layout.get(name)
        if ent is None:
            continue
        ofs, kind = ent
        if kind == "mat4":
            k = ofs
            for col in range(4):
                for row in range(4):
                    buf[k] = v[row][col]
                    k += 1
        elif kind == "vec3":
            buf[ofs] = v[0]
            buf[ofs + 1] = v[1]
            buf[ofs + 2] = v[2]
        else:
            buf[ofs] = float(v)
    renderer.ubo.update(gpu.types.Buffer("FLOAT", len(buf), buf))
    try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
        renderer.shader.uniform_block("sdm_params", renderer.ubo)
    except ValueError:
        pass  # block optimised out (degenerate debug scene)


def warmup_pso(renderer: PartRenderer) -> float:
    """Force the Metal pipeline (PSO) compile NOW, off the draw handler.

    ``gpu.shader.create_from_info`` only runs the GLSL->MSL translation
    (fast); Metal compiles the actual pipeline lazily on the shader's FIRST
    draw. For heavy emitted scenes that compile is multi-second (measured
    31 s on a pre-LOD assembly), and without this warm-up it lands inside the
    viewport draw handler on the main thread: the whole UI beachballs and
    reads as a hang. A 1x1 offscreen draw with the initial uniform values
    pays that cost once, here, where the import operator can report it.

    Returns the warm-up wall-clock in seconds.
    """
    import time

    t0 = time.perf_counter()
    shader = renderer.shader
    offscreen = gpu.types.GPUOffScreen(1, 1)
    try:
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            shader.bind()
            _bind_table_textures(renderer)

            ident = Matrix.Identity(4)
            binds = [
                ("u_persp", ident),
                ("u_persp_inv", ident),
                ("u_object_inv", ident),
                ("u_bbox_min", Vector(renderer.artifacts.bbox[0])),
                ("u_bbox_max", Vector(renderer.artifacts.bbox[1])),
                ("u_cut_normal", Vector((0.0, 1.0, 0.0))),
                ("u_cut_offset", 0.0),
                ("u_cut_on", 0.0),
            ]
            binds += [(spec.name, float(spec.initial)) for spec in renderer.artifacts.uniforms]
            _upload_uniforms(renderer, binds)
            renderer.batch.draw(shader)
            # Read a pixel back: blocks until the GPU finished the command
            # buffer, i.e. until the PSO compile actually happened.
            fb.read_color(0, 0, 1, 1, 4, 0, "FLOAT")
    finally:
        offscreen.free()
    dt = time.perf_counter() - t0
    flightrec.log("pso_warmup", part=Path(renderer.artifacts.sdm_path).stem, ms=round(dt * 1000, 1))
    return dt


# ---------------------------------------------------------------------------
# Active renderer registry + the single draw handler
# ---------------------------------------------------------------------------

# Keyed by Blender object name so renderers survive name lookups across
# undo / Python state resets. Values disappear when the object is deleted.
_RENDERER_REGISTRY: dict[str, PartRenderer] = {}
_DRAW_HANDLER_REF: list[object] = []  # at most one element; holds the handle


def register_renderer(obj_name: str, renderer: PartRenderer) -> None:
    renderer.object_name = obj_name
    _RENDERER_REGISTRY[obj_name] = renderer


def unregister_renderer(obj_name: str) -> None:
    renderer = _RENDERER_REGISTRY.pop(obj_name, None)
    _REFINE_REQUESTS.pop(obj_name, None)
    _REFINE_LAST.pop(obj_name, None)
    if renderer is None:
        return
    for attr in ("offscreen", "refined"):
        offs = getattr(renderer, attr)
        if offs is not None:
            try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
                offs.free()
            except Exception:
                pass
            setattr(renderer, attr, None)


def get_renderer(obj_name: str) -> PartRenderer | None:
    return _RENDERER_REGISTRY.get(obj_name)


# ---------------------------------------------------------------------------
# Progressive refinement
# ---------------------------------------------------------------------------
# Interaction runs at draft resolution; once the inputs (view, uniforms,
# size) have been STABLE for a beat, a timer renders ONE full-resolution,
# full-quality frame off the interaction path and the draw handler
# composites that instead: crisp when idle, fluid when moving. Requests
# carry a monotonic timestamp; the timer only renders a request that hasn't
# been superseded for REFINE_SETTLE_S (so dragging never hiccups).

REFINE_SETTLE_S = 0.25
REFINE_MIN_INTERVAL_S = 1.0  # never re-render a part's refined frame faster
REFINE_RUNAWAY_N = 8  # renders within REFINE_RUNAWAY_WINDOW_S ...
REFINE_RUNAWAY_WINDOW_S = 10.0  # ... = a ping-pong loop; disable refinement

# name -> (key, binds, (w, h), t_requested)
_REFINE_REQUESTS: dict[str, tuple] = {}
_REFINE_LAST: dict[str, tuple] = {}  # name -> (key, t_rendered)
_REFINE_TIMES: list[float] = []  # recent render timestamps (runaway det.)
_REFINE_DISABLED: list[bool] = []


def _request_refine(name: str, key, binds, size) -> None:
    import time

    if _REFINE_DISABLED:
        return
    last = _REFINE_LAST.get(name)
    if last is not None and last[0] == key:
        return  # the refined frame for exactly these inputs already exists
    prev = _REFINE_REQUESTS.get(name)
    if prev is not None and prev[0] == key:
        return  # same inputs already queued: keep its timestamp
    _REFINE_REQUESTS[name] = (key, binds, size, time.perf_counter())
    if not bpy.app.timers.is_registered(_refine_tick):
        bpy.app.timers.register(_refine_tick, first_interval=REFINE_SETTLE_S)


def _refine_tick():
    import time

    now = time.perf_counter()
    for name in list(_REFINE_REQUESTS):
        key, binds, (w, h), t_req = _REFINE_REQUESTS[name]
        if now - t_req < REFINE_SETTLE_S:
            continue  # still moving: check again next tick
        last = _REFINE_LAST.get(name)
        if last is not None and now - last[1] < REFINE_MIN_INTERVAL_S:
            continue  # cooldown: re-check next tick
        # Runaway breaker: refined frames should be RARE (once per pause).
        # A steady stream means the inputs never truly settle (a key that
        # drifts every redraw) and each render blocks the main thread:
        # the "Blender is not responsive" failure. Turn the feature off.
        _REFINE_TIMES.append(now)
        while _REFINE_TIMES and now - _REFINE_TIMES[0] > REFINE_RUNAWAY_WINDOW_S:
            _REFINE_TIMES.pop(0)
        if len(_REFINE_TIMES) > REFINE_RUNAWAY_N:
            _REFINE_DISABLED.append(True)
            _REFINE_REQUESTS.clear()
            flightrec.log("refine_runaway_disabled", part=name)
            print(
                "[sdm-viewer] progressive refinement looped (inputs never "
                "settle): disabled for this session."
            )
            return None
        _REFINE_REQUESTS.pop(name)
        r = _RENDERER_REGISTRY.get(name)
        if r is None:
            continue
        try:
            if r.refined is None or r.refined_size != (w, h):
                if r.refined is not None:
                    r.refined.free()
                    r.refined = None
                r.refined = gpu.types.GPUOffScreen(w, h, format="RGBA16F")
                r.refined_size = (w, h)
            shader = r.shader
            with r.refined.bind():
                fb = gpu.state.active_framebuffer_get()
                fb.clear(color=(0.0, 0.0, 0.0, -1.0), depth=1.0)
                gpu.state.depth_test_set("NONE")
                shader.bind()
                _bind_table_textures(r)
                # Full RESOLUTION, fast-march quality: the visible win of the
                # refined frame is pixel count; the accurate march's tighter
                # epsilon is invisible at viewport scale but ~2.5x the cost;
                # and this render blocks the main thread, so every saved ms
                # is one less ms of post-interaction hitch.
                _upload_uniforms(r, binds, {"u_fast": 1.0, "u_pack_depth": 1.0})
                r.batch.draw(shader)
            prev_key = r.refined_key
            r.refined_key = key
            _REFINE_LAST[name] = (key, time.perf_counter())
            flightrec.log(
                "refine",
                part=name,
                w=w,
                h=h,
                ms=round((time.perf_counter() - now) * 1000, 1),
                key=hash(key) % 10**9,
                prev=(hash(prev_key) % 10**9 if prev_key is not None else None),
            )
        except Exception as exc:
            _FAST_PATH_BROKEN.append(True)
            flightrec.log("fast_path_broken", where="refine_tick", part=name, exc=repr(exc))
            import traceback

            traceback.print_exc()
            continue
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
                    break  # one region shows it; don't broadcast redraws
            break
    return REFINE_SETTLE_S if _REFINE_REQUESTS else None


def _draw_one(renderer: PartRenderer, context) -> None:
    obj = bpy.data.objects.get(renderer.object_name)
    if obj is None:
        # Backing object was deleted. Prune the renderer so its GPU shader +
        # batch are released, and so a future object that reuses this name can't
        # inherit this stale SDF. Safe to mutate here: the draw handler walks a
        # snapshot (list(...)) of the registry, not the live dict.
        unregister_renderer(renderer.object_name)
        return

    if obj.hide_viewport or not obj.visible_get():
        return

    region_data = context.region_data
    if region_data is None:
        return

    persp = region_data.perspective_matrix
    persp_inv = persp.inverted()

    object_inv = obj.matrix_world.inverted()

    bbox_min = Vector(renderer.artifacts.bbox[0])
    bbox_max = Vector(renderer.artifacts.bbox[1])

    # Everything the shader needs, collected as (name, value) so the SAME
    # snapshot can be replayed by the refinement timer. Matrices are copies:
    # region matrices are owned by Blender and mutate underneath us.
    binds = [
        ("u_persp", persp.copy()),
        ("u_persp_inv", persp_inv),
        ("u_object_inv", object_inv),
        ("u_bbox_min", bbox_min),
        ("u_bbox_max", bbox_max),
    ]

    # Cutaway plane from the placeholder's custom props (panel-driven).
    axis = int(obj.get("sdm_cut_axis", 1)) % 3
    sign = -1.0 if obj.get("sdm_cut_flip", 0) else 1.0
    normal = [0.0, 0.0, 0.0]
    normal[axis] = sign
    b_fast = bool(obj.get("sdm_fast", 1))
    binds += [
        ("u_cut_normal", Vector(normal)),
        ("u_cut_offset", float(obj.get("sdm_cut_offset", 0.0))),
        ("u_cut_on", 1.0 if obj.get("sdm_cut_on", 0) else 0.0),
        ("u_cut_caps", 1.0 if obj.get("sdm_cut_caps", 0) else 0.0),
        ("u_comp_tint", 1.0 if obj.get("sdm_comp_tint", 1) else 0.0),
        ("u_pattern", 1.0 if obj.get("sdm_pattern", 1) else 0.0),
    ]

    # Per-Param uniforms from the placeholder's custom properties. Two
    # sources of truth, each stale in the other's case:
    #   - KEYFRAMED props are written to the depsgraph's evaluated copy
    #     during playback/scrub; the original keeps the rest value.
    #   - SCRIPT-written props (MCP sessions, operators) land on the
    #     original; an idprop write doesn't reliably tag a depsgraph update,
    #     so the evaluated copy can hold a stale old value indefinitely.
    # So: read the evaluated copy only for props that actually have an
    # f-curve; everything else reads the original. Missing properties fall
    # back to the spec's "initial" so a stale registry can't crash a redraw.
    src = obj
    try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
        src = obj.evaluated_get(context.evaluated_depsgraph_get())
    except (AttributeError, RuntimeError):
        pass
    animated = set()
    try:
        ad = obj.animation_data
        if ad is not None and ad.action is not None:
            for fc in ad.action.fcurves:
                dp = fc.data_path
                if dp.startswith('["') and dp.endswith('"]'):
                    animated.add(dp[2:-2])
    except (AttributeError, RuntimeError):
        pass  # NLA-only setups fall back to evaluated-for-nothing; original wins
    for spec in renderer.artifacts.uniforms:
        holder = src if spec.name in animated else obj
        val = holder.get(spec.name, obj.get(spec.name, spec.initial))
        binds.append((spec.name, float(val)))

    shader = renderer.shader
    shader.bind()
    _bind_table_textures(renderer)
    _upload_uniforms(
        renderer,
        binds,
        {
            "u_fast": 1.0 if b_fast else 0.0,
            "u_pack_depth": 1.0 if b_fast else 0.0,
        },
    )

    offscreen = None
    b_playing = False
    key = None
    if b_fast and not _FAST_PATH_BROKEN and _COMPOSITE:
        # Viewport size must be read BEFORE binding the offscreen (bind()
        # swaps the framebuffer). A size mismatch defers (re)allocation to a
        # main-thread timer and falls back to the accurate path this frame.
        # During ANIMATION PLAYBACK drop to quarter res: every frame
        # re-marches the scene, so playback runs at render speed: draft
        # resolution buys ~4x smoother motion and snaps back to half-res
        # the moment playback stops.
        div = 2
        try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
            b_playing = bool(context.screen is not None and context.screen.is_animation_playing)
        except AttributeError:
            pass
        if b_playing:
            div = 4
        vx, vy, vw, vh = gpu.state.viewport_get()

        # Progressive refinement: if a full-quality frame matching EXACTLY
        # these inputs is ready, composite it and skip marching entirely.
        key = (vw, vh) + tuple(
            round(f, 6)
            for _, v in binds
            for f in (
                [x for row in v for x in row]
                if isinstance(v, Matrix)
                else list(v)
                if isinstance(v, Vector)
                else [v]
            )
        )
        if (
            renderer.refined is not None
            and renderer.refined_key == key
            and renderer.refined_size == (vw, vh)
        ):
            _composite(renderer.refined)
            return

        offscreen = _request_offscreen(renderer, max(1, vw // div), max(1, vh // div))

    if offscreen is None:
        # Accurate path: march at full viewport resolution, depth-composed
        # directly. Enable depth test so gl_FragDepth z-composes us with the
        # rest of the 3D viewport. (Partial UBO update: the scratch keeps
        # every other member's value.)
        _upload_uniforms(renderer, (), {"u_pack_depth": 0.0})
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)
        try:
            renderer.batch.draw(shader)
        finally:
            gpu.state.depth_test_set("NONE")
            gpu.state.depth_mask_set(False)
        return

    # Fast path: march into the half-size offscreen (4x fewer evaluations),
    # then composite up with depth restored from alpha. If ANYTHING here
    # throws, the fast path is disabled for the session: a slow viewport
    # beats a dead one.
    try:
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            # alpha = -1 is the miss sentinel the composite pass discards on.
            # Depth MUST be cleared and the inherited viewport depth test
            # disabled: stale offscreen depth + LESS_EQUAL rejects random
            # fragments and shreds the image.
            fb.clear(color=(0.0, 0.0, 0.0, -1.0), depth=1.0)
            gpu.state.depth_test_set("NONE")
            renderer.batch.draw(shader)

        _composite(offscreen)

        # Queue the full-quality replacement frame for when things settle.
        if not b_playing and key is not None:
            _request_refine(renderer.object_name, key, binds, (vw, vh))
    except Exception as exc:
        _FAST_PATH_BROKEN.append(True)
        flightrec.log(
            "fast_path_broken", where="draw_one", part=renderer.object_name, exc=repr(exc)
        )
        import traceback

        traceback.print_exc()
        print(
            "[sdm-viewer] fast viewport path failed: disabled for this "
            "session; using the accurate path."
        )


def _composite(offscreen) -> None:
    """Upscale an offscreen (color + alpha-packed depth) onto the viewport."""
    comp_shader, comp_batch = _COMPOSITE[0], _COMPOSITE[1]
    comp_shader.bind()
    comp_shader.uniform_sampler("img", offscreen.texture_color)
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(True)
    try:
        comp_batch.draw(comp_shader)
    finally:
        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)


# A draw over this long is already a UI hitch worth a forensic record; the
# WindowServer-starving failure mode shows up as a back-to-back stream of
# these at 10x the threshold.
_SLOW_DRAW_S = 0.15


def _draw_handler() -> None:
    import time

    context = bpy.context
    if context.region_data is None:
        return
    # Iterate over a snapshot: drawing must not mutate the registry but
    # delete-during-draw is hard to rule out across Blender updates.
    for renderer in list(_RENDERER_REGISTRY.values()):
        t0 = time.perf_counter()
        _draw_one(renderer, context)
        dt = time.perf_counter() - t0
        if dt > _SLOW_DRAW_S:
            flightrec.log(
                "slow_draw",
                part=renderer.object_name,
                ms=round(dt * 1000, 1),
                fast_broken=bool(_FAST_PATH_BROKEN),
                refine_disabled=bool(_REFINE_DISABLED),
                viewport=list(gpu.state.viewport_get()[2:]),
            )


def _on_depsgraph_update(scene, depsgraph) -> None:
    """Redraw viewports when an SDM placeholder's properties change.

    Editing a custom property in the N-panel redraws the UI region but NOT
    the 3D viewport: the ray-march only re-renders on a viewport redraw, so
    without this, slider drags appear to do nothing until the user orbits.
    """
    if not _RENDERER_REGISTRY:
        return
    for upd in depsgraph.updates:
        if upd.id.__class__.__name__ != "Object":
            continue
        if upd.id.name in _RENDERER_REGISTRY:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
            return


# ---------------------------------------------------------------------------
# (Un)register
# ---------------------------------------------------------------------------


def register() -> None:
    handle = bpy.types.SpaceView3D.draw_handler_add(_draw_handler, (), "WINDOW", "POST_VIEW")
    _DRAW_HANDLER_REF.append(handle)
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister() -> None:
    from .sidecar import cleanup_tmp_dirs

    while _DRAW_HANDLER_REF:
        handle = _DRAW_HANDLER_REF.pop()
        try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except (ValueError, RuntimeError):
            pass
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    _RENDERER_REGISTRY.clear()
    cleanup_tmp_dirs()
