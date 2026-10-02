"""``File > Import > Software Defined Matter (.sdm)`` operator.

Runs the sdm-core CLI on the chosen ``.sdm``, compiles the resulting GLSL
into a viewport shader, creates a placeholder Empty sized to the SDF's
inferred bbox, and stores each uniform's initial value as a custom property
on the placeholder so the sidebar panel can drive them as sliders.
"""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy_extras.io_utils import ImportHelper

from .prefs import get_prefs
from .sidecar import SidecarError, emit_artifacts
from .viewport import (
    compile_part_shader,
    register_renderer,
    unregister_renderer,
    warmup_pso,
)

# Matches EmissionArtifacts.bbox in sidecar.py -- the (lo, hi) corner pair
# every function below receives as `artifacts.bbox`.
Bbox = tuple[tuple[float, float, float], tuple[float, float, float]]


class SDM_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = "sdm.import_sdm"
    bl_label = "Import Software Defined Matter (.sdm)"
    bl_description = (
        "Run the sdm-core GLSL emitter on a .sdm file and render the SDF "
        "in the 3D viewport via ray-marching."
    )
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".sdm"
    # bpy.props.*Property() as an annotation is how Blender registers this
    # field; mypy rejects a call in annotation position (see prefs.py's
    # SDMViewerPreferences for the full explanation). Structural, not a bug.
    filter_glob: bpy.props.StringProperty(  # type: ignore[valid-type]
        default="*.sdm",
        options={"HIDDEN"},
        maxlen=255,
    )

    def execute(self, context):
        prefs = get_prefs(context)
        sdm_path = Path(self.filepath)

        try:
            artifacts = emit_artifacts(
                sdm_path,
                python_executable=prefs.python_executable,
                keep_artifacts=prefs.keep_emitted_artifacts,
            )
        except SidecarError as exc:
            self.report({"ERROR"}, f"sdm-core emission failed:\n{exc}")
            return {"CANCELLED"}

        try:
            renderer = compile_part_shader(artifacts)
        except Exception as exc:  # noqa: BLE001  # GPU compile errors are opaque
            self.report({"ERROR"}, f"Shader compile failed: {exc}")
            return {"CANCELLED"}

        # Metal compiles the pipeline lazily on first draw; force it here so
        # a heavy scene stalls the import (once, reported) instead of the
        # viewport draw handler (which reads as a whole-UI hang).
        try:
            warmup_s = warmup_pso(renderer)
        except Exception as exc:  # noqa: BLE001  # treat like a compile failure
            self.report({"ERROR"}, f"Shader pipeline warm-up failed: {exc}")
            return {"CANCELLED"}
        if warmup_s > 2.0:
            self.report(
                {"WARNING"},
                f"GPU pipeline compile took {warmup_s:.1f} s: this part's "
                f"emitted scene is heavy; consider a lower --poly-lod.",
            )

        # Placeholder: an Empty at the world origin, sized to the local bbox
        # so the user can see *where* the SDF lives even before camera framing.
        # Re-importing the same .sdm reuses its existing placeholder: a new
        # object would get a ``.001`` name while the old renderer (keyed by
        # the old name) kept drawing over the new one.
        obj = _reuse_or_make_placeholder(sdm_path, artifacts)

        # Custom properties drive uniforms via the sidebar sliders.
        install_control_properties(obj, artifacts)
        _install_cutaway_properties(obj, artifacts.bbox)

        # Stash a back-pointer so the panel can find the right artifacts even
        # if the user duplicates / renames the object.
        obj["sdm_source_path"] = str(sdm_path.resolve())

        register_renderer(obj.name, renderer)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        _frame_view_to_selected(context)

        self.report({"INFO"}, f"Imported {sdm_path.name}: {len(artifacts.uniforms)} uniform(s)")
        return {"FINISHED"}


def _frame_view_to_selected(context) -> None:
    """Best-effort: frame the placeholder in the first 3D viewport.

    The import operator runs with the File Browser as the active area, so
    a bare ``view3d.view_selected`` call fails its poll(). We walk the
    screen, find a VIEW_3D area + its WINDOW region, and re-issue the op
    with a context override. Failure is silent: framing is a nicety, not
    a requirement, and a missing 3D viewport (e.g. animation workspace)
    shouldn't make the whole import fail.
    """
    for area in context.screen.areas:
        if area.type != "VIEW_3D":
            continue
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if region is None:
            continue
        try:
            with context.temp_override(area=area, region=region):
                bpy.ops.view3d.view_selected("INVOKE_DEFAULT")
        except (RuntimeError, AttributeError):
            pass
        return


def _reuse_or_make_placeholder(sdm_path: Path, artifacts) -> bpy.types.Object:
    """Return the placeholder for this ``.sdm``, reusing an existing one.

    Matching is by the ``sdm_source_path`` back-pointer, so renames don't
    defeat it. On reuse: every stale renderer for this source stops drawing,
    the bbox mesh is refreshed (the part's extent may have changed), and the
    object, with the user's transform and view-state props, is kept.
    """
    key = str(sdm_path.resolve())
    matches = [o for o in bpy.data.objects if o.get("sdm_source_path") == key]
    if not matches:
        return _make_placeholder(sdm_path.stem, artifacts.bbox)
    obj = matches[0]
    for stale in matches:
        unregister_renderer(stale.name)
    old_mesh = obj.data if obj.type == "MESH" else None
    obj.data = placeholder_mesh(sdm_path.stem, artifacts.bbox)
    if old_mesh is not None and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
        obj.data.name = f"SDM_{sdm_path.stem}_bbox"  # reclaim the un-suffixed name
    return obj


def _make_placeholder(name: str, bbox: Bbox) -> bpy.types.Object:
    """A wireframe box outlining the SDF bbox, with an IDENTITY transform.

    The box is a mesh whose *vertices* are the exact (possibly off-centre) bbox
    corners in authored coordinates, so it bounds the part precisely even when
    the bbox isn't centred on the origin (e.g. a tall part spanning z 0..270).

    The transform stays identity: the SDF is authored in absolute (mm)
    coordinates and the shader maps world->local via ``matrix_world.inverted()``
    with ``u_bbox_*`` and ``sdf_scene()`` in those same units. Encoding the bbox
    in the *mesh* (not the object scale/location) keeps the mapping rigid: an
    earlier version baked the extent into the empty's scale, normalising the
    march to a unit cube while the uniforms stayed in mm (~20x mismatch: wrong
    scale, flat normals). The user may still translate/rotate this object to
    move the part; ``u_object_inv`` handles rigid transforms correctly.
    """
    mesh = placeholder_mesh(name, bbox)

    obj = bpy.data.objects.new(name=f"SDM_{name}", object_data=mesh)
    return _finish_placeholder(obj)


def placeholder_mesh(name: str, bbox: Bbox) -> bpy.types.Mesh:
    """Wireframe bbox mesh in authored coordinates (see _make_placeholder)."""
    (xlo, ylo, zlo), (xhi, yhi, zhi) = bbox

    idx: dict[tuple[int, int, int], int] = {}
    verts: list[tuple[float, float, float]] = []
    for i, x in enumerate((xlo, xhi)):
        for j, y in enumerate((ylo, yhi)):
            for k, z in enumerate((zlo, zhi)):
                idx[(i, j, k)] = len(verts)
                verts.append((x, y, z))
    edges = []
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                a = idx[(i, j, k)]
                if i == 0:
                    edges.append((a, idx[(1, j, k)]))
                if j == 0:
                    edges.append((a, idx[(i, 1, k)]))
                if k == 0:
                    edges.append((a, idx[(i, j, 1)]))

    mesh = bpy.data.meshes.new(f"SDM_{name}_bbox")
    mesh.from_pydata(verts, edges, [])
    mesh.update()
    return mesh


def _finish_placeholder(obj: bpy.types.Object) -> bpy.types.Object:
    obj.display_type = "WIRE"  # edges only: doesn't occlude the ray-march
    obj.hide_render = True  # a viewport cue, not render geometry
    bpy.context.collection.objects.link(obj)
    return obj


CTL_PROP_PREFIX = "sdm_ctl_"  # custom-prop prefix for non-live controls


def install_control_properties(obj: bpy.types.Object, artifacts) -> None:
    """Create custom properties for EVERY control in the manifest.

    - live controls    -> a property named after the ``u_p_*`` uniform (the
      draw handler reads these each redraw, so they are also KEYFRAMABLE:
      parameter animation works with plain Blender keyframes)
    - re-emit/topology -> ``sdm_ctl_<param>`` properties; they take effect on
      the Rebuild operator, which writes them into the .sdm and re-runs the
      part's generator

    Slider ranges prefer ``ui.explore_bounds`` (the authored scrub range) over
    the optimiser ``bounds``; ``ui.step`` sets the drag increment. Falls back
    to the plain per-uniform install for pre-manifest emitters.
    """
    if not artifacts.controls:
        _install_uniform_properties(obj, artifacts.uniforms)
        return

    ui_data = obj.id_properties_ui

    for c in artifacts.controls:
        prop = c.uniform if c.cls == "live" else f"{CTL_PROP_PREFIX}{c.param}"
        obj[prop] = float(c.value)
        meta = {
            "default": float(c.value),
            "description": _control_description(c),
        }
        rng = c.ui.get("explore_bounds") or c.bounds
        if rng is not None:
            meta["soft_min"] = float(rng[0])
            meta["soft_max"] = float(rng[1])
        step = c.ui.get("step")
        if step:
            # id_properties_ui float step is in absolute UI units.
            meta["step"] = float(step)
        ui_data(prop).update(**meta)

    # Per-group open/closed toggles for groups authored ui.collapsed:
    # installed HERE because the panel's draw() may not write ID props.
    # Existing toggles survive a rebuild (user's fold state is preserved).
    closed_groups = {
        (c.ui or {}).get("group", "params")
        for c in artifacts.controls
        if (c.ui or {}).get("collapsed")
    }
    for group in sorted(closed_groups):
        key = f"sdm_grpopen_{group}"
        if key not in obj:
            obj[key] = False


def _install_cutaway_properties(obj: bpy.types.Object, bbox: Bbox) -> None:
    """Cutaway plane props (drawn by the panel, read by the draw handler).

    Offset range spans the bbox with a small margin so the plane can clear
    the part entirely on either side.
    """
    (xlo, ylo, zlo), (xhi, yhi, zhi) = bbox
    span = max(xhi - xlo, yhi - ylo, zhi - zlo)
    ui_data = obj.id_properties_ui

    # View state, not part data: a re-import onto an existing placeholder
    # keeps the user's cutaway/fast/tint settings: only the UI ranges are
    # refreshed (the bbox span may have changed).
    def _default(prop, value):
        if prop not in obj:
            obj[prop] = value
        elif not isinstance(obj[prop], type(value)):
            # An external script writing e.g. an int over a bool retypes the
            # ID property and the panel checkbox becomes a bare number field
            # (observed: sdm_cut_on dragged to 5, cutaway impossible to turn
            # off). Keep the user's value, restore the intended type.
            obj[prop] = type(value)(obj[prop])

    _default("sdm_cut_on", False)
    _default("sdm_cut_axis", 1)
    ui_data("sdm_cut_axis").update(min=0, max=2, description="Cut axis: 0=X 1=Y 2=Z")
    _default("sdm_cut_flip", False)
    ui_data("sdm_cut_flip").update(description="Keep the other half-space")
    _default("sdm_cut_caps", False)
    ui_data("sdm_cut_caps").update(
        description="Carve the field at the plane (flat tinted section "
        "faces). Off: uncapped. See into interiors through "
        "the cut."
    )
    _default("sdm_cut_offset", 0.0)
    ui_data("sdm_cut_offset").update(
        soft_min=-span,
        soft_max=span,
        step=max(span / 200.0, 0.01),
        description="Cutaway plane offset along the cut axis (mm)",
    )
    _default("sdm_fast", True)
    ui_data("sdm_fast").update(
        description="Fast viewport: half the ray-march step budget + a "
        "relaxed hit epsilon. Turn off for stills.",
    )
    _default("sdm_comp_tint", True)
    ui_data("sdm_comp_tint").update(
        description="Tint each component (root-union child) with a stable "
        "distinct hue: SDF segmentation.",
    )
    _default("sdm_pattern", True)
    ui_data("sdm_pattern").update(
        description="Rest-space checker pattern: cells ride with each "
        "component's material, so rotations and twists stay "
        "visible during animation.",
    )


def _control_description(c) -> str:
    bits = [f"Param: {c.param} ({c.cls})"]
    if c.unit:
        bits.append(f"unit: {c.unit}")
    if c.cls != "live":
        bits.append("takes effect on Rebuild")
    if c.bounds is not None:
        bits.append(f"optimiser bounds: [{c.bounds[0]}, {c.bounds[1]}]")
    return " | ".join(bits)


def _install_uniform_properties(obj: bpy.types.Object, uniforms) -> None:
    """Create a custom property per uniform with min/max from Param.bounds.

    Fixed Params (``bounds is None``) show up as type-in fields with no
    slider range; the addon doesn't try to guess a range for them.
    """
    ui_data = obj.id_properties_ui  # 4.x convenience API for custom-prop UI

    for spec in uniforms:
        obj[spec.name] = float(spec.initial)
        meta = {"default": float(spec.initial), "description": _uniform_description(spec)}
        if spec.bounds is not None:
            lo, hi = spec.bounds
            meta["min"] = float(lo)
            meta["max"] = float(hi)
            meta["soft_min"] = float(lo)
            meta["soft_max"] = float(hi)
        # Run-time update of UI metadata; survives file save.
        ui_data(spec.name).update(**meta)


def _uniform_description(spec) -> str:
    bits = [f"Param: {spec.source_param}"]
    if spec.unit:
        bits.append(f"unit: {spec.unit}")
    if spec.bounds is not None:
        bits.append(f"bounds: [{spec.bounds[0]}, {spec.bounds[1]}]")
    else:
        bits.append("fixed (no optimisation bounds)")
    return " | ".join(bits)


def menu_func_import(self, _context):
    self.layout.operator(SDM_OT_import.bl_idname, text="Software Defined Matter (.sdm)")


def register() -> None:
    bpy.utils.register_class(SDM_OT_import)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister() -> None:
    # Remove the menu entry first so a half-unregistered state doesn't
    # leave a dead operator in the File menu.
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(SDM_OT_import)

    # Best-effort: drop renderers for objects we authored. The viewport
    # module's own unregister clears the rest.
    for obj in list(bpy.data.objects):
        if obj.get("sdm_source_path") is not None:
            unregister_renderer(obj.name)
