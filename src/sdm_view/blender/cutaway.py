"""Boolean-cutter cube cross-section ("cutaway") for SDM part views.

A *live* cross-section: a single big wireframe cube is positioned so its near
face sits at a chosen offset along X / Y / Z, and a BOOLEAN DIFFERENCE modifier
against that cube is toggled on every solid mesh in the scene. The modifier
evaluates lazily through the depsgraph, so dragging the position slider (or
grabbing the cube in the viewport) re-cuts the geometry live with no re-mesh.

Generalised from an earlier per-part implementation in a consuming CEM,
which cut a fixed component list, to "every solid mesh in the scene", so any
SDM part can use it without naming its geometry up front.

Registers a sub-panel under the SDM N-panel tab (child of ``SDMVIEW_PT_main``)
plus a scene-level ``PropertyGroup``, matching the other ``sdm_view.blender``
panel modules' ``register()`` / ``unregister()`` convention. A consumer's
launch script registers it alongside the project/parts/sketches panels.

(No ``from __future__ import annotations`` here on purpose: Blender registers
``PropertyGroup`` fields by introspecting ``__annotations__`` for the actual
``bpy.props`` objects, and stringised annotations would hide them.)
"""

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    PointerProperty,
)
from mathutils import Vector

CUTTER_NAME = "SDMView_Cutter"
MODIFIER_NAME = "SDMView_Cutaway"
_CUBE_HALF_MM = 100.0  # half-extent of the 200 mm cutter cube


def make_cutter() -> bpy.types.Object:
    """Return the cutter cube, creating the 200 mm wireframe cube if absent."""
    obj = bpy.data.objects.get(CUTTER_NAME)
    if obj is not None:
        return obj
    mesh = bpy.data.meshes.new(f"{CUTTER_NAME}_mesh")
    obj = bpy.data.objects.new(CUTTER_NAME, mesh)
    bpy.context.collection.objects.link(obj)
    h = _CUBE_HALF_MM
    verts = [
        (-h, -h, -h),
        (h, -h, -h),
        (h, h, -h),
        (-h, h, -h),
        (-h, -h, h),
        (h, -h, h),
        (h, h, h),
        (-h, h, h),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update(calc_edges=True)
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def cutaway_targets() -> list[bpy.types.Object]:
    """Solid meshes to cut: every mesh except the cutter and wire-display
    helpers (e.g. a bounding-volume reference cylinder)."""
    return [
        o
        for o in bpy.data.objects
        if o.type == "MESH" and o.name != CUTTER_NAME and o.display_type not in {"WIRE", "BOUNDS"}
    ]


def apply_cutaway(
    *,
    enabled: bool,
    axis: str,
    position_mm: float,
    targets: list[bpy.types.Object] | None = None,
) -> None:
    """Position the cutter and toggle a boolean-difference modifier on each
    target, so geometry on the far side of ``position_mm`` along ``axis`` is
    cut away (revealing the section plane at ``position_mm``)."""
    cutter = make_cutter()
    cutter.hide_viewport = not enabled
    cutter.hide_render = True

    # The cube spans [loc - half, loc + half]; placing loc = offset + half puts
    # its near face exactly at ``offset``, so DIFFERENCE removes everything with
    # (axis coord) > offset and the cut face sits on the section plane.
    off = float(position_mm) + _CUBE_HALF_MM
    if axis == "X":
        cutter.location = Vector((off, 0.0, 0.0))
    elif axis == "Y":
        cutter.location = Vector((0.0, off, 0.0))
    else:  # "Z"
        cutter.location = Vector((0.0, 0.0, off))

    if targets is None:
        targets = cutaway_targets()
    for obj in targets:
        mod = obj.modifiers.get(MODIFIER_NAME)
        if enabled:
            if mod is None:
                mod = obj.modifiers.new(MODIFIER_NAME, "BOOLEAN")
                mod.operation = "DIFFERENCE"
                # Blender 5.x solver enum: FLOAT (cheapest), EXACT, MANIFOLD.
                # FLOAT is fine for cross-section preview; bump to EXACT only if
                # marching-cubes meshes show boolean artifacts.
                try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
                    mod.solver = "FLOAT"
                except TypeError:
                    pass  # older Blender without the solver enum
            mod.object = cutter
            mod.show_viewport = True
            mod.show_render = True
        elif mod is not None:
            mod.show_viewport = False
            mod.show_render = False


def clear_cutaway() -> None:
    """Remove the cutter object and every cutaway modifier (teardown / for an
    idempotent re-run of a launch script)."""
    for o in bpy.data.objects:
        if o.type == "MESH":
            mod = o.modifiers.get(MODIFIER_NAME)
            if mod is not None:
                o.modifiers.remove(mod)
    cutter = bpy.data.objects.get(CUTTER_NAME)
    if cutter is not None:
        bpy.data.objects.remove(cutter, do_unlink=True)


# ── Scene properties + panel ────────────────────────────────────────────────


def _on_change(self, context) -> None:
    apply_cutaway(
        enabled=self.b_cutaway_enabled,
        axis=self.s_cutaway_axis,
        position_mm=self.d_cutaway_position_mm,
    )


class SDMViewCutawayProps(bpy.types.PropertyGroup):
    b_cutaway_enabled: BoolProperty(  # type: ignore[valid-type]
        name="Cross-section",
        description="Cut away the geometry on the far side of the section plane",
        default=False,
        update=_on_change,
    )
    s_cutaway_axis: EnumProperty(  # type: ignore[valid-type]
        name="Axis",
        description="Axis normal to the section plane",
        items=[
            ("X", "X", "Section plane normal to X"),
            ("Y", "Y", "Section plane normal to Y"),
            ("Z", "Z", "Section plane normal to Z"),
        ],
        default="Y",
        update=_on_change,
    )
    d_cutaway_position_mm: FloatProperty(  # type: ignore[valid-type]
        name="Position (mm)",
        description="Section-plane offset along the axis; geometry beyond it is cut",
        default=0.0,
        min=-100.0,
        max=100.0,
        update=_on_change,
    )


class SDMVIEW_PT_cutaway(bpy.types.Panel):
    bl_label = "Cutaway"
    bl_idname = "SDMVIEW_PT_cutaway"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SDM"
    bl_parent_id = "SDMVIEW_PT_main"

    def draw(self, context):
        props = context.scene.sdm_view_cutaway
        col = self.layout.column(align=True)
        col.prop(props, "b_cutaway_enabled", toggle=True, icon="MOD_BOOLEAN")
        sub = col.column(align=True)
        sub.enabled = props.b_cutaway_enabled
        sub.prop(props, "s_cutaway_axis", expand=True)
        sub.prop(props, "d_cutaway_position_mm", slider=True)


_classes = (SDMViewCutawayProps, SDMVIEW_PT_cutaway)


def register() -> None:
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.sdm_view_cutaway = PointerProperty(
        type=SDMViewCutawayProps,
    )


def unregister() -> None:
    clear_cutaway()
    if hasattr(bpy.types.Scene, "sdm_view_cutaway"):
        del bpy.types.Scene.sdm_view_cutaway
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
