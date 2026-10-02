"""SDM part auto-discovery: load an `.sdm` file and dynamically generate
a Blender slider panel from its `params` block.

The first instance of "the .sdm file IS the registration": drop in a
new CEM with a `.sdm` and the Blender authoring stack discovers its
parameters automatically, with no hand-written panel code. See project
memory `auto_discovery_via_sdm.md`.

This first cut handles `params` only: name, value, free, bounds, unit.
Materials / couplings / objectives / constraints are parsed and stashed
on the loaded `SdmPart` but not yet rendered. Likewise edit-back (slider
move → write `.sdm`) is not wired yet.

Usage from Blender after registering the sdm-view bundle:

    1. SDM tab → "SDM part" panel → "Load .sdm…"
    2. Pick e.g. `emergent-matter-sdm-core/examples/hollow_cylinder_with_hinge.sdm`
    3. Sliders appear, one per param, with `bounds` enforced and `unit` in tooltips.
"""

from __future__ import annotations

import bpy
from bpy.props import FloatProperty, StringProperty

from sdm_view.io.sdm_loader import load_sdm_part

# Dynamic-prop tracking. Each loaded part registers FloatProperties on
# `bpy.types.Scene` named `sdm_param_<safe_param_name>`. We track the
# active set of attribute names so a fresh load can clear them.
_ACTIVE_PARAM_ATTRS: list[str] = []
# Per-attr metadata so the panel can read tooltip / unit / bounds.
_PARAM_META: dict[str, dict] = {}


def _safe_attr(name: str) -> str:
    return "sdm_param_" + "".join(ch if ch.isalnum() or ch in "_" else "_" for ch in name)


def _clear_dynamic_params() -> None:
    for attr in list(_ACTIVE_PARAM_ATTRS):
        if hasattr(bpy.types.Scene, attr):
            try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
                delattr(bpy.types.Scene, attr)
            except Exception:
                pass
    _ACTIVE_PARAM_ATTRS.clear()
    _PARAM_META.clear()


def _register_part_as_dynamic_props(part) -> None:
    """For each param on the SdmPart, register a Scene.FloatProperty so
    the panel can render it as a slider with the right bounds + tooltip."""
    _clear_dynamic_params()
    for p in part.params:
        attr = _safe_attr(p.name)
        # Avoid collisions if someone names a param the same as an existing
        # property: bump with a numeric suffix.
        base_attr = attr
        i = 1
        while hasattr(bpy.types.Scene, attr):
            attr = f"{base_attr}_{i}"
            i += 1
        kwargs: dict = {
            "name": p.name,
            "description": (
                f"{p.name} ({p.unit}); free={p.free}"
                + (f"; bounds=[{p.bounds[0]}, {p.bounds[1]}]" if p.bounds else "")
            ),
            "default": float(p.value),
            "subtype": "DISTANCE" if p.unit == "mm" else "NONE",
        }
        if p.bounds is not None:
            kwargs["min"] = float(p.bounds[0])
            kwargs["max"] = float(p.bounds[1])
            kwargs["soft_min"] = float(p.bounds[0])
            kwargs["soft_max"] = float(p.bounds[1])
        setattr(bpy.types.Scene, attr, FloatProperty(**kwargs))
        _ACTIVE_PARAM_ATTRS.append(attr)
        _PARAM_META[attr] = {
            "name": p.name,
            "free": p.free,
            "bounds": p.bounds,
            "unit": p.unit,
        }


# ── Operators ──────────────────────────────────────────────────────────────


class SDMVIEW_OT_load_sdm_part(bpy.types.Operator):
    """Open an `.sdm` file (sdm-core schema v0.1) and generate sliders for
    its parameters in the SDM part sub-panel."""

    bl_idname = "sdmview.load_sdm_part"
    bl_label = "Load .sdm part"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH", default="")  # type: ignore[valid-type]
    filter_glob: StringProperty(default="*.sdm;*.json", options={"HIDDEN"})  # type: ignore[valid-type]

    def invoke(self, context, event):  # noqa: ARG002  # Blender operator invoke() signature requires this parameter even unused
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            part = load_sdm_part(self.filepath)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            self.report({"ERROR"}, f"Couldn't load {self.filepath}: {exc}")
            return {"CANCELLED"}
        _register_part_as_dynamic_props(part)
        context.scene.sdm_view_loaded_part_name = part.name
        context.scene.sdm_view_loaded_part_path = self.filepath
        context.scene.sdm_view_loaded_param_count = len(part.params)
        context.scene.sdm_view_project_status = (
            f"Loaded .sdm part '{part.name}' with {len(part.params)} param(s) from {self.filepath}"
        )
        return {"FINISHED"}


class SDMVIEW_OT_clear_sdm_part(bpy.types.Operator):
    """Remove the dynamically-registered SDM part sliders."""

    bl_idname = "sdmview.clear_sdm_part"
    bl_label = "Clear loaded .sdm"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _clear_dynamic_params()
        context.scene.sdm_view_loaded_part_name = ""
        context.scene.sdm_view_loaded_part_path = ""
        context.scene.sdm_view_loaded_param_count = 0
        context.scene.sdm_view_project_status = "Cleared loaded .sdm part"
        return {"FINISHED"}


# ── Panel ──────────────────────────────────────────────────────────────────


class SDMVIEW_PT_part(bpy.types.Panel):
    bl_label = "SDM part"
    bl_idname = "SDMVIEW_PT_part"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SDM"
    bl_parent_id = "SDMVIEW_PT_main"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        row = layout.row(align=True)
        row.scale_y = 1.2
        row.operator("sdmview.load_sdm_part", icon="FILEBROWSER", text="Load .sdm…")
        row.operator("sdmview.clear_sdm_part", icon="X", text="")

        name = scene.sdm_view_loaded_part_name
        if not name:
            layout.label(text="No .sdm part loaded.", icon="INFO")
            return

        box = layout.box()
        box.label(text=f"Part: {name}", icon="OUTLINER_OB_MESH")
        box.label(text=f"{scene.sdm_view_loaded_param_count} params discovered")

        if not _ACTIVE_PARAM_ATTRS:
            layout.label(text="(no params in this .sdm)", icon="ERROR")
            return

        col = layout.column(align=True)
        for attr in _ACTIVE_PARAM_ATTRS:
            meta = _PARAM_META.get(attr, {})
            row = col.row(align=True)
            row.enabled = bool(meta.get("free", False))
            row.prop(scene, attr)
            unit = meta.get("unit", "")
            if unit:
                row.label(text=unit)


# ── Registration ───────────────────────────────────────────────────────────

_classes = (
    SDMVIEW_OT_load_sdm_part,
    SDMVIEW_OT_clear_sdm_part,
    SDMVIEW_PT_part,
)


def register() -> None:
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.sdm_view_loaded_part_name = StringProperty(
        name="Loaded .sdm part name",
        default="",
    )
    bpy.types.Scene.sdm_view_loaded_part_path = StringProperty(
        name="Loaded .sdm path",
        default="",
        subtype="FILE_PATH",
    )
    from bpy.props import IntProperty  # local to keep top of module slim

    bpy.types.Scene.sdm_view_loaded_param_count = IntProperty(
        name="Loaded param count",
        default=0,
    )


def unregister() -> None:
    _clear_dynamic_params()
    del bpy.types.Scene.sdm_view_loaded_part_name
    del bpy.types.Scene.sdm_view_loaded_part_path
    del bpy.types.Scene.sdm_view_loaded_param_count
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
