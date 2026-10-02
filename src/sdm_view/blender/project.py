"""Top-level SDM project panel and shared state file.

Owns the shared `sdm_view_project_path` and `sdm_view_project_status` scene
properties, plus the top-level "SDM" panel that surfaces them. Feature
sub-panels (principle sketches, annotations) appear under this top-level
panel via `bl_parent_id`.

The project file is a single JSON document holding sketches + annotations
together (`/tmp/sdm-project.json` by default). Each feature's IO does
read-modify-write on its own keys, so the file remains coherent across
features.
"""

from __future__ import annotations

import bpy
from bpy.props import StringProperty

from sdm_view.io.annotation_state import read_annotations, write_annotations
from sdm_view.io.sketch_state import read_sketches, write_sketches

DEFAULT_PROJECT_PATH = "/tmp/sdm-project.json"


# ── Operators ──────────────────────────────────────────────────────────────


class SDMVIEW_OT_project_sync(bpy.types.Operator):
    """Write the full SDM project (sketches + annotations) to the JSON
    project file at the configured path."""

    bl_idname = "sdmview.project_sync"
    bl_label = "Sync project → JSON"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from sdm_view.blender.annotations import (
            _harvest_dots_from_scene,
            _harvest_vectors_from_scene,
        )
        from sdm_view.blender.sketches import _harvest_sketches_from_scene

        scene = context.scene
        path = scene.sdm_view_project_path or DEFAULT_PROJECT_PATH
        sketches = _harvest_sketches_from_scene()
        dots = _harvest_dots_from_scene()
        vectors = _harvest_vectors_from_scene()
        # Each feature writer does read-modify-write, so order is just for
        # diff predictability.
        write_sketches(path, sketches)
        write_annotations(path, dots=dots, vectors=vectors)
        n_curves = sum(len(s.curves) for s in sketches)
        scene.sdm_view_project_status = (
            f"Synced {n_curves} curve(s) + {len(dots)} dot(s) + {len(vectors)} vector(s) → {path}"
        )
        return {"FINISHED"}


class SDMVIEW_OT_project_load(bpy.types.Operator):
    """Replace the in-scene sketches + annotations with the contents of
    the JSON project file. Useful for picking up edits authored outside
    Blender (e.g. by Claude)."""

    bl_idname = "sdmview.project_load"
    bl_label = "Load project ← JSON"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from mathutils import Vector as MVec

        from sdm_view.blender.annotations import (
            DEFAULT_VECTOR_LENGTH_MM,
            DOT_PREFIX,
            VECTOR_PREFIX,
            _all_dots,
            _all_vectors,
            _make_dot_empty,
            _make_vector_empty,
        )
        from sdm_view.blender.sketches import (
            PLANES,
            _apply_sketches_to_scene,
            _create_plane,
            _refresh_all_dimensions,
        )

        scene = context.scene
        path = scene.sdm_view_project_path or DEFAULT_PROJECT_PATH
        try:
            sketches = read_sketches(path)
        except FileNotFoundError:
            self.report({"ERROR"}, f"no project JSON at {path}")
            return {"CANCELLED"}
        try:
            dots, vectors = read_annotations(path)
        except FileNotFoundError:
            dots, vectors = [], []

        for p in PLANES:
            _create_plane(p)
        n_curves = _apply_sketches_to_scene(sketches)
        _refresh_all_dimensions()

        for obj in list(_all_vectors()):
            bpy.data.objects.remove(obj, do_unlink=True)
        for v in vectors:
            tail = MVec((v.tail_x, v.tail_y, v.tail_z))
            head = MVec((v.head_x, v.head_y, v.head_z))
            direction = head - tail
            if direction.length == 0.0:
                direction = MVec((DEFAULT_VECTOR_LENGTH_MM, 0.0, 0.0))
            safe = v.label
            name = (
                f"{VECTOR_PREFIX}{v.index:03d}_{safe}" if safe else f"{VECTOR_PREFIX}{v.index:03d}"
            )
            _make_vector_empty(name, tail, direction, v.color)

        for obj in list(_all_dots()):
            bpy.data.objects.remove(obj, do_unlink=True)
        for d in dots:
            safe = d.label
            name = f"{DOT_PREFIX}{d.index:03d}_{safe}" if safe else f"{DOT_PREFIX}{d.index:03d}"
            _make_dot_empty(name, MVec((d.x, d.y, d.z)), d.color)

        scene.sdm_view_project_status = (
            f"Loaded {n_curves} curve(s) + {len(dots)} dot(s) + "
            f"{len(vectors)} vector(s) from {path}"
        )
        return {"FINISHED"}


# ── Panel ──────────────────────────────────────────────────────────────────


class SDMVIEW_PT_main(bpy.types.Panel):
    bl_label = "SDM project"
    bl_idname = "SDMVIEW_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SDM"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)
        col.label(text="Project file:", icon="FILE_TICK")
        col.prop(scene, "sdm_view_project_path", text="")

        row = layout.row(align=True)
        row.scale_y = 1.2
        row.operator("sdmview.project_sync", icon="EXPORT", text="Sync → JSON")
        row.operator("sdmview.project_load", icon="IMPORT", text="Load ← JSON")

        if scene.sdm_view_project_status:
            box = layout.box()
            box.label(text=scene.sdm_view_project_status, icon="INFO")


# ── Registration ───────────────────────────────────────────────────────────

_classes = (
    SDMVIEW_OT_project_sync,
    SDMVIEW_OT_project_load,
    SDMVIEW_PT_main,
)


def register() -> None:
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.sdm_view_project_path = StringProperty(
        name="SDM project file",
        description="Single JSON document holding sketches + annotations for "
        "this project. Both sides (Blender + JAX) read and write "
        "this file as the shared source of truth.",
        default=DEFAULT_PROJECT_PATH,
        subtype="FILE_PATH",
    )
    bpy.types.Scene.sdm_view_project_status = StringProperty(
        name="SDM project status",
        default="",
    )


def unregister() -> None:
    del bpy.types.Scene.sdm_view_project_path
    del bpy.types.Scene.sdm_view_project_status
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
