"""3D scene annotations: dots (3D points) and vectors (tail + direction).

The 3D sibling of the principle-sketches feature. Dots and vectors are
draggable in the 3D viewport (G/R/S), bidirectional via the shared SDM
project JSON file (default `/tmp/sdm-project.json`).

- **Dots** are Blender Empty objects with `SPHERE` display: small
  wireframe sphere markers, lightweight, visible in any shading mode.
- **Vectors** are Blender Empty objects with `SINGLE_ARROW` display.
  Magnitude lives in `empty_display_size`; direction lives in
  `rotation_euler` (the arrow points along local +Z).

Usage:

    from sdm_view.blender import annotations
    annotations.register()

Drop dot / vector at the 3D cursor with the panel buttons. G/R/S works
on each as a standard Empty. Auto-syncs to the project JSON on every
drop and clear (G/R/S edits do not auto-sync; click "Sync → JSON"
in the top-level SDM panel after manual transforms).
"""

from __future__ import annotations

import bpy
from bpy.props import StringProperty
from mathutils import Vector

from sdm_view.io.annotation_state import (
    Dot as IODot,
)
from sdm_view.io.annotation_state import (
    Vector3 as IOVector,
)

# ── Constants ───────────────────────────────────────────────────────────────

VECTOR_PREFIX = "sdm_vector_"
DOT_PREFIX = "sdm_dot_"
DEFAULT_VECTOR_LENGTH_MM = 5.0
DEFAULT_DOT_RADIUS_MM = 0.6

ANNOTATION_PALETTE: list[tuple[str, tuple[float, float, float, float]]] = [
    ("orange", (1.00, 0.50, 0.05, 1.0)),
    ("cyan", (0.00, 0.85, 1.00, 1.0)),
    ("magenta", (1.00, 0.20, 0.80, 1.0)),
    ("yellow", (1.00, 0.95, 0.20, 1.0)),
    ("lime", (0.50, 1.00, 0.20, 1.0)),
    ("pink", (1.00, 0.55, 0.75, 1.0)),
]

# Back-compat alias for the vector-only palette name used by project.py imports
VECTOR_PALETTE = ANNOTATION_PALETTE


def _color_for_index(i: int) -> tuple[str, tuple[float, float, float, float]]:
    return ANNOTATION_PALETTE[(i - 1) % len(ANNOTATION_PALETTE)]


def _safe_label(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw.strip())


def _next_index(prefix: str) -> int:
    used: set[int] = set()
    for o in bpy.data.objects:
        if not o.name.startswith(prefix):
            continue
        tail = o.name[len(prefix) :]
        idx_str = tail.split("_", 1)[0]
        try:
            used.add(int(idx_str))
        except ValueError:
            continue
    i = 1
    while i in used:
        i += 1
    return i


# ── Vector helpers ─────────────────────────────────────────────────────────


def _all_vectors() -> list[bpy.types.Object]:
    return sorted(
        [o for o in bpy.data.objects if o.name.startswith(VECTOR_PREFIX)],
        key=lambda o: o.name,
    )


def _orient_arrow(empty: bpy.types.Object, direction: Vector) -> None:
    """Orient an Empty with SINGLE_ARROW display so its arrow points along
    `direction`. The SINGLE_ARROW gizmo natively points along local +Z."""
    d = direction.normalized()
    quat = Vector((0.0, 0.0, 1.0)).rotation_difference(d)
    empty.rotation_euler = quat.to_euler()


def _make_vector_empty(
    name: str,
    tail: Vector,
    direction: Vector,
    color_name: str,
) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "SINGLE_ARROW"
    empty.empty_display_size = max(direction.length, 0.001)
    empty.location = tail
    _orient_arrow(empty, direction)
    palette_entry = next(
        ((n, rgba) for n, rgba in ANNOTATION_PALETTE if n == color_name),
        None,
    )
    if palette_entry is not None:
        empty.color = palette_entry[1]
    empty["sdm_view_role"] = "annotation_vector"
    empty["sdm_view_color"] = color_name
    bpy.context.collection.objects.link(empty)
    return empty


def _harvest_vectors_from_scene() -> list[IOVector]:
    out: list[IOVector] = []
    for obj in _all_vectors():
        tail_str = obj.name[len(VECTOR_PREFIX) :]
        idx_part, _, label_part = tail_str.partition("_")
        try:
            index = int(idx_part)
        except ValueError:
            continue
        color = str(obj.get("sdm_view_color", "orange"))
        magnitude = float(obj.empty_display_size)
        direction = obj.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0))
        direction.normalize()
        tail = obj.matrix_world.translation
        head = tail + direction * magnitude
        out.append(
            IOVector(
                index=index,
                label=label_part,
                color=color,
                tail_x=tail.x,
                tail_y=tail.y,
                tail_z=tail.z,
                head_x=head.x,
                head_y=head.y,
                head_z=head.z,
            )
        )
    return out


# ── Dot helpers ────────────────────────────────────────────────────────────


def _all_dots() -> list[bpy.types.Object]:
    return sorted(
        [o for o in bpy.data.objects if o.name.startswith(DOT_PREFIX)],
        key=lambda o: o.name,
    )


def _get_or_make_dot_material(
    color_name: str,
    color_rgba: tuple[float, float, float, float],
) -> bpy.types.Material:
    """Per-colour material with Principled BSDF + matching emission so the
    dot reads as a glowing sphere in any viewport shading mode (solid /
    material preview / rendered).  Re-used across all dots of the same
    colour so the material count stays bounded by palette length."""
    mat_name = f"sdm_dot_mat_{color_name}"
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color_rgba
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Metallic"].default_value = 0.0
    # Emission so the dot remains visible against any backdrop in solid
    # shading mode (Principled BSDF emission renamed in Blender 4.x).
    try:
        bsdf.inputs["Emission Color"].default_value = color_rgba
        bsdf.inputs["Emission Strength"].default_value = 0.6
    except KeyError:
        try:
            bsdf.inputs["Emission"].default_value = color_rgba
            bsdf.inputs["Emission Strength"].default_value = 0.6
        except KeyError:
            pass
    mat.diffuse_color = color_rgba  # viewport object-color fallback
    return mat


def _make_dot_mesh(
    name: str,
    location: Vector,
    color_name: str,
) -> bpy.types.Object:
    """Tiny mesh icosphere with an emissive per-colour material: the
    colour is visible in every viewport shading mode (unlike an Empty
    whose colour only shows in object-colour mode)."""
    import math as _math

    palette_entry = next(
        ((n, rgba) for n, rgba in ANNOTATION_PALETTE if n == color_name),
        None,
    )
    if palette_entry is not None:  # noqa: SIM108  # kept as an explicit if/else; the packed ternary reads worse than the branch it replaces
        _color_rgba = palette_entry[1]
    else:
        _color_rgba = (1.0, 1.0, 1.0, 1.0)

    # 12-vertex icosphere, unit radius then scaled to DEFAULT_DOT_RADIUS_MM.
    phi = (1.0 + _math.sqrt(5.0)) * 0.5
    raw = [
        (-1.0, phi, 0.0),
        (1.0, phi, 0.0),
        (-1.0, -phi, 0.0),
        (1.0, -phi, 0.0),
        (0.0, -1.0, phi),
        (0.0, 1.0, phi),
        (0.0, -1.0, -phi),
        (0.0, 1.0, -phi),
        (phi, 0.0, -1.0),
        (phi, 0.0, 1.0),
        (-phi, 0.0, -1.0),
        (-phi, 0.0, 1.0),
    ]
    scale = DEFAULT_DOT_RADIUS_MM / _math.sqrt(1.0 + phi * phi)
    verts = [(x * scale, y * scale, z * scale) for (x, y, z) in raw]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.data.materials.append(_get_or_make_dot_material(color_name, _color_rgba))
    obj["sdm_view_role"] = "annotation_dot"
    obj["sdm_view_color"] = color_name
    bpy.context.collection.objects.link(obj)
    return obj


# Back-compat alias: older callers still import this name.
_make_dot_empty = _make_dot_mesh


def load_dots_from_json(json_path: str = "/tmp/sdm-project.json") -> int:
    """Recreate dot meshes from the on-disk project JSON.  Used on
    Blender startup so dots survive across a restart.  Existing dots
    are removed first to avoid duplicates.  Returns count loaded."""
    import json
    import pathlib

    p = pathlib.Path(json_path)
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        print(f"[sdm-view] load_dots_from_json: {exc}")
        return 0
    dots = data.get("dots", [])
    # Wipe existing dots first (no duplicates from rerun).
    for obj in list(_all_dots()):
        bpy.data.objects.remove(obj, do_unlink=True)
    n = 0
    for d in dots:
        idx = int(d.get("index", n + 1))
        label = _safe_label(str(d.get("label", "")))
        color = str(d.get("color", "orange"))
        loc = Vector((float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("z", 0.0))))
        name = f"{DOT_PREFIX}{idx:03d}_{label}" if label else f"{DOT_PREFIX}{idx:03d}"
        _make_dot_mesh(name, loc, color)
        n += 1
    return n


def _harvest_dots_from_scene() -> list[IODot]:
    out: list[IODot] = []
    for obj in _all_dots():
        tail_str = obj.name[len(DOT_PREFIX) :]
        idx_part, _, label_part = tail_str.partition("_")
        try:
            index = int(idx_part)
        except ValueError:
            continue
        color = str(obj.get("sdm_view_color", "orange"))
        loc = obj.matrix_world.translation
        out.append(
            IODot(
                index=index,
                label=label_part,
                color=color,
                x=loc.x,
                y=loc.y,
                z=loc.z,
            )
        )
    return out


# ── Operators ──────────────────────────────────────────────────────────────


class SDMVIEW_OT_drop_vector(bpy.types.Operator):
    """Drop a vector annotation at the 3D cursor.

    Defaults: direction = world +X, magnitude = 5 mm. After dropping,
    G/R/S the arrow to position, orient, and scale. Auto-syncs the
    project JSON on drop."""

    bl_idname = "sdmview.drop_vector"
    bl_label = "Drop vector at cursor"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        safe = _safe_label(str(scene.sdm_view_vector_label))
        idx = _next_index(VECTOR_PREFIX)
        color_name, _rgba = _color_for_index(idx)
        name = f"{VECTOR_PREFIX}{idx:03d}_{safe}" if safe else f"{VECTOR_PREFIX}{idx:03d}"
        tail = Vector(scene.cursor.location)
        direction = Vector((DEFAULT_VECTOR_LENGTH_MM, 0.0, 0.0))
        _make_vector_empty(name, tail, direction, color_name)
        scene.sdm_view_project_status = f"+ vector {name}  [{color_name}]"
        scene.sdm_view_vector_label = ""
        bpy.ops.sdmview.project_sync()
        return {"FINISHED"}


class SDMVIEW_OT_drop_dot(bpy.types.Operator):
    """Drop a labeled 3D dot at the 3D cursor.

    Dots are small wireframe spheres, lightweight visible markers for
    parametric points, contact locations, named coordinates. Auto-syncs
    the project JSON on drop."""

    bl_idname = "sdmview.drop_dot"
    bl_label = "Drop dot at cursor"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        safe = _safe_label(str(scene.sdm_view_dot_label))
        idx = _next_index(DOT_PREFIX)
        color_name, _rgba = _color_for_index(idx)
        name = f"{DOT_PREFIX}{idx:03d}_{safe}" if safe else f"{DOT_PREFIX}{idx:03d}"
        location = Vector(scene.cursor.location)
        _make_dot_empty(name, location, color_name)
        scene.sdm_view_project_status = (
            f"+ dot {name}  [{color_name}] @ ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})"
        )
        scene.sdm_view_dot_label = ""
        bpy.ops.sdmview.project_sync()
        return {"FINISHED"}


class SDMVIEW_OT_load_dots(bpy.types.Operator):
    """Reload dots from `/tmp/sdm-project.json` into the scene.

    Use after restarting Blender to recover dots dropped in an earlier
    session.  Wipes existing dots first to avoid duplicates."""

    bl_idname = "sdmview.load_dots"
    bl_label = "Load dots from JSON"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        n = load_dots_from_json()
        context.scene.sdm_view_project_status = f"Loaded {n} dot(s) from JSON"
        return {"FINISHED"}


class SDMVIEW_OT_clear_annotations(bpy.types.Operator):
    """Remove every dot + vector annotation from the scene; resync JSON."""

    bl_idname = "sdmview.clear_annotations"
    bl_label = "Clear annotations"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        n_vec = 0
        for obj in list(_all_vectors()):
            bpy.data.objects.remove(obj, do_unlink=True)
            n_vec += 1
        n_dot = 0
        for obj in list(_all_dots()):
            bpy.data.objects.remove(obj, do_unlink=True)
            n_dot += 1
        context.scene.sdm_view_project_status = f"Cleared {n_dot} dot(s) + {n_vec} vector(s)"
        bpy.ops.sdmview.project_sync()
        return {"FINISHED"}


# ── Panel ──────────────────────────────────────────────────────────────────


class SDMVIEW_PT_annotations(bpy.types.Panel):
    bl_label = "Annotations"
    bl_idname = "SDMVIEW_PT_annotations"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SDM"
    bl_parent_id = "SDMVIEW_PT_main"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        cur = scene.cursor.location
        layout.label(
            text=f"3D cursor: ({cur.x:.1f}, {cur.y:.1f}, {cur.z:.1f})",
            icon="PIVOT_CURSOR",
        )

        box = layout.box()
        box.label(text="Dots", icon="OUTLINER_OB_LIGHT")
        row = box.row(align=True)
        row.prop(scene, "sdm_view_dot_label", text="", icon="GREASEPENCIL")
        row.operator("sdmview.drop_dot", icon="ADD", text="Drop @ cursor")
        row = box.row(align=True)
        row.operator("sdmview.load_dots", icon="FILE_REFRESH", text="Load from JSON")
        box.label(text=f"{len(_all_dots())} dot(s) in scene")

        box = layout.box()
        box.label(text="Vectors", icon="EMPTY_SINGLE_ARROW")
        row = box.row(align=True)
        row.prop(scene, "sdm_view_vector_label", text="", icon="GREASEPENCIL")
        row.operator("sdmview.drop_vector", icon="ADD", text="Drop @ cursor")
        box.label(text=f"{len(_all_vectors())} vector(s) in scene")

        layout.separator()
        layout.operator("sdmview.clear_annotations", icon="TRASH", text="Clear all annotations")


# ── Registration ───────────────────────────────────────────────────────────

_classes = (
    SDMVIEW_OT_drop_vector,
    SDMVIEW_OT_drop_dot,
    SDMVIEW_OT_load_dots,
    SDMVIEW_OT_clear_annotations,
    SDMVIEW_PT_annotations,
)


def register() -> None:
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.sdm_view_vector_label = StringProperty(
        name="Next vector label",
        description="Optional label for the next vector you drop. Auto-cleared.",
        default="",
    )
    bpy.types.Scene.sdm_view_dot_label = StringProperty(
        name="Next dot label",
        description="Optional label for the next dot you drop. Auto-cleared.",
        default="",
    )


def unregister() -> None:
    del bpy.types.Scene.sdm_view_vector_label
    del bpy.types.Scene.sdm_view_dot_label
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
