"""Principle sketches: three transparent reference planes (XY/XZ/YZ) with
named draggable curves, syncing bidirectionally to JSON.

The 2D sibling of annotation dots: dots are 3D points, sketches are
plane-locked curves with internal control points. See memory
`sdm_view_drawings.md` for design intent.

Usage from a Blender script (or addon):

    from sdm_view.blender import sketches
    sketches.register()

Then the "SDM" tab appears in the 3D view N-panel. Click "Init principle
sketches" to create the three planes; "Add polyline" to drop a default
polyline on a plane; edit control points with Blender's Edit Mode (Tab → G);
"Sync → JSON" to export current state; "Load ← JSON" to reload (so Claude
can edit the JSON and have the sketch update in Blender).
"""

from __future__ import annotations

import math

import bpy
from bpy.props import BoolProperty, StringProperty
from mathutils import Vector

from sdm_view.io.sketch_state import (
    Curve as IOCurve,
)
from sdm_view.io.sketch_state import (
    Point as IOPoint,
)
from sdm_view.io.sketch_state import (
    Sketch as IOSketch,
)

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_JSON_PATH = "/tmp/sdm-sketches.json"
PLANE_PREFIX = "sdm_sketch_plane_"
CURVE_PREFIX = "sdm_sketch_curve_"
DIMENSION_PREFIX = "sdm_sketch_dim_"
DIMENSION_TEXT_SIZE = 1.0  # mm
DIMENSION_OFFSET = 1.5  # mm: perpendicular offset from segment midpoint
PLANES: tuple[str, str, str] = ("xy", "xz", "yz")

# Palette cycled when adding new curves without an explicit color.
COLOR_PALETTE: list[tuple[str, tuple[float, float, float, float]]] = [
    ("orange", (1.00, 0.50, 0.05, 1.0)),
    ("cyan", (0.00, 0.85, 1.00, 1.0)),
    ("magenta", (1.00, 0.20, 0.80, 1.0)),
    ("yellow", (1.00, 0.95, 0.20, 1.0)),
    ("lime", (0.50, 1.00, 0.20, 1.0)),
    ("pink", (1.00, 0.55, 0.75, 1.0)),
]


# ── Coordinate helpers ─────────────────────────────────────────────────────


def _plane_2d_to_world(u: float, v: float, plane: str) -> Vector:
    if plane == "xy":
        return Vector((u, v, 0.0))
    if plane == "xz":
        return Vector((u, 0.0, v))
    if plane == "yz":
        return Vector((0.0, u, v))
    raise ValueError(f"unknown plane {plane!r}")


def _world_to_plane_2d(world: Vector, plane: str) -> tuple[float, float]:
    if plane == "xy":
        return world.x, world.y
    if plane == "xz":
        return world.x, world.z
    if plane == "yz":
        return world.y, world.z
    raise ValueError(f"unknown plane {plane!r}")


def _perp_in_plane(direction: Vector, plane: str) -> Vector:
    """Unit vector perpendicular to `direction`, in the given plane (90° CCW
    around the plane's normal axis). Used to offset dimension labels off the
    segment line."""
    if plane == "xy":
        n = Vector((-direction.y, direction.x, 0.0))
    elif plane == "xz":
        n = Vector((direction.z, 0.0, -direction.x))
    elif plane == "yz":
        n = Vector((0.0, -direction.z, direction.y))
    else:
        raise ValueError(plane)
    if n.length == 0.0:
        return n
    return n.normalized()


# ── Materials ──────────────────────────────────────────────────────────────


def _ensure_plane_material() -> bpy.types.Material:
    name = "sdm_sketch_plane_mat"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    nodes = mat.node_tree.nodes
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.25, 0.45, 0.85, 1.0)
    bsdf.inputs["Alpha"].default_value = 0.06
    try:
        bsdf.inputs["Emission Color"].default_value = (0.25, 0.45, 0.85, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 0.2
    except KeyError:
        pass
    mat.node_tree.links.new(bsdf.outputs[0], out.inputs[0])
    return mat


def _ensure_curve_material(color_name: str, color_rgba) -> bpy.types.Material:
    mat_name = f"sdm_sketch_curve_mat_{color_name}"
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color_rgba
    try:
        bsdf.inputs["Emission Color"].default_value = color_rgba
        bsdf.inputs["Emission Strength"].default_value = 0.8
    except KeyError:
        pass
    mat.diffuse_color = color_rgba
    return mat


def _color_for_index(i: int) -> tuple[str, tuple[float, float, float, float]]:
    return COLOR_PALETTE[i % len(COLOR_PALETTE)]


# ── Plane creation ─────────────────────────────────────────────────────────


def _create_plane(plane: str, size: float = 50.0) -> bpy.types.Object:
    name = PLANE_PREFIX + plane
    existing = bpy.data.objects.get(name)
    if existing is not None:
        return existing
    half = size * 0.5
    verts: list[tuple[float, float, float]]
    if plane == "xy":
        verts = [(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)]
    elif plane == "xz":
        verts = [(-half, 0, -half), (half, 0, -half), (half, 0, half), (-half, 0, half)]
    else:  # yz
        verts = [(0, -half, -half), (0, half, -half), (0, half, half), (0, -half, half)]

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(_ensure_plane_material())
    obj.display_type = "WIRE"
    obj["sdm_view_role"] = "principle_plane"
    obj["sdm_view_plane"] = plane
    obj.lock_location = (True, True, True)
    obj.lock_rotation = (True, True, True)
    obj.lock_scale = (True, True, True)
    bpy.context.collection.objects.link(obj)
    return obj


# ── Curve creation ─────────────────────────────────────────────────────────


def _make_polyline_object(
    plane: str,
    curve_id: str,
    points_2d: list[tuple[float, float, str]],  # (u, v, label)
    color_name: str,
    color_rgba,
    label: str,
) -> bpy.types.Object:
    """Create a Blender Curve object representing a polyline on `plane`.
    Each `points_2d` entry is (u, v, point_label)."""
    name = f"{CURVE_PREFIX}{plane}__{curve_id}"
    existing = bpy.data.objects.get(name)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)

    curve = bpy.data.curves.new(name + "_data", type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.18
    curve.bevel_resolution = 3

    spline = curve.splines.new(type="POLY")
    if len(points_2d) > 1:
        spline.points.add(len(points_2d) - 1)
    for i, (u, v, _label) in enumerate(points_2d):
        world = _plane_2d_to_world(u, v, plane)
        spline.points[i].co = (world.x, world.y, world.z, 1.0)

    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(_ensure_curve_material(color_name, color_rgba))
    obj["sdm_view_role"] = "principle_curve"
    obj["sdm_view_plane"] = plane
    obj["sdm_view_curve_id"] = curve_id
    obj["sdm_view_curve_type"] = "polyline"
    obj["sdm_view_color"] = color_name
    obj["sdm_view_label"] = label
    # Lock object transform so editing happens at vertex level only.
    obj.lock_location = (True, True, True)
    obj.lock_rotation = (True, True, True)
    obj.lock_scale = (True, True, True)
    bpy.context.collection.objects.link(obj)

    # Point labels live as a flattened custom property keyed by index.
    for i, (_u, _v, point_label) in enumerate(points_2d):
        obj[f"sdm_view_point_label_{i}"] = point_label

    return obj


# ── Dimensions (in-plane text labels) ──────────────────────────────────────


def _make_dimension_text(
    name: str,
    world_pos: Vector,
    body: str,
    plane: str,
    color_rgba,
) -> bpy.types.Object:
    """Create a flat text object on `plane` showing `body` at `world_pos`.
    Orientation is set so the text lies in the plane facing the natural
    viewing direction for that plane."""
    text_curve = bpy.data.curves.new(name + "_data", type="FONT")
    text_curve.body = body
    text_curve.size = DIMENSION_TEXT_SIZE
    text_curve.align_x = "CENTER"
    text_curve.align_y = "CENTER"
    obj = bpy.data.objects.new(name, text_curve)
    obj.location = world_pos
    if plane == "xy":
        obj.rotation_euler = (0.0, 0.0, 0.0)
    elif plane == "xz":
        obj.rotation_euler = (math.pi / 2, 0.0, 0.0)
    else:  # yz
        obj.rotation_euler = (math.pi / 2, 0.0, math.pi / 2)
    mat_color_key = (
        f"{int(color_rgba[0] * 255):03d}_"
        f"{int(color_rgba[1] * 255):03d}_"
        f"{int(color_rgba[2] * 255):03d}"
    )
    obj.data.materials.append(_ensure_curve_material(f"dim_{mat_color_key}", color_rgba))
    obj["sdm_view_role"] = "dimension"
    obj["sdm_view_plane"] = plane
    obj.lock_location = (True, True, True)
    obj.lock_rotation = (True, True, True)
    obj.lock_scale = (True, True, True)
    scene = bpy.context.scene
    show = getattr(scene, "sdm_view_show_dimensions", True)
    obj.hide_viewport = not bool(show)
    bpy.context.collection.objects.link(obj)
    return obj


def _create_polyline_dimensions(
    plane: str,
    curve_id: str,
    points_2d: list[tuple[float, float, str]],
    color_rgba,
) -> None:
    """Place one segment-length label per polyline segment, at the midpoint
    offset perpendicular to the segment within the plane."""
    if len(points_2d) < 2:
        return
    for i in range(len(points_2d) - 1):
        u0, v0, _ = points_2d[i]
        u1, v1, _ = points_2d[i + 1]
        p0 = _plane_2d_to_world(u0, v0, plane)
        p1 = _plane_2d_to_world(u1, v1, plane)
        segment = p1 - p0
        length = segment.length
        midpoint = (p0 + p1) * 0.5
        perp = _perp_in_plane(segment, plane)
        text_pos = midpoint + perp * DIMENSION_OFFSET
        body = f"{length:.2f}"
        name = f"{DIMENSION_PREFIX}{plane}__{curve_id}__seg{i:02d}"
        existing = bpy.data.objects.get(name)
        if existing is not None:
            bpy.data.objects.remove(existing, do_unlink=True)
        _make_dimension_text(name, text_pos, body, plane, color_rgba)


def _clear_all_dimensions() -> None:
    for obj in [o for o in bpy.data.objects if o.get("sdm_view_role") == "dimension"]:
        bpy.data.objects.remove(obj, do_unlink=True)


def _refresh_all_dimensions() -> None:
    """Wipe and regenerate dimension labels for every principle curve
    currently in the scene. Called after any operator that adds, edits,
    or loads sketch curves."""
    _clear_all_dimensions()
    for obj in bpy.data.objects:
        if obj.get("sdm_view_role") != "principle_curve":
            continue
        if obj.get("sdm_view_curve_type") != "polyline":
            continue
        plane = obj.get("sdm_view_plane")
        if plane not in PLANES:
            continue
        curve_id = obj.get("sdm_view_curve_id", obj.name)
        color_name = obj.get("sdm_view_color", "orange")
        color_rgba = next(
            (rgba for n, rgba in COLOR_PALETTE if n == color_name),
            COLOR_PALETTE[0][1],
        )
        spline = obj.data.splines[0]
        pts: list[tuple[float, float, str]] = []
        for i, p in enumerate(spline.points):
            world = obj.matrix_world @ Vector((p.co.x, p.co.y, p.co.z))
            u, v = _world_to_plane_2d(world, plane)
            label = str(obj.get(f"sdm_view_point_label_{i}", ""))
            pts.append((u, v, label))
        _create_polyline_dimensions(plane, curve_id, pts, color_rgba)


def _on_show_dimensions_change(self, context):  # noqa: ARG001  # Blender property-update callback signature requires this parameter even unused
    visible = bool(context.scene.sdm_view_show_dimensions)
    for obj in bpy.data.objects:
        if obj.get("sdm_view_role") == "dimension":
            obj.hide_viewport = not visible


# ── Harvest / apply ────────────────────────────────────────────────────────


def _harvest_sketches_from_scene() -> list[IOSketch]:
    sketches = {
        p: IOSketch(
            id=f"{p}_sketch",
            plane=p,  # type: ignore[arg-type]
            label=f"{p.upper()} plane sketch",
            curves=[],
        )
        for p in PLANES
    }
    for obj in bpy.data.objects:
        if obj.get("sdm_view_role") != "principle_curve":
            continue
        plane = obj.get("sdm_view_plane")
        if plane not in sketches:
            continue
        curve_type = obj.get("sdm_view_curve_type", "polyline")
        curve_id = obj.get("sdm_view_curve_id", obj.name)
        color = obj.get("sdm_view_color", "orange")
        label = obj.get("sdm_view_label", curve_id)
        if curve_type != "polyline":
            continue  # extend with bezier/circle/arc later
        spline = obj.data.splines[0]
        pts: list[IOPoint] = []
        for i, p in enumerate(spline.points):
            world = obj.matrix_world @ Vector((p.co.x, p.co.y, p.co.z))
            u, v = _world_to_plane_2d(world, plane)
            point_label = str(obj.get(f"sdm_view_point_label_{i}", ""))
            pts.append(IOPoint(x=round(u, 4), y=round(v, 4), label=point_label))
        sketches[plane].curves.append(
            IOCurve(id=curve_id, type="polyline", points=pts, color=color, label=label)
        )
    return list(sketches.values())


def _apply_sketches_to_scene(sketches: list[IOSketch]) -> int:
    """Clear all current sketch curves, recreate from the provided sketches.
    Planes are not touched. Returns the number of curves created."""
    for obj in [o for o in bpy.data.objects if o.get("sdm_view_role") == "principle_curve"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    n = 0
    for s in sketches:
        for i, c in enumerate(s.curves):
            if c.type != "polyline":
                continue
            color_name = c.color or _color_for_index(i)[0]
            rgba = next(
                (rgba for name, rgba in COLOR_PALETTE if name == color_name),
                _color_for_index(i)[1],
            )
            _make_polyline_object(
                plane=s.plane,
                curve_id=c.id,
                points_2d=[(p.x, p.y, p.label) for p in c.points],
                color_name=color_name,
                color_rgba=rgba,
                label=c.label,
            )
            n += 1
    return n


# ── Operators ──────────────────────────────────────────────────────────────


class SDMVIEW_OT_sketches_init(bpy.types.Operator):
    """Create the three principle reference planes (XY, XZ, YZ) and seed
    one empty sketch per plane."""

    bl_idname = "sdmview.sketches_init"
    bl_label = "Init principle sketches"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        for p in PLANES:
            _create_plane(p)
        context.scene.sdm_view_project_status = f"Initialised principle planes: {', '.join(PLANES)}"
        return {"FINISHED"}


class SDMVIEW_OT_sketches_add_polyline(bpy.types.Operator):
    """Add a default polyline to the selected plane sketch: three points
    laid out in the plane, ready to be edited in Edit Mode (Tab → G to grab
    vertices)."""

    bl_idname = "sdmview.sketches_add_polyline"
    bl_label = "Add polyline"
    bl_options = {"REGISTER", "UNDO"}

    plane: StringProperty(default="xy")  # type: ignore[valid-type]

    def execute(self, context):
        plane = self.plane.lower()
        if plane not in PLANES:
            self.report({"ERROR"}, f"unknown plane {plane!r}")
            return {"CANCELLED"}
        # Find the next free id within this plane.
        existing = [
            o
            for o in bpy.data.objects
            if o.get("sdm_view_role") == "principle_curve" and o.get("sdm_view_plane") == plane
        ]
        idx = len(existing) + 1
        color_name, color_rgba = _color_for_index(idx - 1)
        curve_id = f"polyline_{idx:02d}"
        _make_polyline_object(
            plane=plane,
            curve_id=curve_id,
            points_2d=[(0.0, 0.0, "p0"), (5.0, 0.0, "p1"), (5.0, 5.0, "p2")],
            color_name=color_name,
            color_rgba=color_rgba,
            label=curve_id,
        )
        context.scene.sdm_view_project_status = f"+ {curve_id} on {plane} ({color_name})"
        _refresh_all_dimensions()
        # Auto-sync so the project JSON reflects the new curve immediately.
        bpy.ops.sdmview.project_sync()
        return {"FINISHED"}


class SDMVIEW_OT_sketches_clear(bpy.types.Operator):
    """Remove every principle-curve from the scene (planes are kept)."""

    bl_idname = "sdmview.sketches_clear"
    bl_label = "Clear sketch curves"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        n = 0
        for obj in [o for o in bpy.data.objects if o.get("sdm_view_role") == "principle_curve"]:
            bpy.data.objects.remove(obj, do_unlink=True)
            n += 1
        _clear_all_dimensions()
        context.scene.sdm_view_project_status = f"Cleared {n} curve(s)"
        # Persist the empty state too, so JSON matches scene.
        bpy.ops.sdmview.project_sync()
        return {"FINISHED"}


# ── Panel ──────────────────────────────────────────────────────────────────


class SDMVIEW_PT_sketches(bpy.types.Panel):
    bl_label = "Principle sketches"
    bl_idname = "SDMVIEW_PT_sketches"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SDM"
    bl_parent_id = "SDMVIEW_PT_main"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(
            scene,
            "sdm_view_show_dimensions",
            toggle=True,
            icon="DRIVER_DISTANCE",
        )

        layout.operator("sdmview.sketches_init", icon="MESH_PLANE", text="Init principle planes")
        layout.separator()

        for p in PLANES:
            box = layout.box()
            box.label(text=f"{p.upper()} sketch", icon="EMPTY_AXIS")
            op = box.operator(
                "sdmview.sketches_add_polyline", icon="ADD", text=f"Add polyline on {p}"
            )
            op.plane = p

        layout.separator()
        layout.operator("sdmview.sketches_clear", icon="TRASH", text="Clear all sketch curves")


# ── Registration ───────────────────────────────────────────────────────────

_classes = (
    SDMVIEW_OT_sketches_init,
    SDMVIEW_OT_sketches_add_polyline,
    SDMVIEW_OT_sketches_clear,
    SDMVIEW_PT_sketches,
)


def register() -> None:
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.sdm_view_show_dimensions = BoolProperty(
        name="Show dimensions",
        description="Show segment-length labels on sketch curves.",
        default=True,
        update=_on_show_dimensions_change,
    )


def unregister() -> None:
    del bpy.types.Scene.sdm_view_show_dimensions
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
