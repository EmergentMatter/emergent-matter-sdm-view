"""Import a `.vdb` file as `bpy.data.volumes` and attach a VolumeToMesh
modifier.

This is the primary geometry path in sdm-view (decided 2026-05-21). The
`.vdb` carries the full signed-distance field; a `VOLUME_TO_MESH` modifier
extracts the iso-surface. The modifier `threshold` is the iso level, so it
doubles as a "what if the wall is thinner / the neck is wider" knob without
re-evaluating the JAX SDF host-side.

**Blender-only.** `bpy` is imported lazily inside each function so the rest of
the package stays importable from system Python.

The `volume_import` + `VOLUME_TO_MESH` modifier settings follow the
pattern this was generalised from, where a consuming CEM did the same import
per part before it moved here. (On Blender 5.x the
modifier is a *mesh* modifier that references the Volume as input. It can no
longer be added to the Volume object directly, so `add_volume_to_mesh` hosts
it on a generated mesh object; see that function's docstring.)

The live-iso GeometryNodes variant (a `GeometryNodeVolumeToMesh` tree with a
threshold socket exposed as a viewport slider) is future work; these helpers
cover the import + apply-to-mesh path first.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only for the return-type annotation below; bpy itself stays a lazy,
    # function-local import (see the module docstring) so this module is
    # importable from system Python without Blender on the path.
    import bpy


def import_vdb_as_volume(vdb_path: str | Path, name: str | None = None) -> bpy.types.Object:
    """Import a `.vdb` file as a Blender Volume object.

    Args:
        vdb_path: Path to the `.vdb` file.
        name: Optional name for the resulting object. If given, the active
            object is renamed to it.

    Returns:
        The imported Volume object (`obj.type == "VOLUME"`).
    """
    import bpy

    bpy.ops.object.volume_import(filepath=str(vdb_path))
    obj = bpy.context.active_object
    if name is not None:
        obj.name = name

    # Blender loads VDB grids lazily (on first viewport draw). Force-load so
    # headless callers can introspect `obj.data.grids` right away.
    try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
        obj.data.grids.load()
    except Exception:
        pass

    return obj


def add_volume_to_mesh(
    vol_obj, voxel_size_mm: float, *, threshold: float = 0.0, apply: bool = True
):
    """Mesh a Volume object via a VolumeToMesh modifier.

    In Blender 5.x the `VOLUME_TO_MESH` modifier is a *mesh* modifier that
    references a Volume as input (it cannot be added to a Volume object; the
    operator's context enum rejects it). So this creates a host mesh object,
    drops the modifier on it, and points `modifier.object` at `vol_obj`.

    Args:
        vol_obj: A Blender Volume object (e.g. from `import_vdb_as_volume`).
        voxel_size_mm: Mesher voxel size in mm. Matches the grid's voxel size.
        threshold: Iso level to extract (0.0 = the level-set surface).
        apply: If True, apply the modifier to bake a real mesh. In headless
            (`--background`) runs the apply is a no-op and is swallowed.

    Returns:
        The host mesh object carrying the VolumeToMesh modifier.
    """
    import bpy

    mesh_data = bpy.data.meshes.new(f"{vol_obj.name}_VolumeToMesh")
    mesh_obj = bpy.data.objects.new(f"{vol_obj.name}_mesh", mesh_data)
    bpy.context.collection.objects.link(mesh_obj)
    # Share the volume's world transform so the meshed surface lands in place.
    mesh_obj.matrix_world = vol_obj.matrix_world.copy()

    mod = mesh_obj.modifiers.new("SDMView_VolumeToMesh", "VOLUME_TO_MESH")
    mod.object = vol_obj
    mod.threshold = threshold
    mod.adaptivity = 0.0
    # Drive the mesher off our explicit voxel size. The 5.x default
    # resolution_mode is "GRID" (uses the VDB's own voxel size); we set
    # VOXEL_SIZE so voxel_size_mm is honored regardless of grid metadata.
    if hasattr(mod, "resolution_mode"):
        mod.resolution_mode = "VOXEL_SIZE"
    mod.voxel_amount = 0
    mod.voxel_size = voxel_size_mm

    if apply:
        bpy.context.view_layer.objects.active = mesh_obj
        try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except RuntimeError:
            # Leave the modifier live if apply fails (e.g. headless context).
            pass

    return mesh_obj
