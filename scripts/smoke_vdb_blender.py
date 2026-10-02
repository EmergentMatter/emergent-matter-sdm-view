"""Headless Blender smoke test for the sdm-view VDB pipeline.

Exercises the real round trip end-to-end inside Blender's bundled Python:
  1. import openvdb (this is the whole reason the export path exists)
  2. build a synthetic sphere level-set grid in numpy (NO JAX)
  3. write it with `sdm_view.io.write_level_set_vdb`
  4. import it back with `sdm_view.blender.vdb_import.import_vdb_as_volume`
  5. attach a VolumeToMesh modifier with `add_volume_to_mesh`

This is NOT pytest-collected (lives in scripts/, needs Blender). Run:

    /Applications/Blender.app/Contents/MacOS/Blender --background \\
        --python scripts/smoke_vdb_blender.py

We deliberately do NOT assert mesh vertex counts: Blender's depsgraph does
not evaluate the VolumeToMesh modifier in --background mode, so a baked mesh
is not available headless. We verify the modifier is attached instead.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make sdm-view's src/ importable (scripts/ -> repo root -> src).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    print("\n=== sdm-view VDB smoke test ===")

    try:
        import openvdb  # noqa: F401  # presence check: import succeeding is the signal, not the module's contents

        print("  [OK] openvdb imported")
    except ModuleNotFoundError as e:
        print(f"  [FAIL] openvdb import: {e}")
        print("  This script must run inside Blender's bundled Python.")
        return 1

    import numpy as np

    from sdm_view.blender.vdb_import import add_volume_to_mesh, import_vdb_as_volume
    from sdm_view.io import write_level_set_vdb

    print("  [OK] sdm_view imported")

    # 2. Synthetic sphere level set: 24^3 grid, 1.0mm voxels, radius 8 centered.
    n = 24
    voxel = 1.0
    radius = 8.0
    center = (n - 1) / 2.0
    ax = np.arange(n, dtype=np.float32) - center
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    grid = (np.sqrt(xx * xx + yy * yy + zz * zz) - radius).astype(np.float32)
    print(
        f"  [OK] synthetic sphere grid: shape={grid.shape}, "
        f"min={float(grid.min()):.2f}, max={float(grid.max()):.2f}"
    )

    with tempfile.TemporaryDirectory(prefix="sdm_view_vdb_smoke_") as tmpdir:
        out_path = Path(tmpdir) / "sphere.vdb"
        origin = (-center * voxel, -center * voxel, -center * voxel)

        # 3. Write
        written = write_level_set_vdb(grid, voxel, origin, out_path, grid_name="sdf")
        assert written.exists(), f"vdb not written: {written}"
        size = written.stat().st_size
        assert size > 0, f"vdb is empty: {written}"
        print(f"  [OK] write_level_set_vdb: {written} ({size} bytes)")

        # 4. Import back as a Volume
        obj = import_vdb_as_volume(written, name="SmokeSphere")
        assert obj.type == "VOLUME", f"expected VOLUME, got {obj.type}"
        grids = list(obj.data.grids) if hasattr(obj.data, "grids") else []
        if not grids:
            try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
                obj.data.grids.load()
            except Exception:
                pass
            grids = list(obj.data.grids)
        assert grids, "Volume loaded but grid list is empty"
        print(
            f"  [OK] import_vdb_as_volume: object={obj.name!r}, type={obj.type}, "
            f"grids={[g.name for g in grids]}"
        )

        # 5. Mesh the volume (not applied: headless depsgraph won't bake)
        mesh_obj = add_volume_to_mesh(obj, voxel, apply=False)
        vtm = [m for m in mesh_obj.modifiers if m.type == "VOLUME_TO_MESH"]
        assert vtm, "no VOLUME_TO_MESH modifier attached"
        assert vtm[0].object is obj, "VolumeToMesh modifier not pointed at the volume"
        print(
            f"  [OK] add_volume_to_mesh: host={mesh_obj.name!r}, "
            f"modifier={vtm[0].name!r} -> volume={vtm[0].object.name!r}"
        )

    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
