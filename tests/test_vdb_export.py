"""Tests for the `sdm_view.io.vdb_export` writers.

Two layers:

- **Validation + backend gating** (plain Python, no Blender). Every writer
  validates its arguments before importing `openvdb`, so the ValueError
  cases run unconditionally. The ModuleNotFoundError path is only meaningful
  when `openvdb` is absent (the normal sdm-view venv) and is skipped if
  Blender's `openvdb` somehow is on the path.
- **Round-trips** (require `openvdb`). Skipped via `importorskip` in a plain
  `uv run pytest`; run when conda's `openvdb` is installed
  (`conda install -c conda-forge openvdb`).
"""

import numpy as np
import pytest

from sdm_view.io import (
    write_flat_sdf_grid_vdb,
    write_level_set_vdb,
    write_multi_level_set_vdb,
)

# ── write_level_set_vdb: validation + gating ──────────────────────────────


def test_bad_shape_raises(tmp_path):
    grid_2d = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="3D"):
        write_level_set_vdb(grid_2d, 1.0, (0.0, 0.0, 0.0), tmp_path / "x.vdb")


def test_nonpositive_voxel_raises(tmp_path):
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="voxel_size_mm"):
        write_level_set_vdb(grid, 0.0, (0.0, 0.0, 0.0), tmp_path / "x.vdb")
    with pytest.raises(ValueError, match="voxel_size_mm"):
        write_level_set_vdb(grid, -1.0, (0.0, 0.0, 0.0), tmp_path / "x.vdb")


def test_bad_origin_length_raises(tmp_path):
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="origin_mm"):
        write_level_set_vdb(grid, 1.0, (0.0, 0.0), tmp_path / "x.vdb")


def test_missing_openvdb_raises_helpful_error(tmp_path):
    """A valid call without openvdb must raise ModuleNotFoundError naming Blender."""
    try:
        import openvdb  # noqa: F401  # presence check: import succeeding is the signal being tested, not the module's contents
    except ModuleNotFoundError:
        pass
    else:
        pytest.skip("openvdb is present; ModuleNotFoundError path not exercised")

    grid = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ModuleNotFoundError, match="Blender"):
        write_level_set_vdb(grid, 1.0, (0.0, 0.0, 0.0), tmp_path / "x.vdb")


# ── write_multi_level_set_vdb: validation + gating ────────────────────────


def test_multi_empty_grids_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_multi_level_set_vdb({}, 1.0, tmp_path / "x.vdb")


def test_multi_bad_shape_raises(tmp_path):
    grids = {"a": (np.zeros((4, 4), dtype=np.float32), (0.0, 0.0, 0.0))}
    with pytest.raises(ValueError, match="3D"):
        write_multi_level_set_vdb(grids, 1.0, tmp_path / "x.vdb")


def test_multi_nonpositive_voxel_raises(tmp_path):
    grids = {"a": (np.zeros((4, 4, 4), dtype=np.float32), (0.0, 0.0, 0.0))}
    with pytest.raises(ValueError, match="voxel_size_mm"):
        write_multi_level_set_vdb(grids, 0.0, tmp_path / "x.vdb")


def test_multi_missing_openvdb_raises_helpful_error(tmp_path):
    try:
        import openvdb  # noqa: F401  # presence check: import succeeding is the signal being tested, not the module's contents
    except ModuleNotFoundError:
        pass
    else:
        pytest.skip("openvdb is present; ModuleNotFoundError path not exercised")

    grids = {"a": (np.zeros((2, 2, 2), dtype=np.float32), (0.0, 0.0, 0.0))}
    with pytest.raises(ModuleNotFoundError, match="Blender"):
        write_multi_level_set_vdb(grids, 1.0, tmp_path / "x.vdb")


# ── write_flat_sdf_grid_vdb: validation + gating ──────────────────────────


def test_flat_size_mismatch_raises(tmp_path):
    flat = np.zeros(7, dtype=np.float32)  # 7 != 2*2*2
    with pytest.raises(ValueError, match="implies 8"):
        write_flat_sdf_grid_vdb(flat, (2, 2, 2), (0.0, 0.0, 0.0), 1.0, tmp_path / "x.vdb")


def test_flat_missing_openvdb_raises_helpful_error(tmp_path):
    """flat delegates to write_level_set_vdb, so it inherits the Blender error."""
    try:
        import openvdb  # noqa: F401  # presence check: import succeeding is the signal being tested, not the module's contents
    except ModuleNotFoundError:
        pass
    else:
        pytest.skip("openvdb is present; ModuleNotFoundError path not exercised")

    flat = np.zeros(8, dtype=np.float32)
    with pytest.raises(ModuleNotFoundError, match="Blender"):
        write_flat_sdf_grid_vdb(flat, (2, 2, 2), (0.0, 0.0, 0.0), 1.0, tmp_path / "x.vdb")


# ── Round-trips (require openvdb) ─────────────────────────────────────────


@pytest.fixture
def openvdb():
    return pytest.importorskip(
        "openvdb",
        reason="openvdb not installed (conda install -c conda-forge openvdb)",
    )


def _sphere_sdf_grid(
    d_radius_mm: float, d_voxel_mm: float, n: int
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """A (n, n, n) SDF grid sampling a sphere centred in the cube.
    Returns (grid, origin_mm)."""
    half = (n - 1) * d_voxel_mm / 2.0
    coords = np.linspace(-half, half, n, dtype=np.float64)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    sdf = np.sqrt(x * x + y * y + z * z) - d_radius_mm
    return sdf.astype(np.float32), (-half, -half, -half)


def test_level_set_roundtrip_preserves_metadata(openvdb, tmp_path):
    grid, origin = _sphere_sdf_grid(2.0, 0.5, 16)
    out = tmp_path / "sphere.vdb"
    written = write_level_set_vdb(grid, 0.5, origin, out, grid_name="my_sdf")
    assert written == out and out.stat().st_size > 0
    (g,) = openvdb.readAllGridMetadata(str(out))
    assert g.name == "my_sdf"
    assert g.gridClass == openvdb.GridClass.LEVEL_SET
    assert g.transform.voxelSize()[0] == pytest.approx(0.5)


def test_multi_writes_named_grids(openvdb, tmp_path):
    g_a, origin_a = _sphere_sdf_grid(1.0, 0.5, 8)
    g_b, origin_b = _sphere_sdf_grid(2.0, 0.5, 8)
    out = tmp_path / "two.vdb"
    write_multi_level_set_vdb({"small": (g_a, origin_a), "large": (g_b, origin_b)}, 0.5, out)
    metas = openvdb.readAllGridMetadata(str(out))
    assert sorted(g.name for g in metas) == ["large", "small"]
    assert all(g.gridClass == openvdb.GridClass.LEVEL_SET for g in metas)


def test_flat_reshapes_and_writes(openvdb, tmp_path):
    grid, origin = _sphere_sdf_grid(2.0, 0.5, 8)
    nx, ny, nz = grid.shape
    out = tmp_path / "flat.vdb"
    write_flat_sdf_grid_vdb(grid.reshape(-1), (nx, ny, nz), origin, 0.5, out, grid_name="from_flat")
    (g,) = openvdb.readAllGridMetadata(str(out))
    assert g.name == "from_flat"
