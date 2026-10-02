"""Export a numpy SDF grid to an OpenVDB `.vdb` file as a LEVEL_SET FloatGrid.

Target consumer: a CEM's Layer-1 SDF code evaluates `eval_sdf_grid` on its
JAX SDF, gets back a `(nx, ny, nz)` numpy array of signed distances in mm,
and calls `write_level_set_vdb(grid, voxel_size_mm, origin_mm, out_path)`. The
result is a `.vdb` file ready for `sdm_view.blender.vdb_import` to consume.

Two convenience writers sit alongside it, ported from an earlier writer:
`write_multi_level_set_vdb` packs several named grids into one file (each
stays individually selectable in Blender), and `write_flat_sdf_grid_vdb`
takes a flat `(N,)` sample array + shape + bbox-min and routes through
`write_level_set_vdb` so the caller skips the manual reshape.

Backend: **Blender's bundled `openvdb` module**. PyPI `pyopenvdb` has no
cp313 wheels and requires a source-built system OpenVDB; we sidestep that by
running this module inside Blender's Python (either via the addon, or via
`blender --background --python <script>` for headless export).

This module imports cleanly without Blender (only numpy is needed at import
time). `openvdb` is imported lazily, inside `write_level_set_vdb`, so that the
pure-Python `io/` layer stays consumable from system Python.

Two details here are load-bearing and were learned the hard way: the
numpy-compat dance for Blender's bundled numpy, and the integer-ijk origin
offset passed to `copyFromArray`, which is needed because
`transform.translate` is buggy.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def write_level_set_vdb(
    sdf_grid: np.ndarray,
    voxel_size_mm: float,
    origin_mm: tuple[float, float, float],
    out_path: str | Path,
    *,
    grid_name: str = "sdf",
) -> Path:
    """Write a 3D SDF array to an OpenVDB `.vdb` file as a LEVEL_SET grid.

    Args:
        sdf_grid: 3D array of signed distances, shape (nx, ny, nz), in mm.
            Negative values are inside the surface.
        voxel_size_mm: Grid spacing in mm.
        origin_mm: World-space (x, y, z) position of `sdf_grid[0, 0, 0]` in mm.
        out_path: Output `.vdb` path. Parent directories are created.
        grid_name: Name for the VDB grid (e.g. "sdf", "density").

    Returns:
        The output path as a `Path`.

    Raises:
        ValueError: If `sdf_grid` is not 3D, `voxel_size_mm` is not positive,
            or `origin_mm` does not have exactly 3 components.
        ModuleNotFoundError: If `openvdb` is unavailable (i.e. not running
            inside Blender's bundled Python).
    """
    # Validate everything BEFORE touching openvdb so that pure-Python callers
    # get clean argument errors even where the backend is unavailable.
    if sdf_grid.ndim != 3:
        raise ValueError(
            f"sdf_grid must be 3D (nx, ny, nz); got ndim={sdf_grid.ndim} "
            f"with shape {sdf_grid.shape}"
        )
    if voxel_size_mm <= 0:
        raise ValueError(f"voxel_size_mm must be > 0; got {voxel_size_mm}")
    if len(origin_mm) != 3:
        raise ValueError(f"origin_mm must have 3 components (x, y, z); got {len(origin_mm)}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import openvdb
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Blender's bundled `openvdb` module is required to write .vdb files. "
            "PyPI `pyopenvdb` has no cp313 wheels, so run this inside Blender: "
            "either from the sdm-view addon, or headless via "
            "`blender --background --python <script>`."
        ) from e

    # Blender's openvdb is compiled against Blender's bundled numpy; a fresh
    # contiguous float32 buffer is accepted regardless of which numpy produced
    # the input array, so reconstruct one before copyFromArray.
    safe = np.array(sdf_grid, dtype=np.float32, copy=True, order="C")

    g = openvdb.FloatGrid()
    g.name = grid_name
    g.transform = openvdb.createLinearTransform(voxelSize=float(voxel_size_mm))

    # Apply the world origin as an INTEGER ijk index offset on copyFromArray.
    # transform.translate is buggy with copyFromArray, so we offset the index
    # space instead (matches em_sdm/vdb.py).
    ox, oy, oz = origin_mm
    ijk = (
        round(ox / voxel_size_mm),
        round(oy / voxel_size_mm),
        round(oz / voxel_size_mm),
    )
    g.copyFromArray(safe, ijk)
    g.gridClass = openvdb.GridClass.LEVEL_SET

    openvdb.write(str(out_path), grids=[g])
    return out_path


def write_multi_level_set_vdb(
    grids: Mapping[str, tuple[np.ndarray, tuple[float, float, float]]],
    voxel_size_mm: float,
    out_path: str | Path,
) -> Path:
    """Write multiple named SDF grids into one `.vdb` file.

    Each entry lands as a separately selectable LEVEL_SET grid inside
    Blender: useful for a compound part whose sub-components (e.g. sun /
    planets / ring) should stay individually pickable rather than fuse into
    a single SDF. All grids share `voxel_size_mm`.

    Args:
        grids: Mapping of `grid_name -> (sdf_grid, origin_mm)`. Each
            `sdf_grid` is `(nx, ny, nz)` in mm; `origin_mm` is the
            world-space `(x, y, z)` of that grid's voxel `(0, 0, 0)`.
        voxel_size_mm: Voxel spacing in mm, applied to every grid.
        out_path: Output `.vdb` path. Parent directories are created.

    Returns:
        The output path as a `Path`.

    Raises:
        ValueError: empty `grids`, non-positive `voxel_size_mm`, or any grid
            that is not 3D / origin without exactly 3 components.
        ModuleNotFoundError: If `openvdb` is unavailable (not inside Blender).
    """
    # Validate before touching openvdb so pure-Python callers get clean errors.
    if not grids:
        raise ValueError("grids is empty; pass at least one named grid")
    if voxel_size_mm <= 0:
        raise ValueError(f"voxel_size_mm must be > 0; got {voxel_size_mm}")
    for name, (grid, origin_mm) in grids.items():
        if np.ndim(grid) != 3:
            raise ValueError(f"grid {name!r} must be 3D (nx, ny, nz); got ndim={np.ndim(grid)}")
        if len(origin_mm) != 3:
            raise ValueError(
                f"origin_mm for grid {name!r} must have 3 components; got {len(origin_mm)}"
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import openvdb
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Blender's bundled `openvdb` module is required to write .vdb files. "
            "PyPI `pyopenvdb` has no cp313 wheels, so run this inside Blender: "
            "either from the sdm-view addon, or headless via "
            "`blender --background --python <script>`."
        ) from e

    vdb_grids = []
    for grid_name, (sdf_grid, origin_mm) in grids.items():
        # Same buffer-protocol rebuild as the single-grid writer: Blender's
        # openvdb wants a contiguous float32 array from its own numpy.
        safe = np.array(sdf_grid, dtype=np.float32, copy=True, order="C")
        g = openvdb.FloatGrid()
        g.name = grid_name
        g.transform = openvdb.createLinearTransform(voxelSize=float(voxel_size_mm))
        ox, oy, oz = origin_mm
        ijk = (
            round(ox / voxel_size_mm),
            round(oy / voxel_size_mm),
            round(oz / voxel_size_mm),
        )
        g.copyFromArray(safe, ijk)
        g.gridClass = openvdb.GridClass.LEVEL_SET
        vdb_grids.append(g)

    openvdb.write(str(out_path), grids=vdb_grids)
    return out_path


def write_flat_sdf_grid_vdb(
    sdf_values: np.ndarray,
    grid_shape: tuple[int, int, int],
    bbox_min_mm: tuple[float, float, float],
    voxel_size_mm: float,
    out_path: str | Path,
    *,
    grid_name: str = "sdf",
) -> Path:
    """Convenience: pipe `(N,)`-flat SDF samples straight to a `.vdb`.

    The canonical Layer-1 → VDB path hands back a flat `(N,)` array plus a
    `(nx, ny, nz)` shape and a bbox min; this reshapes and routes through
    `write_level_set_vdb` (which validates inputs and creates parent dirs),
    so the caller skips the manual `.reshape()`.

    Args:
        sdf_values: Flat `(N,)` array of signed distances, `N == nx*ny*nz`.
        grid_shape: `(nx, ny, nz)`.
        bbox_min_mm: World-space `(x, y, z)` in mm of voxel `(0, 0, 0)`,
            the minimum corner of the sampled bbox.
        voxel_size_mm: Voxel spacing in mm.
        out_path: Output `.vdb` path. Parent directories are created.
        grid_name: VDB grid name.

    Returns:
        The output path as a `Path`.

    Raises:
        ValueError: `sdf_values` size does not match `grid_shape` (or any
            error raised by `write_level_set_vdb`).
        ModuleNotFoundError: If `openvdb` is unavailable (not inside Blender).
    """
    nx, ny, nz = grid_shape
    n_expected = nx * ny * nz
    flat = np.asarray(sdf_values)
    if flat.size != n_expected:
        raise ValueError(
            f"sdf_values has {flat.size} elements but grid_shape {grid_shape} implies {n_expected}"
        )
    return write_level_set_vdb(
        flat.reshape(nx, ny, nz),
        voxel_size_mm,
        (float(bbox_min_mm[0]), float(bbox_min_mm[1]), float(bbox_min_mm[2])),
        out_path,
        grid_name=grid_name,
    )


__all__ = [
    "write_level_set_vdb",
    "write_multi_level_set_vdb",
    "write_flat_sdf_grid_vdb",
]
