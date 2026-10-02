"""Pure-Python helpers: no bpy. Safe to import from system Python.

Modules:
- `vdb_export`        : numpy SDF grid → `.vdb` (OpenVDB FloatGrid LEVEL_SET)
- `annotation_state`  : JSON I/O for /tmp/*-user-dots.json
"""

from .vdb_export import (
    write_flat_sdf_grid_vdb,
    write_level_set_vdb,
    write_multi_level_set_vdb,
)

__all__ = [
    "write_level_set_vdb",
    "write_multi_level_set_vdb",
    "write_flat_sdf_grid_vdb",
]
