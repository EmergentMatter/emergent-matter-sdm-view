"""sdm-view: shared Blender authoring layer for EmergentMatter SDM CEMs.

The public surface is split:

- `sdm_view.io.*`      : pure-Python helpers (safe outside Blender)
- `sdm_view.blender.*` : bpy-using operators / panels (only inside Blender)

See `CLAUDE.md` for the feature module map.
"""

__version__ = "0.0.0"
