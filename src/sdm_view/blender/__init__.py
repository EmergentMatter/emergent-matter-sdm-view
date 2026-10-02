"""Blender-side of sdm-view: bpy-using operators and panels.

Only loaded inside Blender. The submodules each cover one feature in the
catalog (see CLAUDE.md for the module map). A consuming CEM typically:

    1. Defines its frozen `Parameters` dataclass and `make_components(params)`.
    2. Builds its addon `register()` by composing the sdm_view feature modules.

Not every feature in the catalog has a module here yet; see the org issue
tracker for what's planned but unbuilt.
"""
