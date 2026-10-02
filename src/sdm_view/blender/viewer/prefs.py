"""Viewer settings: primarily the path to the sdm-core Python executable.

Two sources, in priority order:

1. **AddonPreferences**: when ``sdm_view`` is installed and enabled as a
   Blender add-on, the usual Preferences UI applies.
2. **Runtime fallback**: when sdm-view is registered from a launcher
   script (``scripts/open_viewer.py``, a CEM's dev script), there is no
   addon entry to hang preferences on. ``get_prefs`` then returns the
   module-level ``runtime`` object, seeded from ``$SDM_PYTHON`` or an
   auto-discovered sibling ``emergent-matter-sdm-core/.venv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import bpy

_PACKAGE = __package__ or "sdm_view.blender.viewer"


def _discover_sdm_core_python() -> str:
    env = os.environ.get("SDM_PYTHON")
    if env:
        return env
    # Walk up from this file looking for a checkout of sdm-core next to
    # sdm-view (the standard ~/Code layout).
    for ancestor in Path(__file__).resolve().parents:
        cand = ancestor / "emergent-matter-sdm-core" / ".venv" / "bin" / "python"
        if cand.exists():
            return str(cand)
    return "python"


@dataclass
class RuntimePrefs:
    """Script-session settings; mirrors the AddonPreferences fields."""

    python_executable: str = field(default_factory=_discover_sdm_core_python)
    keep_emitted_artifacts: bool = True


runtime = RuntimePrefs()


class SDMViewerPreferences(bpy.types.AddonPreferences):
    bl_idname = _PACKAGE

    # Blender registers a property by introspecting __annotations__ for the
    # actual bpy.props.*Property() object, so these annotations are load-bearing
    # calls, not real types -- mypy rejects a call in annotation position (PEP
    # 484), and no typed stub exists for bpy.props here (see CLAUDE.md's "bpy
    # dependency is implicit"). type: ignore[valid-type] on each is structural,
    # not a bug to fix.
    python_executable: bpy.props.StringProperty(  # type: ignore[valid-type]
        name="sdm-core Python",
        description=(
            "Path to the Python interpreter that has ``software_defined_matter`` "
            "installed. Typically the python inside a uv-managed venv, e.g. "
            "/path/to/emergent-matter-sdm-core/.venv/bin/python."
        ),
        subtype="FILE_PATH",
        default="python",
    )

    keep_emitted_artifacts: bpy.props.BoolProperty(  # type: ignore[valid-type]
        name="Keep emitted artifacts",
        description=(
            "Keep the generated sdf_lib.glsl / sdf_scene.glsl / meta.json next to "
            "the .sdm (handy for debugging). Otherwise write to a temp dir."
        ),
        default=True,
    )

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "python_executable")
        layout.prop(self, "keep_emitted_artifacts")
        layout.label(
            text=("Run `which python` inside your sdm-core venv to find the right path."),
            icon="INFO",
        )


def get_prefs(context):
    try:
        return context.preferences.addons[_PACKAGE].preferences
    except KeyError:
        return runtime


def register() -> None:
    # Fails when sdm_view is not an installed add-on (script-registered
    # session); the runtime fallback covers that path.
    try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
        bpy.utils.register_class(SDMViewerPreferences)
    except (ValueError, RuntimeError):
        pass


def unregister() -> None:
    try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
        bpy.utils.unregister_class(SDMViewerPreferences)
    except (ValueError, RuntimeError):
        pass
