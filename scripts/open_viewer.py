"""Open the SDM GLSL ray-march viewer live in Blender and LEAVE IT OPEN.

Registers sdm-view's viewer module straight from this repo's ``src/`` (no
addon install needed), points it at your sdm-core venv, imports a ``.sdm``
part, opens the SDM N-panel, frames the part, then hands back to Blender's
event loop so you can orbit and drag the Param sliders yourself. Nothing is
captured or quit; it's an interactive demo entry point.

Prerequisites:
  * A Python interpreter with ``software_defined_matter`` installed
    (typically a uv-managed sdm-core venv). Auto-discovered from a sibling
    ``emergent-matter-sdm-core/.venv``; override with ``SDM_PYTHON``.

Usage:
    SDM_PART=/path/to/part.sdm \\
    blender --factory-startup -P scripts/open_viewer.py

macOS: a Blender launched from a script opens UNFOCUSED, behind other
windows: it looks like it never opened, especially while a first import
sits in the Metal pipeline compile. Raise it after launching:
    (sleep 2 && osascript -e 'tell application "Blender" to activate') &

You may also pass the part path after ``--``:
    blender --factory-startup -P scripts/open_viewer.py -- /path/to/part.sdm
"""

import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Euler

SDM_VIEW_SRC = Path(__file__).resolve().parents[1] / "src"
if str(SDM_VIEW_SRC) not in sys.path:
    sys.path.insert(0, str(SDM_VIEW_SRC))


def _part_path() -> str:
    if "--" in sys.argv:
        rest = sys.argv[sys.argv.index("--") + 1 :]
        if rest:
            return rest[0]
    p = os.environ.get("SDM_PART")
    if not p:
        raise SystemExit("Set SDM_PART=/path/to/part.sdm (env) or pass it after '--'.")
    return p


def _register_viewer() -> None:
    from sdm_view.blender import viewer

    try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
        viewer.unregister()  # tolerate a re-run in the same session
    except Exception:  # noqa: BLE001  # paired with the SIM105 suppression above: best-effort, must not block
        pass
    viewer.register()

    from sdm_view.blender.viewer.prefs import get_prefs

    prefs = get_prefs(bpy.context)
    if prefs.python_executable in ("", "python"):
        raise SystemExit(
            "No sdm-core Python found. Pass SDM_PYTHON=/path/to/.venv/bin/python "
            "or check out emergent-matter-sdm-core next to this repo and "
            "`uv sync` it."
        )
    print(f"[open_viewer] sdm-core python: {prefs.python_executable}")


def _view3d():
    win = bpy.context.window_manager.windows[0]
    for area in win.screen.areas:
        if area.type == "VIEW_3D":
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    # N-panel (show_region_ui) is switched on in _settle():
                    # flipping region visibility at -P time segfaults 5.2's
                    # ED_area_init (areas not yet initialised).
                    space.shading.type = "SOLID"
            return win, area, region
    return win, None, None


_STATE = {"phase": 0, "area": None, "distance": None}


def _frame(win, area, region) -> None:
    r3d = None
    for space in area.spaces:
        if space.type == "VIEW_3D":
            r3d = space.region_3d
            r3d.view_rotation = Euler(
                (math.radians(62), 0.0, math.radians(40)), "XYZ"
            ).to_quaternion()
            r3d.view_perspective = "PERSP"
    try:
        with bpy.context.temp_override(window=win, area=area, region=region):
            bpy.ops.view3d.view_selected()
    except Exception:  # noqa: BLE001  # framing is a nicety
        pass
    if r3d is not None:
        r3d.view_distance *= 2.4
        _STATE["distance"] = r3d.view_distance


def _settle():
    """Switch the N-panel to the SDM tab + reassert framing once the UI has
    drawn, then stop and leave Blender to the user."""
    area = _STATE["area"]
    if area is not None:
        for space in area.spaces:
            if space.type == "VIEW_3D":
                try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
                    space.show_region_ui = True  # N-panel on (safe post-init)
                except Exception:  # noqa: BLE001  # paired with the SIM105 suppression above: best-effort, must not block
                    pass
        for region in area.regions:
            if region.type == "UI":
                try:
                    region.active_panel_category = "SDM"
                    region.tag_redraw()
                except Exception:  # noqa: BLE001  # paired with the SIM105 suppression above: best-effort, must not block
                    pass
        if _STATE["distance"]:
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.region_3d.view_distance = _STATE["distance"]
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            a.tag_redraw()
    _STATE["phase"] += 1
    return 0.5 if _STATE["phase"] < 3 else None


def main() -> None:
    # Clean slate. MCP sessions can't use --factory-startup (the extension
    # won't load), so the user's startup file leaks in: splash screen,
    # default cube/camera/light, last-saved workspace. Persist splash-off
    # (covers every future launch too), delete the startup objects via the
    # data API (a wm.read_homefile here + UI pokes segfaults 5.2's
    # ED_area_init; areas aren't initialised yet at -P time), and pin the
    # Layout workspace.
    try:
        bpy.context.preferences.view.show_splash = False
        bpy.ops.wm.save_userpref()
    except Exception:  # noqa: BLE001  # paired with the SIM105 suppression above: best-effort, must not block
        pass
    for stale in list(bpy.data.objects):
        try:  # noqa: SIM105  # best-effort cleanup; a failure here must not block the surrounding operation
            bpy.data.objects.remove(stale, do_unlink=True)
        except Exception:  # noqa: BLE001  # paired with the SIM105 suppression above: best-effort, must not block
            pass
    try:
        ws = bpy.data.workspaces.get("Layout")
        if ws is not None:
            bpy.context.window_manager.windows[0].workspace = ws
    except Exception:  # noqa: BLE001  # paired with the SIM105 suppression above: best-effort, must not block
        pass
    part = _part_path()
    _register_viewer()
    win, area, region = _view3d()
    if area is None:
        raise SystemExit("No VIEW_3D area to render into.")
    print(f"[open_viewer] importing {part}")
    bpy.ops.sdm.import_sdm(filepath=part)  # runs emitter + compiles shader
    _frame(win, area, region)
    _STATE["area"] = area
    bpy.app.timers.register(_settle, first_interval=0.8)

    if os.environ.get("SDM_AUTOPLAY"):
        # "Show me": start the first authored animation once the window has
        # settled: no panel hunting required.
        def _autoplay():
            try:
                with bpy.context.temp_override(window=win, area=area, region=region):
                    bpy.ops.sdm.play_animation(anim_index=0)
            except Exception:
                import traceback

                traceback.print_exc()
            return

        bpy.app.timers.register(_autoplay, first_interval=1.5)

    print("[open_viewer] ready: orbit with MMB, scroll to zoom, drag the SDM panel sliders.")


main()
