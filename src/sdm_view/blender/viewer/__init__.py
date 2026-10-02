"""GLSL ray-march viewer for ``.sdm`` parts: sdm-view feature module.

Imports a ``.sdm`` part, runs the ``sdm-core`` GLSL emitter as a subprocess,
and renders the resulting SDF as a ray-marched fullscreen pass in the 3D
viewport. Free Params surface as sliders in the N-panel; staged params
rebuild through the part's declared generator; authored animations play
back as keyframed pose sweeps.

Talks to ``sdm-core`` over a single subprocess boundary:
``<python> -m software_defined_matter.glsl <file>.sdm --out <tmpdir>``:
the interpreter comes from the addon preference or the ``prefs.runtime``
fallback (see ``prefs.py``).

Started life as a standalone EmergentMatter Blender addon, merged here
on 2026-07-07 with its history; now registered like every other sdm-view
feature module. Launcher: ``scripts/open_viewer.py``.
"""

from __future__ import annotations

from . import import_operator, panel, prefs, viewport

_MODULES = (prefs, viewport, import_operator, panel)


def register() -> None:
    for m in _MODULES:
        m.register()


def unregister() -> None:
    # Tear down the viewport draw handler first; otherwise unregistering the
    # operator/panel leaves a dangling callback that crashes on next redraw.
    for m in reversed(_MODULES):
        m.unregister()
