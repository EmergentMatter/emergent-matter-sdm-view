"""Debounced, threaded auto-rebuild for re-emit / topology control edits.

The panel calls :func:`check` every draw. When a staged (non-live) control
value changes, the part is marked dirty; once the value has been stable for
``DEBOUNCE_S`` the rebuild pipeline runs:

  main thread: stage values into the .sdm (+ a ``rebuild_hint`` naming the
               params the user actually moved, so the generator can let
               intent edits win reconciliation against derived live pairs)
  worker thread: run ``metadata.generator``, then the sdm-core emitter
               (both are subprocess / file work; no bpy access)
  main thread: compile the GPU shader, swap the renderer, refresh props,
               resize the bbox wire, redraw

Edits made while a rebuild is running are preserved and trigger a follow-up
rebuild. The Rebuild Part button remains as a manual "rebuild now".
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import bpy

from . import flightrec
from .prefs import get_prefs
from .sidecar import emit_artifacts
from .viewport import (
    compile_part_shader,
    get_renderer,
    register_renderer,
    unregister_renderer,
    warmup_pso,
)

DEBOUNCE_S = 0.8
_TICK_S = 0.25

# obj_name -> mutable state dict:
#   sig      last-seen staged-value signature (change detection)
#   dirty    a staged value changed; rebuild once stable
#   last     time.monotonic() of the most recent change
#   running  worker thread in flight
#   done     worker finished; finish step pending on main thread
#   staged   {prop_key: value} written to the .sdm at start
#   artifacts / error   worker results
_STATE: dict = {}


def _prop_key(c) -> str:
    from .import_operator import CTL_PROP_PREFIX

    return c.uniform if c.cls == "live" else f"{CTL_PROP_PREFIX}{c.param}"


def check(obj, renderer) -> None:
    """Called from the panel's draw(): detect control-value edits.

    ALL controls are watched: including live ones. A live uniform gives the
    instant preview, but several live params also feed BAKED geometry (e.g.
    a blade's polygon vertices derive from a bore radius), so a live
    drag leaves the part internally inconsistent (rings move, blades hang in
    space) until the generator re-derives everything. The debounced rebuild
    reconciles after the drag settles; the shader swap is seamless because
    prop values are preserved across it.
    """
    controls = renderer.artifacts.controls
    if not controls:
        return
    # POSE params are pure kinematic DOFs: by contract they feed live
    # uniforms only, never baked geometry, and animation sweeps them every
    # frame: watching them would queue a useless rebuild per pose change
    # (whose prop refresh then snaps the pose back to rest).
    sig = tuple(
        float(obj.get(_prop_key(c), 0.0)) for c in controls if (c.ui or {}).get("role") != "pose"
    )
    e = _STATE.setdefault(obj.name, {"sig": None})
    if e["sig"] is None:
        e["sig"] = sig
        return
    if sig != e["sig"]:
        e["sig"] = sig
        e["dirty"] = True
        e["last"] = time.monotonic()
        e.pop("error", None)
        flightrec.log("rebuild_dirty", part=obj.name, trigger="control_edit")
        _ensure_timer()


def force(obj_name: str) -> None:
    """Manual trigger (the Rebuild Part button): skip the debounce."""
    e = _STATE.setdefault(obj_name, {"sig": None})
    e["dirty"] = True
    e["last"] = time.monotonic() - DEBOUNCE_S
    e.pop("error", None)
    flightrec.log("rebuild_dirty", part=obj_name, trigger="force")
    _ensure_timer()


def status(obj_name: str) -> tuple:
    """(running, error_or_None) for the panel's status row."""
    e = _STATE.get(obj_name) or {}
    return bool(e.get("running") or e.get("dirty") or e.get("done")), e.get("error")


def clear(obj_name: str) -> None:
    _STATE.pop(obj_name, None)


def _ensure_timer() -> None:
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=_TICK_S)


def _tick():
    now = time.monotonic()
    active = False
    for name, e in list(_STATE.items()):
        if e.get("done"):
            e["done"] = False
            _finish(name, e)
        if e.get("running") or e.get("done"):
            active = True
        elif e.get("dirty") and now - e.get("last", 0.0) >= DEBOUNCE_S:
            _start(name, e)
            active = True
        elif e.get("dirty"):
            active = True
    return _TICK_S if active else None


def _start(name: str, e: dict) -> None:
    obj = bpy.data.objects.get(name)
    renderer = get_renderer(name)
    if obj is None or renderer is None:
        _STATE.pop(name, None)
        return

    sdm_path = Path(obj["sdm_source_path"])
    try:
        staged, generator = _stage_values(obj, renderer, sdm_path)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        e["error"] = f"stage failed: {exc}"
        e["dirty"] = False
        return

    prefs = get_prefs(bpy.context)
    e.update(dirty=False, running=True, staged=staged)
    flightrec.log("rebuild_start", part=name, staged=sorted(staged))
    thread = threading.Thread(
        target=_worker,
        args=(e, sdm_path, generator, prefs.python_executable, prefs.keep_emitted_artifacts),
        daemon=True,
    )
    thread.start()


def _stage_values(obj, renderer, sdm_path: Path):
    """Write every control's current value into the .sdm. Returns the staged
    {prop_key: value} snapshot and the generator spec.

    ``metadata.rebuild_hint.changed`` names the params whose value differs
    from what the file held: the generator uses it to decide which side of
    a derived/intent pair the user actually touched.
    """
    doc = json.loads(sdm_path.read_text())
    params = doc.get("params", {})
    staged, changed = {}, []
    for c in renderer.artifacts.controls:
        # POSE params are kinematic DOFs (animation sweeps them constantly):
        # persisting their transient value would bake a mid-animation
        # pose into the .sdm as the part's REST state, after which "Reset
        # to .sdm values" restores a bent part. Design params only.
        if (c.ui or {}).get("role") == "pose":
            continue
        key = _prop_key(c)
        if c.param not in params or key not in obj:
            continue
        val = float(obj[key])
        staged[key] = val
        if abs(val - float(params[c.param].get("value", 0.0))) > 1e-9:
            changed.append(c.param)
        params[c.param]["value"] = val
    doc.setdefault("metadata", {})["rebuild_hint"] = {"changed": changed}
    sdm_path.write_text(json.dumps(doc, indent=2) + "\n")
    return staged, (doc.get("metadata") or {}).get("generator")


def _worker(e: dict, sdm_path: Path, generator, py_exec: str, keep: bool) -> None:
    """Thread body: generator subprocess + emitter CLI. No bpy access."""
    t0 = time.monotonic()
    t_gen = 0.0
    try:
        if generator:
            argv = list(generator.get("argv", []))
            cwd = (sdm_path.parent / generator.get("cwd", ".")).resolve()
            env = dict(os.environ)
            extra = ("/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".local" / "bin"))
            env["PATH"] = os.pathsep.join((*extra, env.get("PATH", "")))
            proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
            t_gen = time.monotonic() - t0
            if proc.returncode != 0:
                raise RuntimeError(
                    f"generator exit {proc.returncode}: {proc.stderr.strip()[-400:]}"
                )
        e["artifacts"] = emit_artifacts(
            sdm_path,
            python_executable=py_exec,
            keep_artifacts=keep,
        )
    except Exception as exc:  # noqa: BLE001  # surfaced in the panel
        e["error"] = str(exc)
    finally:
        e["running"] = False
        e["done"] = True
        flightrec.log(
            "rebuild_worker_done",
            part=sdm_path.stem,
            error=e.get("error"),
            gen_s=round(t_gen, 2),
            total_s=round(time.monotonic() - t0, 2),
        )


def _finish(name: str, e: dict) -> None:
    """Main-thread completion: GPU compile, renderer swap, prop refresh."""
    from .import_operator import install_control_properties, placeholder_mesh

    obj = bpy.data.objects.get(name)
    artifacts = e.pop("artifacts", None)
    staged = e.pop("staged", {})
    if obj is None:
        _STATE.pop(name, None)
        return
    if artifacts is None:
        _redraw()
        return  # worker error; e["error"] is shown by the panel

    try:
        new_renderer = compile_part_shader(artifacts)
        # Pay the Metal PSO compile HERE, before the swap. Without this the
        # multi-second pipeline compile lands inside the viewport draw
        # handler on the new shader's first frame (import_operator warms up
        # for exactly the same reason).
        warmup_pso(new_renderer)
    except Exception as exc:  # noqa: BLE001  # GPU compile errors are opaque
        e["error"] = f"shader compile: {exc}"
        flightrec.log("rebuild_swap_failed", part=name, exc=repr(exc))
        _redraw()
        return

    unregister_renderer(name)
    register_renderer(name, new_renderer)
    flightrec.log("rebuild_swap", part=name)

    # Preserve values the user moved while the rebuild was running, then
    # refresh everything else from the regenerated file. Pose DOFs are
    # never staged (transient by design): carry the current pose across
    # the refresh so a rebuild doesn't snap the part back to rest.
    moved = {
        k: float(obj[k]) for k, v in staged.items() if k in obj and abs(float(obj[k]) - v) > 1e-9
    }
    pose = {
        k: float(obj[k])
        for k in (
            _prop_key(c)
            for c in new_renderer.artifacts.controls
            if (c.ui or {}).get("role") == "pose"
        )
        if k in obj
    }
    install_control_properties(obj, artifacts)
    for k, v in {**pose, **moved}.items():
        obj[k] = v

    sdm_path = Path(obj["sdm_source_path"])
    old_mesh = obj.data
    obj.data = placeholder_mesh(sdm_path.stem, artifacts.bbox)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

    # New file values => new baseline signature (avoids a rebuild loop from
    # the generator's own reconciliation of derived params).
    e["sig"] = None
    if moved:
        e["dirty"] = True
        e["last"] = time.monotonic()

    _redraw()


def _redraw() -> None:
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type in ("VIEW_3D", "PROPERTIES"):
                area.tag_redraw()
