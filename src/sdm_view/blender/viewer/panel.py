"""Sidebar panel (N-key): the full control suite for the active SDM part.

Built from the emitter's per-param control manifest (meta.json "controls"):

- **live** controls scrub a GLSL uniform: instant, no recompilation, and
  keyframable (the draw handler reads the custom prop every redraw, so
  Blender keyframes on it play back as parameter animation).
- **re-emit** and **topology** controls edit staged values; the *Rebuild
  Part* operator writes them into the .sdm, re-runs the part's declared
  generator (``metadata.generator``), re-emits GLSL, and swaps the shader.

Falls back to the flat uniform list for pre-manifest emitters.
"""

from __future__ import annotations

import bpy

from . import rebuild
from .import_operator import CTL_PROP_PREFIX
from .viewport import get_renderer

_PANEL_CATEGORY = "SDM"

_CLASS_ICON = {
    "live": "PROP_ON",
    "re-emit": "FILE_REFRESH",
    "topology": "MOD_BUILD",
}


def _active_sdm_object(context) -> bpy.types.Object | None:
    obj = context.active_object
    if obj is not None and obj.get("sdm_source_path") is not None:
        return obj
    # Clicking empty space deselects and the panel used to vanish ("no
    # controls"). Fall back to the scene's SDM part: with one part (the
    # common case) the panel just stays up; with several, select one.
    for cand in context.scene.objects:
        if cand.get("sdm_source_path") is not None and get_renderer(cand.name):
            return cand
    return None


def _prop_key(c) -> str:
    return c.uniform if c.cls == "live" else f"{CTL_PROP_PREFIX}{c.param}"


# Last-drawn control values per object. The panel region redraws
# continuously while a slider is dragged, so comparing against this in
# draw() lets us tag the 3D viewport the moment a value moves: idprop
# edits don't reliably push a depsgraph update on their own, so without
# this the ray-march only catches up when the user orbits.
_LAST_VALUES: dict = {}


def _sync_viewport_on_change(context, obj, keys) -> None:
    sig = tuple(float(obj.get(k, 0.0)) for k in keys)
    if _LAST_VALUES.get(obj.name) != sig:
        _LAST_VALUES[obj.name] = sig
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _grouped_controls(controls):
    """[(group, [controls sorted by ui.order])] in first-seen group order."""
    groups = {}
    for c in controls:
        groups.setdefault(c.ui.get("group", "params"), []).append(c)
    for members in groups.values():
        members.sort(key=lambda c: c.ui.get("order", 1e9))
    return list(groups.items())


class SDM_PT_part(bpy.types.Panel):
    bl_label = "SDM Part"
    bl_idname = "SDM_PT_part"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = _PANEL_CATEGORY

    @classmethod
    def poll(cls, context):
        return _active_sdm_object(context) is not None

    def draw(self, context):
        layout = self.layout
        obj = _active_sdm_object(context)
        if obj is None:
            return

        layout.label(text=obj.get("sdm_source_path", obj.name), icon="MESH_DATA")

        renderer = get_renderer(obj.name)
        if renderer is None:
            layout.label(text="No active shader for this object.", icon="ERROR")
            layout.label(text="(Re-import the .sdm to rebuild it.)")
            return

        controls = renderer.artifacts.controls
        if not controls:
            # Pre-manifest emitter: flat uniform list, as before.
            col = layout.column(align=True)
            for spec in renderer.artifacts.uniforms:
                if spec.name in obj:
                    col.prop(obj, f'["{spec.name}"]', text=spec.source_param)
        else:
            for group, members in _grouped_controls(controls):
                box = layout.box()

                # Composed-CEM groups (a dependency's passed-up controls)
                # author ui.collapsed and start folded; a bool prop per group
                # (installed at import/rebuild; draw() may not write IDs)
                # remembers the user's toggle.
                open_key = f"sdm_grpopen_{group}"
                if open_key in obj:
                    box.prop(obj, f'["{open_key}"]', text=group.capitalize(), toggle=True)
                    if not obj.get(open_key):
                        continue
                else:
                    box.label(text=group.capitalize())

                col = box.column(align=True)
                for c in members:
                    key = _prop_key(c)
                    if key not in obj:
                        continue
                    label = f"{c.param} [{c.unit}]" if c.unit else c.param
                    row = col.row(align=True)
                    if (c.ui or {}).get("driven"):
                        # Relation-owned: show, don't edit; the generator
                        # re-derives it from the free params on rebuild.
                        val = float(obj.get(key, c.value))
                        row.label(text=f"{label} = {val:.4g}", icon="LINKED")
                        continue
                    row.prop(obj, f'["{key}"]', text=label)
                    choices = (c.ui or {}).get("choices")
                    if choices:
                        i = int(round(float(obj.get(key, c.value))))
                        if 0 <= i < len(choices):
                            row.label(text=choices[i])
                    row.label(text="", icon=_CLASS_ICON.get(c.cls, "DOT"))
            # ALL control edits rebuild AUTOMATICALLY (debounced, threaded):
            # staged params take effect, and live params get their baked
            # derivations (e.g. blade vertices) reconciled after the drag.
            # The button stays as a manual "rebuild now".
            rebuild.check(obj, renderer)
            busy, err = rebuild.status(obj.name)
            row = layout.row(align=True)
            row.operator(SDM_OT_rebuild.bl_idname, icon="FILE_REFRESH")
            if busy:
                row.label(text="Rebuilding…", icon="SORTTIME")
            if err:
                layout.label(text=err[:64], icon="ERROR")

        # -- Animations (from .sdm metadata.animations) --
        anims = renderer.artifacts.animations
        if anims:
            box = layout.box()
            box.label(text="Animations", icon="PLAY")
            live_params = {c.param for c in controls if c.cls == "live"}
            for i, anim in enumerate(anims):
                row = box.row(align=True)
                tracks = anim.get("tracks", [])
                playable = tracks and all(t.get("param") in live_params for t in tracks)
                if playable:
                    op = row.operator(
                        SDM_OT_play_animation.bl_idname,
                        text=anim.get("name", f"anim {i}"),
                        icon="PLAY",
                    )
                    op.anim_index = i
                    # Stop = back to neutral. Every animated axis gets the
                    # pair; there is no "stopped mid-pose" state to manage.
                    row.operator(SDM_OT_stop_animation.bl_idname, text="", icon="SNAP_FACE")
                else:
                    row.label(text=anim.get("name", f"anim {i}"), icon="DECORATE_KEYFRAME")
                    row.label(text="(awaits kinematics evaluator)")

        # -- Components (SDF segmentation) --
        comps = getattr(renderer.artifacts, "components", ()) or ()
        if comps:
            box = layout.box()
            row = box.row(align=True)
            row.label(text="Components", icon="OUTLINER_OB_GROUP_INSTANCE")
            if "sdm_comp_tint" in obj:
                row.prop(obj, '["sdm_comp_tint"]', text="tint")
            col = box.column(align=True)
            for c in comps:
                col.label(text=f"{c.get('id')} · {c.get('label')}")

        # -- Cutaway --
        if "sdm_cut_on" in obj:
            box = layout.box()
            row = box.row(align=True)
            row.prop(obj, '["sdm_cut_on"]', text="Cutaway")
            # Axis as X/Y/Z segment buttons (the raw idprop is 0/1/2, which
            # read as magic numbers and drag like a slider).
            axis_row = row.row(align=True)
            cur_axis = int(obj.get("sdm_cut_axis", 1)) % 3
            for i, lbl in enumerate(("X", "Y", "Z")):
                op = axis_row.operator(
                    SDM_OT_set_cut_axis.bl_idname, text=lbl, depress=(cur_axis == i)
                )
                op.axis = i
            row.prop(obj, '["sdm_cut_flip"]', text="flip")
            if "sdm_cut_caps" in obj:
                row.prop(obj, '["sdm_cut_caps"]', text="caps")
            box.prop(obj, '["sdm_cut_offset"]', text="offset")

        # -- Display aids --
        # `sdm_pattern` shades in rest space and `sdm_fast` trades march
        # quality for frame rate. Neither has anything to do with component
        # segmentation, so neither may sit in the Components box: a part that
        # emits no components hides that box, and the toggle goes with it
        # with no way to reach it. A raster field with a single body is
        # exactly that part.
        keys = [
            (k, t)
            for k, t in (("sdm_fast", "Fast viewport"), ("sdm_pattern", "pattern"))
            if k in obj
        ]
        if keys:
            row = layout.row(align=True)
            for key, text in keys:
                row.prop(obj, f'["{key}"]', text=text)

        layout.separator()
        row = layout.row(align=True)
        row.operator(SDM_OT_rest_pose.bl_idname, icon="ARMATURE_DATA")
        row.operator(SDM_OT_reset_to_sdm.bl_idname, icon="LOOP_BACK")

        # Live-sync the ray-march while sliders are dragged.
        keys = (
            [_prop_key(c) for c in controls]
            if controls
            else [s.name for s in renderer.artifacts.uniforms]
        )
        keys += [
            "sdm_cut_on",
            "sdm_cut_axis",
            "sdm_cut_flip",
            "sdm_cut_offset",
            "sdm_cut_caps",
            "sdm_fast",
            "sdm_comp_tint",
            "sdm_pattern",
        ]
        _sync_viewport_on_change(context, obj, keys)


class SDM_OT_set_cut_axis(bpy.types.Operator):
    bl_idname = "sdm.set_cut_axis"
    bl_label = "Cutaway Axis"
    bl_description = "Cut along this axis"
    bl_options = {"INTERNAL", "UNDO"}

    # bpy.props.*Property() as an annotation is how Blender registers this
    # field (see prefs.py's SDMViewerPreferences for the full explanation).
    axis: bpy.props.IntProperty(min=0, max=2, default=1)  # type: ignore[valid-type]

    def execute(self, context):
        obj = _active_sdm_object(context)
        if obj is None:
            return {"CANCELLED"}
        obj["sdm_cut_axis"] = int(self.axis)
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


class SDM_OT_rebuild(bpy.types.Operator):
    bl_idname = "sdm.rebuild_part"
    bl_label = "Rebuild Part"
    bl_description = (
        "Write every control value into the .sdm, re-run the part's declared "
        "generator (metadata.generator) so re-emit/topology params take "
        "effect, then re-emit GLSL and swap the shader"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = _active_sdm_object(context)
        return obj is not None and get_renderer(obj.name) is not None

    def execute(self, context):
        obj = _active_sdm_object(context)
        # The rebuild pipeline (stage -> generator -> re-emit -> shader swap)
        # runs debounced + threaded in rebuild.py; this just skips the wait.
        rebuild.force(obj.name)
        self.report({"INFO"}, "Rebuild started")
        return {"FINISHED"}


def _fcurves_for(obj, data_path):
    """F-curves on ``obj`` matching ``data_path``, across Blender API eras.

    Blender <= 4.x exposes ``action.fcurves``; 5.x actions are layered +
    slotted and the legacy attribute is gone: curves live in the channelbag
    of (layer strip, action slot). Returns [] rather than raising: a missing
    CYCLES modifier degrades the loop, it must never block playback.
    """
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    action = ad.action
    if hasattr(action, "fcurves"):  # legacy API (<= 4.x)
        return [fc for fc in action.fcurves if fc.data_path == data_path]
    found = []
    slot = getattr(ad, "action_slot", None)
    try:
        for layer in action.layers:
            for strip in layer.strips:
                bag = strip.channelbag(slot) if slot is not None else None
                if bag is None:
                    continue
                found.extend(fc for fc in bag.fcurves if fc.data_path == data_path)
    except (AttributeError, TypeError, RuntimeError):
        return []
    return found


class SDM_OT_play_animation(bpy.types.Operator):
    bl_idname = "sdm.play_animation"
    bl_label = "Play SDM animation"
    bl_description = (
        "Keyframe the animation's live params over their authored range "
        "(rest -> +range -> -range -> rest, cyclic) and toggle playback"
    )
    bl_options = {"REGISTER"}

    anim_index: bpy.props.IntProperty(default=0)  # type: ignore[valid-type]

    @classmethod
    def poll(cls, context):
        obj = _active_sdm_object(context)
        return obj is not None and get_renderer(obj.name) is not None

    def execute(self, context):
        obj = _active_sdm_object(context)
        renderer = get_renderer(obj.name)
        anims = renderer.artifacts.animations
        if not (0 <= self.anim_index < len(anims)):
            return {"CANCELLED"}
        anim = anims[self.anim_index]
        by_param = {c.param: c for c in renderer.artifacts.controls}

        scene = context.scene
        fps = scene.render.fps
        # Cycle duration from the authored rate: one full sweep spans
        # 4x the half-range; fall back to 4 s if no rate given.
        rng_deg = anim.get("range_deg")
        rate = anim.get("rate_deg_per_s")
        if rng_deg and rate:  # noqa: SIM108  # kept as an explicit if/else; the packed ternary reads worse than the branch it replaces
            seconds = 2.0 * (rng_deg[1] - rng_deg[0]) / rate
        else:
            seconds = 4.0
        n_frames = max(int(seconds * fps), 8)
        q = n_frames // 4

        for t in anim.get("tracks", []):
            c = by_param.get(t.get("param"))
            if c is None or c.cls != "live":
                continue
            key = c.uniform
            lo, hi = t.get("range", [c.value, c.value])
            rest = float(t.get("rest", c.value))
            path = f'["{key}"]'
            # rest -> hi -> rest -> lo -> rest, then cycle.
            for frame, value in ((1, rest), (q, hi), (2 * q, rest), (3 * q, lo), (n_frames, rest)):
                obj[key] = float(value)
                obj.keyframe_insert(data_path=path, frame=frame)
            for fc in _fcurves_for(obj, path):
                if not any(m.type == "CYCLES" for m in fc.modifiers):
                    fc.modifiers.new("CYCLES")

        scene.frame_start = 1
        scene.frame_end = n_frames
        bpy.ops.screen.animation_play()
        return {"FINISHED"}


def _zero_pose(context, obj, renderer) -> int:
    """Stop playback, clear animation keys, restore pose DOFs to rest.

    The shared neutral-return used by Rest Pose and every animation Stop.
    Returns the number of DOFs restored.
    """
    try:
        if context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
    except (AttributeError, RuntimeError):
        pass
    # Play only ever keyframes pose params on the placeholder, so clearing
    # its animation wholesale is safe, and required: keyed values override
    # property writes.
    obj.animation_data_clear()

    n = 0
    for c in renderer.artifacts.controls:
        if (c.ui or {}).get("role") != "pose":
            continue
        obj[_prop_key(c)] = float(c.value)
        n += 1
    if n == 0:
        # Pre-role .sdm (or no pose params): fall back to zeroing every
        # radian-unit live control: the pose DOFs are the only ones.
        for c in renderer.artifacts.controls:
            if c.cls == "live" and c.unit == "rad":
                obj[_prop_key(c)] = 0.0
                n += 1

    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()
    return n


class SDM_OT_rest_pose(bpy.types.Operator):
    bl_idname = "sdm.rest_pose"
    bl_label = "Rest pose"
    bl_description = (
        "Zero the part's pose: stop playback, clear animation keys, and "
        'restore every role="pose" DOF (deflections, twists) to its '
        "authored rest. Design sliders are left untouched."
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = _active_sdm_object(context)
        return obj is not None and get_renderer(obj.name) is not None

    def execute(self, context):
        obj = _active_sdm_object(context)
        renderer = get_renderer(obj.name)
        if obj is None or renderer is None:
            return {"CANCELLED"}
        n = _zero_pose(context, obj, renderer)
        self.report({"INFO"}, f"Pose zeroed ({n} DOF)")
        return {"FINISHED"}


class SDM_OT_stop_animation(bpy.types.Operator):
    bl_idname = "sdm.stop_animation"
    bl_label = "Stop"
    bl_description = "Stop the animation and return the part to neutral"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = _active_sdm_object(context)
        return obj is not None and get_renderer(obj.name) is not None

    def execute(self, context):
        obj = _active_sdm_object(context)
        renderer = get_renderer(obj.name)
        if obj is None or renderer is None:
            return {"CANCELLED"}
        _zero_pose(context, obj, renderer)
        return {"FINISHED"}


class SDM_OT_reset_to_sdm(bpy.types.Operator):
    bl_idname = "sdm.reset_to_sdm"
    bl_label = "Reset to .sdm values"
    bl_description = (
        "Stop playback, clear any SDM animation keys, and restore every "
        "control to the value declared in the source .sdm (rest pose). "
        "Does not re-emit GLSL."
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = _active_sdm_object(context)
        return obj is not None and get_renderer(obj.name) is not None

    def execute(self, context):
        obj = _active_sdm_object(context)
        renderer = get_renderer(obj.name)
        if obj is None or renderer is None:
            return {"CANCELLED"}

        # Keyframed params override property writes (the draw handler reads
        # the EVALUATED object), so a reset that leaves the action in place
        # appears to do nothing. Stop playback and clear the placeholder's
        # animation: it carries nothing but SDM pose keys.
        try:
            if context.screen.is_animation_playing:
                bpy.ops.screen.animation_cancel(restore_frame=False)
        except (AttributeError, RuntimeError):
            pass
        obj.animation_data_clear()

        if renderer.artifacts.controls:
            for c in renderer.artifacts.controls:
                obj[_prop_key(c)] = float(c.value)
        else:
            for spec in renderer.artifacts.uniforms:
                obj[spec.name] = float(spec.initial)

        # Tag the viewport so the new uniform values are picked up immediately.
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


def register() -> None:
    bpy.utils.register_class(SDM_OT_set_cut_axis)
    bpy.utils.register_class(SDM_OT_rebuild)
    bpy.utils.register_class(SDM_OT_play_animation)
    bpy.utils.register_class(SDM_OT_stop_animation)
    bpy.utils.register_class(SDM_OT_rest_pose)
    bpy.utils.register_class(SDM_OT_reset_to_sdm)
    bpy.utils.register_class(SDM_PT_part)


def unregister() -> None:
    bpy.utils.unregister_class(SDM_PT_part)
    bpy.utils.unregister_class(SDM_OT_reset_to_sdm)
    bpy.utils.unregister_class(SDM_OT_rest_pose)
    bpy.utils.unregister_class(SDM_OT_stop_animation)
    bpy.utils.unregister_class(SDM_OT_play_animation)
    bpy.utils.unregister_class(SDM_OT_rebuild)
    bpy.utils.unregister_class(SDM_OT_set_cut_axis)
