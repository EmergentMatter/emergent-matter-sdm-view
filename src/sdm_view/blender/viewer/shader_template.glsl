// shader_template.glsl
//
// Fragment shader for the SDM viewport ray-marcher.
//
// Assembly order (see viewport.py / assemble_fragment_source):
//   1. This file up to the first marker
//   2. @@SDF_LIB@@      - inlined contents of sdf_lib.glsl from sdm-core
//   3. This file between markers
//   4. @@SDF_SCENE@@    - inlined contents of sdf_scene.glsl (declares u_p_* and sdf_scene)
//   5. This file after the second marker
//
// All SDF math runs in object-local space (the frame the .sdm is authored
// in). The placeholder Blender object's world matrix is supplied as
// u_object_inv, so moving / rotating the placeholder in the scene moves
// the part along with it.
//
// NOTE: no ``#version`` directive here. Blender's gpu module prepends its
// own GLSL preamble (including the version), so any `#version` we write
// ends up on a non-first line and the driver fails compilation with
// C0204 "version directive must be first statement". Let Blender choose.

// ---------------------------------------------------------------------------
// Per-frame uniforms supplied by the Python draw handler
// ---------------------------------------------------------------------------
uniform mat4 u_persp;        // region_data.perspective_matrix  (proj * view)
uniform mat4 u_persp_inv;    // its inverse
uniform mat4 u_object_inv;   // inverse of placeholder.matrix_world
uniform vec3 u_bbox_min;     // local-space SDF bounding box
uniform vec3 u_bbox_max;
uniform vec3 u_cut_normal;   // cutaway plane normal (object-local space)
uniform float u_cut_offset;  // plane offset along the normal (mm)
uniform float u_cut_on;      // > 0.5 enables the cutaway clip
uniform float u_cut_caps;    // > 0.5: carve the field (flat tinted section
                             // faces). <= 0.5 (default): UNCAPPED: removed
                             // material is skipped by the march, exposing
                             // the interior surfaces behind the plane.

// ---------------------------------------------------------------------------
// Inputs / outputs
// ---------------------------------------------------------------------------
in vec2 v_ndc;               // -1..1 across the screen, supplied by the vertex shader
out vec4 fragColor;

// @@SDF_LIB@@

// ---------------------------------------------------------------------------
// AABB intersection: limits march range to inside the bbox
// ---------------------------------------------------------------------------
bool ray_box_intersect(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax,
                       out float t_near, out float t_far) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (bmin - ro) * inv;
    vec3 t1 = (bmax - ro) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    t_near = max(max(tmin.x, tmin.y), tmin.z);
    t_far  = min(min(tmax.x, tmax.y), tmax.z);
    return (t_far >= max(t_near, 0.0));
}

// @@SDF_SCENE@@

// ---------------------------------------------------------------------------
// Cutaway wrapper: all marching/normals go through scene_d so the clip
// plane carves the field itself (flat cut faces get the plane's normal).
// The cut is MATERIAL-SPACE (sdf_scene_rcut): each component intersects the
// plane at its rest point, so the slice is taken once in material
// coordinates and deforms/rotates with the part during animation instead of
// re-clipping the deformed geometry against a static plane every frame.
// ---------------------------------------------------------------------------
float scene_d(vec3 p) {
    if (u_cut_on > 0.5 && u_cut_caps > 0.5) {
        return sdf_scene_rcut(p, u_cut_normal, u_cut_offset);
    }
    // Uncapped cut: march/normals use the RAW field; the march loop skips
    // removed-side hits instead (sdm_removed_side).
    return sdf_scene(p);
}

// Is p removed material? Material-space plane test of the component that
// owns the nearest surface: evaluated at surface encounters only, never
// per step.
bool sdm_removed_side(vec3 p) {
    int cid = sdf_scene_comp(p);
    vec3 q = sdm_rest_point(p, cid);
    return dot(q, u_cut_normal) - u_cut_offset > 0.0;
}

// ---------------------------------------------------------------------------
// Surface normal via central differences on scene_d (cutaway-aware: flat
// cut faces get the clip plane's normal)
// ---------------------------------------------------------------------------
vec3 estimate_normal(vec3 p) {
    const float h = 5e-4;  // local-space units (mm in the canonical .sdm)
    vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * scene_d(p + k.xyy * h) +
        k.yyx * scene_d(p + k.yyx * h) +
        k.yxy * scene_d(p + k.yxy * h) +
        k.xxx * scene_d(p + k.xxx * h)
    );
}

// ---------------------------------------------------------------------------
// Sphere-tracing loop
// ---------------------------------------------------------------------------
const int   MAX_STEPS  = 192;
const float HIT_EPS    = 5e-4;
const float STEP_FLOOR = 5e-4;  // avoid stalls near sharp CSG seams

// Fast-viewport mode (u_fast > 0.5, panel "Fast viewport" toggle, default
// on): half the step budget and a relaxed hit epsilon. Sphere tracing's
// cost tail is the slow creep at grazing incidence: a looser epsilon cuts
// it disproportionately. Turn it off for silhouette-accurate stills.
uniform float u_fast;

// Half-resolution path (u_pack_depth > 0.5): the draw handler renders this
// shader into a half-size offscreen and composites it up: 4x fewer marches
// per redraw, which is the actual responsiveness lever on Retina (the march
// runs on the UI thread every viewport redraw). Depth can't reach the
// offscreen's depth buffer through Blender's GPUOffScreen API, so it rides
// in ALPHA (RGBA16F): a < 0 means "no hit" (cleared sentinel), a >= 0 is
// window-space depth for the composite pass to write to gl_FragDepth.
uniform float u_pack_depth;

// Component tint toggle (panel "Color components"); sdf_scene_comp comes
// from the emitted scene (the assembler injects a return-0 stub for
// artifacts emitted before segmentation existed).
uniform float u_comp_tint;

// Rest-space checker toggle (panel "pattern"): a subtle checkerboard
// evaluated at the hit's MATERIAL coordinates (sdm_rest_point), so surfaces
// visibly carry rotation/twist during animation: a monochrome cylinder
// spinning about its own axis is otherwise indistinguishable from standing
// still.
uniform float u_pattern;

bool march(vec3 ro, vec3 rd, float t_near, float t_far, out vec3 p_hit, out float t_hit) {
    float t = max(t_near, 0.0);
    int   cap = u_fast > 0.5 ? MAX_STEPS / 2 : MAX_STEPS;
    float eps = u_fast > 0.5 ? 4.0 * HIT_EPS : HIT_EPS;
    bool clip = u_cut_on > 0.5 && u_cut_caps <= 0.5;   // uncapped cut
    bool skipping = false;   // traversing removed-side solid interior
    for (int i = 0; i < MAX_STEPS; ++i) {
        if (i >= cap) break;
        vec3 p = ro + rd * t;
        float d = scene_d(p);
        if (!skipping && d < eps) {
            if (!clip || !sdm_removed_side(p)) {
                p_hit = p;
                t_hit = t;
                return true;
            }
            skipping = true;   // removed material: march THROUGH it
        }
        if (skipping && d >= eps) {
            skipping = false;  // exited the removed solid; resume tracing
        }
        t += skipping ? max(abs(d), STEP_FLOOR) : max(d, STEP_FLOOR);
        if (t > t_far) break;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
void main() {
    // World-space ray through this pixel (unproject NDC at near & far planes).
    // The origin is the unprojected NEAR-PLANE point, NOT the eye position:
    // in perspective the near point lies on the eye ray anyway, and in
    // orthographic views (axis gizmo / numpad snaps) rays are parallel with
    // per-pixel origins: a shared eye origin renders garbage there.
    vec4 near_h = u_persp_inv * vec4(v_ndc, -1.0, 1.0);
    vec4 far_h  = u_persp_inv * vec4(v_ndc,  1.0, 1.0);
    vec3 near_w = near_h.xyz / near_h.w;
    vec3 far_w  = far_h.xyz  / far_h.w;
    vec3 ro_w   = near_w;
    vec3 rd_w   = normalize(far_w - near_w);

    // Transform ray into object-local space; SDF is authored there.
    mat3 obj_inv3 = mat3(u_object_inv);
    vec3 ro_local = (u_object_inv * vec4(ro_w, 1.0)).xyz;
    vec3 rd_local = normalize(obj_inv3 * rd_w);

    // Cull rays that miss the bbox entirely. Saves >90% of pixels on a
    // typical viewport that has the part in a corner of the frame.
    float t_near, t_far;
    if (!ray_box_intersect(ro_local, rd_local, u_bbox_min, u_bbox_max, t_near, t_far)) {
        discard;
    }

    // Camera inside the SOLID (zoomed through a wall): the march would "hit"
    // at t=0 for every pixel and the viewport becomes one undifferentiated
    // grey fill. Instead, march THROUGH the solid to its exit surface and
    // shade that as a warm cutaway: the interior stays readable and you can
    // see your way back out.
    bool inside_start = scene_d(ro_local + rd_local * max(t_near, 0.0)) < 0.0;
    if (inside_start) {
        float t = max(t_near, 0.0);
        for (int i = 0; i < MAX_STEPS; ++i) {
            vec3 p = ro_local + rd_local * t;
            float d = scene_d(p);
            if (d >= -HIT_EPS) break;              // reached the exit surface
            t += max(-d, STEP_FLOOR);
            if (t > t_far) break;                  // solid extends past bbox
        }
        vec3 p_exit = ro_local + rd_local * t;
        vec3 n_in = -estimate_normal(p_exit);       // face the camera
        mat3 xf = transpose(mat3(u_object_inv));
        vec3 n_w = normalize(xf * n_in);
        float lam = max(dot(n_w, normalize(vec3(0.5, 0.7, 0.8))), 0.0);
        vec3 cut = vec3(0.55, 0.30, 0.24) * (0.35 + 0.65 * lam);

        vec3 pw = (inverse(u_object_inv) * vec4(p_exit, 1.0)).xyz;
        vec4 cl = u_persp * vec4(pw, 1.0);
        float depth01_cut = 0.5 + 0.5 * (cl.z / cl.w);
        gl_FragDepth = depth01_cut;   // harmless in the offscreen pass
        fragColor = u_pack_depth > 0.5 ? vec4(cut, depth01_cut)
                                       : vec4(cut, 1.0);
        return;
    }

    vec3 p_local;
    float t_hit;
    if (!march(ro_local, rd_local, t_near, t_far, p_local, t_hit)) {
        discard;
    }

    // Normal in local space; transform to world via inverse-transpose
    // (so non-uniform placeholder scale still produces correct lighting).
    vec3 n_local = estimate_normal(p_local);
    mat3 normal_xform = transpose(obj_inv3);
    vec3 n_world = normalize(normal_xform * n_local);

    // Simple two-light Lambert with a fill term: readable on any background.
    vec3 light_key  = normalize(vec3( 0.5,  0.7,  0.8));
    vec3 light_fill = normalize(vec3(-0.4, -0.3,  0.6));
    float lambert_key  = max(dot(n_world, light_key),  0.0);
    float lambert_fill = max(dot(n_world, light_fill), 0.0);
    vec3 base = vec3(0.78, 0.80, 0.84);

    // Component segmentation tint (u_comp_tint > 0.5): the emitted
    // sdf_scene_comp reports which root-union child owns this surface;
    // golden-ratio hue spacing gives every component a stable, distinct
    // color with zero palette uniforms. One extra scene eval, hit pixels
    // only. The id also selects the rest-point chain for the pattern.
    bool b_caps = u_cut_on > 0.5 && u_cut_caps > 0.5;
    int cid = 0;
    if (u_comp_tint > 0.5 || u_pattern > 0.5 || b_caps) {
        cid = sdf_scene_comp(p_local);
    }
    if (u_comp_tint > 0.5) {
        float hue = fract(0.61803398875 * float(cid));
        vec3 hk = clamp(abs(mod(hue * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0)
                        - 1.0, 0.0, 1.0);
        vec3 tint = mix(vec3(1.0), hk, 0.55);
        base = mix(base, tint, 0.55);
    }

    // Rest-space checker: material coordinates, so the cells ride WITH the
    // twist/rotation. Cell size tracks the part's largest span.
    if (u_pattern > 0.5) {
        vec3 q = sdm_rest_point(p_local, cid);
        vec3 span = u_bbox_max - u_bbox_min;
        float cell = max(span.x, max(span.y, span.z)) / 12.0;
        vec3 idx = floor(q / cell);
        float parity = mod(idx.x + idx.y + idx.z, 2.0);
        base *= mix(0.78, 1.0, parity);
    }

    // Capped mode: section faces (hits ON the material-space plane) read as
    // sectioned material: a binary solid/void indicator, not fake surface.
    if (b_caps) {
        float d_plane = abs(dot(sdm_rest_point(p_local, cid), u_cut_normal)
                            - u_cut_offset);
        if (d_plane < 8.0 * HIT_EPS) {
            base = vec3(0.62, 0.36, 0.28);
        }
    }
    vec3 color = base * (0.18 + 0.72 * lambert_key + 0.18 * lambert_fill);

    // Map the local-space hit back to world for correct depth compositing
    // with the rest of the Blender viewport.
    vec3 p_world = (inverse(u_object_inv) * vec4(p_local, 1.0)).xyz;
    vec4 clip = u_persp * vec4(p_world, 1.0);
    float depth01 = 0.5 + 0.5 * (clip.z / clip.w);
    gl_FragDepth = depth01;           // harmless in the offscreen pass
    fragColor = u_pack_depth > 0.5 ? vec4(color, depth01)
                                   : vec4(color, 1.0);
}
