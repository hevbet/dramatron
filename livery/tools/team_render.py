#!/usr/bin/env python3
"""Studio «launch key visual» renderer: two smp_audi_rs3_lms (Audi RS3 LMS TCR) cars together.

Usage:
    python3 team_render.py --skins A_dir B_dir --out DIR
                           [--views front34_pair,rear34_pair,side_pair,top_pair]
                           [--width 2400 --height 1350] [--ss 2] [--jobs 4] [--title]
                           [--gap M --stagger M] [--dof F] [--back-ev EV] [--exposure E] [--zoom Z]
                           [--camera az,el,dist[,tx,ty,tz]] [--eye x,y,z --target x,y,z]
                           [--kn5 path] [--no-interior] [--no-ks-detail] [--fonts DIR] [--cache DIR] [--fast]

Each skin folder is applied to its own instance of the car with every texture override found in
it (Skin.dds, glass_sticker.dds, caliper.dds, rim_d.dds, tyre_d.dds, ...), exactly like
render_rs3.py / Assetto Corsa: a file named like a texture of the .kn5 replaces that texture.
The first skin is car A, the second car B.  Driver names and numbers for the title band and the
contact sheet come from each folder's ui_skin.json.

Views (each one has its own car layout, camera and look; the hero car is the one nearest the camera):
  front34_pair  eye-level camera (1.7 m), ~50 mm lens: car A hero on the right third, car B behind
                it on the left (turned a little so its door number shows); the cars fill ~60 % of the
                frame height with ~10 % headroom
  rear34_pair   the same from the rear, car B hero
  side_pair     true profile at bumper height (0.62 m), long lens (~115 mm): car A in front, car B
                behind it and half a car ahead (its door number clear of car A)
  top_pair      plan view straight down (90 degrees), the cars parallel in echelon
  custom        --camera az,el,dist[,target] (az from the car front towards the car left) or --eye/--target
--gap/--stagger replace the preset layouts with two parallel cars (gap centre to centre, stagger front to back).

Scene: glossy black studio floor (planar mirror reflection weighted by Fresnel, sharp at the contact
and blurred with the height of the reflected point, faded towards the camera) with a soft light pool
under the cars, contact shadows and an occlusion footprint (height field of the cars), in a near-black
cyclorama with a maroon-pink back glow, a dashed light line on the back wall (the butcher-chart motif),
a touch of coloured haze and a vignette.  Light rig (procedural environment rotated with the camera
like a photographer's rig; every light a hard-edged rectangle with a gradient): a long overhead
softbox, two long horizontal light-line strips behind the cars (the crisp streaks along hood, roof and
trunk), a side strip on each side (one clean streak along the shoulder line), a front three-quarter
key, cyan and magenta rim strips low behind the cars (rim light on roof line, wing and tyres), a
ceiling cove (gradient reflections on glass and the horizontal panels), a dim backdrop panel and a
front fill; the top view adds floor-standing reflector panels for the glass.  Diffuse light: order-2
spherical harmonics x per-vertex directional occlusion (128 shadow maps), the overhead part with a
near-field fall-off.  Reflections are per pixel and analytic, with specular occlusion interpolated
between the visibility directions.  Paint: pearl pink (a slight hue flop, neutral sheen, so the pink
stays the same in every view) under a Fresnel clearcoat; the holographic regions of the livery (found
in the texture) keep their designed rainbow and shift it a little with the view angle, with a
soft-clamped tinted film reflection.  Painted rims get extra fill so white rims read white.  DRL,
projector and tail lamps are emissive and bloom.  The car further from the camera is exposed --back-ev
lower and slightly blurred by the depth of field (--dof).  Textures: anisotropic filtering (up to 8
taps along the footprint) over mip pyramids padded outside the UV islands; alpha-tested decals test
the filtered alpha.  Smoothed shading normals on the body.  Post: SSAA (--ss) with a triangle-filter
float downsample (no ringing), bloom, ACES tone mapping, dithering before quantising.
--title adds the team band at the bottom (overlaid on the floor).  --fast skips the visibility
precomputation (previews).  sheet.png: the views without the band, cropped to the cars.

Cost: per layout the visibility (128 shadow maps) takes ~0.5-1 min on 4 CPUs (--cache keeps it between
runs); the four 2400x1350 views at --ss 2 in parallel (--jobs 4) take ~6-7 min, peak ~2.5 GB per worker
(~3.5 GB with depth of field).

World space = AC/kn5: x = car LEFT, y = up, z = FRONT.  Writes <view>.png and sheet.png.
"""
import argparse
import json
import math
import os
import re
import sys
import time
import zlib
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_rs3 as rr  # noqa: E402
from kn5 import R  # noqa: E402

LIV = os.path.dirname(HERE)
FONT_DIRS = [os.path.join(LIV, "rs3_concepts", "fonts"), os.path.join(LIV, "fonts"),
             os.path.join(LIV, "fonts", "extra"), os.path.join(LIV, "rs3_pink", "fonts"),
             "/usr/share/fonts/truetype/dejavu"]
TEAM_LINE = ("КОМАНДА ЭДМ", "BUTCHER CHART CHROME")

ALL_VIEWS = ["front34_pair", "rear34_pair", "side_pair", "top_pair"]
CAPTIONS = {"front34_pair": "Три четверти спереди", "rear34_pair": "Три четверти сзади",
            "side_pair": "Профиль", "top_pair": "Вид сверху", "custom": "Свой ракурс"}


def _behind(az_deg, left, deeper):
    """Offset (x, z) of a second car seen `left` m to the left of the hero on screen and `deeper` m further
    from a camera at azimuth az_deg."""
    a = math.radians(az_deg)
    return (round(-left * math.cos(a) - deeper * math.sin(a), 3), round(left * math.sin(a) - deeper * math.cos(a), 3))


_F34 = _behind(34.0, 2.5, 3.6)
_R34 = _behind(214.0, 2.5, 3.6)
# cars: (x, z, yaw degrees; + turns the nose towards +x) for car A and car B.
# camera: az degrees from the car front (+z) towards the car left (+x); eye_h camera height (m) or el
# (degrees above the target); dist horizontal distance (m) from the layout centre (long lens when large).
# margins: framing (left, right, top, bottom) fractions of the frame (the bottom one leaves room for the
# floor reflection and the title band).  dof: depth of field strength; back_ev: exposure of the car further
# away; floor_alb: floor diffuse albedo in the light pool; refl: floor reflection weight (x Fresnel);
# line_h: height (m) of the dashed light line on the back wall (None: no line)
PRESETS = {
    "front34_pair": dict(az=34.0, eye_h=1.7, dist=10.0, ty=0.6, cars=((0.0, 0.0, 0.0), (_F34[0], _F34[1], -8.0)),
                         margins=(0.04, 0.04, 0.10, 0.30), dof=0.30, back_ev=-0.3, exposure=1.0, floor_alb=0.16,
                         refl=0.55),
    "rear34_pair": dict(az=214.0, eye_h=1.7, dist=10.0, ty=0.6, cars=((_R34[0], _R34[1], -8.0), (0.0, 0.0, 0.0)),
                        margins=(0.04, 0.04, 0.10, 0.30), dof=0.30, back_ev=-0.3, exposure=1.0, floor_alb=0.16,
                        refl=0.55),
    "side_pair": dict(az=90.0, eye_h=0.62, dist=26.0, ty=0.6, cars=((1.6, -1.3, 0.0), (-1.6, 1.3, 0.0)),
                      margins=(0.04, 0.04, 0.22, 0.36), dof=0.0, back_ev=-0.2, exposure=1.0, floor_alb=0.16,
                      refl=0.40, line_h=0.95),
    "top_pair": dict(az=-90.0, el=90.0, dist=15.0, ty=0.3, cars=((1.175, 0.7, 0.0), (-1.175, -0.7, 0.0)),
                     margins=(0.08, 0.08, 0.06, 0.13), dof=0.0, back_ev=0.0, exposure=1.0, floor_alb=0.14,
                     refl=0.30),
}
CUSTOM = dict(ty=0.62, cars=((1.225, 0.4, 0.0), (-1.225, -0.4, 0.0)), margins=(0.06, 0.06, 0.12, 0.27), dof=0.0,
              back_ev=0.0, exposure=1.0, floor_alb=0.16, refl=0.5)

# ---------------------------------------------------------------- materials
# kd diffuse weight | cc clearcoat weight (Schlick F0 0.04, lobe cc_s rad) | sp dielectric spec weight
# (Schlick F0 0.04, lobe sp_s rad, tinted by the base colour by sp_tint) | mt metallic weight
# (F0 = base colour lifted by mt_lift, lobe mt_s rad) | fill: extra diffuse light (wheels deep in the arches
# get a photographer's fill) | occ_min: floor of the diffuse occlusion
MAT_DEFAULT = dict(kd=1.0, cc=0.0, cc_s=0.02, sp=0.0, sp_s=0.3, sp_tint=0.0, mt=0.0, mt_s=0.2, mt_lift=0.0,
                   fill=1.0, occ_min=0.0)
MATS = {
    "skin": dict(cc=1.0, cc_s=0.010, sp=0.40, sp_s=0.22, sp_tint=0.5),       # pearl base + clearcoat
    "glass_sticker": dict(cc=0.15, cc_s=0.04),
    "ext_sticker": dict(cc=0.45, cc_s=0.03),
    "caliper": dict(cc=1.0, cc_s=0.03, fill=1.4, occ_min=0.3),
    "rim": dict(kd=1.0, mt=0.04, mt_s=0.16, mt_lift=0.05, cc=0.6, cc_s=0.03, fill=1.9, occ_min=0.5),  # painted rims
    "ture": dict(sp=0.6, sp_s=0.40),
    "brakedisc": dict(kd=0.5, mt=0.6, mt_s=0.30, mt_lift=0.3),
    "chrome": dict(kd=0.05, mt=1.0, mt_s=0.012, mt_lift=0.55),
    "mirror": dict(kd=0.02, mt=1.0, mt_s=0.003, mt_lift=0.7),
    "lights": dict(kd=0.5, mt=0.65, mt_s=0.06, mt_lift=0.35, cc=1.0, cc_s=0.004),
    "reflector": dict(kd=0.5, mt=0.65, mt_s=0.06, mt_lift=0.35, cc=1.0, cc_s=0.004),
    "exhaust": dict(kd=0.4, mt=0.7, mt_s=0.2, mt_lift=0.3),
    "ext_carbon": dict(cc=0.55, cc_s=0.03),
    "glassblack": dict(cc=1.0, cc_s=0.006),
    "ext_plastic": dict(sp=0.8, sp_s=0.20),
    "pl_black": dict(sp=0.8, sp_s=0.22),
    "black": dict(sp=0.5, sp_s=0.30),
    "reshotka": dict(sp=0.6, sp_s=0.30),
    "ext_metal": dict(kd=0.7, mt=0.3, mt_s=0.28, mt_lift=0.1),
    "screw": dict(kd=0.5, mt=0.6, mt_s=0.2, mt_lift=0.3),
    "screw.001": dict(kd=0.5, mt=0.6, mt_s=0.2, mt_lift=0.3),
    "ext_rubber": dict(sp=0.4, sp_s=0.45),
    "__interior__": dict(sp=0.3, sp_s=0.4),
}
# glass: transmittance (linear rgb) and Fresnel F0.  Window glass is dark tinted, lamp lenses clear.
GLASS_T = {"window": (0.16, 0.17, 0.19), "lens": (0.80, 0.80, 0.82), "lights_glass": (0.62, 0.07, 0.08)}
GLASS_F0 = 0.045
DIFFUSE_GAIN = 0.62
OCC_RAD = np.array([0.012, 0.009, 0.010], np.float32)   # what an occluded reflection sees (car body, dark)
# emissive lamps by kn5 node name (linear radiance): daytime running lights, projectors, tail and brake lamps
EMIT = {"LIGHT_DRL": (9.0, 9.3, 10.5), "LIGHT_FRONT.001": (1.4, 1.35, 1.25), "LIGHT_FRONT": (1.0, 0.97, 0.9),
        "LIGHT_TAIL": (9.0, 0.70, 0.50), "LIGHT_BRAKE": (4.5, 0.30, 0.22)}
PEARL_HUE = 338.0                       # hue of the pearl pink (degrees) for the flop mask
PEARL_FLOP_DEG = -7.0                   # pearl hue shift at grazing angles (small: the pink must match across views)
PEARL_SHEEN = np.array([1.0, 0.93, 0.96], np.float32)    # neutral-warm pearl sheen
NEAR_TOP_H = 4.0                        # height of the overhead softbox for its near-field fall-off (m)
MAX_ANISO = 8
LOD_BIAS = 0.0


def srgb_to_lin(c):
    return np.power(np.clip(np.asarray(c, np.float32) / 255.0, 0, 1), 2.2)


def schlick(cos, f0):
    m = np.clip(1.0 - cos, 0.0, 1.0)
    m2 = m * m
    return f0 + (1.0 - f0) * (m2 * m2 * m)


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def near_field(y):
    """Irradiance fall-off of the overhead softbox with the height of the lit point (1 at y = 1 m)."""
    return np.clip(((NEAR_TOP_H - 1.0) / np.maximum(NEAR_TOP_H - y, 0.5)) ** 2, 0.3, 1.6).astype(np.float32)


# ---------------------------------------------------------------- kn5 with transforms and normals
def load_geometry(path, interior=True):
    """render_rs3.load_scene (same node walk, transforms and skip rules) plus world vertex normals."""
    r = R(open(path, "rb").read())
    assert r.raw(6) == b"sc6969"
    ver = r.i()
    if ver > 5:
        r.i()
    textures = {}
    for _ in range(r.i()):
        r.i(); name = r.s(); size = r.i(); textures[name] = r.raw(size)
    mats = []
    for _ in range(r.i()):
        name = r.s(); shader = r.s(); r.u8(); r.u8()
        if ver > 4:
            r.i()
        props = {}
        for _ in range(r.i()):
            pn = r.s(); props[pn] = r.f(); r.raw(36)
        samplers = {}
        for _ in range(r.i()):
            sn = r.s(); r.i(); samplers[sn] = r.s()
        mats.append(dict(name=name, shader=shader, samplers=samplers, props=props))
    meshes = []

    def node(M, path_, skip):
        cls = r.i(); name = r.s(); nch = r.i(); r.u8()
        full = path_ + "/" + name
        up = name.upper()
        if any(k in up for k in ("COCKPIT_HR", "FLYCAM", "DAMAGE")) or (not interior and "COCKPIT_LR" in up):
            skip = True
        if cls == 1:
            L = np.frombuffer(r.raw(64), dtype="<f4").reshape(4, 4).astype(np.float64)
            M = L @ M
        elif cls in (2, 3):
            r.u8(); r.u8(); r.u8()
            if cls == 3:
                for _ in range(r.i()):
                    r.s(); r.raw(64)
                stride = 19
            else:
                stride = 11
            nv = r.i(); v = np.frombuffer(r.raw(nv * stride * 4), dtype="<f4").reshape(nv, stride)
            ni = r.i(); idx = np.frombuffer(r.raw(ni * 2), dtype="<u2").reshape(-1, 3)
            mid = r.i(); r.i()
            if cls == 2:
                r.f(2); r.f(3); r.f(); r.u8()
            else:
                r.raw(8)
            if not skip:
                A = M[:3, :3]
                pw = v[:, 0:3].astype(np.float64) @ A + M[3, :3]
                n = v[:, 3:6].astype(np.float64)
                if abs(np.linalg.det(A)) > 1e-9:
                    n = n @ np.linalg.inv(A).T
                meshes.append(dict(name=full, node=name, pos=pw, nrm=n, uv=v[:, 6:8].astype(np.float64),
                                   idx=idx.astype(np.int64), mat=mats[mid]["name"],
                                   interior="COCKPIT" in full.upper()))
        for _ in range(nch):
            node(M, full, skip)

    node(np.eye(4), "", False)
    return textures, mats, meshes


def smooth_normals(P, VN, verts, radius=0.03, sigma=0.015, min_dot=0.985):
    """Spatially smoothed shading normals for the body panels: every vertex averages the normals of the
    vertices within `radius` (gaussian weights) that point within ~10 degrees of its own, so small mesh
    ripples and seam mismatches do not wobble the thin softbox streaks while real creases stay crisp."""
    from scipy.spatial import cKDTree
    pts = P[verts]; nrm = VN[verts].astype(np.float64)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return VN
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(pts[i] - pts[j], axis=1)
    dots = (nrm[i] * nrm[j]).sum(1)
    w = np.exp(-0.5 * (d / sigma) ** 2) * np.clip((dots - min_dot) / (1 - min_dot), 0, 1)
    acc = nrm.copy()
    np.add.at(acc, i, nrm[j] * w[:, None])
    np.add.at(acc, j, nrm[i] * w[:, None])
    out = VN.copy()
    out[verts] = normalize(acc)
    return out


class Model:
    """Car geometry, shared by every instance (same filtering as render_rs3.Scene)."""

    def __init__(self, kn5_path, interior=True):
        textures, mats, meshes = load_geometry(kn5_path, interior)
        self.path, self.interior = os.path.abspath(kn5_path), interior
        self.textures = textures
        self.mats = {m["name"]: m for m in mats}
        self.mat_names, mat_id = [], {}
        emit_names = list(EMIT)
        P, N, UV, T, TM, EM = [], [], [], [], [], []
        off = 0
        for m in meshes:
            mn = m["mat"]
            if mn in rr.SKIP_MATS or not rr.in_car_bounds(m["pos"]):
                continue
            key = "__interior__" if (m["interior"] and mn not in rr.GLASS_MATS) else mn
            if key not in mat_id:
                mat_id[key] = len(self.mat_names); self.mat_names.append(key)
            P.append(m["pos"]); N.append(m["nrm"]); UV.append(m["uv"]); T.append(m["idx"] + off)
            TM.append(np.full(len(m["idx"]), mat_id[key], np.int32))
            nn = m["node"].upper()
            EM.append(np.full(len(m["idx"]), emit_names.index(nn) + 1 if nn in emit_names else 0, np.int8))
            off += len(m["pos"])
        self.P = np.concatenate(P); self.UV = np.concatenate(UV)
        self.T = np.concatenate(T); self.TM = np.concatenate(TM); self.EMIS = np.concatenate(EM)
        self.emit_rgb = np.array([(0, 0, 0)] + [EMIT[k] for k in emit_names], np.float32)
        a, b, c = self.P[self.T[:, 0]], self.P[self.T[:, 1]], self.P[self.T[:, 2]]
        fn = np.cross(b - a, c - a)
        ln = np.linalg.norm(fn, axis=1, keepdims=True)
        self.FN = (fn / np.maximum(ln, 1e-12)).astype(np.float32)
        self.tri_ok = ln[:, 0] > 1e-12
        # vertex normals: from the kn5, falling back to area-weighted face normals where degenerate
        vn = np.concatenate(N)
        ok = np.linalg.norm(vn, axis=1) > 0.3
        if not ok.all():
            acc = np.zeros_like(self.P)
            for k in range(3):
                np.add.at(acc, self.T[:, k], fn)
            vn[~ok] = acc[~ok]
        vn = normalize(vn)
        if "skin" in self.mat_names:
            vn = smooth_normals(self.P, vn, np.unique(self.T[self.TM == self.mat_names.index("skin")].ravel()))
        self.VN = vn.astype(np.float32)
        uva = self.UV[self.T[:, 1]] - self.UV[self.T[:, 0]]; uvb = self.UV[self.T[:, 2]] - self.UV[self.T[:, 0]]
        self.uv_area = np.abs(uva[:, 0] * uvb[:, 1] - uva[:, 1] * uvb[:, 0]).astype(np.float32)
        tyre = self.TM == self.mat_names.index("ture") if "ture" in self.mat_names else np.zeros(len(self.T), bool)
        self.floor_y = float(self.P[self.T[tyre]].reshape(-1, 3)[:, 1].min()) if tyre.any() else float(self.P[:, 1].min())
        self.is_glass = np.array([n in rr.GLASS_MATS for n in self.mat_names])
        self.skin_id = self.mat_names.index("skin") if "skin" in self.mat_names else -1
        # window glass vs. headlamp lenses (both use the material «glass»)
        cen = (a + b + c) / 3
        self.lens = (cen[:, 1] < 0.80) & (cen[:, 2] > 1.4)
        ext = ~np.array([n == "__interior__" for n in self.mat_names])[self.TM]
        used = np.unique(self.T[ext & self.tri_ok].ravel())
        self.ext_vertices = used
        # material parameter tables
        prm = {k: [] for k in MAT_DEFAULT}
        for n in self.mat_names:
            d = dict(MAT_DEFAULT); d.update(MATS.get(n, {}))
            for k in prm:
                prm[k].append(d[k])
        self.prm = {k: np.array(v, np.float32) for k, v in prm.items()}
        self._cov = {}

    def uv_coverage(self, mat_name, size):
        """Texels covered by the UV islands of a material (wrapped UVs), dilated by one texel."""
        key = (mat_name, size)
        if key not in self._cov:
            mid = self.mat_names.index(mat_name)
            uv = self.UV[self.T[self.TM == mid]]
            uv = uv - np.floor(uv.mean(1, keepdims=True))
            im = Image.new("L", (size, size), 0)
            d = ImageDraw.Draw(im)
            for tri in (uv * size).tolist():
                d.polygon([tuple(p) for p in tri], fill=255, outline=255)
            cov = ndimage.binary_dilation(np.asarray(im) > 0, iterations=1)
            self._cov[key] = cov
        return self._cov[key]


# ---------------------------------------------------------------- textures
def build_mips(tex, at=False, min_size=8, cov=None):
    """Mip pyramid (uint8 levels).  at: alpha-weighted colour (alpha-tested decals).  cov: texel coverage
    of the UV islands; the colours outside the islands are ignored and filled from the islands
    (push-pull), so lower mips never bleed the empty texture background into the island edges."""
    if cov is not None:
        H, W, C = tex.shape
        w = cov.astype(np.float32)[..., None]
        pre = [tex.astype(np.float32) * w]; wts = [w]
        while min(pre[-1].shape[:2]) > 1:
            p, q = pre[-1], wts[-1]
            h2, w2 = p.shape[0] // 2, p.shape[1] // 2
            pre.append(p[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, C).mean((1, 3)))
            wts.append(q[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, 1).mean((1, 3)))
        filled = [None] * len(pre)
        k = len(pre) - 1
        filled[k] = np.where(wts[k] > 1e-6, pre[k] / np.maximum(wts[k], 1e-6), tex.reshape(-1, C).mean(0))
        for k in range(len(pre) - 2, -1, -1):
            up = np.repeat(np.repeat(filled[k + 1], 2, 0), 2, 1)
            hh, ww = pre[k].shape[:2]
            if up.shape[0] < hh or up.shape[1] < ww:
                up = np.pad(up, ((0, hh - up.shape[0]), (0, ww - up.shape[1]), (0, 0)), mode="edge")
            up = up[:hh, :ww]
            col = pre[k] / np.maximum(wts[k], 1e-6)
            filled[k] = np.where(wts[k] > 1e-6, col, up)
            pre[k + 1] = None
        levels = []
        for f in filled:
            if levels and min(f.shape[:2]) < min_size:
                break
            levels.append(np.clip(f + 0.5, 0, 255).astype(np.uint8))
        return levels
    levels = [tex]
    cur = tex.astype(np.float32)
    while min(cur.shape[:2]) > min_size:
        h2, w2 = cur.shape[0] // 2, cur.shape[1] // 2
        c = cur[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, -1)
        if at and cur.shape[2] == 4:
            a = c[..., 3:4]
            sa = a.sum((1, 3))
            rgb = np.where(sa > 0, (c[..., :3] * a).sum((1, 3)) / np.maximum(sa, 1e-6), c[..., :3].mean((1, 3)))
            nxt = np.concatenate([rgb, sa / 4], -1)
        else:
            nxt = c.mean((1, 3))
        levels.append(np.clip(nxt + 0.5, 0, 255).astype(np.uint8))
        cur = nxt
    return levels


def sample_mip(levels, u, v, lod):
    """Trilinear sample; lod per sample (level 0 = full size). Returns float32 (N, C) in 0..255."""
    nl = len(levels)
    lod = np.clip(lod, 0, nl - 1)
    l0 = np.floor(lod).astype(np.int32)
    fr = (lod - l0).astype(np.float32)[:, None]
    out = np.empty((len(u), levels[0].shape[2]), np.float32)
    for lv in np.unique(l0):
        jj = np.nonzero(l0 == lv)[0]
        a = rr.sample(levels[lv], u[jj], v[jj])
        if lv + 1 < nl:
            b = rr.sample(levels[lv + 1], u[jj], v[jj])
            out[jj] = a + (b - a) * fr[jj]
        else:
            out[jj] = a
    return out


def footprint(J, W0, H0):
    """Texel footprint of one pixel from the per-triangle screen->uv Jacobian J (n, 4: du/dx, du/dy,
    dv/dx, dv/dy) for a W0 x H0 texture: (major, minor) axis lengths in texels and the major axis
    direction (unit vector in texel space)."""
    a = J[:, 0] * W0; b = J[:, 1] * W0; c = J[:, 2] * H0; d = J[:, 3] * H0
    p = a * a + b * b; q = c * c + d * d; r = a * c + b * d
    S = p + q; D = np.sqrt(np.maximum((p - q) ** 2 + 4 * r * r, 0))
    s1 = np.sqrt(np.maximum((S + D) / 2, 0)); s2 = np.sqrt(np.maximum((S - D) / 2, 0))
    lam = (S + D) / 2
    # eigenvector of M M^T for the larger eigenvalue
    ex = np.where(np.abs(r) > 1e-12, r, np.where(p >= q, 1.0, 0.0))
    ey = np.where(np.abs(r) > 1e-12, lam - p, np.where(p >= q, 0.0, 1.0))
    n = np.maximum(np.hypot(ex, ey), 1e-12)
    return s1.astype(np.float32), s2.astype(np.float32), (ex / n).astype(np.float32), (ey / n).astype(np.float32)


def sample_aniso(levels, u, v, J, bias=0.0, max_aniso=MAX_ANISO):
    """Anisotropic filtering: up to max_aniso trilinear taps spread along the major axis of the pixel's
    texel footprint, at the mip level of footprint / taps (sharp along the minor axis, no aliasing along
    the major one: thin keylines and decals at grazing angles neither sparkle nor smear)."""
    H0, W0 = levels[0].shape[:2]
    s1, s2, ex, ey = footprint(J, W0, H0)
    N = np.clip(np.ceil(s1 / np.maximum(s2, 1e-3)), 1, max_aniso).astype(np.int32)
    N = np.where(s1 <= 1.0, 1, N)
    lod = np.log2(np.maximum(s1 / N, 1e-6)) + bias
    lod = np.where(np.isfinite(lod), lod, 0.0)
    out = np.zeros((len(u), levels[0].shape[2]), np.float32)
    for n in np.unique(N):
        jj = np.nonzero(N == n)[0]
        if n == 1:
            out[jj] = sample_mip(levels, u[jj], v[jj], lod[jj])
            continue
        acc = np.zeros((len(jj), levels[0].shape[2]), np.float32)
        step = s1[jj] / n
        for k in range(n):
            t = ((k + 0.5) - n / 2.0) * step
            acc += sample_mip(levels, u[jj] + t * ex[jj] / W0, v[jj] + t * ey[jj] / H0, lod[jj])
        out[jj] = acc / n
    return out


def tri_uv_jacobian(X, Y, T, UV):
    """Per triangle screen -> uv Jacobian (n, 4: du/dx, du/dy, dv/dx, dv/dy; uv units per pixel), affine over
    the triangle (zero for degenerate triangles)."""
    x0, y0 = X[T[:, 0]], Y[T[:, 0]]
    e1x = X[T[:, 1]] - x0; e1y = Y[T[:, 1]] - y0
    e2x = X[T[:, 2]] - x0; e2y = Y[T[:, 2]] - y0
    uv0 = UV[T[:, 0]]
    f1 = UV[T[:, 1]] - uv0; f2 = UV[T[:, 2]] - uv0
    det = e1x * e2y - e2x * e1y
    ok = np.abs(det) > 1e-9
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    J = np.stack([(f1[:, 0] * e2y - f2[:, 0] * e1y) * inv, (-f1[:, 0] * e2x + f2[:, 0] * e1x) * inv,
                  (f1[:, 1] * e2y - f2[:, 1] * e1y) * inv, (-f1[:, 1] * e2x + f2[:, 1] * e1x) * inv], 1)
    J = np.where(np.isfinite(J), J, 0.0)
    return np.clip(J, -1e3, 1e3).astype(np.float32)


def holo_mask(rgb, cov):
    """Holographic regions of a livery texture (float 0..1, same size as rgb): pastel pixels whose hue is
    far from the pearl pink and every pastel pixel around them (the pink part of the rainbow); saturated
    logos, dark text and the pink paint stay out."""
    s = rgb.astype(np.float32) / 255.0
    mx = s.max(-1); mn = s.min(-1)
    sat = np.where(mx > 1e-4, (mx - mn) / np.maximum(mx, 1e-4), 0)
    r, g, b = s[..., 0], s[..., 1], s[..., 2]
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    hd = np.abs(((h - PEARL_HUE) + 180) % 360 - 180)
    pastel = cov & (mx > 0.55) & (sat > 0.10) & (sat < 0.62)
    core = (pastel & (hd > 30)).astype(np.float32)
    sc = s.shape[0] / 1024.0
    reg = np.clip((ndimage.gaussian_filter(core, 10 * sc) - 0.12) / 0.2, 0, 1)
    m = ndimage.gaussian_filter(reg * pastel, 1.0 * sc)
    m = np.where(cov, m, ndimage.grey_dilation(m, size=5))
    return np.clip(m, 0, 1).astype(np.float32)


def read_ui_skin(skin_dir):
    info = {}
    fp = os.path.join(skin_dir, "ui_skin.json")
    try:
        with open(fp, encoding="utf-8-sig") as fh:
            info = json.load(fh)
    except Exception:
        info = {}
    base = os.path.basename(os.path.normpath(skin_dir))
    if not info.get("number"):
        m = re.search(r"_(\d+)$", base)
        info["number"] = m.group(1) if m else ""
    if not info.get("drivername"):
        info["drivername"] = re.sub(r"_\d+$", "", base).replace("_", " ")
    info.setdefault("team", "")
    info["folder"] = base
    return info


class Livery:
    """One skin folder applied to the model: per-material mip pyramids or flat colours (AC overrides)."""

    def __init__(self, model, skin_dir, ks_detail=True):
        self.skin_dir = skin_dir
        self.info = read_ui_skin(skin_dir)
        files = rr.skin_files(skin_dir)
        self.overrides = {}
        cache = {}

        def load_tex(tn):
            if not tn:
                return None
            if tn.lower() not in cache:
                img = None
                fp = files.get(tn.lower())
                if fp is not None:
                    try:
                        img = Image.open(fp); img.load()
                        self.overrides[tn] = fp
                    except Exception as e:
                        print(f"warning: cannot read {fp} ({e}); using the embedded texture", file=sys.stderr)
                        img = None
                if img is None and tn in model.textures:
                    img = rr.decode(model.textures[tn])
                cache[tn.lower()] = img
            return cache[tn.lower()]

        self.kind, self.pyr, self.flat = [], [], []
        self.refl = None            # Skin_map reflection mask pyramid for «skin»
        self.holo = None            # holographic-region mask pyramid for «skin»
        for name in model.mat_names:
            kind, pyr, flat = "flat", None, rr.FLAT.get(name, (40, 40, 42))
            if name == "__interior__":
                flat = rr.INTERIOR_COL
            elif name == "skin":
                flat = (235, 150, 185)
            if name in rr.GLASS_MATS:
                kind = "glass"; flat = rr.GLASS_MATS[name][0]
            elif name in rr.TEX_OK:
                mat = model.mats.get(name, {})
                img = load_tex(mat.get("samplers", {}).get("txDiffuse"))
                if img is not None:
                    at = name in rr.AT_MATS
                    tex = rr.tex_array(img, rgba=at)
                    if ks_detail:
                        tex = rr.apply_const_detail(tex, img, mat, load_tex)
                    cov = None
                    if name == "skin" and tex.shape[0] == tex.shape[1]:
                        cov = model.uv_coverage(name, tex.shape[0])
                    kind, pyr = "tex", build_mips(tex, at=at, cov=cov)
                    if name == "skin":
                        lv = min(len(pyr) - 1, max(0, int(round(math.log2(max(tex.shape[0], 1) / 1024)))))
                        cov_s = model.uv_coverage(name, pyr[lv].shape[0])
                        hm = holo_mask(pyr[lv][..., :3], cov_s)
                        self.holo = build_mips((hm[..., None] * 255 + 0.5).astype(np.uint8))
                if name == "skin":
                    mp = load_tex(mat.get("samplers", {}).get("txMaps"))
                    if mp is not None:
                        b = np.asarray(mp.convert("RGB"))[..., 2:3].copy()
                        self.refl = build_mips(b)
            self.kind.append(kind); self.pyr.append(pyr); self.flat.append(srgb_to_lin(flat))


# ---------------------------------------------------------------- studio environment
SQ2 = math.sqrt(2.0)


def _box(x, a, s):
    """Box [-a, a] convolved with a gaussian of sigma s (exact in 1D)."""
    return 0.5 * (erf((x + a) / (s * SQ2)) - erf((x - a) / (s * SQ2)))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


MAGENTA = np.array([1.0, 0.30, 0.78], np.float32)
TEAL = np.array([0.25, 0.85, 1.0], np.float32)          # cool cyan


class Studio:
    """Procedural lat-long studio, rotated with the camera like a photographer's light rig.
    Lights (group, kind, centre, half sizes, rgb, gradient); every light is a hard-edged rectangle (its
    edges only softened by the reflection lobe) with a linear gradient across it:
      top    long overhead softbox (world aligned, long along the cars) with a hotter core
      line   two long thin horizontal strips behind the cars: from an eye-level camera the hood, roof and
             trunk mirror the directions just above the horizon behind the car, so these draw the crisp
             light lines along them
      strip  a horizontal strip where the car sides mirror the camera, a little high: one clean streak
             along the shoulder line
      key    a large front three-quarter softbox above the camera side (diffuse key, soft sheen)
      rim    cyan and magenta strips low behind the cars, left and right of the view axis: the silhouettes
             (roof line, wing, tyres) mirror the directions straight behind the car, so they draw thin
             coloured rim lines
      cove   the ceiling cove: a dim gradient above ~24 degrees (glass and horizontal panels show a sweep)
      back   a dim panel behind the cars, fill a dim panel behind the camera
      tstrip (top view) two lengthwise overhead strips beside the softbox (streaks along roof rails)
      gpanel (top view) tall reflector panels in front of and behind the cars (seen in the glass)
    The walls are near black, the floor a little lighter near the horizon (the light pool bounces some
    light under the cars).  radiance(d, sigma) is the environment convolved with a gaussian lobe of sigma
    rad (analytic)."""

    def __init__(self, cam_az_deg=34.0, cam_el_deg=6.0):
        ca = math.radians(cam_az_deg)
        back = ca + math.pi
        mirror = math.pi - ca                    # where a vertical side panel reflects the camera
        R_ = math.radians
        W_ = lambda v: np.array([v, v, v], np.float32)
        top_view = cam_el_deg > 55
        self.top_view = top_view
        if top_view:
            # lengthwise overhead strips moved out to where the roof rails and shoulders mirror them (a streak
            # along the car, none across the roof number); rims at the horizon left and right of the cars
            # (the side edges of the cars seen from above); reflector panels in front of / behind the cars
            tilt = math.radians(90 - cam_el_deg) * math.copysign(1.0, -math.sin(ca))
            ts = (tilt + 0.55, tilt - 0.45)
            self.lights = [
                ("top", "top", 0.0, 0.0, 0.40, 0.78, W_(2.4), 0.0),
                ("top", "top", 0.0, 0.0, 0.16, 0.60, W_(1.8), 0.0),
                ("tstrip", "top", ts[0], 0.0, 0.035, 0.85, W_(10.0), 0.0),
                ("tstrip", "top", ts[1], 0.0, 0.035, 0.85, W_(10.0), 0.0),
                ("rim", "azel", R_(90), R_(6), R_(60), R_(6), MAGENTA * 6.0, 0.0),
                ("rim", "azel", R_(-90), R_(6), R_(60), R_(6), TEAL * 6.0, 0.0),
                ("gpanel", "azel", R_(16), R_(31), R_(22), R_(17), W_(3.4), 0.6),
                ("gpanel", "azel", math.pi - R_(16), R_(36), R_(22), R_(17), W_(2.8), 0.6),
                ("cove", "cove", R_(24), 0.0, 0.0, 0.0, W_(0.10), 0.6),
                ("back", "azel", back, R_(22), R_(62), R_(18), W_(0.30), 0.6),
                ("fill", "azel", ca, R_(14), R_(40), R_(14), W_(0.30), 0.0),
            ]
            self.spec_gains = {"top": 0.18, "tstrip": 1.0, "gpanel": 0.12, "cove": 0.4}
            self.diff_gains = {"tstrip": 0.12, "rim": 0.02, "gpanel": 0.0, "cove": 0.5}
            self.glass_gains = {"top": 0.3, "tstrip": 0.6, "rim": 0.5, "gpanel": 1.0, "cove": 2.5, "back": 1.0,
                                "fill": 0.5}
        else:
            # a surface tilted t towards the camera mirrors the elevation cam_el + 2 t: the light lines sit just
            # above the camera elevation (the roof crown and the hood draw one streak each) and the rims at the
            # mirror of the camera elevation (the silhouettes, whose normals are square to the view)
            e = math.radians(cam_el_deg)
            self.lights = [
                ("top", "top", 0.0, 0.0, 0.40, 0.78, W_(2.4), 0.0),
                ("top", "top", 0.0, 0.0, 0.16, 0.60, W_(1.8), 0.0),
                ("line", "azel", back, e + R_(3.5), R_(66), R_(0.9), W_(9.0), 0.0),
                ("line", "azel", back, e + R_(15.0), R_(58), R_(1.2), W_(4.5), 0.0),
                ("strip", "azel", mirror, R_(17.0), R_(38), R_(2.3), W_(22.0), 0.0),
                ("key", "azel", ca + R_(40), R_(36), R_(22), R_(15), W_(2.6), -0.4),
                ("rim", "azel", back + R_(10), -e + R_(3.0), R_(9), R_(5.5), MAGENTA * 8.0, 0.0),
                ("rim", "azel", back - R_(10), -e + R_(3.0), R_(9), R_(5.5), TEAL * 8.0, 0.0),
                ("cove", "cove", R_(24), 0.0, 0.0, 0.0, W_(0.10), 0.6),
                ("back", "azel", back, R_(22), R_(62), R_(18), W_(0.30), 0.6),
                ("fill", "azel", ca, R_(14), R_(40), R_(14), W_(0.35), 0.0),
            ]
            # reflections seen by the camera: from high above the overhead softbox would mirror flat over the
            # roof (a photographer shoots through it), so its specular share drops with camera elevation
            self.spec_gains = {"top": float(np.interp(cam_el_deg, [30, 75], [1.0, 0.18])), "key": 0.25,
                               "cove": 0.5}
            # diffuse: the strips and rims are tuned for crisp reflections, their diffuse share is small
            self.diff_gains = {"line": 0.06, "strip": 0.06, "rim": 0.03, "key": 1.0, "cove": 0.5}
            self.glass_gains = {"top": self.spec_gains["top"], "line": 0.7, "strip": 0.4, "key": 0.8, "rim": 0.7,
                                "cove": 2.5, "back": 1.0, "fill": 0.6}
        # what the glossy floor mirrors beyond the cars: the dark studio and a faint trace of the big panels
        self.floor_gains = {"*": 0.0, "base": 1.0, "top": 0.02, "back": 0.5, "fill": 0.5}

    def radiance(self, d, sig, gains=None):
        d = np.asarray(d, np.float32)
        n = len(d)
        gains = gains or {}
        dflt = gains.get("*", 1.0)
        sig = np.broadcast_to(np.asarray(sig, np.float32), (n,)).astype(np.float32)
        sig = np.maximum(sig, 0.002)
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        el = np.arcsin(np.clip(y, -1, 1))
        az = np.arctan2(x, z)
        # walls: near black, a little lighter towards the horizon (backdrop sweep); floor: dark, lighter
        # towards the horizon (the light pool around the cars)
        wall = 0.0025 + 0.0040 * np.exp(-(el / 0.30) ** 2)
        floor = 0.0035 + 0.0080 * np.exp(-((el + 0.12) / 0.30) ** 2)
        hz = 0.5 * (1 + erf(el / (np.maximum(sig, 0.004) * SQ2)))
        base = (floor + (wall - floor) * hz).astype(np.float32) * gains.get("base", 1.0)
        out = np.repeat(base[:, None], 3, 1)
        out *= np.array([1.0, 0.98, 1.0], np.float32)
        top_u = top_v = None
        for group, kind, c1, c2, a, b, rgb, grad in self.lights:
            g = gains.get(group, dflt)
            if g <= 0:
                continue
            if kind == "top":
                if top_u is None:
                    top_u = np.arctan2(x, np.maximum(y, 1e-3)); top_v = np.arctan2(z, np.maximum(y, 1e-3))
                w = _box(top_u - c1, a, sig) * _box(top_v - c2, b, sig) * (y > 0)
            elif kind == "cove":
                w = 0.5 * (1 + erf((el - c1) / (sig * SQ2))) * (1 + grad * np.clip((el - c1) / (np.pi / 2 - c1), 0, 1))
            else:
                du = _wrap(az - c1); dv = el - c2
                sel = np.abs(dv) < b + 5 * sig + 0.05
                w = np.zeros(n, np.float32)
                if sel.any():
                    cs = np.maximum(np.cos(el[sel]), 0.15)
                    w[sel] = _box(du[sel] * cs, a, sig[sel]) * _box(dv[sel], b, sig[sel])
                    if grad:
                        w[sel] *= np.clip(1 + grad * dv[sel] / b, 0, None)
            out += (w * g)[:, None].astype(np.float32) * rgb[None, :]
        return out


def sh9_basis(n):
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    return np.stack([0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                     1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1),
                     1.092548 * x * z, 0.546274 * (x * x - y * y)], 1).astype(np.float32)


def sh_irradiance_coeffs(studio, gains=None, nlat=96, nlon=192):
    th = (np.arange(nlat) + 0.5) / nlat * np.pi
    ph = (np.arange(nlon) + 0.5) / nlon * 2 * np.pi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(TH) * np.sin(PH), np.cos(TH), np.sin(TH) * np.cos(PH)], -1).reshape(-1, 3)
    dw = (np.sin(TH) * (np.pi / nlat) * (2 * np.pi / nlon)).reshape(-1)
    L = studio.radiance(d, 0.05, gains)
    Y = sh9_basis(d)
    c = (Y * dw[:, None]).T @ L                     # 9 x 3
    A = np.array([np.pi, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3, np.pi / 4, np.pi / 4, np.pi / 4,
                  np.pi / 4, np.pi / 4], np.float32)
    return (c * A[:, None]).astype(np.float32)


def sh_irradiance(coef, n):
    return np.maximum(sh9_basis(n) @ coef, 0.0)


# ---------------------------------------------------------------- layout: instances, visibility, floor maps
def fib_hemisphere(K, ymin=-0.08):
    i = np.arange(K) + 0.5
    y = 1 - i / K * (1 - ymin)
    r = np.sqrt(np.maximum(0, 1 - y * y))
    ph = i * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.sin(ph), y, r * np.cos(ph)], 1)


VIS_K = 128
LUT_NB = 6
_VIS = None


def _vis_chunk(ks):
    P, VN, T_occ, dirs, res, centre, radius = _VIS
    s = res / (2 * radius)
    out = np.zeros((len(P), len(ks)), np.uint8)
    rel = P - centre
    for j, k in enumerate(ks):
        d = dirs[k]
        helper = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = normalize(np.cross(d, helper)); e2 = np.cross(d, e1)
        X = rel @ e1 * s + res / 2; Y = rel @ e2 * s + res / 2; D = 1000.0 - rel @ d
        zb, _, _, _ = rr.rasterize(X, Y, D, T_occ, np.zeros(len(T_occ)), res, res)
        zb = zb.reshape(res, res)
        q = rel + VN * 0.012 + d * 0.006
        qx = q @ e1 * s + res / 2 - 0.5; qy = q @ e2 * s + res / 2 - 0.5; qd = 1000.0 - q @ d
        x0 = np.floor(qx).astype(np.int64); y0 = np.floor(qy).astype(np.int64)
        fx = qx - x0; fy = qy - y0
        acc = np.zeros(len(P))
        for dy in (0, 1):
            for dx in (0, 1):
                xi = x0 + dx; yi = y0 + dy
                inside = (xi >= 0) & (xi < res) & (yi >= 0) & (yi < res)
                z = np.full(len(P), np.inf)
                z[inside] = zb[yi[inside], xi[inside]]
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                acc += w * (qd <= z + 0.025)
        out[:, j] = np.clip(acc * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


def yaw_matrix(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])     # row vectors: p @ M


class Layout:
    """Car instances placed in the studio (x, z, yaw each) + everything that depends only on their placement."""

    def __init__(self, model, cars, jobs=1, log=print, cache_dir=None, fast=False):
        t0 = time.time()
        self.model = model
        self.cars = [tuple(float(v) for v in c) for c in cars]
        n_cars = len(cars)
        self.n = n_cars
        nv, nt = len(model.P), len(model.T)
        Ps, VNs, FNs, offs = [], [], [], []
        for x, z, yaw in self.cars:
            M = yaw_matrix(yaw)
            off = np.array([x, -model.floor_y, z])
            Ps.append(model.P @ M + off)
            VNs.append((model.VN @ M.astype(np.float32)).astype(np.float32))
            FNs.append((model.FN @ M.astype(np.float32)).astype(np.float32))
            offs.append(off)
        self.offsets = offs
        self.P = np.concatenate(Ps)
        self.VN = np.concatenate(VNs)
        self.UV = np.concatenate([model.UV] * n_cars)
        self.T = np.concatenate([model.T + i * nv for i in range(n_cars)])
        self.TM = np.concatenate([model.TM] * n_cars)
        self.EMIS = np.concatenate([model.EMIS] * n_cars)
        self.CAR = np.repeat(np.arange(n_cars, dtype=np.int32), nt)
        self.FN = np.concatenate(FNs)
        self.tri_ok = np.concatenate([model.tri_ok] * n_cars)
        self.uv_area = np.concatenate([model.uv_area] * n_cars)
        self.lens = np.concatenate([model.lens] * n_cars)
        self.is_glass_tri = model.is_glass[self.TM]
        self.ext_vertices = np.concatenate([model.ext_vertices + i * nv for i in range(n_cars)])
        self.car_centres = [np.array([x, 0.62, z]) for x, z, _ in self.cars]
        lo = self.P[self.ext_vertices].min(0); hi = self.P[self.ext_vertices].max(0)
        self.lo, self.hi = lo, hi
        # ---- directional visibility per vertex (shadow maps from VIS_K directions over the hemisphere)
        self.dirs = fib_hemisphere(VIS_K)
        occ = self.tri_ok & ~self.is_glass_tri
        centre = (lo + hi) / 2; radius = float(np.linalg.norm(hi - lo) / 2 + 0.2)
        cache = None
        if cache_dir and not fast:
            import hashlib
            st = os.stat(model.path)
            cars_key = ";".join(",".join(f"{v:.4f}" for v in c) for c in self.cars)
            key = f"{model.path}|{st.st_size}|{st.st_mtime}|{model.interior}|{cars_key}|{VIS_K}|512|v3"
            cache = os.path.join(cache_dir, "vis_" + hashlib.md5(key.encode()).hexdigest()[:16] + ".npy")
        if fast:
            self.vis = np.full((len(self.P), VIS_K), 255, np.uint8)
        elif cache and os.path.exists(cache):
            self.vis = np.load(cache)
        else:
            global _VIS
            _VIS = (self.P, self.VN.astype(np.float64), self.T[occ], self.dirs, 512, centre, radius)
            chunks = np.array_split(np.arange(VIS_K), max(1, jobs) * 4)
            if jobs > 1:
                import multiprocessing as mp
                with mp.get_context("fork").Pool(jobs) as pool:
                    parts = pool.map(_vis_chunk, chunks, chunksize=1)
            else:
                parts = [_vis_chunk(c) for c in chunks]
            _VIS = None
            self.vis = np.concatenate(parts, 1)                    # (nv_all, K) uint8
            if cache:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache, self.vis)
        # specular occlusion: visibility smoothed over a ~12 degree cone
        cosang = np.clip(self.dirs @ self.dirs.T, -1, 1)
        Wk = np.exp(-(np.arccos(cosang) / 0.21) ** 2)
        Wk /= Wk.sum(1, keepdims=True)
        self.so = np.clip(self.vis.astype(np.float32) @ Wk.T + 0.5, 0, 255).astype(np.uint8)
        # direction -> the LUT_NB nearest visibility directions with smooth weights (lat-long 256 x 128 over
        # the hemisphere): interpolating between them avoids the patchy cells of a nearest lookup
        az = (np.arange(256) + 0.5) / 256 * 2 * np.pi - np.pi
        el = (np.arange(128) + 0.5) / 128 * (np.pi / 2 + 0.1) - 0.1
        AZ, EL = np.meshgrid(az, el, indexing="ij")
        dd = np.stack([np.cos(EL) * np.sin(AZ), np.sin(EL), np.cos(EL) * np.cos(AZ)], -1).reshape(-1, 3)
        dots = dd @ self.dirs.T
        nb = np.argsort(-dots, 1)[:, :LUT_NB]
        ang = np.arccos(np.clip(np.take_along_axis(dots, nb, 1), -1, 1))
        wgt = np.exp(-(ang / 0.17) ** 2)
        wgt /= wgt.sum(1, keepdims=True)
        self.lut_k = nb.reshape(256, 128, LUT_NB).astype(np.int16)
        self.lut_w = wgt.reshape(256, 128, LUT_NB).astype(np.float32)
        t1 = time.time()
        # ---- floor height field (lowest car surface over each floor cell) -> contact shadows / occlusion
        cell = 0.02
        x0, z0 = lo[0] - 3.5, lo[2] - 3.5
        nx = int((hi[0] + 3.5 - x0) / cell) + 1; nz = int((hi[2] + 3.5 - z0) / cell) + 1
        X = (self.P[:, 0] - x0) / cell; Y = (self.P[:, 2] - z0) / cell; D = 1000.0 + self.P[:, 1]
        zb, _, _, _ = rr.rasterize(X, Y, D, self.T[occ], np.zeros(int(occ.sum())), nx, nz)
        hmin = (zb.reshape(nz, nx) - 1000.0).astype(np.float32)          # inf where nothing above
        O = np.zeros((nz, nx), np.float32)
        for h in (0.015, 0.04, 0.08, 0.13, 0.2, 0.3, 0.45, 0.65, 0.9, 1.2, 1.5):
            m = (hmin < h).astype(np.float32)
            O = np.maximum(O, ndimage.gaussian_filter(m, max(h * 0.42, 0.012) / cell))
        ao = ndimage.gaussian_filter((hmin < 0.7).astype(np.float32), 0.38 / cell)
        contact = ndimage.gaussian_filter((hmin < 0.035).astype(np.float32), 0.025 / cell)
        self.floor_light = np.clip((1 - 0.96 * O) * (1 - 0.62 * ao) * (1 - 0.8 * contact), 0, 1).astype(np.float32)
        self.floor_grid = (x0, z0, cell, nx, nz)
        log(f"layout {self.cars}: visibility {t1 - t0:.1f}s{' (skipped)' if fast else ''}, floor {time.time() - t1:.1f}s")

    def floor_light_at(self, x, z):
        x0, z0, cell, nx, nz = self.floor_grid
        gx = (x - x0) / cell; gz = (z - z0) / cell
        v = ndimage.map_coordinates(self.floor_light, [gz, gx], order=1, mode="constant", cval=1.0)
        return v.astype(np.float32)


# ---------------------------------------------------------------- camera
class Camera:
    def __init__(self, eye, target, up_hint=(0, 1, 0)):
        self.eye, self.f, self.rgt, self.up = rr.look_at(eye, target, up_hint)
        self.target = np.asarray(target, float)

    def rel(self, P):
        rel = P - self.eye
        D = rel @ self.f
        return rel @ self.rgt / D, rel @ self.up / D, D

    def fit(self, L, Ws, Hs, margins, zoom=1.0):
        """Frame the exterior vertices of all cars: margins = (left, right, top, bottom) fractions."""
        ux, uy, _ = self.rel(L.P[L.ext_vertices])
        ml, mr, mt, mb = margins
        lo_x, hi_x, lo_y, hi_y = ux.min(), ux.max(), uy.min(), uy.max()
        fpx = min(Ws * (1 - ml - mr) / (hi_x - lo_x), Hs * (1 - mt - mb) / (hi_y - lo_y)) * zoom
        self.fpx = fpx
        self.mx, self.my = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
        self.cx = Ws * (ml + (1 - ml - mr) / 2); self.cy = Hs * (mt + (1 - mt - mb) / 2)
        self.Ws, self.Hs = Ws, Hs

    def focal_mm(self):
        """35 mm equivalent focal length of the framing (36 mm wide frame)."""
        return float(self.fpx / self.Ws * 36.0)

    def project(self, P, scale=1.0):
        a, b, D = self.rel(P)
        X = ((a - self.mx) * self.fpx + self.cx) * scale
        Y = (-(b - self.my) * self.fpx + self.cy) * scale
        return X, Y, D

    def pixel_dirs(self, px, py, scale=1.0):
        a = (px + 0.5) / scale - self.cx
        b = (py + 0.5) / scale - self.cy
        a = a / self.fpx + self.mx; b = -b / self.fpx + self.my
        d = self.f[None, :] + a[:, None] * self.rgt[None, :] + b[:, None] * self.up[None, :]
        return normalize(d).astype(np.float32)


def view_setup(name, args, n_cars):
    """(camera, cam_az, look) for a view: preset camera + layout + look, or the custom camera."""
    p = dict(PRESETS.get(name, CUSTOM))
    if args.gap is not None or args.stagger is not None or n_cars != len(p["cars"]):
        gap = args.gap if args.gap is not None else 2.45
        st = args.stagger if args.stagger is not None else 0.8
        cars = []
        for i in range(n_cars):
            k = (n_cars - 1) / 2 - i
            f = k / ((n_cars - 1) / 2) if n_cars > 1 else 0.0
            cars.append((k * gap, f * st / 2, 0.0))
        p["cars"] = tuple(cars)
    if args.dof is not None:
        p["dof"] = args.dof
    if args.back_ev is not None:
        p["back_ev"] = args.back_ev
    cars = p["cars"]
    cx = float(np.mean([c[0] for c in cars])); cz = float(np.mean([c[1] for c in cars]))
    if name == "custom":
        if args.eye and args.target:
            eye = [float(t) for t in args.eye.split(",")]; tgt = [float(t) for t in args.target.split(",")]
            cam = Camera(eye, tgt)
            dv = cam.eye - cam.target
            return cam, math.degrees(math.atan2(dv[0], dv[2])), p
        vals = [float(t) for t in args.camera.split(",")]
        az, el, dist = vals[:3]
        tgt = vals[3:6] if len(vals) >= 6 else [cx, p["ty"], cz]
        a, e = math.radians(az), math.radians(el)
        d = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
        eye = np.asarray(tgt) + d * dist
    else:
        az, dist = p["az"], p["dist"]
        tgt = np.array([cx, p["ty"], cz])
        a = math.radians(az)
        if p.get("eye_h") is not None:
            eye = tgt + np.array([math.sin(a) * dist, p["eye_h"] - p["ty"], math.cos(a) * dist])
            el = math.degrees(math.atan2(p["eye_h"] - p["ty"], dist))
        else:
            el = p["el"]
            e = math.radians(el)
            eye = tgt + np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)]) * dist
    up_hint = (0, 1, 0)
    if el > 85:
        up_hint = (math.sin(a + math.pi), 0, math.cos(a + math.pi))
    return Camera(eye, tgt, up_hint), az, p


# ---------------------------------------------------------------- shading
def hue_rotate(g, ang):
    """Rotate colours (n, 3) about the grey axis by ang (n,) radians (+ = red -> green -> blue)."""
    mean = g.mean(1, keepdims=True)
    c = g - mean
    k = 1.0 / math.sqrt(3.0)
    cr = k * np.stack([c[:, 2] - c[:, 1], c[:, 0] - c[:, 2], c[:, 1] - c[:, 0]], 1)
    return mean + c * np.cos(ang)[:, None] + cr * np.sin(ang)[:, None]


def pearl_holo(alb, NV, ry, holo, phase):
    """Pearl flop (a slight hue shift and lift at grazing angles, so the pink reads the same in every view)
    on the pink paint, and a gentle view dependent shift of the designed rainbow on the holographic
    regions (it moves with the angle but stays the band the texture draws).  Returns (albedo, pearl
    paint weight)."""
    g = np.power(np.clip(alb, 0, 1), 1 / 2.2)
    mx = g.max(1); mn = g.min(1)
    d = np.maximum(mx - mn, 1e-6)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    r_, g_, b_ = g[:, 0], g[:, 1], g[:, 2]
    hue = np.where(mx == r_, ((g_ - b_) / d) % 6, np.where(mx == g_, (b_ - r_) / d + 2, (r_ - g_) / d + 4)) * 60
    hd = np.abs(((hue - PEARL_HUE) + 180) % 360 - 180)
    pm = (np.clip((34 - hd) / 14, 0, 1) * np.clip((sat - 0.12) / 0.12, 0, 1) * np.clip((mx - 0.45) / 0.2, 0, 1)
          * (1 - holo)).astype(np.float32)
    flop = np.power(1 - NV, 1.6)
    ang = (math.radians(PEARL_FLOP_DEG) * flop * pm
           + holo * (phase + math.radians(75) * np.power(1 - NV, 1.3) + math.radians(22) * ry))
    g2 = hue_rotate(g, ang)
    lift = (0.06 * flop * flop * pm)[:, None]
    g2 = g2 + (np.array([0.98, 0.93, 0.95], np.float32) - g2) * lift
    return np.power(np.clip(g2, 0, 1), 2.2).astype(np.float32), pm


class ShadeCtx:
    def __init__(self, model, liveries, L, studio, car_gain=None, lod_bias=LOD_BIAS):
        self.m, self.liv, self.L, self.env = model, liveries, L, studio
        self.lod_bias = lod_bias
        self.car_gain = np.ones(L.n, np.float32) if car_gain is None else np.asarray(car_gain, np.float32)
        self.holo_phase = np.array([0.0, 0.45, 0.9, 1.35][:L.n] + [0.0] * max(0, L.n - 4), np.float32)
        # diffuse: the overhead softbox (with its near-field fall-off) and the rest of the rig separately
        dg = dict(studio.diff_gains)
        # (the overhead softbox at 0.85 of its reflected strength: the upper surfaces stay close in value to
        # the sides, so the pink reads as one colour on roof and doors)
        self.sh_top = sh_irradiance_coeffs(studio, {"*": 0.0, "base": 0.0, "top": 1.0}) * DIFFUSE_GAIN * 0.85
        self.sh_rest = sh_irradiance_coeffs(studio, dict(dg, top=0.0)) * DIFFUSE_GAIN
        # per-vertex directional occlusion for this environment
        Lk = studio.radiance(L.dirs.astype(np.float32), 0.22, dg).mean(1)
        cosk = np.clip(L.VN @ L.dirs.T.astype(np.float32), 0, None)          # (nv, K)
        wl = cosk * Lk[None, :]
        vis = L.vis.astype(np.float32) / 255.0
        num = (vis * wl).sum(1); den = wl.sum(1)
        occ_l = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 1.0)
        num2 = (vis * cosk).sum(1); den2 = cosk.sum(1)
        ao = np.where(den2 > 1e-6, num2 / np.maximum(den2, 1e-6), 1.0)
        self.occ = np.clip(0.7 * occ_l + 0.3 * ao, 0, 1).astype(np.float32)
        self.P32 = L.P.astype(np.float32)
        self.jac = None                       # per-triangle screen -> uv Jacobian of the current pass

    def irradiance(self, ns, y, occ):
        E = sh_irradiance(self.sh_top, ns) * near_field(y)[:, None] + sh_irradiance(self.sh_rest, ns)
        return E * occ[:, None]

    def spec_occ(self, tri, bw, Rd):
        L = self.L
        az = np.arctan2(Rd[:, 0], Rd[:, 2]); el = np.arcsin(np.clip(Rd[:, 1], -1, 1))
        ia = np.clip(((az + np.pi) / (2 * np.pi) * 256).astype(np.int32), 0, 255)
        ie = np.clip(((el + 0.1) / (np.pi / 2 + 0.1) * 128).astype(np.int32), 0, 127)
        kk = L.lut_k[ia, ie].astype(np.int64); ww = L.lut_w[ia, ie]
        so = np.zeros(len(Rd), np.float32)
        for j in range(kk.shape[1]):
            k = kk[:, j]
            so += ww[:, j] * (L.so[tri[:, 0], k] * bw[:, 0] + L.so[tri[:, 1], k] * bw[:, 1] + L.so[tri[:, 2], k] * bw[:, 2])
        so /= 255.0
        so = np.where(Rd[:, 1] < -0.06, 1.0, so)
        # sharpen a little: partial occlusion of the reflection lobe
        x = np.clip((so - 0.15) / 0.75, 0, 1)
        return (x * x * (3 - 2 * x)).astype(np.float32)

    def albedo(self, tt, uv):
        """Base colour (linear), Skin_map reflection weight and holographic mask per sample: anisotropic
        filtering from the per-triangle Jacobian of the current pass."""
        L, m = self.L, self.m
        n = len(tt)
        col = np.zeros((n, 3), np.float32)
        refl = np.ones(n, np.float32)
        holo = np.zeros(n, np.float32)
        car = L.CAR[tt]; mt = L.TM[tt]
        key = car.astype(np.int64) * 1000 + mt
        for kv in np.unique(key):
            jj = np.nonzero(key == kv)[0]
            c, mid = int(kv // 1000), int(kv % 1000)
            lv = self.liv[c]
            if lv.kind[mid] == "tex":
                pyr = lv.pyr[mid]
                J = self.jac[tt[jj]]
                s = sample_aniso(pyr, uv[jj, 0], uv[jj, 1], J, self.lod_bias)
                col[jj] = srgb_to_lin(s[:, :3])
                if mid == m.skin_id:
                    H0 = pyr[0].shape[0]
                    s1 = footprint(J, pyr[0].shape[1], H0)[0]
                    lod = np.log2(np.maximum(s1, 1e-6))
                    if lv.refl is not None:
                        r = sample_mip(lv.refl, uv[jj, 0], uv[jj, 1], lod - math.log2(H0 / lv.refl[0].shape[0]))[:, 0] / 255.0
                        refl[jj] = 0.55 + 0.45 * r
                    if lv.holo is not None:
                        holo[jj] = sample_mip(lv.holo, uv[jj, 0], uv[jj, 1], lod - math.log2(H0 / lv.holo[0].shape[0]))[:, 0] / 255.0
            else:
                col[jj] = lv.flat[mid]
        return col, refl, holo

    def gbuffer_points(self, tt, b1, b2):
        L = self.L
        tri = L.T[tt]
        bw = np.stack([1 - b1 - b2, b1, b2], 1).astype(np.float32)
        pos = (self.P32[tri[:, 0]] * bw[:, 0:1] + self.P32[tri[:, 1]] * bw[:, 1:2] + self.P32[tri[:, 2]] * bw[:, 2:3])
        ns = (L.VN[tri[:, 0]] * bw[:, 0:1] + L.VN[tri[:, 1]] * bw[:, 1:2] + L.VN[tri[:, 2]] * bw[:, 2:3])
        uv = (L.UV[tri[:, 0]] * bw[:, 0:1] + L.UV[tri[:, 1]] * bw[:, 1:2] + L.UV[tri[:, 2]] * bw[:, 2:3])
        return tri, bw, pos, ns, uv

    def orient(self, ns, ng, V):
        ng = np.where(((ng * V).sum(1) < 0)[:, None], -ng, ng)
        ns = normalize(ns)
        ns = np.where(((ns * ng).sum(1) < 0)[:, None], -ns, ns)
        nv = (ns * V).sum(1)
        bend = nv < 0.05
        if bend.any():
            ns[bend] = normalize(ns[bend] + V[bend] * (0.05 - nv[bend])[:, None])
        return ns.astype(np.float32), ng

    def shade_opaque(self, tt, b1, b2, eye):
        """Returns (radiance, world position, emitted radiance) of the opaque samples."""
        L, m, env = self.L, self.m, self.env
        tri, bw, pos, ns, uv = self.gbuffer_points(tt, b1, b2)
        V = normalize(eye[None, :].astype(np.float32) - pos).astype(np.float32)
        ns, ng = self.orient(ns, L.FN[tt], V)
        NV = np.clip((ns * V).sum(1), 0, 1)
        Rd = (2 * NV[:, None] * ns - V).astype(np.float32)
        mid = L.TM[tt]; car = L.CAR[tt]
        p = {k: v[mid].copy() for k, v in m.prm.items()}
        alb, refl, holo = self.albedo(tt, uv)
        occ = (self.occ[tri[:, 0]] * bw[:, 0] + self.occ[tri[:, 1]] * bw[:, 1] + self.occ[tri[:, 2]] * bw[:, 2])
        occ = np.maximum(occ, p["occ_min"])
        E = self.irradiance(ns, pos[:, 1], occ) * p["fill"][:, None]
        so = self.spec_occ(tri, bw, Rd)
        F = np.minimum(schlick(NV, 0.04), 0.55)      # grazing Fresnel limited (lobe width / horizon occlusion)
        # pearl paint and holographic film on the livery
        sk = np.nonzero(mid == m.skin_id)[0]
        pm = np.zeros(len(tt), np.float32)
        if len(sk):
            alb[sk], pm[sk] = pearl_holo(alb[sk], NV[sk], Rd[sk, 1], holo[sk], self.holo_phase[car[sk]])
        out = np.zeros_like(alb)
        Fcc = F * p["cc"] * refl
        mt = p["mt"]
        kd = p["kd"] * (1 - 0.25 * holo)
        out += alb * E / np.pi * (kd * (1 - Fcc) * (1 - mt))[:, None]

        def envlobe(sel, sig):
            r = env.radiance(Rd[sel], sig if np.ndim(sig) == 0 else sig[sel], env.spec_gains)
            s = so[sel][:, None]
            return r * s + OCC_RAD[None, :] * (1 - s)

        sel = p["cc"] > 0
        if sel.any():
            out[sel] += envlobe(sel, p["cc_s"]) * Fcc[sel][:, None]
        sel = p["sp"] > 0
        if sel.any():
            tint = p["sp_tint"][sel][:, None]
            col = (1 - tint) + tint * np.clip(alb[sel] / np.maximum(alb[sel].max(1, keepdims=True), 1e-4), 0, 1)
            w = pm[sel][:, None]
            col = col * (1 - w) + PEARL_SHEEN[None, :] * w
            out[sel] += envlobe(sel, p["sp_s"]) * (F[sel] * p["sp"][sel] * (1 - Fcc[sel]))[:, None] * col
        sel = mt > 0
        if sel.any():
            f0 = alb[sel] + (1 - alb[sel]) * p["mt_lift"][sel][:, None]
            fm = f0 + (1 - f0) * schlick(NV[sel], 0.0)[:, None]
            out[sel] += envlobe(sel, p["mt_s"]) * fm * (mt[sel] * (1 - Fcc[sel]))[:, None]
        # holographic film: a tinted metallic reflection with a soft lobe, soft clamped so a bright light
        # never washes the band out to flat white (it keeps its colour)
        sel = holo > 0.02
        if sel.any():
            hc = np.clip(alb[sel] / np.maximum(alb[sel].max(1, keepdims=True), 1e-4), 0, 1) ** 1.8
            fh = (0.25 + 0.75 * schlick(NV[sel], 0.20)) * 0.40 * holo[sel]
            rh = envlobe(sel, 0.10) * hc * fh[:, None]
            lum = rh.max(1, keepdims=True)
            out[sel] += rh / (1.0 + lum / 0.6)
        em = L.EMIS[tt]
        emis = None
        if (em > 0).any():
            emis = m.emit_rgb[em] * self.car_gain[car][:, None]
        out *= self.car_gain[car][:, None]
        if emis is not None:
            out += emis
        out = np.minimum(np.nan_to_num(out, nan=0.0, posinf=0.0), 12.0)
        return out, pos, emis

    def shade_glass(self, tt, b1, b2, eye):
        """Returns (reflected radiance, transmittance) for the nearest glass layer."""
        L, m, env = self.L, self.m, self.env
        tri, bw, pos, ns, uv = self.gbuffer_points(tt, b1, b2)
        V = normalize(eye[None, :].astype(np.float32) - pos).astype(np.float32)
        ns, ng = self.orient(ns, L.FN[tt], V)
        NV = np.clip((ns * V).sum(1), 0, 1)
        Rd = (2 * NV[:, None] * ns - V).astype(np.float32)
        so = self.spec_occ(tri, bw, Rd)
        F = np.minimum(schlick(NV, GLASS_F0), 0.5)
        r = env.radiance(Rd, 0.008, env.glass_gains)
        r = r * so[:, None] + OCC_RAD[None, :] * (1 - so[:, None])
        Tm = np.empty((len(tt), 3), np.float32)
        isl = L.lens[tt]
        is_red = np.array([n == "lights_glass" for n in m.mat_names])[L.TM[tt]]
        Tm[:] = GLASS_T["window"]
        Tm[isl] = GLASS_T["lens"]
        Tm[is_red] = GLASS_T["lights_glass"]
        occ = (self.occ[tri[:, 0]] * bw[:, 0] + self.occ[tri[:, 1]] * bw[:, 1] + self.occ[tri[:, 2]] * bw[:, 2])
        E = self.irradiance(ns, pos[:, 1], occ)
        haze = E / np.pi * 0.010
        haze[is_red] = E[is_red] / np.pi * np.array([0.10, 0.006, 0.008], np.float32)
        refl = r * F[:, None]
        refl = refl / (1.0 + refl / 0.42)                 # soft clamp: softbox reflections never blow out
        refl = refl + haze
        g = self.car_gain[L.CAR[tt]][:, None]
        T = Tm * (1 - F)[:, None]
        return np.minimum(np.nan_to_num(refl), 8.0) * g, T


# ---------------------------------------------------------------- render passes
def tri_screen_area(X, Y, T):
    x0, x1, x2 = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    y0, y1, y2 = Y[T[:, 0]], Y[T[:, 1]], Y[T[:, 2]]
    return (np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) * 0.5).astype(np.float32)


def raster_scene(ctx, X, Y, D, W, H):
    """Opaque (alpha tested) and glass G-buffers.  ctx.jac must hold the Jacobian of this projection: the
    alpha test reads the filtered alpha at the pixel's mip level (no crawling decal edges)."""
    L, m = ctx.L, ctx.m
    is_at = np.array([n in rr.AT_MATS for n in m.mat_names])
    opaque = ~L.is_glass_tri & L.tri_ok
    glass = L.is_glass_tri & L.tri_ok
    gid_op = np.nonzero(opaque)[0]
    zero = np.zeros(len(L.T))
    bias = np.array([rr.DEPTH_BIAS.get(n, 0.0) for n in m.mat_names])[L.TM]

    def at_filter(tl, b1, b2):
        tt = gid_op[tl]
        keep = np.ones(len(tt), bool)
        mt = L.TM[tt]
        atm = is_at[mt]
        if atm.any():
            ii = np.nonzero(atm)[0]
            tri = L.T[tt[ii]]
            b0 = 1 - b1[ii] - b2[ii]
            uv = L.UV[tri[:, 0]] * b0[:, None] + L.UV[tri[:, 1]] * b1[ii, None] + L.UV[tri[:, 2]] * b2[ii, None]
            key = L.CAR[tt[ii]].astype(np.int64) * 1000 + mt[ii]
            for kv in np.unique(key):
                sel = np.nonzero(key == kv)[0]
                lv = ctx.liv[int(kv // 1000)]
                pyr = lv.pyr[int(kv % 1000)]
                if pyr is None or pyr[0].shape[2] < 4:
                    continue
                s1 = footprint(ctx.jac[tt[ii[sel]]], pyr[0].shape[1], pyr[0].shape[0])[0]
                lod = np.log2(np.maximum(s1, 1e-6))
                a = sample_mip(pyr, uv[sel, 0], uv[sel, 1], lod)[:, 3]
                keep[ii[sel]] = a >= 128
        return keep

    zb, tid, B1, B2 = rr.rasterize(X, Y, D, L.T[opaque], bias[opaque], W, H, at_filter)
    tid = np.where(tid >= 0, gid_op[np.maximum(tid, 0)], -1).astype(np.int32)
    gid_gl = np.nonzero(glass)[0]
    zg, tidg, G1, G2 = rr.rasterize(X, Y, D, L.T[glass], zero[glass], W, H, None)
    tidg = np.where(tidg >= 0, gid_gl[np.maximum(tidg, 0)], -1).astype(np.int32)
    return zb.astype(np.float32), tid, B1, B2, zg.astype(np.float32), tidg, G1, G2


def shade_buffers(ctx, gb, eye, npx, chunk=1_000_000, want_pos=False, want_emis=False):
    """Shade the opaque G-buffer: (radiance, world height or None, emitted radiance or None)."""
    zb, tid, B1, B2, zg, tidg, G1, G2 = gb
    img = np.zeros((npx, 3), np.float32)
    hit = np.nonzero(tid >= 0)[0]
    posy = np.full(npx, np.nan, np.float32) if want_pos else None
    emi = np.zeros((npx, 3), np.float32) if want_emis else None
    for k in range(0, len(hit), chunk):
        h = hit[k:k + chunk]
        c, pos, e = ctx.shade_opaque(tid[h], B1[h], B2[h], eye)
        img[h] = c
        if want_pos:
            posy[h] = pos[:, 1]
        if want_emis and e is not None:
            emi[h] = e
    return img, posy, emi


def composite_glass(ctx, gb, img, eye, emi=None):
    zb, tid, B1, B2, zg, tidg, G1, G2 = gb
    gh = np.nonzero((tidg >= 0) & (zg < zb))[0]
    for k in range(0, len(gh), 1_000_000):
        h = gh[k:k + 1_000_000]
        refl, T = ctx.shade_glass(tidg[h], G1[h], G2[h], eye)
        img[h] = img[h] * T + refl
        if emi is not None:
            emi[h] *= T
    return gh


def blur_stack_select(chans, sig_px, levels=(0.0, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0)):
    """Per-pixel variable gaussian blur by interpolating a stack of blurred images. chans: (H, W, C)."""
    H, W, C = chans.shape
    sig = np.clip(sig_px, levels[0], levels[-1])
    out = np.zeros_like(chans)
    prev = None
    for i, s in enumerate(levels):
        cur = chans if s == 0 else np.stack([ndimage.gaussian_filter(chans[..., c], s, mode="nearest") for c in range(C)], -1)
        if i > 0:
            lo = levels[i - 1]
            w = np.clip((sig - lo) / (s - lo), 0, 1)
            inb = (sig > lo) & (sig <= s)
            w = np.where(inb, w, 0.0)[..., None]
            out += np.where(inb[..., None], prev * (1 - w) + cur * w, 0.0)
        else:
            out += np.where((sig <= s)[..., None], cur, 0.0)
        prev = cur
    return out


FLOOR_Y = 0.0            # the layout puts the tyre contact patches on y = 0


def mirror_reflection(ctx, cam, eye, Ws, Hs, ms=0.5):
    """Planar mirror pass at reduced resolution: the cars mirrored in the floor, shaded from the mirrored
    eye, faded gently with height above the floor and blurred by the floor gloss: the blur of every
    reflected point grows with its own height (sharp at the tyre contact, soft at the roof), so the body
    colour never smears down over the contact patches.
    Returns (Hm, Wm, 4): premultiplied rgb + coverage, and the scale."""
    L = ctx.L
    Wm, Hm = int(Ws * ms), int(Hs * ms)
    Pm = L.P.copy(); Pm[:, 1] = 2 * FLOOR_Y - Pm[:, 1]
    Xm, Ym, Dm = cam.project(Pm, ms)
    jac_main = ctx.jac
    ctx.jac = tri_uv_jacobian(Xm, Ym, L.T, L.UV)
    gbm = raster_scene(ctx, Xm, Ym, Dm, Wm, Hm)
    eye_m = eye.copy(); eye_m[1] = 2 * FLOOR_Y - eye_m[1]
    mimg, mposy, _ = shade_buffers(ctx, gbm, eye_m, Wm * Hm, want_pos=True)
    composite_glass(ctx, gbm, mimg, eye_m)
    ctx.jac = jac_main
    cov = (gbm[1] >= 0).astype(np.float32)
    del gbm
    hgt = np.where(cov > 0, np.nan_to_num(mposy, nan=0.0) - FLOOR_Y, 0.0).astype(np.float32)
    fade = np.exp(-np.maximum(hgt, 0) / 1.8) * (cov > 0)
    mimg = mimg.reshape(Hm, Wm, 3); cov2 = cov.reshape(Hm, Wm); hgt2 = hgt.reshape(Hm, Wm)
    fade2 = fade.reshape(Hm, Wm)
    gy, gx = np.mgrid[0:Hm, 0:Wm]
    dirs_m = cam.pixel_dirs(gx.ravel(), gy.ravel(), ms)
    tfl = np.where(dirs_m[:, 1] < -1e-4, (FLOOR_Y - eye[1]) / np.minimum(dirs_m[:, 1], -1e-4), 1e3)
    depth_fl = (tfl * (dirs_m @ cam.f.astype(np.float32))).reshape(Hm, Wm)
    del dirs_m, gx, gy, tfl
    # blur radius from the pixel's own reflected height; outside the mirror image (where the blurred image
    # spreads) from the nearest reflected point, smoothed, so the soft halo never stops at a hard edge
    if (cov2 > 0).any():
        near = ndimage.distance_transform_edt(cov2 <= 0, return_distances=False, return_indices=True)
        h_near = hgt2[near[0], near[1]]
        del near
        h_near = ndimage.gaussian_filter(h_near, 4)
    else:
        h_near = np.zeros_like(hgt2)
    h_use = np.where(cov2 > 0, hgt2, h_near)
    beta = 0.035                                                    # floor gloss lobe (rad)
    sig_px = beta * 2 * h_use * cam.fpx * ms / np.maximum(depth_fl, 1.0) + 0.5
    chans = np.concatenate([mimg * fade2[..., None], cov2[..., None]], -1).astype(np.float32)
    return blur_stack_select(chans, sig_px).astype(np.float32), ms


def shade_background(ctx, cam, eye, idx, Ws, mb, ms, look):
    """Glossy black floor (diffuse light pool x contact shadows and occlusion footprint + Fresnel
    reflection of the cars / studio, faded towards the camera) fading into the near-black cyclorama with a
    maroon-pink glow behind the cars and a dashed light line on the back wall."""
    L, env = ctx.L, ctx.env
    py, px = np.divmod(idx, Ws)
    gyf = (py + 0.5) * ms - 0.5; gxf = (px + 0.5) * ms - 0.5
    refl_car = np.stack([ndimage.map_coordinates(mb[..., c], [gyf, gxf], order=1, mode="nearest")
                         for c in range(4)], 1).astype(np.float32)
    dirs = cam.pixel_dirs(px, py)
    down = dirs[:, 1] < -1e-4
    t = np.where(down, (FLOOR_Y - eye[1]) / np.minimum(dirs[:, 1], -1e-4), 0.0)
    hx = eye[0] + dirs[:, 0] * t; hz = eye[2] + dirs[:, 2] * t
    centre = (L.lo + L.hi) / 2
    r = np.hypot(hx - centre[0], hz - centre[2]).astype(np.float32)
    up = np.array([[0, 1, 0]], np.float32)
    E_up = float((sh_irradiance(ctx.sh_top, up) * near_field(np.zeros(1, np.float32))[:, None]
                  + sh_irradiance(ctx.sh_rest, up))[0].mean())
    # light pool: a soft spot centred under the cars
    pool = (1.0 + (r / look.pool_r) ** 2) ** -2.0
    fl = L.floor_light_at(hx, hz)
    floor_col = (look.floor_alb * E_up / np.pi * pool * fl)[:, None] * np.array([1.0, 0.97, 0.98], np.float32)
    # floor reflection: Fresnel x (mirror image of the cars, else the studio), weaker towards the camera
    cosv = np.clip(-dirs[:, 1], 0, 1)
    Ff = schlick(cosv, 0.04) * look.refl
    depth = np.where(down, t * (dirs @ cam.f.astype(np.float32)), 1e4).astype(np.float32)
    fc = np.clip((depth - look.d_near) / max(look.d_car - look.d_near, 1e-3), 0, 1)
    fc = 0.30 + 0.70 * fc * fc * (3 - 2 * fc)
    Rf = dirs.copy(); Rf[:, 1] = -Rf[:, 1]
    env_r = env.radiance(Rf, 0.08, env.floor_gains)
    refl = refl_car[:, :3] + env_r * (1 - np.clip(refl_car[:, 3:4], 0, 1))
    floor_rad = floor_col + refl * (Ff * fc)[:, None]
    # backdrop (infinite cyclorama): near black with a maroon-pink glow behind the cars
    gx = (px + 0.5 - look.pool_cx) / look.pool_rx; gy = (py + 0.5 - look.pool_cy) / look.pool_ry
    glow = np.exp(-1.6 * (gx * gx + gy * gy))
    wide = np.exp(-0.5 * (gx * gx * 0.25 + gy * gy * 0.6))
    back = look.back * look.back_tint[None, :] + (look.pool_glow * glow)[:, None] * look.glow_rgb[None, :] \
        + (look.pool_glow * 0.35 * wide)[:, None] * look.back_tint[None, :]
    if look.line_y is not None:
        # dashed light line on the back wall (the butcher-chart dash), with a soft glow around it
        ph = ((px + 0.5 - look.line_x0) % look.line_period) / look.line_period
        dash = np.clip((look.line_duty - np.abs(ph - 0.5) * 2 + 0.5 * look.line_soft) / look.line_soft, 0, 1)
        inx = np.clip((px - look.line_x0) / (0.06 * Ws), 0, 1) * np.clip((look.line_x1 - px) / (0.06 * Ws), 0, 1)
        dy = (py + 0.5 - look.line_y)
        core = np.exp(-0.5 * (dy / look.line_w) ** 2)
        halo = np.exp(-0.5 * (dy / (look.line_w * 9)) ** 2)
        back = back + ((core * dash * look.line_core + halo * (0.35 + 0.65 * dash) * look.line_glow) * inx)[:, None] \
            * look.line_rgb[None, :]
    fade_far = np.clip((r - look.far0) / look.far1, 0, 1) ** 1.5
    fade_far = np.where(down, fade_far, 1.0)[:, None]
    return (floor_rad * (1 - fade_far) + back * fade_far).astype(np.float32), depth


def render_view(factory, name, args, log=print):
    """Returns (image with the title band if --title, image without it, car bounding box in output px)."""
    t0 = time.time()
    W, H, ss = args.width, args.height, args.ss
    Ws, Hs = W * ss, H * ss
    n_cars = len(factory.liveries)
    cam, cam_az, look = view_setup(name, args, n_cars)
    dv = cam.eye - cam.target
    cam_el = math.degrees(math.atan2(dv[1], math.hypot(dv[0], dv[2])))
    L = factory.layout(look["cars"])
    # the car further from the camera is the second plane: darker (and blurred by the depth of field)
    dep = np.array([float((np.asarray(c) - cam.eye) @ cam.f) for c in L.car_centres])
    gain = np.ones(L.n, np.float32)
    if L.n > 1 and look.get("back_ev", 0.0) != 0.0:
        gain[int(np.argmax(dep))] = 2.0 ** look["back_ev"]
    ctx = ShadeCtx(factory.model, factory.liveries, L, Studio(cam_az, cam_el), gain)
    margins = tuple(look.get("margins", (0.06, 0.06, 0.12, 0.27)))
    cam.fit(L, Ws, Hs, margins, args.zoom)
    eye = cam.eye.astype(np.float32)
    X, Y, D = cam.project(L.P)
    ctx.jac = tri_uv_jacobian(X, Y, L.T, L.UV)
    gb = raster_scene(ctx, X, Y, D, Ws, Hs)
    t1 = time.time()
    npx = Ws * Hs
    img, _, emi = shade_buffers(ctx, gb, eye, npx, want_emis=True)
    t2 = time.time()

    # ---- look of the floor and backdrop
    cen = (L.lo + L.hi) / 2
    pcx, pcy, _ = cam.project(np.array([[cen[0], 1.15, cen[2]]]))
    ux, uy, _ = cam.project(L.P[L.ext_vertices])
    top = name == "top_pair" or cam_el > 55
    lk = SimpleNamespace(
        floor_alb=look.get("floor_alb", 0.16), refl=look.get("refl", 0.5),
        pool_r=float(max(L.hi[0] - L.lo[0], L.hi[2] - L.lo[2]) * 0.55),
        d_car=float(dep.min()) - 1.5, d_near=float(dep.min()) * 0.45,
        pool_cx=float(pcx[0]), pool_cy=float(min(pcy[0], uy.min() + 0.45 * (uy.max() - uy.min()))),
        pool_rx=float((ux.max() - ux.min()) * 0.55), pool_ry=float(max(uy.max() - uy.min(), Hs * 0.3) * 0.85),
        back=0.0006, pool_glow=0.022, back_tint=np.array([1.0, 0.86, 0.92], np.float32),
        glow_rgb=np.array([1.0, 0.24, 0.40], np.float32),
        far0=float(np.linalg.norm(L.hi - L.lo) * 0.5 + 3.0), far1=10.0, line_y=None)
    if look.get("line_h") is not None and not top:
        # dashed light line on the back wall (seen behind the cars from a low camera)
        fdir = np.array([cam.f[0], 0.0, cam.f[2]]); fdir /= max(np.linalg.norm(fdir), 1e-6)
        lx, ly, _ = cam.project((np.array([cen[0], look["line_h"], cen[2]]) + fdir * 14.0)[None, :])
        lk.line_y = float(ly[0]); lk.line_w = 1.1 * ss
        lk.line_x0, lk.line_x1 = 0.03 * Ws, 0.97 * Ws
        lk.line_period = 46.0 * ss * (W / 2400.0); lk.line_duty = 0.62; lk.line_soft = 0.08
        lk.line_core = 0.55; lk.line_glow = 0.020; lk.line_rgb = np.array([1.0, 0.16, 0.36], np.float32)
    if top:
        lk.pool_r = float(max(L.hi[0] - L.lo[0], L.hi[2] - L.lo[2]) * 0.75)
        lk.far0, lk.far1 = 30.0, 10.0

    mb, ms = mirror_reflection(ctx, cam, eye, Ws, Hs)
    t3 = time.time()
    bg = np.nonzero(gb[1] < 0)[0]
    want_depth = look.get("dof", 0.0) > 0
    if want_depth:
        depth = np.where(gb[1] >= 0, gb[0], 1e4).astype(np.float32)
    for k in range(0, len(bg), 1_500_000):
        idx = bg[k:k + 1_500_000]
        img[idx], dpt = shade_background(ctx, cam, eye, idx, Ws, mb, ms, lk)
        if want_depth:
            depth[idx] = dpt
    del mb, bg
    # glass (also over the floor seen through the windows)
    gh = composite_glass(ctx, gb, img, eye, emi)
    layer = None
    if want_depth:
        layer = np.where(gb[1] >= 0, L.CAR[np.maximum(gb[1], 0)], -1).astype(np.int8)
        gl = gh[gb[1][gh] < 0]
        layer[gl] = L.CAR[gb[5][gl]]
        depth[gl] = gb[4][gl]
    t4 = time.time()

    # ---- post: DOF, bloom, haze, vignette, tone mapping, float downsample, dither
    img = img.reshape(Hs, Ws, 3); emi = emi.reshape(Hs, Ws, 3)
    del gb
    dof = None
    if want_depth:
        dof = (depth.reshape(Hs, Ws), layer.reshape(Hs, Ws), list(dep))
    disp = post_process(img, emi, dof, look, args, ss, lk, exposure=args.exposure * look.get("exposure", 1.0))
    del img, emi
    out = downsample_dither(disp, W, H, seed=zlib.crc32(name.encode()))
    t5 = time.time()
    # screen position of each car (for the title band ordering) and the cars' bounding box
    car_xy = []
    for c in L.car_centres:
        cx_, cy_, _ = cam.project(np.asarray(c)[None, :])
        car_xy.append((float(cx_[0]) / ss, float(cy_[0]) / ss))
    bbox = (float(ux.min()) / ss, float(uy.min()) / ss, float(ux.max()) / ss, float(uy.max()) / ss)
    raw = out
    if args.title:
        out = draw_title(out, ctx.liv, car_xy, args)
    log(f"{name}: focal {cam.focal_mm():.0f} mm eq., eye {np.round(cam.eye, 2).tolist()}, el {cam_el:.1f}, "
        f"cars y {bbox[1] / H:.2f}-{bbox[3] / H:.2f} ({(bbox[3] - bbox[1]) / H:.0%} of the height), "
        f"x {bbox[0] / W:.2f}-{bbox[2] / W:.2f}; raster {t1 - t0:.0f}s, shade {t2 - t1:.0f}s, "
        f"mirror {t3 - t2:.0f}s, floor+glass {t4 - t3:.0f}s, post {t5 - t4:.0f}s, total {time.time() - t0:.0f}s, "
        f"peak {_peak_mb():.0f} MB")
    return out, raw, bbox


def _peak_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def _dilate_max(a, r):
    """Max filter of radius r pixels, computed on a 4x max-pooled grid (fast, slightly generous)."""
    H, W = a.shape
    k = 4
    h, w = -(-H // k), -(-W // k)
    pad = np.zeros((h * k, w * k), a.dtype); pad[:H, :W] = a
    small = pad.reshape(h, k, w, k).max((1, 3))
    rs = int(math.ceil(r / k))
    if rs > 0:
        small = ndimage.maximum_filter(small, size=2 * rs + 1)
    return np.repeat(np.repeat(small, k, 0), k, 1)[:H, :W]


def depth_of_field(img, depth, layer, car_depths, strength, ss):
    """Layered depth of field focused on the nearest car: studio layer then each car back to front, every
    layer blurred by its own circle of confusion (normalized convolution) and composited with its blurred
    coverage, so a sharp car never bleeds into the blurred background behind it."""
    Hs, Ws = depth.shape
    zf = min(car_depths)
    cmax = 20.0 * ss
    coc = np.minimum(strength * ss * (Ws / ss / 2400.0) * 9.0 * np.abs(1 - zf / np.maximum(depth, 0.1)) * (zf / 11.0), cmax)
    coc = coc.astype(np.float32)
    levels = tuple(float(v) * ss for v in (0, 0.75, 1.5, 3, 6, 12, 20))
    # studio (floor + backdrop): its own blur everywhere (behind the cars the nearby studio blur is used)
    m = (layer < 0).astype(np.float32)
    cs = ndimage.gaussian_filter(coc * m, 8 * ss) / np.maximum(ndimage.gaussian_filter(m, 8 * ss), 1e-4)
    field = np.where(m > 0, coc, cs)
    st = blur_stack_select(np.concatenate([img * m[..., None], m[..., None]], -1), field, levels)
    out = st[..., :3] / np.maximum(st[..., 3:4], 1e-4)
    for c in np.argsort(car_depths)[::-1]:                    # far car first
        m = (layer == c).astype(np.float32)
        if not m.any():
            continue
        cc = coc * m
        field = np.maximum(cc, _dilate_max(cc, cmax) * (1 - m))
        st = blur_stack_select(np.concatenate([img * m[..., None], m[..., None]], -1), field, levels)
        a = np.clip(st[..., 3:4], 0, 1)
        out = st[..., :3] + out * (1 - a)
    return out.astype(np.float32)


def post_process(img, emi, dof, look, args, ss, lk, exposure=1.0):
    """HDR (Hs, Ws, 3) -> display-referred float (Hs, Ws, 3) in 0..1 (sRGB encoded).  emi: the emitted part
    of img (lamps, after the lens glass): it blooms more than the reflected highlights."""
    Hs, Ws = img.shape[:2]
    if dof is not None and look.get("dof", 0) > 0:
        img = depth_of_field(img, dof[0], dof[1], dof[2], look["dof"], ss)
    # bloom from the bright part of the HDR image (softbox streaks) and from the lamps (DRL, tail lights)
    k = 4
    h4, w4 = Hs // k, Ws // k
    small = img[:h4 * k, :w4 * k].reshape(h4, k, w4, k, 3).mean((1, 3))
    lum = small.max(2, keepdims=True)
    bright = small * np.clip((lum - 1.1) / np.maximum(lum, 1e-4), 0, None)
    if emi is not None:
        bright = bright + 0.9 * emi[:h4 * k, :w4 * k].reshape(h4, k, w4, k, 3).mean((1, 3))
    bloom = np.zeros_like(small)
    px = (Ws / ss) / 2400.0 * ss / k                 # one output pixel of a 2400-wide frame, in bloom pixels
    for s, wgt in ((2.0 * px, 0.40), (7.0 * px, 0.35), (22.0 * px, 0.25)):
        bloom += wgt * np.stack([ndimage.gaussian_filter(bright[..., c], max(s, 0.5)) for c in range(3)], -1)
    for c in range(3):
        up = ndimage.zoom(bloom[..., c], (Hs / h4, Ws / w4), order=1)
        img[:up.shape[0], :up.shape[1], c] += 0.12 * up[:Hs, :Ws]
    del bloom, bright, small
    yy = (np.arange(Hs, dtype=np.float32) + 0.5) / Hs
    xx = (np.arange(Ws, dtype=np.float32) + 0.5) / Ws
    # haze: a faint maroon-pink veil in the light pool behind the cars (cooler towards one side)
    hx = np.exp(-((xx * Ws - lk.pool_cx) / (lk.pool_rx * 0.9)) ** 2)
    hy = np.exp(-((yy * Hs - lk.pool_cy) / (lk.pool_ry * 1.1)) ** 2)
    mix = np.clip((xx * Ws - lk.pool_cx) / (lk.pool_rx * 1.4) * 0.5 + 0.5, 0, 1)
    rose = np.array([1.0, 0.32, 0.52], np.float32)
    tint = rose[None, :] * (1 - 0.35 * mix[:, None]) + TEAL[None, :] * (0.35 * mix[:, None])   # (Ws, 3)
    haze_amt = 0.0035
    yv = yy * 2 - 1
    xv = xx * 2 - 1
    out = np.empty((Hs, Ws, 3), np.float32)
    for r0 in range(0, Hs, 256):
        r1 = min(Hs, r0 + 256)
        blk = img[r0:r1] + haze_amt * (hy[r0:r1, None, None] * hx[None, :, None]) * tint[None, :, :]
        r2 = (xv[None, :] ** 2) * 0.80 + (yv[r0:r1, None] ** 2) * 0.95
        vig = 1 - 0.38 * (np.clip(r2, 0, 1.75) / 1.75) ** 1.6
        v = aces(blk * exposure) * vig[..., None]
        out[r0:r1] = np.power(np.clip(v, 0, 1), 1 / 2.2)
    return out


def downsample_dither(disp, W, H, seed=0):
    """Float display image -> (W, H) uint8 RGB: triangle-filter downsample in float (no negative lobes, so
    no ringing halos at high-contrast edges), then ~1 LSB triangular dither before quantising (no banding
    in the dark gradients)."""
    Hs, Ws = disp.shape[:2]
    chans = []
    for c in range(3):
        im = Image.fromarray(np.ascontiguousarray(disp[..., c]).astype(np.float32), mode="F")
        if (Ws, Hs) != (W, H):
            im = im.resize((W, H), Image.BILINEAR)
        chans.append(np.asarray(im, np.float32))
    v = np.stack(chans, -1) * 255.0
    rng = np.random.default_rng(seed)
    v += rng.random(v.shape, dtype=np.float32) + rng.random(v.shape, dtype=np.float32) - 1.0
    return Image.fromarray(np.clip(np.round(v), 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- title band
def find_font(name, size, var=None, extra_dirs=()):
    for d in list(extra_dirs) + FONT_DIRS:
        fp = os.path.join(d, name)
        if os.path.exists(fp):
            try:
                f = ImageFont.truetype(fp, int(size))
                if var:
                    try:
                        f.set_variation_by_name(var)
                    except Exception:
                        pass
                return f
            except Exception:
                pass
    for fp in rr.FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, int(size))
            except Exception:
                pass
    return ImageFont.load_default()


_FONT_EXTRA = []


def holo_strip(w, h, phase=0.0):
    x = np.linspace(0, 1, w)[None, :]
    t = x * 2.2 * np.pi + phase
    ones = np.ones((h, 1))
    r = 205 + 50 * np.sin(t); g = 195 + 55 * np.sin(t + 2.1); b = 220 + 35 * np.sin(t + 4.2)
    arr = np.stack([r * ones, g * ones, b * ones], -1)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _spaced(draw, xy, text, font, fill, spacing):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, anchor="ls")
        x += draw.textlength(ch, font=font) + spacing
    return x


def _spaced_len(draw, text, font, spacing):
    return sum(draw.textlength(ch, font=font) + spacing for ch in text) - spacing


def screen_order(car_xy):
    """Order of the cars for captions: left to right, or top to bottom when they are stacked vertically."""
    xy = np.asarray(car_xy, float)
    if len(xy) > 1 and np.ptp(xy[:, 1]) > np.ptp(xy[:, 0]):
        return list(np.argsort(xy[:, 1]))
    return list(np.argsort(xy[:, 0]))


def draw_title(im, liveries, car_xy, args):
    """Team band: «КОМАНДА ЭДМ · BUTCHER CHART CHROME» + driver number plates and names (ui_skin.json),
    drivers ordered like the cars on screen."""
    W, H = im.size
    s = min(W / 2400.0, H / 1350.0) * 0.8           # a slim band (~9 % of the frame), overlaid on the floor
    im = im.convert("RGBA")
    bh = int(150 * s)
    y0 = H - bh
    # dark fade under the band (the floor reflection shows through above it)
    grad_h = int(bh * 2.0)
    a = np.clip(np.arange(grad_h) / (grad_h - bh * 0.85), 0, 1) ** 1.8 * 200
    rng = np.random.default_rng(7)
    a = a[:, None] + rng.random((grad_h, W)) - 0.5           # dithered alpha ramp
    shade = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")
    over = Image.new("RGBA", (W, grad_h), (8, 4, 8, 255)); over.putalpha(shade)
    im.alpha_composite(over, (0, H - grad_h))
    d = ImageDraw.Draw(im)
    pad = int(70 * s)
    # holographic hairline + maroon dashed butcher-chart rule, with air between them
    hl = holo_strip(W - 2 * pad, max(2, int(3 * s))).convert("RGBA")
    im.alpha_composite(hl, (pad, y0 - int(8 * s)))
    dash, gap_ = int(30 * s), int(14 * s)
    yl = y0 + int(20 * s)
    x = pad
    while x < W - pad:
        d.rounded_rectangle([x, yl, min(x + dash, W - pad), yl + max(2, int(4 * s))], radius=max(1, int(2 * s)),
                            fill=(170, 34, 68, 255))
        x += dash + gap_
    yb = y0 + int(bh * 0.74)
    # team line
    f_team = find_font("MontserratAlternates-BlackItalic.ttf", 54 * s, extra_dirs=_FONT_EXTRA)
    f_sub = find_font("Exo2-Italic[wght].ttf", 36 * s, "ExtraBold Italic", extra_dirs=_FONT_EXTRA)
    team = (liveries[0].info.get("team") or TEAM_LINE[0]).upper()
    sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sh).text((pad + int(3 * s), yb + int(4 * s)), team, font=f_team, fill=170, anchor="ls")
    sh = sh.filter(ImageFilter.GaussianBlur(4 * s))
    shl = Image.new("RGBA", (W, H), (70, 0, 30, 255)); shl.putalpha(sh)
    im.alpha_composite(shl)
    d = ImageDraw.Draw(im)
    d.text((pad, yb), team, font=f_team, fill=(255, 150, 198, 255), anchor="ls")
    x = pad + d.textlength(team, font=f_team) + int(26 * s)
    # holo diamond separator
    cap = f_sub.getbbox("B", anchor="ls")
    cy = yb + (cap[1] + cap[3]) / 2
    r_ = 9 * s
    dm = Image.new("L", (W, H), 0)
    ImageDraw.Draw(dm).polygon([(x, cy), (x + r_, cy - r_), (x + 2 * r_, cy), (x + r_, cy + r_)], fill=255)
    hol = holo_strip(W, H, 1.3).convert("RGBA"); hol.putalpha(dm)
    im.alpha_composite(hol)
    x += 2 * r_ + int(26 * s)
    # subtitle: flat light pearl, crisp on the dark band
    _spaced(d, (x, yb), TEAM_LINE[1], f_sub, (244, 232, 242, 255), 3 * s)
    d = ImageDraw.Draw(im)
    # drivers
    f_num = find_font("Unbounded[wght].ttf", 36 * s, "Black", extra_dirs=_FONT_EXTRA)
    f_first = find_font("SofiaSansCondensed-Italic[wght].ttf", 25 * s, "Bold Italic", extra_dirs=_FONT_EXTRA)
    f_last = find_font("SofiaSansCondensed-Italic[wght].ttf", 46 * s, "Black Italic", extra_dirs=_FONT_EXTRA)
    blocks = []
    for i in screen_order(car_xy):
        info = liveries[i].info
        num = str(info.get("number", "")).strip()
        parts = str(info.get("drivername", "")).strip().upper().split()
        first, last = (" ".join(parts[:-1]), parts[-1]) if len(parts) > 1 else ("", " ".join(parts))
        pw = (d.textlength(num, font=f_num) + 40 * s) if num else 0
        tw = max(d.textlength(last, font=f_last), _spaced_len(d, first, f_first, 2 * s) if first else 0)
        blocks.append((num, first, last, pw, tw))
    gapb = int(56 * s)
    total = sum(pw + 18 * s + tw for _, _, _, pw, tw in blocks) + gapb * (len(blocks) - 1)
    x = W - pad - total
    ph = int(66 * s)
    pc = yb - int(18 * s)
    for num, first, last, pw, tw in blocks:
        if num:
            d.rounded_rectangle([x, pc - ph // 2, x + pw, pc + ph // 2], radius=int(15 * s), fill=(128, 20, 50, 255),
                                outline=(255, 255, 255, 255), width=max(2, int(4 * s)))
            nb = f_num.getbbox(num, anchor="ls")
            d.text((x + pw / 2, pc - (nb[1] + nb[3]) / 2), num, font=f_num, fill=(255, 255, 255, 255), anchor="ms")
        xn = x + pw + 18 * s
        lb = f_last.getbbox(last, anchor="ls")
        if first:
            _spaced(d, (xn + 2 * s, pc - int(9 * s)), first, f_first, (255, 172, 208, 255), 2 * s)
            d.text((xn, pc + int(26 * s)), last, font=f_last, fill=(246, 240, 244, 255), anchor="ls")
        else:
            d.text((xn, pc - (lb[1] + lb[3]) / 2), last, font=f_last, fill=(246, 240, 244, 255), anchor="ls")
        x = xn + tw + gapb
    return im.convert("RGB")


def _crop_box(bbox, W, H, aspect, pad=0.07, below=0.16):
    """A crop of the given aspect around a car bounding box (with air around it and room below for the
    floor reflection), inside the W x H frame where possible; when the cars do not fit a crop of that
    aspect inside the frame, the box reaches past the frame edge (the caller extends the image)."""
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    cw = bw * (1 + 2 * pad); ch = bh * (1 + 2 * pad + below)
    cw = max(cw, ch * aspect); ch = cw / aspect
    if cw > W and ch * (W / cw) >= bh * (1 + pad):
        cw = W; ch = cw / aspect
    cx = (x0 + x1) / 2; cy = (y0 + y1) / 2 + bh * below / 2
    l = min(max(cx - cw / 2, 0), W - cw) if cw <= W else cx - cw / 2
    t = min(max(cy - ch / 2, 0), H - ch) if ch <= H else cy - ch / 2
    return int(round(l)), int(round(t)), int(round(l + cw)), int(round(t + ch))


def _crop_ext(im, box):
    """Crop, extending the image by its edge pixels where the box reaches past the frame."""
    W, H = im.size
    l, t, r, b = box
    pl, pt, pr, pb = max(0, -l), max(0, -t), max(0, r - W), max(0, b - H)
    if pl or pt or pr or pb:
        a = np.pad(np.asarray(im), ((pt, pb), (pl, pr), (0, 0)), mode="edge")
        im = Image.fromarray(a); l += pl; r += pl; t += pt; b += pt
    return im.crop((l, t, r, b))


def team_sheet(imgs, bboxes, views, liveries, cols=2, tw=1200, th=572):
    """Shareable contact sheet: the team header once (team, livery, drivers from ui_skin.json), then every
    view without its title band, cropped to the cars, with a readable caption."""
    s = tw / 1200.0
    pad, head, capt = int(28 * s), int(118 * s), int(64 * s)
    rows = (len(imgs) + cols - 1) // cols
    Wt = cols * tw + (cols + 1) * pad
    Ht = head + rows * (th + capt) + (rows - 1) * int(10 * s) + pad
    sheet = Image.new("RGB", (Wt, Ht), (11, 8, 11))
    d = ImageDraw.Draw(sheet)
    f_team = find_font("MontserratAlternates-BlackItalic.ttf", 44 * s, extra_dirs=_FONT_EXTRA)
    f_sub = find_font("Exo2-Italic[wght].ttf", 28 * s, "ExtraBold Italic", extra_dirs=_FONT_EXTRA)
    f_drv = find_font("SofiaSansCondensed-Italic[wght].ttf", 34 * s, "Black Italic", extra_dirs=_FONT_EXTRA)
    f_cap = find_font("SofiaSansCondensed-Italic[wght].ttf", 40 * s, "Black Italic", extra_dirs=_FONT_EXTRA)
    yb = int(66 * s)
    team = (liveries[0].info.get("team") or TEAM_LINE[0]).upper()
    d.text((pad, yb), team, font=f_team, fill=(255, 150, 198), anchor="ls")
    x = pad + d.textlength(team, font=f_team) + int(22 * s)
    _spaced(d, (x, yb), TEAM_LINE[1], f_sub, (244, 232, 242), 2 * s)
    drv = "   ·   ".join(f"№{lv.info.get('number', '')}  {str(lv.info.get('drivername', '')).upper()}" for lv in liveries)
    d.text((Wt - pad, yb), drv, font=f_drv, fill=(246, 240, 244), anchor="rs")
    hl = holo_strip(Wt - 2 * pad, max(2, int(3 * s))).convert("RGB")
    sheet.paste(hl, (pad, int(86 * s)))
    x = pad
    while x < Wt - pad:
        d.rounded_rectangle([x, int(100 * s), min(x + int(24 * s), Wt - pad), int(103 * s)], radius=1, fill=(170, 34, 68))
        x += int(36 * s)
    for k, (im, bb, v) in enumerate(zip(imgs, bboxes, views)):
        r_, c_ = divmod(k, cols)
        x = pad + c_ * (tw + pad); y = head + r_ * (th + capt + int(10 * s))
        box = _crop_box(bb, im.size[0], im.size[1], tw / th, below=0.0 if v == "top_pair" else 0.16)
        sheet.paste(_crop_ext(im, box).resize((tw, th), Image.LANCZOS), (x, y))
        cap = CAPTIONS.get(v, v).upper()
        d.rounded_rectangle([x, y + th + int(14 * s), x + int(8 * s), y + th + int(48 * s)], radius=2, fill=(170, 34, 68))
        _spaced(d, (x + int(20 * s), y + th + int(46 * s)), cap, f_cap, (244, 232, 242), 1.5 * s)
    return sheet


# ---------------------------------------------------------------- driver
class Factory:
    """Model, liveries and the layouts the views need (visibility precomputed before forking)."""

    def __init__(self, model, liveries, jobs, cache, fast, log):
        self.model, self.liveries = model, liveries
        self.jobs, self.cache, self.fast, self.log = jobs, cache, fast, log
        self.layouts = {}

    def layout(self, cars):
        key = tuple(tuple(round(float(v), 4) for v in c) for c in cars)
        if key not in self.layouts:
            self.layouts[key] = Layout(self.model, key, jobs=self.jobs, log=self.log, cache_dir=self.cache, fast=self.fast)
        return self.layouts[key]


_JOB = None


def _render_one(view):
    args, factory = _JOB
    im, raw, bbox = render_view(factory, view, args, log=lambda s: print(s, flush=True))
    fp = os.path.join(args.out, view + ".png")
    im.save(fp)
    return fp, np.asarray(raw), bbox


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skins", nargs="+", required=True, help="skin folders (car A, car B, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default=",".join(ALL_VIEWS))
    ap.add_argument("--width", type=int, default=2400); ap.add_argument("--height", type=int, default=1350)
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--title", action="store_true", help="add the team title band")
    ap.add_argument("--gap", type=float, default=None, help="replace the preset layouts: distance between the cars (m)")
    ap.add_argument("--stagger", type=float, default=None, help="replace the preset layouts: front-to-back offset (m)")
    ap.add_argument("--dof", type=float, default=None, help="depth of field strength (0 = off; default per view)")
    ap.add_argument("--back-ev", type=float, default=None, help="exposure of the car further away (EV; default per view)")
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--zoom", type=float, default=1.0, help="scale of the auto framing")
    ap.add_argument("--camera", default=None, metavar="az,el,dist[,tx,ty,tz]", help="custom camera (view «custom»)")
    ap.add_argument("--eye", default=None, metavar="x,y,z", help="custom camera position (write --eye=-6,1.2,-7 when it starts negative)")
    ap.add_argument("--target", default=None, metavar="x,y,z", help="custom camera target (with --eye)")
    ap.add_argument("--kn5", default=rr.DEFAULT_KN5)
    ap.add_argument("--no-interior", action="store_true")
    ap.add_argument("--no-ks-detail", action="store_true", help="skip the ksPerPixelMultiMap constant detail colour")
    ap.add_argument("--fonts", default=None, help="extra font folder searched first")
    ap.add_argument("--cache", default=None, help="folder to cache the per-layout visibility (speeds up re-renders)")
    ap.add_argument("--fast", action="store_true", help="previews: no visibility / occlusion precomputation")
    ap.add_argument("--no-sheet", action="store_true", help="skip the contact sheet")
    a = ap.parse_args()
    if a.fonts:
        _FONT_EXTRA.append(a.fonts)
    views = [v for v in a.views.split(",") if v]
    if (a.camera or (a.eye and a.target)) and "custom" not in views:
        views.append("custom")
    for v in views:
        if v not in PRESETS and v != "custom":
            ap.error(f"unknown view {v}")
        if v == "custom" and not (a.camera or (a.eye and a.target)):
            ap.error("view custom needs --camera or --eye/--target")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model = Model(a.kn5, interior=not a.no_interior)
    liveries = [Livery(model, d, ks_detail=not a.no_ks_detail) for d in a.skins]
    print(f"model: {len(model.T)} tris x {len(liveries)} cars, floor y {model.floor_y:.4f}, {time.time() - t0:.1f}s", flush=True)
    for lv in liveries:
        print(f"  {lv.info['folder']}: #{lv.info.get('number')} {lv.info.get('drivername')}; overrides: "
              + ", ".join(sorted(os.path.basename(p) for p in lv.overrides.values())), flush=True)
    log = lambda s: print(s, flush=True)
    factory = Factory(model, liveries, a.jobs, a.cache, a.fast, log)
    for v in views:                       # layouts (visibility) before forking
        factory.layout(view_setup(v, a, len(liveries))[2]["cars"])
    global _JOB
    _JOB = (a, factory)
    jobs = max(1, min(a.jobs, len(views)))
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as pool:
            res = pool.map(_render_one, views, chunksize=1)
    else:
        res = [_render_one(v) for v in views]
    if not a.no_sheet:
        raws = [Image.fromarray(r[1]) for r in res]
        team_sheet(raws, [r[2] for r in res], views, liveries).save(os.path.join(a.out, "sheet.png"))
    print(f"done in {time.time() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
