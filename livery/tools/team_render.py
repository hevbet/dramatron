#!/usr/bin/env python3
"""Studio «hero shot» renderer: two smp_audi_rs3_lms (Audi RS3 LMS TCR) cars side by side.

Usage:
    python3 team_render.py --skins A_dir B_dir --out DIR
                           [--views front34_pair,rear34_pair,side_pair,top_pair]
                           [--width 2400 --height 1350] [--ss 2] [--jobs 4] [--title]
                           [--gap 2.45] [--stagger M] [--dof F] [--exposure E]
                           [--camera az,el,dist[,tx,ty,tz]] [--eye x,y,z --target x,y,z]
                           [--kn5 path] [--no-interior] [--no-ks-detail] [--fonts DIR]

Each skin folder is applied to its own instance of the car with every texture override found in
it (Skin.dds, glass_sticker.dds, caliper.dds, rim_d.dds, tyre_d.dds, ...), exactly like
render_rs3.py / Assetto Corsa: a file named like a texture of the .kn5 replaces that texture.
The first skin is car A (left side of the cars, x > 0, staggered forward), the second car B.
Driver names and numbers for the title band come from each folder's ui_skin.json.

Scene: both cars parallel, --gap metres apart centre to centre and --stagger metres front to back,
on a dark glossy studio floor (blurred planar mirror reflection that fades with height, soft
contact shadows and occlusion from a height field of the cars), lit by a procedural studio
environment: a large overhead softbox, two long side strip softboxes, a rim light behind the cars
(placed opposite the camera) and a dim front fill.  The environment is used for diffuse
irradiance (order-2 spherical harmonics x per-vertex directional occlusion from 128 shadow maps)
and for per-pixel reflections (clearcoat with Schlick Fresnel F0 0.04 over the base colour,
metallic rims/chrome, dark tinted glass), evaluated analytically so the softbox streaks stay
crisp on the bodywork.  Shading is deferred: the render_rs3 rasterizer produces a G-buffer
(triangle id + perspective-correct barycentrics -> position, smooth normal, UV, material, car)
and every pixel is shaded on its own.  Textures are sampled trilinearly from mip pyramids.

Views: front34_pair (low hero angle, the nearer car in front), rear34_pair, side_pair (profile in
echelon, cars overlapping slightly: this preset uses a longer stagger unless --stagger is given),
top_pair, and custom (--camera az,el,dist[,target] with az measured from the car front towards
the car left, or --eye/--target).  Post: SSAA (--ss), bloom, filmic tone mapping, vignette and
optional depth of field (--dof, 0 = off).  --title adds the team band at the bottom.

World space = AC/kn5: x = car LEFT, y = up, z = FRONT.  Writes <view>.png and sheet.png.
"""
import argparse
import json
import math
import os
import re
import sys
import time

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
# az: degrees from the car front (+z) towards the car left (+x); el: degrees above the floor
PRESETS = {
    "front34_pair": dict(az=34.0, el=6.0, dist=11.0, target=(0.0, 0.62, 0.0), stagger=0.8),
    "rear34_pair": dict(az=214.0, el=8.5, dist=11.0, target=(0.0, 0.66, 0.0), stagger=0.8),
    "side_pair": dict(az=90.0, el=5.0, dist=15.0, target=(0.0, 0.62, 0.0), stagger=3.6),
    "top_pair": dict(az=-90.0, el=80.0, dist=15.0, target=(0.0, 0.3, 0.0), stagger=0.8),
}

# ---------------------------------------------------------------- materials
# kd diffuse weight | cc clearcoat weight (Schlick F0 0.04, lobe cc_s rad) | sp dielectric spec weight
# (Schlick F0 0.04, lobe sp_s rad, tinted by the base colour by sp_tint) | mt metallic weight
# (F0 = base colour lifted by mt_lift, lobe mt_s rad)
MAT_DEFAULT = dict(kd=1.0, cc=0.0, cc_s=0.02, sp=0.0, sp_s=0.3, sp_tint=0.0, mt=0.0, mt_s=0.2, mt_lift=0.0)
MATS = {
    "skin": dict(cc=1.0, cc_s=0.010, sp=0.5, sp_s=0.28, sp_tint=0.85),       # pearl base + clearcoat
    "glass_sticker": dict(cc=0.75, cc_s=0.02),
    "ext_sticker": dict(cc=0.75, cc_s=0.02),
    "caliper": dict(cc=1.0, cc_s=0.03),
    "rim": dict(kd=0.6, mt=0.55, mt_s=0.14, mt_lift=0.18, cc=0.9, cc_s=0.02),
    "ture": dict(sp=0.6, sp_s=0.40),
    "brakedisc": dict(kd=0.5, mt=0.6, mt_s=0.30, mt_lift=0.3),
    "chrome": dict(kd=0.05, mt=1.0, mt_s=0.012, mt_lift=0.55),
    "mirror": dict(kd=0.02, mt=1.0, mt_s=0.003, mt_lift=0.7),
    "lights": dict(kd=0.5, mt=0.65, mt_s=0.06, mt_lift=0.35, cc=1.0, cc_s=0.004),
    "reflector": dict(kd=0.5, mt=0.65, mt_s=0.06, mt_lift=0.35, cc=1.0, cc_s=0.004),
    "exhaust": dict(kd=0.4, mt=0.7, mt_s=0.2, mt_lift=0.3),
    "ext_carbon": dict(cc=1.0, cc_s=0.015),
    "glassblack": dict(cc=1.0, cc_s=0.004),
    "ext_plastic": dict(sp=1.0, sp_s=0.20),
    "pl_black": dict(sp=0.8, sp_s=0.25),
    "black": dict(sp=0.4, sp_s=0.35),
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
DIFFUSE_GAIN = 0.36
OCC_RAD = np.array([0.050, 0.030, 0.038], np.float32)   # what an occluded reflection sees (car body)


def srgb_to_lin(c):
    return np.power(np.clip(np.asarray(c, np.float32) / 255.0, 0, 1), 2.2)


def schlick(cos, f0):
    m = np.clip(1.0 - cos, 0.0, 1.0)
    m2 = m * m
    return f0 + (1.0 - f0) * (m2 * m2 * m)


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


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
                meshes.append(dict(name=full, pos=pw, nrm=n, uv=v[:, 6:8].astype(np.float64),
                                   idx=idx.astype(np.int64), mat=mats[mid]["name"],
                                   interior="COCKPIT" in full.upper()))
        for _ in range(nch):
            node(M, full, skip)

    node(np.eye(4), "", False)
    return textures, mats, meshes


class Model:
    """Car geometry, shared by every instance (same filtering as render_rs3.Scene)."""

    def __init__(self, kn5_path, interior=True):
        textures, mats, meshes = load_geometry(kn5_path, interior)
        self.path, self.interior = os.path.abspath(kn5_path), interior
        self.textures = textures
        self.mats = {m["name"]: m for m in mats}
        self.mat_names, mat_id = [], {}
        P, N, UV, T, TM = [], [], [], [], []
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
            off += len(m["pos"])
        self.P = np.concatenate(P); self.UV = np.concatenate(UV)
        self.T = np.concatenate(T); self.TM = np.concatenate(TM)
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
        self.VN = normalize(vn).astype(np.float32)
        uva = self.UV[self.T[:, 1]] - self.UV[self.T[:, 0]]; uvb = self.UV[self.T[:, 2]] - self.UV[self.T[:, 0]]
        self.uv_area = np.abs(uva[:, 0] * uvb[:, 1] - uva[:, 1] * uvb[:, 0]).astype(np.float32)
        tyre = self.TM == self.mat_names.index("ture") if "ture" in self.mat_names else np.zeros(len(self.T), bool)
        self.floor_y = float(self.P[self.T[tyre]].reshape(-1, 3)[:, 1].min()) if tyre.any() else float(self.P[:, 1].min())
        self.is_glass = np.array([n in rr.GLASS_MATS for n in self.mat_names])
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


# ---------------------------------------------------------------- textures
def build_mips(tex, at=False, min_size=8):
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
                    kind, pyr = "tex", build_mips(tex, at=at)
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


class Studio:
    """Procedural lat-long studio, rotated with the camera like a photographer's light rig:
    a large overhead softbox (with a hotter core), two pairs of long horizontal strip softboxes placed
    where the car sides mirror them towards the camera (they draw the streak highlights along the
    shoulders and doors), a big soft panel behind the cars (hood/roof/windshield reflections), two
    vertical rim strips behind the cars, a dim front fill, dark walls and a dark floor.
    radiance(d, sigma) is the environment convolved with a gaussian lobe of sigma rad (analytic)."""

    def __init__(self, cam_az_deg=34.0, cam_el_deg=6.0):
        ca = math.radians(cam_az_deg)
        mirror = math.pi - ca                    # where a vertical side panel reflects the camera
        R_ = math.radians
        W_ = lambda v: np.array([v, v, v * 1.03], np.float32)
        self.lights = [
            # (group, kind, centre az, centre el, half-width u, half-width v, rgb, vertical gradient)
            ("top", "top", 0.0, 0.0, 0.50, 0.72, W_(4.2), 0.0),
            ("top", "top", 0.0, 0.0, 0.26, 0.50, W_(3.4), 0.0),
            ("strip", "azel", mirror, R_(9), R_(48), R_(1.4), W_(30.0), 0.0),
            ("strip", "azel", -mirror, R_(9), R_(48), R_(1.4), W_(30.0), 0.0),
            ("strip", "azel", mirror, R_(25), R_(42), R_(1.0), W_(22.0), 0.0),
            ("strip", "azel", -mirror, R_(25), R_(42), R_(1.0), W_(22.0), 0.0),
            ("back", "azel", ca + math.pi, R_(30), R_(60), R_(17), W_(0.95), 0.6),
            ("rim", "azel", ca + math.pi + R_(68), R_(22), R_(5), R_(20), W_(9.0), 0.0),
            ("rim", "azel", ca + math.pi - R_(68), R_(22), R_(5), R_(20), W_(9.0), 0.0),
            ("fill", "azel", ca, R_(12), R_(45), R_(16), W_(0.42), 0.0),
        ]
        # reflections seen by the camera: from high above the overhead softbox would mirror flat over the
        # roof (a photographer shoots through it), so its specular share drops with camera elevation
        self.spec_gains = {"top": float(np.interp(cam_el_deg, [30, 75], [1.0, 0.22]))}
        self.floor_gains = {g: 0.03 for g in ("top", "strip", "back", "rim", "fill")}

    def radiance(self, d, sig, gains=None):
        d = np.asarray(d, np.float32)
        n = len(d)
        sig = np.broadcast_to(np.asarray(sig, np.float32), (n,)).astype(np.float32)
        sig = np.maximum(sig, 0.002)
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        el = np.arcsin(np.clip(y, -1, 1))
        az = np.arctan2(x, z)
        # walls: dark grey, a bit lighter towards the horizon (backdrop sweep); floor: dark, lit pool below
        wall = 0.026 + 0.034 * np.exp(-(el / 0.28) ** 2)
        floor = 0.016 + 0.040 * np.clip(-y, 0, 1)
        hz = 0.5 * (1 + erf(el / (np.maximum(sig, 0.004) * SQ2)))
        base = (floor + (wall - floor) * hz).astype(np.float32)
        out = np.repeat(base[:, None], 3, 1)
        out *= np.array([1.0, 0.985, 1.02], np.float32)
        top_u = top_v = None
        for group, kind, c1, c2, a, b, rgb, grad in self.lights:
            g = 1.0 if gains is None else gains.get(group, 1.0)
            if g <= 0:
                continue
            if kind == "top":
                if top_u is None:
                    top_u = np.arctan2(x, y); top_v = np.arctan2(z, y)
                w = _box(top_u, a, sig) * _box(top_v, b, sig)
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


def sh_irradiance_coeffs(studio, nlat=96, nlon=192):
    th = (np.arange(nlat) + 0.5) / nlat * np.pi
    ph = (np.arange(nlon) + 0.5) / nlon * 2 * np.pi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(TH) * np.sin(PH), np.cos(TH), np.sin(TH) * np.cos(PH)], -1).reshape(-1, 3)
    dw = (np.sin(TH) * (np.pi / nlat) * (2 * np.pi / nlon)).reshape(-1)
    L = studio.radiance(d, 0.05)
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


class Layout:
    """Car instances placed in the studio + everything that depends only on their placement."""

    def __init__(self, model, n_cars, gap, stagger, jobs=1, log=print, cache_dir=None):
        t0 = time.time()
        self.model = model
        self.gap, self.stagger = gap, stagger
        self.n = n_cars
        nv, nt = len(model.P), len(model.T)
        offs = []
        for i in range(n_cars):
            k = (n_cars - 1) / 2 - i
            offs.append(np.array([k * gap, -model.floor_y, (k / max(1, (n_cars - 1) / 2)) * stagger / 2 if n_cars > 1 else 0.0]))
        self.offsets = offs
        self.P = np.concatenate([model.P + o for o in offs])
        self.VN = np.concatenate([model.VN] * n_cars)
        self.UV = np.concatenate([model.UV] * n_cars)
        self.T = np.concatenate([model.T + i * nv for i in range(n_cars)])
        self.TM = np.concatenate([model.TM] * n_cars)
        self.CAR = np.repeat(np.arange(n_cars, dtype=np.int32), nt)
        self.FN = np.concatenate([model.FN] * n_cars)
        self.tri_ok = np.concatenate([model.tri_ok] * n_cars)
        self.uv_area = np.concatenate([model.uv_area] * n_cars)
        self.lens = np.concatenate([model.lens] * n_cars)
        self.is_glass_tri = model.is_glass[self.TM]
        self.ext_vertices = np.concatenate([model.ext_vertices + i * nv for i in range(n_cars)])
        self.car_centres = [np.array([o[0], 0.62, o[2]]) for o in offs]
        lo = self.P[self.ext_vertices].min(0); hi = self.P[self.ext_vertices].max(0)
        self.lo, self.hi = lo, hi
        # ---- directional visibility per vertex (shadow maps from VIS_K directions over the hemisphere)
        self.dirs = fib_hemisphere(VIS_K)
        occ = self.tri_ok & ~self.is_glass_tri
        centre = (lo + hi) / 2; radius = float(np.linalg.norm(hi - lo) / 2 + 0.2)
        cache = None
        if cache_dir:
            import hashlib
            st = os.stat(model.path)
            key = f"{model.path}|{st.st_size}|{st.st_mtime}|{model.interior}|{n_cars}|{gap:.4f}|{stagger:.4f}|{VIS_K}|512|v1"
            cache = os.path.join(cache_dir, "vis_" + hashlib.md5(key.encode()).hexdigest()[:16] + ".npy")
        if cache and os.path.exists(cache):
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
        # direction -> nearest visibility direction lookup (lat-long 256 x 128 over the hemisphere)
        az = (np.arange(256) + 0.5) / 256 * 2 * np.pi - np.pi
        el = (np.arange(128) + 0.5) / 128 * (np.pi / 2 + 0.1) - 0.1
        AZ, EL = np.meshgrid(az, el, indexing="ij")
        dd = np.stack([np.cos(EL) * np.sin(AZ), np.sin(EL), np.cos(EL) * np.cos(AZ)], -1).reshape(-1, 3)
        self.dir_lut = np.argmax(dd @ self.dirs.T, 1).reshape(256, 128).astype(np.int16)
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
        ao = ndimage.gaussian_filter((hmin < 0.7).astype(np.float32), 0.45 / cell)
        contact = ndimage.gaussian_filter((hmin < 0.03).astype(np.float32), 0.02 / cell)
        self.floor_light = np.clip((1 - 0.93 * O) * (1 - 0.40 * ao) * (1 - 0.5 * contact), 0, 1).astype(np.float32)
        self.floor_grid = (x0, z0, cell, nx, nz)
        log(f"layout gap {gap:.2f} stagger {stagger:.2f}: visibility {t1 - t0:.1f}s, floor {time.time() - t1:.1f}s")

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


def preset_camera(name, args, L_target=None):
    if name == "custom":
        if args.eye and args.target:
            eye = [float(t) for t in args.eye.split(",")]; tgt = [float(t) for t in args.target.split(",")]
            return Camera(eye, tgt), None
        vals = [float(t) for t in args.camera.split(",")]
        az, el, dist = vals[:3]
        tgt = vals[3:6] if len(vals) >= 6 else [0.0, 0.62, 0.0]
    else:
        p = PRESETS[name]
        az, el, dist, tgt = p["az"], p["el"], p["dist"], list(p["target"])
    a, e = math.radians(az), math.radians(el)
    d = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
    eye = np.asarray(tgt) + d * dist
    up_hint = (0, 1, 0)
    if el > 85:
        up_hint = (math.sin(a + math.pi), 0, math.cos(a + math.pi))
    return Camera(eye, tgt, up_hint), az


# ---------------------------------------------------------------- shading
class ShadeCtx:
    def __init__(self, model, liveries, L, studio, lod_bias=-0.35):
        self.m, self.liv, self.L, self.env = model, liveries, L, studio
        self.lod_bias = lod_bias
        # diffuse gain: the softboxes are tuned for crisp reflections; their diffuse share is scaled
        # separately so the paint keeps its colour (photographer's exposure/fill balance)
        self.sh = sh_irradiance_coeffs(studio) * DIFFUSE_GAIN
        # per-vertex directional occlusion for this environment
        Lk = studio.radiance(L.dirs.astype(np.float32), 0.22).mean(1)
        cosk = np.clip(L.VN @ L.dirs.T.astype(np.float32), 0, None)          # (nv, K)
        wl = cosk * Lk[None, :]
        vis = L.vis.astype(np.float32) / 255.0
        num = (vis * wl).sum(1); den = wl.sum(1)
        occ_l = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 1.0)
        num2 = (vis * cosk).sum(1); den2 = cosk.sum(1)
        ao = np.where(den2 > 1e-6, num2 / np.maximum(den2, 1e-6), 1.0)
        self.occ = np.clip(0.7 * occ_l + 0.3 * ao, 0, 1).astype(np.float32)
        self.P32 = L.P.astype(np.float32)

    def spec_occ(self, tri, bw, Rd):
        L = self.L
        az = np.arctan2(Rd[:, 0], Rd[:, 2]); el = np.arcsin(np.clip(Rd[:, 1], -1, 1))
        ia = np.clip(((az + np.pi) / (2 * np.pi) * 256).astype(np.int32), 0, 255)
        ie = np.clip(((el + 0.1) / (np.pi / 2 + 0.1) * 128).astype(np.int32), 0, 127)
        k = L.dir_lut[ia, ie].astype(np.int64)
        so = (L.so[tri[:, 0], k] * bw[:, 0] + L.so[tri[:, 1], k] * bw[:, 1] + L.so[tri[:, 2], k] * bw[:, 2]) / 255.0
        so = np.where(Rd[:, 1] < -0.06, 1.0, so)
        # sharpen a little: partial occlusion of the reflection lobe
        return np.clip((so - 0.15) / 0.75, 0, 1).astype(np.float32)

    def albedo(self, tt, uv, tri_scr_area):
        L, m = self.L, self.m
        n = len(tt)
        col = np.zeros((n, 3), np.float32)
        refl = np.ones(n, np.float32)
        car = L.CAR[tt]; mt = L.TM[tt]
        key = car.astype(np.int64) * 1000 + mt
        for kv in np.unique(key):
            jj = np.nonzero(key == kv)[0]
            c, mid = int(kv // 1000), int(kv % 1000)
            lv = self.liv[c]
            if lv.kind[mid] == "tex":
                pyr = lv.pyr[mid]
                H0, W0 = pyr[0].shape[:2]
                t = tt[jj]
                lod = 0.5 * np.log2(np.maximum(L.uv_area[t] * W0 * H0, 1e-9) / np.maximum(tri_scr_area[t], 1e-6)) + self.lod_bias
                lod = np.where(L.uv_area[t] > 1e-12, lod, 0.0)
                s = sample_mip(pyr, uv[jj, 0], uv[jj, 1], lod)
                col[jj] = srgb_to_lin(s[:, :3])
                if m.mat_names[mid] == "skin" and lv.refl is not None:
                    h1 = lv.refl[0].shape[0]
                    lod2 = lod - math.log2(H0 / h1)
                    r = sample_mip(lv.refl, uv[jj, 0], uv[jj, 1], lod2)[:, 0] / 255.0
                    refl[jj] = 0.55 + 0.45 * r
            else:
                col[jj] = lv.flat[mid]
        return col, refl

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

    def shade_opaque(self, tt, b1, b2, eye, tri_scr_area):
        L, m, env = self.L, self.m, self.env
        tri, bw, pos, ns, uv = self.gbuffer_points(tt, b1, b2)
        V = normalize(eye[None, :].astype(np.float32) - pos).astype(np.float32)
        ns, ng = self.orient(ns, L.FN[tt], V)
        NV = np.clip((ns * V).sum(1), 0, 1)
        Rd = (2 * NV[:, None] * ns - V).astype(np.float32)
        mid = L.TM[tt]
        p = {k: v[mid] for k, v in m.prm.items()}
        alb, refl = self.albedo(tt, uv, tri_scr_area)
        occ = (self.occ[tri[:, 0]] * bw[:, 0] + self.occ[tri[:, 1]] * bw[:, 1] + self.occ[tri[:, 2]] * bw[:, 2])
        E = sh_irradiance(self.sh, ns) * occ[:, None]
        so = self.spec_occ(tri, bw, Rd)
        F = schlick(NV, 0.04)
        out = np.zeros_like(alb)
        Fcc = F * p["cc"] * refl
        mt = p["mt"]
        out += alb * E / np.pi * (p["kd"] * (1 - Fcc) * (1 - mt))[:, None]

        def envlobe(sel, sig):
            r = env.radiance(Rd[sel], sig[sel], env.spec_gains)
            s = so[sel][:, None]
            return r * s + OCC_RAD[None, :] * (1 - s)

        sel = p["cc"] > 0
        if sel.any():
            out[sel] += envlobe(sel, p["cc_s"]) * Fcc[sel][:, None]
        sel = p["sp"] > 0
        if sel.any():
            tint = p["sp_tint"][sel][:, None]
            col = (1 - tint) + tint * np.clip(alb[sel] / np.maximum(alb[sel].max(1, keepdims=True), 1e-4), 0, 1)
            out[sel] += envlobe(sel, p["sp_s"]) * (F[sel] * p["sp"][sel] * (1 - Fcc[sel]))[:, None] * col
        sel = mt > 0
        if sel.any():
            f0 = alb[sel] + (1 - alb[sel]) * p["mt_lift"][sel][:, None]
            fm = f0 + (1 - f0) * schlick(NV[sel], 0.0)[:, None]
            out[sel] += envlobe(sel, p["mt_s"]) * fm * (mt[sel] * (1 - Fcc[sel]))[:, None]
        return np.nan_to_num(out, nan=0.0, posinf=0.0), pos

    def shade_glass(self, tt, b1, b2, eye):
        """Returns (reflected radiance, transmittance) for the nearest glass layer."""
        L, m, env = self.L, self.m, self.env
        tri, bw, pos, ns, uv = self.gbuffer_points(tt, b1, b2)
        V = normalize(eye[None, :].astype(np.float32) - pos).astype(np.float32)
        ns, ng = self.orient(ns, L.FN[tt], V)
        NV = np.clip((ns * V).sum(1), 0, 1)
        Rd = (2 * NV[:, None] * ns - V).astype(np.float32)
        so = self.spec_occ(tri, bw, Rd)
        F = schlick(NV, GLASS_F0)
        r = env.radiance(Rd, 0.004, env.spec_gains)
        r = r * so[:, None] + OCC_RAD[None, :] * (1 - so[:, None])
        Tm = np.empty((len(tt), 3), np.float32)
        isl = L.lens[tt]
        is_red = np.array([n == "lights_glass" for n in m.mat_names])[L.TM[tt]]
        Tm[:] = GLASS_T["window"]
        Tm[isl] = GLASS_T["lens"]
        Tm[is_red] = GLASS_T["lights_glass"]
        occ = (self.occ[tri[:, 0]] * bw[:, 0] + self.occ[tri[:, 1]] * bw[:, 1] + self.occ[tri[:, 2]] * bw[:, 2])
        E = sh_irradiance(self.sh, ns) * occ[:, None]
        haze = E / np.pi * 0.012
        haze[is_red] = E[is_red] / np.pi * np.array([0.10, 0.006, 0.008], np.float32)
        refl = r * F[:, None] + haze
        T = Tm * (1 - F)[:, None]
        return np.nan_to_num(refl), T


# ---------------------------------------------------------------- render passes
def tri_screen_area(X, Y, T):
    x0, x1, x2 = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    y0, y1, y2 = Y[T[:, 0]], Y[T[:, 1]], Y[T[:, 2]]
    return (np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) * 0.5).astype(np.float32)


def raster_scene(ctx, X, Y, D, W, H):
    """Opaque (alpha tested) and glass G-buffers."""
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
                sel = key == kv
                lv = ctx.liv[int(kv // 1000)]
                pyr = lv.pyr[int(kv % 1000)]
                if pyr is None or pyr[0].shape[2] < 4:
                    continue
                a = rr.sample(pyr[0], uv[sel, 0], uv[sel, 1], bilinear=False)[:, 3]
                keep[ii[sel]] = a >= 128
        return keep

    zb, tid, B1, B2 = rr.rasterize(X, Y, D, L.T[opaque], bias[opaque], W, H, at_filter)
    tid = np.where(tid >= 0, gid_op[np.maximum(tid, 0)], -1).astype(np.int32)
    gid_gl = np.nonzero(glass)[0]
    zg, tidg, G1, G2 = rr.rasterize(X, Y, D, L.T[glass], zero[glass], W, H, None)
    tidg = np.where(tidg >= 0, gid_gl[np.maximum(tidg, 0)], -1).astype(np.int32)
    return zb.astype(np.float32), tid, B1, B2, zg.astype(np.float32), tidg, G1, G2


def shade_buffers(ctx, gb, eye, area, npx, chunk=1_200_000, want_pos=False):
    zb, tid, B1, B2, zg, tidg, G1, G2 = gb
    img = np.zeros((npx, 3), np.float32)
    hit = np.nonzero(tid >= 0)[0]
    posy = np.full(npx, np.nan, np.float32) if want_pos else None
    for k in range(0, len(hit), chunk):
        h = hit[k:k + chunk]
        c, pos = ctx.shade_opaque(tid[h], B1[h], B2[h], eye, area)
        img[h] = c
        if want_pos:
            posy[h] = pos[:, 1]
    return img, posy


def composite_glass(ctx, gb, img, eye):
    zb, tid, B1, B2, zg, tidg, G1, G2 = gb
    gh = np.nonzero((tidg >= 0) & (zg < zb))[0]
    for k in range(0, len(gh), 1_200_000):
        h = gh[k:k + 1_200_000]
        refl, T = ctx.shade_glass(tidg[h], G1[h], G2[h], eye)
        img[h] = img[h] * T + refl
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
    eye, faded with height above the floor and blurred by the floor gloss lobe (blur grows with the
    reflected path length).  Returns (Hm, Wm, 4): premultiplied rgb + coverage, and the scale."""
    L = ctx.L
    Wm, Hm = int(Ws * ms), int(Hs * ms)
    Pm = L.P.copy(); Pm[:, 1] = 2 * FLOOR_Y - Pm[:, 1]
    Xm, Ym, Dm = cam.project(Pm, ms)
    aream = tri_screen_area(Xm, Ym, L.T)
    gbm = raster_scene(ctx, Xm, Ym, Dm, Wm, Hm)
    eye_m = eye.copy(); eye_m[1] = 2 * FLOOR_Y - eye_m[1]
    mimg, mposy = shade_buffers(ctx, gbm, eye_m, aream, Wm * Hm, want_pos=True)
    composite_glass(ctx, gbm, mimg, eye_m)
    cov = (gbm[1] >= 0).astype(np.float32)
    del gbm
    hgt = np.where(cov > 0, np.nan_to_num(mposy, nan=0.0) - FLOOR_Y, 0.0).astype(np.float32)
    fade = np.exp(-np.maximum(hgt, 0) / 0.65) * (cov > 0)
    mimg = mimg.reshape(Hm, Wm, 3); cov2 = cov.reshape(Hm, Wm); hgt2 = hgt.reshape(Hm, Wm)
    fade2 = fade.reshape(Hm, Wm)
    gy, gx = np.mgrid[0:Hm, 0:Wm]
    dirs_m = cam.pixel_dirs(gx.ravel(), gy.ravel(), ms)
    tfl = np.where(dirs_m[:, 1] < -1e-4, (FLOOR_Y - eye[1]) / np.minimum(dirs_m[:, 1], -1e-4), 1e3)
    depth_fl = (tfl * (dirs_m @ cam.f.astype(np.float32))).reshape(Hm, Wm)
    del dirs_m, gx, gy, tfl
    sm_c = ndimage.gaussian_filter(cov2, 6)
    h_est = ndimage.gaussian_filter(hgt2 * cov2, 6) / np.maximum(sm_c, 1e-3)
    h_est = np.where(sm_c > 1e-3, h_est, 1.0)
    beta = 0.035                                                    # floor gloss lobe (rad)
    sig_px = beta * 2 * h_est * cam.fpx * ms / np.maximum(depth_fl, 1.0) + 0.6
    chans = np.concatenate([mimg * fade2[..., None], cov2[..., None]], -1).astype(np.float32)
    return blur_stack_select(chans, sig_px).astype(np.float32), ms


def shade_background(ctx, cam, eye, idx, Ws, mb, ms):
    """Floor (diffuse pool x contact shadows + Fresnel reflection of the cars / studio) fading into the
    dark cyclorama backdrop, for the flat pixel indices idx."""
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
    # floor diffuse: overhead pool x contact shadows/occlusion
    E_up = float(sh_irradiance(ctx.sh, np.array([[0, 1, 0]], np.float32))[0].mean())
    pool = (1.0 + (r / 5.0) ** 2) ** -2.0
    fl = L.floor_light_at(hx, hz)
    floor_alb = 0.020
    floor_col = (floor_alb * E_up / np.pi * pool * fl)[:, None] * np.array([1.0, 0.99, 1.0], np.float32)
    # floor reflection: Fresnel x (mirror image of the cars, else the studio)
    cosv = np.clip(-dirs[:, 1], 0, 1)
    Ff = schlick(cosv, 0.04) * 0.65
    Rf = dirs.copy(); Rf[:, 1] = -Rf[:, 1]
    env_r = env.radiance(Rf, 0.06, env.floor_gains)
    refl = refl_car[:, :3] + env_r * (1 - np.clip(refl_car[:, 3:4], 0, 1))
    floor_rad = floor_col + refl * (Ff * np.sqrt(fl))[:, None]
    # backdrop (infinite cyclorama): the floor fades into it with distance
    tgt_dir = normalize((cam.target - cam.eye)[None, :])[0].astype(np.float32)
    glow = np.exp(-((1 - np.clip(dirs @ tgt_dir, -1, 1)) / 0.02))
    back = (0.010 + 0.022 * glow)[:, None] * np.array([1.0, 0.97, 1.02], np.float32)
    fade_far = np.clip((r - 9.0) / 9.0, 0, 1) ** 1.5
    fade_far = np.where(down, fade_far, 1.0)[:, None]
    return (floor_rad * (1 - fade_far) + back * fade_far).astype(np.float32)


def render_view(ctx_factory, name, args, log=print):
    t0 = time.time()
    W, H, ss = args.width, args.height, args.ss
    Ws, Hs = W * ss, H * ss
    preset = PRESETS.get(name, {})
    stagger = args.stagger if args.stagger is not None else preset.get("stagger", 0.8)
    cam, cam_az = preset_camera(name, args)
    dv = cam.eye - cam.target
    if cam_az is None:
        cam_az = math.degrees(math.atan2(dv[0], dv[2]))
    cam_el = math.degrees(math.atan2(dv[1], math.hypot(dv[0], dv[2])))
    ctx = ctx_factory(stagger, cam_az, cam_el)
    L, m = ctx.L, ctx.m
    margins = (0.075, 0.075, 0.13, 0.28 if args.title else 0.20)
    if name == "top_pair":
        margins = (0.08, 0.08, 0.07, 0.20 if args.title else 0.08)
    cam.fit(L, Ws, Hs, margins, args.zoom)
    eye = cam.eye.astype(np.float32)
    X, Y, D = cam.project(L.P)
    area = tri_screen_area(X, Y, L.T)
    gb = raster_scene(ctx, X, Y, D, Ws, Hs)
    t1 = time.time()
    npx = Ws * Hs
    img, _ = shade_buffers(ctx, gb, eye, area, npx)
    t2 = time.time()

    # ---- floor + backdrop for every pixel without an opaque car surface
    mb, ms = mirror_reflection(ctx, cam, eye, Ws, Hs)
    t3 = time.time()
    bg = np.nonzero(gb[1] < 0)[0]
    for k in range(0, len(bg), 1_500_000):
        idx = bg[k:k + 1_500_000]
        img[idx] = shade_background(ctx, cam, eye, idx, Ws, mb, ms)
    del mb, bg
    # glass (also over the floor seen through the windows)
    composite_glass(ctx, gb, img, eye)
    t4 = time.time()

    # ---- post: bloom, DOF, tone mapping, vignette, downsample
    img = img.reshape(Hs, Ws, 3)
    depth = np.where(gb[1] >= 0, gb[0], np.inf).reshape(Hs, Ws) if args.dof > 0 else None
    del gb
    img = post_process(img, depth, cam, args, ss)
    out = Image.fromarray(img)
    if ss > 1:
        out = out.resize((W, H), Image.LANCZOS)
    t5 = time.time()
    # screen x of each car (for the title band ordering)
    car_x = []
    for c in L.car_centres:
        cx_, _, _ = cam.project(np.asarray(c)[None, :])
        car_x.append(float(cx_[0]) / ss)
    if args.title:
        out = draw_title(out, ctx.liv, car_x, args)
    log(f"{name}: raster {t1 - t0:.0f}s, shade {t2 - t1:.0f}s, mirror {t3 - t2:.0f}s, floor+glass {t4 - t3:.0f}s, "
        f"post {t5 - t4:.0f}s, total {time.time() - t0:.0f}s, peak {_peak_mb():.0f} MB")
    return out


def _peak_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def post_process(img, depth, cam, args, ss):
    Hs, Ws = img.shape[:2]
    # optional depth of field: focus on the nearest car centre
    if args.dof and args.dof > 0:
        zf = float(np.nanpercentile(depth[np.isfinite(depth)], 25)) if np.isfinite(depth).any() else 10.0
        dd = np.where(np.isfinite(depth), depth, zf * 3)
        coc = args.dof * ss * np.abs(1 - zf / np.maximum(dd, 0.1)) * 6.0
        coc = ndimage.maximum_filter(coc, size=3)
        img = blur_stack_select(img, np.minimum(coc, 24.0), levels=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0))
    # bloom from the bright part of the HDR image
    k = 4
    h4, w4 = Hs // k, Ws // k
    small = img[:h4 * k, :w4 * k].reshape(h4, k, w4, k, 3).mean((1, 3))
    lum = small.max(2, keepdims=True)
    bright = small * np.clip((lum - 1.2) / np.maximum(lum, 1e-4), 0, None)
    bloom = np.zeros_like(small)
    for s, wgt in ((1.5 * ss / k * 2, 0.5), (5.0 * ss / k * 2, 0.3), (16.0 * ss / k * 2, 0.2)):
        bloom += wgt * np.stack([ndimage.gaussian_filter(bright[..., c], s) for c in range(3)], -1)
    for c in range(3):
        up = ndimage.zoom(bloom[..., c], (Hs / h4, Ws / w4), order=1)
        img[:up.shape[0], :up.shape[1], c] += 0.10 * up[:Hs, :Ws]
    del bloom, bright, small
    exposure = args.exposure
    # vignette + filmic tone mapping + gamma, in row chunks (memory)
    yy = (np.arange(Hs, dtype=np.float32) + 0.5) / Hs * 2 - 1
    xx = (np.arange(Ws, dtype=np.float32) + 0.5) / Ws * 2 - 1
    out = np.empty((Hs, Ws, 3), np.uint8)
    for r0 in range(0, Hs, 256):
        r1 = min(Hs, r0 + 256)
        r2 = (xx[None, :] ** 2) * 0.75 + (yy[r0:r1, None] ** 2) * 0.9
        vig = 1 - 0.30 * (np.clip(r2, 0, 1.65) / 1.65) ** 1.8
        v = aces(img[r0:r1] * exposure) * vig[..., None]
        out[r0:r1] = np.clip(np.power(np.clip(v, 0, 1), 1 / 2.2) * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


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


def chrome_fill(w, h, y0, y1):
    """Chrome gradient over rows y0..y1 (cap height of the text), clamped outside."""
    y = np.clip((np.arange(h)[:, None] - y0) / max(1, y1 - y0), 0, 1)
    st = [(0, 250), (0.40, 255), (0.50, 150), (0.60, 112), (0.82, 210), (1, 240)]
    v = np.interp(y, [q[0] for q in st], [q[1] for q in st]) * np.ones((1, w))
    rgb = np.stack([v * 0.98 + 5, v * 0.93 + 8, v * 0.97 + 8], -1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")


def _spaced(draw, xy, text, font, fill, spacing):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, anchor="ls")
        x += draw.textlength(ch, font=font) + spacing
    return x


def _spaced_len(draw, text, font, spacing):
    return sum(draw.textlength(ch, font=font) + spacing for ch in text) - spacing


def draw_title(im, liveries, car_x, args):
    """Team band: «КОМАНДА ЭДМ · BUTCHER CHART CHROME» + driver number plates and names (ui_skin.json),
    drivers ordered like the cars on screen."""
    W, H = im.size
    s = min(W / 2400.0, H / 1350.0)
    im = im.convert("RGBA")
    bh = int(150 * s)
    y0 = H - bh
    # dark fade under the band
    grad_h = int(bh * 2.3)
    a = np.clip(np.arange(grad_h) / (grad_h - bh), 0, 1) ** 1.6 * 228
    shade = Image.fromarray(np.repeat(a[:, None], W, 1).astype(np.uint8), "L")
    over = Image.new("RGBA", (W, grad_h), (9, 5, 9, 255)); over.putalpha(shade)
    im.alpha_composite(over, (0, H - grad_h))
    d = ImageDraw.Draw(im)
    pad = int(70 * s)
    # holographic hairline + maroon dashed butcher-chart line
    hl = holo_strip(W - 2 * pad, max(2, int(3 * s))).convert("RGBA")
    im.alpha_composite(hl, (pad, y0))
    dash, gap_ = int(30 * s), int(14 * s)
    yl = y0 + int(14 * s)
    x = pad
    while x < W - pad:
        d.rounded_rectangle([x, yl, min(x + dash, W - pad), yl + max(2, int(4 * s))], radius=max(1, int(2 * s)),
                            fill=(160, 30, 62, 255))
        x += dash + gap_
    yb = y0 + int(bh * 0.70)
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
    tm = Image.new("L", (W, H), 0)
    _spaced(ImageDraw.Draw(tm), (x, yb), TEAM_LINE[1], f_sub, 255, 3 * s)
    ch = chrome_fill(W, H, yb + cap[1], yb + cap[3]); ch.putalpha(tm)
    im.alpha_composite(ch)
    d = ImageDraw.Draw(im)
    # drivers
    f_num = find_font("Unbounded[wght].ttf", 36 * s, "Black", extra_dirs=_FONT_EXTRA)
    f_first = find_font("SofiaSansCondensed-Italic[wght].ttf", 25 * s, "Bold Italic", extra_dirs=_FONT_EXTRA)
    f_last = find_font("SofiaSansCondensed-Italic[wght].ttf", 46 * s, "Black Italic", extra_dirs=_FONT_EXTRA)
    blocks = []
    for i in np.argsort(car_x):
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


# ---------------------------------------------------------------- driver
_JOB = None


def _render_one(view):
    args, factory = _JOB
    im = render_view(factory, view, args, log=lambda s: print(s, flush=True))
    fp = os.path.join(args.out, view + ".png")
    im.save(fp)
    return fp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skins", nargs="+", required=True, help="skin folders (car A, car B, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default=",".join(ALL_VIEWS))
    ap.add_argument("--width", type=int, default=2400); ap.add_argument("--height", type=int, default=1350)
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--title", action="store_true", help="add the team title band")
    ap.add_argument("--gap", type=float, default=2.45, help="distance between the cars, centre to centre (m)")
    ap.add_argument("--stagger", type=float, default=None, help="front-to-back offset between the cars (m)")
    ap.add_argument("--dof", type=float, default=0.0, help="depth of field strength (0 = off, ~1 = shallow)")
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--zoom", type=float, default=1.0, help="scale of the auto framing")
    ap.add_argument("--camera", default=None, metavar="az,el,dist[,tx,ty,tz]", help="custom camera (view «custom»)")
    ap.add_argument("--eye", default=None); ap.add_argument("--target", default=None)
    ap.add_argument("--kn5", default=rr.DEFAULT_KN5)
    ap.add_argument("--no-interior", action="store_true")
    ap.add_argument("--no-ks-detail", action="store_true", help="skip the ksPerPixelMultiMap constant detail colour")
    ap.add_argument("--fonts", default=None, help="extra font folder searched first")
    ap.add_argument("--cache", default=None, help="folder to cache the per-layout visibility (speeds up re-renders)")
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
    # layouts needed by the views (visibility is precomputed once per layout, before forking)
    layouts = {}
    for v in views:
        st = a.stagger if a.stagger is not None else PRESETS.get(v, {}).get("stagger", 0.8)
        key = round(st, 4)
        if key not in layouts:
            layouts[key] = Layout(model, len(liveries), a.gap, st, jobs=a.jobs, log=lambda s: print(s, flush=True),
                                  cache_dir=a.cache)

    def factory(stagger, cam_az, cam_el):
        return ShadeCtx(model, liveries, layouts[round(stagger, 4)], Studio(cam_az, cam_el))

    global _JOB
    _JOB = (a, factory)
    jobs = max(1, min(a.jobs, len(views)))
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as pool:
            paths = pool.map(_render_one, views, chunksize=1)
    else:
        paths = [_render_one(v) for v in views]
    imgs = [Image.open(p) for p in paths]
    title = " · ".join(lv.info["folder"] for lv in liveries)
    rr.contact_sheet(imgs, views, title, cols=2, tw=960, th=540).save(os.path.join(a.out, "sheet.png"))
    print(f"done in {time.time() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
