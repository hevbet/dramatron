#!/usr/bin/env python3
"""Software 3D preview renderer for the Assetto Corsa car smp_audi_rs3_lms (Audi RS3 LMS TCR).

Usage:
    python3 render_rs3.py <skin_dir> <out_dir> [--views side_left,front34_left,...] [--ss 2]
                          [--width 1600 --height 900] [--kn5 path] [--no-interior] [--jobs N]

<skin_dir> must contain Skin.dds (4096) and optionally glass_sticker.dds (1024, RGBA).
Files without overrides fall back to the textures embedded in the .kn5.
Writes <view>.png for each view plus sheet.png (labelled contact sheet).

Pure numpy rasterizer: z-buffer, perspective-correct UV, alpha test for *_sticker / AT
materials, translucent glass pass, lambert + ambient + soft spec, SSAA.

World space = AC/kn5 world after node transforms: x = car LEFT, y = up, z = FRONT.
UV convention (verified against in-game screenshots): tex_x = frac(u) * W, tex_y = frac(v) * H.
"""
import argparse
import io
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kn5 import R  # noqa: E402

DEFAULT_KN5 = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad/kn5/smp_audi_rs3_lms.kn5"
FONT_CANDIDATES = [
    "/home/user/dramatron/livery/rs3_pink/fonts/Oswald-Medium.ttf",
    "/home/user/dramatron/livery/br03/brand/fonts/Exo2-Italic[wght].ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

ALL_VIEWS = ["front34_left", "front34_right", "side_left", "side_right",
             "rear34_left", "rear34_right", "front", "rear", "top"]

# ---------------------------------------------------------------- materials
# kind: 'tex' textured (embedded or overridden), 'flat' colour, 'glass' translucent pass, 'skip'
SKIP_MATS = {"hud", "hud_leds", "display", "DAMAGE_GLASS", "rimblur", "holes", "int_glass", "int_glassblack"}
GLASS_MATS = {"glass": ((18, 20, 24), 0.62), "lights_glass": ((150, 20, 20), 0.45)}
FLAT = {
    "ext_carbon": (26, 26, 28), "ext_plastic": (30, 30, 32), "pl_black": (22, 22, 24), "black": (18, 18, 20),
    "reshotka": (20, 20, 22), "intplastic": (34, 34, 36), "ext_rubber": (16, 16, 16), "glassblack": (10, 10, 12),
    "chrome": (175, 175, 180), "ext_metal": (34, 34, 36), "exhaust": (110, 105, 100), "screw": (90, 90, 90),
    "screw.001": (90, 90, 90), "rim": (150, 152, 158), "ture": (24, 24, 24), "brakedisc": (70, 70, 72),
    "caliper": (170, 30, 30), "lights": (205, 205, 210), "reflector": (170, 170, 175), "mirror": (150, 160, 170),
}
INTERIOR_COL = (48, 48, 50)
TEX_OK = {"skin", "glass_sticker", "ext_sticker", "rim", "ture", "lights", "reflector", "caliper"}   # use textures for these
AT_MATS = {"glass_sticker", "ext_sticker"}
SPEC = {"skin": 0.35, "glass_sticker": 0.3, "ext_sticker": 0.25, "chrome": 0.6, "rim": 0.35, "lights": 0.5}
DEPTH_BIAS = {"glass_sticker": 0.004, "ext_sticker": 0.004}


# ---------------------------------------------------------------- kn5 with transforms
def load_scene(path, interior=True):
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
        for _ in range(r.i()):
            r.s(); r.f(); r.raw(36)
        samplers = {}
        for _ in range(r.i()):
            sn = r.s(); r.i(); samplers[sn] = r.s()
        mats.append(dict(name=name, shader=shader, samplers=samplers))
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
                p = v[:, 0:3].astype(np.float64)
                pw = p @ M[:3, :3] + M[3, :3]
                meshes.append(dict(name=full, pos=pw, uv=v[:, 6:8].astype(np.float64),
                                   idx=idx.astype(np.int64), mat=mats[mid]["name"],
                                   interior="COCKPIT" in full.upper()))
        for _ in range(nch):
            node(M, full, skip)

    node(np.eye(4), "", False)
    return textures, mats, meshes


def in_car_bounds(p):
    lo, hi = p.min(0), p.max(0)
    return (lo[0] > -1.2 and hi[0] < 1.2 and lo[1] > -0.15 and hi[1] < 1.9 and lo[2] > -2.7 and hi[2] < 2.7)


# ---------------------------------------------------------------- textures
def decode(data):
    try:
        return Image.open(io.BytesIO(data))
    except Exception:
        return None


def tex_array(img, rgba=False):
    img = img.convert("RGBA" if rgba else "RGB")
    return np.ascontiguousarray(np.asarray(img))


def sample(tex, u, v, bilinear=True):
    """tex HxWxC uint8; u,v arrays (wrapped). returns float32 (N,C)."""
    H, W = tex.shape[:2]
    x = np.mod(u, 1.0) * W - 0.5
    y = np.mod(v, 1.0) * H - 0.5
    if not bilinear:
        xi = np.clip(np.round(x).astype(np.int64), 0, W - 1)
        yi = np.clip(np.round(y).astype(np.int64), 0, H - 1)
        return tex[yi, xi].astype(np.float32)
    x0 = np.floor(x); y0 = np.floor(y)
    fx = (x - x0).astype(np.float32)[:, None]; fy = (y - y0).astype(np.float32)[:, None]
    x0 = x0.astype(np.int64) % W; y0 = y0.astype(np.int64) % H
    x1 = (x0 + 1) % W; y1 = (y0 + 1) % H
    a = tex[y0, x0].astype(np.float32); b = tex[y0, x1].astype(np.float32)
    c = tex[y1, x0].astype(np.float32); d = tex[y1, x1].astype(np.float32)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


# ---------------------------------------------------------------- scene assembly
class Scene:
    def __init__(self, kn5_path, skin_dir, interior=True):
        textures, mats, meshes = load_scene(kn5_path, interior)
        self.mat_names = []
        mat_id = {}
        P, UV, T, TM = [], [], [], []
        off = 0
        for m in meshes:
            mn = m["mat"]
            if mn in SKIP_MATS:
                continue
            if not in_car_bounds(m["pos"]):
                continue
            key = "__interior__" if (m["interior"] and mn not in GLASS_MATS) else mn
            if key not in mat_id:
                mat_id[key] = len(self.mat_names); self.mat_names.append(key)
            P.append(m["pos"]); UV.append(m["uv"]); T.append(m["idx"] + off)
            TM.append(np.full(len(m["idx"]), mat_id[key], np.int32))
            off += len(m["pos"])
        self.P = np.concatenate(P); self.UV = np.concatenate(UV)
        self.T = np.concatenate(T); self.TM = np.concatenate(TM)
        # face normals (world)
        a, b, c = self.P[self.T[:, 0]], self.P[self.T[:, 1]], self.P[self.T[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        self.N = n / np.maximum(ln, 1e-12)
        self.tri_ok = ln[:, 0] > 1e-12
        # textures per material
        tex_by_name = {}
        for k, mat in enumerate(mats):
            tex_by_name[mat["name"]] = mat["samplers"].get("txDiffuse")
        override = {}
        for fn in ("Skin.dds", "glass_sticker.dds"):
            fp = os.path.join(skin_dir, fn)
            if os.path.exists(fp):
                override[fn.lower()] = Image.open(fp)
        self.kind, self.tex, self.flat, self.spec, self.bias, self.galpha = [], [], [], [], [], []
        for name in self.mat_names:
            kind, tex, flat = "flat", None, FLAT.get(name, (40, 40, 42))
            if name == "__interior__":
                flat = INTERIOR_COL
            elif name in GLASS_MATS:
                kind = "glass"; flat = GLASS_MATS[name][0]
            elif name in TEX_OK:
                tn = tex_by_name.get(name)
                img = None
                if tn:
                    img = override.get(tn.lower())
                    if img is None and tn in textures:
                        img = decode(textures[tn])
                if img is not None:
                    kind = "tex"; tex = tex_array(img, rgba=name in AT_MATS)
            self.kind.append(kind); self.tex.append(tex); self.flat.append(flat)
            self.spec.append(SPEC.get(name, 0.15)); self.bias.append(DEPTH_BIAS.get(name, 0.0))
            self.galpha.append(GLASS_MATS.get(name, (None, 0))[1])
        self.is_glass = np.array([k == "glass" for k in self.kind])
        self.is_at = np.array([n in AT_MATS for n in self.mat_names])
        self.bias = np.array(self.bias)
        self.spec = np.array(self.spec, np.float32)


# ---------------------------------------------------------------- camera
def look_at(eye, target, up_hint=(0, 1, 0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    rgt = np.cross(f, up_hint); rgt /= np.linalg.norm(rgt)
    up = np.cross(rgt, f)
    return eye, f, rgt, up


CENTER = np.array([0.0, 0.62, -0.02])


def view_camera(name):
    D = 9.0
    c = CENTER
    if name == "side_left":
        return look_at(c + [D, 0.25, 0], c)
    if name == "side_right":
        return look_at(c + [-D, 0.25, 0], c)
    if name in ("front34_left", "front34_right", "rear34_left", "rear34_right"):
        sx = 1 if name.endswith("left") else -1
        sz = 1 if name.startswith("front") else -1
        ang = np.radians(38)
        el = np.radians(24)
        d = np.array([sx * np.sin(ang) * np.cos(el), np.sin(el), sz * np.cos(ang) * np.cos(el)])
        return look_at(c + d * D, c)
    if name == "front":
        return look_at(c + [0, 2.2, D], c)
    if name == "rear":
        return look_at(c + [0, 2.2, -D], c)
    if name == "top":
        return look_at(c + [0, D * 1.3, 0], c, up_hint=(1, 0, 0))   # front of car -> image right
    raise ValueError(name)


# ---------------------------------------------------------------- rasterizer
def rasterize(X, Y, Dv, tris, tri_bias, W, H, at_filter=None, max_frag=3_000_000):
    """X,Y screen px (pixel centre at i+0.5), Dv camera depth per vertex.
    Returns zbuf (H*W), tri id (-1 none), perspective-correct barycentrics b1,b2."""
    npx = W * H
    zbuf = np.full(npx, np.inf); tid = np.full(npx, -1, np.int64)
    B1 = np.zeros(npx, np.float32); B2 = np.zeros(npx, np.float32)
    if len(tris) == 0:
        return zbuf, tid, B1, B2
    i0, i1, i2 = tris[:, 0], tris[:, 1], tris[:, 2]
    x0, x1, x2 = X[i0], X[i1], X[i2]
    y0, y1, y2 = Y[i0], Y[i1], Y[i2]
    d0, d1, d2 = Dv[i0], Dv[i1], Dv[i2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    xmin = np.floor(np.minimum(np.minimum(x0, x1), x2) - 0.5).astype(np.int64) + 1
    xmax = np.floor(np.maximum(np.maximum(x0, x1), x2) - 0.5).astype(np.int64)
    ymin = np.floor(np.minimum(np.minimum(y0, y1), y2) - 0.5).astype(np.int64) + 1
    ymax = np.floor(np.maximum(np.maximum(y0, y1), y2) - 0.5).astype(np.int64)
    xmin = np.maximum(xmin, 0); ymin = np.maximum(ymin, 0)
    xmax = np.minimum(xmax, W - 1); ymax = np.minimum(ymax, H - 1)
    ok = (np.abs(area) > 1e-9) & (xmax >= xmin) & (ymax >= ymin) & (np.minimum(np.minimum(d0, d1), d2) > 0.05)
    ids = np.nonzero(ok)[0]
    size = np.maximum(xmax - xmin, ymax - ymin)[ids] + 1
    S_bins = 2 ** np.ceil(np.log2(np.maximum(size, 1))).astype(np.int64)
    for S in np.unique(S_bins):
        sel = ids[S_bins == S]
        oy, ox = np.mgrid[0:S, 0:S]
        ox = ox.ravel(); oy = oy.ravel()
        per = max(1, max_frag // (S * S))
        for k in range(0, len(sel), per):
            t = sel[k:k + per]
            px = xmin[t][:, None] + ox[None, :]
            py = ymin[t][:, None] + oy[None, :]
            inb = (px <= xmax[t][:, None]) & (py <= ymax[t][:, None])
            fx = px + 0.5; fy = py + 0.5
            A = area[t][:, None]
            w0 = ((x1[t][:, None] - fx) * (y2[t][:, None] - fy) - (x2[t][:, None] - fx) * (y1[t][:, None] - fy)) / A
            w1 = ((x2[t][:, None] - fx) * (y0[t][:, None] - fy) - (x0[t][:, None] - fx) * (y2[t][:, None] - fy)) / A
            w2 = 1.0 - w0 - w1
            eps = -1e-6
            m = inb & (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
            if not m.any():
                continue
            r_, c_ = np.nonzero(m)
            tt = t[r_]
            w0 = w0[r_, c_]; w1 = w1[r_, c_]; w2 = w2[r_, c_]
            iz = w0 / d0[tt] + w1 / d1[tt] + w2 / d2[tt]
            z = 1.0 / iz - tri_bias[tt]
            pb1 = (w1 / d1[tt]) / iz; pb2 = (w2 / d2[tt]) / iz
            pix = py[r_, c_] * W + px[r_, c_]
            if at_filter is not None:
                keep = at_filter(tt, pb1, pb2)
                if not keep.all():
                    pix, z, tt, pb1, pb2 = pix[keep], z[keep], tt[keep], pb1[keep], pb2[keep]
                    if len(pix) == 0:
                        continue
            o = np.lexsort((z, pix))
            pix = pix[o]; z = z[o]
            first = np.ones(len(pix), bool); first[1:] = pix[1:] != pix[:-1]
            pix = pix[first]; z = z[first]; o = o[first]
            better = z < zbuf[pix]
            pix = pix[better]; o = o[better]
            zbuf[pix] = z[better]; tid[pix] = tt[o]
            B1[pix] = pb1[o]; B2[pix] = pb2[o]
    return zbuf, tid, B1, B2


# ---------------------------------------------------------------- render one view
def render_view(sc, name, W=1600, H=900, ss=2, margin=0.07):
    eye, f, rgt, up = view_camera(name)
    rel = sc.P - eye
    Dv = rel @ f
    cx_ = rel @ rgt / Dv; cy_ = rel @ up / Dv
    # auto-fit: use exterior (non-interior) vertices
    used = np.unique(sc.T.ravel())
    ux, uy = cx_[used], cy_[used]
    lo_x, hi_x = ux.min(), ux.max(); lo_y, hi_y = uy.min(), uy.max()
    Ws, Hs = W * ss, H * ss
    fpx = min(Ws * (1 - 2 * margin) / (hi_x - lo_x), Hs * (1 - 2 * margin) / (hi_y - lo_y))
    mx, my = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
    X = (cx_ - mx) * fpx + Ws / 2
    Y = -(cy_ - my) * fpx + Hs / 2

    UVs = sc.UV
    tm = sc.TM

    def bary_uv(tt, b1, b2):
        tri = sc.T[tt]
        b0 = 1.0 - b1 - b2
        uv = UVs[tri[:, 0]] * b0[:, None] + UVs[tri[:, 1]] * b1[:, None] + UVs[tri[:, 2]] * b2[:, None]
        return uv

    def at_filter(tt, b1, b2):
        keep = np.ones(len(tt), bool)
        mt = tm[tt]
        atm = sc.is_at[mt]
        if atm.any():
            ii = np.nonzero(atm)[0]
            uv = bary_uv(tt[ii], b1[ii], b2[ii])
            for mid in np.unique(mt[ii]):
                jj = ii[mt[ii] == mid]
                tex = sc.tex[mid]
                if tex is None or tex.shape[2] < 4:
                    continue
                a = sample(tex, uv[mt[ii] == mid, 0], uv[mt[ii] == mid, 1], bilinear=False)[:, 3]
                keep[jj] = a >= 128
        return keep

    opaque = ~sc.is_glass[tm] & sc.tri_ok
    glass = sc.is_glass[tm] & sc.tri_ok
    tri_bias = sc.bias[tm]
    gid_op = np.nonzero(opaque)[0]
    zb, tid, B1, B2 = rasterize(X, Y, Dv, sc.T[opaque], tri_bias[opaque], Ws, Hs,
                                lambda tt, b1, b2: at_filter(gid_op[tt], b1, b2))
    # map local ids back to global triangle ids
    tid = np.where(tid >= 0, gid_op[np.maximum(tid, 0)], -1)
    zg, tidg, G1, G2 = rasterize(X, Y, Dv, sc.T[glass], tri_bias[glass], Ws, Hs, None)
    gid_gl = np.nonzero(glass)[0]
    tidg = np.where(tidg >= 0, gid_gl[np.maximum(tidg, 0)], -1)

    # background: dark studio gradient
    yy = np.linspace(0, 1, Hs, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, Ws, dtype=np.float32)[None, :]
    bg = 58 - 26 * yy - 10 * xx ** 2
    img = np.repeat(np.broadcast_to(bg, (Hs, Ws))[..., None], 3, axis=2).astype(np.float32).copy()
    # soft ground shadow
    if name != "top":
        foot = np.array([[1.0, 0.0, 2.35], [-1.0, 0.0, 2.35], [-1.0, 0.0, -2.35], [1.0, 0.0, -2.35]])
        r2 = foot - eye; d2 = r2 @ f
        sxs = (r2 @ rgt / d2 - mx) * fpx + Ws / 2; sys_ = -(r2 @ up / d2 - my) * fpx + Hs / 2
        sh = Image.new("L", (Ws, Hs), 0)
        ImageDraw.Draw(sh).polygon(list(zip(sxs.tolist(), sys_.tolist())), fill=200)
        sh = sh.filter(ImageFilter.GaussianBlur(18 * ss))
        img *= (1 - 0.75 * np.asarray(sh, np.float32)[..., None] / 255)
    flat_img = img.reshape(-1, 3)

    key = rgt * -0.45 + np.array([0, 1.0, 0]) * 0.85 - f * 0.55
    key /= np.linalg.norm(key)
    fill = -rgt * -0.45 + np.array([0, 0.3, 0]) - f * 0.6
    fill /= np.linalg.norm(fill)

    def shade(pix, tt, b1, b2, glass_pass=False):
        mt = tm[tt]
        n = sc.N[tt]
        # view vector
        tri = sc.T[tt]
        b0 = 1 - b1 - b2
        pos = sc.P[tri[:, 0]] * b0[:, None] + sc.P[tri[:, 1]] * b1[:, None] + sc.P[tri[:, 2]] * b2[:, None]
        vdir = eye - pos; vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)
        flip = (n * vdir).sum(1) < 0
        n = np.where(flip[:, None], -n, n)
        lam = np.clip(n @ key, 0, 1) * 0.75 + np.clip(n @ fill, 0, 1) * 0.25
        sky = 0.5 + 0.5 * n[:, 1]
        light = 0.32 + 0.18 * sky + 0.75 * lam
        h = key[None, :] + vdir; h /= np.linalg.norm(h, axis=1, keepdims=True)
        spec = np.clip((n * h).sum(1), 0, 1) ** 40
        fres = (1 - np.clip((n * vdir).sum(1), 0, 1)) ** 4
        col = np.zeros((len(tt), 3), np.float32)
        for mid in np.unique(mt):
            jj = mt == mid
            if sc.kind[mid] == "tex":
                uv = bary_uv(tt[jj], b1[jj], b2[jj])
                col[jj] = sample(sc.tex[mid], uv[:, 0], uv[:, 1])[:, :3]
            else:
                col[jj] = sc.flat[mid]
        ks = sc.spec[mt]
        out = col * light[:, None] + (255 * (ks * spec + 0.25 * ks * fres))[:, None]
        return np.clip(out, 0, 255)

    hit = np.nonzero(tid >= 0)[0]
    if len(hit):
        flat_img[hit] = shade(hit, tid[hit], B1[hit], B2[hit])
    gh = np.nonzero((tidg >= 0) & (zg < zb))[0]
    if len(gh):
        gc = shade(gh, tidg[gh], G1[gh], G2[gh], True)
        al = np.array(sc.galpha, np.float32)[tm[tidg[gh]]][:, None]
        flat_img[gh] = flat_img[gh] * (1 - al) + gc * al
        # reflections on glass: add spec-only highlight already included in gc via alpha
    img = flat_img.reshape(Hs, Ws, 3)
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if ss > 1:
        out = out.resize((W, H), Image.LANCZOS)
    return out


def get_font(size):
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def contact_sheet(images, names, title, cols=3, tw=800, th=450):
    rows = (len(images) + cols - 1) // cols
    pad, head = 10, 56
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad, head + rows * (th + pad) + pad), (24, 24, 26))
    d = ImageDraw.Draw(sheet)
    d.text((pad + 4, 12), title, fill=(235, 235, 235), font=get_font(30))
    lf = get_font(22)
    for k, (im, n) in enumerate(zip(images, names)):
        r_, c_ = divmod(k, cols)
        x = pad + c_ * (tw + pad); y = head + r_ * (th + pad)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        d.rectangle([x, y, x + 12 + d.textlength(n, font=lf), y + 32], fill=(0, 0, 0))
        d.text((x + 6, y + 3), n, fill=(255, 255, 255), font=lf)
    return sheet


_JOB = None


def _render_one(v):
    sc, W, H, ss, out_dir = _JOB
    t1 = time.time()
    im = render_view(sc, v, W, H, ss)
    im.save(os.path.join(out_dir, v + ".png"))
    print(f"{v}: {time.time() - t1:.1f}s", flush=True)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("skin_dir"); ap.add_argument("out_dir")
    ap.add_argument("--views", default=",".join(ALL_VIEWS))
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor (1=fast, 2=default)")
    ap.add_argument("--width", type=int, default=1600); ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--kn5", default=DEFAULT_KN5)
    ap.add_argument("--no-interior", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel processes (one view each)")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    sc = Scene(a.kn5, a.skin_dir, interior=not a.no_interior)
    print(f"scene: {len(sc.T)} tris, {len(sc.mat_names)} materials, {time.time() - t0:.1f}s", flush=True)
    views = [v for v in a.views.split(",") if v]
    global _JOB
    _JOB = (sc, a.width, a.height, a.ss, a.out_dir)
    jobs = max(1, min(a.jobs, len(views)))
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as pool:
            imgs = pool.map(_render_one, views, chunksize=1)
    else:
        imgs = [_render_one(v) for v in views]
    title = a.title or os.path.basename(os.path.normpath(a.skin_dir))
    contact_sheet(imgs, views, title).save(os.path.join(a.out_dir, "sheet.png"))
    print(f"done in {time.time() - t0:.1f}s -> {a.out_dir}")


if __name__ == "__main__":
    main()
