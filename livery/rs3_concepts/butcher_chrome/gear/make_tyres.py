#!/usr/bin/env python3
"""Pink sidewall lettering for the «Butcher Chart Chrome» skins: SIMKARTING.RU / KARTING64.RU on the tyres.

Writes skin-folder overrides of the two diffuse textures of the tyre material «ture» (shader ksTyres):
    out/<skin>/tyre_d.dds        2048x2048 DXT5, 12 mips  (txDiffuse = txDirty)
    out/<skin>/tyre_d_blur.dds   1024x1024 DXT5, 11 mips  (txBlur, used while the wheel spins)
    identical files for Pozdnyakov_23 and Konopelko_00, plus the preview gear/tyre_preview.png.

Everything is measured from the .kn5 at run time (and printed); the numbers below are what it finds.

Sidewall ring in UV (texel units of the 2048 map, centre (1024, 1024) = UV (0.5, 0.5)):
  * mesh: the sidewall is a 60-segment ring from r = 813.5 (bead, world radius 252.0 mm) through 871.4 / 932.5 /
    980.5 / 993.6 to r = 1013.0 (shoulder, 307.6 mm). The UV radius is NOT proportional to the world radius
    (871.4-932.5 carries 4071 texels per metre radially but 3270 per metre around the wheel), so the lettering is
    laid out in world millimetres and mapped to UV through that piecewise-linear profile -> no radial stretch.
  * texture: painted ring r = 811 .. 1014; moulded bead double line at r = 829 / 836, shoulder double lines at
    975.5 / 979 and 991.5 / 995. The moulded band for lettering lies between them: r = 838 .. 973
    (world 258.8 .. 295.6 mm, 36.9 mm tall); 16 radial mould seams every 22.5 degrees.
  * stock lettering (dark moulded print, also embossed in tyre_nm.dds): «FOR COMPETITION USE ONLY / 240/610R17»
    at theta = 206..245 deg and «RADIAL / A-005 23100-1 MADE IN JAPAN» at 54..83 deg (theta measured in the image,
    clockwise from +x, y down). Glyph tops face OUTWARD and the text runs CLOCKWISE (increasing theta).
Mesh orientation (all four tyres share the UVs):
  * every OUTER sidewall maps the ring unmirrored (det of UV->screen as seen from that side > 0 on 600/600
    sidewall triangles); every inner sidewall is the mirrored copy, which reads correctly from inside the car;
  * left wheels show the texture as is (0 deg), right wheels rotated by 180 deg, so a block in the upper half of
    the map stands upright on top of the left wheels at rest and the opposite block on top of the right wheels.
Layout: the two blocks are diametrically opposite, glyph tops outward, reading clockwise (as the stock print),
each glyph rigidly rotated to the arc (Exo 2 ExtraBold Italic, the face of the URLs on the body). They sit in the
two arcs that are free of stock lettering (centred to maximise the clearance), so no stock print is covered: the
stock print is also embossed in tyre_nm.dds, which is not overridden, and pink paint over it would show its relief.
Four blocks would not fit between the stock print, so two blocks is the layout (alternating with the stock print).
Colour: the body pink PIG (242,158,178) itself, same hue and saturation (no desaturation), scaled UP by GAIN
so that the brightest mottled texel lands exactly on 255 in red (PAINT ~ (250,163,184), +3.3 %; the
hue-preserving ceiling is +5.4 %). Reason: the kn5 lights the tyre material «ture» (ksTyres) with
ksAmbient 0.25 / ksDiffuse 0.30, the body «skin» with 0.45 / 0.50, so the same texel shows at only 56-60 % of the
body's brightness in game (the software renderer lights every material alike and hides this). Any darkening in
the texture would stack on top of that, so the print is kept at the brightest exact-hue pink a texel can hold;
the remaining in-game gap cannot be closed from the texture (it would need ~1.7x PIG, far above 255).
The rubber mottling (TONE) is only a +-2 % shimmer and the grain breakup (GRAIN, BLOT) only thins the paint on
the darkest specks, so the solid-lettering mean stays above PIG (printed: ~1.02-1.03 x PIG).
Alpha: the DXT5 alpha half of every block is copied from the original file (all levels identical).
Only the 4x4 blocks that change are re-encoded; every other block is byte-identical to the original.
tyre_d_blur.dds is the angular (motion) blurred sidewall: it gets a soft pink ring at the lettering radius whose
opacity is the angular mean of the lettering coverage (what the spinning lettering averages to), mixed in
linear light (peak ~0.19).

Run:  python3 make_tyres.py [--out DIR] [--preview PNG] [--renders DIR]
      (--renders: a folder with wheel close-ups wheel_LF.png ... and full/side_*.png to add to the preview;
       --skins "" rebuilds only the preview)
"""
import argparse
import io
import math
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = "/home/user/dramatron/livery/tools"
sys.path.insert(0, TOOLS)
import render_rs3  # noqa: E402

KN5 = render_rs3.DEFAULT_KN5
SKINS = ("Pozdnyakov_23", "Konopelko_00")
TEX, TEX_BLUR = "tyre_d.dds", "tyre_d_blur.dds"
FONT = "/home/user/dramatron/livery/fonts/Exo2-Italic[wght].ttf"
FONT_VAR = b"ExtraBold Italic"
TEXTS = ("SIMKARTING.RU", "KARTING64.RU")      # block 0 in the upper half of the map, block 1 opposite

CAP_MM = 25.0          # cap height (world mm) - 68 % of the 36.9 mm moulded band
TRACK = 0.07           # extra letter spacing, fraction of the cap height
SPM = 16000.0          # glyph raster resolution, px per metre (~4.9x the texel density)
SS = 4                 # supersampling per texel axis
CLEAR_MIN = 8.0        # min clearance (deg) between a block and the stock print
MOTT = (0.8, 1.2)      # clip range of the rubber mottling ratio (local / median rubber luma)
TONE = 0.1             # strength of the rubber mottling on the paint (tone 0.98..1.02, symmetric)
GRAIN = 0.20           # max paint loss on the darkest grain specks
BLOT = 0.05            # max paint loss on the darkest rubber clouds

# palette (butcher_chrome/make_skin.py): PIG (242,158,178) body pink. The print is that exact hue and saturation,
# brightened (never darkened): ksTyres 0.25/0.30 vs the body's 0.45/0.50 already dims it to ~0.58x in game.
PIG = np.array((242, 158, 178), np.float32)
GAIN = 255.0 / (float(PIG.max()) * (1 + TONE * (MOTT[1] - 1)))   # brightest tone -> R = 255 exactly (1.033)
PAINT = PIG * GAIN                                                # ~ (250, 163, 184)
BLUR_GAIN = 1.0        # blur ring opacity = BLUR_GAIN x angular mean coverage
BLUR_SOFT = 6.0        # radial softening of the blur ring (sigma, 2048 texels = 3 texels of the 1024 map)


# ---------------------------------------------------------------- DDS helpers
def dds_info(data):
    hh = struct.unpack_from("<31I", data, 4)
    assert data[:4] == b"DDS " and data[84:88] == b"DXT5", data[84:88]
    return hh[3], hh[2], max(1, hh[6])          # W, H, mips


def level_layout(data):
    W, H, mips = dds_info(data)
    off, out = 128, []
    for lv in range(mips):
        w, h = max(1, W >> lv), max(1, H >> lv)
        bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
        out.append((w, h, bw, bh, off))
        off += bw * bh * 16
    assert off == len(data), (off, len(data))
    return out


def decode_levels(data):
    """RGBA uint8 of every mip level (Pillow decodes the top level only -> re-wrap each level)."""
    levels = []
    for w, h, bw, bh, off in level_layout(data):
        hdr = bytearray(data[:128])
        struct.pack_into("<I", hdr, 12, bh * 4); struct.pack_into("<I", hdr, 16, bw * 4)
        struct.pack_into("<I", hdr, 28, 1)
        im = Image.open(io.BytesIO(bytes(hdr) + data[off:off + bw * bh * 16])).convert("RGBA")
        levels.append(np.asarray(im)[:h, :w].copy())
    return levels


def encode_blocks(rgba):
    """DXT5 blocks (bw*bh*16 bytes, row-major) of one level with Pillow's BC3 encoder (edge-padded to 4x4)."""
    h, w = rgba.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4
    a = np.pad(rgba, ((0, ph), (0, pw), (0, 0)), mode="edge") if (ph or pw) else rgba
    bio = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a), "RGBA").save(bio, format="DDS", pixel_format="DXT5")
    raw = bio.getvalue()
    assert raw[84:88] == b"DXT5"
    body = raw[128:]
    assert len(body) == (a.shape[0] // 4) * (a.shape[1] // 4) * 16
    return body


def box_down(x, f):
    if f == 1:
        return x
    h, w = x.shape[0] // f, x.shape[1] // f
    return x.reshape(h, f, w, f, -1).mean((1, 3))


def write_patched(orig, new0_rgb, path, thr=0.75):
    """New DDS with the ORIGINAL header and level layout. Level k colour = original level k + box-filtered
    (new0 - original0); only blocks whose colour changes by >= thr are re-encoded (colour half only); the
    alpha half of every block and every unchanged block stay byte-identical to the original."""
    lv_orig = decode_levels(orig)
    lay = level_layout(orig)
    W = lay[0][0]
    delta0 = new0_rgb.astype(np.float32) - lv_orig[0][..., :3].astype(np.float32)
    out = bytearray(orig)
    stats = []
    for k, ((w, h, bw, bh, off), lo) in enumerate(zip(lay, lv_orig)):
        d = box_down(delta0, W // w)
        tgt = np.clip(lo[..., :3].astype(np.float32) + d, 0, 255)
        ch = np.abs(d).max(-1) >= thr
        ph, pw = bh * 4 - h, bw * 4 - w
        chb = np.pad(ch, ((0, ph), (0, pw))).reshape(bh, 4, bw, 4).any((1, 3))
        n = int(chb.sum())
        if n:
            rgba = np.dstack([np.clip(tgt + 0.5, 0, 255).astype(np.uint8), lo[..., 3]])
            enc = encode_blocks(rgba)
            for bi in np.flatnonzero(chb.ravel()):
                o = off + bi * 16
                out[o + 8:o + 16] = enc[bi * 16 + 8:bi * 16 + 16]          # colour half only
        stats.append((w, n, bw * bh))
    with open(path + ".tmp", "wb") as fh:
        fh.write(out)
    os.replace(path + ".tmp", path)
    return stats


# ---------------------------------------------------------------- measurements
def load_kn5():
    tex, mats, meshes = render_rs3.load_scene(KN5, interior=False)
    mat = [m for m in mats if m["name"] == "ture"][0]
    tyres = [m for m in meshes if m["mat"] == "ture"]
    assert len(tyres) == 4, len(tyres)
    for t in tyres[1:]:
        assert np.allclose(t["uv"], tyres[0]["uv"]) and np.array_equal(t["idx"], tyres[0]["idx"])
    return tex, mat, tyres


def wheel_tag(m):
    return [p for p in m["name"].split("/") if p.startswith("WHEEL_")][0][6:]


def mesh_analysis(tyres, size=2048):
    """Ring profile (UV radius -> world radius) and the orientation of each wheel's outer sidewall."""
    info, prof = [], None
    for m in tyres:
        P, T, UV = m["pos"], m["idx"], m["uv"]
        tx, ty = np.mod(UV[:, 0], 1) * size, np.mod(UV[:, 1], 1) * size
        c = (P.min(0) + P.max(0)) / 2
        side = 1.0 if c[0] > 0 else -1.0          # x = car LEFT
        ru = np.hypot(tx - size / 2, ty - size / 2)
        rw = np.hypot(P[:, 1] - c[1], P[:, 2] - c[2])
        xo = (P[:, 0] - c[0]) * side
        # screen coordinates seen from outside that side (x right, y down): left side camera looks -x, right +x
        sx = -(P[:, 2] - c[2]) if side > 0 else (P[:, 2] - c[2])
        sy = -(P[:, 1] - c[1])
        ring = ru > 800
        a, b, cc = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        nrm = np.cross(b - a, cc - a)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        res = {}
        for lab, sgn in (("outer", 1), ("inner", -1)):
            sel = np.flatnonzero((nrm[:, 0] * side * sgn > 0.5) & ring[T].all(1))
            q = np.stack([tx[T[sel]], ty[T[sel]]], -1)
            s = np.stack([sx[T[sel]], sy[T[sel]]], -1)
            dq = np.stack([q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]], -1)
            ds = np.stack([s[:, 1] - s[:, 0], s[:, 2] - s[:, 0]], -1)
            det = np.linalg.det(ds) / np.linalg.det(dq)
            res[lab] = (int((det > 0).sum()), int((det < 0).sum()))
        vo = ring & (xo > 0.05)
        au = np.arctan2(ty[vo] - size / 2, tx[vo] - size / 2)
        aw = np.arctan2(sy[vo], sx[vo])
        rot = math.degrees(np.angle(np.exp(1j * (aw - au)).mean()))
        info.append((wheel_tag(m), res, rot))
        if prof is None:
            order = np.argsort(ru[vo])
            rs = ru[vo][order]
            grp = np.concatenate([[0], np.cumsum(np.diff(rs) > 2.0)])       # rings: radii within 2 texels
            sel = np.flatnonzero(vo)[order]
            prof = [(float(ru[sel][grp == g].mean()), float(rw[sel][grp == g].mean()),
                     float(xo[sel][grp == g].mean())) for g in np.unique(grp)]
    return info, np.array(prof)


def polar(img, r0, r1, dr=0.5, dth=0.25, size=2048):
    """Polar unwrap: rows = radius r1 -> r0 (outer at the top), cols = theta 0..360 (deg, clockwise from +x)."""
    k = img.shape[0] / size
    th = np.radians(np.arange(0, 360, dth))
    r = np.arange(r1, r0, -dr)
    TT, RR = np.meshgrid(th, r)
    x = img.shape[1] / 2 + RR * k * np.cos(TT) - 0.5
    y = img.shape[0] / 2 + RR * k * np.sin(TT) - 0.5
    if img.ndim == 2:
        return ndimage.map_coordinates(img.astype(np.float32), [y, x], order=1), r
    return np.stack([ndimage.map_coordinates(img[..., i].astype(np.float32), [y, x], order=1)
                     for i in range(img.shape[2])], -1), r


def texture_ring(L):
    """Painted ring edges, moulded grooves and the lettering band from the radial profile of the diffuse."""
    pu, r = polar(L, 790, 1022, dr=0.5)
    med = np.median(pu, 1)
    on = np.flatnonzero(med > 20)
    edge_out, edge_in = r[on[0]], r[on[-1]]
    groove = (med < 30) & (med > -1)
    groove[: on[0]] = groove[on[-1] + 1:] = False
    runs, cur = [], []
    for i in np.flatnonzero(groove):
        if cur and i != cur[-1] + 1:
            runs.append(cur); cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    grooves = sorted(float(r[c].mean()) for c in runs)              # radii of the moulded lines
    grooves = [g for g in grooves if edge_in + 3 < g < edge_out - 3]  # not the antialiased ring edges
    mid = (edge_in + edge_out) / 2
    bead = [g for g in grooves if g < mid]
    shoulder = [g for g in grooves if g > mid]
    band = (max(bead) + 2.0, min(shoulder) - 2.0)                   # clear of the line edges
    return dict(edge=(float(edge_in), float(edge_out)), grooves=grooves, band=band)


def stock_zones(L, band):
    """Angular extents (deg) of the stock print inside the band; the 16 radial mould seams are ignored."""
    pu, r = polar(L, band[0], band[1], dr=0.5, dth=0.25)
    loc = ndimage.median_filter(pu, size=(1, 41), mode="wrap")
    dark = pu < loc - 10
    frac = dark.mean(0)
    seam = frac > 0.45                                               # thin full-height radial lines
    seam = ndimage.binary_dilation(seam, iterations=2)
    col = (dark.sum(0) >= 2) & ~seam
    th = np.arange(0, 360, 0.25)
    zones, cur = [], None
    for i in np.flatnonzero(col):
        t = th[i]
        if cur and t - cur[1] <= 4.0:
            cur[1] = t; cur[2] += int(dark[:, i].sum())
        else:
            if cur:
                zones.append(cur)
            cur = [t, t, int(dark[:, i].sum())]
    if cur:
        zones.append(cur)
    if len(zones) > 1 and zones[0][0] < 4.0 and zones[-1][1] > 356.0:     # wrap at 0/360
        z = zones.pop(); zones[0] = [z[0] - 360, zones[0][1], zones[0][2] + z[2]]
    return [(a, b) for a, b, n in zones if n > 150]


# ---------------------------------------------------------------- lettering
def get_font(px):
    f = ImageFont.truetype(FONT, int(round(px)))
    f.set_variation_by_name(FONT_VAR)
    return f


def glyphs(text, cap_px, track_px):
    """Per-glyph rasters: (img (float 0..1), x_ref, y_ref, centre along the line) + total line length (px).
    (x_ref, y_ref) = raster position of the glyph's advance centre at mid cap height."""
    f0 = get_font(1000)
    capr = -f0.getbbox("H", anchor="ls")[1] / 1000.0
    f = get_font(cap_px / capr)
    cap = -f.getbbox("H", anchor="ls")[1]
    out, pad = [], 8
    for k, ch in enumerate(text):
        x0 = f.getlength(text[:k]) + k * track_px
        adv = f.getlength(text[:k + 1]) - f.getlength(text[:k])
        l, t, r, b = f.getbbox(ch, anchor="ls")
        im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
        ImageDraw.Draw(im).text((pad - l, pad - t), ch, font=f, fill=255, anchor="ls")
        ox, oy = pad - l, pad - t                                    # baseline origin in the raster
        out.append((np.asarray(im, np.float32) / 255.0, ox + adv / 2, oy - cap / 2, x0 + adv / 2))
    total = f.getlength(text) + (len(text) - 1) * track_px
    return out, total, cap


class Profile:
    """UV radius (texels) <-> world radius (m), piecewise linear through the mesh rings."""
    def __init__(self, prof):
        self.ru, self.rw = prof[:, 0], prof[:, 1]

    def w(self, ru):
        return np.interp(ru, self.ru, self.rw)

    def u(self, rw):
        return np.interp(rw, self.rw, self.ru)


def block_span(text, rho_m):
    gl, total, cap = glyphs(text, CAP_MM / 1000 * SPM, TRACK * CAP_MM / 1000 * SPM)
    return math.degrees(total / SPM / rho_m), gl, total


def choose_layout(zones, spans):
    """Block 0 centre in the upper half of the map (theta 180..360), block 1 opposite; maximise the smallest
    angular clearance between a block and the stock print."""
    def clearance(c, half):
        best = 1e9
        for a, b in zones:
            for k in (-360, 0, 360):
                lo, hi = a + k, b + k
                g = max(lo - (c + half), (c - half) - hi)
                best = min(best, g)
        return best
    best = None
    for c in np.arange(180, 360, 0.25):
        g = min(clearance(c, spans[0] / 2), clearance(c - 180, spans[1] / 2))
        if best is None or g > best[1] + 1e-9:
            best = (float(c), float(g))
    return best


def coverage(size, prof, blocks, rho_m):
    """Lettering coverage (0..1) on the size x size UV map (SS x SS supersampled)."""
    cov = np.zeros((size, size), np.float32)
    k = size / 2048.0
    for theta_c, gl, total in blocks:
        for img, xr, yr, xc in gl:
            th = math.radians(theta_c) + (xc - total / 2) / SPM / rho_m
            ct, st = math.cos(th), math.sin(th)
            C = np.array((rho_m * ct, rho_m * st))
            tvec, uvec = np.array((-st, ct)), np.array((ct, st))
            # texel box of the glyph (generous): world corners -> UV
            hh, ww = img.shape
            corners = []
            for cx_, cy_ in ((0, 0), (ww, 0), (0, hh), (ww, hh)):
                gx, gy = (cx_ - xr) / SPM, (yr - cy_) / SPM
                p = C + gx * tvec + gy * uvec
                rr = np.hypot(*p); a = math.atan2(p[1], p[0])
                ru = prof.u(rr) * k
                corners.append((size / 2 + ru * math.cos(a), size / 2 + ru * math.sin(a)))
            corners = np.array(corners)
            x0, y0 = np.floor(corners.min(0)).astype(int) - 3
            x1, y1 = np.ceil(corners.max(0)).astype(int) + 3
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, size), min(y1, size)
            sub = (np.arange(SS) + 0.5) / SS
            X = (np.arange(x0, x1)[None, :, None, None] + sub[None, None, None, :])
            Y = (np.arange(y0, y1)[:, None, None, None] + sub[None, None, :, None])
            X, Y = np.broadcast_arrays(X, Y)
            dx, dy = X - size / 2, Y - size / 2
            ru = np.hypot(dx, dy) / k
            rw = prof.w(ru)
            a = np.arctan2(dy, dx)
            px, py = rw * np.cos(a) - C[0], rw * np.sin(a) - C[1]
            gx = px * tvec[0] + py * tvec[1]
            gy = px * uvec[0] + py * uvec[1]
            col, row = xr + gx * SPM, yr - gy * SPM
            v = ndimage.map_coordinates(img, [row.ravel(), col.ravel()], order=1, cval=0.0).reshape(row.shape)
            v = v.mean((2, 3))
            cov[y0:y1, x0:x1] = np.maximum(cov[y0:y1, x0:x1], v)
    return cov


# ---------------------------------------------------------------- compositing
def to_lin(x):
    x = np.clip(x, 0, 255) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = np.clip(x, 0, 1)
    return 255.0 * np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def paint(orig_rgb, cov, ring_mask, grain=True, linear=False):
    """Pink print over the rubber: tone from the original mottling, light breakup from the original grain.
    linear=True mixes in linear light (time average of a spinning wheel, used for the blur map)."""
    L = orig_rgb.mean(-1)
    Lmed = ndimage.median_filter(L, size=5)                          # rubber without the thin moulded lines
    ref = float(np.median(Lmed[ring_mask]))
    mott = np.clip(Lmed / ref, *MOTT)
    tone = 1 + TONE * (mott - 1)
    a = cov.copy()
    if grain:
        g = L - Lmed
        sd = float(g[ring_mask].std()) or 1.0
        speck = np.clip((-g / sd - 0.8) / 1.6, 0, 1)                 # dark grain specks -> thinner paint
        blot = np.clip((0.97 - mott) / 0.08, 0, 1)                   # dark rubber clouds -> slightly thinner
        a = a * (1 - GRAIN * speck) * (1 - BLOT * blot)
    col = PAINT[None, None, :] * tone[..., None]
    if linear:
        out = to_srgb(to_lin(orig_rgb) * (1 - a[..., None]) + to_lin(col) * a[..., None])
    else:
        out = orig_rgb * (1 - a[..., None]) + col * a[..., None]
    return np.clip(out, 0, 255), a


def blur_ring(orig_b, cov, prof):
    """Soft low-opacity pink ring on the blur map: opacity(r) = BLUR_GAIN x angular mean coverage at r (what the
    two blocks average to over a turn), softened radially and mixed in linear light like a real motion blur."""
    pu, r = polar(cov, 800, 1020, dr=0.25, dth=0.1)
    m = pu.mean(1)                                                  # angular mean per radius (2048 units)
    m = ndimage.gaussian_filter1d(m, BLUR_SOFT / 0.25)              # soften (sigma in 2048 texels)
    size = orig_b.shape[0]
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    rr = np.hypot(xx - size / 2, yy - size / 2) * (2048 / size)
    op = BLUR_GAIN * np.interp(rr, r[::-1], m[::-1], left=0, right=0)
    ring = (rr > 812) & (rr < 1013)
    out, _ = paint(orig_b, op.astype(np.float32), ring, grain=False, linear=True)
    return out, op, (r, m)


# ---------------------------------------------------------------- preview
def lift(x):
    """Preview tone curve: lifts the dark rubber without clipping the pink."""
    return np.clip(255.0 * (np.clip(x, 0, 255) / 255.0) ** 0.6, 0, 255)


def preview(path, new0, blur_new, theta_blocks, zones, band, renders=None):
    W = 2400
    canvas = Image.new("RGB", (W, 3000), (22, 22, 26))
    d = ImageDraw.Draw(canvas)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        fnt = ImageFont.load_default()
    white = (230, 230, 230)
    L0 = lift(new0)
    canvas.paste(Image.fromarray(L0.astype(np.uint8)).resize((1000, 1000), Image.LANCZOS), (20, 50))
    d.text((20, 15), "tyre_d.dds 2048 (preview tone curve)", fill=white, font=fnt)
    canvas.paste(Image.fromarray(lift(blur_new).astype(np.uint8)).resize((560, 560), Image.LANCZOS), (20, 1100))
    d.text((20, 1065), "tyre_d_blur.dds 1024: soft ring", fill=white, font=fnt)
    x0, sw = 1040, 1340
    rmid = (band[0] + band[1]) / 2
    # aspect-true unwrap of the whole ring in two halves (outer edge up, theta increases to the right)
    dth = 180.0 / sw
    dr = math.radians(dth) * rmid
    pu, _ = polar(L0, 805, 1021, dr=dr, dth=dth)
    pu = np.clip(pu, 0, 255).astype(np.uint8)
    y = 15
    d.text((x0, y), "unwrapped ring, aspect true (outer edge up); pink bars = our blocks, grey = stock print kept",
           fill=white, font=fnt)
    y += 50
    for i in range(2):
        strip = Image.fromarray(pu[:, i * sw:(i + 1) * sw])
        canvas.paste(strip, (x0, y))
        for deg in range(i * 180, (i + 1) * 180 + 1, 15):
            x = x0 + (deg - i * 180) / dth
            d.line([(x, y - 6), (x, y)], fill=(255, 220, 0))
            d.text((x + 3, y - 26), f"{deg}", fill=(255, 220, 0), font=fnt)
        yb = y + strip.height + 4
        for spans, colr in ((zones, (150, 150, 160)), ([(c - h, c + h) for c, h, _ in theta_blocks], (242, 158, 178))):
            for a, b in spans:
                for k in (-360, 0, 360):
                    aa, bb = a + k - i * 180, b + k - i * 180
                    if bb < 0 or aa > 180:
                        continue
                    d.rectangle([x0 + max(aa, 0) / dth, yb, x0 + min(bb, 180) / dth, yb + 6], fill=colr)
        y = yb + 50
    # full-resolution upright views of both blocks (0.5 texel per pixel, aspect true)
    for c, half, txt in theta_blocks:
        dth2 = math.degrees(0.5 / rmid)
        pz, _ = polar(L0, band[0] - 25, band[1] + 25, dr=0.5, dth=dth2)
        n = int((2 * half + 6) / dth2)
        lo = int(((c - half - 3) % 360) / dth2)
        idx = np.arange(lo, lo + n) % pz.shape[1]
        crop = Image.fromarray(np.clip(pz[:, idx], 0, 255).astype(np.uint8))
        crop = crop.resize((sw, int(crop.height * sw / crop.width)), Image.LANCZOS)
        canvas.paste(crop, (x0, y))
        d.text((x0, y + crop.height + 4), f"{txt}: centre {c:.1f} deg, {2 * half:.1f} deg of arc "
               f"(texture unwrapped upright)", fill=white, font=fnt)
        y += crop.height + 44
    if renders:                                                      # whole-car side views, if rendered
        xs = x0
        for nm in ("side_left", "side_right"):
            p = os.path.join(renders, "full", nm + ".png")
            if os.path.exists(p):
                im = Image.open(p).convert("RGB")
                im = im.crop((int(im.width * 0.05), int(im.height * 0.22), int(im.width * 0.95),
                              int(im.height * 0.78)))
                im.thumbnail((660, 660))
                d.text((xs, y), f"render {nm}", fill=white, font=fnt)
                canvas.paste(im, (xs, y + 30))
                xs += 680
        y += 30 + 400
    y = max(y, 1690)
    if renders:
        x = 20
        for nm, lab in (("wheel_LF", "LF"), ("wheel_LR", "LR"), ("wheel_RF", "RF"), ("wheel_RR", "RR")):
            p = os.path.join(renders, nm + ".png")
            if os.path.exists(p):
                im = Image.open(p).convert("RGB")
                im.thumbnail((575, 575))
                d.text((x, y), f"render {lab} ({'left' if lab[0] == 'L' else 'right'} side, outer sidewall)",
                       fill=white, font=fnt)
                canvas.paste(im, (x, y + 30))
            x += 590
        y += 30 + 580
    canvas.crop((0, 0, W, y + 10)).save(path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--skins", default=",".join(SKINS))
    ap.add_argument("--preview", default=os.path.join(HERE, "tyre_preview.png"))
    ap.add_argument("--renders", default=None, help="folder with wheel_LF/LR/RF/RR.png close-ups (and "
                    "full/side_left.png, full/side_right.png) to add to the preview")
    a = ap.parse_args()

    tex, mat, tyres = load_kn5()
    print(f"material ture ({mat['shader']}): {mat['samplers']}")
    data, data_b = tex[TEX], tex[TEX_BLUR]
    for nm, dd in ((TEX, data), (TEX_BLUR, data_b)):
        w, h, mp = dds_info(dd)
        print(f"original {nm}: {w}x{h} DXT5, {mp} mips, {len(dd)} bytes")

    info, prof_tab = mesh_analysis(tyres)
    print("mesh rings (UV radius texels -> world radius mm, sidewall offset mm):")
    for ru, rw, xo in prof_tab:
        print(f"   r_uv {ru:7.1f} -> {rw * 1000:6.1f} mm  (x {xo * 1000:5.1f} mm)")
    for tag, res, rot in info:
        print(f"   wheel {tag}: outer sidewall det+/det- {res['outer']}, inner {res['inner']}, "
              f"texture rotation on screen {rot:+.1f} deg")
        assert res["outer"][1] == 0 and res["outer"][0] > 0, "outer sidewall mirrored"
    prof = Profile(prof_tab)

    lv0 = decode_levels(data)[0]
    orig = lv0[..., :3].astype(np.float32)
    L = orig.mean(-1)
    ring = texture_ring(L)
    band = ring["band"]
    print(f"texture ring: painted r {ring['edge'][0]:.1f}..{ring['edge'][1]:.1f}, moulded lines at "
          f"{', '.join(f'{g:.1f}' for g in ring['grooves'])}; lettering band r {band[0]:.1f}..{band[1]:.1f} "
          f"= world {prof.w(band[0]) * 1000:.1f}..{prof.w(band[1]) * 1000:.1f} mm")
    zones = stock_zones(L, band)
    print("stock print (deg, clockwise from +x): " + ", ".join(f"{a_:.1f}..{b_:.1f}" for a_, b_ in zones))

    rw0, rw1 = prof.w(band[0]), prof.w(band[1])
    rho_m = (rw0 + rw1) / 2
    cap = CAP_MM / 1000
    margin = (rw1 - rw0 - cap) / 2 * 1000
    print(f"type: Exo 2 ExtraBold Italic, cap {CAP_MM:.1f} mm, centred at r {rho_m * 1000:.1f} mm "
          f"(margins {margin:.1f} mm to the moulded lines), tracking {TRACK * 100:.0f} % cap")
    spans, gls = [], []
    for t in TEXTS:
        sp, gl, total = block_span(t, rho_m)
        spans.append(sp); gls.append((gl, total))
    c0, clear = choose_layout(zones, spans)
    assert clear >= CLEAR_MIN, f"blocks too close to the stock print ({clear:.1f} deg)"
    cents = (c0, (c0 - 180) % 360)
    for t, sp, c in zip(TEXTS, spans, cents):
        print(f"block «{t}»: centre {c:.1f} deg, arc {sp:.1f} deg ({c - sp / 2:.1f}..{c + sp / 2:.1f}), "
              f"length {math.radians(sp) * rho_m * 1000:.0f} mm")
    print(f"smallest clearance to the stock print: {clear:.1f} deg")

    blocks = [(c, gl, total) for c, (gl, total) in zip(cents, gls)]
    cov = coverage(2048, prof, blocks, rho_m)
    yy, xx = np.mgrid[0:2048, 0:2048] + 0.5
    rr = np.hypot(xx - 1024, yy - 1024)
    ring_mask = (rr > band[0]) & (rr < band[1])
    ys, xs = np.nonzero(cov > 0.02)
    rcov = np.hypot(xs + 0.5 - 1024, ys + 0.5 - 1024)
    print(f"lettering texels: {len(ys)}, UV radius {rcov.min():.1f}..{rcov.max():.1f} "
          f"(band {band[0]:.1f}..{band[1]:.1f})")
    assert rcov.min() > band[0] and rcov.max() < band[1]
    new0, aeff = paint(orig, cov, ring_mask)
    m = cov > 0.98
    sm = new0[m].mean(0)
    print(f"paint: PIG x {GAIN:.3f} = PAINT {tuple(int(round(v)) for v in PAINT)}; solid lettering mean "
          f"{tuple(int(round(v)) for v in sm)} (= {', '.join(f'{v:.3f}' for v in sm / PIG)} x PIG), "
          f"min/max luma {new0[m].mean(-1).min():.0f}/{new0[m].mean(-1).max():.0f}, "
          f"max R {new0[m][:, 0].max():.1f}")
    assert (sm >= PIG).all(), "lettering darker than the body pink"

    lvb = decode_levels(data_b)[0]
    orig_b = lvb[..., :3].astype(np.float32)
    new_b, op_b, (rb, mb) = blur_ring(orig_b, cov, prof)
    pk = int(np.argmax(mb))
    print(f"blur ring: peak opacity {mb[pk] * BLUR_GAIN:.3f} at r {rb[pk]:.1f}, "
          f"opacity > 0.02 for r {rb[mb * BLUR_GAIN > 0.02].min():.1f}..{rb[mb * BLUR_GAIN > 0.02].max():.1f}")

    origs = {TEX: data, TEX_BLUR: data_b}
    news = {TEX: new0, TEX_BLUR: new_b}
    for skin in [s for s in a.skins.split(",") if s]:
        dd = os.path.join(a.out, skin)
        os.makedirs(dd, exist_ok=True)
        for nm in (TEX, TEX_BLUR):
            p = os.path.join(dd, nm)
            st = write_patched(origs[nm], news[nm], p)
            got = open(p, "rb").read()
            lv_new, lv_old = decode_levels(got), decode_levels(origs[nm])
            same_hdr = got[:128] == origs[nm][:128] and len(got) == len(origs[nm])
            same_a = all(np.array_equal(x[..., 3], y[..., 3]) for x, y in zip(lv_new, lv_old))
            alpha_bytes = all(got[o:o + 8] == origs[nm][o:o + 8] for o in range(128, len(got), 16))
            err = np.abs(lv_new[0][..., :3].astype(np.float32) - np.clip(news[nm] + 0.5, 0, 255).astype(np.uint8))
            ch = np.abs(news[nm] - origs_lv0(origs[nm])).max(-1) > 0.75
            print(f"wrote {p}: {len(got)} bytes, header+layout identical {same_hdr}, alpha identical on all "
                  f"{len(lv_new)} levels {same_a} (alpha bytes {alpha_bytes}); re-encoded blocks per level "
                  f"{[n for _, n, _ in st]}; DXT error in changed texels mean {err[ch].mean():.2f}")
    if a.preview:
        tb = [(c, sp / 2, t) for c, sp, t in zip(cents, spans, TEXTS)]
        preview(a.preview, new0, new_b, tb, zones, band, a.renders)
        print(f"preview {a.preview}")


def origs_lv0(data):
    return decode_levels(data)[0][..., :3].astype(np.float32)


if __name__ == "__main__":
    main()
