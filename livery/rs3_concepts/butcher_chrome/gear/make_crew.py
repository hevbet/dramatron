#!/usr/bin/env python3
"""Pit-crew kit for the «Butcher Chart Chrome» skins (Pozdnyakov_23 / Konopelko_00).

Source: the SMP01 skin of the car. Every texture below is written with the SAME name and pixel size as
the original, into out/<skin>/ for both skins (identical files):

  ac_crew.dds            512x512   DXT5, no mips   crew shirt (flat navy + 3 SMP logos in the original)
  Crew_HELMET_Color.dds  1024x512  DXT5, no mips   crew helmet + chin straps (2x SMP ESPORTS on the straps)
  Brands_Crew.dds        512x256   DXT5, no mips   logo decal sheet; its ALPHA is the logo cut-out
  Meccanico_Gadgets.png  256x256   PNG RGBA        headset ear cups (СМП РСКГ / mirrored SMP ESPORTS) etc.
  (Brands_Crew_NM.dds, the flat normal map of the decal sheet, is NOT written: the logo shapes do not change,
   so a copy of it would override nothing.)

Design (palette from make_skin.py): deep maroon-plum shirt and helmet shell; flesh-pink accents (PIG);
«арка» + SMP RACING ESPORTS on a pink butcher price tag with a dashed burgundy cut line (the car's
cut-chart motif) and a holo hairline; SMP RACING ESPORTS in its own white/blue (on a deep-ink pill where
it sits on pink, as on the car); Симкарт on its night plate with a chrome frame (as on the car).

Rules kept
  * alpha channel of every file is copied unchanged from the original (DDS: the DXT5 alpha blocks are
    copied byte for byte from the source file, so even the gradient alpha of Brands_Crew is bit-identical);
  * shading is kept by recolouring in luminance (new = colour x original-luminance ratio), never a flat
    fill over shaded areas; fabric grain of the straps is kept;
  * logos/text are placed ONLY where the original has readable logos/text, in the same orientation
    (horizontal stays horizontal, the vertical shirt slot reads bottom->top, the strap slots read
    top->bottom); nothing is mirrored, aspect ratios are kept;
  * the right ear-cup disc of Meccanico_Gadgets carries MIRRORED text in the original (its UV is
    mirrored), so it gets no text: only a mirror-symmetric holo sparkle;
  * Brands_Crew: the alpha is the decal cut-out and must stay as is, so the logo SHAPES cannot change:
    BR ENGINEERING is recoloured flesh pink, the two SMP RACING keep their white/blue, and the marks that are
    not team partners (the KRB monogram and BR03 - on the driver suit BR03 became «арка») are filled with the
    shirt colour, so they melt into the shirt instead of being highlighted. The RGB under
    alpha=0 is filled with the nearest logo colour so DXT blocks at the logo edges do not pull black.

Usage: python3 make_crew.py [--out DIR] [--preview PATH]
"""
import argparse
import math
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
LIV = "/home/user/dramatron/livery"
SRC = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad/rs3/SMP01"
BX = os.path.join(LIV, "br03_pro", "brand_extra")
FONTS = os.path.join(LIV, "fonts")
sys.path.insert(0, os.path.join(LIV, "br03_pro"))
import lib as blib  # noqa: E402  (arka_logo, simkart_logo, save_dxt5)

SKINS = ("Pozdnyakov_23", "Konopelko_00")

# ---------------------------------------------------------------- palette (make_skin.py)
PIG = np.array((242, 158, 178), np.float32)       # flesh pink, the 917/20 tone
INK = (96, 18, 42)                                 # butcher's burgundy ink
INK_D = (40, 10, 22)                               # deep ink (outlines)
NIGHT = (16, 12, 20)                               # Simkart night plate
WHITE = (252, 250, 255)
ARKA_Y = (242, 206, 92)                            # «арка» yellow (as on the car)
SMP_BLUE = (0, 181, 239)
HOLO = np.array([(255, 120, 200), (180, 140, 255), (110, 200, 255), (110, 245, 210), (255, 230, 150),
                 (255, 120, 200)], np.float32)
# crew kit
SHIRT = np.array((82, 20, 49), np.float32)        # deep maroon-plum shirt; exact RGB565 value, so the
#                                                    flat shirt and the hidden KRB decal encode to the same DXT colour
SHELL = np.array((80, 18, 44), np.float32)        # helmet shell at the original's base grey (L=16.3)


# ---------------------------------------------------------------- small art helpers (as in make_skin.py)
def solid(mask, col):
    t = Image.new("RGBA", mask.size, tuple(int(c) for c in col[:3]) + (255,))
    t.putalpha(mask)
    return t


def grow(mask, r):
    if r <= 0:
        return mask
    a = np.asarray(mask) > 127
    d = ndimage.distance_transform_edt(~a)
    return Image.fromarray((np.clip(r + 0.5 - d, 0, 1) * 255).astype(np.uint8))


def holo_lookup(t):
    t = (np.asarray(t, np.float32) % 1.0) * (len(HOLO) - 1)
    k = np.floor(t).astype(int)
    f = (t - k)[..., None]
    return HOLO[k] * (1 - f) + HOLO[np.minimum(k + 1, len(HOLO) - 1)] * f


def holo_rgb(w, h, scale=1.0, phase=0.0):
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    t = ((x * 1.3 + y * 0.45) * scale + phase) % 1.0
    return Image.fromarray(holo_lookup(t).astype(np.uint8))


def chrome_rgb(w, h, tint=(1.0, 0.97, 1.04)):
    y = np.linspace(0, 1, h)[:, None]
    st = [(0, 252), (0.38, 232), (0.48, 150), (0.53, 96), (0.6, 205), (0.82, 246), (1, 200)]
    v = np.interp(y, [s[0] for s in st], [s[1] for s in st]) * np.ones((1, w))
    return Image.fromarray(np.clip(np.stack([v * tint[0], v * tint[1], v * tint[2]], -1), 0, 255).astype(np.uint8))


def fill_mask(mask, rgb_img):
    t = rgb_img.convert("RGBA").resize(mask.size)
    t.putalpha(mask)
    return t


def fit_box(img, w, h):
    k = min(w / img.width, h / img.height)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def sparkle(r, col=WHITE):
    """4-point Y2K sparkle (mirror-symmetric left/right), radius r px, holo fill, deep-ink keyline."""
    s = int(r * 2.6) * 2
    m = Image.new("L", (s, s), 0)
    c = s / 2
    pts = []
    for k in range(8):
        a = math.radians(k * 45 - 90)
        rr = r if k % 2 == 0 else r * 0.22
        if k % 4 == 2:
            rr = r * 0.62
        pts.append((c + math.cos(a) * rr, c + math.sin(a) * rr))
    ImageDraw.Draw(m).polygon(pts, fill=255)
    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    g = m.filter(ImageFilter.GaussianBlur(r * 0.25)).point(lambda v: int(min(255, v * 1.2)))
    out.alpha_composite(solid(g, (255, 210, 240)))
    out.alpha_composite(solid(grow(m, max(1, r // 9)), INK_D))
    hx = holo_rgb(s, s, 0.8, 0.1)
    hx = Image.blend(hx, hx.transpose(Image.FLIP_LEFT_RIGHT), 0.5)          # symmetric holo sheen
    out.alpha_composite(fill_mask(m, hx))
    core = m.filter(ImageFilter.MinFilter(3)) if r > 14 else m
    out.alpha_composite(solid(core.point(lambda v: int(v * 0.7)), col))
    return out.crop(out.getbbox())


def dashed_path(pts, closed, on, off, phase=0.0):
    """Split a polyline into dash segments by arc length."""
    if closed:
        pts = list(pts) + [pts[0]]
    segs, cur, s = [], [], -phase
    period = on + off
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(L / 0.5))
        for k in range(n):
            t = k / n
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            if (s % period) < on:
                cur.append((x, y))
            elif cur:
                segs.append(cur)
                cur = []
            s += L / n
    if cur:
        segs.append(cur)
    return [g for g in segs if len(g) > 1]


def rrect_path(x0, y0, x1, y1, r, n=24):
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def circle_path(cx, cy, r, n=180):
    return [(cx + r * math.cos(2 * math.pi * k / n - math.pi / 2), cy + r * math.sin(2 * math.pi * k / n - math.pi / 2))
            for k in range(n)]


def draw_dashes(size, pts, lw, on, off, ss=4, phase=0.0):
    """Anti-aliased dashed line mask (supersampled)."""
    m = Image.new("L", (size[0] * ss, size[1] * ss), 0)
    d = ImageDraw.Draw(m)
    for seg in dashed_path([(x * ss, y * ss) for x, y in pts], True, on * ss, off * ss, phase * ss):
        d.line(seg, fill=255, width=max(1, int(lw * ss)), joint="curve")
    return m.resize(size, Image.LANCZOS)


# ---------------------------------------------------------------- brand art
def asset_arka(h, taimcafe=False):
    return blib.arka_logo(int(h), fill=ARKA_Y, dark=INK_D, taimcafe=taimcafe).crop()


def asset_smp(h):
    """SMP RACING ESPORTS, brand colours (white + blue accent), height h px."""
    w = Image.open(os.path.join(BX, "smp_racing_esports_white.png")).convert("L")
    a = Image.open(os.path.join(BX, "smp_racing_esports_accent.png")).convert("L")
    im = Image.new("RGBA", w.size, (0, 0, 0, 0))
    im.alpha_composite(solid(w, WHITE))
    im.alpha_composite(solid(a, SMP_BLUE))
    im = im.crop(im.getbbox())
    return im.resize((max(1, round(im.width * h / im.height)), int(h)), Image.LANCZOS)


def simkart_plate(w, h):
    """Simkart wordmark on its night plate with a chrome bevel + holo hairline (car tag, without url)."""
    S = 4
    W, H = w * S, h * S
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    r, bw = int(H * 0.22), max(4, int(H * 0.08))
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, W - 1, H - 1], radius=r, fill=255)
    out.alpha_composite(fill_mask(m, chrome_rgb(W, H)))
    m2 = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m2).rounded_rectangle([bw, bw, W - 1 - bw, H - 1 - bw], radius=max(2, r - bw), fill=255)
    out.alpha_composite(fill_mask(m2, holo_rgb(W, H, 1.2)))
    k = bw + max(2, bw // 3)
    m3 = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m3).rounded_rectangle([k, k, W - 1 - k, H - 1 - k], radius=max(2, r - k), fill=255)
    out.alpha_composite(solid(m3, NIGHT))
    logo = blib.simkart_logo(int(H * 0.62), silver=(222, 220, 232), red=(236, 52, 40), glow=True)
    logo = fit_box(logo, W - 2 * k - int(H * 0.30), H - 2 * k - int(H * 0.10))
    out.alpha_composite(logo, ((W - logo.width) // 2, (H - logo.height) // 2))
    return out.resize((w, h), Image.LANCZOS)


def butcher_tag(w, h, content):
    """Flesh-pink pearl price tag: holo hairline edge, dashed burgundy cut line inset, content centred."""
    S = 4
    W, H = w * S, h * S
    r = int(H * 0.26)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, W - 1, H - 1], radius=r, fill=255)
    out.alpha_composite(fill_mask(m, holo_rgb(W, H, 1.6)))
    e = max(3, int(H * 0.045))
    m2 = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m2).rounded_rectangle([e, e, W - 1 - e, H - 1 - e], radius=r - e, fill=255)
    y = np.linspace(0, 1, H)[:, None, None]
    pearl = PIG[None, None, :] * (1.06 - 0.12 * y) + np.array((10, 6, 12), np.float32) * (1 - y) * 0.6
    pearl = Image.fromarray(np.clip(pearl * np.ones((1, W, 1)), 0, 255).astype(np.uint8))
    out.alpha_composite(fill_mask(m2, pearl))
    ins = e + int(H * 0.085)
    lw = max(2.0, H * 0.035) / S
    dm = draw_dashes((w, h), rrect_path(ins / S, ins / S, (W - ins) / S, (H - ins) / S, (r - ins) / S),
                     lw, on=h * 0.11, off=h * 0.07)
    out.alpha_composite(solid(dm.resize((W, H), Image.LANCZOS), INK))
    out = out.resize((w, h), Image.LANCZOS)
    out.alpha_composite(content, ((w - content.width) // 2, (h - content.height) // 2))
    return out


def pill(art, px, py, fill):
    """Rounded pill (supersampled) behind art."""
    w, h = art.width + 2 * px, art.height + 2 * py
    t = Image.new("RGBA", (w * 4, h * 4), (0, 0, 0, 0))
    ImageDraw.Draw(t).rounded_rectangle([0, 0, w * 4 - 1, h * 4 - 1], radius=h * 2, fill=tuple(fill) + (255,))
    t = t.resize((w, h), Image.LANCZOS)
    t.alpha_composite(art, (px, py))
    return t


def paste(dst, art, cx, cy, angle=0):
    """Composite art centred at (cx, cy). angle: PIL rotate (CCW, degrees), multiples of 90 only."""
    if angle:
        art = art.rotate(angle, expand=True)          # 90-degree steps: exact, no resampling, no mirroring
    dst.alpha_composite(art, (int(round(cx - art.width / 2)), int(round(cy - art.height / 2))))
    return (int(round(cx - art.width / 2)), int(round(cy - art.height / 2)), art.width, art.height)


# ---------------------------------------------------------------- texture io
def load(name):
    return Image.open(os.path.join(SRC, name)).convert("RGBA")


def lum(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def with_alpha(rgb, src):
    """RGB from the restyle, alpha copied unchanged from the source."""
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    out.putalpha(src.getchannel("A"))
    return out


def copy_alpha_blocks(src_path, dst_path):
    """DXT5 alpha is lossy: re-encoding a decoded gradient alpha changes it. Both files are single-level
    DXT5 of the same size, so the 8-byte alpha half of every 16-byte block is copied from the original
    -> the alpha channel is bit-identical to the source."""
    src = open(src_path, "rb").read()
    dst = bytearray(open(dst_path, "rb").read())
    assert len(src) == len(dst) and src[84:88] == dst[84:88] == b"DXT5"
    for o in range(128, len(dst), 16):
        dst[o:o + 8] = src[o:o + 8]
    open(dst_path, "wb").write(bytes(dst))


def dds_header(path):
    b = open(path, "rb").read(128)
    h, w = struct.unpack("<II", b[12:20])
    mips = struct.unpack("<I", b[28:32])[0]
    return w, h, mips, b[84:88]


# ================================================================= ac_crew.dds (shirt)
NAVY = np.array((33, 58, 90), np.float32)
CREW_SLOTS = {                     # original logos (x0, y0, x1, y1) and their orientation
    "A_big": ((41, 90, 354, 152), 0),        # SMP ESPORTS, horizontal
    "B_vert": ((428, 79, 450, 179), 90),     # SMP RACING, reads bottom -> top (rotated 90 CCW)
    "C_small": ((182, 471, 330, 503), 0),    # SMP RACING, horizontal
}


def make_ac_crew():
    src = load("ac_crew.dds")
    a = np.asarray(src).astype(np.float32)[..., :3]
    old_logo = ndimage.binary_dilation(np.abs(a - NAVY).sum(-1) > 30, iterations=2)
    ratio = np.clip(lum(a) / lum(NAVY), 0.0, 1.25)
    ratio[old_logo] = 1.0                                          # flat shirt: logos are simply erased
    rgb = SHIRT[None, None, :] * ratio[..., None]
    img = with_alpha(rgb, src)
    art = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # A: pink butcher tag in the old SMP ESPORTS box: «арка» | sparkle | SMP RACING ESPORTS on a deep-ink pill
    (x0, y0, x1, y1), _ = CREW_SLOTS["A_big"]
    tw, th = (x1 - x0) - 6, (y1 - y0) + 6
    ar = fit_box(asset_arka(int(th * 1.2)), tw * 0.40, th * 0.66)     # stays inside the dashed cut line
    sp = sparkle(int(th * 0.16))
    smp = pill(asset_smp(int(th * 0.36)), 10, 6, INK_D)
    gap = int(th * 0.15)
    row = Image.new("RGBA", (ar.width + sp.width + smp.width + 2 * gap, max(ar.height, smp.height)), (0, 0, 0, 0))
    x = 0
    for it in (ar, sp, smp):
        row.alpha_composite(it, (x, (row.height - it.height) // 2))
        x += it.width + gap
    paste(art, butcher_tag(tw, th, row), (x0 + x1) / 2, (y0 + y1) / 2)

    # B: SMP RACING ESPORTS, vertical, same reading direction as the old SMP RACING (bottom -> top)
    (x0, y0, x1, y1), ang = CREW_SLOTS["B_vert"]
    smp = asset_smp(x1 - x0 + 2)                                     # thickness = old logo thickness + 2
    smp = fit_box(smp, y1 - y0, x1 - x0 + 2)
    paste(art, smp, (x0 + x1) / 2, (y0 + y1) / 2, angle=ang)

    # C: Симкарт night plate (old small SMP RACING slot, +3 px)
    (x0, y0, x1, y1), _ = CREW_SLOTS["C_small"]
    paste(art, simkart_plate((x1 - x0) + 4, (y1 - y0) + 6), (x0 + x1) / 2, (y0 + y1) / 2)

    rgb_img = img.convert("RGB").convert("RGBA")
    rgb_img.alpha_composite(art)
    out = rgb_img.convert("RGB").convert("RGBA")
    out.putalpha(src.getchannel("A"))
    return src, out


# ================================================================= Crew_HELMET_Color.dds
STRAP_X = (976, 1010)                       # columns of the strap that held the logos
STRAP_SLOTS = [((979, 64, 1006, 195), "arka"), ((980, 313, 1006, 444), "smp")]   # read top -> bottom
STRAP_CLEAN = [(22, 60), (200, 248), (268, 309), (448, 482)]                      # fabric rows w/o logos


def make_helmet():
    src = load("Crew_HELMET_Color.dds")
    a = np.asarray(src).astype(np.float32)[..., :3]
    H, W = a.shape[:2]

    # 1) strap logos out: rebuild the fabric grain row by row from clean rows of the same strap
    rng = np.random.default_rng(64)
    x0, x1 = STRAP_X
    clean = [y for lo, hi in STRAP_CLEAN for y in range(lo, hi)
             if a[y, x0:x1].max() <= 30 and (a[y, x0:x1, 0] - a[y, x0:x1, 1:].max(-1)).max() <= 10]
    for (sx0, sy0, sx1, sy1), _ in STRAP_SLOTS:
        for y in range(sy0 - 4, sy1 + 5):
            a[y, x0:x1] = a[clean[rng.integers(len(clean))], x0:x1]

    # 2) recolour in luminance
    Y = lum(a)
    redness = a[..., 0] - a[..., 1:].max(-1)
    red_w = np.clip((redness - 12) / 24, 0, 1)[..., None]          # soft class edge (no DXT speckle)
    sat = a.max(-1) - a.min(-1)
    shell_w = np.clip((40 - Y) / 16, 0, 1) * (sat < 30)            # dark neutral -> plum shell (metal stays)
    shell = SHELL[None, None, :] * np.clip(Y / 16.3, 0, 1.6)[..., None]
    rgb = a * (1 - shell_w[..., None]) + shell * shell_w[..., None]
    pink = PIG[None, None, :] * np.clip(a[..., 0] / 128.0, 0, 1.06)[..., None]
    rgb = rgb * (1 - red_w) + pink * red_w                          # red trims -> flesh pink, shading kept
    img = with_alpha(rgb, src)

    # 3) new strap logos, same orientation as the old ones (rotated 90 CW, read top -> bottom)
    art = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for (sx0, sy0, sx1, sy1), what in STRAP_SLOTS:
        cx, cy, L, T = (sx0 + sx1) / 2, (sy0 + sy1) / 2, sy1 - sy0, sx1 - sx0
        if what == "arka":
            ar = asset_arka(T + 3)
            sp = sparkle(int(T * 0.30))
            row = Image.new("RGBA", (ar.width + 2 * sp.width + 16, max(ar.height, sp.height)), (0, 0, 0, 0))
            row.alpha_composite(sp, (0, (row.height - sp.height) // 2))
            row.alpha_composite(ar, ((row.width - ar.width) // 2, (row.height - ar.height) // 2))
            row.alpha_composite(sp, (row.width - sp.width, (row.height - sp.height) // 2))
            lg = fit_box(row, L, T + 6)
        else:
            lg = fit_box(asset_smp(T), L, T)
        paste(art, lg, cx, cy, angle=-90)
    rgb_img = img.convert("RGB").convert("RGBA")
    rgb_img.alpha_composite(art)
    out = rgb_img.convert("RGB").convert("RGBA")
    out.putalpha(src.getchannel("A"))
    return src, out


# ================================================================= Brands_Crew.dds (decal sheet)
BRANDS = {                      # component boxes on the sheet (alpha), what to do
    "br_engineering": ((38, 14, 112, 118), "pink"),
    "smp_racing_top": ((170, 20, 329, 51), "keep"),
    "krb_monogram": ((399, 13, 491, 110), "shirt"),
    "br03": ((15, 145, 120, 185), "shirt"),       # not a listed partner -> hidden like KRB
    "smp_racing_bot": ((159, 208, 353, 247), "keep"),
}


def make_brands():
    src = load("Brands_Crew.dds")
    a = np.asarray(src).astype(np.float32)
    rgb = a[..., :3].copy()
    alpha = a[..., 3]
    vis = alpha > 0
    for (x0, y0, x1, y1), mode in BRANDS.values():
        sl = (slice(y0 - 2, y1 + 3), slice(x0 - 2, x1 + 3))
        m = vis[sl]
        if mode == "pink":
            rgb[sl][m] = PIG
        elif mode == "shirt":
            rgb[sl][m] = SHIRT
        elif mode == "keep":       # brand colours; lift dark edge texels (black matte) to the logo colour
            sub = rgb[sl]
            dark = m & (sub.max(-1) < 200) & ~((sub[..., 2] > sub[..., 0] + 40))
            sub[dark] = WHITE
    # RGB under alpha=0: nearest visible colour (no black fringes from DXT blocks / filtering)
    _, (iy, ix) = ndimage.distance_transform_edt(~vis, return_indices=True)
    rgb = rgb[iy, ix]
    return src, with_alpha(rgb, src)


# ================================================================= Meccanico_Gadgets.png
CUP_BIG = (70.5, 82.5, 66.0)          # centre x, y, inner radius (navy disc, old «СМП РСКГ», readable)
CUP_SMALL = (199.0, 66.0, 44.0)       # old SMP ESPORTS, MIRRORED in the texture -> no text here


def make_gadgets():
    src = Image.open(os.path.join(SRC, "Meccanico_Gadgets.png")).convert("RGBA")
    a = np.asarray(src).astype(np.float32)[..., :3]
    H, W = a.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    rgb = a.copy()
    red_w = np.clip((a[..., 0] - a[..., 1:].max(-1) - 8) / 24, 0, 1)
    navy = (a[..., 2] > a[..., 0] + 20) & (a[..., 2] > 25)
    inside = np.zeros((H, W), bool)
    for cx, cy, r in (CUP_BIG, CUP_SMALL):
        inside |= np.hypot(xx - cx, yy - cy) <= r + 1.5
    pink = PIG[None, None, :] * np.clip(a[..., 0] / 255.0, 0, 1)[..., None]
    w = (red_w * ~inside)[..., None]
    rgb = rgb * (1 - w) + pink * w                                   # headset shell: red -> flesh pink
    plum = SHIRT[None, None, :] * np.clip(a[..., 2] / 84.0, 0, 1.1)[..., None]
    rgb[navy] = plum[navy]                                           # disc edges (navy <-> black ring)
    for cx, cy, r in (CUP_BIG, CUP_SMALL):                           # wipe the old logos, flat plum disc
        d = np.hypot(xx - cx, yy - cy)
        w = np.clip(r + 0.5 - d, 0, 1)[..., None]
        rgb = rgb * (1 - w) + SHIRT[None, None, :] * w
    img = with_alpha(rgb, src)

    art = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for (cx, cy, r), lw in ((CUP_BIG, 2.0), (CUP_SMALL, 1.6)):       # dashed pink "cut line" ring
        dm = draw_dashes((W, H), circle_path(cx, cy, r - 5.5, 240), lw, on=7.0, off=4.5)
        art.alpha_composite(solid(dm, PIG))
    cx, cy, r = CUP_BIG                                              # «арка · тайм-кафе», readable disc
    ar = fit_box(asset_arka(64, taimcafe=True), r * 1.42, r * 1.25)
    paste(art, ar, cx, cy + 2)
    cx, cy, r = CUP_SMALL                                            # mirrored disc: symmetric sparkle only
    paste(art, sparkle(int(r * 0.52)), cx, cy)
    rgb_img = img.convert("RGB").convert("RGBA")
    rgb_img.alpha_composite(art)
    out = rgb_img.convert("RGB").convert("RGBA")
    out.putalpha(src.getchannel("A"))
    return src, out


# ================================================================= preview
def label(img, s, xy, size=26, col=(235, 235, 240)):
    f = ImageFont.truetype(os.path.join(FONTS, "Exo2-Italic[wght].ttf"), size)
    f.set_variation_by_name("Bold Italic")
    ImageDraw.Draw(img).text(xy, s, font=f, fill=col)


def over(img, bg):
    b = Image.new("RGBA", img.size, tuple(int(c) for c in bg) + (255,))
    b.alpha_composite(img)
    return b.convert("RGB")


def checker(size, c=12):
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    v = np.where(((xx // c) + (yy // c)) % 2 == 0, 70, 50).astype(np.uint8)
    return Image.fromarray(np.stack([v] * 3, -1)).convert("RGBA")


def boost(img, k):
    a = np.asarray(img.convert("RGB")).astype(np.float32) * k
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make_preview(res, path):
    G, LBL = 16, 40
    BG = (24, 22, 28)
    crew_o, crew_n = res["ac_crew.dds"]
    hel_o, hel_n = res["Crew_HELMET_Color.dds"]
    br_o, br_n = res["Brands_Crew.dds"]
    gd_o, gd_n = res["Meccanico_Gadgets.png"]
    strap = (948, 0, 1024, 512)
    sw = (strap[2] - strap[0]) * 2
    Wc = G + 2048 + G + 2 * sw + G * 3
    Hc = G + LBL + 512 + G + LBL + 512 + G + LBL + 512 + G + 40
    cv = Image.new("RGB", (Wc, Hc), BG)
    y = G
    # row 1: shirt, ear cups
    x = G
    for im, t in ((crew_o.convert("RGB"), "ac_crew  original"), (crew_n.convert("RGB"), "ac_crew  restyle")):
        label(cv, t, (x, y)); cv.paste(im, (x, y + LBL)); x += 512 + G
    for im, t in ((gd_o, "Meccanico_Gadgets  original (x2)"), (gd_n, "Meccanico_Gadgets  restyle (x2)")):
        ck = checker((512, 512)); ck.alpha_composite(im.resize((512, 512), Image.NEAREST))
        label(cv, t, (x, y)); cv.paste(ck.convert("RGB"), (x, y + LBL)); x += 512 + G
    y += LBL + 512 + G
    # row 2: helmet + strap zoom
    x = G
    label(cv, "Crew_HELMET_Color  original  /  restyle   (straps: x2, exposure x2.5)", (x, y))
    cv.paste(hel_o.convert("RGB"), (x, y + LBL)); x += 1024 + G
    cv.paste(hel_n.convert("RGB"), (x, y + LBL)); x += 1024 + G
    for im in (hel_o, hel_n):
        z = boost(im.crop(strap), 2.5).resize((sw, 1024), Image.NEAREST)
        cv.paste(z, (x, y + LBL)); x += sw + G
    y += LBL + 512 + G
    # row 3: decal sheet as it lands on the shirt (alpha-composited over the shirt colour)
    x = G
    label(cv, "Brands_Crew (x2)  original over the old shirt  /  restyle over the new shirt  (alpha unchanged)",
          (x, y))
    cv.paste(over(br_o, NAVY).resize((1024, 512), Image.LANCZOS), (x, y + LBL)); x += 1024 + G
    cv.paste(over(br_n, SHIRT).resize((1024, 512), Image.LANCZOS), (x, y + LBL))
    label(cv, "Brands_Crew_NM: flat normal map, logo shapes unchanged -> not overridden (no file written).",
          (G, Hc - 38), size=22, col=(180, 176, 190))
    cv.save(path)


# ================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--preview", default=os.path.join(HERE, "crew_preview.png"))
    args = ap.parse_args()

    res = {
        "ac_crew.dds": make_ac_crew(),
        "Crew_HELMET_Color.dds": make_helmet(),
        "Brands_Crew.dds": make_brands(),
        "Meccanico_Gadgets.png": make_gadgets(),
    }
    shown = {}                                     # preview shows the files as written (DXT5 decoded)
    for skin in SKINS:
        od = os.path.join(args.out, skin)
        os.makedirs(od, exist_ok=True)
        for name, (src, new) in res.items():
            assert new.size == src.size, name
            assert np.array_equal(np.asarray(new.getchannel("A")), np.asarray(src.getchannel("A"))), name
            p = os.path.join(od, name)
            if name.endswith(".dds"):
                blib.save_dxt5(new, p)
                copy_alpha_blocks(os.path.join(SRC, name), p)
                o, n = dds_header(os.path.join(SRC, name)), dds_header(p)
                assert o == n, (name, o, n)          # size, mip count, DXT5
            else:
                new.save(p, optimize=True)
            shown[name] = (src, Image.open(p).convert("RGBA"))
            assert np.array_equal(np.asarray(shown[name][1].getchannel("A")),
                                  np.asarray(src.getchannel("A"))), ("alpha changed", name)
        print(skin, "->", od)
    make_preview(shown, args.preview)
    print("preview ->", args.preview)


if __name__ == "__main__":
    main()
