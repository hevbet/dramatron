"""Concept 1 «Butcher Chart Chrome» — Audi RS3 LMS (smp_audi_rs3_lms).

Pink Pig 2.0: flesh-pink car divided into butcher "cuts" by burgundy dashed lines that follow the real
character lines of the body (Audi shoulder crease, door-bottom crease, wide-body flare), vintage shop-sign
labels (Yeseva One), plus a Y2K layer: holographic lower band, chrome keylines, chrome numbers, sparkles.

Everything that crosses panels is painted in WORLD SPACE (texel -> 3D position map at 4096), so lines are
continuous across door gaps and UV seams and logos/text are projected level to the ground with correct
aspect. Every logo/text decal is checked automatically (inside its zone mask, clearance to panel edge,
clearance to the cut lines, surface tilt) — see check_report.txt.

Run:  python3 make_skin.py            (writes Pozdnyakov_00/, Konopelko_00/, texture_preview.png, zips)
"""
import json
import math
import os
import struct
import sys
import zipfile

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
LIV = "/home/user/dramatron/livery"
SCRATCH = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad"
SRC = os.path.join(SCRATCH, "rs3", "SMP01")
ZONES = os.path.join(LIV, "rs3_concepts", "zones")
FONTS = os.path.join(LIV, "fonts")
BRAND = os.path.join(LIV, "br03", "brand")
BX = os.path.join(LIV, "br03_pro", "brand_extra")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(LIV, "br03_pro"))
import posmap4k  # noqa: E402
import lib as blib  # noqa: E402  (br03_pro helpers: arka_logo, simkart_logo, driveoil_logo)

N = 4096
TPM = 795.0                     # texels per metre (median density) – for cm reporting

# ---------------------------------------------------------------- palette
PIG = np.array((242, 158, 178), np.float32)       # flesh pink, the 917/20 tone
INK = (96, 18, 42)                                 # butcher's burgundy ink
INK_D = (40, 10, 22)                               # deep ink (outlines)
NIGHT = (16, 12, 20)                               # Simkart night plate
STAMP = (104, 64, 168)                             # violet vet-stamp ink
WHITE = (252, 250, 255)
HOLO = np.array([(255, 150, 205), (214, 168, 255), (150, 206, 255), (152, 246, 222), (255, 240, 196),
                 (255, 168, 214)], np.float32)


# ---------------------------------------------------------------- fonts
def font(name, size, var=None, axes=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), int(size))
    if var:
        f.set_variation_by_name(var)
    if axes:
        f.set_variation_by_axes(axes)
    return f


F_NUM = lambda s: font("Unbounded[wght].ttf", s, "Black")                       # numbers
F_NAME = lambda s: font("SofiaSansCondensed-Italic[wght].ttf", s, "Black Italic")  # driver names
F_LABEL = lambda s: font("YesevaOne-Regular.ttf", s)                            # butcher cut labels
F_STAMP = lambda s: font("Podkova[wght].ttf", s, "ExtraBold")                   # vet stamp
F_SPON = lambda s: font("Exo2-Italic[wght].ttf", s, "ExtraBold Italic")         # urls / partner text
F_TEAM = lambda s: font("MontserratAlternates-BlackItalic.ttf", s)              # «Команда ЭДМ»
F_TECH = lambda s: font("Tektur[wdth,wght].ttf", s, axes=[100, 800])            # KARTING64.RU


# ================================================================= 2D art helpers
def text_mask(s, f, track=0.0):
    if not track:
        bb = f.getbbox(s)
        m = Image.new("L", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), 0)
        ImageDraw.Draw(m).text((4 - bb[0], 4 - bb[1]), s, font=f, fill=255)
        return m
    gap = f.size * track
    w = int(sum(f.getlength(c) for c in s) + gap * len(s)) + 20
    asc, desc = f.getmetrics()
    m = Image.new("L", (w, asc + desc + 20), 0)
    d = ImageDraw.Draw(m)
    x = 10
    for c in s:
        d.text((x, 10), c, font=f, fill=255)
        x += f.getlength(c) + gap
    return m.crop(m.getbbox())


def solid(mask, col):
    t = Image.new("RGBA", mask.size, tuple(col[:3]) + (255,))
    t.putalpha(mask)
    return t


def pad(mask, p):
    b = Image.new("L", (mask.width + 2 * p, mask.height + 2 * p), 0)
    b.paste(mask, (p, p))
    return b


def grow(mask, r):
    """Round dilation by r px (outline)."""
    if r <= 0:
        return mask
    a = np.asarray(mask) > 127
    d = ndimage.distance_transform_edt(~a)
    return Image.fromarray((np.clip(r + 0.5 - d, 0, 1) * 255).astype(np.uint8))


def chrome_rgb(w, h, tint=(1.0, 0.97, 1.04)):
    y = np.linspace(0, 1, h)[:, None]
    st = [(0, 252), (0.38, 232), (0.48, 150), (0.53, 96), (0.6, 205), (0.82, 246), (1, 200)]
    v = np.interp(y, [s[0] for s in st], [s[1] for s in st]) * np.ones((1, w))
    rgb = np.stack([v * tint[0], v * tint[1], v * tint[2]], -1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def holo_rgb(w, h, scale=1.0, phase=0.0):
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    t = ((x * 1.3 + y * 0.45) * scale + phase) % 1.0
    return Image.fromarray(holo_lookup(t).astype(np.uint8))


def holo_lookup(t):
    t = (np.asarray(t, np.float32) % 1.0) * (len(HOLO) - 1)
    k = np.floor(t).astype(int)
    f = (t - k)[..., None]
    return HOLO[k] * (1 - f) + HOLO[np.minimum(k + 1, len(HOLO) - 1)] * f


def fill_mask(mask, rgb_img):
    t = rgb_img.convert("RGBA").resize(mask.size)
    t.putalpha(mask)
    return t


def chrome_text(s, f, ow, outline=INK_D, holo_edge=True, track=0.0):
    """Chrome lettering, inner holo hairline, dark outline, soft drop shadow."""
    m = pad(text_mask(s, f, track), ow * 2 + 8)
    ol = grow(m, ow)
    out = Image.new("RGBA", m.size, (0, 0, 0, 0))
    sh = ImageChops.offset(ol, max(2, ow // 2), max(3, ow // 2 + 1)).filter(ImageFilter.GaussianBlur(ow * 0.4))
    out.alpha_composite(solid(sh.point(lambda v: int(v * 0.45)), INK_D))
    out.alpha_composite(solid(ol, outline))
    if holo_edge:
        out.alpha_composite(fill_mask(m, holo_rgb(m.width, m.height, 1.4)))
        inner = Image.fromarray(((ndimage.distance_transform_edt(np.asarray(m) > 127) > max(2, ow * 0.35)) * 255)
                                .astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
        out.alpha_composite(fill_mask(inner, chrome_rgb(m.width, m.height)))
    else:
        out.alpha_composite(fill_mask(m, chrome_rgb(m.width, m.height)))
    return out.crop(out.getbbox())


def ink_text(s, f, col=INK, track=0.0):
    return solid(text_mask(s, f, track), col).crop()


def sparkle(r, col=WHITE, glow=True):
    """4-point Y2K sparkle star of radius r px."""
    s = int(r * 2.6) * 2
    m = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(m)
    c = s / 2
    pts = []
    for k in range(8):
        a = math.radians(k * 45 - 90)
        rr = r if k % 2 == 0 else r * 0.22
        if k % 4 == 2:
            rr = r * 0.62
        pts.append((c + math.cos(a) * rr, c + math.sin(a) * rr))
    d.polygon(pts, fill=255)
    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    if glow:
        g = m.filter(ImageFilter.GaussianBlur(r * 0.25)).point(lambda v: int(min(255, v * 1.2)))
        out.alpha_composite(solid(g, (255, 210, 240)))
    out.alpha_composite(solid(grow(m, max(1, r // 9)), INK_D))
    out.alpha_composite(fill_mask(m, holo_rgb(s, s, 0.8, 0.1)))
    core = m.filter(ImageFilter.MinFilter(3)) if r > 14 else m
    out.alpha_composite(solid(core.point(lambda v: int(v * 0.7)), col))
    return out


def rrect(w, h, r, fill, outline=None, ow=0):
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(t).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=tuple(fill) + (255,),
                                        outline=(tuple(outline) + (255,)) if outline else None, width=ow)
    return t


def chrome_frame(w, h, r, bw, fill):
    """Rounded plate: chrome bevel border + thin holo inner line + fill."""
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    out.alpha_composite(fill_mask(m, chrome_rgb(w, h)))
    m2 = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m2).rounded_rectangle([bw, bw, w - 1 - bw, h - 1 - bw], radius=max(2, r - bw), fill=255)
    out.alpha_composite(fill_mask(m2, holo_rgb(w, h, 1.2)))
    m3 = Image.new("L", (w, h), 0)
    k = bw + max(2, bw // 3)
    ImageDraw.Draw(m3).rounded_rectangle([k, k, w - 1 - k, h - 1 - k], radius=max(2, r - k), fill=255)
    out.alpha_composite(solid(m3, fill))
    return out


def fit_h(img, h):
    return img.resize((max(1, round(img.width * h / img.height)), int(h)), Image.LANCZOS)


def fit_w(img, w):
    return img.resize((int(w), max(1, round(img.height * w / img.width))), Image.LANCZOS)


def fit_box(img, w, h):
    k = min(w / img.width, h / img.height)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def stack(items, gap, align="c"):
    w = max(i.width for i in items)
    h = sum(i.height for i in items) + gap * (len(items) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y = 0
    for i in items:
        x = (w - i.width) // 2 if align == "c" else 0
        out.alpha_composite(i, (x, y))
        y += i.height + gap
    return out


def row(items, gap):
    h = max(i.height for i in items)
    w = sum(i.width for i in items) + gap * (len(items) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for i in items:
        out.alpha_composite(i, (x, (h - i.height) // 2))
        x += i.width + gap
    return out


def on_canvas(img, p):
    out = Image.new("RGBA", (img.width + 2 * p, img.height + 2 * p), (0, 0, 0, 0))
    out.alpha_composite(img, (p, p))
    return out


# ================================================================= brand assets
def _L(path):
    return Image.open(path).convert("L")


def asset_arka(h, taimcafe=True):
    return blib.arka_logo(int(h), fill=(242, 206, 92), dark=INK_D, taimcafe=taimcafe)


def asset_simkart_tag(w, url=True):
    """Simkart wordmark on its own night plate (as on the brand site), chrome frame; url under the logo."""
    logo = blib.simkart_logo(int(w * 0.16), silver=(222, 220, 232), red=(236, 52, 40), glow=True)
    logo = fit_w(logo, int(w * 0.80))
    items = [logo]
    if url:
        u = ink_text("simkart.vercel.app", F_SPON(int(w * 0.07)), (236, 236, 244))
        items.append(u)
    inner = stack(items, int(w * 0.012))
    bw = max(4, int(w * 0.018))
    H = inner.height + int(w * 0.07) + 2 * bw
    plate = chrome_frame(int(w), H, int(H * 0.22), bw, NIGHT)
    plate.alpha_composite(inner, ((plate.width - inner.width) // 2, (plate.height - inner.height) // 2))
    return plate


def asset_smp(h, mono=None):
    w = _L(os.path.join(BX, "smp_racing_esports_white.png"))
    a = _L(os.path.join(BX, "smp_racing_esports_accent.png"))
    im = Image.new("RGBA", w.size, (0, 0, 0, 0))
    im.alpha_composite(solid(w, mono or WHITE))
    im.alpha_composite(solid(a, mono or (0, 181, 239)))
    return fit_h(im.crop(im.getbbox()), h)


def asset_mono(name, h, col):
    m = _L(os.path.join(BX, name))
    m = m.crop(m.getbbox())
    return fit_h(solid(m, col), h)


def asset_coat(h):
    sh = _L(os.path.join(BX, "saratov_coa_shield.png"))
    im = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    im.alpha_composite(solid(sh, (20, 30, 60)))
    im.alpha_composite(solid(sh.filter(ImageFilter.MinFilter(31)), (22, 148, 220)))
    im.alpha_composite(solid(_L(os.path.join(BX, "saratov_coa_crown.png")), (222, 178, 60)))
    im.alpha_composite(solid(_L(os.path.join(BX, "saratov_coa_crown_lines.png")), (20, 30, 60)))
    im.alpha_composite(solid(_L(os.path.join(BX, "saratov_coa_fish.png")), (232, 236, 244)))
    im.alpha_composite(solid(_L(os.path.join(BX, "saratov_coa_lines.png")), (20, 30, 60)))
    return fit_h(im.crop(im.getbbox()), h)


def asset_mbu(h):
    im = Image.open(os.path.join(BX, "edm_mbu_logo.png")).convert("RGBA")
    return fit_h(im.crop(im.getbbox()), h)


def asset_karting64(h, text_col=INK, flip=False):
    """Kart + pilot + flag from the KARTING64 logo (top part only), KARTING64.RU set in Tektur."""
    k = _L(os.path.join(BX, "karting64_black.png"))
    cut = 1500
    im = Image.new("RGBA", (k.width, cut), (0, 0, 0, 0))
    for n, c in (("white", WHITE), ("blue", (0, 57, 166)), ("red", (213, 43, 30)), ("black", INK_D)):
        im.alpha_composite(solid(_L(os.path.join(BX, f"karting64_{n}.png")).crop((0, 0, k.width, cut)), c))
    im = im.crop(im.getbbox())
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    kart = fit_h(im, int(h * 0.62))
    t = ink_text("KARTING64.RU", F_TECH(int(h * 0.42)), text_col)
    t = fit_w(t, int(kart.width * 1.05))
    return stack([kart, t], int(h * 0.05))


def asset_driveoil(h):
    return blib.driveoil_logo(int(h), plate=True)


def ru_flag(w, h, border=INK_D):
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    for i, c in enumerate([(255, 255, 255), (0, 57, 166), (213, 43, 30)]):
        d.rectangle([0, i * h // 3, w - 1, (i + 1) * h // 3 - 1], fill=c + (255,))
    d.rectangle([0, 0, w - 1, h - 1], outline=border + (255,), width=max(1, h // 22))
    return t


def saratov_flag(w, h):
    """Saratov oblast flag: white field, red band at the hoist with the coat of arms."""
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, w - 1, h - 1], fill=(255, 255, 255, 255))
    d.rectangle([0, 0, int(w * 0.38), h - 1], fill=(206, 30, 40, 255))
    c = asset_coat(int(h * 0.7))
    t.alpha_composite(c, (int(w * 0.19) - c.width // 2, (h - c.height) // 2))
    d.rectangle([0, 0, w - 1, h - 1], outline=INK_D + (255,), width=max(1, h // 22))
    return t


def number_plate(w, h, num="00"):
    """Race-number plate: burgundy price-tag plate, chrome frame, chrome Unbounded digits."""
    bw = max(5, int(h * 0.045))
    p = chrome_frame(w, h, int(h * 0.2), bw, INK)
    digits = chrome_text(num, F_NUM(int(h * 0.9)), max(3, int(h * 0.035)), outline=INK_D)
    digits = fit_box(digits, w * 0.80, h * 0.74)
    p.alpha_composite(digits, ((w - digits.width) // 2, (h - digits.height) // 2 + int(h * 0.01)))
    return p


def vet_stamp(w, lines, col=STAMP, seed=7):
    """Oval veterinary meat stamp (double ring, worn ink)."""
    h = int(w * 0.6)
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    lw = max(4, w // 34)
    d.ellipse([lw, lw, w - lw, h - lw], outline=255, width=lw)
    k = lw * 3
    d.ellipse([k, k, w - k, h - k], outline=255, width=max(2, lw // 2))
    sizes = [0.13, 0.25, 0.13]
    ys = [0.30, 0.52, 0.73]
    for s, y, f in zip(lines, ys, sizes):
        ft = F_STAMP(int(h * f)) if f < 0.2 else F_LABEL(int(h * f))
        d.text((w / 2, h * y), s, font=ft, fill=255, anchor="mm")
    rng = np.random.default_rng(seed)
    noise = Image.fromarray(((rng.random((h, w)) > 0.12) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.1))
    m = ImageChops.multiply(m, noise.point(lambda v: 255 if v > 110 else int(v * 1.6)))
    return solid(m.point(lambda v: int(v * 0.9)), col)


def cut_label(s, h_px, col=INK, underline=True):
    """Butcher cut label (Yeseva One) with a fine flourish underline: ——◆——."""
    t = ink_text(s, F_LABEL(int(h_px * 1.38)), col, track=0.06)
    t = fit_h(t, h_px)
    if not underline:
        return t
    lw = max(2, h_px // 14)
    u = Image.new("RGBA", (t.width, lw * 7), (0, 0, 0, 0))
    d = ImageDraw.Draw(u)
    c = t.width / 2
    d.line([(0, lw * 3.5), (c - lw * 4, lw * 3.5)], fill=col + (255,), width=lw)
    d.line([(c + lw * 4, lw * 3.5), (t.width, lw * 3.5)], fill=col + (255,), width=lw)
    d.polygon([(c - lw * 3, lw * 3.5), (c, 0), (c + lw * 3, lw * 3.5), (c, lw * 7)], fill=col + (255,))
    return stack([t, u], int(h_px * 0.12))


# ================================================================= source skin: body mask, baked shading
print("loading source skin + 4k posmap ...")
src = Image.open(os.path.join(SRC, "Skin.dds")).convert("RGB")
A0 = np.asarray(src).astype(np.float32)
mx = A0.max(-1)
logo = ndimage.binary_dilation(mx > 110, iterations=6)
rings = np.zeros_like(logo)
rings[720:790, 1950:2140] = True
logo &= ~rings
_, (iy, ix) = ndimage.distance_transform_edt(logo, return_indices=True)
A = A0.copy()
A[logo] = A[iy[logo], ix[logo]]
mx = A.max(-1)
BODY_COL = (A[..., 2] > A[..., 0] + 12) & (mx > 18)
SHADE = np.clip(0.55 + 0.45 * np.clip(mx / 76.0, 0, 1.12), 0.42, 1.06)

POS, NRM, PART, COV = posmap4k.load()
PART_NAMES = json.load(open(os.path.join(ZONES, "zones.json")))["part_names"]
PID = {n: i for i, n in enumerate(PART_NAMES)}
# paint where the source has body colour, plus the seam padding of skin islands (mip-map bleed)
PAINT = (BODY_COL | (~COV & (PART >= 0))) & ~rings
_, (sy, sx) = ndimage.distance_transform_edt(~(COV & BODY_COL), return_indices=True)
SHADE = np.where(COV & BODY_COL, SHADE, SHADE[sy, sx])
del sy, sx, iy, ix
POSF = POS.reshape(-1, 3)
NRMF = NRM.reshape(-1, 3)
PARTF = PART.reshape(-1)
PAINTF = PAINT.reshape(-1)

CAN = np.empty((N * N, 3), np.float32)          # design albedo (before baked shading)
CAN[:] = PIG
OBST = np.zeros(N * N, bool)                     # cut lines / keylines (obstacles for logo clearance)
INK_LAYER = np.zeros(N * N, np.float32)          # alpha of all logos (to keep lines off logos)

_idx_cache = {}


def cand(parts, side=None):
    key = (tuple(sorted(parts)), side)
    if key not in _idx_cache:
        m = np.isin(PARTF, [PID[p] for p in parts]) & PAINTF
        if side == "L":
            m &= POSF[:, 0] > 0.02
        elif side == "R":
            m &= POSF[:, 0] < -0.02
        _idx_cache[key] = np.flatnonzero(m)
    return _idx_cache[key]


SIDE_PARTS = ["front_bumper_corner", "front_fender", "front_fender_top", "front_door", "front_door_low", "rear_door",
              "rear_fender", "rear_shoulder", "sill", "rear_bumper_corner", "c_pillar", "a_pillar", "roof_rail",
              "mirror"]
ALL_PARTS = [p for p in PART_NAMES if p not in ("hidden",)]


# ================================================================= world-space paint
def blend(idx, rgb, a, obstacle=False):
    a = np.clip(a, 0, 1)[:, None]
    CAN[idx] = CAN[idx] * (1 - a) + np.asarray(rgb, np.float32) * a
    if obstacle:
        OBST[idx[a[:, 0] > 0.3]] = True


def smoothstep_aa(d, half_w, aa=0.0013):
    return np.clip((half_w - d) / aa + 0.5, 0, 1)


# -------- belly line B(y): top of the holographic lower band (follows door crease + wide-body flare)
B_PTS = [(-2.40, 0.315), (-1.72, 0.315), (-1.06, 0.40), (0.30, 0.40), (0.42, 0.418), (0.52, 0.442), (0.62, 0.478),
         (0.72, 0.528), (0.82, 0.598), (0.92, 0.676), (0.985, 0.728), (1.03, 0.47), (1.10, 0.45), (2.50, 0.45)]


def b_curve(y):
    ys, zs = np.array(B_PTS).T
    # monotone cubic (PCHIP) keeps the flare smooth and never overshoots
    from scipy.interpolate import PchipInterpolator
    f = PchipInterpolator(ys, zs)
    return f(y), f.derivative()(y)


def arc_y():
    yy = np.linspace(-2.5, 2.6, 6000)
    z, dz = b_curve(yy)
    s = np.concatenate([[0], np.cumsum(np.sqrt(1 + dz[1:] ** 2) * np.diff(yy))])
    return yy, s


_AY, _AS = arc_y()


def paint_belly():
    idx = np.flatnonzero(PAINTF & np.isin(PARTF, [PID[p] for p in ALL_PARTS]))
    P = POSF[idx]
    z, dz = b_curve(P[:, 1])
    k = np.sqrt(1 + dz ** 2)
    dist = (P[:, 2] - z) / k                        # signed perpendicular distance to B (+ above)
    # holo band below B: iridescent field from world position, slightly brighter toward the top edge
    t = 0.22 * P[:, 1] + 0.9 * P[:, 2] + 0.18 * np.abs(P[:, 0]) + 0.08 * np.sin(P[:, 1] * 3.1)
    col = holo_lookup(t * 0.9 + 0.1)
    lift = np.clip(1 + dist * 1.4, 0.82, 1.0)[:, None]          # deeper toward the sill
    col = col * lift + (1 - lift) * np.array([150, 90, 160], np.float32)
    band = smoothstep_aa(dist, 0.0)                  # dist<0 -> 1
    band = np.clip((-dist) / 0.0013 + 0.5, 0, 1)
    blend(idx, col, band)
    # chrome keyline straddling the band edge (8 mm), with a dark hairline under it for separation
    # lines stop where B dives steeply into the rear wheel arch (the band edge meets the arch lip there)
    flat = (np.abs(dz) < 2.5).astype(np.float32)
    blend(idx, INK_D, smoothstep_aa(np.abs(dist + 0.0002), 0.0056) * flat, obstacle=True)
    blend(idx, (248, 246, 255), smoothstep_aa(np.abs(dist), 0.0040) * flat, obstacle=True)
    # dashed butcher line 2.2 cm above the band edge, rounded dash ends, continuous around the car
    # arc-length parameter: along the side it is the arc of B(y); around the nose / tail corners the lateral
    # term continues it (sign follows the direction of travel, so dashes keep their length when the
    # line wraps forward around the front bumper corner instead of stretching into one long bar)
    s = belly_s(P)
    # over the rear door the dashed line leaves B and sweeps up into the shoulder line (connector below),
    # so it never ends in mid-panel
    keep = (flat > 0) & ~((P[:, 1] > BELLY_SPLIT) & (P[:, 1] < 1.10) & (P[:, 2] > 0.55))
    dash_line(idx[keep], s[keep], np.abs(dist - 0.022)[keep], 0.011, 0.075, 0.045, INK)


BELLY_SPLIT = 0.86


def belly_s(P):
    return np.interp(P[:, 1], _AY, _AS) + np.tanh(P[:, 1] / 0.5) * (0.96 - np.minimum(np.abs(P[:, 0]), 0.96))


def belly_connector(side, a_pts):
    """Dashed sweep that carries the belly cut line from the rear-door flare up into the shoulder line
    (tangent-continuous at the start, meets the shoulder line at a shallow angle), dash phase continuous."""
    y0 = BELLY_SPLIT
    z0, dz0 = b_curve(np.array([y0]))
    k0 = math.sqrt(1 + dz0[0] ** 2)
    # point on the dashed centre line (2.2 cm perpendicular above B) and its tangent
    py, pz = y0 - 0.022 * dz0[0] / k0, z0[0] + 0.022 / k0
    t0 = np.array([1.0, dz0[0]]) / k0
    ay, az = np.array(a_pts).T
    y3 = 1.18
    z3 = float(np.interp(y3, ay, az))
    t3 = np.array([1.0, 0.36]) / math.hypot(1.0, 0.36)
    P0, P3 = np.array([py, pz]), np.array([y3, z3])
    P1, P2 = P0 + t0 * 0.10, P3 - t3 * 0.09
    tt = np.linspace(0, 1, 60)[:, None]
    bz = ((1 - tt) ** 3) * P0 + 3 * ((1 - tt) ** 2) * tt * P1 + 3 * (1 - tt) * tt ** 2 * P2 + tt ** 3 * P3
    c3 = snap([tuple(q) for q in bz], "side", SIDE_PARTS, side)
    x0 = c3[0, 0]
    s0 = float(belly_s(np.array([[x0, py, pz]]))[0])
    curve_line(c3, SIDE_PARTS, side, 0.011, INK, dash=(0.075, 0.045), phase=s0, resnap_mode="side")


def dash_line(idx, s, d, half_w, on, off, col, phase=0.0):
    per = on + off
    loc = np.mod(s + phase, per)
    along = np.where(loc < on, 0.0, np.minimum(loc - on, per - loc))
    dd = np.sqrt(along ** 2 + d ** 2)
    a = smoothstep_aa(dd, half_w)
    blend(idx, col, a, obstacle=True)


# -------- 3D curves snapped to the surface (shoulder line, hood / roof cut lines)
_tree_cache = {}


def snap(points, mode, parts, side=None):
    """points: list of 2D coords; mode 'side' -> (y,z) finds outermost x on `side`;
    'top' -> (x,y) finds highest z. Returns (n,3) world points on the surface."""
    idx = cand(parts, side)
    P = POSF[idx]
    key = (mode, tuple(parts), side)
    if key not in _tree_cache:
        sel = np.arange(0, len(P), 2)
        q = P[sel][:, [1, 2]] if mode == "side" else P[sel][:, [0, 1]]
        _tree_cache[key] = (cKDTree(q), P[sel])
    tree, PS = _tree_cache[key]
    out = []
    for p in points:
        nb = tree.query_ball_point(p, 0.006)
        if not nb:
            _, nb = tree.query(p, k=8)
            nb = list(np.atleast_1d(nb))
        c = PS[nb]
        # keep the requested 2D coordinates exactly; take the depth from the outermost surface sheet
        if mode == "side":
            sx = c[:, 0] if side == "L" else -c[:, 0]
            sheet = sx > sx.max() - 0.008
            out.append((np.mean(c[sheet, 0]), p[0], p[1]))
        else:
            sheet = c[:, 2] > c[:, 2].max() - 0.008
            out.append((p[0], p[1], np.mean(c[sheet, 2])))
    return np.array(out)


def catmull3(pts, n=40):
    pts = np.asarray(pts, float)
    P = np.vstack([pts[0], pts, pts[-1]])
    res = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            res.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    res.append(P[-2])
    return np.array(res)


def curve_line(ctrl3d, parts, side, half_w, col, dash=None, phase=0.0, resnap_mode=None, ctrl2d=None):
    """Paint a 3D curve (through surface points) as a line of constant 3D width."""
    C = catmull3(ctrl3d, 60)
    if resnap_mode is not None:          # keep the dense curve glued to the surface
        if resnap_mode == "side":
            C = snap([(p[1], p[2]) for p in C], "side", parts, side)
        else:
            C = snap([(p[0], p[1]) for p in C], "top", parts, side)
        C[:, 0] = ndimage.uniform_filter1d(C[:, 0], 9, mode="nearest")
        C[:, 2] = ndimage.uniform_filter1d(C[:, 2], 9, mode="nearest")
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    S = np.concatenate([[0], np.cumsum(seg)])
    idx = cand(parts, side)
    P = POSF[idx]
    lo, hi = C.min(0) - 0.05, C.max(0) + 0.05
    m = np.all((P >= lo) & (P <= hi), axis=1)
    idx, P = idx[m], P[m]
    d, j = cKDTree(C).query(P, k=1, distance_upper_bound=0.05)
    ok = np.isfinite(d)
    idx, P, d, j = idx[ok], P[ok], d[ok], j[ok]
    # beyond the curve ends: no paint (rounded end only if dash)
    if dash:
        dash_line(idx, S[j], d, half_w, dash[0], dash[1], col, phase)
    else:
        blend(idx, col, smoothstep_aa(d, half_w), obstacle=True)
    return C


# ================================================================= decals (projected artwork) + checks
CHECKS = []


def frame(normal, up):
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    u = np.asarray(up, float)
    u = u - n * (u @ n)
    u /= np.linalg.norm(u)
    r = np.cross(u, n)
    return n, u, r


def decal(name, art, center, normal, up, ppm, parts, side=None, depth_tol=0.09, paint_angle=82, check_angle=48,
          min_clear_cm=1.0, check=True, obstacle_check=True, kind="logo", dry=False):
    """Project RGBA `art` (ppm pixels per metre) onto the body along `normal`, art 'up' = `up` (world).
    Returns painted texel indices. Records a check: footprint coverage, edge clearance, tilt, line clearance."""
    n, u, r = frame(normal, up)
    c = np.asarray(center, float)
    W, H = art.size
    wm, hm = W / ppm, H / ppm
    idx = cand(parts, side)
    P = POSF[idx]
    rel = P - c
    sa, sb, dep = rel @ r, rel @ u, rel @ n
    px = (sa / wm + 0.5) * W - 0.5
    py = (0.5 - sb / hm) * H - 0.5
    cosn = NRMF[idx] @ n
    m = (px > -2) & (px < W + 1) & (py > -2) & (py < H + 1) & (np.abs(dep) < depth_tol) & (cosn > math.cos(math.radians(paint_angle)))
    a = np.asarray(art.convert("RGBA")).astype(np.float32) / 255.0
    pre = a.copy()
    pre[..., :3] *= pre[..., 3:4]
    co = np.stack([py[m], px[m]])
    vals = [ndimage.map_coordinates(pre[..., ch], co, order=1, mode="constant", cval=0.0) for ch in range(4)]
    al = vals[3]
    rgb = np.stack(vals[:3], -1) / np.maximum(al[:, None], 1e-6) * 255.0
    sel = idx[m]
    keep = al > 0.002
    painted = sel[al > 0.05]
    if dry:
        return _check(name, art, a[..., 3], c, n, u, r, wm, hm, idx, sa, sb, dep, cosn, depth_tol, check_angle, painted,
                      parts, side, min_clear_cm, obstacle_check, kind)
    blend(sel[keep], rgb[keep], al[keep])
    if kind in ("logo", "text", "number"):
        INK_LAYER[painted] = 1.0
    if check:
        CHECKS.append(_check(name, art, a[..., 3], c, n, u, r, wm, hm, idx, sa, sb, dep, cosn, depth_tol,
                             check_angle, painted, parts, side, min_clear_cm, obstacle_check, kind))
    return painted


def _check(name, art, alpha, c, n, u, r, wm, hm, idx, sa, sb, dep, cosn, depth_tol, check_angle, painted, parts,
           side, min_clear_cm, obstacle_check, kind):
    g = 500.0                                       # check grid: 2 mm cells
    margin = 0.06
    Wg, Hg = int((wm + 2 * margin) * g), int((hm + 2 * margin) * g)
    gx = ((sa + wm / 2 + margin) * g).astype(int)
    gy = ((hm / 2 + margin - sb) * g).astype(int)
    ok = (gx >= 0) & (gx < Wg) & (gy >= 0) & (gy < Hg) & (np.abs(dep) < depth_tol)
    good = ok & (cosn > math.cos(math.radians(check_angle)))
    cov = np.zeros((Hg, Wg), bool)
    cov[gy[good], gx[good]] = True
    cov = ndimage.binary_closing(cov, iterations=2) | cov
    # footprint of the art in the same grid
    fp = Image.fromarray((alpha * 255).astype(np.uint8)).resize((max(1, int(wm * g)), max(1, int(hm * g))), Image.BILINEAR)
    fpa = np.zeros((Hg, Wg), bool)
    o = int(margin * g)
    fpa[o:o + fp.height, o:o + fp.width] = np.asarray(fp) > 25
    edt = ndimage.distance_transform_edt(cov) / g * 100      # cm to nearest non-surface cell
    clear_edge = float(edt[fpa].min()) - 0.2 if fpa.any() else 0.0
    covered = float(cov[fpa].mean()) if fpa.any() else 0.0
    # tilt: max angle between decal normal and surface normal under the footprint
    under = ok & fpa[np.clip(gy, 0, Hg - 1), np.clip(gx, 0, Wg - 1)]
    tilt = float(np.degrees(np.arccos(np.clip(cosn[under].min(), -1, 1)))) if under.any() else 0.0
    # texture-space: painted texels inside zone mask; clearance to mask edge (px -> cm) and to cut lines
    res = dict(name=name, kind=kind, parts=parts, side=side, size_cm=[round(wm * 100, 1), round(hm * 100, 1)],
               coverage=round(covered, 4), edge_clear_cm=round(clear_edge, 1), max_tilt_deg=round(tilt, 1))
    if len(painted):
        ys, xs = np.divmod(painted, N)
        y0, y1, x0, x1 = max(0, ys.min() - 120), min(N, ys.max() + 121), max(0, xs.min() - 120), min(N, xs.max() + 121)
        zm = np.isin(PART[y0:y1, x0:x1], [PID[p] for p in parts])
        inside = zm[ys - y0, xs - x0].mean()
        zedt = ndimage.distance_transform_edt(zm)
        res["in_zone_mask"] = round(float(inside), 4)
        res["mask_clear_cm"] = round(float(zedt[ys - y0, xs - x0].min()) / TPM * 100, 1)
        if obstacle_check:
            ob = OBST.reshape(N, N)[y0:y1, x0:x1]
            if ob.any():
                oedt = ndimage.distance_transform_edt(~ob)
                res["line_clear_cm"] = round(float(oedt[ys - y0, xs - x0].min()) / TPM * 100, 1)
            else:
                res["line_clear_cm"] = 99.0
    fails = []
    if covered < 0.999:
        fails.append("CLIPPED (footprint not fully on flat surface)")
    if clear_edge < min_clear_cm:
        fails.append(f"edge clearance {clear_edge:.1f} cm < {min_clear_cm}")
    if res.get("line_clear_cm", 99) < 0.8:
        fails.append("touches a cut line")
    res["status"] = "OK" if not fails else "FAIL: " + "; ".join(fails)
    return res


def decal_best(name, art, candidates, **kw):
    """Try candidate placements [(center, normal, up), ...] (dry run) and paint the one with the
    largest min(edge clearance, line clearance). Keeps the designer's order as tie-break."""
    best, bscore = None, -1e9
    for cand_ in candidates:
        c, n, u = cand_
        r = decal(name, art, c, n, u, dry=True, **kw)
        sc = min(r["edge_clear_cm"], r.get("line_clear_cm", 99)) if r["coverage"] >= 0.999 else -100 + r["coverage"]
        if sc > bscore + 0.05:
            best, bscore = cand_, sc
    c, n, u = best
    return decal(name, art, c, n, u, **kw)


def mean_normal(center, parts, side=None, radius=0.08):
    idx = cand(parts, side)
    P = POSF[idx]
    d = np.linalg.norm(P - np.asarray(center), axis=1)
    k = d < radius
    v = NRMF[idx[k]].mean(0)
    return v / np.linalg.norm(v)


def nflat(c, parts, side=None, r=0.12):
    """mean surface normal around c with the lateral component removed (keeps text level)."""
    v = mean_normal(c, parts, side, r)
    v[0] = 0
    return v / np.linalg.norm(v)


def surf_point(mode, a, b, parts, side=None):
    return snap([(a, b)], mode, parts, side)[0]


# ================================================================= LAYOUT
PPM = 2400          # art resolution for decals (px per metre) – 3x the texel density


def px(m):
    return int(round(m * PPM))


def paint_pearl():
    """Pearlescent flesh-pink base: warm peach-pink at the nose drifting to a cool lilac-pink at the tail,
    a touch lighter on the upper surfaces (Y2K pearl paint, subtle so the cut chart stays the hero)."""
    idx = np.flatnonzero(PAINTF & (PARTF >= 0))
    P = POSF[idx]
    nz = NRMF[idx, 2]
    t = np.clip((P[:, 1] + 2.1) / 4.4, 0, 1)[:, None]
    warm = np.array((246, 160, 168), np.float32)
    cool = np.array((236, 160, 206), np.float32)
    col = warm * (1 - t) + cool * t
    col = col + np.clip(nz, 0, 1)[:, None] * np.array((6, 8, 10), np.float32)
    CAN[idx] = col


def paint_body():
    print("base: pearl pink, holo belly band, keylines, butcher lines ...")
    paint_pearl()
    paint_belly()

    # --- shoulder cut line along the Audi tornado crease (headlight -> tail light), both sides
    A_PTS = [(-1.70, 0.800), (-1.55, 0.832), (-1.30, 0.842), (-1.05, 0.848), (-0.90, 0.855), (-0.60, 0.862),
             (0.00, 0.866), (0.60, 0.866), (0.92, 0.860), (1.06, 0.845), (1.30, 0.836), (1.60, 0.835), (1.86, 0.832)]
    for s in ("L", "R"):
        c3 = snap(A_PTS, "side", SIDE_PARTS, s)
        curve_line(c3, SIDE_PARTS, s, 0.011, INK, dash=(0.075, 0.045), resnap_mode="side")
        belly_connector(s, A_PTS)
        kl = snap([(y, z + 0.021) for y, z in A_PTS if y > -0.95] , "side", SIDE_PARTS, s)
        C = curve_line(kl, SIDE_PARTS, s, 0.0036, (250, 248, 255), resnap_mode="side")

    # --- hood / roof transverse cut lines (meet the shoulder crease on the fenders)
    TOP = ["hood", "front_fender_top", "front_fender", "roof", "roof_rail"]
    # parallel to the curved hood rear edge, 15 cm ahead of it, ending on the fender crease (shoulder line)
    hx = np.array([0.0, 0.3, 0.5, 0.6, 0.7])
    he = np.array([-1.116, -1.095, -1.048, -1.002, -0.923])
    xs = np.linspace(-0.70, 0.70, 29)
    pts = [(x, np.interp(abs(x), hx, he) - 0.15) for x in xs]
    pts = [(-0.78, -1.02)] + pts + [(0.78, -1.02)]
    hood = snap(pts, "top", TOP)
    curve_line(hood, TOP, None, 0.011, INK, dash=(0.075, 0.045), resnap_mode="top")
    for yl in (-0.24, 1.10):
        rl = snap([(x, yl) for x in np.linspace(-0.60, 0.60, 17)], "top", ["roof", "roof_rail"])
        curve_line(rl, ["roof", "roof_rail"], None, 0.011, INK, dash=(0.075, 0.045), resnap_mode="top")
        kl = snap([(x, yl + (0.024 if yl < 0 else -0.024)) for x in np.linspace(-0.60, 0.60, 17)], "top", ["roof", "roof_rail"])
        curve_line(kl, ["roof", "roof_rail"], None, 0.0036, (250, 248, 255), resnap_mode="top")


def side_decals(s):
    sg = 1 if s == "L" else -1
    nrm = (sg, 0, 0)
    up = (0, 0, 1)

    def at(y, z, parts):
        p = surf_point("side", y, z, parts, s)
        return (p[0], y, z)

    # race number plate – front door
    fd = ["front_door"]
    decal(f"number_door_{s}", number_plate(px(0.40), px(0.29)), at(-0.50, 0.628, fd), nrm, up, PPM, fd, s, kind="number")
    # main partner Симкарт – front door, behind the number
    tag = asset_simkart_tag(px(0.36))
    decal(f"simkart_door_{s}", tag, at(-0.055, 0.585, fd), nrm, up, PPM, fd, s)
    # main partner «арка · тайм-кафе» – rear door
    rd = ["rear_door"]
    ark = asset_arka(px(0.165))
    decal(f"arka_door_{s}", ark, at(0.47, 0.655, rd), nrm, up, PPM, rd, s)
    # cut labels
    decal(f"label_koreika_{s}", cut_label("КОРЕЙКА", px(0.036)), at(0.80, 0.757, rd), nrm, up, PPM, rd, s,
          kind="text", min_clear_cm=0.8)
    decal(f"label_grudinka_{s}", cut_label("ГРУДИНКА", px(0.034), underline=False), at(-0.47, 0.343, ["front_door_low"]),
          nrm, up, PPM, ["front_door_low"], s, kind="text", min_clear_cm=0.8)
    rf = ["rear_fender"]
    # (right side: kept behind the fuel-filler flap, which sits on the right rear fender at y~1.3-1.45)
    decal(f"label_okorok_{s}", cut_label("ОКОРОК", px(0.028), underline=False), at(1.36 if s == "L" else 1.64, 0.889, rf), nrm, up, PPM, rf, s,
          kind="text", min_clear_cm=0.8)
    # DriveOil – rear quarter behind the wheel
    rq = ["rear_fender", "rear_bumper_corner"]
    decal(f"driveoil_{s}", asset_driveoil(px(0.044)), at(1.83, 0.615, rq), nrm, up, PPM, rq, s)
    # sill: series + partners, burgundy ink on the holo band
    sl = ["sill"]
    # (sizes are the largest that keep >= 0.5 cm air to where the sill starts curving under the car)
    decal_best(f"sill_smp_{s}", asset_smp(px(0.042), mono=INK),
               [(at(yy, zz, sl), nrm, up) for yy in (-0.45, -0.50, -0.40) for zz in (0.229, 0.226, 0.232)],
               ppm=PPM, parts=sl, side=s, min_clear_cm=0.5)
    decal_best(f"sill_breng_{s}", asset_mono("br_engineering_black.png", px(0.036), INK),
               [(at(0.25, zz, sl), nrm, up) for zz in (0.235, 0.232, 0.238, 0.229)], ppm=PPM, parts=sl, side=s,
               min_clear_cm=0.5)
    decal_best(f"sill_karting64_{s}", asset_karting64(px(0.085), flip=(s == "L")),
               [(at(yy, zz, sl), nrm, up) for yy in (0.80, 0.82, 0.78) for zz in (0.240, 0.237, 0.243)],
               ppm=PPM, parts=sl, side=s, min_clear_cm=0.5)
    # sparkles (Y2K) around the number plate
    for (yy, zz, rr) in ((-0.28, 0.79, 0.020), (-0.72, 0.46, 0.014), (0.17, 0.48, 0.011)):
        decal(f"sparkle_{s}", sparkle(px(rr)), at(yy, zz, fd), nrm, up, PPM, fd, s, check=False, kind="fx")


def top_decals():
    # Hood: Симкарт night plate, read from the front
    hood = ["hood"]
    c = surf_point("top", 0.0, -1.46, hood)
    decal("simkart_hood", asset_simkart_tag(px(0.64)), c, mean_normal(c, hood), (0, 1, 0), PPM, hood, depth_tol=0.05)
    c = surf_point("top", 0.0, -1.19, hood)
    decal("label_sheika_hood", cut_label("ШЕЙКА", px(0.042)), c, mean_normal(c, hood), (0, 1, 0), PPM, hood, kind="text")
    # front fender tops: ЛОПАТКА (reads from the side, inboard of the crease)
    for s, sg in (("L", 1), ("R", -1)):
        ft = ["front_fender_top", "front_fender"]
        cands = []
        for yy in (-1.30, -1.36, -1.42, -1.48, -1.24):
            for xx in (0.80, 0.79, 0.81, 0.78):
                c = surf_point("top", sg * xx, yy, ft)
                cands.append((c, mean_normal(c, ft, s, 0.06), (-sg, 0, 0.0001)))
        decal_best(f"label_lopatka_{s}", cut_label("ЛОПАТКА", px(0.027), underline=False), cands, ppm=PPM, parts=ft,
                   side=s, kind="text", min_clear_cm=0.6, check_angle=40)
    # Roof: number (reads from the car's left side), vet stamp = ВЫРЕЗКА label, Saratov region block at the rear
    roof = ["roof"]
    c = surf_point("top", 0.0, 0.47, roof)
    decal("number_roof", number_plate(px(0.58), px(0.42)), c, (0, 0, 1), (-1, 0, 0), PPM, roof, kind="number")
    c = surf_point("top", 0.0, -0.045, roof)
    decal("stamp_vyrezka_roof", vet_stamp(px(0.38), ["ВЫСШИЙ СОРТ", "ВЫРЕЗКА", "ГОСТ · 64 · САРАТОВ"]), c, (0, 0, 1),
          (0, 1, 0), PPM, roof, kind="text")
    c = surf_point("top", 0.0, 0.93, roof)
    reg = row([asset_coat(px(0.15)), stack([ink_text("САРАТОВСКАЯ", F_TEAM(px(0.036)), INK),
                                             ink_text("ОБЛАСТЬ · 64", F_TEAM(px(0.036)), INK)], px(0.010))], px(0.03))
    decal("saratov_roof", reg, c, (0, 0, 1), (0, -1, 0), PPM, roof)
    # Y2K sparkles (accents only, kept off logos and lines)
    for (xx, yy, rr, pp) in ((0.27, 0.12, 0.030, roof), (-0.27, 0.81, 0.022, roof), (-0.33, 0.20, 0.016, roof),
                             (0.37, -1.30, 0.026, hood), (-0.37, -1.62, 0.018, hood)):
        c = surf_point("top", xx, yy, pp)
        decal("sparkle_top", sparkle(px(rr)), c, (0, 0, 1), (0, 1, 0), PPM, pp, check=False, kind="fx")
    # Trunk lid: «арка» reads from behind
    tr = ["trunk_lid"]
    c = surf_point("top", 0.0, 1.985, tr)
    decal("arka_trunk", asset_arka(px(0.13), taimcafe=False), c, mean_normal(c, tr, None, 0.05), (0, -1, 0), PPM, tr,
          depth_tol=0.06, check_angle=40)


def rear_front_decals():
    # rear panel between the tail lights: team + МБУ sticker
    rp = ["rear_panel"]
    n = (0, 1, 0)
    team = row([asset_mbu(px(0.125)), stack([ink_text("КОМАНДА", F_TEAM(px(0.040)), INK),
                                              ink_text("ЭДМ", F_TEAM(px(0.062)), INK)], px(0.008))], px(0.02))
    decal("team_edm_rear", team, (0.0, 2.17, 0.775), n, (0, 0, 1), PPM, rp, depth_tol=0.10)
    # rear bumper: Simkart url, big
    rb = ["rear_bumper"]
    c = (0.0, 2.25, 0.585)
    decal("url_simkart_rear", ink_text("simkart.vercel.app", F_SPON(px(0.070)), INK), c, nflat(c, rb), (0, 0, 1),
          PPM, rb, depth_tol=0.10)
    # front: ПЯТАЧОК (snout) on the hood nose ahead of the vent, reads from the front
    hood = ["hood"]
    # nose strip between the vent lip (separate grille mesh, front edge y=-1.951) and the hood front edge:
    # 2.4 cm letters keep 1 cm air to the lip and 1.8 cm to the edge
    c = surf_point("top", 0.0, -1.972, hood)
    decal("label_pyatachok", cut_label("ПЯТАЧОК", px(0.024), underline=False), c, mean_normal(c, hood, None, 0.05),
          (0, 1, 0), PPM, hood, kind="text", min_clear_cm=0.8, check_angle=40)


# ================================================================= glass_sticker.dds (1024, alpha kept)
GCHECKS = []


def gplace(g, name, art, box, cx=None, cy=None, rot=0.0):
    x0, y0, x1, y1 = box
    if rot:
        art = art.rotate(rot, expand=True, resample=Image.BICUBIC)
    cx = (x0 + x1) / 2 if cx is None else cx
    cy = (y0 + y1) / 2 if cy is None else cy
    ox, oy = int(round(cx - art.width / 2)), int(round(cy - art.height / 2))
    bb = art.getbbox()
    ax0, ay0, ax1, ay1 = ox + bb[0], oy + bb[1], ox + bb[2], oy + bb[3]
    clear = min(ax0 - x0, ay0 - y0, x1 - ax1, y1 - ay1)
    GCHECKS.append(dict(name=name, box=box, art_bbox=[ax0, ay0, ax1, ay1], clear_px=int(clear),
                        status="OK" if clear >= 2 else "FAIL: outside zone"))
    g.alpha_composite(art, (ox, oy))


def glass_sticker(driver):
    g = Image.open(os.path.join(SRC, "glass_sticker.dds")).convert("RGBA")
    S = 4                                     # draw at 4x, downsample at the end (crisp small text)
    G = Image.new("RGBA", (1024 * S, 1024 * S), (0, 0, 0, 0))
    G.alpha_composite(g.resize((1024 * S, 1024 * S), Image.NEAREST))
    d = ImageDraw.Draw(G)
    k = lambda b: [v * S for v in b]

    # windscreen banner: burgundy, chrome edge, SMP RACING ESPORTS + BR + РАФ (all white)
    d.rectangle(k([0, 0, 1024, 152]), fill=INK + (255,))
    d.rectangle(k([0, 140, 1024, 146]), fill=(250, 248, 255, 255))
    gplace(G, "ws_smp", asset_smp(62 * S), k([296, 22, 728, 128]))
    gplace(G, "ws_br", asset_mono("br_symbol_white.png", 52 * S, WHITE), k([200, 30, 290, 118]))
    gplace(G, "ws_raf", asset_mono("raf_black.png", 64 * S, WHITE), k([734, 24, 824, 122]))
    for x in (160, 864):
        gplace(G, "ws_sparkle", sparkle(15 * S), k([x - 32, 40, x + 32, 110]))

    # front number: windscreen passenger corner (zone rot -8.8)
    gplace(G, "ws_number", number_plate(150 * S, 110 * S), k([590, 308, 845, 532]), rot=-8.8)

    # rear window banner: Симкарт on night band, chrome edge (reads from behind)
    d.rectangle(k([135, 168, 864, 253]), fill=NIGHT + (255,))
    d.rectangle(k([135, 168, 864, 172]), fill=(250, 248, 255, 255))
    lg = fit_h(blib.simkart_logo(50 * S, glow=True), 54 * S)
    gplace(G, "rw_simkart", lg, k([140, 172, 864, 253]), cx=380 * S)
    gplace(G, "rw_url", ink_text("simkart.vercel.app", F_SPON(28 * S), (236, 236, 244)), k([140, 172, 864, 253]), cx=700 * S)

    # rear number: rear-window corner (zone rot -16.2)
    # (disc decal Circle.038, centre ~(732,682) r~136; plate pushed up toward the roof so more of it
    #  shows above the wing from a following car; rotation -16.1 deg computed from the mesh)
    gplace(G, "rw_number", number_plate(150 * S, 108 * S), k([586, 547, 878, 818]), cy=660 * S, rot=-16.2)

    # side windows: driver name + Russian flag on a burgundy strip (lower part of the rear door window)
    first, last = driver
    for nm, box in (("L", [115, 272, 513, 496]), ("R", [114, 518, 512, 741])):
        bx0, by0, bx1, by1 = box
        w, h = 360 * S, 70 * S
        strip = chrome_frame(w, h, 16 * S, 4 * S, INK)
        fl = ru_flag(54 * S, 36 * S)
        t1 = ink_text(first.upper(), F_NAME(22 * S), (255, 214, 232))
        t2 = ink_text(last.upper(), F_NAME(40 * S), WHITE)
        tx = stack([t1, t2], 4 * S, align="l")
        tx = fit_box(tx, w - 100 * S, h - 16 * S)
        strip.alpha_composite(fl, (18 * S, (h - fl.height) // 2))
        strip.alpha_composite(tx, (86 * S + (w - 104 * S - tx.width) // 2, (h - tx.height) // 2))
        gplace(G, f"name_{nm}", strip, k(box), cy=(by1 - 54) * S)

    # rear wing (ext_sticker): whole area opaque; top = pink with dashed cut line + «арка · тайм-кафе»
    wx0, wy0, wx1, wy1 = 100, 856, 945, 1010
    d.rectangle(k([wx0, wy0, wx1, wy1]), fill=tuple(int(v) for v in PIG) + (255,))
    # wing top in the mesh = x 105..940, y 863 (trailing edge) .. 1006 (leading edge), 570 px/m both ways
    for yl in (868, 1001):                                    # chrome keyline + ink dashes along both edges
        d.rectangle(k([wx0, yl - 1, wx1, yl + 1]), fill=(250, 248, 255, 255))
    x = 120
    while x < 925:
        d.rounded_rectangle(k([x, 873, min(x + 26, 925), 878]), radius=3 * S, fill=INK + (255,))
        d.rounded_rectangle(k([x, 991, min(x + 26, 925), 996]), radius=3 * S, fill=INK + (255,))
        x += 42
    # horizontal lockup so the main partner fills the wing span: «арка» + «· тайм-кафе ·»
    word = asset_arka(78 * S, taimcafe=False)
    tc = blib.outlined(blib.brand_mask("arka_taimcafe.png", 26 * S), (242, 206, 92), INK_D, 3 * S, INK_D, (-2 * S, 3 * S))
    arka = row([word, tc], 18 * S)
    gplace(G, "wing_arka", fit_box(arka, 600 * S, 94 * S), k([212, 882, 833, 987]), rot=180)
    for cx in (160, 885):
        gplace(G, "wing_sparkle", sparkle(18 * S), k([cx - 42, 886, cx + 42, 982]), rot=180)
    # endplates: burgundy, Saratov flag + «64»
    for nm, (ex0, ex1) in (("R", (110, 305)), ("L", (326, 521))):
        d.rectangle(k([ex0, 757, ex1, 859]), fill=INK + (255,))
        bl = row([saratov_flag(66 * S, 44 * S), chrome_text("64", F_NUM(44 * S), 3 * S)], 10 * S)
        gplace(G, f"endplate_{nm}", fit_box(bl, 150 * S, 64 * S), k([ex0 + 5, 766, ex1 - 5, 851]))
    out = G.resize((1024, 1024), Image.LANCZOS)
    # alpha: binary where we painted (alpha-tested material); keep original transparency elsewhere
    a = np.asarray(out.getchannel("A"))
    out.putalpha(Image.fromarray(np.where(a >= 128, 255, 0).astype(np.uint8)))
    return out


# ================================================================= output
def save_dxt5(im, path):
    im.convert("RGBA").save(path, pixel_format="DXT5")
    with open(path, "r+b") as fh:
        fh.seek(20)
        fh.write(struct.pack("<I", (im.width // 4) * (im.height // 4) * 16))
        fh.seek(88)
        fh.write(struct.pack("<I", 0))


DRIVERS = [("Pozdnyakov_00", ("Станислав", "Поздняков")), ("Konopelko_00", ("Матвей", "Конопелько"))]


def main():
    paint_body()
    print("decals ...")
    for s in ("L", "R"):
        side_decals(s)
    top_decals()
    rear_front_decals()

    out = A0.copy().reshape(-1, 3)
    pm = PAINTF
    out[pm] = CAN[pm] * SHADE.reshape(-1)[pm, None]
    skin = Image.fromarray(np.clip(out.reshape(N, N, 3), 0, 255).astype(np.uint8))
    skin.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, "texture_preview.png"))

    for folder, drv in DRIVERS:
        od = os.path.join(HERE, folder)
        os.makedirs(od, exist_ok=True)
        save_dxt5(skin, os.path.join(od, "Skin.dds"))
        n_before = len(GCHECKS)
        gl = glass_sticker(drv)
        if folder != DRIVERS[0][0]:          # second car: identical layout, keep only its name checks
            mine = [dict(g, name=g["name"] + "@" + folder) for g in GCHECKS[n_before:] if g["name"].startswith("name_")]
            del GCHECKS[n_before:]
            GCHECKS.extend(mine)
        save_dxt5(gl, os.path.join(od, "glass_sticker.dds"))
        if folder == DRIVERS[0][0]:
            gl.save(os.path.join(HERE, "glass_preview.png"))
        json.dump({"skinname": folder, "drivername": f"{drv[0]} {drv[1]}", "country": "Russia",
                   "team": "Команда ЭДМ", "number": "00", "priority": 1},
                  open(os.path.join(od, "ui_skin.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        livery_icon(od)
        with zipfile.ZipFile(os.path.join(HERE, folder + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(os.listdir(od)):
                z.write(os.path.join(od, f), folder + "/" + f)
    rep = []
    for r in CHECKS:
        rep.append(f"{r['status']:<6} {r['name']:<26} {str(r['size_cm']):<14} cov={r['coverage']:.3f} "
                   f"edge={r['edge_clear_cm']:>5}cm tilt={r['max_tilt_deg']:>5}° in_mask={r.get('in_zone_mask')} "
                   f"mask_clear={r.get('mask_clear_cm')}cm lines={r.get('line_clear_cm')}cm")
    for r in GCHECKS:  # glass checks of the first car (second car differs only in the name text)
        rep.append(f"{r['status']:<6} glass:{r['name']:<20} box={r['box']} art={r['art_bbox']} clear={r['clear_px']}px")
    txt = "\n".join(rep)
    print(txt)
    with open(os.path.join(HERE, "check_report.txt"), "w") as fh:
        fh.write(txt + "\n")
    json.dump(dict(skin=CHECKS, glass=GCHECKS), open(os.path.join(HERE, "check_report.json"), "w"),
              ensure_ascii=False, indent=1, default=str)

    print("done")


def livery_icon(od):
    """185x185 livery swatch: pink field, holo lower band with chrome keyline, dashed cut line, chrome 00."""
    S = 4
    w = 185 * S
    im = Image.new("RGBA", (w, w), tuple(int(v) for v in PIG) + (255,))
    band = holo_rgb(w, w // 3, 1.0, 0.2).convert("RGBA")
    im.alpha_composite(band, (0, w - w // 3))
    d = ImageDraw.Draw(im)
    d.rectangle([0, w - w // 3 - 5 * S, w, w - w // 3 + 1 * S], fill=(250, 248, 255, 255))
    x = 0
    while x < w:
        d.rounded_rectangle([x, w - w // 3 - 22 * S, x + 22 * S, w - w // 3 - 16 * S], radius=3 * S, fill=INK + (255,))
        x += 34 * S
    p = number_plate(120 * S, 80 * S)
    im.alpha_composite(p, ((w - p.width) // 2, 16 * S))
    im.alpha_composite(sparkle(14 * S), (w - 46 * S, 8 * S))
    im.convert("RGB").resize((185, 185), Image.LANCZOS).save(os.path.join(od, "livery.png"))


if __name__ == "__main__":
    main()
