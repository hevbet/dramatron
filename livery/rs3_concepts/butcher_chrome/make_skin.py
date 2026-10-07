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
HOLO = np.array([(255, 120, 200), (180, 140, 255), (110, 200, 255), (110, 245, 210), (255, 230, 150),
                 (255, 120, 200)], np.float32)


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
F_LABEL = lambda s: font("Podkova[wght].ttf", s, "ExtraBold")                   # butcher cut labels (heavy slab)
F_STAMP = lambda s: font("Podkova[wght].ttf", s, "ExtraBold")                   # vet stamp
F_SPON = lambda s: font("Exo2-Italic[wght].ttf", s, "ExtraBold Italic")         # urls / partner text
F_TEAM = lambda s: font("MontserratAlternates-BlackItalic.ttf", s)              # «Команда ЭДМ»


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


def asset_simkart_tag(w, url=True, pin=False):
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
    if pin:        # slim holo-on-chrome pinstripe around the plate, a pink gap between
        g, lw = int(w * 0.022), max(3, int(w * 0.007))
        W2, H2 = plate.width + 2 * (g + lw), plate.height + 2 * (g + lw)
        ring = Image.new("L", (W2, H2), 0)
        ImageDraw.Draw(ring).rounded_rectangle([0, 0, W2 - 1, H2 - 1], radius=int(H * 0.22) + g + lw, outline=255,
                                               width=lw)
        out = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
        out.alpha_composite(fill_mask(ring, holo_rgb(W2, H2, 2.0)))
        out.alpha_composite(plate, (g + lw, g + lw))
        return out
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


def asset_karting64(h, flip=False):
    """KARTING64 logo artwork ONLY (kart + pilot + RU-flag stripes), no lettering.
    Source layers are 7535x2682: artwork occupies rows 0..1461, the «KARTING64.RU» letters start at row 1550
    (slogan + underline below) -> cutting at row 1500 keeps zero letter pixels. Colours = the brand layers."""
    k = _L(os.path.join(BX, "karting64_black.png"))
    cut = 1500
    im = Image.new("RGBA", (k.width, cut), (0, 0, 0, 0))
    for n, c in (("white", WHITE), ("blue", (0, 57, 166)), ("red", (213, 43, 30)), ("black", INK_D)):
        im.alpha_composite(solid(_L(os.path.join(BX, f"karting64_{n}.png")).crop((0, 0, k.width, cut)), c))
    im = im.crop(im.getbbox())
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return fit_h(im, int(h))          # uniform scale: aspect ratio kept


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
    """Butcher cut label (Podkova ExtraBold, shop-sign slab), h_px = cap height; optional ——◆—— underline."""
    t = ink_text(s, F_LABEL(int(h_px * 1.45)), col, track=0.05)
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
# (behind the rear door the swoosh no longer dives vertically at the door / quarter shut line (y ~1.0): it crests
#  over the shut line onto the wide-body flare face and rolls down into the rear wheel-arch lip at y ~1.12-1.15,
#  so the holo fill and its chrome edge close against the arch instead of being cut by the seam; under the arch
#  (no surface) it drops back to the 0.45 band height of the rear quarter)
B_PTS = [(-2.40, 0.315), (-1.72, 0.315), (-1.06, 0.40), (0.30, 0.40), (0.42, 0.418), (0.52, 0.442), (0.62, 0.478),
         (0.72, 0.528), (0.82, 0.598), (0.92, 0.676), (0.985, 0.728), (1.035, 0.740), (1.085, 0.728), (1.13, 0.700),
         (1.20, 0.620), (1.40, 0.480), (1.60, 0.450), (2.50, 0.45)]


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
    # ~2.5 full spectrum cycles along one side
    t = 0.58 * P[:, 1] + 0.9 * P[:, 2] + 0.18 * np.abs(P[:, 0]) + 0.08 * np.sin(P[:, 1] * 3.1)
    col = holo_lookup(t + 0.1)
    lift = np.clip(1 + dist * 1.4, 0.82, 1.0)[:, None]          # deeper toward the sill
    col = col * lift + (1 - lift) * np.array([150, 90, 160], np.float32)
    band = smoothstep_aa(dist, 0.0)                  # dist<0 -> 1
    band = np.clip((-dist) / 0.0013 + 0.5, 0, 1)
    blend(idx, col, band)
    # chrome keyline straddling the band edge (8 mm), with a dark hairline under it for separation
    # lines stop where B dives steeply into the rear wheel arch (the band edge meets the arch lip there)
    # (the keyline follows B down the rear-arch dive so it runs into the arch lip instead of stopping short)
    flat = (np.abs(dz) < 2.5).astype(np.float32)
    blend(idx, INK_D, smoothstep_aa(np.abs(dist + 0.0003), 0.0092), obstacle=True)
    # 15 mm graded chrome: bright white edge -> grey core -> white edge
    u = np.clip(np.abs(dist) / 0.0075, 0, 1)
    chrome = np.interp(u, [0, 0.35, 0.7, 1.0], [150, 205, 250, 238])[:, None] * np.array((1.0, 0.98, 1.04), np.float32)
    blend(idx, np.clip(chrome, 0, 255), smoothstep_aa(np.abs(dist), 0.0075), obstacle=True)
    # dashed butcher line 2.2 cm above the band edge, rounded dash ends, continuous around the car
    # arc-length parameter: along the side it is the arc of B(y); around the nose / tail corners the lateral
    # term continues it (sign follows the direction of travel, so dashes keep their length when the
    # line wraps forward around the front bumper corner instead of stretching into one long bar)
    s = belly_s(P)
    # over the rear door the dashed line leaves B and sweeps up into the shoulder line (connector below),
    # so it never ends in mid-panel
    keep = (flat > 0) & ~((P[:, 1] > BELLY_SPLIT) & (P[:, 1] < 1.62) & (P[:, 2] > 0.46))
    # at the nose the dashed line does not run on across the bumper into the air intakes: it ends where the front
    # connector leaves it and turns up behind the headlight (closed corner of the chart, see front_connector)
    keep &= P[:, 1] >= FRONT_TURN_Y
    dash_line(idx[keep], s[keep], np.abs(dist - 0.022)[keep], 0.011, 0.075, 0.045, INK)


BELLY_SPLIT = 0.86
FRONT_TURN_Y = -1.86           # set from the front connector geometry (front_turn_y) before the belly is painted


def belly_s(P):
    return np.interp(P[:, 1], _AY, _AS) + np.tanh(P[:, 1] / 0.5) * (0.96 - np.minimum(np.abs(P[:, 0]), 0.96))


DASH = (0.075, 0.045)
DPER = DASH[0] + DASH[1]


def _wrap(v):
    """wrap a dash-phase difference into [-DPER/2, DPER/2)"""
    return (v + DPER / 2) % DPER - DPER / 2


def sweep(side, P0, t0, s_start, P3, t3, s_end, h0=0.10, h3=0.09, clip=None, extra=None, ymin=None):
    """Dashed cubic sweep in the side (y,z) plane from P0 (tangent t0, dash coordinate s_start) to P3 (tangent t3).
    The dash coordinate is stretched by a few percent so it arrives at P3 exactly in phase with s_end: where the
    sweep merges into another dashed line, the dashes coincide instead of doubling into a blob.
    `clip` = (n,3) centre-line of the line it merges into; sweep paint closer than 1.6 cm to it is dropped."""
    t0 = np.asarray(t0, float) / np.linalg.norm(t0)
    t3 = np.asarray(t3, float) / np.linalg.norm(t3)
    P0, P3 = np.asarray(P0, float), np.asarray(P3, float)
    ctrl = [P0, P0 + t0 * h0] + ([np.asarray(e, float) for e in extra] if extra else []) + [P3 - t3 * h3, P3]
    if extra:
        pts2 = catmull3([tuple(c) for c in [P0] + [np.asarray(e, float) for e in extra] + [P3]], 30)
    else:
        tt = np.linspace(0, 1, 60)[:, None]
        q0, q1, q2, q3 = ctrl
        pts2 = ((1 - tt) ** 3) * q0 + 3 * ((1 - tt) ** 2) * tt * q1 + 3 * (1 - tt) * tt ** 2 * q2 + tt ** 3 * q3
    c3 = snap([tuple(q) for q in pts2], "side", SIDE_PARTS, side)
    C = catmull3(c3, 60)
    C = snap([(p[1], p[2]) for p in C], "side", SIDE_PARTS, side)
    C[:, 0] = ndimage.uniform_filter1d(C[:, 0], 9, mode="nearest")
    L = float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1)))
    delta = L + _wrap(s_end - s_start - L)
    curve_line(c3, SIDE_PARTS, side, 0.011, INK, dash=DASH, phase=s_start, resnap_mode="side",
               scale=delta / L, clip=clip, ymin=ymin)
    return C


def belly_connector(side, a_pts, shoulder_C, y3=1.34):
    """Dashed sweep that carries the belly cut line from the rear-door flare up into the shoulder line
    (tangent-continuous at the start, merges into the shoulder line tangentially well behind the rear-door gap),
    dash phase continuous at both ends."""
    y0 = BELLY_SPLIT
    z0, dz0 = b_curve(np.array([y0]))
    k0 = math.sqrt(1 + dz0[0] ** 2)
    # point on the dashed centre line (2.2 cm perpendicular above B) and its tangent
    py, pz = y0 - 0.022 * dz0[0] / k0, z0[0] + 0.022 / k0
    t0 = np.array([1.0, dz0[0]])
    ay, az = np.array(a_pts).T
    z3 = float(np.interp(y3, ay, az))
    slope = (np.interp(y3 + 0.02, ay, az) - np.interp(y3 - 0.02, ay, az)) / 0.04
    x0 = snap([(py, pz)], "side", SIDE_PARTS, side)[0, 0]
    s0 = float(belly_s(np.array([[x0, py, pz]]))[0])
    S = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(shoulder_C, axis=0), axis=1))])
    s3 = float(np.interp(y3, shoulder_C[:, 1], S))
    sweep(side, (py, pz), t0, s0, (y3, z3), (1.0, slope), s3, h0=0.12, h3=0.16, clip=shoulder_C)


A_FRONT = (-1.45, 0.836)       # where the front connector hands over to the shoulder line
FRONT_EXTRA = [(-1.80, 0.352), (-1.705, 0.42), (-1.668, 0.53), (-1.656, 0.64), (-1.645, 0.725), (-1.612, 0.790),
               (-1.56, 0.823)]
FRONT_YB = -1.90


def front_turn_y():
    """y where the front connector has left the belly dash line (centre lines < 4 mm apart before it), moved forward
    into the middle of the nearest dash gap so neither line ends in a cut-off dash. The belly dashes stop there and
    the connector starts there: the cut line turns up behind the headlight instead of running on into the intake."""
    zb = float(b_curve(np.array([FRONT_YB]))[0][0]) + 0.022
    C = catmull3([(FRONT_YB, zb)] + FRONT_EXTRA + [A_FRONT], 30)
    k = int(np.argmax(C[:, 1] - zb > 0.004))
    y_sep = float(C[k, 0])
    ys = np.linspace(max(y_sep - DPER, FRONT_YB + 0.005), y_sep, 241)
    pts = snap([(y, zb) for y in ys], "side", SIDE_PARTS, "L")
    loc = np.mod(belly_s(pts), DPER)
    gap_mid = DASH[0] + DASH[1] / 2
    return float(ys[np.argmin(np.abs(loc - gap_mid))])


def front_connector(side, shoulder_C):
    """Closes the chart at the front: the shoulder line turns down behind the headlight, between the lamp and
    the front wheel arch, and merges into the belly dash on the front bumper corner (phase-matched)."""
    yb = FRONT_YB
    zb = float(b_curve(np.array([yb]))[0][0]) + 0.022
    xb = snap([(yb, zb)], "side", SIDE_PARTS, side)[0, 0]
    sb = float(belly_s(np.array([[xb, yb, zb]]))[0])
    # ends exactly at the first shoulder point, in phase with the shoulder dashes (their arc length 0)
    y1, z1 = A_FRONT
    # (no clip: the part that still runs on the belly line is identical to it - same dash phase - and both start at
    #  FRONT_TURN_Y, so the corner is one rounded turn and nothing is left running forward into the intake)
    sweep(side, (yb, zb), (1.0, 0.0), sb, (y1, z1), (1.0, 0.25), 0.0, extra=FRONT_EXTRA, ymin=FRONT_TURN_Y)


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


def curve_line(ctrl3d, parts, side, half_w, col, dash=None, phase=0.0, resnap_mode=None, ctrl2d=None, scale=1.0,
               clip=None, trim=None, taper=None, ymin=None):
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
    if clip is not None:                  # drop the part that runs onto the line this curve merges into
        dc, _ = cKDTree(clip).query(C, k=1)
        far = dc > 0.016
        keep = far[j]
        idx, P, d, j = idx[keep], P[keep], d[keep], j[keep]
    if ymin is not None:                  # nothing ahead of world y = ymin (measured on the curve, not the texel)
        keep = C[j, 1] >= ymin
        idx, P, d, j = idx[keep], P[keep], d[keep], j[keep]
    if trim is not None:                  # (s0, s1) arc-length window of this curve left unpainted
        keep = (S[j] < trim[0]) | (S[j] > trim[1])
        idx, P, d, j = idx[keep], P[keep], d[keep], j[keep]
    # beyond the curve ends: no paint (rounded end only if dash)
    if dash:
        dash_line(idx, S[j] * scale, d, half_w, dash[0], dash[1], col, phase)
    else:
        hw = half_w
        if taper is not None:             # (start, end) taper lengths: the line narrows to a point instead of stopping
            sj = S[j]
            hw = half_w * np.clip(np.minimum(sj / max(taper[0], 1e-6), (S[-1] - sj) / max(taper[1], 1e-6)), 0, 1)
        blend(idx, col, smoothstep_aa(d, hw), obstacle=True)
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
          min_clear_cm=1.0, check=True, obstacle_check=True, kind="logo", dry=False, keepout=None):
    """Project RGBA `art` (ppm pixels per metre) onto the body along `normal`, art 'up' = `up` (world).
    Returns painted texel indices. Records a check: footprint coverage, edge clearance, tilt, line clearance."""
    n, u, r = frame(normal, up)
    c = np.asarray(center, float)
    W, H = art.size
    wm, hm = W / ppm, H / ppm
    idx = cand(parts, side)
    R = math.hypot(wm / 2 + 0.07, hm / 2 + 0.07) + depth_tol
    bx = np.all(np.abs(POSF[idx] - c) < R, axis=1)          # cheap box prefilter (check margin included)
    idx = idx[bx]
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
                      parts, side, min_clear_cm, obstacle_check, kind, keepout)
    if check:      # (checked before this decal joins INK_LAYER, so its air to the other decals is measured)
        CHECKS.append(_check(name, art, a[..., 3], c, n, u, r, wm, hm, idx, sa, sb, dep, cosn, depth_tol,
                             check_angle, painted, parts, side, min_clear_cm, obstacle_check, kind, keepout))
    blend(sel[keep], rgb[keep], al[keep])
    if kind in ("logo", "text", "number"):
        INK_LAYER[painted] = 1.0
    return painted


def _check(name, art, alpha, c, n, u, r, wm, hm, idx, sa, sb, dep, cosn, depth_tol, check_angle, painted, parts,
           side, min_clear_cm, obstacle_check, kind, keepout=None):
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
        zedt = zone_edt(parts)
        inside = (zedt[painted] > 0).mean()
        res["in_zone_mask"] = round(float(inside), 4)
        res["mask_clear_cm"] = round(float(zedt[painted].min()) / TPM * 100, 1)
        if obstacle_check:
            # clearance to every painted line (dashes, keylines, band edge) measured in 3D, so lines that sit on
            # another UV island (e.g. the band keyline next to the sill logos) are seen too
            res["line_clear_cm"] = round(line_clear_3d(painted), 1)
        res["decal_clear_cm"] = round(decal_clear_3d(painted), 1)
    fails = []
    if covered < 0.999:
        fails.append("CLIPPED (footprint not fully on flat surface)")
    if clear_edge < min_clear_cm:
        fails.append(f"edge clearance {clear_edge:.1f} cm < {min_clear_cm}")
    if res.get("mask_clear_cm", 99) < 1.0:
        fails.append("touches zone mask edge")
    if res.get("line_clear_cm", 99) < 0.8:
        fails.append("touches a cut line")
    if keepout and len(painted):         # 3D keep-out (e.g. what the rear wing hides in the standard views)
        Pp = POSF[painted]
        ko = (Pp[:, 1].min() >= keepout.get("y_min", -9)) and (np.abs(Pp[:, 0]).max() <= keepout.get("x_abs_max", 9))
        res["keepout_ok"] = bool(ko)
        if not ko:
            fails.append("inside a keep-out (hidden by the wing in a standard view)")
    tmax = TILT_MAX.get(kind)
    if tmax is not None and tilt > tmax:
        fails.append(f"distortion risk (tilt {tilt:.1f} > {tmax})")
    res["status"] = "OK" if not fails else "FAIL: " + "; ".join(fails)
    return res


_zedt = {}


def zone_edt(parts):
    """texel distance to the edge of the zone mask of `parts` (whole texture, cached)"""
    key = tuple(sorted(parts))
    if key not in _zedt:
        _zedt[key] = ndimage.distance_transform_edt(np.isin(PART, [PID[p] for p in parts])).astype(np.float32).reshape(-1)
    return _zedt[key]


TILT_MAX = {"text": 12.0, "logo": 15.0, "number": 15.0}
_obst_tree = {}


_ink_tree = {}


def decal_clear_3d(painted):
    """3D air (cm) to every logo/text/number painted before this one."""
    ink = INK_LAYER > 0
    n = int(ink.sum())
    if n == 0:
        return 99.0
    if _ink_tree.get("n") != n:
        ii = np.flatnonzero(ink)
        _ink_tree["t"] = cKDTree(POSF[ii[:: max(1, len(ii) // 600000)]])
        _ink_tree["n"] = n
    pts = POSF[painted[:: max(1, len(painted) // 20000)]]
    d, _ = _ink_tree["t"].query(pts, k=1, distance_upper_bound=0.99)
    d = d[np.isfinite(d)]
    return float(d.min()) * 100 if len(d) else 99.0


def line_clear_3d(painted):
    if "t" not in _obst_tree or _obst_tree["n"] != int(OBST.sum()):
        ob = np.flatnonzero(OBST)
        _obst_tree["t"] = cKDTree(POSF[ob])
        _obst_tree["n"] = int(OBST.sum())
    pts = POSF[painted[:: max(1, len(painted) // 40000)]]
    d, _ = _obst_tree["t"].query(pts, k=1, distance_upper_bound=0.99)
    d = d[np.isfinite(d)]
    return float(d.min()) * 100 - 0.1 if len(d) else 99.0


def _gates(r, kw, tilt_gate, line_min, air):
    kind = kw.get("kind", "logo")
    tg = tilt_gate if tilt_gate is not None else TILT_MAX.get(kind, 90)
    ok = (r["coverage"] >= 0.999 and r["max_tilt_deg"] <= tg and r.get("mask_clear_cm", 99) >= max(1.0, air)
          and r["edge_clear_cm"] >= max(kw.get("min_clear_cm", 1.0), air) + 0.1 and r.get("line_clear_cm", 99) >= line_min
          and r.get("keepout_ok", True))
    sc = min(r["edge_clear_cm"], r.get("line_clear_cm", 99), r.get("mask_clear_cm", 99)) - 0.05 * r["max_tilt_deg"]
    return ok, sc


def decal_best(name, art, candidates, tilt_gate=None, line_min=1.0, air=0.0, deco_min=2.0, want=False, **kw):
    """Try candidate placements [(center, normal, up), ...] (dry run). Candidates that pass every gate
    (fully on surface, tilt <= gate, >= 1 cm to the zone-mask edge, >= line_min to cut lines) win, ranked by
    min(edge, line, mask clearance) with a small tilt penalty; designer order breaks ties.
    want=True -> dry run only: returns (passed, best candidate, report)."""
    best, bscore, bok, brep = None, -1e9, False, None
    for cand_ in candidates:
        c, n, u = cand_
        r = decal(name, art, c, n, u, dry=True, **kw)
        ok, sc = _gates(r, kw, tilt_gate, line_min, air)
        ok = ok and r.get("decal_clear_cm", 99) >= deco_min
        sc = min(sc, r.get("decal_clear_cm", 99) - deco_min + 1.0)
        if not ok:
            sc -= 1000
        if sc > bscore + 0.05:
            best, bscore, bok, brep = cand_, sc, ok, r
    if want:
        return bok, best, brep
    c, n, u = best
    return decal(name, art, c, n, u, **kw)


def decal_fit(name, make_art, sizes, make_cands, **kw):
    """Largest size (first in `sizes`) for which some candidate passes every gate; paints it.
    Falls back to the smallest size's best candidate (its check will then report the FAIL)."""
    # binary search over the (descending) sizes: feasibility is monotone in size
    res = {}

    def trial(i):
        if i not in res:
            art = make_art(sizes[i])
            res[i] = (art,) + decal_best(name, art, make_cands(sizes[i]), want=True, **kw)
        return res[i][1]
    lo, hi = 0, len(sizes) - 1           # find the smallest index i that passes
    if not trial(hi):
        pick = hi
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if trial(mid):
                hi = mid
            else:
                lo = mid + 1
        pick = lo
    art, ok, best, rep_ = res[pick]
    kw2 = {k: v for k, v in kw.items() if k not in ("tilt_gate", "line_min", "air", "deco_min")}
    print(f"   fit {name}: size {sizes[pick]} m -> {'OK' if ok else 'NO FIT'} tilt={rep_['max_tilt_deg']} "
          f"edge={rep_['edge_clear_cm']} mask={rep_.get('mask_clear_cm')} line={rep_.get('line_clear_cm')} "
          f"decals={rep_.get('decal_clear_cm')}", flush=True)
    c, n, u = best
    decal(name, art, c, n, u, **kw2)
    return sizes[pick], best


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
    Nn = NRMF[idx]
    t = np.clip((P[:, 1] + 2.1) / 4.4, 0, 1)[:, None]
    warm = np.array((240, 138, 160), np.float32)
    cool = np.array((226, 140, 200), np.float32)
    col = warm * (1 - t) + cool * t
    # baked pearl flip (view independent): side-facing faces drift to lilac/cyan, up-facing faces to peach/gold
    side = np.clip(np.abs(Nn[:, 0]) - 0.35, 0, 1)[:, None] / 0.65
    upf = np.clip(Nn[:, 2] - 0.30, 0, 1)[:, None] / 0.70
    col = col + side * np.array((-10, 2, 16), np.float32) + upf * np.array((14, 10, -8), np.float32)
    # low-frequency world-space mottling (+-3) so big panels are not one flat value
    nse = (np.sin(P[:, 1] * 2.3 + P[:, 2] * 3.1) + np.sin(P[:, 0] * 2.9 - P[:, 1] * 1.7 + 1.3)
           + np.sin(P[:, 2] * 4.1 + P[:, 0] * 1.1 + 0.7)) / 3.0
    col = col + nse[:, None] * np.array((3, 2.5, 3), np.float32)
    CAN[idx] = col
    # mirror shells: holo-on-chrome (Y2K)
    mi = np.flatnonzero(PAINTF & (PARTF == PID["mirror"]))
    if len(mi):
        Q = POSF[mi]
        h = holo_lookup(Q[:, 1] * 6.0 + Q[:, 2] * 4.0 + np.abs(Q[:, 0]) * 3.0)
        g = np.clip(0.55 + 0.45 * np.cos((Q[:, 2] - 1.07) * 60.0), 0, 1)[:, None]      # chrome bands
        CAN[mi] = h * 0.55 + np.array((250, 248, 255), np.float32) * 0.45 * g + h * 0.45 * (1 - g)


def paint_body():
    print("base: pearl pink, holo belly band, keylines, butcher lines ...")
    paint_pearl()
    global FRONT_TURN_Y
    FRONT_TURN_Y = front_turn_y()
    print(f"   front turn of the belly cut line at y = {FRONT_TURN_Y:.3f}")
    paint_belly()

    # --- shoulder cut line along the Audi tornado crease (headlight -> tail light), both sides
    # (rear fender, y >= 1.06: the line rides on the side-facing band 0.855-0.880 just under the tornado crease;
    #  probed normals: below 0.855 the wide-body flare top faces UP (nz ~0.95) and made the line look thin)
    A_PTS = [A_FRONT, (-1.30, 0.842), (-1.05, 0.848), (-0.90, 0.855), (-0.60, 0.862),
             (0.00, 0.866), (0.60, 0.867), (0.92, 0.867), (1.06, 0.867), (1.30, 0.868), (1.60, 0.868), (1.86, 0.867)]
    for s in ("L", "R"):
        c3 = snap(A_PTS, "side", SIDE_PARTS, s)
        C = curve_line(c3, SIDE_PARTS, s, 0.011, INK, dash=DASH, resnap_mode="side")
        belly_connector(s, A_PTS, C)
        front_connector(s, C)
        # chrome keyline above the whole shoulder line (6 mm half width), white core with grey edges
        kl = snap([(y, z + 0.021) for y, z in A_PTS], "side", SIDE_PARTS, s)
        # (tapers to a point at the front, where the cut line turns down behind the headlight, and into the tail lamp)
        curve_line(kl, SIDE_PARTS, s, 0.0072, (176, 172, 190), resnap_mode="side", taper=(0.30, 0.12))
        curve_line(kl, SIDE_PARTS, s, 0.0045, (252, 250, 255), resnap_mode="side", taper=(0.30, 0.12))

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


def pill(art, padx, pady, fill, line=(255, 214, 232), lw=None):
    """Rounded backer plate behind a logo (fill + thin hairline)."""
    w, h = art.width + 2 * padx, art.height + 2 * pady
    lw = lw or max(2, h // 26)
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=tuple(line) + (255,))
    d.rounded_rectangle([lw, lw, w - 1 - lw, h - 1 - lw], radius=h // 2 - lw, fill=tuple(fill) + (255,))
    t.alpha_composite(art, (padx, pady))
    return t


def side_cands(ys, zs, parts, s, w):
    """Candidate placements on a side panel. Normals are computed from the mesh around each centre with either
    the fore-aft (y) or the vertical (z) component removed, or purely lateral: all three keep the art's baseline
    exactly horizontal along the car (text parallel to the ground); the tilt gate then picks the flattest."""
    sg = 1 if s == "L" else -1
    out = []
    for y in ys:
        for z in zs:
            p = surf_point("side", y, z, parts, s)
            c = (p[0], y, z)
            mn = mean_normal(c, parts, s, max(0.05, min(0.25, w / 2)))
            for nv in ((sg, 0.0, 0.0), (mn[0], 0.0, mn[2]), (mn[0], mn[1], 0.0)):
                out.append((c, nv, (0, 0, 1)))
    return out


def side_decals(s):
    sg = 1 if s == "L" else -1

    def sc(ys, zs, parts, w):
        return side_cands(ys, zs, parts, s, w)

    fd, rd, rf, fl, sl = ["front_door"], ["rear_door"], ["rear_fender"], ["front_door_low"], ["sill"]
    kw = dict(ppm=PPM, side=s)
    # race number plate – front door; >= 6 cm clear pink to the shoulder line above and the belly line below
    _, nb = decal_fit(f"number_door_{s}", lambda w: number_plate(px(w), px(w * 0.674)), [0.46, 0.45, 0.44, 0.42],
                      lambda w: sc((-0.52, -0.54, -0.50), (0.645, 0.635, 0.655), fd, w),
                      parts=fd, kind="number", line_min=6.0, air=2.0, **kw)
    ny_ = nb[0][1]
    # main partner Симкарт – front door, behind the number (>= 7 cm air to the plate, inside the door)
    decal_fit(f"simkart_door_{s}", lambda w: asset_simkart_tag(px(w)), [0.42, 0.41, 0.40, 0.38, 0.36],
              lambda w: sc((ny_ + 0.23 + 0.07 + w / 2, ny_ + 0.23 + 0.09 + w / 2, ny_ + 0.23 + 0.11 + w / 2),
                           (0.625, 0.610, 0.640), fd, w),
              parts=fd, line_min=4.0, air=2.0, deco_min=6.5, **kw)
    # main partner «арка · тайм-кафе» – rear door
    decal_best(f"arka_door_{s}", asset_arka(px(0.165)), sc((0.455, 0.445, 0.465), (0.655, 0.665, 0.645), rd, 0.37),
               parts=rd, line_min=3.0, air=2.0, **kw)
    # ---- butcher cut labels (Podkova ExtraBold, ink), each centred in its cut, largest size that keeps air
    # (loin: the upper rear-door cut between Арка, the door handle (y 0.87-1.04, z 0.80-0.82), the swoosh and
    #  the shoulder line - the tightest cut on the car)
    decal_fit(f"label_koreika_{s}", lambda h: cut_label("КОРЕЙКА", px(h)),
              [0.044, 0.040, 0.037, 0.034, 0.031],
              lambda h: sc((0.775, 0.765, 0.785), (0.790, 0.782, 0.798), rd, h * 6),
              parts=rd, kind="text", tilt_gate=10.0, line_min=1.0, deco_min=2.5, min_clear_cm=1.0, **kw)
    decal_fit(f"label_grudinka_{s}", lambda h: cut_label("ГРУДИНКА", px(h), underline=False),
              [0.060, 0.055, 0.050, 0.046, 0.042, 0.038],
              lambda h: sc((-0.50, -0.48, -0.53), (0.316, 0.308, 0.324), fl, h * 7),
              parts=fl, kind="text", tilt_gate=10.0, line_min=1.0, deco_min=4.0, **kw)
    # ham: the haunch in front of the rear wheel = the flat lower rear-door strip under the door crease (z 0.29-0.38,
    # same strip as ГРУДИНКА on the front door; the rear quarter behind the wheel is only 21 cm wide)
    decal_fit(f"label_okorok_{s}", lambda h: cut_label("ОКОРОК", px(h), underline=False),
              [0.064, 0.060, 0.056, 0.052, 0.048, 0.044],
              lambda h: sc((0.60, 0.56, 0.64), (0.330, 0.322, 0.338), rd, h * 6),
              parts=rd, kind="text", tilt_gate=10.0, line_min=1.5, deco_min=4.0, min_clear_cm=1.5, **kw)
    # ---- partners on the holo band: dark / pearl backers so they read on the iridescence
    # SMP RACING ESPORTS (series) – full colour on a deep-ink pill, flat lower front door strip
    decal_fit(f"smp_low_{s}", lambda h: pill(asset_smp(px(h)), px(0.012), px(0.008), INK_D),
              [0.058, 0.054, 0.050, 0.046, 0.042],
              lambda h: sc((0.02, -0.04, 0.08), (0.316, 0.308, 0.324), fl, h * 5),
              parts=fl, tilt_gate=12.0, line_min=1.0, deco_min=6.0, **kw)
    # DriveOil – rear quarter behind the wheel (one panel: arch lip y~1.70 .. bumper seam y~1.95)
    rq = ["rear_fender"]
    decal_fit(f"driveoil_{s}", lambda h: asset_driveoil(px(h)), [0.065, 0.060, 0.056, 0.052, 0.048, 0.044],
              lambda h: sc((1.83, 1.825, 1.835), (0.615, 0.60, 0.63), rq, h * 3.6),
              parts=rq, tilt_gate=12.0, line_min=2.0, deco_min=4.0, depth_tol=0.03, **kw)
    # sill: BR ENGINEERING (white on a deep-ink pill) + KARTING64 artwork (pearl pad, real orientation)
    decal_fit(f"sill_breng_{s}", lambda h: pill(asset_mono("br_engineering_black.png", px(h), WHITE), px(0.012),
                                                px(0.005), INK_D), [0.045, 0.042, 0.039, 0.036, 0.033],
              lambda h: sc((0.10, 0.06, 0.14), (0.234, 0.229, 0.239), sl, h * 11),
              parts=sl, tilt_gate=13.0, line_min=1.0, min_clear_cm=0.5, **kw)
    decal_fit(f"sill_karting64_{s}", lambda h: pill(asset_karting64(px(h)), px(0.010), px(0.006), (250, 244, 250),
                                                    line=INK_D),
              [0.066, 0.062, 0.058, 0.054, 0.050],
              lambda h: sc((0.84, 0.80, 0.88), (0.236, 0.231, 0.241), sl, h * 4),
              parts=sl, tilt_gate=13.0, line_min=1.0, min_clear_cm=0.5, **kw)
    # karting64.ru as a separate url (same face as simkart.vercel.app), deep ink on the sill next to the kart logo
    decal_fit(f"url_karting64_{s}", lambda h: ink_text("karting64.ru", F_SPON(px(h)), INK_D),
              [0.052, 0.050, 0.048, 0.045, 0.042],
              lambda h: sc((0.52, 0.50, 0.54), (0.236, 0.232, 0.240), sl, h * 6),
              parts=sl, kind="text", tilt_gate=10.0, min_clear_cm=0.5, line_min=1.0, deco_min=5.0, **kw)
    # sparkles (Y2K) – kept >= 4 cm from the plates and logos, >= 1.5 cm from lines
    for k, (ys, zs, rr, pp) in enumerate((((-0.84, -0.82, -0.80), (0.74, 0.70, 0.66), 0.016, fd),
                                          ((0.90, 0.93, 0.87), (0.50, 0.47, 0.44), 0.012, rd))):
        place_fx(f"sparkle_{s}{k}", sparkle(px(rr)), sc(ys, zs, pp, rr * 5), parts=pp, **kw)


def place_fx(name, art, cands, **kw):
    """Y2K accent: painted only where it keeps >= 4 cm from every logo / number and >= 1.5 cm from lines."""
    ok, best, _ = decal_best(name, art, cands, kind="fx", line_min=1.5, deco_min=4.0, want=True, **kw)
    if ok:
        c, n, u = best
        decal(name, art, c, n, u, kind="fx", check=False, **kw)
    else:
        print("   (skipped accent", name, ")")


def top_cands(xs, ys, parts, up, normal="mean", side=None, r=0.10):
    out = []
    for x in xs:
        for y in ys:
            c = surf_point("top", x, y, parts, side)
            n = (0, 0, 1) if normal == "z" else mean_normal(c, parts, side, r)
            out.append((c, n, up))
    return out


def top_decals():
    # Hood: Симкарт night plate (+ thin holo pinstripe), read from the front; clear of the bonnet vent cut-out
    hood = ["hood"]
    decal_fit("simkart_hood", lambda w: asset_simkart_tag(px(w), pin=True), [0.64, 0.62, 0.60, 0.58],
              lambda w: top_cands((0.0,), (-1.43, -1.42, -1.44, -1.41, -1.45), hood, (0, 1, 0), r=0.15),
              ppm=PPM, parts=hood, depth_tol=0.05, air=1.5, line_min=2.0)
    # ШЕЙКА: the neck cut between the hood cut line and the windscreen
    decal_fit("label_sheika_hood", lambda h: cut_label("ШЕЙКА", px(h)), [0.075, 0.070, 0.065, 0.060, 0.055, 0.050],
              lambda h: top_cands((0.0,), (-1.19, -1.18, -1.20, -1.17, -1.21), hood, (0, 1, 0), r=0.08),
              ppm=PPM, parts=hood, kind="text", tilt_gate=10.0, line_min=1.0)
    # ЛОПАТКА (shoulder): on the side face of the front fender behind the wheel arch, under the shoulder cut line.
    # (it used to sit on the fender top, which rises ~6 deg toward the windscreen: projected there the word followed
    #  that slope - 9.5 deg tilt. On the side face the art is projected along a normal with its fore-aft component
    #  removed and up = world +z, so the baseline is exactly horizontal, like КОРЕЙКА / ОКОРОК.)
    for s in ("L", "R"):
        ff = ["front_fender"]
        decal_fit(f"label_lopatka_{s}", lambda h: cut_label("ЛОПАТКА", px(h)),
                  [0.036, 0.034, 0.032, 0.030, 0.028, 0.026],
                  lambda h, s=s, ff=ff: side_cands((-1.00, -0.99, -1.01, -0.98, -1.02), (0.770, 0.765, 0.775, 0.760),
                                                   ff, s, h * 7),
                  ppm=PPM, parts=ff, side=s, kind="text", tilt_gate=10.0, line_min=1.5, deco_min=3.0, min_clear_cm=1.0)
    # Roof – one reading direction for the whole roof: everything reads from BEHIND (TV helicopter / following
    # car), digit tops toward the nose. Order: number > region > stamp.
    roof = ["roof"]
    decal_fit("number_roof", lambda w: number_plate(px(w), px(w * 0.70)), [0.74, 0.72, 0.70, 0.66],
              lambda w: top_cands((0.0,), (0.40, 0.42, 0.38, 0.44), roof, (0, -1, 0), normal="z"),
              ppm=PPM, parts=roof, kind="number", air=6.0, line_min=6.0)
    decal_fit("stamp_vyrezka_roof", lambda w: vet_stamp(px(w), ["ВЫСШИЙ СОРТ", "ВЫРЕЗКА", "ГОСТ · 64 · САРАТОВ"]),
              [0.32, 0.30, 0.28],
              lambda w: top_cands((0.0,), (-0.10, -0.09, -0.11, -0.08), roof, (0, -1, 0), normal="z"),
              ppm=PPM, parts=roof, kind="text", line_min=1.5, deco_min=5.0)
    reg = lambda h: row([asset_coat(px(h)), stack([ink_text("САРАТОВСКАЯ", F_TEAM(px(h * 0.31)), INK),
                                                    ink_text("ОБЛАСТЬ · 64", F_TEAM(px(h * 0.31)), INK)],
                                                   px(h * 0.07))], px(h * 0.2))
    decal_fit("saratov_roof", reg, [0.20, 0.19, 0.18, 0.17, 0.16, 0.15],
              lambda h: top_cands((0.0,), (0.93, 0.92, 0.94, 0.91), roof, (0, -1, 0), normal="z"),
              ppm=PPM, parts=roof, line_min=1.5, deco_min=5.0)
    # Y2K sparkles (accents only, kept off logos and lines)
    for k, (xs, ys, rr, pp) in enumerate((((0.30, 0.32), (0.05, 0.0), 0.026, roof), ((-0.30, -0.32), (0.80, 0.84), 0.020, roof),
                                          ((0.37, 0.40), (-1.30, -1.26), 0.024, hood), ((-0.37, -0.40), (-1.62, -1.58), 0.018, hood))):
        place_fx(f"sparkle_top{k}", sparkle(px(rr)), top_cands(xs, ys, pp, (0, 1, 0), normal="z"), ppm=PPM, parts=pp)
    # Trunk lid: «арка» reads from behind. Only the rear strip of the lid is clear of the rear wing in every view:
    # the wing blade hides the lid ahead of y ~2.05 in both rear 3/4 views, and its two stays (x = +-0.20) cross the
    # lid from above / behind (measured with a z-buffer of the model in the 9 standard views). The logo therefore
    # sits between the stays (|x| <= 0.17) on y 2.055-2.135, the КРЕСТЕЦ label it displaces is dropped (it is not one
    # of the chart's brief labels; the rump is still drawn by the cut lines).
    tr = ["trunk_lid"]
    decal_fit("arka_trunk", lambda h: asset_arka(px(h), taimcafe=False),
              [0.075, 0.072, 0.070, 0.067, 0.064, 0.060, 0.056, 0.052],
              lambda h: top_cands((0.0,), (2.098, 2.095, 2.100, 2.092, 2.102, 2.090), tr, (0, -1, 0), r=0.04),
              ppm=PPM, parts=tr, depth_tol=0.06, check_angle=40, air=1.0,
              keepout=dict(y_min=2.052, x_abs_max=0.170))


def rear_front_decals():
    # rear panel between the tail lights: team + МБУ sticker
    rp = ["rear_panel"]
    n = (0, 1, 0)
    team = row([asset_mbu(px(0.140)), stack([ink_text("КОМАНДА", F_TEAM(px(0.040)), INK),
                                              ink_text("ЭДМ", F_TEAM(px(0.062)), INK)], px(0.008))], px(0.02))
    decal_best("team_edm_rear", team, [((0.0, 2.17, z), nflat((0.0, 2.17, z), rp), (0, 0, 1)) for z in (0.775, 0.785)],
               ppm=PPM, parts=rp, depth_tol=0.10)
    # rear bumper: Simkart url, as big as stays flat (< 12 deg) on the curved bumper
    rb = ["rear_bumper"]

    # (the vertical band of the bumper is z ~0.51-0.57; above it the bumper rolls over to the top)
    def rbc(h):
        out = []
        Q = POSF[cand(rb)]
        for z in (0.540, 0.545, 0.535, 0.550):
            m = (np.abs(Q[:, 0]) < 0.01) & (np.abs(Q[:, 2] - z) < 0.005)
            c = (0.0, float(Q[m][:, 1].max()), z)
            for nv in (nflat(c, rb), (0, 1, 0)):
                out.append((c, nv, (0, 0, 1)))
        return out
    decal_fit("url_simkart_rear", lambda h: ink_text("simkart.vercel.app", F_SPON(px(h)), INK),
              [0.062, 0.058, 0.055, 0.052, 0.048], rbc, ppm=PPM, parts=rb, depth_tol=0.10, kind="text",
              tilt_gate=12.0, line_min=1.5, deco_min=3.0)
    # front: ПЯТАЧОК (snout) on the hood nose strip ahead of the vent, reads from the front
    hood = ["hood"]

    def pc(h):
        out = []
        for y in (-1.972, -1.970, -1.974, -1.968, -1.976):
            c = surf_point("top", 0.0, y, hood)
            out.append((c, mean_normal(c, hood, None, 0.05), (0, 1, 0)))
        return out
    decal_fit("label_pyatachok", lambda h: cut_label("ПЯТАЧОК", px(h), underline=False),
              [0.034, 0.032, 0.030, 0.028, 0.026, 0.024], pc, ppm=PPM, parts=hood, kind="text", min_clear_cm=0.8,
              check_angle=40, tilt_gate=10.0, line_min=1.0)
    # (front bumper corners: no Симкарт line - outboard of the intakes the corner is ~10 cm wide and turns > 40 deg,
    #  every candidate failed the flatness / edge gates)


# ================================================================= glass_sticker.dds (1024, alpha kept)
GCHECKS = []


def gplace(g, name, art, box, cx=None, cy=None, rot=0.0, edge=4):
    """edge > 0: a deep-ink rim of `edge` (4x) px under the art, so the alpha-test cut (>=128 after the downsample)
    lands on a dark line instead of stair-stepping across pink / chrome."""
    x0, y0, x1, y1 = box
    if rot:
        art = art.rotate(rot, expand=True, resample=Image.BICUBIC)
    if edge:
        a = art.getchannel("A")
        rim = grow(pad(a.point(lambda v: 255 if v >= 128 else 0), edge), edge)
        base = solid(rim, INK_D)
        base.alpha_composite(art, (edge, edge))
        art = base
    cx = (x0 + x1) / 2 if cx is None else cx
    cy = (y0 + y1) / 2 if cy is None else cy
    ox, oy = int(round(cx - art.width / 2)), int(round(cy - art.height / 2))
    bb = art.getbbox()
    ax0, ay0, ax1, ay1 = ox + bb[0], oy + bb[1], ox + bb[2], oy + bb[3]
    clear = min(ax0 - x0, ay0 - y0, x1 - ax1, y1 - ay1)
    GCHECKS.append(dict(name=name, box=box, art_bbox=[ax0, ay0, ax1, ay1], clear_px=int(clear),
                        status="OK" if clear >= 2 else "FAIL: outside zone"))
    g.alpha_composite(art, (ox, oy))


_GLASS_FIT = {}


def glass_fit(mesh, bbox):
    """Affine fit texture px (1024) -> world (AC frame: x left, y up, z front) of the glass_sticker mesh `mesh`
    inside `bbox`, from the car's kn5. Returns the 3x3 matrix M with world = [tx, ty, 1] @ M."""
    key = (mesh, tuple(bbox))
    if key not in _GLASS_FIT:
        sys.path.insert(0, os.path.join(LIV, "tools"))
        import render_rs3
        _, _, meshes = render_rs3.load_scene(render_rs3.DEFAULT_KN5, True)
        rows, pts = [], []
        for m in meshes:
            if m["mat"] != "glass_sticker" or not m["name"].endswith("/" + mesh):
                continue
            tx, ty = (m["uv"][:, 0] % 1) * 1024, (m["uv"][:, 1] % 1) * 1024
            sel = (tx >= bbox[0]) & (tx <= bbox[2]) & (ty >= bbox[1]) & (ty <= bbox[3])
            rows.append(np.stack([tx[sel], ty[sel], np.ones(sel.sum())], 1))
            pts.append(m["pos"][sel])
        M, *_ = np.linalg.lstsq(np.concatenate(rows), np.concatenate(pts), rcond=None)
        _GLASS_FIT[key] = M
    return _GLASS_FIT[key]


def glass_level_rot(mesh, bbox):
    """PIL rotate angle that puts the art's baseline exactly level (zero world-up component) on the glass.
    The UV map there is conformal (texture axes 90.0 deg apart, equal mm/px), so a pure rotation is enough."""
    M = glass_fit(mesh, bbox)
    ex, ey = M[0], M[1]                      # world displacement per texture px along +x / +y
    a = math.degrees(math.atan2(ex[1], -ey[1]))
    return -a


def glass_sticker(driver):
    g = Image.open(os.path.join(SRC, "glass_sticker.dds")).convert("RGBA")
    S = 4                                     # draw at 4x, downsample at the end (crisp small text)
    G = Image.new("RGBA", (1024 * S, 1024 * S), (0, 0, 0, 0))
    G.alpha_composite(g.resize((1024 * S, 1024 * S), Image.NEAREST))
    d = ImageDraw.Draw(G)
    k = lambda b: [v * S for v in b]

    # windscreen banner: burgundy, chrome edge; SMP RACING ESPORTS | title partner Арка (yellow) | BR + РАФ
    # (visible banner on the mesh spans u ~ 115..909)
    d.rectangle(k([0, 0, 1024, 152]), fill=INK + (255,))
    d.rectangle(k([0, 140, 1024, 146]), fill=(250, 248, 255, 255))
    gplace(G, "ws_smp", fit_box(asset_smp(60 * S), 214 * S, 76 * S), k([150, 22, 376, 128]), edge=0)
    gplace(G, "ws_arka", fit_box(asset_arka(96 * S, taimcafe=False), 236 * S, 100 * S), k([392, 14, 640, 132]),
           edge=0)
    br = asset_mono("br_symbol_white.png", 52 * S, WHITE)
    raf = asset_mono("raf_black.png", 64 * S, WHITE)
    gplace(G, "ws_br_raf", row([br, raf], 34 * S), k([656, 22, 876, 128]), edge=0)

    # front number: windscreen passenger corner. Rotation computed from the glass mesh (Circle.077): -9.5 deg puts
    # the plate's top / bottom edges exactly level in 3D (the zone map's -8.8 is the mean over the zone, 0.7 deg off)
    ws_box = [590, 308, 845, 532]
    gplace(G, "ws_number", number_plate(150 * S, 110 * S), k(ws_box), rot=glass_level_rot("Circle.077", ws_box), edge=S)

    # rear window banner: Симкарт on night band, chrome edge (reads from behind)
    d.rectangle(k([135, 168, 864, 253]), fill=NIGHT + (255,))
    d.rectangle(k([135, 168, 864, 172]), fill=(250, 248, 255, 255))
    lg = fit_h(blib.simkart_logo(50 * S, glow=True), 54 * S)
    gplace(G, "rw_simkart", lg, k([140, 172, 864, 253]), cx=380 * S, edge=0)
    gplace(G, "rw_url", ink_text("simkart.vercel.app", F_SPON(28 * S), (236, 236, 244)), k([140, 172, 864, 253]),
           cx=700 * S, edge=0)

    # rear number: rear-window disc decal (Circle.038). Rotation computed from the mesh (-16.8 deg = exactly level).
    # The rear wing hides the disc below a line from ty ~683 (left) to ~715 (right) in the straight rear view and a
    # wing stay crosses it at tx ~850 (z-buffer of the model): the plate was 150x108 at cy 660 and lost its lower
    # third behind the blade, which made it look sheared. Largest plate (same 150:108 proportions) that is >= 6 px
    # inside both the disc and the visible part in rear / rear 3/4 views: 130x94 centred (672, 628).
    rw_box = [586, 547, 878, 818]
    gplace(G, "rw_number", number_plate(130 * S, 94 * S), k(rw_box), cx=672 * S, cy=628 * S,
           rot=glass_level_rot("Circle.038", rw_box), edge=S)

    # side windows: driver name + Russian flag on a burgundy strip (lower part of the rear door window)
    first, last = driver
    for nm, box in (("L", [115, 272, 513, 496]), ("R", [114, 518, 512, 741])):
        bx0, by0, bx1, by1 = box
        # (+18 %: 386 x 82 is the largest strip that keeps >= 2 px to the window zone box)
        w, h = 386 * S, 82 * S
        strip = chrome_frame(w, h, 18 * S, 4 * S, INK)
        fl = ru_flag(60 * S, 40 * S)
        t1 = ink_text(first.upper(), F_NAME(24 * S), (255, 214, 232))
        t2 = ink_text(last.upper(), F_NAME(50 * S), WHITE)
        tx = stack([t1, t2], 4 * S, align="l")
        tx = fit_box(tx, w - 108 * S, h - 16 * S)
        strip.alpha_composite(fl, (18 * S, (h - fl.height) // 2))
        strip.alpha_composite(tx, (92 * S + (w - 110 * S - tx.width) // 2, (h - tx.height) // 2))
        gplace(G, f"name_{nm}", strip, k(box), cy=(by1 - 60) * S, edge=S)

    # rear wing (ext_sticker): whole area opaque; top = pearl pink, body-scale dashed cut lines, chrome
    # trailing-edge keyline, «арка» big in the middle (тайм-кафе sub-line dropped: unreadable at TV distance)
    wx0, wy0, wx1, wy1 = 100, 856, 945, 1010
    # wing top in the mesh = x 105..940, y 863 (trailing edge) .. 1006 (leading edge), 570 px/m both ways
    gx = np.linspace(0, 1, (wx1 - wx0) * S)[None, :]
    gy = np.linspace(0, 1, (wy1 - wy0) * S)[:, None]
    # same pearl as the body at the tail on an up-facing surface: cool lilac-pink + peach flip
    wing = np.array((236, 148, 194), np.float32)[None, None, :] + (np.sin(gx * 9.0 + gy * 2.0) * 3.0)[..., None]
    G.alpha_composite(Image.fromarray(np.dstack([np.clip(wing, 0, 255).astype(np.uint8),
                                                 np.full(wing.shape[:2], 255, np.uint8)]), "RGBA"), (wx0 * S, wy0 * S))
    # chrome keyline on the trailing edge (graded white-grey-white), thin chrome on the leading edge
    ch = chrome_rgb((wx1 - wx0) * S, 8 * S).convert("RGBA")
    G.alpha_composite(ch, (wx0 * S, 863 * S))
    d.rectangle(k([wx0, 1001, wx1, 1003]), fill=(250, 248, 255, 255))
    # body-scale dashes: 7.5 cm on / 4.5 cm off = 43 / 26 px, 1.6 cm thick = 9 px, rounded
    x = 118
    while x < 927:
        x2 = min(x + 43, 927)
        d.rounded_rectangle(k([x, 874, x2, 883]), radius=4 * S, fill=INK + (255,))
        d.rounded_rectangle(k([x, 988, x2, 997]), radius=4 * S, fill=INK + (255,))
        x += 69
    # «арка»: ~ 60 % of the chord, >= 2 cm (11 px) air to the dashes
    gplace(G, "wing_arka", asset_arka(66 * S, taimcafe=False), k([300, 893, 745, 978]), rot=180, edge=0)
    for cx in (175, 870):
        gplace(G, "wing_sparkle", sparkle(17 * S), k([cx - 42, 894, cx + 42, 977]), rot=180, edge=0)
    # endplates: burgundy, Saratov flag + «64», holo band along both long edges
    for nm, (ex0, ex1) in (("R", (110, 305)), ("L", (326, 521))):
        d.rectangle(k([ex0, 757, ex1, 859]), fill=INK + (255,))
        hb = holo_rgb((ex1 - ex0) * S, 7 * S, 1.6).convert("RGBA")
        G.alpha_composite(hb, (ex0 * S, 757 * S))
        G.alpha_composite(hb, (ex0 * S, 852 * S))
        d.rectangle(k([ex0, 764, ex1, 765]), fill=(250, 248, 255, 255))
        d.rectangle(k([ex0, 850, ex1, 851]), fill=(250, 248, 255, 255))
        bl = row([saratov_flag(66 * S, 44 * S), chrome_text("64", F_NUM(44 * S), 3 * S)], 10 * S)
        gplace(G, f"endplate_{nm}", fit_box(bl, 150 * S, 64 * S), k([ex0 + 5, 768, ex1 - 5, 849]), edge=0)
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
