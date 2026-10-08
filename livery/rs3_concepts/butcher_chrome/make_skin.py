"""Concept 1 «Butcher Chart Chrome» — Audi RS3 LMS (smp_audi_rs3_lms).

Pink Pig 2.0: flesh-pink car divided into butcher "cuts" by burgundy dashed lines that follow the real
character lines of the body (Audi shoulder crease, door-bottom crease, wide-body flare), vintage shop-sign
labels (Yeseva One), plus a Y2K layer: holographic lower band, chrome keylines, chrome numbers, sparkles.

Everything that crosses panels is painted in WORLD SPACE (texel -> 3D position map at 4096), so lines are
continuous across door gaps and UV seams and logos/text are projected level to the ground with correct
aspect. Every logo/text decal is checked automatically (inside its zone mask, clearance to panel edge,
clearance to the cut lines, surface tilt) — see check_report.txt. Dashed lines are drawn in whole dashes only: a
dash that would be cut by an arch lip / trim notch or half-hidden behind a part mounted on the body (tow strap) is
left out whole and listed in the report; lines that meet another line end on the middle of one of its dashes.

Run:  python3 make_skin.py            (writes one skin folder per driver, named Surname_Number - Pozdnyakov_23/,
                                      Konopelko_00/ - their zips, the team zip ../butcher_chrome.zip,
                                      texture_preview.png, check_report.txt/.json; each skin folder also gets
                                      every file of gear/out/<skin>/ - calipers, driver gear, crew)
"""
import json
import math
import os
import shutil
import struct
import sys
import zipfile

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

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


SIMKART_URL = "simkarting.ru"     # Симкарт web address
K64_URL = "karting64.ru"          # KARTING64 web address


def asset_simkart_tag(w, url=True, pin=False):
    """Simkart wordmark on its own night plate (as on the brand site), chrome frame; url under the logo."""
    logo = blib.simkart_logo(int(w * 0.16), silver=(222, 220, 232), red=(236, 52, 40), glow=True)
    logo = fit_w(logo, int(w * 0.80))
    items = [logo]
    if url:
        # «simkarting.ru» (13 characters) is set at 0.078 w, 11 % larger than the old 18-character address (0.07 w):
        # it spans ~0.51 w (old 0.64 w) under the 0.80 w wordmark and the plate grows only ~3 % in height (the hood
        # plate keeps >= 2 cm to the hood cut line at its size); font size only (uniform glyph scale, no stretching)
        u = ink_text(SIMKART_URL, F_SPON(int(w * 0.078)), (236, 236, 244))
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


RACE_NUM = "00"        # number painted by the body decals in progress (set per driver in main from NUMBERS)
NUM_REF = "00"         # the digits every number is sized against (the approved plates were fitted to «00»)


def number_digits(w, h, num):
    """(digits art, x, y, scale) of `num` on a w x h plate. The reference «00» is fitted once (80 % of the plate
    width, 74 % of its height - as the approved plates) and every number is set at that SAME scale: same cap
    height, stroke and outline, i.e. the same optical size and the same top / bottom air as «00», never stretched
    (one uniform scale). It is centred across the plate on its own ink and keeps the font baseline of «00» (all
    digits share the cap height, the offset term only matters for a digit whose ink top differs). A number wider
    than 80 % of the plate at that scale would be scaled down uniformly (none is: «23» is narrower than «00»)."""
    f = F_NUM(int(h * 0.9))
    ow = max(3, int(h * 0.035))
    ref = chrome_text(NUM_REF, f, ow, outline=INK_D)
    k = min(w * 0.80 / ref.width, h * 0.74 / ref.height)
    art = ref if num == NUM_REF else chrome_text(num, f, ow, outline=INK_D)
    k = min(k, w * 0.80 / art.width)
    art = art.resize((max(1, round(art.width * k)), max(1, round(art.height * k))), Image.LANCZOS)
    y = (h - max(1, round(ref.height * k))) // 2 + int(h * 0.01) + round((f.getbbox(num)[1] - f.getbbox(NUM_REF)[1]) * k)
    return art, (w - art.width) // 2, y, k


def number_plate(w, h, num=None):
    """Race-number plate: burgundy price-tag plate, chrome frame, chrome Unbounded digits (see number_digits).
    The plate itself (size, frame, fill) is the same for every number."""
    num = RACE_NUM if num is None else num
    bw = max(5, int(h * 0.045))
    p = chrome_frame(w, h, int(h * 0.2), bw, INK)
    digits, x, y, _ = number_digits(w, h, num)
    p.alpha_composite(digits, (x, y))
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


def url_pair(h, col=INK, two_lines=False):
    """«simkarting.ru ◆ karting64.ru»: both web addresses in the partner face (Exo 2 ExtraBold Italic), same size, one
    baseline, a small ink diamond between them (Exo 2 has no ◆ glyph, so it is drawn: 0.64 x-height, centred on the
    x-height). two_lines=True: the two addresses stacked, same size."""
    f = F_SPON(h)
    if two_lines:
        return stack([ink_text(SIMKART_URL, f, col), ink_text(K64_URL, f, col)], int(h * 0.18))
    xh = -f.getbbox("x", anchor="ls")[1]
    gap = 1.7 * xh
    la, lb = f.getlength(SIMKART_URL), f.getlength(K64_URL)
    asc, desc = f.getmetrics()
    p = 8
    m = Image.new("L", (int(la + gap + lb) + 2 * p, asc + desc + 2 * p), 0)
    d = ImageDraw.Draw(m)
    d.text((p, p + asc), SIMKART_URL, font=f, fill=255, anchor="ls")
    d.text((p + la + gap, p + asc), K64_URL, font=f, fill=255, anchor="ls")
    cx, cy, r = p + la + gap / 2, p + asc - xh / 2, 0.32 * xh
    d.polygon([(cx - r, cy), (cx, cy - r), (cx + r, cy), (cx, cy + r)], fill=255)
    m = m.crop(m.getbbox())
    return solid(pad(m, 4), col)


def pig_tail(h, loops=3.0, a=0.30, R0=1.0, R1=0.72, w0=0.40, w1=0.19):
    """Curly pig tail laid out sideways as a coil (the cartoon corkscrew): a short root rising from the left runs
    tangentially into `loops` round curls (prolate trochoid x = a t + R sin t, y = -R cos t: the curls sit on top,
    each crossing the stroke once) that shrink slightly to a tapered, rounded tip on the right. The stroke is drawn
    in arc-length order, and where a later part of the stroke crosses over an earlier one the under-part gets a
    deep-ink edge, so every crossing reads as over/under. Chrome with a holo tint, holo hairline and deep-ink
    outline + soft shadow (the chrome lettering treatment). h = art height (px), aspect kept (~3.3 : 1)."""
    S = 4
    tend = 2 * math.pi * loops + 0.55 * math.pi
    th = np.linspace(0.0, tend, 3000)
    R = R0 + (R1 - R0) * th / tend
    cx, cy = a * th + R * np.sin(th), -R * np.cos(th)          # (y up) starts at the bottom heading +x
    tq = np.linspace(0, 1, 400)[:-1, None]
    q0, q1, q2 = np.array([-1.3, -R0 + 0.55]), np.array([-0.585, -R0]), np.array([0.0, -R0])
    stem = (1 - tq) ** 2 * q0 + 2 * (1 - tq) * tq * q1 + tq ** 2 * q2
    x, y = np.concatenate([stem[:, 0], cx]), np.concatenate([stem[:, 1], cy])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    sw = w0 + (w1 - w0) * (s / s[-1]) ** 0.9                  # stroke width, root -> tip
    h0 = h / 1.16                                             # (outline, shadow and padding add ~16 %)
    sc = (h0 * S) / ((y + sw / 2).max() - (y - sw / 2).min())
    ow = max(2, int(h0 * 0.05))
    mp = int(h0 * 0.12) * S
    X = (x - (x - sw / 2).min()) * sc + mp
    Y = ((y + sw / 2).max() - y) * sc + mp                    # image y down
    W, Hh = int(X.max() + sw.max() * sc + 2 * mp), int(Y.max() + sw.max() * sc + 2 * mp)
    um = np.full((Hh, W), -1.0, np.float32)                   # arc length of the topmost stroke part per pixel
    gy, gx = np.mgrid[0:Hh, 0:W]
    for i in range(0, len(X), max(1, len(X) // 1500)):
        r = sw[i] * sc / 2
        x0, x1, y0, y1 = int(max(0, X[i] - r - 1)), int(min(W, X[i] + r + 2)), int(max(0, Y[i] - r - 1)), \
            int(min(Hh, Y[i] + r + 2))
        disc = (gx[y0:y1, x0:x1] - X[i]) ** 2 + (gy[y0:y1, x0:x1] - Y[i]) ** 2 <= r * r
        um[y0:y1, x0:x1][disc] = s[i] * sc
    on = um >= 0
    oh = ow * S
    fp = np.hypot(*np.mgrid[-oh:oh + 1, -oh:oh + 1]) <= oh
    later = ndimage.grey_dilation(np.where(on, um, -1.0), footprint=fp)
    cross = on & (later - um > 5 * oh + 0.6 * sw.max() * sc)   # an earlier part right under a later part's edge
    m = Image.fromarray((on * 255).astype(np.uint8)).resize((W // S, Hh // S), Image.LANCZOS)
    cr = Image.fromarray((cross * 255).astype(np.uint8)).resize((W // S, Hh // S), Image.LANCZOS)
    bb = m.getbbox()
    m, cr = pad(m.crop(bb), ow * 2 + 8), pad(cr.crop(bb), ow * 2 + 8)
    ol = grow(m, ow)
    out = Image.new("RGBA", m.size, (0, 0, 0, 0))
    sh = ImageChops.offset(ol, max(2, ow // 2), max(3, ow // 2 + 1)).filter(ImageFilter.GaussianBlur(ow * 0.4))
    out.alpha_composite(solid(sh.point(lambda v: int(v * 0.45)), INK_D))
    out.alpha_composite(solid(ol, INK_D))
    out.alpha_composite(fill_mask(m, holo_rgb(m.width, m.height, 1.4)))
    inner = Image.fromarray(((ndimage.distance_transform_edt(np.asarray(m) > 127) > max(2, ow * 0.35)) * 255)
                            .astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    tint = Image.blend(chrome_rgb(m.width, m.height), holo_rgb(m.width, m.height, 0.9, 0.3), 0.28)
    out.alpha_composite(fill_mask(inner, tint))
    out.alpha_composite(solid(cr, INK_D))
    out = out.crop(out.getbbox())
    return fit_h(out, int(h))                                  # (uniform: aspect kept)


def tail_with_label(h):
    """Trunk art: curly chrome tail + «ХВОСТИК» cut label (cut_label: Podkova ExtraBold, ink, as ГРУДИНКА / ОКОРОК
    without the ——◆—— underline: the tail itself is the ornament); h = label cap height (m). The strip between the
    wing stays that every rear / top view sees is ~37 cm wide but only ~12 cm deep (see trunk_tail), and the word is
    7.8 x its cap height wide, so the tail sits centred ABOVE the word and is laid out sideways as a coil
    (pig_tail, ~3.3 : 1): the word and the tail together fill the strip in both directions - the tail art is
    TAIL_K x the cap height tall and spans ~2/3 of the word."""
    lab = cut_label("ХВОСТИК", px(h), underline=False)
    tail = pig_tail(px(h) * TAIL_K)
    return stack([tail, lab], px(h * 0.25))          # (a quarter cap of air between the tail and the word)


TAIL_K = 1.4


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
rings[720:790, 1950:2140] = True           # baked Audi rings on the rear panel (under the 3D chrome badge)
logo &= ~rings
_, (iy, ix) = ndimage.distance_transform_edt(logo, return_indices=True)
A = A0.copy()
A[logo] = A[iy[logo], ix[logo]]
# Audi rings box: the box is painted like its surroundings (it used to be left out of PAINT, so the source navy showed
# as a dark rectangle). The ring strokes are thin lines clearly darker than the local navy: (grey closing 11 px, > the
# 3-4 px stroke) - source gives the strokes as a soft alpha that is drawn back on top in deep ink (paint_audi_rings).
# The shading under the rings: inside the ring hull (the four discs + 5 px) every row is interpolated linearly
# between the clean source just left and right of the hull, so the pink runs on continuously - no darker lenses
# where the rings overlap (a closing keeps the source's darker anti-aliased navy there) and no box edge (outside the
# hull the box keeps the source itself, incl. the dark rim along the top edge of the panel).
_rp = 12
_rs = A0[720 - _rp:790 + _rp, 1950 - _rp:2140 + _rp]
_rl = _rs.mean(-1)
_rk = ndimage.grey_closing(_rl, size=(11, 11)) - _rl
RING_A = np.zeros(A0.shape[:2], np.float32)
RING_A[720:790, 1950:2140] = np.clip((_rk - 5.0) / 12.0, 0, 1)[_rp:-_rp, _rp:-_rp]
_lab, _n = ndimage.label(ndimage.binary_dilation(_rk > 2.0, iterations=2))
_big = 1 + int(np.argmax(ndimage.sum(np.ones_like(_rl), _lab, range(1, _n + 1))))
_hull = ndimage.binary_dilation(ndimage.binary_fill_holes(
    ndimage.binary_dilation((_rk > 2.0) & (_lab == _big), iterations=3)), iterations=2)
_rf = _rs.copy()
for _r in np.flatnonzero(_hull.any(1)):
    _c = np.flatnonzero(_hull[_r])
    _a, _b = int(_c.min()), int(_c.max())
    _t = ((np.arange(_a, _b + 1) - (_a - 2.5)) / (_b - _a + 5.0))[:, None]
    _rf[_r, _a:_b + 1] = _rs[_r, _a - 4:_a].mean(0)[None] * (1 - _t) + _rs[_r, _b + 1:_b + 5].mean(0)[None] * _t
A[720:790, 1950:2140] = _rf[_rp:-_rp, _rp:-_rp]
del _rs, _rl, _rk, _lab, _hull, _rf
mx = A.max(-1)
BODY_COL = (A[..., 2] > A[..., 0] + 12) & (mx > 18)
SHADE = np.clip(0.55 + 0.45 * np.clip(mx / 76.0, 0, 1.12), 0.42, 1.06)

POS, NRM, PART, COV = posmap4k.load()
PART_NAMES = json.load(open(os.path.join(ZONES, "zones.json")))["part_names"]
PID = {n: i for i, n in enumerate(PART_NAMES)}
# Paint coverage comes from the part map, not from the source colour: every texel of a body part's skin islands is
# painted (plus the seam padding of the islands, for mip-map bleed). The source-colour test (BODY_COL) used to leave
# out texels that are dark in the source - deep ambient occlusion on the inner faces of the rear flares and bumper
# returns, the door-handle recesses and undersides, the trunk-lid lip under the rings, the mirror bases, arch lips -
# so the source navy / black showed through as jagged dark patches. Left as in the source: the parts that are black
# plastic in the design (front splitter, rear diffuser band), the 'hidden' part, and the floor (the down-facing
# underside of the sill). The Audi rings box is painted too (its strokes come back as ink in paint_audi_rings).
PAINT_EXCL = ("hidden", "front_splitter", "rear_diffuser_band")
FLOOR = (PART == PID["sill"]) & (NRM[..., 2] < -0.7)
PAINT = BODY_COL | ((PART >= 0) & ~np.isin(PART, [PID[n] for n in PAINT_EXCL]) & ~FLOOR)
# baked shading of the newly covered texels: their own source shading (keeps the AO gradient of a recess) but never
# brighter than the nearest body-colour texel and never below SHADE_MIN - deep AO reads as dark pink, not black
SHADE_MIN = 0.55
_, (sy, sx) = ndimage.distance_transform_edt(~(COV & BODY_COL), return_indices=True)
SHADE = np.where(COV & BODY_COL, SHADE,
                 np.where(COV, np.maximum(np.minimum(SHADE, SHADE[sy, sx]), SHADE_MIN), SHADE[sy, sx]))
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
    global REAR_K
    REAR_K = rear_k_centred(P, dist)
    s = belly_s(P)
    # over the rear door the dashed line leaves B and sweeps up into the shoulder line (connector below),
    # so it never ends in mid-panel
    keep = (flat > 0) & ~((P[:, 1] > BELLY_SPLIT) & (P[:, 1] < 1.62) & (P[:, 2] > 0.46))
    # at the nose the dashed line does not run on along the bumper corner ahead of the front wheel (toward the air
    # intakes): it ends in the front wheel opening; the front of the chart is closed by the cut line along the
    # headlight (eyeliner_curve)
    keep &= P[:, 1] >= FRONT_ARCH_Y
    # Whole dashes only. Around the tail the line crosses the notches of the black plastic lower-trim fins (no
    # bumper skin there: the separate ext_plastic part fills them) and passes behind the tow-strap tab; at the
    # wheel arches it runs off the lip. A dash that loses part of its length to any of these would show as a stub,
    # so every such dash is left out whole and the line goes behind the part / across the opening in clear paint.
    d_c = np.abs(dist - 0.022)
    loc = np.mod(s, DPER)
    near = np.floor(s / DPER).astype(int) + ((loc >= DASH[0]) & (loc - DASH[0] > DPER - loc))
    key = near * 2 + (P[:, 0] > -0.05)        # (the centre dash of the tail, |x| < 4 cm, is one group)
    covf = COV.reshape(-1)[idx] & OUTER[idx]          # only the outer skin the renderer shows
    kmin = int(key.min())
    kspan = int(key.max()) - kmin + 1
    # (1) truncated: the dash body is checked in three strips across its width (lower / centre / upper 6 mm);
    #     each strip must run the full 7.5 cm with no hole > 1.2 cm (door shut lines are < 0.6 cm). A strip that
    #     is missing (the trim notch takes the lower half of a dash) or broken counts as a cut dash. Dashes at the
    #     hand-over to the rear-door connector are cut there on purpose (the connector carries them on): exempt.
    off = dist - 0.022
    body = keep & covf & (d_c < 0.009) & (loc < DASH[0])
    bi = np.flatnonzero(body)
    order = np.lexsort((loc[bi], key[bi]))
    bi = bi[order]
    kb, lb, ob = key[bi], loc[bi], np.digitize(off[bi], [-0.003, 0.003])
    cuts = np.flatnonzero(np.diff(kb)) + 1
    trunc = []
    for g in np.split(np.arange(len(bi)), cuts):
        if not len(g):
            continue
        ys = P[bi[g], 1]
        if (ys.max() > BELLY_SPLIT - 0.10) and (ys.min() < BELLY_SPLIT + 0.10):
            continue
        if ys.min() > 2.0 and np.abs(P[bi[g], 0]).min() < 0.005:
            continue                   # the tail centre dash: its two mirrored halves each run 0 .. on/2
        worst = 0.0
        for band in range(3):
            lg = lb[g][ob[g] == band]
            worst = max(worst, float(np.diff(np.concatenate([[0.0], lg, [DASH[0]]])).max()))
        if worst > 0.012:
            trunc.append(int(kb[g[0]]))
    # (2) partly hidden: in some standard view part of the dash faces the camera but a separate part (tow-strap
    #     tab, trim) is drawn in front of it while the rest of the same dash is visible -> it would read as a stub
    hid = set()
    base = np.flatnonzero(keep & OUTER[idx] & (d_c < 0.008) & (loc < DASH[0]))
    for v in VIEWS9:
        e = VISD["cam"][v][0]
        dv = e[None, :] - P[base]
        dv /= np.linalg.norm(dv, axis=1, keepdims=True)
        fc = np.einsum("ij,ij->i", dv, NRMF[idx[base]])
        ci = base[fc > 0.3]
        hb = hidden_by_part(P[ci], v)
        kh = key[ci]
        nh = np.bincount(kh[hb] - kmin, minlength=kspan)
        ns = np.bincount(kh[~hb] - kmin, minlength=kspan)
        for k in np.flatnonzero((nh > 3) & (ns > 25)):
            hid.add(int(k + kmin))
            HIDDEN_IN.setdefault(int(k + kmin), set()).add(v)
    hid = sorted(hid)
    bad = sorted(set(trunc) | set(hid))
    for k in bad:
        m = key == k
        m &= OUTER[idx] & (d_c < 0.008)
        BELLY_DROPPED.append(dict(dash=k // 2, side="L" if k % 2 else "R",
                                  why=("hidden in " + ",".join(sorted(HIDDEN_IN[k]))) if k in hid else "truncated",
                                  y=round(float(np.median(P[m, 1])), 3), x=round(float(np.median(P[m, 0])), 3)))
    LINE_NOTES.append(f"tail centre: one {DASH[0] * 100:.1f} cm dash centred on x=0 (lateral stretch {REAR_K:.4f})")
    print(f"   rear dash centred on the car (lateral stretch {REAR_K:.4f}); whole dashes left out: "
          f"{len(trunc)} truncated (arch lip / trim notch), {len(hid)} partly hidden (tow strap / trim)")
    for r in BELLY_DROPPED:
        print(f"      dropped dash {r['dash']}{r['side']} ({r['why']}) at x={r['x']:+.3f} y={r['y']:+.3f}")
    keep &= ~np.isin(key, bad)
    dash_line(idx[keep], s[keep], d_c[keep], 0.011, 0.075, 0.045, INK)


BELLY_DROPPED = []
LINE_NOTES = []
URL_NOTES = []
HIDDEN_IN = {}
VISD = {}           # renderer visibility of the 9 standard views (filled in paint_body)
OUTER = None        # texels visible in at least one standard view


BELLY_SPLIT = 0.86
FRONT_ARCH_Y = -1.32           # middle of the front wheel opening at belly height (set by front_arch_y)


REAR_K = 1.0                   # lateral stretch of the dash coordinate around the tail (set in paint_belly)


def belly_s(P):
    lat = np.tanh(P[:, 1] / 0.5) * (0.96 - np.minimum(np.abs(P[:, 0]), 0.96))
    return np.interp(P[:, 1], _AY, _AS) + np.where(P[:, 1] > 0, REAR_K, 1.0) * lat


def rear_k_centred(P, dist):
    """Around the tail the dash coordinate is mirror-symmetric in x, so the two halves of the bumper meet on the
    centreline with whatever phase they have there; when that is not a dash middle, the two half-dashes merge into
    one double-length dash (or a dash is cut into two stubs). Stretch the lateral term by a few percent so the
    centreline falls exactly on the middle of a dash: one normal 7.5 cm dash, centred on the car."""
    c = (P[:, 1] > 1.5) & (np.abs(P[:, 0]) < 0.006) & (np.abs(dist - 0.022) < 0.002)
    if not c.any():
        return 1.0
    yc = float(np.median(P[c, 1]))
    arc = float(np.interp(yc, _AY, _AS))
    lat = math.tanh(yc / 0.5) * 0.96
    s0 = arc + lat
    want = DASH[0] / 2
    delta = _wrap(want - (s0 % DPER))
    return (lat + delta) / lat


VIEWS9 = ["front34_left", "front34_right", "side_left", "side_right", "rear34_left", "rear34_right", "front", "rear",
          "top"]
VIS_CACHE = os.path.join(SCRATCH, "bc", "visibility_9views_v2.npz")


def view_visibility():
    """What the renderer shows in each of the 9 standard views (its own cameras and rasteriser, opaque pass, ss=2):
    'vis'  {view: bool (N*N)}  Skin.dds texels that appear in the view,
    'zns'  {view: (Hs,Ws) float32}  depth of the front-most NON-skin surface per pixel (trim, tow strap ...), inf
           where the pixel shows the body or nothing,
    'cam'  {view: (eye, f, rgt, up, mx, my, fpx, Ws, Hs)} in the posmap frame, to project texels into the view.
    Geometry only, so cached in the scratchpad."""
    if os.path.exists(VIS_CACHE):
        d = np.load(VIS_CACHE, allow_pickle=True)
        return dict(vis={v: np.unpackbits(d["vis_" + v])[:N * N].astype(bool) for v in VIEWS9},
                    zns={v: d["zns_" + v] for v in VIEWS9}, cam={v: tuple(d["cam_" + v]) for v in VIEWS9})
    sys.path.insert(0, os.path.join(LIV, "tools"))
    import render_rs3 as R
    sc = R.Scene(R.DEFAULT_KN5, SRC, interior=False)
    skin_mid = sc.mat_names.index("skin")
    W, H, ss, margin = 1600, 900, 2, 0.07
    vis, zns, cam = {}, {}, {}
    to_pm = lambda v: np.array([v[0], -v[2], v[1]])            # render (x left, y up, z front) -> posmap axes
    for name in VIEWS9:
        eye, f, rgt, up = R.view_camera(name)
        rel = sc.P - eye
        Dv = rel @ f
        cx_, cy_ = rel @ rgt / Dv, rel @ up / Dv
        used = np.unique(sc.T.ravel())
        ux, uy = cx_[used], cy_[used]
        Ws, Hs = W * ss, H * ss
        fpx = min(Ws * (1 - 2 * margin) / (ux.max() - ux.min()), Hs * (1 - 2 * margin) / (uy.max() - uy.min()))
        mx, my = (ux.min() + ux.max()) / 2, (uy.min() + uy.max()) / 2
        X = (cx_ - mx) * fpx + Ws / 2
        Y = -(cy_ - my) * fpx + Hs / 2
        opaque = ~sc.is_glass[sc.TM] & sc.tri_ok
        gid = np.nonzero(opaque)[0]
        zb, tid, B1, B2 = R.rasterize(X, Y, Dv, sc.T[opaque], sc.bias[sc.TM][opaque], Ws, Hs, None)
        tid = np.where(tid >= 0, gid[np.maximum(tid, 0)], -1)
        hit = tid >= 0
        isskin = np.zeros_like(hit)
        isskin[hit] = sc.TM[tid[hit]] == skin_mid
        z = np.where(hit & ~isskin, zb, np.inf).astype(np.float32).reshape(Hs, Ws)
        tt, b1, b2 = tid[isskin], B1[isskin], B2[isskin]
        tri = sc.T[tt]
        uv = (sc.UV[tri[:, 0]] * (1 - b1 - b2)[:, None] + sc.UV[tri[:, 1]] * b1[:, None]
              + sc.UV[tri[:, 2]] * b2[:, None])
        tx = np.clip(((uv[:, 0] % 1) * N).astype(int), 0, N - 1)
        ty = np.clip((((1 + uv[:, 1]) % 1) * N).astype(int), 0, N - 1)
        m = np.zeros((N, N), bool)
        m[ty, tx] = True
        # a 1.7 mm pixel lands on every 2nd-3rd texel of a foreshortened panel: close the sampling holes
        vis[name] = ndimage.binary_dilation(m, iterations=3).reshape(-1)
        zns[name] = z
        e = to_pm(eye) + np.array([0, 0, 0.07237756])          # the kn5 node transform puts the posmap 7.24 cm higher
        cam[name] = (e, to_pm(f), to_pm(rgt), to_pm(up), mx, my, fpx, Ws, Hs)
    os.makedirs(os.path.dirname(VIS_CACHE), exist_ok=True)
    blob = {}
    for v in VIEWS9:
        blob["vis_" + v] = np.packbits(vis[v])
        blob["zns_" + v] = zns[v]
        blob["cam_" + v] = np.array(cam[v], dtype=object)
    np.savez_compressed(VIS_CACHE, **blob)
    return dict(vis=vis, zns=zns, cam=cam)


def hidden_by_part(pts, view):
    """True where a world point lies 0.5-10 cm behind a non-skin part in `view`: a part mounted right on the body
    (tow-strap tab, trim fin). Things further in front (a wheel seen across the car) are ordinary perspective."""
    e, f, rgt, up, mx, my, fpx, Ws, Hs = VISD["cam"][view]
    rel = pts - e
    dv = rel @ f
    X = ((rel @ rgt) / dv - mx) * fpx + Ws / 2
    Y = -((rel @ up) / dv - my) * fpx + Hs / 2
    xi, yi = np.floor(X).astype(int), np.floor(Y).astype(int)
    ok = (xi >= 0) & (xi < Ws) & (yi >= 0) & (yi < Hs)
    out = np.zeros(len(pts), bool)
    zp = VISD["zns"][view][yi[ok], xi[ok]]
    out[ok] = (zp < dv[ok] - 0.005) & (zp > dv[ok] - 0.10)
    return out


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
    C = snap([(p[1], p[2]) for p in C], "side", SIDE_PARTS, side, exact=True)
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


# -------- the shoulder cut line's path: the Audi «tornado» crease, extracted from the mesh (not drawn by eye)
# Vertical sections of the skin mesh every 1 cm from the headlight to the tail light, on each side: the outer profile
# (outermost |x| per z, 0.5 mm), its surface elevation and the convex curvature d(elevation)/d(arc) (deg per mm of
# surface). The crease is the ridge with the sharpest turn of the normal:
#   doors + rear quarter (y >= -0.83): the tornado line - the sharp edge on top of the shadowed under-cut band
#     (~5-8 deg/mm), rising from z 0.848 at the front door to 0.908 over the rear quarter and into the tail lamp;
#   front fender (y <= -0.885): the wide-body flare's top edge (vertical side face -> up-facing flare top, ~7-9
#     deg/mm), the edge the side views show running from the headlight back to the door (z 0.79-0.81).
# Each ridge is followed station to station (nearest height, short gaps such as the fuel flap edges bridged), smoothed
# (Gaussian, sigma 4 cm). Across the fender / door shut line the two edges are 5 cm apart in height and 12 cm apart in
# depth (the bolt-on flare is wider than the door) below z ~0.82, no surface ridge in between: the path blends from
# one to the other with one smooth S (smootherstep over CREASE_BLEND). The blend is finished just behind the shut
# line, so the line crosses it at z ~0.845, where the fender's rear end is flush with the door (no 12 cm step under
# a dash). On the front fender the crease runs on forward and down in front of the flare into the bumper's feature
# edge under the headlight - the front cut line follows that edge (eyeliner_curve).
CREASE_STEP = 0.01
CREASE_Y = (-1.60, 1.90)
CREASE_FENDER_END = -0.885     # last fender station (its rear edge at the door shut line, y ~ -0.86)
CREASE_DOOR_START = -0.83      # first front-door station
CREASE_BLEND = (-1.08, -0.80)  # S-blend fender flare edge -> door crease, done behind the shut line
SHOULDER_OFF = 0.0             # line centre -> crease (m, along the surface, + up): the centre line sits ON the crease
SH_REAR_AIR = 0.031            # centre line of the shoulder line's last dash end -> tail lamp (cap air >= 2 cm)
CREASE = {}                    # per side: the extracted path + raw ridge heights (report)
_MESH = {}


def _mesh():
    if not _MESH:
        D = np.load(os.path.join(ZONES, "_analysis.npz"))
        pt = np.load(os.path.join(ZONES, "part_tri.npy"))
        _MESH.update(P=D["P"].astype(float), N=D["N"].astype(float),
                     ok=~np.isin(pt, [PID[n] for n in ("hidden", "hood_underside", "mirror", "door_handle")]))
    return _MESH


def crease_section(y0, sg, zlo=0.70, zhi=0.98, dz=0.0005):
    """Vertical section of the outer skin at y0 (sg +1 left / -1 right): outermost |x| per z, whether there is surface,
    and the convex curvature d(elevation)/d(arc) in deg per mm (elevation = angle of the normal above horizontal,
    smoothed over ~1.5 mm of profile)."""
    m = _mesh()
    P, Nt = m["P"], m["N"]
    v = P[:, :, 1] - y0
    sv = np.sign(v)
    cross = (sv.min(1) < 0) & (sv.max(1) > 0) & m["ok"] & (P[:, :, 0].mean(1) * sg > 0.45) & (Nt[:, 0] * sg > 0.02)
    zs = np.arange(zlo, zhi, dz)
    X = np.full(len(zs), -9.0)
    T = np.full(len(zs), -1)
    for t in np.flatnonzero(cross):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if v[t, a] * v[t, b] < 0:
                u = v[t, a] / (v[t, a] - v[t, b])
                pts.append(P[t, a] + (P[t, b] - P[t, a]) * u)
        if len(pts) != 2:
            continue
        a, b = pts
        if a[2] > b[2]:
            a, b = b, a
        if b[2] - a[2] < 1e-6:
            continue
        k = np.flatnonzero((zs >= a[2]) & (zs <= b[2]))
        x = (a[0] + (b[0] - a[0]) * (zs[k] - a[2]) / (b[2] - a[2])) * sg
        w = x > X[k]
        X[k[w]] = x[w]
        T[k[w]] = t
    ok = T >= 0
    n = Nt[np.maximum(T, 0)]
    el = np.degrees(np.arctan2(n[:, 2], np.abs(n[:, 0])))
    arc = np.concatenate([[0], np.cumsum(np.hypot(dz, np.where(ok[1:] & ok[:-1], np.diff(X), 0.0)))])
    w = ndimage.gaussian_filter1d(ok.astype(float), 3)
    el_s = ndimage.gaussian_filter1d(np.where(ok, el, 0.0), 3) / np.maximum(w, 1e-6)
    return zs, X, ok, np.gradient(el_s, arc) / 1000.0


def crease_peaks(y0, sg):
    """(z, |x|, deg/mm) of every convex ridge of the section (local maxima of the curvature > 1 deg/mm)"""
    zs, X, ok, de = crease_section(y0, sg)
    return [(float(zs[k]), float(X[k]), float(de[k])) for k in range(4, len(zs) - 4)
            if ok[k - 4:k + 5].all() and de[k] > 1.0 and de[k] == de[k - 4:k + 5].max()]


def _crease_track(ys, peaks, start_y, zwin, xwin, min_k, step_tol=0.008, max_gap=6):
    """Follow one ridge: the sharpest peak inside the window at start_y, then station by station forward and backward
    the peak nearest the previous height (< step_tol); up to max_gap stations without one are bridged linearly."""
    i0 = int(np.argmin(np.abs(ys - start_y)))
    good = lambda p: zwin[0] < p[0] < zwin[1] and xwin[0] < p[1] < xwin[1] and p[2] > min_k
    z = np.full(len(ys), np.nan)
    z[i0] = max([p for p in peaks[i0] if good(p)], key=lambda p: p[2])[0]
    for rng in (range(i0 + 1, len(ys)), range(i0 - 1, -1, -1)):
        prev, miss = z[i0], 0
        for i in rng:
            c = [p for p in peaks[i] if good(p) and abs(p[0] - prev) < step_tol]
            if not c:
                miss += 1
                if miss > max_gap:
                    break
                continue
            miss = 0
            z[i] = prev = min(c, key=lambda p: abs(p[0] - prev))[0]
    f = np.isfinite(z)
    i = np.arange(len(z))
    inn = ~f & (i > i[f].min()) & (i < i[f].max())
    z[inn] = np.interp(i[inn], i[f], z[f])
    return z


def crease_path(side, sigma=0.04):
    """Smoothed crease height z(y) on one side (1 cm stations, CREASE_Y), see above. Returns (ys, z) and keeps the raw
    ridge heights in CREASE[side]."""
    if side in CREASE:
        return CREASE[side]["y"], CREASE[side]["z"]
    sg = 1 if side == "L" else -1
    ys = np.round(np.arange(CREASE_Y[0], CREASE_Y[1] + 1e-9, CREASE_STEP), 3)
    peaks = [crease_peaks(y, sg) for y in ys]
    fz = _crease_track(ys, peaks, -1.20, (0.70, 0.83), (0.88, 1.0), 3.0)       # front fender: flare top edge
    dz = _crease_track(ys, peaks, 0.0, (0.83, 0.93), (0.70, 0.87), 3.0)         # doors / rear quarter: tornado line
    fz[ys > CREASE_FENDER_END] = np.nan
    dz[ys < CREASE_DOOR_START] = np.nan

    def smooth(a):
        out = np.full_like(a, np.nan)
        i = np.flatnonzero(np.isfinite(a))
        out[i[0]:i[-1] + 1] = ndimage.gaussian_filter1d(a[i[0]:i[-1] + 1], sigma / CREASE_STEP, mode="nearest")
        return out
    fs, ds = smooth(fz), smooth(dz)
    # straight continuations into the blend (fender: its last 12 cm; door: the 12 cm behind its rounded leading edge)
    mf = (ys > CREASE_FENDER_END - 0.12) & (ys <= CREASE_FENDER_END) & np.isfinite(fs)
    md = (ys >= -0.78) & (ys <= -0.66) & np.isfinite(ds)
    zf = np.where(ys <= CREASE_FENDER_END, fs, np.polyval(np.polyfit(ys[mf], fs[mf], 1), ys))
    zd = np.where(ys >= -0.78, ds, np.polyval(np.polyfit(ys[md], ds[md], 1), ys))
    t = np.clip((ys - CREASE_BLEND[0]) / (CREASE_BLEND[1] - CREASE_BLEND[0]), 0, 1)
    w = t * t * t * (t * (t * 6 - 15) + 10)
    z = (1 - w) * zf + w * zd
    ok = np.isfinite(z)
    CREASE[side] = dict(y=ys[ok], z=z[ok], raw=np.where(ys <= CREASE_FENDER_END, fz, dz)[ok])
    return CREASE[side]["y"], CREASE[side]["z"]


def crease_z(side, y):
    ys, z = crease_path(side)
    return np.interp(y, ys, z)


SH_FRONT_Y = -1.45              # first control station of the shoulder line (it is painted from the eyeliner branch on)
TL_CACHE = os.path.join(SCRATCH, "bc", "taillamp_pts_v1.npz")


def taillamp_points():
    """Points on the tail-lamp meshes (lights / lights_glass / reflector at the rear corners), posmap frame, from the
    kn5 (3 mm sampling of their triangles). Both sides. Geometry only -> cached."""
    if os.path.exists(TL_CACHE):
        return np.load(TL_CACHE)["P"]
    sys.path.insert(0, os.path.join(LIV, "tools"))
    import render_rs3
    _, _, meshes = render_rs3.load_scene(render_rs3.DEFAULT_KN5, True)
    out = []
    for m in meshes:
        if m["mat"] not in ("lights", "lights_glass", "reflector"):
            continue
        P = np.stack([m["pos"][:, 0], -m["pos"][:, 2], m["pos"][:, 1] + 0.07237756], 1)
        for t in m["idx"].reshape(-1, 3):
            a, b, c = P[t]
            if min(a[1], b[1], c[1]) < 1.6 or max(abs(a[0]), abs(b[0]), abs(c[0])) < 0.25:
                continue
            n = int(max(2, math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / 0.003)))
            u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
            k = (u + v) <= 1
            out.append(a + np.outer(u[k], b - a) + np.outer(v[k], c - a))
    P = np.concatenate(out).astype(np.float32)
    os.makedirs(os.path.dirname(TL_CACHE), exist_ok=True)
    np.savez_compressed(TL_CACHE, P=P)
    return P


def shoulder_ctrl(side):
    """Control points (y, z) of the shoulder cut line on one side: the extracted crease (crease_z) every 4 cm from
    SH_FRONT_Y back to the station where the line's centre comes SH_REAR_AIR from the tail lamp."""
    sg = 1 if side == "L" else -1
    ys = np.arange(SH_FRONT_Y, CREASE_Y[1] - 0.005, 0.005)
    z = crease_z(side, ys)
    C = snap(list(zip(ys[ys > 1.5], z[ys > 1.5])), "side", SIDE_PARTS, side)
    tl = taillamp_points()
    tl = tl[tl[:, 0] * sg > 0]
    dl = cKDTree(tl).query(C)[0]
    k = np.flatnonzero(dl < SH_REAR_AIR)
    y_end = float(C[k[0] - 1, 1]) if len(k) else float(ys[-1])
    yc = list(np.arange(SH_FRONT_Y, y_end - 0.02, 0.04)) + [y_end]
    CREASE.setdefault(side, {})["y_end"] = y_end
    return [(float(y), float(crease_z(side, y))) for y in yc]




def front_arch_y():
    """y of the middle of the front wheel opening at the height of the belly dash line (the largest gap in the body
    surface there, measured on the posmap). The belly dash line is not painted ahead of it: in front of the front
    wheel there is no dashed run on the bumper corner any more (the front of the chart is closed by the cut line
    along the headlight, see eyeliner_curve)."""
    m = COV.reshape(-1) & (PARTF >= 0) & (np.abs(POSF[:, 2] - 0.337) < 0.01) & (POSF[:, 0] > 0.5) & \
        (POSF[:, 1] > -1.9) & (POSF[:, 1] < -0.7)
    y = np.sort(POSF[m, 1])
    k = int(np.argmax(np.diff(y)))
    return float(0.5 * (y[k] + y[k + 1]))


# -------- front cut line along the headlight («eyeliner»)
# Seen from the side the shoulder line no longer drops vertically behind the headlight to the belly line: it leaves the
# fender top in one smooth arc, comes down behind the lamp following its rear edge, runs forward along the lamp's
# LOWER contour at a constant gap (like eyeliner under an eye, following the step behind the lamp's lower tooth) and
# wraps round the front corner onto the bumper face, where it runs into the corner of the grille (the snout) with
# one whole dash and the usual air - the line goes on «behind the snout» like the belly line behind the wheel arches.
# The gap is computed, not drawn by eye: the lamp outline is taken from the kn5 (lamp meshes, rasterised with the
# renderer's own cameras), the line centre is the iso-line of the on-surface distance to that outline.
EYE_PARTS = ["front_bumper", "front_bumper_corner", "front_fender", "front_fender_top"]
EYE_C = 0.050          # line centre -> lamp outline (m) under the lamp's outer part and behind it (side view air ~2-3 cm)
EYE_C_IN = 0.040       # ... under the lamp's raised inner part (forward-facing: the side view sees that gap unshortened)
EYE_RAMP = (90.0, 120.0)   # polar angle range (deg, around the lamp in front34_left) of the smooth step EYE_C -> EYE_C_IN
EYE_BRANCH_Y = -1.30   # the arc leaves the shoulder line here (shoulder's outermost point on the fender top, at its
#                        first control point behind A_FRONT); the shoulder line is not painted ahead of it any more
EYE_END_CLEAR = 0.0335  # centre-line distance of the last dash end to the nearest other part (grille): cap air ~2.3 cm
HL_CACHE = os.path.join(SCRATCH, "bc", "headlight_rims_v1.npz")
HL_BOX = (0.40, 0.42, 1.38, 1.00, 0.98, 2.18)       # close-up box around the left lamp (renderer world: x, y up, z front)
HL_VIEWS = ("side_left", "front", "front34_left")
EYE = {}               # filled by paint_body: curves, phases, checks (for the report)
LINE_CHECKS = []


def headlight_rims():
    """Outline of the left headlight as the renderer shows it, from the kn5: close-ups of the lamp in side_left, front
    and front34_left (renderer cameras and z-buffer, opaque pass, 1600x900). Lamp = the 'lights' / 'reflector' meshes
    of the front lamp plus the black / plastic housing triangles that lie entirely inside their box (+1 cm, +3.5 cm to
    the rear for the housing tip). Returns the 3D points (posmap frame) of skin pixels that touch the lamp (rim_hl) or
    any other part (rim_other: grille, intakes, trim, hood gap ...), and per view the lamp mask + camera (for the
    visible-air report). Geometry only, cached in the scratchpad. The car is symmetric: the right side is mirrored."""
    if os.path.exists(HL_CACHE):
        d = np.load(HL_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    sys.path.insert(0, os.path.join(LIV, "tools"))
    import render_rs3 as R
    sc = R.Scene(R.DEFAULT_KN5, SRC, interior=False)
    to_pm = lambda p: np.stack([p[..., 0], -p[..., 2], p[..., 1] + 0.07237756], -1)
    V = to_pm(sc.P[sc.T])
    c = V.mean(1)
    mat = np.array(sc.mat_names)[sc.TM]
    front = (c[:, 1] < -1.45) & (c[:, 1] > -2.2) & (c[:, 2] > 0.5) & (c[:, 2] < 0.9) & (np.abs(c[:, 0]) > 0.3)
    core = np.isin(mat, ["lights", "reflector"]) & front
    lo, hi = V[core].reshape(-1, 3).min(0), V[core].reshape(-1, 3).max(0)
    ax = np.abs(V[..., 0])
    xl, xh = np.abs(V[core][..., 0]).min(), np.abs(V[core][..., 0]).max()
    inb = (ax >= xl - 0.01) & (ax <= xh + 0.01) & (V[..., 1] >= lo[1] - 0.01) & (V[..., 1] <= hi[1] + 0.035) & \
          (V[..., 2] >= lo[2] - 0.008) & (V[..., 2] <= hi[2] + 0.008)
    is_hl = core | (np.isin(mat, ["black", "ext_plastic"]) & inb.all(1))
    skin_mid = sc.mat_names.index("skin")
    W, H = 1600, 900
    out = dict(hl_tris=np.array(int(is_hl.sum())))
    rim_h, rim_o = [], []
    for v in HL_VIEWS:
        eye, f, rgt, up = R.view_camera(v)
        rel = sc.P - eye
        Dv = rel @ f
        cx_, cy_ = rel @ rgt / Dv, rel @ up / Dv
        blo, bhi = np.array(HL_BOX[:3]), np.array(HL_BOX[3:])
        corners = np.array([[(blo, bhi)[i][0], (blo, bhi)[j][1], (blo, bhi)[k][2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        rc = corners - eye
        dc = rc @ f
        ux, uy = rc @ rgt / dc, rc @ up / dc
        fpx = min(W * 0.86 / (ux.max() - ux.min()), H * 0.86 / (uy.max() - uy.min()))
        mx, my = (ux.min() + ux.max()) / 2, (uy.min() + uy.max()) / 2
        X = (cx_ - mx) * fpx + W / 2
        Y = -(cy_ - my) * fpx + H / 2
        opaque = ~sc.is_glass[sc.TM] & sc.tri_ok
        gid = np.nonzero(opaque)[0]
        zb, tid, B1, B2 = R.rasterize(X, Y, Dv, sc.T[opaque], sc.bias[sc.TM][opaque], W, H, None)
        tid = np.where(tid >= 0, gid[np.maximum(tid, 0)], -1)
        hit = tid >= 0
        lab = np.zeros(W * H, np.int8)               # 0 none, 1 skin, 2 lamp, 3 any other part
        lab[hit] = 3
        lab[hit & (sc.TM[np.maximum(tid, 0)] == skin_mid)] = 1
        lab[hit & is_hl[np.maximum(tid, 0)]] = 2
        tri = sc.T[np.maximum(tid, 0)]
        p = sc.P[tri[:, 0]] * (1 - B1 - B2)[:, None] + sc.P[tri[:, 1]] * B1[:, None] + sc.P[tri[:, 2]] * B2[:, None]
        pm = to_pm(p).reshape(H, W, 3).astype(np.float32)
        lab = lab.reshape(H, W)
        sk = lab == 1
        for k_, acc in ((2, rim_h), (3, rim_o)):
            nb = ndimage.binary_dilation(lab == k_, iterations=2) & sk
            acc.append(pm[nb])
        out["mask_" + v] = np.packbits(lab == 2)
        out["cam_" + v] = np.array([*to_pm(eye[None])[0], *to_pm(f[None])[0] - np.array([0, 0, 0.07237756]),
                                    *(to_pm(rgt[None])[0] - np.array([0, 0, 0.07237756])),
                                    *(to_pm(up[None])[0] - np.array([0, 0, 0.07237756])), mx, my, fpx, W, H])
    out["rim_hl"] = np.concatenate(rim_h)
    out["rim_other"] = np.concatenate(rim_o)
    os.makedirs(os.path.dirname(HL_CACHE), exist_ok=True)
    np.savez_compressed(HL_CACHE, **out)
    return out


class _OuterSurf:
    """Outer (renderer-visible) skin of EYE_PARTS on one side, for gluing 3D curve points onto the mesh."""
    def __init__(self, side):
        idx = cand(EYE_PARTS, side)
        self.idx = idx[OUTER[idx] & (POSF[idx, 1] < -1.25)]
        self.P = POSF[self.idx]
        self.N = NRMF[self.idx]
        self.tree = cKDTree(self.P)

    def snap(self, pts, k=6):
        out = np.atleast_2d(np.asarray(pts, float)).copy()
        for _ in range(3):
            _, j = self.tree.query(out, k=k)
            q = self.P[j].mean(1)
            n = self.N[j].mean(1)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            out = out - ((out - q) * n).sum(1, keepdims=True) * n
        return out


# -------- nose: the front cut line goes on past the headlight around the TOP of the grille (Singleframe) and joins
# the other side's line on the centreline. Its outline is taken from the kn5 like the lamp's: close-ups of the left
# half of the nose in the front and front 3/4 views (renderer cameras and z-buffer), every pixel labelled skin / lamp /
# grille (frame, honeycomb, rings: the plastic / chrome parts in front of the bumper inside |x| < 0.47) / other part.
NOSE_CACHE = os.path.join(SCRATCH, "bc", "nose_rims_v1.npz")
NOSE_BOX = (-0.10, 0.25, 1.75, 0.95, 0.95, 2.25)     # renderer world (x, y up, z front) - left half of the nose
NOSE_VIEWS = ("front", "front34_left")
NOSE_PARTS = ["hood", "nose"]
GRILLE_GAP = 0.035     # line centre -> grille outline (m): 2.4 cm of clear paint to the grille edge
GRILLE_C = (0.0, 0.50)  # (x, z) centre of the grille in the front plane (orders the grille contour by angle)


def nose_rims():
    """3D points (posmap frame) of skin pixels touching the grille (rim_gr), the lamp (rim_hl) or any other part
    (rim_ot) in close-ups of the left half of the nose, plus per view the lamp / grille masks and the camera (for the
    visible-air report). Geometry only, cached in the scratchpad; the right side is the mirror image."""
    if os.path.exists(NOSE_CACHE):
        d = np.load(NOSE_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    sys.path.insert(0, os.path.join(LIV, "tools"))
    import render_rs3 as R
    sc = R.Scene(R.DEFAULT_KN5, SRC, interior=False)
    to_pm = lambda p: np.stack([p[..., 0], -p[..., 2], p[..., 1] + 0.07237756], -1)
    V = to_pm(sc.P[sc.T])
    c = V.mean(1)
    mat = np.array(sc.mat_names)[sc.TM]
    front = (c[:, 1] < -1.45) & (c[:, 1] > -2.2) & (c[:, 2] > 0.5) & (c[:, 2] < 0.9) & (np.abs(c[:, 0]) > 0.3)
    core = np.isin(mat, ["lights", "reflector"]) & front
    lo, hi = V[core].reshape(-1, 3).min(0), V[core].reshape(-1, 3).max(0)
    ax = np.abs(V[..., 0])
    xl, xh = np.abs(V[core][..., 0]).min(), np.abs(V[core][..., 0]).max()
    inb = (ax >= xl - 0.01) & (ax <= xh + 0.01) & (V[..., 1] >= lo[1] - 0.01) & (V[..., 1] <= hi[1] + 0.035) & \
          (V[..., 2] >= lo[2] - 0.008) & (V[..., 2] <= hi[2] + 0.008)
    is_hl = core | (np.isin(mat, ["black", "ext_plastic"]) & inb.all(1))          # (as in headlight_rims)
    is_gr = np.isin(mat, ["ext_plastic", "reshotka", "chrome", "black"]) & (c[:, 1] < -2.04) & \
        (np.abs(c[:, 0]) < 0.47) & (c[:, 2] > 0.29) & (c[:, 2] < 0.74) & ~is_hl
    skin_mid = sc.mat_names.index("skin")
    W, H = 1600, 900
    out = dict(gr_tris=np.array(int(is_gr.sum())))
    acc = {"hl": [], "gr": [], "ot": []}
    for v in NOSE_VIEWS:
        eye, f, rgt, up = R.view_camera(v)
        rel = sc.P - eye
        Dv = rel @ f
        cx_, cy_ = rel @ rgt / Dv, rel @ up / Dv
        blo, bhi = np.array(NOSE_BOX[:3]), np.array(NOSE_BOX[3:])
        corners = np.array([[(blo, bhi)[i][0], (blo, bhi)[j][1], (blo, bhi)[k][2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        rc = corners - eye
        dc = rc @ f
        ux, uy = rc @ rgt / dc, rc @ up / dc
        fpx = min(W * 0.86 / (ux.max() - ux.min()), H * 0.86 / (uy.max() - uy.min()))
        mx, my = (ux.min() + ux.max()) / 2, (uy.min() + uy.max()) / 2
        X = (cx_ - mx) * fpx + W / 2
        Y = -(cy_ - my) * fpx + H / 2
        opaque = ~sc.is_glass[sc.TM] & sc.tri_ok
        gid = np.nonzero(opaque)[0]
        zb, tid, B1, B2 = R.rasterize(X, Y, Dv, sc.T[opaque], sc.bias[sc.TM][opaque], W, H, None)
        tid = np.where(tid >= 0, gid[np.maximum(tid, 0)], -1)
        hit = tid >= 0
        t_ = np.maximum(tid, 0)
        lab = np.zeros(W * H, np.int8)               # 0 none, 1 skin, 2 lamp, 3 grille, 4 any other part
        lab[hit] = 4
        lab[hit & (sc.TM[t_] == skin_mid)] = 1
        lab[hit & is_hl[t_]] = 2
        lab[hit & is_gr[t_]] = 3
        tri = sc.T[t_]
        p = sc.P[tri[:, 0]] * (1 - B1 - B2)[:, None] + sc.P[tri[:, 1]] * B1[:, None] + sc.P[tri[:, 2]] * B2[:, None]
        pm = to_pm(p).reshape(H, W, 3).astype(np.float32)
        lab = lab.reshape(H, W)
        sk = lab == 1
        for k_, nm in ((2, "hl"), (3, "gr"), (4, "ot")):
            nb = ndimage.binary_dilation(lab == k_, iterations=2) & sk
            acc[nm].append(pm[nb])
        out["lamp_" + v] = np.packbits(lab == 2)
        out["grille_" + v] = np.packbits(lab == 3)
        out["cam_" + v] = np.array([*to_pm(eye[None])[0], *to_pm(f[None])[0] - np.array([0, 0, 0.07237756]),
                                    *(to_pm(rgt[None])[0] - np.array([0, 0, 0.07237756])),
                                    *(to_pm(up[None])[0] - np.array([0, 0, 0.07237756])), mx, my, fpx, W, H])
    for k_ in acc:
        out["rim_" + k_] = np.concatenate(acc[k_])
    os.makedirs(os.path.dirname(NOSE_CACHE), exist_ok=True)
    np.savez_compressed(NOSE_CACHE, **out)
    return out


class _EyeSurf:
    """Outer (renderer-visible) skin of the front cut line's panels on one side, centreline strip included (|x| up
    to 4 mm past the centre), for gluing 3D curve points onto the mesh."""
    def __init__(self, side):
        sg = 1 if side == "L" else -1
        idx = np.flatnonzero(np.isin(PARTF, [PID[p] for p in EYE_PARTS + NOSE_PARTS]) & PAINTF)
        P = POSF[idx]
        self.idx = idx[OUTER[idx] & (P[:, 1] < -1.25) & (P[:, 0] * sg > -0.004)]
        self.P = POSF[self.idx]
        self.N = NRMF[self.idx]
        self.tree = cKDTree(self.P)

    def snap(self, pts, k=6):
        out = np.atleast_2d(np.asarray(pts, float)).copy()
        for _ in range(3):
            _, j = self.tree.query(out, k=k)
            q = self.P[j].mean(1)
            n = self.N[j].mean(1)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            out = out - ((out - q) * n).sum(1, keepdims=True) * n
        return out


class _OuterSurf:
    """Outer (renderer-visible) skin of EYE_PARTS on one side, for gluing 3D curve points onto the mesh."""
    def __init__(self, side):
        idx = cand(EYE_PARTS, side)
        self.idx = idx[OUTER[idx] & (POSF[idx, 1] < -1.25)]
        self.P = POSF[self.idx]
        self.N = NRMF[self.idx]
        self.tree = cKDTree(self.P)

    def snap(self, pts, k=6):
        out = np.atleast_2d(np.asarray(pts, float)).copy()
        for _ in range(3):
            _, j = self.tree.query(out, k=k)
            q = self.P[j].mean(1)
            n = self.N[j].mean(1)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            out = out - ((out - q) * n).sum(1, keepdims=True) * n
        return out


def _resample(C, step=0.002):
    s = arclen(C)
    t = np.arange(0, s[-1], step)
    return np.stack([np.interp(t, s, C[:, k]) for k in range(3)], 1)


def eyeliner_curve(shoulder_C):
    """Centre line of the front cut line on the LEFT side (dense, 2 mm), starting on the shoulder line at
    EYE_BRANCH_Y (tangent-continuous) and ending at the grille corner. Returns (C, S_B): S_B = shoulder arc length
    (dash coordinate) at the branch point."""
    from scipy.interpolate import splprep, splev
    rim = headlight_rims()
    surf = _OuterSurf("L")
    th, to = cKDTree(rim["rim_hl"]), cKDTree(rim["rim_other"])
    dh = th.query(surf.P)[0]
    # iso-lines of the distance to the lamp outline; their outer branch ordered by the polar angle around the lamp in
    # the front34_left view (image y down: 90 deg = straight below the lamp)
    e, f, rgt, up, mx, my, fpx, Ws, Hs = VISD["cam"]["front34_left"]

    def proj(Q):
        rel = Q - e
        dv = rel @ f
        return ((rel @ rgt) / dv - mx) * fpx + Ws / 2, -((rel @ up) / dv - my) * fpx + Hs / 2

    hx, hy = proj(rim["rim_hl"])
    cx, cy = hx.mean(), hy.mean()

    def contour(c, a0=-12.0, a1=135.0, step=1.5):
        m = np.abs(dh - c) < 0.001
        Q = surf.P[m]
        X, Y = proj(Q)
        ang = np.degrees(np.arctan2(Y - cy, X - cx))
        rad = np.hypot(X - cx, Y - cy)
        pts, angs = [], []
        for a in np.arange(a0, a1, step):
            s_ = (ang >= a) & (ang < a + step)
            if s_.sum() < 3:
                continue
            k_ = rad[s_] > rad[s_].max() - 4
            pts.append(np.median(Q[s_][k_], 0))
            angs.append(a + step / 2)
        return np.array(pts), np.array(angs)

    O1, a1 = contour(EYE_C)
    O2, a2 = contour(EYE_C_IN)
    # behind and under the lamp's outer part: EYE_C; under its raised inner part a smooth step down to EYE_C_IN
    ra = np.arange(EYE_RAMP[0], 128.0, 1.5)
    at = lambda O, a: np.stack([np.interp(ra, a, O[:, k]) for k in range(3)], 1)
    tr = np.clip((ra - EYE_RAMP[0]) / (EYE_RAMP[1] - EYE_RAMP[0]), 0, 1)[:, None]
    tr = tr * tr * (3 - 2 * tr)
    ramp = surf.snap(at(O1, a1) * (1 - tr) + at(O2, a2) * tr)
    sel = (a1 > -4) & (a1 < EYE_RAMP[0])
    body = np.vstack([O1[sel], ramp])
    inner = np.concatenate([np.zeros(sel.sum(), bool), ra > 99])
    # end: the inner run carried straight on toward the grille corner until the dash end would come closer than
    # EYE_END_CLEAR to the next part
    a_, b_ = body[-8], body[-1]
    dv = (b_ - a_) / np.linalg.norm(b_ - a_)
    E = b_
    for t in np.arange(0, 0.10, 0.0025):
        p = surf.snap(b_ + dv * t)[0]
        if to.query(p)[0] < EYE_END_CLEAR:
            break
        E = p
    # branch on the shoulder line (its own dense curve, so the dash coordinate continues exactly)
    S_sh = arclen(shoulder_C)
    ib = int(np.argmin(np.abs(shoulder_C[:, 1] - EYE_BRANCH_Y)))
    PB = shoulder_C[ib].copy()
    back = [shoulder_C[min(ib + 20, len(shoulder_C) - 1)], shoulder_C[min(ib + 10, len(shoulder_C) - 1)]]
    # two guide points on the outer surface between the branch and the top of the descent (on the chord, glued on)
    top = list(surf.snap(np.array([PB + (body[0] - PB) * t for t in (0.35, 0.70)])))
    G = np.vstack(back + [PB] + top + list(body) + [E])
    w = np.ones(len(G))
    w[:3] = 40.0
    w[3:5] = 6.0
    w[5:5 + len(body)][inner] = 4.0
    w[-1] = 40.0
    for _ in range(8):
        # smoothing B-spline through the guide points (flowing line), then glued to the surface; where it comes closer
        # to the lamp than the target - 4 mm (3 cm at the outer lower corner, which the arc rounds) the guide points
        # nearby get more weight
        tck, _u = splprep(G.T, w=w, s=0.0002 * len(G), k=3)
        C = np.array(splev(np.linspace(0, 1, 3000), tck)).T
        k0 = int(np.argmin(np.linalg.norm(C - PB, axis=1)))
        C = _resample(surf.snap(C[k0:]), 0.002)
        d_ = th.query(C)[0]
        s_ = arclen(C)
        corner = (C[:, 1] < -1.70) & (C[:, 1] > -1.86) & (C[:, 2] < 0.64)
        lim = np.where(corner, 0.040, np.where(C[:, 0] < 0.60, EYE_C_IN - 0.004, EYE_C - 0.004))
        bad = (d_ < lim) & (s_ > 0.2)
        if not bad.any():
            break
        w[np.unique(cKDTree(G).query(C[bad])[1])] *= 2.5
    C[0] = PB
    return C, float(S_sh[ib])


def paint_eyeliner(C, side, S_B, k):
    """Dashed, constant 2.2 cm width, dash coordinate S_B - k*arc (continues the shoulder line's dashes through the
    branch; k stretches the dashes by a few percent so the last one ends exactly at the end of the curve)."""
    S = arclen(C)
    idx = cand(EYE_PARTS, side)
    P = POSF[idx]
    lo, hi = C.min(0) - 0.03, C.max(0) + 0.03
    m = np.all((P >= lo) & (P <= hi), axis=1)
    idx, P = idx[m], P[m]
    d, j = cKDTree(C).query(P, k=1, distance_upper_bound=0.03)
    ok = np.isfinite(d)
    idx, d, j = idx[ok], d[ok], j[ok]
    s = S_B - k * S[j]
    dash_line(idx, s, d, 0.011, DASH[0], DASH[1], INK)
    loc = np.mod(s, DPER)
    along = np.where(loc < DASH[0], 0.0, np.minimum(loc - DASH[0], DPER - loc))
    return idx[np.sqrt(along ** 2 + d ** 2) < 0.011]


def front_line_report():
    """front cut line along the headlight: whole dashes, air to the lamp / other parts / the labels nearby"""
    for s in ("L", "R"):
        e = EYE[s]
        r = eyeliner_checks(e["C"], s, e["S_B"], e["k"], e["ink"])
        lab = {"ЛОПАТКА": f"label_lopatka_{s}", "ПЯТАЧОК": "label_pyatachok", "ШЕЙКА": "label_sheika_hood",
               "Симкарт hood tag": "simkart_hood"}
        tree = cKDTree(POSF[e["ink"]])
        r["label_air_cm"] = {k: round(float(tree.query(POSF[PAINTED[v]])[0].min()) * 100, 1)
                             for k, v in lab.items() if len(PAINTED.get(v, ())) > 0}
        if len(r["label_air_cm"]) < len(lab) or min(r["label_air_cm"].values()) < 2.0:
            r["status"] = "FAIL"
        LINE_CHECKS.append(r)
        print(f"   front cut line {s}: {r}", flush=True)


def eyeliner_checks(C, side, S_B, k, ink):
    """Whole dashes only: every dash of the line is checked in three strips across its width (centre, +-6 mm) for
    holes in the outer skin (> 1.2 cm = cut by a seam / opening) and, in each of the 9 standard views that faces it,
    for being partly hidden behind another part. Plus the air to the lamp (3D, along the surface) and to every other
    part, and the visible air to the lamp in the side, front 3/4 and front close-ups (mirrored for the right side)."""
    S = arclen(C)
    s = S_B - k * S
    loc = np.mod(s, DPER)
    body = loc < DASH[0]
    runs = np.split(np.arange(len(C)), np.flatnonzero(np.diff(body.astype(int))) + 1)
    dashes = [r for r in runs if body[r[0]]]
    surf = _OuterSurf(side)
    T = np.gradient(C, S, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Nn = surf.N[surf.tree.query(C)[1]]
    lat = np.cross(Nn, T)
    lat /= np.linalg.norm(lat, axis=1, keepdims=True)
    trunc, hid = [], []
    for i, r in enumerate(dashes):
        if i == 0 and r[0] == 0:
            continue                       # the dash through the branch: half of it is the shoulder line's
        worst = 0.0
        for off in (-0.006, 0.0, 0.006):
            q = surf.snap(C[r] + lat[r] * off)
            dd = surf.tree.query(q)[0]
            hole = dd > 0.0025
            if hole.any():
                runs_h = np.split(np.flatnonzero(hole), np.flatnonzero(np.diff(np.flatnonzero(hole)) > 1) + 1)
                worst = max(worst, max(len(h) for h in runs_h) * 0.002)
        if worst > 0.012:
            trunc.append(i)
    did = np.full(len(C), -1)
    for i, r in enumerate(dashes):
        did[r] = i
    near = np.flatnonzero(did >= 0)
    did = did[near[np.abs(near[None, :] - np.arange(len(C))[:, None]).argmin(1)]]   # caps belong to their dash
    dash_of = did[cKDTree(C).query(POSF[ink])[1]]
    for v in VIEWS9:
        e = VISD["cam"][v][0]
        dv = e[None, :] - POSF[ink]
        dv /= np.linalg.norm(dv, axis=1, keepdims=True)
        # (texels seen at more than ~66 deg incidence are left out: where the line wraps away from a camera - the
        #  part under the lamp seen from straight above - the last slivers graze the lamp's lower lip)
        fc = np.einsum("ij,ij->i", dv, NRMF[ink]) > 0.4
        if fc.sum() == 0:
            continue
        hb = hidden_by_part(POSF[ink[fc]], v)
        dk = dash_of[fc]
        for dsh in np.unique(dk):
            nh, ns = int((hb & (dk == dsh)).sum()), int((~hb & (dk == dsh)).sum())
            if nh > 3 and ns > 25:
                hid.append((int(dsh), v))
    rim = headlight_rims()
    sg = 1.0 if side == "L" else -1.0
    Pm = POSF[ink] * np.array([sg, 1, 1])                     # (right side mirrored onto the left close-ups)
    dl = cKDTree(rim["rim_hl"]).query(Pm)[0]
    do = cKDTree(rim["rim_other"]).query(Pm)[0]
    jc = cKDTree(C).query(POSF[ink])[1]
    # segments: «behind» = from the top of the arc behind the lamp down to its lower outer corner, «under» = along
    # the lower contour to the end (the curve is past the corner once it runs forward of y -1.77 below z 0.62)
    under = (C[jc, 1] < -1.77) & (C[jc, 2] < 0.62)
    behind = (S[jc] > 0.25) & ~under
    after = behind | under
    sa = S[jc]
    mins3 = []
    for b0 in np.arange(0.0, S[-1] - 0.06, 0.06):         # narrowest air on the paint per 6 cm, along the lamp
        m = (sa >= b0) & (sa < b0 + 0.06)
        if m.sum() > 20 and dl[m].min() < 0.07:
            mins3.append(float(dl[m].min()))
    res = dict(side=side, dashes=len(dashes), stretch=round(k, 4), length_cm=round(float(S[-1]) * 100, 1),
               truncated=trunc, hidden=hid, air_3d_cm=[round(min(mins3) * 100, 1), round(max(mins3) * 100, 1)],
               air_other_cm=round(float(do.min()) * 100, 1))
    vis_air = {}
    for v in HL_VIEWS:
        cam = rim["cam_" + v]
        e, f, rgt, up = cam[0:3], cam[3:6], cam[6:9], cam[9:12]
        mx, my, fpx, Wc, Hc = cam[12:17]
        Wc, Hc = int(Wc), int(Hc)
        mask = np.unpackbits(rim["mask_" + v])[:Wc * Hc].reshape(Hc, Wc).astype(bool)
        edt = ndimage.distance_transform_edt(~mask)
        stdv = {"front": "front"}.get(v, v if side == "L" else v.replace("left", "right"))
        vis = VISD["vis"][stdv][ink] & after
        if vis.sum() == 0:
            continue
        rel = Pm - e
        dv = rel @ f
        X = ((rel @ rgt) / dv - mx) * fpx + Wc / 2
        Y = -((rel @ up) / dv - my) * fpx + Hc / 2
        xi = np.clip(X.astype(int), 0, Wc - 1)
        yi = np.clip(Y.astype(int), 0, Hc - 1)
        g = edt[yi, xi] * dv / fpx
        # the air seen along the lamp: the narrowest air in every 6 cm of line, over the stretch where the line runs
        # alongside the lamp in this view (air < 6 cm), the last 6 cm (the run into the grille corner) left out
        sa = S[jc]
        mins = []
        for b0 in np.arange(0.0, S[-1] - 0.06, 0.06):
            m = vis & (sa >= b0) & (sa < b0 + 0.06)
            if m.sum() > 20 and g[m].min() < 0.06:
                mins.append(float(g[m].min()))
        if mins:
            vis_air[v] = [round(min(mins) * 100, 1), round(max(mins) * 100, 1)]
    res["visible_air_cm"] = vis_air
    res["status"] = "OK" if not trunc and not hid else "FAIL"
    return res


def dash_line(idx, s, d, half_w, on, off, col, phase=0.0):
    per = on + off
    loc = np.mod(s + phase, per)
    along = np.where(loc < on, 0.0, np.minimum(loc - on, per - loc))
    dd = np.sqrt(along ** 2 + d ** 2)
    a = smoothstep_aa(dd, half_w)
    blend(idx, col, a, obstacle=True)


# -------- 3D curves snapped to the surface (shoulder line, hood / roof cut lines)
_tree_cache = {}


def snap(points, mode, parts, side=None, exact=False):
    """points: list of 2D coords; mode 'side' -> (y,z) finds outermost x on `side`;
    'top' -> (x,y) finds highest z. Returns (n,3) world points on the surface.
    exact=True (used for painted lines): the depth is then solved on the tangent plane of the mesh triangle
    nearest to that first estimate (posmap normals are the flat triangle normals, so this lands on the mesh).
    On a sloped sheet (the up-facing front-fender top, nz ~0.95) the plain mean sat ~4 mm off the surface, so
    the band painted around the curve slid down the slope and the shoulder line read lower on the fender than
    on the door."""
    idx = cand(parts, side)
    P = POSF[idx]
    key = (mode, tuple(parts), side)
    if key not in _tree_cache:
        sel = np.arange(0, len(P), 2)
        q = P[sel][:, [1, 2]] if mode == "side" else P[sel][:, [0, 1]]
        _tree_cache[key] = (cKDTree(q), P[sel], NRMF[idx[sel]])
    tree, PS, NS = _tree_cache[key]
    ax = 0 if mode == "side" else 2                     # the depth axis that is solved for
    out = []
    for p in points:
        nb = tree.query_ball_point(p, 0.006)
        if not nb:
            _, nb = tree.query(p, k=8)
            nb = list(np.atleast_1d(nb))
        nb = np.asarray(nb)
        c = PS[nb]
        # keep the requested 2D coordinates exactly; take the depth from the outermost surface sheet
        if mode == "side":
            sx = c[:, 0] if side == "L" else -c[:, 0]
            sheet = sx > sx.max() - 0.008
            pt = np.array([np.mean(c[sheet, 0]), p[0], p[1]])
        else:
            sheet = c[:, 2] > c[:, 2].max() - 0.008
            pt = np.array([p[0], p[1], np.mean(c[sheet, 2])])
        if exact:
            cs, ns = c[sheet], NS[nb[sheet]]
            for _ in range(2):
                j = int(np.argmin(np.linalg.norm(cs - pt, axis=1)))
                q, n = cs[j], ns[j]
                if abs(n[ax]) < 0.2:
                    break
                other = [k for k in range(3) if k != ax]
                pt[ax] = q[ax] - sum((pt[k] - q[k]) * n[k] for k in other) / n[ax]
        out.append(tuple(pt))
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


def dense_curve(ctrl3d, parts, side, resnap_mode):
    """the dense centre line curve_line paints (catmull-rom through ctrl3d, re-glued to the surface)"""
    C = catmull3(ctrl3d, 60)
    if resnap_mode is not None:          # keep the dense curve glued to the surface
        if resnap_mode == "side":
            C = snap([(p[1], p[2]) for p in C], "side", parts, side, exact=True)
        else:
            C = snap([(p[0], p[1]) for p in C], "top", parts, side, exact=True)
        C[:, 0] = ndimage.uniform_filter1d(C[:, 0], 9, mode="nearest")
        C[:, 2] = ndimage.uniform_filter1d(C[:, 2], 9, mode="nearest")
    return C


def arclen(C):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])


def curve_line(ctrl3d, parts, side, half_w, col, dash=None, phase=0.0, resnap_mode=None, ctrl2d=None, scale=1.0,
               clip=None, trim=None, taper=None, ymin=None):
    """Paint a 3D curve (through surface points) as a line of constant 3D width."""
    C = dense_curve(ctrl3d, parts, side, resnap_mode)
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
        fade = 1.0
        if taper is not None:             # (start, end) taper lengths: the line narrows to a point instead of stopping
            sj = S[j]
            tf = np.clip(np.minimum(sj / max(taper[0], 1e-6), (S[-1] - sj) / max(taper[1], 1e-6)), 0, 1)
            # below ~2 texels a narrowing line stair-steps across the texel grid (a ragged white sliver on the
            # fender top): the last part keeps a 2.5 mm half width and fades out in opacity instead
            hw = np.maximum(half_w * tf, min(half_w, 0.0025))
            fade = np.clip(half_w * tf / min(half_w, 0.0025), 0, 1)
        blend(idx, col, smoothstep_aa(d, hw) * fade, obstacle=True)
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
          min_clear_cm=1.0, check=True, obstacle_check=True, kind="logo", dry=False, keepout=None, occl_views=None):
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
                      parts, side, min_clear_cm, obstacle_check, kind, keepout, occl_views)
    if check:      # (checked before this decal joins INK_LAYER, so its air to the other decals is measured)
        CHECKS.append(_check(name, art, a[..., 3], c, n, u, r, wm, hm, idx, sa, sb, dep, cosn, depth_tol,
                             check_angle, painted, parts, side, min_clear_cm, obstacle_check, kind, keepout,
                             occl_views))
    if kind == "number":       # what lies under the plate: another driver's digits are painted on the same plate later
        NUM_UNDER[name] = (sel[keep], CAN[sel[keep]].copy())
    blend(sel[keep], rgb[keep], al[keep])
    if kind in ("logo", "text", "number"):
        INK_LAYER[painted] = 1.0
    PAINTED[name] = painted
    return painted


PAINTED = {}          # decal name -> painted texels (for the clearance report of the front cut line)
NUM_UNDER = {}        # number decal name -> (texels under the plate, design colour there before the plate)
NUM_JOBS = []         # body number plates as placed: repainted with each driver's own number (paint_numbers)


def paint_numbers(num):
    """Paint race number `num` on every body number plate: same plates, same placements and frames as fitted (the
    plate's outline does not depend on the digits), only the digits change. The colour under each plate is put
    back first, so nothing of the previous digits remains."""
    global RACE_NUM
    RACE_NUM = num
    for j in NUM_JOBS:
        sel, before = NUM_UNDER[j["name"]]
        CAN[sel] = before
        decal(j["name"], j["make_art"](j["size"]), j["c"], j["n"], j["u"], check=False, **j["kw"])
        assert np.array_equal(NUM_UNDER[j["name"]][0], sel), j["name"]


def view_occlusion(painted, views):
    """{view: fraction of the decal texels that face the camera (cos > 0.3) but are NOT drawn by the renderer in that
    view}, i.e. hidden behind another part of the model (wheel, wing, stay, trim) or the body itself. 0 = fully seen."""
    out = {}
    P = POSF[painted]
    Nn = NRMF[painted]
    for v in views:
        e = VISD["cam"][v][0]
        dv = e[None, :] - P
        dv /= np.linalg.norm(dv, axis=1, keepdims=True)
        f = np.einsum("ij,ij->i", dv, Nn) > 0.3
        if f.sum() < 0.5 * len(painted):
            out[v] = None                   # (seen edge-on / from behind in this view: not a view of this decal)
            continue
        out[v] = float((~VISD["vis"][v][painted[f]]).mean())
    return out


def _check(name, art, alpha, c, n, u, r, wm, hm, idx, sa, sb, dep, cosn, depth_tol, check_angle, painted, parts,
           side, min_clear_cm, obstacle_check, kind, keepout=None, occl_views=None):
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
    # (max_tilt_deg = largest angle between the projection axis and the surface normal under the art: how far the
    #  surface turns away from the projection, i.e. a distortion measure - not a rotation. The rotation of the art
    #  in the view that faces it is level_deg.)
    res["level_deg"], res["level_rows_deg"], res["level_view"] = _level(idx, sa, sb, good, n, wm, hm)
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
        ko = (Pp[:, 1].min() >= keepout.get("y_min", -9)) and (Pp[:, 1].max() <= keepout.get("y_max", 9)) and \
            (np.abs(Pp[:, 0]).max() <= keepout.get("x_abs_max", 9))
        res["keepout_ok"] = bool(ko)
        if not ko:
            fails.append("inside a keep-out (hidden by the wing in a standard view)")
    if occl_views and len(painted) and VISD:    # 3D occlusion in the renderer's own standard views
        occ = view_occlusion(painted, occl_views)
        res["hidden_frac"] = {v: (None if f is None else round(f, 3)) for v, f in occ.items()}
        bad = [v for v, f in occ.items() if f is None or f > 0.01]
        res["occl_ok"] = not bad
        if bad:
            fails.append("not fully visible in " + ",".join(bad))
    tmax = TILT_MAX.get(kind)
    if tmax is not None and tilt > tmax:
        fails.append(f"distortion risk (tilt {tilt:.1f} > {tmax})")
    res["status"] = "OK" if not fails else "FAIL: " + "; ".join(fails)
    return res


def _level(idx, sa, sb, good, n, wm, hm):
    """Measured rotation of the art in the view that faces it (side / top / front / rear, by the dominant axis of the
    projection normal), from the position map: a straight-line fit through the surface texels of three art rows
    (at +35 %, 0 and -35 % of the half height, central 90 % of the width) gives each row's slope in that view.
    Side / front / rear views: rise over run (0 = parallel to the ground); top view: the row's fore-aft drift over
    its run across the car (0 = square across the car). Returns (largest |angle| with its sign, the three row angles,
    view), degrees."""
    ax = int(np.argmax(np.abs(n)))
    h_ax, v_ax = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[ax]
    view = {0: "side", 1: "rear" if n[1] > 0 else "front", 2: "top"}[ax]
    P = POSF[idx]
    rows = []
    for f in (0.35, 0.0, -0.35):
        k = good & (np.abs(sb - f * hm / 2) < 0.0015) & (np.abs(sa) <= 0.45 * wm)
        if k.sum() < 12 or np.ptp(P[k, h_ax]) < 0.25 * wm:
            continue
        rows.append(round(math.degrees(math.atan(np.polyfit(P[k, h_ax], P[k, v_ax], 1)[0])), 2))
    worst = max(rows, key=abs) if rows else None
    return worst, rows, view


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
          and r.get("keepout_ok", True) and r.get("occl_ok", True))
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
          f"level={rep_.get('level_deg')}({rep_.get('level_view')}) "
          f"edge={rep_['edge_clear_cm']} mask={rep_.get('mask_clear_cm')} line={rep_.get('line_clear_cm')} "
          f"decals={rep_.get('decal_clear_cm')}", flush=True)
    c, n, u = best
    decal(name, art, c, n, u, **kw2)
    if kw.get("kind") == "number":
        NUM_JOBS.append(dict(name=name, make_art=make_art, size=sizes[pick], c=c, n=n, u=u, kw=kw2))
    return sizes[pick], best


def fit_search(name, make_art, sizes, make_cands, **kw):
    """decal_fit without painting: (size, passed, (center, normal, up), art, report) for the largest passing size
    (or the smallest size's best candidate when none passes)."""
    res = {}

    def trial(i):
        if i not in res:
            art = make_art(sizes[i])
            res[i] = (art,) + decal_best(name, art, make_cands(sizes[i]), want=True, **kw)
        return res[i][1]
    lo, hi = 0, len(sizes) - 1
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
    art, ok, best, rep = res[pick]
    return sizes[pick], ok, best, art, rep


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


def paint_audi_rings():
    """The baked Audi rings under the 3D chrome badge, back on the pink rear panel: only the ring strokes (soft alpha
    RING_A from the source), in deep ink, no box around them."""
    ra = RING_A.reshape(-1)
    idx = np.flatnonzero((ra > 0.004) & PAINTF)
    blend(idx, INK_D, ra[idx])
    print(f"   Audi rings: {len(idx)} texels of ring stroke in deep ink (box painted pink)")


def paint_body():
    print("base: pearl pink, holo belly band, keylines, butcher lines ...")
    paint_pearl()
    global FRONT_ARCH_Y, OUTER
    VISD.update(view_visibility())
    OUTER = np.logical_or.reduce([VISD["vis"][v] for v in VIEWS9])
    FRONT_ARCH_Y = front_arch_y()
    print(f"   belly cut line ends in the front wheel opening (y = {FRONT_ARCH_Y:.3f})")
    paint_belly()

    # --- shoulder cut line along the Audi tornado crease (headlight -> tail light), both sides
    # (rear fender, y >= 1.06: the line rides on the side-facing band 0.855-0.880 just under the tornado crease;
    #  probed normals: below 0.855 the wide-body flare top faces UP (nz ~0.95) and made the line look thin)
    # (front fender: the line runs on the up-facing flare top, nz ~0.95, where the side view sees it thin, and meets
    #  the 50-deg door crease at the fender / door gap y ~-0.86, where it is seen full height. The rise toward the
    #  rear is done ahead of y -1.05 so the line is almost level across the gap: the thin fender part runs straight
    #  into the middle of the door part instead of climbing 2-3 px right at the gap, which read as a step.)
    A_PTS = [A_FRONT, (-1.30, 0.845), (-1.05, 0.853), (-0.90, 0.857), (-0.60, 0.862),
             (0.00, 0.866), (0.60, 0.867), (0.92, 0.867), (1.06, 0.867), (1.30, 0.868), (1.60, 0.868), (1.86, 0.867)]
    shoulder = {}
    # front cut line along the headlight (left side computed, right side = its mirror image glued onto the right skin)
    c3L = snap(A_PTS, "side", SIDE_PARTS, "L")
    eyeL, S_BL = eyeliner_curve(dense_curve(c3L, SIDE_PARTS, "L", "side"))
    for s in ("L", "R"):
        c3 = snap(A_PTS, "side", SIDE_PARTS, s)
        Csh = dense_curve(c3, SIDE_PARTS, s, "side")
        if s == "L":
            eye, S_B = eyeL, S_BL
        else:
            ib = int(np.argmin(np.abs(Csh[:, 1] - EYE_BRANCH_Y)))
            S_B = float(arclen(Csh)[ib])
            eye = _resample(_OuterSurf("R").snap(eyeL * np.array([-1.0, 1.0, 1.0])), 0.002)
            EYE["mirror_start_dev_cm"] = round(float(np.linalg.norm(eye[0] - Csh[ib])) * 100, 2)
            eye[0] = Csh[ib]
        # the shoulder line ends at the branch (its dashes ahead of it, toward A_FRONT, are not painted any more)
        C = curve_line(c3, SIDE_PARTS, s, 0.011, INK, dash=DASH, resnap_mode="side", trim=(-1.0, S_B))
        shoulder[s] = C
        belly_connector(s, A_PTS, C)
        L_eye = float(arclen(eye)[-1])
        n_eye = int(round((L_eye - S_B) / DPER))
        k_eye = (S_B + n_eye * DPER) / L_eye            # the last dash ends exactly at the end of the line
        ink = paint_eyeliner(eye, s, S_B, k_eye)
        EYE[s] = dict(C=eye, S_B=S_B, k=k_eye, n=n_eye, ink=ink)
        print(f"   front cut line {s}: leaves the shoulder line at y {eye[0, 1]:.3f} (dash coordinate {S_B:.3f}), "
              f"{L_eye * 100:.1f} cm, {n_eye} dash periods, stretch {k_eye:.4f}, ends at "
              f"({eye[-1, 0]:+.3f}, {eye[-1, 1]:.3f}, {eye[-1, 2]:.3f})", flush=True)
        # chrome keyline above the whole shoulder line (6 mm half width), white core with grey edges
        # (tapers to a point at the front, where the cut line turns down behind the headlight - the branch point,
        #  A_PTS[1] - and into the tail lamp)
        kl = snap([(y, z + 0.021) for y, z in A_PTS[1:]], "side", SIDE_PARTS, s)
        curve_line(kl, SIDE_PARTS, s, 0.0072, (176, 172, 190), resnap_mode="side", taper=(0.30, 0.12))
        curve_line(kl, SIDE_PARTS, s, 0.0045, (252, 250, 255), resnap_mode="side", taper=(0.30, 0.12))

    # --- hood / roof transverse cut lines (meet the shoulder crease on the fenders)
    TOP = ["hood", "front_fender_top", "front_fender", "roof", "roof_rail"]
    # parallel to the curved hood rear edge, 15 cm ahead of it, running on across the fender tops until it MEETS
    # the shoulder cut line (it used to stop at x = +-0.78, ~7 cm short of it, ending in a lone dot).
    # The junction is put on the middle of a shoulder dash, and the hood dashes are stretched (< 1 dash period over
    # the whole line) so the line starts and ends with a full dash whose rounded end sits on that shoulder dash:
    # an ink T-junction on both fenders, no stub.
    hx = np.array([0.0, 0.3, 0.5, 0.6, 0.7])
    he = np.array([-1.116, -1.095, -1.048, -1.002, -0.923])
    xs = np.linspace(-0.70, 0.70, 29)
    pts = [(x, np.interp(abs(x), hx, he) - 0.15) for x in xs]
    CL = shoulder["L"]
    SL = arclen(CL)
    y_aim = pts[-1][1] + (0.86 - pts[-1][0]) * 0.79      # straight on along the hood line's end slope (dy/dx 0.79)
    s_aim = float(np.interp(y_aim, CL[:, 1], SL))
    s_j = math.floor(s_aim / DPER) * DPER + DASH[0] / 2
    s_j = min((s_j - DPER, s_j, s_j + DPER), key=lambda v: abs(v - s_aim))
    xj, yj = float(np.interp(s_j, SL, CL[:, 0])), float(np.interp(s_j, SL, CL[:, 1]))
    mid = (0.5 * (0.70 + xj), 0.5 * (pts[-1][1] + yj) - 0.006)
    pts = [(-xj, yj), (-mid[0], mid[1])] + pts + [mid, (xj, yj)]
    hood = snap(pts, "top", TOP, exact=True)
    Ch = dense_curve(hood, TOP, None, "top")
    Lh = float(arclen(Ch)[-1])
    nper = round((Lh - DASH[0]) / DPER)
    curve_line(hood, TOP, None, 0.011, INK, dash=DASH, resnap_mode="top", scale=(nper * DPER + DASH[0]) / Lh)
    LINE_NOTES.append(f"hood cut line meets the shoulder line at x=+-{xj:.3f} y={yj:.3f} on the middle of a shoulder "
                      f"dash; {nper + 1} whole dashes, stretch {(nper * DPER + DASH[0]) / Lh:.4f}")
    print(f"   hood cut line meets the shoulder line at x=+-{xj:.3f} y={yj:.3f} (z {Ch[-1, 2]:.3f} vs shoulder "
          f"{float(np.interp(s_j, SL, CL[:, 2])):.3f}); {nper + 1} dashes, stretch {(nper * DPER + DASH[0]) / Lh:.4f}; "
          f"R shoulder there x={float(np.interp(s_j, arclen(shoulder['R']), shoulder['R'][:, 0])):.3f}")
    # roof cut lines: dashed burgundy only - no white keyline beside them on the roof (client)
    for yl in (-0.24, 1.10):
        rl = snap([(x, yl) for x in np.linspace(-0.60, 0.60, 17)], "top", ["roof", "roof_rail"])
        curve_line(rl, ["roof", "roof_rail"], None, 0.011, INK, dash=(0.075, 0.045), resnap_mode="top")


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


def level_frame(c, parts, side, w, h):
    """Projection frame for a side plate of w x h m centred at c, computed from the position map.
    The art's horizontal axis must follow the car (world +-y) with no z component in the side views, so art up =
    world +z: then r = up x n is horizontal whatever the normal, and every art row is level. The normal is fitted to
    the panel under the footprint: in plan it is square to the panel (the panel's run x(y) is fitted - the door is
    ~1 deg off the car axis), and its elevation is the middle of the elevations of the surface normals under the
    footprint, which halves the largest angle between the projection and the surface (least distortion: the door rolls
    in ~18 deg toward the shoulder crease under the plate's top edge and is ~-3 deg at its bottom).
    Returns (normal, up)."""
    sg = 1 if side == "L" else -1
    idx = cand(parts, side)
    P, Nn = POSF[idx], NRMF[idx]
    rel = P - np.asarray(c, float)
    k = (np.abs(rel[:, 1]) <= w / 2) & (np.abs(rel[:, 2]) <= h / 2) & (np.abs(rel[:, 0]) < 0.09)
    A = np.stack([P[k, 1], P[k, 2], P[k, 2] ** 2, np.ones(int(k.sum()))], 1)
    a = float(np.linalg.lstsq(A, P[k, 0], rcond=None)[0][0])           # dx/dy of the panel under the plate
    t = np.array((a, 1.0, 0.0)) / math.hypot(a, 1.0)
    nh = np.array((sg, 0.0, 0.0)) - t * (sg * t[0])
    nh /= np.linalg.norm(nh)
    el = np.arctan2(Nn[k, 2], Nn[k] @ nh)
    th = 0.5 * (float(el.min()) + float(el.max()))
    return tuple(nh * math.cos(th) + np.array((0.0, 0.0, math.sin(th)))), (0.0, 0.0, 1.0)


def level_side_cands(ys, zs, parts, s, w, h):
    """Side-plate candidates (centres on the panel) with the computed level frame of each (see level_frame)."""
    out = []
    for y in ys:
        for z in zs:
            p = surf_point("side", y, z, parts, s)
            c = (p[0], y, z)
            n, u = level_frame(c, parts, s, w, h)
            out.append((c, n, u))
    return out


def asset_raf(h):
    """РАФ emblem (raf_black.png: the full mono emblem - laurel wings around the «раф» monogram), deep ink."""
    return asset_mono("raf_black.png", int(h), INK_D)


_RAF_CIRCLE = []


def raf_circle():
    """Smallest circle around the РАФ emblem (raf_black.png, convex hull of the mark): (diameter / emblem height,
    centre offset dx, dy from the bbox centre / emblem height). The wing tips reach past the inscribed circle, so the
    emblem needs a circle of ~1.05 x its height."""
    if not _RAF_CIRCLE:
        m = np.asarray(_L(os.path.join(BX, "raf_black.png"))) > 127
        ys, xs = np.nonzero(m)
        hp = np.stack([xs, ys], 1).astype(float)
        hp = hp[ConvexHull(hp).vertices]
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        best, step = (np.sqrt(((hp - (cx, cy)) ** 2).sum(1)).max(), cx, cy), 64.0
        while step >= 0.5:                     # coarse-to-fine search of the minimax centre
            r0, bx, by = best
            for dx in (-step, 0, step):
                for dy in (-step, 0, step):
                    r = np.sqrt(((hp - (bx + dx, by + dy)) ** 2).sum(1)).max()
                    if r < best[0] - 1e-9:
                        best = (r, bx + dx, by + dy)
            if best[0] >= r0 - 1e-9:
                step /= 2
        r, bx, by = best
        hgt = ys.max() - ys.min() + 1
        _RAF_CIRCLE.extend([2 * r / hgt, (bx - cx) / hgt, (by - cy) / hgt])
    return tuple(_RAF_CIRCLE)


def raf_badge(D):
    """РАФ for the sill row: a round badge of diameter D (px) = the row height, so its diameter matches the pills and
    the DriveOil plate of its neighbours. Deep-ink disc with the pink hairline of the BR ENGINEERING pill (same
    treatment), the full emblem in white inside it, scaled uniformly (aspect kept) so that its smallest enclosing
    circle sits concentric with the disc, 4 % of D clear of the hairline."""
    S = 4
    lw = max(2, D // 26)                       # (pill(): hairline = height // 26)
    k, ox, oy = raf_circle()
    eh = int((D - 2 * lw - 2 * 0.04 * D) / k)
    em = asset_mono("raf_black.png", eh * S, WHITE)
    Ds = D * S
    disc = Image.new("RGBA", (Ds, Ds), (0, 0, 0, 0))
    d = ImageDraw.Draw(disc)
    d.ellipse([0, 0, Ds - 1, Ds - 1], fill=(255, 214, 232, 255))
    d.ellipse([lw * S, lw * S, Ds - 1 - lw * S, Ds - 1 - lw * S], fill=INK_D + (255,))
    disc.alpha_composite(em, (int(round(Ds / 2 - em.width / 2 - ox * em.height)),
                              int(round(Ds / 2 - em.height / 2 - oy * em.height))))
    return disc.resize((D, D), Image.LANCZOS)


def raf_pill(D):
    """РАФ for the sill row on the BR ENGINEERING treatment: deep-ink pill of height D (px) with the pink hairline,
    the emblem in white in its left end (scaled uniformly so its smallest enclosing circle sits concentric with the
    rounded end, 4 % of D clear of the hairline - the raf_badge disc), followed by «РАФ» in white Unbounded Black at
    the cap height of the ENGINEERING letters (70 % of the BR logo's height D - 1 cm, measured on
    br_engineering_black.png), centred on the pill. On its own the round badge read as a dot next to the 15-43 cm
    plates; with the word it has the same letter height as its neighbour."""
    S = 4
    lw = max(2, D // 26)                       # (pill(): hairline = height // 26)
    k, ox, oy = raf_circle()
    eh = int((D - 2 * lw - 2 * 0.04 * D) / k)
    em = asset_mono("raf_black.png", eh * S, WHITE)
    cap = 0.70 * (D - px(0.010))
    f = F_NUM(int(cap * S * 1.4))
    bb_p = f.getbbox("Р")
    kc = cap * S / (bb_p[3] - bb_p[1])         # uniform scale: Р (cap height) -> cap
    word = text_mask("РАФ", f, track=0.06)
    word = word.resize((max(1, round(word.width * kc)), max(1, round(word.height * kc))), Image.LANCZOS)
    wb = word.getbbox()
    word = word.crop(wb)
    bb_w = f.getbbox("РАФ")
    top_cap = (bb_p[1] - bb_w[1]) * kc         # Ф reaches above / below the caps: centre the cap band, not the ink
    Ds, gap, padr = D * S, int(0.22 * D * S), int(px(0.012) * S)
    W = Ds + gap + word.width + padr
    out = Image.new("RGBA", (W, Ds), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    d.rounded_rectangle([0, 0, W - 1, Ds - 1], radius=Ds // 2, fill=(255, 214, 232, 255))
    d.rounded_rectangle([lw * S, lw * S, W - 1 - lw * S, Ds - 1 - lw * S], radius=Ds // 2 - lw * S, fill=INK_D + (255,))
    out.alpha_composite(em, (int(round(Ds / 2 - em.width / 2 - ox * em.height)),
                             int(round(Ds / 2 - em.height / 2 - oy * em.height))))
    out.alpha_composite(solid(word, WHITE), (Ds + gap, int(round(Ds / 2 - cap * S / 2 - top_cap))))
    return out.resize((round(W / S), D), Image.LANCZOS)


PEARL_PAD = (250, 244, 250)


def sill_art(kind, H):
    """Sill-row logo on its backer, scaled uniformly to the common outer height H (m)."""
    if kind == "raf":                # deep-ink pill: emblem + «РАФ» at the BR ENGINEERING cap height (raf_pill)
        a = raf_pill(px(H))
    elif kind == "breng":            # white on a deep-ink pill (unchanged treatment)
        a = pill(asset_mono("br_engineering_black.png", px(H - 0.010), WHITE), px(0.012), px(0.005), INK_D)
    elif kind == "driveoil":         # the brand's own orange plate (as behind the rear wheel)
        a = asset_driveoil(px(H))
    elif kind == "karting64":        # kart artwork only, pearl pad (unchanged treatment)
        a = pill(asset_karting64(px(H - 0.012)), px(0.010), px(0.006), PEARL_PAD, line=INK_D)
    else:
        raise ValueError(kind)
    return fit_h(a, px(H))


SILL_ORDER = ["raf", "breng", "driveoil", "karting64"]    # client order, front -> rear
SILL_ARCH_AIR = 0.02                                  # extra air (m) to the first / last passing point at the arches
SILL_LOG = []


SILL_H = (0.048, 0.046, 0.044, 0.042, 0.040)          # common outer height candidates (m), largest first
SILL_ZC = (0.230, 0.228, 0.226, 0.224, 0.222)         # common centre-line candidates (world z)
_SILL_PLAN = {}


def sill_plan():
    """One sill row layout for BOTH sides (mirror-identical), front -> rear РАФ · BR ENGINEERING · DriveOil ·
    KARTING64 (client order). All logos have the same outer (backer / badge) height and one common centre line z_c
    (= one baseline, equal visual height: the round РАФ badge's diameter = the pill / plate height).
    The row spans the whole flat sill: the usable ends are found first with the smallest height - the first / last
    placements, scanned 1 cm at a time from the arches, that pass every gate on both sides (flat within 12 deg, >= 1 cm
    to the sill edges, fully visible in the side and both 3/4 views: the front tyre hides the sill ahead of y ~-0.75 in
    the front 3/4 view), plus SILL_ARCH_AIR - and the first logo starts there, the last one ends there, with equal gaps
    between them. Then the largest common height that passes at those ends wins; among the centre lines the one that
    keeps the most air for the tightest logo (the sill face is lower at the front: top edge z ~0.255 vs ~0.275 at the
    rear)."""
    if _SILL_PLAN:
        return _SILL_PLAN
    sl = ["sill"]
    ctx = {}
    for s in ("L", "R"):
        sd = "left" if s == "L" else "right"
        ctx[s] = dict(sg=1 if s == "L" else -1, Ps=POSF[cand(sl, s)], Ns=NRMF[cand(sl, s)],
                      kwc=dict(ppm=PPM, side=s, parts=sl, min_clear_cm=1.0,
                               occl_views=[f"side_{sd}", f"front34_{sd}", f"rear34_{sd}"]))

    def flattest_normal(s, y, zc, w, h):
        """projection normal with the smallest worst-case angle to the sill normals under the footprint (the face
        leans ~15-25 deg at the front and turns square at the rear). Any normal keeps the baseline level: the art's
        horizontal axis is up x normal with up = world z, i.e. always horizontal."""
        c_ = ctx[s]
        m = (np.abs(c_["Ps"][:, 1] - y) < w / 2) & (np.abs(c_["Ps"][:, 2] - zc) < h / 2)
        nn = c_["Ns"][m][:: max(1, int(m.sum()) // 4000)]
        ty, tz = np.meshgrid(np.arange(-0.30, 0.305, 0.02), np.arange(-0.10, 0.605, 0.02))
        cn = np.stack([np.full(ty.size, float(c_["sg"])), ty.ravel(), tz.ravel()], 1)
        cn /= np.linalg.norm(cn, axis=1, keepdims=True)
        worst = np.arccos(np.clip(nn @ cn.T, -1, 1)).max(0)
        return tuple(cn[int(np.argmin(worst))])

    def trial(s, name, art, y, zc):
        w, h = art.width / PPM, art.height / PPM
        p = surf_point("side", y, zc, sl, s)
        c = (p[0], y, zc)
        cands_ = [(c, nv, (0, 0, 1)) for nv in (flattest_normal(s, y, zc, w, h), (ctx[s]["sg"], 0.0, 0.0))]
        return decal_best(name, art, cands_, want=True, tilt_gate=12.0, line_min=1.0, air=1.0, deco_min=2.0,
                          **ctx[s]["kwc"])

    def both(name, art, y, zc):
        return {s: trial(s, f"{name}_{s}", art, y, zc) for s in ("L", "R")}

    first, last = SILL_ORDER[0], SILL_ORDER[-1]
    ends = {}
    for zc in SILL_ZC:              # usable ends: smallest height, both sides
        a0, a1 = sill_art(first, SILL_H[-1]), sill_art(last, SILL_H[-1])
        w0, w1 = a0.width / PPM, a1.width / PPM
        yf = next((y for y in np.arange(-0.86, -0.50, 0.01)
                   if all(r[0] for r in both("sill_probe", a0, y + w0 / 2, zc).values())), None)
        yr = next((y for y in np.arange(1.02, 0.70, -0.01)
                   if all(r[0] for r in both("sill_probe", a1, y - w1 / 2, zc).values())), None)
        if yf is not None and yr is not None:
            ends[zc] = (float(yf) + SILL_ARCH_AIR, float(yr) - SILL_ARCH_AIR)
    print("   sill usable ends (front edge of the first logo / rear edge of the last) per centre line: "
          + ", ".join(f"z {zc:.3f}: {a:+.3f} .. {b:+.3f}" for zc, (a, b) in ends.items()), flush=True)
    for H in SILL_H:
        best = None
        for zc, (yf, yr) in ends.items():
            kinds = list(SILL_ORDER)
            arts = [sill_art(k, H) for k in kinds]
            W = np.array([a.width / PPM for a in arts])
            gap = (yr - yf - W.sum()) / (len(arts) - 1)
            ys = yf + np.concatenate([[0], np.cumsum(W[:-1] + gap)]) + W / 2
            res = []
            for k, a, y in zip(kinds, arts, ys):
                r = both(f"sill_{k}", a, y, zc)
                res.append(r)
                if not all(v[0] for v in r.values()):
                    break
            if len(res) < len(kinds) or not all(v[0] for r in res for v in r.values()):
                continue
            score = min(min(v[2]["edge_clear_cm"], v[2]["mask_clear_cm"]) for r in res for v in r.values())
            if best is None or score > best["score"] + 0.05:
                best = dict(score=score, H=H, zc=zc, kinds=kinds, arts=arts, W=W, ys=ys, gap=gap, yf=yf, yr=yr,
                            res=res)
        if best is not None:
            break
    if best is None:
        raise RuntimeError("sill row: no layout passes on both sides")
    _SILL_PLAN.update(best)
    b = best
    print(f"   sill row: height {b['H'] * 100:.1f} cm, centre z {b['zc']:.3f}, y {b['yf']:+.3f} .. {b['yr']:+.3f}; "
          f"{len(b['kinds'])} logos, equal gaps {b['gap'] * 100:.1f} cm: "
          + ", ".join(f"{k} {w * 100:.1f} cm @ y {y:+.3f}" for k, w, y in zip(b["kinds"], b["W"], b["ys"])),
          flush=True)
    for s in ("L", "R"):
        SILL_LOG.append(dict(side=s, height_cm=round(b["H"] * 100, 1), z_centre=b["zc"], y_front=round(b["yf"], 3),
                             y_rear=round(b["yr"], 3), gap_cm=round(float(b["gap"]) * 100, 1),
                             logos=[dict(kind=k, y=round(float(y), 3), width_cm=round(float(w) * 100, 1))
                                    for k, y, w in zip(b["kinds"], b["ys"], b["W"])]))
    return _SILL_PLAN


def sill_row(s):
    """Paint the sill row (layout from sill_plan, identical on both sides)."""
    b = sill_plan()
    sl = ["sill"]
    sd = "left" if s == "L" else "right"
    for k, a, r in zip(b["kinds"], b["arts"], b["res"]):
        c, n, u = r[s][1]
        decal(f"sill_{k}_{s}", a, c, n, u, ppm=PPM, side=s, parts=sl, min_clear_cm=1.0,
              occl_views=[f"side_{sd}", f"front34_{sd}", f"rear34_{sd}"])


def side_decals(s):
    sg = 1 if s == "L" else -1

    def sc(ys, zs, parts, w):
        return side_cands(ys, zs, parts, s, w)

    fd, rd, rf, fl, sl = ["front_door"], ["rear_door"], ["rear_fender"], ["front_door_low"], ["sill"]
    kw = dict(ppm=PPM, side=s)
    # race number plate – front door; >= 6 cm clear pink to the shoulder line above and the belly line below.
    # Level with the ground: projected with the frame computed from the position map (level_frame: art up = world
    # +z, horizontal axis along the car with no z component); its measured rotation in the side view is level_deg.
    # Size stays the approved 0.42 m plate (the shoulder and belly lines leave exactly room for it with 6 cm air).
    _, nb = decal_fit(f"number_door_{s}", lambda w: number_plate(px(w), px(w * 0.674)), [0.42],
                      lambda w: level_side_cands((-0.52, -0.54, -0.50), (0.645, 0.635, 0.655), fd, s, w, w * 0.674),
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
    # sill: one evenly spread row of partner logos (see sill_row)
    sill_row(s)
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
              ppm=PPM, parts=hood, kind="text", tilt_gate=10.0, line_min=2.0)      # >= 2 cm air to the hood cut line
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
    trunk_tail()


def trunk_tail():
    # Trunk lid: the pig's curly tail + «ХВОСТИК», reads from behind. Only the rear strip of the lid is clear of the
    # rear wing in every view: the wing blade hides the lid ahead of y ~2.05 in both rear 3/4 views, and its two stays
    # (x = +-0.20) cross the lid from above / behind (measured with a z-buffer of the model in the 9 standard views).
    # The art therefore sits between the stays (|x| <= 0.17) on the flat rear strip of the lid: from y 2.060 (the
    # top view loses the lid ahead of y ~2.045 under the wing-mount cross bar: >= 1.5 cm air) to y 2.166 (the lid
    # turns down into the lip at y ~2.178: >= 1.2 cm air) - keep-out gate - and it is also checked against the
    # renderer's own visibility in rear, both rear 3/4 and top views. Largest label cap that passes wins (tail
    # stacked above the word, see tail_with_label). (The «арка» that sat here repeated the «арка» on the wing, in the
    # same visual plane.)
    tr = ["trunk_lid"]
    decal_fit("tail_trunk", tail_with_label, [round(0.044 - 0.001 * i, 3) for i in range(17)],
              lambda h: top_cands((0.0,), (2.113, 2.111, 2.115, 2.109, 2.117, 2.107, 2.119), tr, (0, -1, 0), r=0.05),
              ppm=PPM, parts=tr, depth_tol=0.06, check_angle=40, air=1.0, kind="text", tilt_gate=10.0,
              keepout=dict(y_min=2.060, y_max=2.166, x_abs_max=0.170),
              occl_views=["rear", "rear34_left", "rear34_right", "top"])


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
    # «simkarting.ru ◆ karting64.ru»: both addresses side by side in one line (same face, size and ink). One line is
    # kept when it fits flat (<= 12 deg), clear of the dashed cut line, at >= 4.0 cm type; otherwise the two
    # addresses are stacked in two lines (same size). Must be fully visible in rear and both rear 3/4 views
    # (renderer z-buffer: tow strap, trim and wing).
    rkw = dict(ppm=PPM, parts=rb, depth_tol=0.10, kind="text", occl_views=["rear", "rear34_left", "rear34_right"])
    rgate = dict(tilt_gate=12.0, line_min=1.5, deco_min=3.0)
    one = fit_search("url_rear", lambda h: url_pair(px(h), INK), [0.052, 0.050, 0.048, 0.046, 0.044, 0.042, 0.040],
                     rbc, **rgate, **rkw)
    if one[1]:
        pick = one
        URL_NOTES.append(f"rear bumper: one line, {one[0] * 100:.1f} cm type")
    else:
        pick = fit_search("url_rear", lambda h: url_pair(px(h), INK, two_lines=True),
                          [0.034, 0.032, 0.030, 0.028, 0.026], rbc, **rgate, **rkw)
        URL_NOTES.append(f"rear bumper: two lines, {pick[0] * 100:.1f} cm type (one line did not fit)")
    size, ok, (c, n, u), art, rep = pick
    print(f"   url_rear: {'one line' if pick is one else 'two lines'} {size} m -> {'OK' if ok else 'NO FIT'} "
          f"tilt={rep['max_tilt_deg']} edge={rep['edge_clear_cm']} line={rep.get('line_clear_cm')} "
          f"hidden={rep.get('hidden_frac')}", flush=True)
    decal("url_rear", art, c, n, u, **rkw)
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


SIDE_WIN = {"L": ([115, 272, 513, 496], ("side_left", "front34_left", "rear34_left")),
            "R": ([114, 518, 512, 741], ("side_right", "front34_right", "rear34_right"))}
GVIS_CACHE = os.path.join(SCRATCH, "bc", "glass_side_visible_v1.npz")


def glass_side_visible():
    """Visible glass of the two rear side windows (glass_sticker.dds px, 1024): the window zone is made opaque and the
    model is rasterised in the side and both 3/4 views of that side with the renderer's own cameras (opaque pass,
    ss=2); a texel counts as visible glass only when it is the front-most surface in all three views (the black frame,
    B-pillar trim and C-pillar corner are separate parts / outside the glass mesh). Geometry only -> cached."""
    if os.path.exists(GVIS_CACHE):
        d = np.load(GVIS_CACHE)
        return {k: d[k] for k in ("L", "R")}
    sys.path.insert(0, os.path.join(LIV, "tools"))
    import render_rs3 as R
    sc = R.Scene(R.DEFAULT_KN5, SRC, interior=False)
    gm = sc.mat_names.index("glass_sticker")
    tex = sc.tex[gm].copy()
    for box, _ in SIDE_WIN.values():
        x0, y0, x1, y1 = box
        tex[y0 - 30:y1 + 30, x0 - 30:x1 + 30, 3] = 255
    sc.tex[gm] = tex
    tm = sc.TM

    def at_filter(tt, b1, b2):
        keep = np.ones(len(tt), bool)
        mt = tm[tt]
        ii = np.nonzero(sc.is_at[mt])[0]
        if len(ii):
            tri = sc.T[tt[ii]]
            b0 = 1 - b1[ii] - b2[ii]
            uv = sc.UV[tri[:, 0]] * b0[:, None] + sc.UV[tri[:, 1]] * b1[ii, None] + sc.UV[tri[:, 2]] * b2[ii, None]
            for mid in np.unique(mt[ii]):
                jj = mt[ii] == mid
                t = sc.tex[mid]
                if t is None or t.shape[2] < 4:
                    continue
                keep[ii[jj]] = R.sample(t, uv[jj, 0], uv[jj, 1], bilinear=False)[:, 3] >= 128
        return keep
    out = {}
    for nm, (box, views) in SIDE_WIN.items():
        vis = np.ones((1024, 1024), bool)
        for name in views:
            eye, f, rgt, up = R.view_camera(name)
            rel = sc.P - eye
            Dv = rel @ f
            cx_, cy_ = rel @ rgt / Dv, rel @ up / Dv
            used = np.unique(sc.T.ravel())
            ux, uy = cx_[used], cy_[used]
            W, H, ss, margin = 1600, 900, 2, 0.07
            Ws, Hs = W * ss, H * ss
            fpx = min(Ws * (1 - 2 * margin) / (ux.max() - ux.min()), Hs * (1 - 2 * margin) / (uy.max() - uy.min()))
            mx_, my_ = (ux.min() + ux.max()) / 2, (uy.min() + uy.max()) / 2
            X = (cx_ - mx_) * fpx + Ws / 2
            Y = -(cy_ - my_) * fpx + Hs / 2
            opaque = ~sc.is_glass[tm] & sc.tri_ok
            gid = np.nonzero(opaque)[0]
            _, tid, B1, B2 = R.rasterize(X, Y, Dv, sc.T[opaque], sc.bias[tm][opaque], Ws, Hs,
                                         lambda tt, b1, b2: at_filter(gid[tt], b1, b2))
            tid = np.where(tid >= 0, gid[np.maximum(tid, 0)], -1)
            hit = tid >= 0
            isg = np.zeros_like(hit)
            isg[hit] = tm[tid[hit]] == gm
            tt, b1, b2 = tid[isg], B1[isg], B2[isg]
            tri = sc.T[tt]
            uv = (sc.UV[tri[:, 0]] * (1 - b1 - b2)[:, None] + sc.UV[tri[:, 1]] * b1[:, None]
                  + sc.UV[tri[:, 2]] * b2[:, None])
            tx = np.clip(((uv[:, 0] % 1) * 1024).astype(int), 0, 1023)
            ty = np.clip(((uv[:, 1] % 1) * 1024).astype(int), 0, 1023)
            m = np.zeros((1024, 1024), bool)
            m[ty, tx] = True
            vis &= ndimage.binary_closing(m, iterations=3)       # close the pixel-sampling holes only
        x0, y0, x1, y1 = box
        sub = np.zeros_like(vis)
        sub[y0 - 25:y1 + 25, x0 - 25:x1 + 25] = vis[y0 - 25:y1 + 25, x0 - 25:x1 + 25]
        sub = ndimage.binary_opening(sub, iterations=1)
        lab, _ = ndimage.label(sub)
        out[nm] = lab == lab[(y0 + y1) // 2, (x0 + x1) // 2]      # the window itself (not the rear-window banner)
    os.makedirs(os.path.dirname(GVIS_CACHE), exist_ok=True)
    np.savez_compressed(GVIS_CACHE, **out)
    return out


def rrect_outline(w, h, r, rot):
    """Outline points of a w x h rounded rectangle (corner radius r) centred on 0, turned like PIL rotate(rot)."""
    pts = []
    for t in np.linspace(0, 1, 400):
        pts += [(-w / 2 + r + t * (w - 2 * r), -h / 2), (-w / 2 + r + t * (w - 2 * r), h / 2)]
    for t in np.linspace(0, 1, 80):
        pts += [(-w / 2, -h / 2 + r + t * (h - 2 * r)), (w / 2, -h / 2 + r + t * (h - 2 * r))]
    for ccx, ccy, a0 in ((w / 2 - r, -h / 2 + r, -90), (w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                         (-w / 2 + r, -h / 2 + r, 180)):
        for a in np.radians(np.linspace(a0, a0 + 90, 40)):
            pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    P = np.array(pts)
    A = math.radians(rot)       # PIL rotate(+a) turns the art counter-clockwise on screen (image y down)
    return np.stack([P[:, 0] * math.cos(A) + P[:, 1] * math.sin(A), -P[:, 0] * math.sin(A) + P[:, 1] * math.cos(A)], 1)


def centre_in_glass(vis, box, w, h, r, rot):
    """Centre (px) of a w x h rounded plate that maximises its smallest clearance to the edge of the visible glass
    (equal air to the frame on the tightest sides). Returns (cx, cy, clearance px)."""
    edt = ndimage.distance_transform_edt(vis)
    P = rrect_outline(w, h, r, rot)
    best = (-1.0, 0.0, 0.0)
    for cy in np.arange(box[1] + h / 2, box[3] - h / 2 + 0.5, 1.0):
        for cx in np.arange(box[0] + w / 2, box[2] - w / 2 + 0.5, 1.0):
            q = P + (cx, cy)
            c = edt[np.clip(np.round(q[:, 1]).astype(int), 0, 1023), np.clip(np.round(q[:, 0]).astype(int), 0, 1023)].min()
            if c > best[0] + 1e-9:
                best = (float(c), float(cx), float(cy))
    return best[1], best[2], best[0]


EP_CACHE = os.path.join(SCRATCH, "bc", "endplate_faces_v1.npz")
EP_MARGIN = 10.0        # px of glass_sticker (570 px/m on the wing): ~1.8 cm clear around the KARTING64 pad
_EP_POS = {}


def endplate_faces():
    """Outer faces of the two rear-wing endplates in glass_sticker.dds px (1024), from the kn5: the sticker
    triangles of the endplate mesh (Plane.033, wing sticker material) rasterised at 4x and reduced to the texels that lie fully inside.
    One face per endplate (normal +-x, outward); texture x runs along world -z (left plate) / +z (right plate) and
    texture y along world -y, i.e. both read un-mirrored from outside. Geometry only -> cached."""
    if os.path.exists(EP_CACHE):
        d = np.load(EP_CACHE)
        return {k: d[k] for k in ("L", "R")}
    sys.path.insert(0, os.path.join(LIV, "tools"))
    import render_rs3
    _, _, meshes = render_rs3.load_scene(render_rs3.DEFAULT_KN5, True)
    S = 4
    out = {}
    for nm in ("L", "R"):
        mk = Image.new("L", (1024 * S, 1024 * S), 0)
        dr = ImageDraw.Draw(mk)
        for m in meshes:
            if m["mat"] not in ("glass_sticker", "ext_sticker") or not m["name"].endswith("Plane.033"):
                continue
            I = m["idx"].reshape(-1, 3)
            tx, ty = (m["uv"][:, 0] % 1) * 1024, (m["uv"][:, 1] % 1) * 1024
            for t in I:
                if ty[t].min() < 745 or ty[t].max() > 870 or tx[t].max() > 545:
                    continue
                if (m["pos"][t, 0].mean() > 0) != (nm == "L"):
                    continue
                dr.polygon([(tx[i] * S, ty[i] * S) for i in t], fill=255)
        a = np.asarray(mk) > 0
        out[nm] = a.reshape(1024, S, 1024, S).all(axis=(1, 3))
    os.makedirs(os.path.dirname(EP_CACHE), exist_ok=True)
    np.savez_compressed(EP_CACHE, **out)
    return out


def endplate_layout():
    """KARTING64 pad on the endplates: one common pad height for both plates (the largest that keeps >= EP_MARGIN
    px from the edge of BOTH visible faces and from the holo bands along their long edges), each centred where its
    smallest clearance is largest. Returns {side: (pad height px, cx, cy, clearance px, face bbox)}."""
    faces = endplate_faces()
    S = 4
    p40 = k64_pad(40 * S)
    asp = p40.width / p40.height
    free, box = {}, {}
    for nm in ("L", "R"):
        f = faces[nm].copy()
        f[:766] = False                        # holo band + white line along the top edge
        f[850:] = False                        # ... and along the bottom edge
        ys_, xs_ = np.nonzero(faces[nm])
        free[nm], box[nm] = f, [int(xs_.min()), int(ys_.min()), int(xs_.max()), int(ys_.max())]
    hp0 = math.floor(min(850 - 766 - 2 * EP_MARGIN, min((b[2] - b[0] - 2 * EP_MARGIN) / asp for b in box.values())))
    for hp in np.arange(float(hp0), 20.0, -1.0):
        pad_ = k64_pad(hp * S)
        w_, h_ = pad_.width / S, pad_.height / S
        res = {nm: centre_in_glass(free[nm], box[nm], w_, h_, h_ / 2, 0.0) for nm in ("L", "R")}
        if min(r[2] for r in res.values()) >= EP_MARGIN:
            break
    return {nm: (float(hp), r[0], r[1], r[2], box[nm]) for nm, r in res.items()}


def k64_pad(h):
    """KARTING64 artwork (kart, pilot, RU-flag stripes; no lettering) on the pearl pad of the sill row, pad height h px
    (same proportions as the sill pad: art 75 % of the pad height, 21 % / 12.5 % padding)."""
    return pill(asset_karting64(int(h * 0.75)), int(h * 0.208), int(h * 0.125), PEARL_PAD, line=INK_D)


NAME_SCALE = 0.85       # side-window name plate: 15 % smaller than the 386 x 82 px strip (same aspect ratio)
_NAME_POS = {}


def glass_sticker(driver, num):
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
    gplace(G, "ws_number", number_plate(150 * S, 110 * S, num), k(ws_box), rot=glass_level_rot("Circle.077", ws_box),
           edge=S)

    # rear window banner: Симкарт on night band, chrome edge (reads from behind)
    d.rectangle(k([135, 168, 864, 253]), fill=NIGHT + (255,))
    d.rectangle(k([135, 168, 864, 172]), fill=(250, 248, 255, 255))
    lg = fit_h(blib.simkart_logo(50 * S, glow=True), 54 * S)
    gplace(G, "rw_simkart", lg, k([140, 172, 864, 253]), cx=380 * S, edge=0)
    # («simkarting.ru» is 13 characters, the old address 18: set at 34 px instead of 28 px -> 31 px letters, 223 px long)
    gplace(G, "rw_url", ink_text(SIMKART_URL, F_SPON(34 * S), (236, 236, 244)), k([140, 172, 864, 253]),
           cx=700 * S, edge=0)

    # rear number: rear-window disc decal (Circle.038). Rotation computed from the mesh (-16.8 deg = exactly level).
    # The rear wing hides the disc below a line from ty ~683 (left) to ~715 (right) in the straight rear view and a
    # wing stay crosses it at tx ~850 (z-buffer of the model): the plate was 150x108 at cy 660 and lost its lower
    # third behind the blade, which made it look sheared. Largest plate (same 150:108 proportions) that is >= 6 px
    # inside both the disc and the visible part in rear / rear 3/4 views: 130x94 centred (672, 628).
    rw_box = [586, 547, 878, 818]
    gplace(G, "rw_number", number_plate(130 * S, 94 * S, num), k(rw_box), cx=672 * S, cy=628 * S,
           rot=glass_level_rot("Circle.038", rw_box), edge=S)

    # side windows: driver name + Russian flag on a burgundy strip (lower part of the rear door window)
    # The 386 x 82 strip filled the window zone box, but the visible glass is a trapezoid inside that box (slanted
    # B-pillar edge, C-pillar corner at the top rear): its front corners ran under the frame. The strip is drawn as
    # before and scaled down uniformly by NAME_SCALE (aspect ratio kept), turned by the computed level angle of the
    # glass, and centred where its smallest clearance to the edge of the visible glass (measured with the renderer's
    # z-buffer in the side + both 3/4 views) is largest: equal air to the frame front, rear and bottom.
    first, last = driver
    gvis = glass_side_visible()
    for nm, (box, _) in SIDE_WIN.items():
        w, h = 386 * S, 82 * S
        strip = chrome_frame(w, h, 18 * S, 4 * S, INK)
        fl = ru_flag(60 * S, 40 * S)
        t1 = ink_text(first.upper(), F_NAME(24 * S), (255, 214, 232))
        t2 = ink_text(last.upper(), F_NAME(50 * S), WHITE)
        tx = stack([t1, t2], 4 * S, align="l")
        tx = fit_box(tx, w - 108 * S, h - 16 * S)
        strip.alpha_composite(fl, (18 * S, (h - fl.height) // 2))
        strip.alpha_composite(tx, (92 * S + (w - 110 * S - tx.width) // 2, (h - tx.height) // 2))
        strip = strip.resize((round(w * NAME_SCALE), round(h * NAME_SCALE)), Image.LANCZOS)
        rot = glass_level_rot("Circle.059", box)
        if nm not in _NAME_POS:
            # (+1 px: the deep-ink rim gplace puts under the art)
            ww, hh = strip.width / S + 2, strip.height / S + 2
            _NAME_POS[nm] = centre_in_glass(gvis[nm], box, ww, hh, 18 * NAME_SCALE + 1, rot) + (rot,)
        cx, cy, clr, rot = _NAME_POS[nm]
        gplace(G, f"name_{nm}", strip, k(box), cx=cx * S, cy=cy * S, rot=rot, edge=S)
        mm_px = float(np.linalg.norm(glass_fit("Circle.059", box)[0])) * 1000
        GCHECKS[-1].update(visible_glass_clear_px=round(clr, 1), visible_glass_clear_mm=round(clr * mm_px),
                           centre_px=[round(cx), round(cy)], rot_deg=round(rot, 2),
                           size_px=[strip.width // S, strip.height // S])
        if clr < 8:
            GCHECKS[-1]["status"] = "FAIL: plate too close to the frame / outside the visible glass"

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
    # endplates: burgundy, holo band along both long edges, and ONLY the KARTING64 logo (kart artwork, no lettering)
    # on its pearl pad (the sill treatment), as large as the visible outer face allows (client: the Saratov flag + «64»
    # are gone). Both faces are mapped un-mirrored seen from outside (texture x = toward the rear on the left plate,
    # toward the front on the right one, texture y = down; see endplate_faces), so the art goes on as drawn, not
    # flipped, on both. Size and centre: the largest pad whose outline keeps >= EP_MARGIN px (~1.8 cm) from the edge
    # of the visible face (the face's chamfered / rounded corners from the kn5 UVs) and from the holo bands.
    if not _EP_POS:
        _EP_POS.update(endplate_layout())
    for nm, (ex0, ex1) in (("R", (110, 305)), ("L", (326, 521))):
        d.rectangle(k([ex0, 757, ex1, 859]), fill=INK + (255,))
        hb = holo_rgb((ex1 - ex0) * S, 7 * S, 1.6).convert("RGBA")
        G.alpha_composite(hb, (ex0 * S, 757 * S))
        G.alpha_composite(hb, (ex0 * S, 852 * S))
        d.rectangle(k([ex0, 764, ex1, 765]), fill=(250, 248, 255, 255))
        d.rectangle(k([ex0, 850, ex1, 851]), fill=(250, 248, 255, 255))
        hp, cx, cy, clr, box = _EP_POS[nm]
        pad_ = k64_pad(hp * S)
        gplace(G, f"endplate_k64_{nm}", pad_, k(box), cx=cx * S, cy=cy * S, edge=0)
        mm_px = 1000.0 / 570.0
        GCHECKS[-1].update(size_px=[round(pad_.width / S), round(pad_.height / S)],
                           size_cm=[round(pad_.width / S * mm_px / 10, 1), round(pad_.height / S * mm_px / 10, 1)],
                           centre_px=[round(cx), round(cy)], visible_face_clear_px=round(clr, 1),
                           visible_face_clear_mm=round(clr * mm_px))
        if clr < EP_MARGIN:
            GCHECKS[-1]["status"] = "FAIL: KARTING64 pad too close to the endplate edge / bands"
    out = G.resize((1024, 1024), Image.LANCZOS)
    # side-window zones (only the name plates there): the transparent texels around the plate take the colour of the
    # nearest plate texel. The downsample leaves random colours in near-zero-alpha texels, and the DXT5 colour
    # endpoints of a 4x4 block are chosen from all 16 texels, so a red / cyan transparent texel turned a visible rim
    # texel bright red (and texture filtering would bleed it into the edge in game).
    arr = np.asarray(out).copy()
    for box, _ in SIDE_WIN.values():
        x0, y0, x1, y1 = box
        sub = arr[y0:y1 + 1, x0:x1 + 1]
        op = sub[..., 3] >= 128
        if op.any():
            _, (iy_, ix_) = ndimage.distance_transform_edt(~op, return_indices=True)
            sub[..., :3] = np.where(op[..., None], sub[..., :3], sub[iy_, ix_, :3])
    out = Image.fromarray(arr, "RGBA")
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


NUMBERS = {"Pozdnyakov": "23", "Konopelko": "00"}      # race number per driver (Конопелько: still 00 for now)
NAMES = {"Pozdnyakov": ("Станислав", "Поздняков"), "Konopelko": ("Матвей", "Конопелько")}
# skin folder = Surname_Number (also the zip name, ui_skin.json skinname, renders/<skin>, gear/out/<skin>)
DRIVERS = [(f"{k}_{NUMBERS[k]}", NAMES[k]) for k in ("Pozdnyakov", "Konopelko")]


def skin_number(folder):
    return NUMBERS[folder.split("_")[0]]


def number_air(w, h, num, unit_per_px):
    """Digits of `num` on a w x h px plate (number_digits): scale, ink size (outline + shadow incl.) and the air from the
    ink to the inner edge of the plate's fill, left / right / top / bottom, in the given unit per px."""
    d, x, y, k = number_digits(w, h, num)
    ys, xs = np.nonzero(np.asarray(d.getchannel("A")) > 128)
    bw = max(5, int(h * 0.045))
    e = bw + max(2, bw // 3)
    x0, x1, y0, y1 = x + xs.min(), x + xs.max(), y + ys.min(), y + ys.max()
    f = lambda v: round(float(v) * unit_per_px, 1)
    return dict(num=num, scale=round(k, 4), ink=[f(x1 - x0 + 1), f(y1 - y0 + 1)],
                air=dict(L=f(x0 - e), R=f(w - 1 - e - x1), T=f(y0 - e), B=f(h - 1 - e - y1)))


NUM_NOTES = []

# gear textures made by gear/make_caliper.py, make_gear.py, make_crew.py: every file of gear/out/<skin>/ goes into the
# skin folder (and so into its zip) as is. Both caliper files are needed: caliper.dds keeps the stock alpha 0 and only
# shows pink together with the white caliper_detail.dds.
GEAR_OUT = os.path.join(HERE, "gear", "out")
GEAR_REQUIRED = ("caliper.dds", "caliper_detail.dds",
                 "2016_Suit_DIFF.dds", "DRIVER_Suit.dds", "2016_Gloves_DIFF.dds", "DRIVER_Gloves.dds",
                 "HELMET_2012.dds", "HELMET_2012_Glass.dds",
                 "ac_crew.dds", "Crew_HELMET_Color.dds", "Brands_Crew.dds", "Meccanico_Gadgets.png")


def copy_gear(folder, od):
    """Byte-for-byte copy of every file in gear/out/<folder>/ into the skin folder od."""
    src = os.path.join(GEAR_OUT, folder)
    have = sorted(f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))) if os.path.isdir(src) else []
    missing = [f for f in GEAR_REQUIRED if f not in have]
    if missing:
        raise SystemExit(f"gear/out/{folder}: missing {', '.join(missing)}")
    for f in have:
        shutil.copyfile(os.path.join(src, f), os.path.join(od, f))
    print(f"   gear -> {folder}/: {len(have)} files ({', '.join(have)})")


def main():
    global RACE_NUM
    RACE_NUM = skin_number(DRIVERS[0][0])          # the body is fitted / checked with the first driver's number
    paint_body()
    print("decals ...")
    for s in ("L", "R"):
        side_decals(s)
    top_decals()
    rear_front_decals()
    paint_audi_rings()
    front_line_report()

    pm = PAINTF
    for folder, drv in DRIVERS:
        num = skin_number(folder)
        if num != RACE_NUM:                  # same plates and placements, this driver's digits
            paint_numbers(num)
        out = A0.copy().reshape(-1, 3)
        out[pm] = CAN[pm] * SHADE.reshape(-1)[pm, None]
        skin = Image.fromarray(np.clip(out.reshape(N, N, 3), 0, 255).astype(np.uint8))
        if folder == DRIVERS[0][0]:
            skin.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, "texture_preview.png"))
        od = os.path.join(HERE, folder)
        os.makedirs(od, exist_ok=True)
        save_dxt5(skin, os.path.join(od, "Skin.dds"))
        n_before = len(GCHECKS)
        gl = glass_sticker(drv, num)
        if folder != DRIVERS[0][0]:          # second car: identical layout, keep its name and number checks
            mine = [dict(g, name=g["name"] + "@" + folder) for g in GCHECKS[n_before:]
                    if g["name"].startswith("name_") or g["name"].endswith("_number")]
            del GCHECKS[n_before:]
            GCHECKS.extend(mine)
        save_dxt5(gl, os.path.join(od, "glass_sticker.dds"))
        if folder == DRIVERS[0][0]:
            gl.save(os.path.join(HERE, "glass_preview.png"))
        json.dump({"skinname": folder, "drivername": f"{drv[0]} {drv[1]}", "country": "Russia",
                   "team": "Команда ЭДМ", "number": num, "priority": 1},
                  open(os.path.join(od, "ui_skin.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        livery_icon(od, num)
        copy_gear(folder, od)
        with zipfile.ZipFile(os.path.join(HERE, folder + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(os.listdir(od)):
                z.write(os.path.join(od, f), folder + "/" + f)
        # digits on every plate of this car (body: cm, glass: px of the 1024 glass_sticker; icon: px of 185)
        plates = [(j["name"], px(j["size"]), j["make_art"](j["size"]).height, 100.0 / PPM, "cm") for j in NUM_JOBS]
        plates += [("ws_number", 600, 440, 0.25, "px"), ("rw_number", 520, 376, 0.25, "px"), ("livery_icon", 480, 320, 0.25, "px")]
        for nm, w_, h_, upp, unit in plates:
            NUM_NOTES.append(dict(skin=folder, plate=nm, unit=unit, plate_size=[round(w_ * upp, 1), round(h_ * upp, 1)],
                                  this=number_air(w_, h_, num, upp), ref=number_air(w_, h_, NUM_REF, upp)))
    # team zip: both skin folders at its root
    with zipfile.ZipFile(os.path.join(os.path.dirname(HERE), "butcher_chrome.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for folder, _ in DRIVERS:
            od = os.path.join(HERE, folder)
            for f in sorted(os.listdir(od)):
                z.write(os.path.join(od, f), folder + "/" + f)
    rep = []
    for r in CHECKS:
        rep.append(f"{r['status']:<6} {r['name']:<26} {str(r['size_cm']):<14} cov={r['coverage']:.3f} "
                   f"edge={r['edge_clear_cm']:>5}cm tilt={r['max_tilt_deg']:>5}° level={r.get('level_deg')}°({r.get('level_view')}) "
                   f"in_mask={r.get('in_zone_mask')} "
                   f"mask_clear={r.get('mask_clear_cm')}cm lines={r.get('line_clear_cm')}cm"
                   + (f" decals={r.get('decal_clear_cm')}cm hidden={r['hidden_frac']}" if "hidden_frac" in r else ""))
    for r in GCHECKS:  # glass checks of the first car + the name / number checks of the second (same layout)
        extra = ""
        if "visible_glass_clear_px" in r:
            extra = (f" size={r['size_px']}px centre={r['centre_px']} rot={r['rot_deg']}deg"
                     f" visible_glass_clear={r['visible_glass_clear_px']}px/{r['visible_glass_clear_mm']}mm")
        rep.append(f"{r['status']:<6} glass:{r['name']:<20} box={r['box']} art={r['art_bbox']} clear={r['clear_px']}px"
                   + extra)
    for r in LINE_CHECKS:        # front cut line along the headlight
        va = ", ".join(f"{v} {a[0]}-{a[1]}" for v, a in r["visible_air_cm"].items())
        a3 = f"{r['air_3d_cm'][0]}-{r['air_3d_cm'][1]}"
        rep.append(f"{r['status']:<6} line:headlight_{r['side']:<14} {r['dashes']} whole dashes, stretch {r['stretch']}, "
                   f"{r['length_cm']} cm; air to the lamp outline on the paint (narrowest per 6 cm): {a3} cm; visible air along the "
                   f"lamp in the close-ups (narrowest per 6 cm, cm): {va}; other parts >= {r['air_other_cm']} cm; labels "
                   + ", ".join(f"{k} {v}" for k, v in r["label_air_cm"].items()) + " cm"
                   + ("" if r["status"] == "OK" else f"; truncated {r['truncated']} hidden {r['hidden']}"))
    for r in BELLY_DROPPED:      # belly cut-line dashes left out whole instead of showing as stubs
        rep.append(f"INFO   line:belly dash {r['dash']}{r['side']:<3} left out whole ({r['why']}) at x={r['x']:+.3f} "
                   f"y={r['y']:+.3f}")
    for r in LINE_NOTES:
        rep.append("INFO   line:" + r)
    for r in SILL_LOG:
        rep.append(f"INFO   sill row {r['side']}: height {r['height_cm']} cm, centre z {r['z_centre']}, y {r['y_front']:+.3f}"
                   f" .. {r['y_rear']:+.3f}, equal gaps {r['gap_cm']} cm: "
                   + ", ".join(f"{g['kind']} {g['width_cm']} cm @ y {g['y']:+.3f}" for g in r["logos"]))
    for r in URL_NOTES:
        rep.append("INFO   url:" + r)
    for r in NUM_NOTES:
        t, f = r["this"], r["ref"]
        a = lambda d: f"L {d['air']['L']} R {d['air']['R']} T {d['air']['T']} B {d['air']['B']}"
        rep.append(f"INFO   number {r['skin']} «{t['num']}» {r['plate']} plate {r['plate_size'][0]} x {r['plate_size'][1]} "
                   f"{r['unit']}: digits {t['ink'][0]} x {t['ink'][1]} {r['unit']} at the «{f['num']}» scale "
                   f"({t['scale']}), air to the plate {a(t)} {r['unit']}"
                   + ("" if t['num'] == f['num'] else f" («{f['num']}»: digits {f['ink'][0]} x {f['ink'][1]}, air {a(f)})"))
    txt = "\n".join(rep)
    print(txt)
    with open(os.path.join(HERE, "check_report.txt"), "w") as fh:
        fh.write(txt + "\n")
    json.dump(dict(skin=CHECKS, glass=GCHECKS, belly_dropped=BELLY_DROPPED, line_notes=LINE_NOTES, sill_row=SILL_LOG,
                   url_notes=URL_NOTES, front_cut_line=LINE_CHECKS, numbers=NUM_NOTES), open(os.path.join(HERE, "check_report.json"), "w"),
              ensure_ascii=False, indent=1, default=str)

    print("done")


def livery_icon(od, num):
    """185x185 livery swatch: pink field, holo lower band with chrome keyline, dashed cut line, chrome race number."""
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
    p = number_plate(120 * S, 80 * S, num)
    im.alpha_composite(p, ((w - p.width) // 2, 16 * S))
    im.alpha_composite(sparkle(14 * S), (w - 46 * S, 8 * S))
    im.convert("RGB").resize((185, 185), Image.LANCZOS).save(os.path.join(od, "livery.png"))


if __name__ == "__main__":
    main()
