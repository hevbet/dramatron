"""2D artwork for concept «Pro Pink TCR»: fonts, plates, brand logos (recoloured traced masks), flags.

Every function returns a tight RGBA PIL image; sizes are in pixels of the artwork, aspect ratio of
brand marks is always preserved (only uniform scaling).
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

LIV = "/home/user/dramatron/livery"
FONTS = os.path.join(LIV, "fonts")
BRAND = os.path.join(LIV, "br03", "brand")
BX = os.path.join(LIV, "br03_pro", "brand_extra")
sys.path.insert(0, os.path.join(LIV, "br03_pro"))
import lib as blib  # noqa: E402  (arka_logo, simkart_logo, driveoil_logo from the BR03 project)

# ---------------------------------------------------------------- palette
MAG = (232, 30, 128)        # «Pro Pink» – hot magenta-pink
MAG_L = (255, 120, 190)     # light pink (hairlines, highlights)
PLUM = (40, 15, 44)         # deep plum (lower body / car-2 upper)
PLUM_D = (22, 8, 26)        # outlines
WHITE = (250, 248, 252)
SILVER = (214, 214, 224)
RU = [(255, 255, 255), (0, 57, 166), (213, 43, 30)]
HOLO = np.array([(255, 128, 205), (208, 160, 255), (140, 206, 255), (150, 248, 222), (255, 242, 196),
                 (255, 150, 210)], np.float32)


# ---------------------------------------------------------------- fonts (decided once for the whole car)
def _font(name, size, var=None, axes=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), max(4, int(size)))
    if var:
        f.set_variation_by_name(var)
    if axes:
        f.set_variation_by_axes(axes)
    return f


F_NUM = lambda s: _font("Unbounded[wght].ttf", s, "Black")                          # race numbers
F_NAME = lambda s: _font("SofiaSansCondensed-Italic[wght].ttf", s, "Black Italic")  # driver names
F_SPON = lambda s: _font("Exo2-Italic[wght].ttf", s, "ExtraBold Italic")            # urls / partner lines
F_TEAM = lambda s: _font("MontserratAlternates-BlackItalic.ttf", s)                 # «Команда ЭДМ»
F_TECH = lambda s: _font("Tektur[wdth,wght].ttf", s, axes=[100, 800])               # KARTING64.RU / 64
F_CUT = lambda s: _font("YesevaOne-Regular.ttf", s)                                 # butcher cut labels
F_Y2K = lambda s: _font("RubikMonoOne-Regular.ttf", s)                              # chrome headline


# ---------------------------------------------------------------- primitives
def text_mask(s, f, track=0.0):
    if not track:
        bb = f.getbbox(s)
        m = Image.new("L", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), 0)
        ImageDraw.Draw(m).text((4 - bb[0], 4 - bb[1]), s, font=f, fill=255)
        return m.crop(m.getbbox())
    gap = f.size * track
    w = int(sum(f.getlength(c) for c in s) + gap * len(s)) + 40
    asc, desc = f.getmetrics()
    m = Image.new("L", (w, asc + desc + 40), 0)
    d = ImageDraw.Draw(m)
    x = 20
    for c in s:
        d.text((x, 20), c, font=f, fill=255)
        x += f.getlength(c) + gap
    return m.crop(m.getbbox())


def solid(mask, col):
    t = Image.new("RGBA", mask.size, tuple(col[:3]) + (255,))
    t.putalpha(mask)
    return t


def text(s, f, col, track=0.0):
    return solid(text_mask(s, f, track), col)


def grow(mask, r):
    a = np.asarray(mask) > 127
    d = ndimage.distance_transform_edt(~a)
    return Image.fromarray((np.clip(r + 0.5 - d, 0, 1) * 255).astype(np.uint8))


def pad(img, p):
    out = Image.new(img.mode, (img.width + 2 * p, img.height + 2 * p), 0 if img.mode == "L" else (0, 0, 0, 0))
    out.paste(img, (p, p))
    return out


def fit_h(img, h):
    return img.resize((max(1, round(img.width * h / img.height)), max(1, int(h))), Image.LANCZOS)


def fit_w(img, w):
    return img.resize((max(1, int(w)), max(1, round(img.height * w / img.width))), Image.LANCZOS)


def fit_box(img, w, h):
    k = min(w / img.width, h / img.height)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def stack(items, gap, align="c"):
    w = max(i.width for i in items)
    h = sum(i.height for i in items) + gap * (len(items) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y = 0
    for i in items:
        x = (w - i.width) // 2 if align == "c" else (0 if align == "l" else w - i.width)
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


def chrome_rgb(w, h, tint=(1.0, 0.97, 1.03)):
    y = np.linspace(0, 1, h)[:, None]
    st = [(0, 250), (0.36, 228), (0.47, 150), (0.52, 92), (0.6, 200), (0.84, 244), (1, 196)]
    v = np.interp(y, [s[0] for s in st], [s[1] for s in st]) * np.ones((1, w))
    rgb = np.stack([v * tint[0], v * tint[1], v * tint[2]], -1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def holo_lookup(t):
    t = (np.asarray(t, np.float32) % 1.0) * (len(HOLO) - 1)
    k = np.floor(t).astype(int)
    f = (t - k)[..., None]
    return HOLO[k] * (1 - f) + HOLO[np.minimum(k + 1, len(HOLO) - 1)] * f


def holo_rgb(w, h, scale=1.0, phase=0.0):
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    return Image.fromarray(holo_lookup((x * 1.3 + y * 0.5) * scale + phase).astype(np.uint8))


def fill(mask, rgb_img):
    t = rgb_img.convert("RGBA").resize(mask.size)
    t.putalpha(mask)
    return t


def chamfer_mask(w, h, c, ss=3):
    """Plate with chamfered (cut) corners: top-left and bottom-right cut larger – the 'blade' plate shape."""
    W, H, C = w * ss, h * ss, c * ss
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon([(C, 0), (W - C * 0.35, 0), (W, C * 0.35), (W, H - C), (W - C, H),
                               (C * 0.35, H), (0, H - C * 0.35), (0, C)], fill=255)
    return m.resize((w, h), Image.LANCZOS)


def blade_plate(w, h, body, frame="chrome", fw=None, c=None, mirror=False):
    """Chamfered plate: chrome (or flat-colour) frame + body colour."""
    fw = fw or max(3, int(h * 0.05))
    c = c or int(h * 0.22)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    outer = chamfer_mask(w, h, c)
    inner = Image.new("L", (w, h), 0)
    inner.paste(chamfer_mask(w - 2 * fw, h - 2 * fw, max(2, c - fw // 2)), (fw, fw))
    if frame == "chrome":
        out.alpha_composite(fill(outer, chrome_rgb(w, h)))
    elif frame is not None:
        out.alpha_composite(solid(outer, frame))
    out.alpha_composite(solid(inner if frame is not None else outer, body))
    if mirror:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    return out


def chrome_text(s, f, ow, outline=PLUM_D, track=0.0, holo_core=False):
    m = pad(text_mask(s, f, track), ow * 2 + 6)
    ol = grow(m, ow)
    out = Image.new("RGBA", m.size, (0, 0, 0, 0))
    out.alpha_composite(solid(ol, outline))
    out.alpha_composite(fill(m, holo_rgb(m.width, m.height, 1.2) if holo_core else chrome_rgb(m.width, m.height)))
    return out.crop(out.getbbox())


# ---------------------------------------------------------------- brand assets
def _L(path):
    return Image.open(path).convert("L")


def arka(h, taimcafe=True):
    """тайм-кафе «Арка»: brand yellow word, dark outline + drop shadow (as on the club's posters)."""
    return blib.arka_logo(int(h), fill=(244, 208, 92), dark=PLUM_D, taimcafe=taimcafe)


def simkart(h, glow=True):
    return blib.simkart_logo(int(h), silver=(226, 224, 236), red=(238, 50, 40), glow=glow)


def simkart_plate(w, url=True, body=PLUM_D):
    """«Симкарт» on its night plate with chrome blade frame; url under the wordmark."""
    logo = fit_w(simkart(int(w * 0.2)), int(w * 0.78))
    items = [logo]
    if url:
        items.append(fit_w(text("simkart.vercel.app", F_SPON(200), (236, 236, 246)), int(w * 0.52)))
    inner = stack(items, int(w * 0.004))
    h = inner.height + int(w * 0.09)
    p = blade_plate(int(w), h, body, fw=max(3, int(h * 0.045)))
    p.alpha_composite(inner, ((p.width - inner.width) // 2, (p.height - inner.height) // 2))
    return p


def smp(h, white=WHITE, accent=(0, 181, 239)):
    w = _L(os.path.join(BX, "smp_racing_esports_white.png"))
    a = _L(os.path.join(BX, "smp_racing_esports_accent.png"))
    im = Image.new("RGBA", w.size, (0, 0, 0, 0))
    im.alpha_composite(solid(w, white))
    im.alpha_composite(solid(a, accent))
    return fit_h(im.crop(im.getbbox()), h)


def mono(name, h, col):
    m = _L(os.path.join(BX, name))
    return fit_h(solid(m.crop(m.getbbox()), col), h)


def coat(h):
    """Saratov oblast coat of arms (azure shield, three silver sterlets, golden crown)."""
    sh = _L(os.path.join(BX, "saratov_coa_shield.png"))
    im = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    im.alpha_composite(solid(sh, (20, 30, 60)))
    im.alpha_composite(solid(sh.filter(ImageFilter.MinFilter(31)), (22, 148, 220)))
    im.alpha_composite(solid(_L(os.path.join(BX, "saratov_coa_crown.png")), (222, 178, 60)))
    im.alpha_composite(solid(_L(os.path.join(BX, "saratov_coa_crown_lines.png")), (20, 30, 60)))
    im.alpha_composite(solid(_L(os.path.join(BX, "saratov_coa_fish.png")), (232, 236, 244)))
    im.alpha_composite(solid(_L(os.path.join(BX, "saratov_coa_lines.png")), (20, 30, 60)))
    return fit_h(im.crop(im.getbbox()), h)


def mbu(h):
    im = Image.open(os.path.join(BX, "edm_mbu_logo.png")).convert("RGBA")
    return fit_h(im.crop(im.getbbox()), h)


def karting64(h, text_col=WHITE, flip=False, outline=None):
    """KARTING64: kart + pilot + flag (top part of the logo), url set in Tektur."""
    k = _L(os.path.join(BX, "karting64_black.png"))
    cut = 1500
    im = Image.new("RGBA", (k.width, cut), (0, 0, 0, 0))
    for n, c in (("white", WHITE), ("blue", (0, 57, 166)), ("red", (213, 43, 30)), ("black", text_col)):
        im.alpha_composite(solid(_L(os.path.join(BX, f"karting64_{n}.png")).crop((0, 0, k.width, cut)), c))
    im = im.crop(im.getbbox())
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    kart = fit_h(im, int(h * 0.6))
    t = fit_w(text("KARTING64.RU", F_TECH(300), text_col, track=0.02), int(kart.width * 1.0))
    return stack([kart, t], int(h * 0.06))


def driveoil(h, plate=True):
    return blib.driveoil_logo(int(h), plate=plate)


def ru_flag(w, h, border=PLUM_D):
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    bw = max(1, h // 14)
    d.rectangle([0, 0, w - 1, h - 1], fill=border + (255,))
    for i, c in enumerate(RU):
        y0 = bw + i * (h - 2 * bw) / 3
        d.rectangle([bw, y0, w - 1 - bw, bw + (i + 1) * (h - 2 * bw) / 3 - 1], fill=c + (255,))
    return t


def saratov_flag(w, h, border=PLUM_D):
    """Saratov oblast flag: white with red lower third, coat of arms in the canton."""
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, w - 1, h - 1], fill=border + (255,))
    bw = max(1, h // 18)
    d.rectangle([bw, bw, w - 1 - bw, h - 1 - bw], fill=(255, 255, 255, 255))
    d.rectangle([bw, int(h * 2 / 3), w - 1 - bw, h - 1 - bw], fill=(206, 32, 42, 255))
    c = coat(int(h * 0.5))
    t.alpha_composite(c, (int(w * 0.08), int(h * 0.08)))
    return t


def number_plate(w, h, num="00", body=WHITE, digit=PLUM, frame="chrome"):
    p = blade_plate(w, h, body, frame=frame, fw=max(4, int(h * 0.045)))
    dmask = text_mask(num, F_NUM(int(h * 1.2)), track=0.02)
    dg = fit_box(solid(dmask, digit), w * 0.78, h * 0.68)
    p.alpha_composite(dg, ((w - dg.width) // 2, (h - dg.height) // 2))
    return p


def name_tag(surname, first, w, flag=True, col=WHITE):
    """Driver name block: Russian flag + SURNAME (Sofia Sans Condensed Black Italic) + first name above."""
    sur = text(surname.upper(), F_NAME(400), col, track=0.02)
    fst = text(first.upper(), F_NAME(400), col, track=0.08)
    fst = fit_h(fst, int(sur.height * 0.42))
    block = stack([fst, sur], int(sur.height * 0.14), align="l")
    if flag:
        fl = ru_flag(int(block.height * 1.0), int(block.height * 0.66))
        block = row([fl, block], int(block.height * 0.18))
    return fit_w(block, w)


def team_edm(h, col=WHITE, sticker=True):
    """«КОМАНДА ЭДМ» lockup with the round МБУ «Клуб Энгельсская молодёжь» sticker."""
    t1 = text("КОМАНДА", F_TEAM(300), col, track=0.04)
    t2 = text("ЭДМ", F_TEAM(300), col, track=0.02)
    t2 = fit_w(t2, t1.width)
    block = stack([t1, t2], int(t1.height * 0.2))
    if not sticker:
        return fit_h(block, h)
    st = mbu(block.height)
    return fit_h(row([st, block], int(block.height * 0.18)), h)


def region64(h, col=WHITE):
    """Region identity: coat of arms + «64» + «САРАТОВСКАЯ ОБЛАСТЬ»."""
    c = coat(400)
    n = fit_h(text("64", F_TECH(500), col), int(c.height * 0.62))
    t = fit_w(text("САРАТОВСКАЯ ОБЛАСТЬ", F_SPON(200), col, track=0.03), int(n.width * 2.4))
    right = stack([n, t], int(c.height * 0.06), align="l")
    right = fit_h(right, c.height)
    return fit_h(row([c, right], int(c.height * 0.12)), h)


def cut_label(s, h, col):
    """Butcher cut label (Yeseva One), letter-spaced, with a small diamond underline."""
    t = fit_h(text(s, F_CUT(300), col, track=0.08), h)
    lw = max(2, h // 12)
    u = Image.new("RGBA", (t.width, lw * 6), (0, 0, 0, 0))
    d = ImageDraw.Draw(u)
    c = t.width / 2
    d.line([(t.width * 0.12, lw * 3), (c - lw * 4, lw * 3)], fill=col + (255,), width=lw)
    d.line([(c + lw * 4, lw * 3), (t.width * 0.88, lw * 3)], fill=col + (255,), width=lw)
    d.polygon([(c - lw * 2.5, lw * 3), (c, 0), (c + lw * 2.5, lw * 3), (c, lw * 6)], fill=col + (255,))
    return stack([t, u], int(h * 0.15))


def sparkle(r, col=WHITE):
    s = int(r * 2.6) * 2
    m = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(m)
    c = s / 2
    pts = []
    for k in range(8):
        a = math.radians(k * 45 - 90)
        rr = r if k % 2 == 0 else r * 0.2
        if k % 4 == 2:
            rr = r * 0.6
        pts.append((c + math.cos(a) * rr, c + math.sin(a) * rr))
    d.polygon(pts, fill=255)
    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    out.alpha_composite(solid(m.filter(ImageFilter.GaussianBlur(r * 0.22)).point(lambda v: int(v * 0.8)), MAG_L))
    out.alpha_composite(fill(m, holo_rgb(s, s, 0.8, 0.1)))
    out.alpha_composite(solid(m.filter(ImageFilter.MinFilter(3)).point(lambda v: int(v * 0.7)), col))
    return out.crop(out.getbbox())


if __name__ == "__main__":
    bg = Image.new("RGBA", (2400, 1800), MAG + (255,))
    d = ImageDraw.Draw(bg)
    d.rectangle([0, 900, 2400, 1800], fill=PLUM + (255,))
    items = [
        (arka(160), (40, 40)), (simkart_plate(520), (520, 40)), (smp(90), (1120, 40)),
        (mono("br_engineering_black.png", 60, WHITE), (40, 300)), (mono("raf_black.png", 200, WHITE), (900, 300)),
        (coat(200), (1150, 300)), (mbu(200), (1350, 300)), (karting64(200), (1600, 300)), (driveoil(80), (40, 420)),
        (number_plate(500, 260), (40, 560)), (name_tag("Поздняков", "Станислав", 600), (600, 600)),
        (team_edm(150), (1300, 600)), (region64(150), (1300, 760)), (saratov_flag(300, 200), (2000, 560)),
        (cut_label("ВЫРЕЗКА", 80, PLUM), (600, 780)),
        (number_plate(500, 260, body=WHITE, digit=PLUM), (40, 950)), (arka(160), (600, 950)),
        (smp(90), (1100, 950)), (karting64(200), (1700, 950)), (team_edm(150), (40, 1300)), (region64(150), (700, 1300)),
        (chrome_text("PINK PIG", F_Y2K(120), 6), (1300, 1300)), (sparkle(60), (2100, 1300)),
        (cut_label("ОКОРОК", 80, MAG_L), (1300, 1550)), (simkart(120), (40, 1550)),
    ]
    for im, p in items:
        bg.alpha_composite(im, p)
    bg.save(sys.argv[1] if len(sys.argv) > 1 else "/tmp/assets.png")
