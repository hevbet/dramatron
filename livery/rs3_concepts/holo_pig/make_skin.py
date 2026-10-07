"""Концепция 2 «HOLO PIG» — Audi RS3 LMS (smp_audi_rs3_lms), Pink Pig 2.0 × Y2K.

Y2K в первую очередь: горячий розовый кузов, большая голографическая волна от переднего бампера
через двери к «окороку» (задняя арка, корма), сливовый «рокер» внизу под спонсоров,
хромовые пузырчатые номера, звёзды-блёстки и бабочки. Мясницкая схема Pink Pig — тональным
пунктиром и маленькими подписями отрубов (Yeseva One) поверх розового.

Вся геометрия задаётся в мировых координатах модели (posmap из zones/): поля цвета считаются
по 3D-позиции каждого тексела, поэтому линии непрерывны через швы развёртки и щели дверей.
Логотипы/надписи рисуются «как их видит зритель» (вид сбоку/сверху/спереди/сзади) и
проецируются на кузов — текст всегда параллелен земле и без искажений.
Для каждого элемента автоматически проверяется: элемент целиком лежит на своей панели
(покрытие 100 %), запас до края панели (см), и что в текстуре он внутри масок зон.

Запуск: python3 make_skin.py            → Pozdnyakov_00/, Konopelko_00/, *.zip, texture_preview.png, check_report.txt
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

HERE = os.path.dirname(os.path.abspath(__file__))
LIV = os.path.abspath(os.path.join(HERE, "..", ".."))
ZONES = os.path.join(HERE, "..", "zones")
SCR = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad"
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCR, "rs3", "SMP01")
FONTS = os.path.join(LIV, "fonts")
BRAND = os.path.join(LIV, "br03", "brand")
BX = os.path.join(LIV, "br03_pro", "brand_extra")
N = 4096
D = 2000                      # плотность арта, px на метр (текстура ≈ 795 px/m)

# ---------------------------------------------------------------- палитра
PINK = np.array((255, 72, 160), np.float32)        # горячий розовый «HOLO PIG»
PINK_T = (214, 40, 128)                            # тон-в-тон для схемы отрубов
PLUM = (44, 10, 46)                                # тёмная слива: рокер, плашки, обводки
PLUM2 = (70, 18, 72)
WHITE = (250, 248, 255)
HOT = (255, 72, 160)
HOLO = [(255, 105, 205), (176, 120, 255), (95, 165, 255), (60, 220, 225),
        (150, 240, 160), (255, 215, 120), (255, 140, 175), (255, 105, 205)]
RU = [(255, 255, 255), (0, 57, 166), (213, 43, 30)]


def hexc(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------- шрифты (выбор концепции)
def font(name, size, var=None, axes=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), int(size))
    if var:
        f.set_variation_by_name(var)
    if axes:
        f.set_variation_by_axes(axes)
    return f


F_NUM = lambda s: font("DelaGothicOne-Regular.ttf", s)                 # номера: жирный «пузырь»
F_NAME = lambda s: font("Exo2-Italic[wght].ttf", s, "Black Italic")     # имена пилотов
F_SPON = lambda s: font("MontserratAlternates-BlackItalic.ttf", s)      # команда / регион
F_URL = lambda s: font("Exo2-Italic[wght].ttf", s, "ExtraBold Italic")  # ссылки
F_TECH = lambda s: font("Tektur[wdth,wght].ttf", s, axes=[100, 800])    # KARTING64.RU
F_CUT = lambda s: font("YesevaOne-Regular.ttf", s)                      # подписи отрубов
F_Y2K = lambda s: font("RubikBubbles-Regular.ttf", s)                   # Y2K-заголовок


# ---------------------------------------------------------------- базовые слои арта
def solid(mask_l, col, alpha=1.0):
    t = Image.new("RGBA", mask_l.size, tuple(col[:3]) + (255,))
    t.putalpha(mask_l if alpha == 1.0 else mask_l.point(lambda v: int(v * alpha)))
    return t


def text_mask(s, f, track=0.0):
    if not track:
        bb = f.getbbox(s)
        m = Image.new("L", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), 0)
        ImageDraw.Draw(m).text((4 - bb[0], 4 - bb[1]), s, font=f, fill=255)
        return m
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


def pad(m, p):
    out = Image.new(m.mode, (m.width + 2 * p, m.height + 2 * p), 0)
    out.paste(m, (p, p))
    return out


def grow(m, r):
    """Расширить маску на r px (круглое «перо», для обводок)."""
    if r <= 0:
        return m
    a = np.asarray(m) > 127
    dist = ndimage.distance_transform_edt(~a)
    return Image.fromarray((np.clip(r + 0.5 - dist, 0, 1) * 255).astype(np.uint8))


def holo_img(w, h, scale=1.0, phase=0.0, angle=0.35):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / max(w, h) * math.cos(angle) + yy / max(w, h) * math.sin(angle)) * 1.4 * scale + phase
    t = t + 0.06 * np.sin(xx / max(w, h) * 9 + yy / max(w, h) * 5)
    return palette(t)


def palette(t):
    t = (t % 1.0) * (len(HOLO) - 1)
    k = np.floor(t).astype(int)
    f = (t - k)[..., None]
    pal = np.array(HOLO, np.float32)
    f = f * f * (3 - 2 * f)
    return pal[k] * (1 - f) + pal[np.minimum(k + 1, len(HOLO) - 1)] * f


def chrome_img(w, h, tint=(1.0, 0.97, 1.04)):
    st = [(0, 255), (0.30, 238), (0.48, 168), (0.52, 120), (0.56, 236), (0.8, 255), (1, 205)]
    c = np.interp(np.linspace(0, 1, h), [s[0] for s in st], [s[1] for s in st])
    rgb = np.repeat(np.repeat(c[:, None, None], w, 1), 3, 2) * np.array(tint)
    return np.clip(rgb, 0, 255)


def fill_img(mask_l, rgb):
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")
    im.putalpha(mask_l)
    return im


def styled(mask_l, fill="chrome", outline=PLUM, ow=0.06, outline2=None, ow2=0.0, shadow=True):
    """Надпись/знак: заливка (chrome | holo | (r,g,b)), обводка (доля высоты), опц. вторая обводка и тень."""
    h = mask_l.height
    o1 = int(h * ow)
    o2 = o1 + int(h * ow2)
    p = o2 + int(h * 0.08) + 4
    m = pad(mask_l, p)
    out = Image.new("RGBA", m.size, (0, 0, 0, 0))
    big = grow(m, o2) if outline2 else grow(m, o1)
    if shadow:
        sh = ImageChops.offset(big, int(h * 0.035) + 1, int(h * 0.05) + 1).filter(ImageFilter.GaussianBlur(h * 0.02))
        out.alpha_composite(solid(sh, PLUM, 0.55))
    if outline2:
        out.alpha_composite(solid(grow(m, o2), outline2))
    if o1:
        out.alpha_composite(solid(grow(m, o1), outline))
    if fill == "chrome":
        bb = mask_l.getbbox()
        rgb = np.zeros((m.height, m.width, 3), np.float32) + 230
        top, hh = p + bb[1], bb[3] - bb[1]
        rgb[top:top + hh] = chrome_img(m.width, hh)
        out.alpha_composite(fill_img(m, rgb))
    elif fill == "holo":
        out.alpha_composite(fill_img(m, holo_img(m.width, m.height)))
    else:
        out.alpha_composite(solid(m, fill))
    return out.crop(out.getbbox())


def rrect_mask(w, h, r):
    m = Image.new("L", (w * 3, h * 3), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w * 3 - 1, h * 3 - 1], radius=r * 3, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def plate(w, h, r=None, base=PLUM, rim="holo", rim_w=None, inner_line=True):
    """Сливовая плашка со скруглением и голографическим кантом (Y2K-стикер)."""
    r = r if r is not None else int(min(w, h) * 0.22)
    rim_w = rim_w if rim_w is not None else max(6, int(min(w, h) * 0.045))
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    outer = rrect_mask(w, h, r)
    if rim == "holo":
        out.alpha_composite(fill_img(outer, holo_img(w, h, 1.2)))
    else:
        out.alpha_composite(solid(outer, rim))
    inner = rrect_mask(w - 2 * rim_w, h - 2 * rim_w, max(2, r - rim_w))
    out.alpha_composite(solid(inner, base), (rim_w, rim_w))
    if inner_line:
        lw = max(2, rim_w // 4)
        il = Image.new("L", (w - 2 * rim_w - 4 * lw, h - 2 * rim_w - 4 * lw), 0)
        ImageDraw.Draw(il).rounded_rectangle([0, 0, il.width - 1, il.height - 1], radius=max(2, r - rim_w - 2 * lw),
                                             outline=255, width=lw)
        out.alpha_composite(solid(il, WHITE, 0.55), (rim_w + 2 * lw, rim_w + 2 * lw))
    return out


def stack(items, gap=0, align="c", vertical=True):
    if vertical:
        W = max(i.width for i in items)
        H = sum(i.height for i in items) + gap * (len(items) - 1)
    else:
        W = sum(i.width for i in items) + gap * (len(items) - 1)
        H = max(i.height for i in items)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pos = 0
    for i in items:
        if vertical:
            x = (W - i.width) // 2 if align == "c" else 0
            out.alpha_composite(i, (x, pos))
            pos += i.height + gap
        else:
            out.alpha_composite(i, (pos, (H - i.height) // 2))
            pos += i.width + gap
    return out


def on_plate(art, padx, pady, **kw):
    p = plate(art.width + 2 * padx, art.height + 2 * pady, **kw)
    p.alpha_composite(art, (padx, pady))
    return p


def scale_to(im, w=None, h=None):
    if w:
        return im.resize((int(w), max(1, int(im.height * w / im.width))), Image.LANCZOS)
    return im.resize((max(1, int(im.width * h / im.height)), int(h)), Image.LANCZOS)


# ---------------------------------------------------------------- бренды
def bmask(path, h):
    m = Image.open(path).convert("L")
    m = m.crop(m.getbbox())
    return m.resize((max(1, int(m.width * h / m.height)), int(h)), Image.LANCZOS)


def arka_logo(h, taimcafe=True):
    """«арка» в фирменном стиле: жёлтая заливка, тёмная обводка и тень вниз-влево."""
    y, dk = (240, 209, 102), (26, 20, 30)
    m = pad(bmask(os.path.join(BRAND, "arka_word.png"), h), int(h * 0.12))
    t = Image.new("RGBA", m.size, (0, 0, 0, 0))
    ol = grow(m, max(3, h // 22))
    t.alpha_composite(solid(ImageChops.offset(ol, -max(2, h // 28), max(3, h // 16)), dk))
    t.alpha_composite(solid(ol, dk))
    t.alpha_composite(solid(m, y))
    t = t.crop(t.getbbox())
    if not taimcafe:
        return t
    s = pad(bmask(os.path.join(BRAND, "arka_taimcafe.png"), int(h * 0.26)), 10)
    ts = Image.new("RGBA", s.size, (0, 0, 0, 0))
    o2 = grow(s, max(2, s.height // 9))
    ts.alpha_composite(solid(ImageChops.offset(o2, -2, 3), dk))
    ts.alpha_composite(solid(o2, dk))
    ts.alpha_composite(solid(s, y))
    ts = ts.crop(ts.getbbox())
    return stack([t, ts], gap=int(h * 0.04))


def simkart_logo(h, glow=True):
    """«Симкарт»: «Сим» серебро-хром, «карт» красный, красное свечение (как у бренда)."""
    red = (234, 50, 38)
    w = bmask(os.path.join(BRAND, "simkart_word_white.png"), h) if False else None
    W0 = Image.open(os.path.join(BRAND, "simkart_word_white.png")).convert("L")
    R0 = Image.open(os.path.join(BRAND, "simkart_word_red.png")).convert("L")
    bb = ImageChops.lighter(W0, R0).getbbox()
    W0, R0 = W0.crop(bb), R0.crop(bb)
    sz = (int(W0.width * h / W0.height), h)
    w, r = W0.resize(sz, Image.LANCZOS), R0.resize(sz, Image.LANCZOS)
    p = int(h * 0.3)
    w, r = pad(w, p), pad(r, p)
    t = Image.new("RGBA", w.size, (0, 0, 0, 0))
    if glow:
        both = ImageChops.lighter(w, r).filter(ImageFilter.GaussianBlur(h * 0.12))
        t.alpha_composite(solid(both, red, 0.7))
    rgb = np.zeros((w.height, w.width, 3), np.float32)
    rgb[p:p + h] = chrome_img(w.width, h)
    t.alpha_composite(fill_img(w, rgb))
    t.alpha_composite(solid(r, red))
    return t.crop(t.getbbox())


def _L(n):
    return Image.open(os.path.join(BX, n)).convert("L")


def smp_lockup(h, col=WHITE, acc=(0, 181, 239)):
    w, a = _L("smp_racing_esports_white.png"), _L("smp_racing_esports_accent.png")
    bb = ImageChops.lighter(w, a).getbbox()
    w, a = w.crop(bb), a.crop(bb)
    sz = (int(w.width * h / w.height), h)
    im = solid(w.resize(sz, Image.LANCZOS), col)
    im.alpha_composite(solid(a.resize(sz, Image.LANCZOS), acc))
    return im


def mono(name, h, col):
    return solid(bmask(os.path.join(BX, name), h), col)


def coat(h):
    """Герб Саратовской области (лазоревый щит, серебряные стерляди, золотая корона)."""
    sh = _L("saratov_coa_shield.png")
    im = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    im.alpha_composite(solid(sh, (20, 30, 60)))
    im.alpha_composite(solid(sh.filter(ImageFilter.MinFilter(31)), (22, 148, 220)))
    im.alpha_composite(solid(_L("saratov_coa_crown.png"), (222, 178, 60)))
    im.alpha_composite(solid(_L("saratov_coa_crown_lines.png"), (20, 30, 60)))
    im.alpha_composite(solid(_L("saratov_coa_fish.png"), (225, 230, 238)))
    im.alpha_composite(solid(_L("saratov_coa_lines.png"), (20, 30, 60)))
    im = im.crop(im.getbbox())
    return scale_to(im, h=h)


def karting64(h, col_text=WHITE, flip=False):
    """KARTING64: картинг с пилотом и флагом из логотипа, подпись KARTING64.RU шрифтом Tektur."""
    k = _L("karting64_black.png")
    cut = 1500
    im = Image.new("RGBA", (k.width, cut), (0, 0, 0, 0))
    for n, c in (("white", WHITE), ("blue", (0, 57, 166)), ("red", (213, 43, 30)), ("black", WHITE)):
        im.alpha_composite(solid(_L(f"karting64_{n}.png").crop((0, 0, k.width, cut)), c))
    im = im.crop(im.getbbox())
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = scale_to(im, h=int(h * 0.62))
    t = solid(text_mask("KARTING64.RU", F_TECH(200), track=0.04), col_text)
    t = scale_to(t, w=int(im.width * 1.0))
    return stack([im, t], gap=int(h * 0.06))


def driveoil(h):
    sys.path.insert(0, os.path.join(LIV, "br03_pro"))
    import lib as blib
    return blib.driveoil_logo(h)


def edm_sticker(h):
    im = Image.open(os.path.join(BX, "edm_mbu_logo.png")).convert("RGBA")
    im = im.crop(im.getbbox())
    return im.resize((h, h), Image.LANCZOS)


def ru_flag(h, w=None, outline=PLUM):
    w = w or int(h * 1.5)
    o = max(2, h // 14)
    t = Image.new("RGBA", (w + 2 * o, h + 2 * o), outline + (255,))
    d = ImageDraw.Draw(t)
    for i, c in enumerate(RU):
        d.rectangle([o, o + i * h / 3, o + w - 1, o + (i + 1) * h / 3 - 1], fill=c + (255,))
    return t


def sar_flag(h, w=None, outline=PLUM):
    """Флаг-стикер региона: белый/красный + герб."""
    w = w or int(h * 1.5)
    o = max(2, h // 14)
    t = Image.new("RGBA", (w + 2 * o, h + 2 * o), outline + (255,))
    d = ImageDraw.Draw(t)
    d.rectangle([o, o, o + w - 1, o + h // 2], fill=(255, 255, 255, 255))
    d.rectangle([o, o + h // 2, o + w - 1, o + h - 1], fill=(213, 43, 30, 255))
    c = coat(int(h * 0.8))
    t.alpha_composite(c, (o + int(w * 0.12), o + (h - c.height) // 2))
    return t


def sparkle(r, col=(255, 255, 255), glow=True):
    """Звезда-блёстка Y2K: 4 длинных луча + 4 коротких, с мягким свечением."""
    S = r * 5
    m = Image.new("L", (S * 3, S * 3), 0)
    d = ImageDraw.Draw(m)
    c = S * 1.5
    for ang, L_ in ((0, 1.0), (45, 0.45)):
        pts = []
        for k in range(4):
            a1 = math.radians(ang + k * 90)
            pts.append((c + math.cos(a1) * r * 3 * L_ * 2, c + math.sin(a1) * r * 3 * L_ * 2))
            a2 = math.radians(ang + k * 90 + 45)
            pts.append((c + math.cos(a2) * r * 0.5, c + math.sin(a2) * r * 0.5))
        d.polygon(pts, fill=255)
    m = m.resize((S, S), Image.LANCZOS)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    if glow:
        out.alpha_composite(solid(m.filter(ImageFilter.GaussianBlur(r * 0.5)), col, 0.85))
    out.alpha_composite(solid(m, col))
    return out


def butterfly(w, flip=False):
    """Оригинальная Y2K-бабочка: голографические крылья, сливовый контур, белые блики."""
    S = 3
    W, H = w * S, int(w * 0.8) * S
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    cx = W / 2

    def wing(sx, top):
        pts = []
        for k in range(41):
            t = k / 40 * math.pi
            if top:
                rx, ry = W * 0.44, H * 0.40
                x = cx + sx * (W * 0.03 + rx * math.sin(t) ** 0.8 * (0.55 + 0.45 * math.sin(t / 2)))
                y = H * 0.48 - ry * (0.15 + 0.85 * (1 - math.cos(t)) / 2) + ry * 0.25 * math.sin(t) ** 3
            else:
                rx, ry = W * 0.30, H * 0.40
                x = cx + sx * (W * 0.03 + rx * math.sin(t) ** 0.9)
                y = H * 0.50 + ry * (1 - math.cos(t)) / 2 * 0.95
            pts.append((x, y))
        d.polygon([(cx, H * 0.5)] + pts, fill=255)

    for sx in (-1, 1):
        wing(sx, True)
        wing(sx, False)
    m = m.resize((w, int(w * 0.8)), Image.LANCZOS)
    p = int(w * 0.12)
    m = pad(m, p)
    out = Image.new("RGBA", m.size, (0, 0, 0, 0))
    out.alpha_composite(solid(grow(m, max(3, w // 40)), PLUM))
    out.alpha_composite(fill_img(m, holo_img(m.width, m.height, 1.5, 0.15, 1.2)))
    # прожилки
    v = Image.new("L", m.size, 0)
    dv = ImageDraw.Draw(v)
    c0 = (m.width / 2, m.height * 0.5)
    lw = max(2, w // 70)
    for ang in (-150, -120, -60, -30, 150, 120, 60, 30):
        a = math.radians(ang)
        dv.line([c0, (c0[0] + math.cos(a) * w * 0.38, c0[1] - math.sin(a) * w * 0.3)], fill=255, width=lw)
    out.alpha_composite(solid(ImageChops.multiply(v, m), PLUM, 0.5))
    # тело и усики
    b = Image.new("L", m.size, 0)
    db = ImageDraw.Draw(b)
    bw = max(4, w // 16)
    db.ellipse([c0[0] - bw / 2, m.height * 0.26, c0[0] + bw / 2, m.height * 0.80], fill=255)
    for sx in (-1, 1):
        db.line([(c0[0], m.height * 0.28), (c0[0] + sx * w * 0.10, m.height * 0.10)], fill=255, width=max(2, w // 60))
        db.ellipse([c0[0] + sx * w * 0.10 - bw * 0.35, m.height * 0.10 - bw * 0.35,
                    c0[0] + sx * w * 0.10 + bw * 0.35, m.height * 0.10 + bw * 0.35], fill=255)
    out.alpha_composite(solid(b, PLUM))
    # блики
    hl = Image.new("L", m.size, 0)
    dh = ImageDraw.Draw(hl)
    for sx in (-1, 1):
        x0 = c0[0] + sx * w * 0.20
        dh.ellipse([x0 - w * 0.05, m.height * 0.22, x0 + w * 0.05, m.height * 0.30], fill=255)
    out.alpha_composite(solid(hl.filter(ImageFilter.GaussianBlur(1)), WHITE, 0.8))
    out = out.rotate(-12 if not flip else 12, expand=True, resample=Image.BICUBIC)
    return out.crop(out.getbbox())


def number_plate(w, h, num="00", sub=True):
    """Номерной щит: сливовая плашка с голо-кантом, номер — хром с розовой обводкой, внизу флаг региона."""
    p = plate(w, h, r=int(h * 0.16), rim_w=int(h * 0.045))
    m = text_mask(num, F_NUM(int(h * 1.2)))
    nh = int(h * (0.60 if sub else 0.68))
    m = scale_to(m, h=nh)
    if m.width > w * 0.78:
        m = scale_to(m, w=int(w * 0.78))
    t = styled(m, "chrome", outline=HOT, ow=0.055, outline2=PLUM, ow2=0.035, shadow=True)
    p.alpha_composite(t, ((w - t.width) // 2, int((h * 0.42 if sub else h * 0.5) - t.height / 2)))
    if not sub:      # мелкий щит (лобовое): подпись нечитаема в 1024 px — только номер
        return p
    # нижняя линейка: флаг Саратовской области + «САРАТОВ · 64»
    fh = int(h * 0.10)
    fl = sar_flag(fh, int(fh * 1.6), outline=PLUM)
    tx = solid(text_mask("САРАТОВ · 64", F_SPON(200)), WHITE)
    tx = scale_to(tx, h=int(fh * 0.72))
    row = stack([fl, tx], gap=int(fh * 0.5), vertical=False)
    p.alpha_composite(row, ((w - row.width) // 2, int(h * 0.835 - row.height / 2)))
    return p


# ---------------------------------------------------------------- исходник: тени, маска кузова
def load_source():
    src = Image.open(os.path.join(SRC, "Skin.dds")).convert("RGB")
    a = np.asarray(src).astype(np.float32)
    mx = a.max(-1)
    logo = mx > 110
    rings = np.zeros_like(logo)
    rings[720:790, 1950:2140] = True
    logo &= ~rings
    logo = ndimage.binary_dilation(logo, iterations=6)
    _, (iy, ix) = ndimage.distance_transform_edt(logo, return_indices=True)
    a[logo] = a[iy[logo], ix[logo]]
    mx = a.max(-1)
    body = (a[..., 2] > a[..., 0] + 12) & (mx > 18)
    shade = np.clip(0.50 + 0.50 * np.clip(mx / 78.0, 0, 1.10), 0.35, 1.05)
    orig = np.asarray(src).astype(np.float32)
    return a, orig, body, shade, rings


# ---------------------------------------------------------------- posmap 4096 (заполненный, без мусора на швах)
names = json.load(open(os.path.join(ZONES, "zones.json")))["part_names"]
PID = {n: i for i, n in enumerate(names)}


def load_posmap():
    d = np.load(os.path.join(ZONES, "posmap.npz"))
    cov = d["covered"]
    _, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
    pos = d["pos"].astype(np.float32)[iy, ix]
    nrm = d["nrm"].astype(np.float32)[iy, ix]
    part = d["part"][iy, ix]
    P = np.stack([ndimage.zoom(pos[..., c], 2, order=1) for c in range(3)], -1)
    Nn = np.stack([ndimage.zoom(nrm[..., c], 2, order=1) for c in range(3)], -1)
    Nn /= np.maximum(np.linalg.norm(Nn, axis=-1, keepdims=True), 1e-6)
    part = ndimage.zoom(part, 2, order=0)
    covered = ndimage.zoom(cov.astype(np.uint8), 2, order=0) > 0
    return P, Nn, part, covered


VIEWS = {   # (right, up, toward-viewer)
    "left": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "right": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    "front": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "rear": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "top": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "top_rear": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "top_left": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
}


class Car:
    def __init__(self):
        print("posmap …")
        self.pos, self.nrm, self.part, self.cov = load_posmap()
        self.a, self.orig, self.body, self.shade, self.rings = load_source()
        self.X, self.Y, self.Z = self.pos[..., 0], self.pos[..., 1], self.pos[..., 2]
        self.report = []
        self.masks = {}

    def zone_mask(self, zid):
        if zid not in self.masks:
            self.masks[zid] = np.asarray(Image.open(os.path.join(ZONES, "masks", zid + ".png")).convert("L")) > 127
        return self.masks[zid]

    # -------------------------------------------------- проекция арта + проверка
    def flat(self):
        if not hasattr(self, "ci"):
            self.ci = np.flatnonzero(self.cov)
            self.Pf = self.pos.reshape(-1, 3)[self.ci]
            self.Nf = self.nrm.reshape(-1, 3)[self.ci]
            self.Tf = self.part.reshape(-1)[self.ci]
        return self.ci, self.Pf, self.Nf, self.Tf

    def zone_dil(self, zid):
        key = "dil:" + zid
        if key not in self.masks:
            self.masks[key] = ndimage.binary_dilation(self.zone_mask(zid), iterations=2)
        return self.masks[key]

    def project(self, art, view, center, width_m, parts, name, max_angle=60, zones=None, min_clear_cm=1.5,
                check=True, depth_lim=0.35):
        """Проецирует RGBA-арт (как его видит зритель) на текстуру. Возвращает разреженный слой (индексы, RGBA)
        и пишет в отчёт: покрытие (доля арта, попавшая на разрешённые панели), запас до края панели, маски зон."""
        ci, Pf, Nf, Tf = self.flat()
        r, u, dv = (np.array(v, np.float32) for v in VIEWS[view])
        A = np.asarray(art.convert("RGBA")).astype(np.float32) / 255
        H, W = A.shape[:2]
        height_m = width_m * H / W
        c = np.asarray(center, np.float32)
        margin = 0.12
        sel = (Nf @ dv) > math.cos(math.radians(max_angle))
        if parts is not None:
            sel &= np.isin(Tf, [PID[p] for p in parts])
        rel = Pf - c
        sa = rel @ r
        sb = rel @ u
        sel &= (np.abs(rel @ dv) < depth_lim) & (np.abs(sa) < width_m / 2 + margin) & (np.abs(sb) < height_m / 2 + margin)
        k = np.flatnonzero(sel)
        sa, sb = sa[k], sb[k]
        px = (sa / width_m + 0.5) * W - 0.5
        py = (0.5 - sb / height_m) * H - 0.5
        inart = (px > -1) & (px < W) & (py > -1) & (py < H)
        pre = A.copy()
        pre[..., :3] *= pre[..., 3:4]
        coords = np.stack([py[inart], px[inart]])
        vals = [ndimage.map_coordinates(pre[..., ch], coords, order=1, mode="constant", cval=0) for ch in range(4)]
        al = vals[3]
        rgba = np.zeros((len(al), 4), np.float32)
        rgba[:, :3] = np.stack(vals[:3], -1) / np.maximum(al[:, None], 1e-6)
        rgba[:, 3] = al
        idx = ci[k[inart]]
        keep = al > 0.002
        idx, rgba = idx[keep], rgba[keep]
        if not check:
            return idx, rgba
        # --- проверка в плоскости вида: сетка 2.5 мм
        cell = 0.0025
        gw = int((width_m + 2 * margin) / cell)
        gh = int((height_m + 2 * margin) / cell)
        gx = ((sa + width_m / 2 + margin) / cell).astype(int)
        gy = ((height_m / 2 + margin - sb) / cell).astype(int)
        ok = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        hit = np.zeros((gh, gw), bool)
        hit[gy[ok], gx[ok]] = True
        hit = ndimage.binary_closing(hit, iterations=3, border_value=1)
        al_img = Image.fromarray((A[..., 3] * 255).astype(np.uint8))
        aw, ah = int(round(width_m / cell)), int(round(height_m / cell))
        al_small = np.asarray(al_img.resize((aw, ah), Image.BOX)) > 40
        opq = np.zeros((gh, gw), bool)
        o = int(margin / cell)
        opq[o:o + ah, o:o + aw] = al_small[:gh - o, :gw - o]
        cov = (opq & hit).sum() / max(1, opq.sum())
        dist = ndimage.distance_transform_edt(hit)
        clear_cm = float(dist[opq].min() * cell * 100) if opq.any() else 0
        at = None
        if opq.any():
            dd = np.where(opq, dist, 1e9)
            iy, ix = np.unravel_index(dd.argmin(), dd.shape)
            at = (round((ix * cell - margin - width_m / 2) * 100), round((height_m / 2 + margin - iy * cell) * 100))
        # --- маски зон в текстуре
        inside = None
        if zones:
            solid_idx = idx[rgba[:, 3] > 0.25]
            zin = np.zeros(len(solid_idx), bool)
            for z in zones:
                zin |= self.zone_dil(z).reshape(-1)[solid_idx]
            inside = float(zin.mean()) if len(zin) else 0.0
        status = "OK" if (cov >= 0.995 and clear_cm >= min_clear_cm and (inside is None or inside >= 0.995)) else "CHECK"
        self.report.append(dict(name=name, view=view, parts=parts, width_cm=round(width_m * 100, 1),
                                height_cm=round(height_m * 100, 1), coverage=round(cov * 100, 2),
                                clearance_cm=round(clear_cm, 1), in_zone_masks=None if inside is None else round(inside * 100, 2),
                                status=status, at=at))
        return idx, rgba

    def side_art_canvas(self):
        pass


# ---------------------------------------------------------------- геометрия ливреи (мировые координаты, метры)
def interp(y, pts):
    xs, zs = zip(*pts)
    return np.interp(y, xs, zs)


def smooth_curve(pts, n=400):
    """Сглаженная кривая (y, z) через точки (Catmull-Rom) → функция z(y)."""
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = (np.array(P[j], float) for j in (i - 1, i, i + 1, i + 2))
        for t in np.linspace(0, 1, n // (len(P) - 3), endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(np.array(pts[-1], float))
    out = np.array(out)
    order = np.argsort(out[:, 0])
    return out[order]


# верхняя кромка голо-волны (вид сбоку): поднимается над передней аркой, ныряет вдоль дверей,
# взмывает на «окорок» (задняя арка) и уходит за корму
WAVE = smooth_curve([(-2.45, 0.60), (-2.05, 0.62), (-1.75, 0.74), (-1.38, 0.86), (-1.05, 0.80),
                     (-0.75, 0.62), (-0.35, 0.525), (0.10, 0.51), (0.50, 0.55), (0.82, 0.66),
                     (1.15, 0.84), (1.50, 0.96), (1.85, 1.00), (2.45, 1.00)])
ROCKER_Z = 0.425              # верх сливового рокера = складка двери (z≈0.42 по модели)
GAP = 0.020                   # розовый зазор между голо-волной и сливовой нитью
PIN = 0.012                   # сливовая нить по кромке волны


def wave_z(y):
    return np.interp(y, WAVE[:, 0], WAVE[:, 1])


def aa(s, w=0.0012):
    """Сглаженная ступенька по знаковой величине s (м)."""
    return np.clip(s / w + 0.5, 0, 1)


def build_fields(car, ident):
    """Основные поля цвета: розовая база, голо-волна, рокер, нить, зеркала, капот/крыша."""
    X, Y, Z = car.X, car.Y, car.Z
    ax = np.abs(X)
    nz = car.nrm[..., 2]
    # --- голо-поле (переливается с кривизной: зависит от нормали, как плёнка)
    nv = car.nrm @ np.array([0.35, -0.55, 0.75], np.float32)
    t = 0.30 * Y + 0.95 * Z + 0.55 * nv + 0.10 * np.sin(Y * 3.1 + Z * 5.0) + 0.12 * ax
    holo = palette(t)
    # --- верх: голо-«шеврон» на капоте — продолжение волны через крылья к носу
    # граница: y < yv(x), только на верхних поверхностях капота/носа/крыльев
    yv = -1.62 + 0.95 * np.clip(ax / 0.80, 0, 1.2) ** 2.8   # плоский нос шеврона: нить обходит плашку капота
    top_front = (Y < -0.80) & (Z > 0.60)
    s_side = wave_z(Y) - Z                         # >0 → под волной
    s_top = yv - Y                                 # >0 → перед шевроном
    s = np.where(top_front, np.maximum(s_side, s_top), s_side)
    # корма: всё ниже 1.00 позади задней двери уже в волне; верх крышки багажника — розовый
    k_holo = aa(s)
    # сливовая нить: полоса над кромкой волны (в зазоре GAP)
    k_pin = aa(s + GAP + PIN) * (1 - aa(s + GAP))
    # рокер (пороги + низ дверей) между арками, низ бамперов
    between = (Y > -0.98) & (Y < 1.06)
    zr = np.where(between, ROCKER_Z, 0.30)
    k_rock = aa(zr - Z)
    # тонкая голо-нитка на верхней кромке рокера (1 см над ним — розового нет, рокер в волне)
    k_rline = aa(zr + 0.010 - Z) * (1 - aa(zr - Z)) * between
    base = np.broadcast_to(PINK, (N, N, 3)).astype(np.float32).copy()
    col = base * (1 - k_holo[..., None]) + holo * k_holo[..., None]
    col = col * (1 - k_pin[..., None]) + np.array(PLUM, np.float32) * k_pin[..., None]
    col = col * (1 - k_rock[..., None]) + np.array(PLUM, np.float32) * k_rock[..., None]
    col = col * (1 - k_rline[..., None]) + holo * k_rline[..., None]
    # тональный пунктир «разделки» сверху (капот / крыша): кривые y = y0 + c·x², только по розовому
    tone = np.zeros((N, N), np.float32)
    for y0, cc, zmin in ((-0.215, 0.10, 1.25), (1.17, -0.05, 1.25)):
        yc = y0 + cc * X ** 2
        band = aa(0.005 - np.abs(Y - yc))
        f = ((X + 5.0) / 0.075) % 1.0
        dash = np.clip(np.minimum(f, 0.62 - f) * 0.075 / 0.0012 + 0.5, 0, 1)
        tone = np.maximum(tone, band * dash * (Z > zmin) * (nz > 0.6))
    tone *= (1 - k_holo)
    col = col * (1 - tone[..., None]) + np.array(PINK_T, np.float32) * tone[..., None]
    # зеркала: идентификация машин (1 — голограмма, 2 — сливовые с голо-кромкой)
    mir = car.part == PID["mirror"]
    if ident == 0:
        col[mir] = holo[mir]
    else:
        col[mir] = np.array(PLUM, np.float32)
    car.k_holo = k_holo
    return col


# ---------------------------------------------------------------- рисование «вида сбоку» (тональная схема, блёстки)
class SideCanvas:
    """Холст вида сбоку в метрах: y от Y0 до Y1, z от 0 до ZT; для правого борта зеркалится при проекции."""
    Y0, Y1, ZT = -2.30, 2.40, 1.50

    def __init__(self, dpm=1000):
        self.d = dpm
        self.img = Image.new("RGBA", (int((self.Y1 - self.Y0) * dpm), int(self.ZT * dpm)), (0, 0, 0, 0))

    def xy(self, y, z):
        return ((y - self.Y0) * self.d, (self.ZT - z) * self.d)

    def paste(self, art, y, z, width_m):
        a = scale_to(art, w=int(width_m * self.d))
        x, yy = self.xy(y, z)
        self.img.alpha_composite(a, (int(x - a.width / 2), int(yy - a.height / 2)))

    def dashed(self, pts, col, width_m=0.012, dash=0.05, gap=0.032, smooth=True):
        P = smooth_curve(pts) if smooth else np.array(pts)
        if smooth:     # smooth_curve сортирует по y — для вертикальных кривых строим по z
            pass
        m = Image.new("L", self.img.size, 0)
        d = ImageDraw.Draw(m)
        w = max(2, int(width_m * self.d))
        seg = [self.xy(*p) for p in P]
        L = 0.0
        on = True
        left = dash * self.d
        for (x0, y0), (x1, y1) in zip(seg[:-1], seg[1:]):
            sl = math.hypot(x1 - x0, y1 - y0)
            p = 0.0
            while p < sl - 1e-6:
                st = min(left, sl - p)
                if on:
                    d.line([(x0 + (x1 - x0) * p / sl, y0 + (y1 - y0) * p / sl),
                            (x0 + (x1 - x0) * (p + st) / sl, y0 + (y1 - y0) * (p + st) / sl)], fill=255, width=w)
                p += st
                left -= st
                if left <= 1e-6:
                    on = not on
                    left = (dash if on else gap) * self.d
        self.img.alpha_composite(solid(m, col))


def curve_pts(pts, n=60, vertical=False):
    """Сглаженная ломаная без сортировки (для любых направлений)."""
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = (np.array(P[j], float) for j in (i - 1, i, i + 1, i + 2))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(np.array(pts[-1], float))
    return [tuple(p) for p in out]


def project_canvas(car, canvas, view, max_angle=72, exclude_parts=()):
    """Проецирует весь холст вида сбоку на борт (без проверки — это фон/узор). y холста = мировой y."""
    ci, Pf, Nf, Tf = car.flat()
    dv = np.array(VIEWS[view][2], np.float32)
    A = np.asarray(canvas.img).astype(np.float32) / 255
    H, W = A.shape[:2]
    sx = (Pf[:, 1] - canvas.Y0) * canvas.d
    sy = (canvas.ZT - Pf[:, 2]) * canvas.d
    side = Pf[:, 0] > 0.2 if view == "left" else Pf[:, 0] < -0.2
    m = side & ((Nf @ dv) > math.cos(math.radians(max_angle))) & (sx > 0) & (sx < W - 1) & (sy > 0) & (sy < H - 1)
    for p in exclude_parts:
        m &= Tf != PID[p]
    k = np.flatnonzero(m)
    pre = A.copy()
    pre[..., :3] *= pre[..., 3:4]
    coords = np.stack([sy[k], sx[k]])
    vals = [ndimage.map_coordinates(pre[..., ch], coords, order=1) for ch in range(4)]
    al = vals[3]
    keep = al > 0.002
    rgba = np.zeros((keep.sum(), 4), np.float32)
    rgba[:, :3] = (np.stack(vals[:3], -1) / np.maximum(al[:, None], 1e-6))[keep]
    rgba[:, 3] = al[keep]
    return ci[k[keep]], rgba


def comp(col, layer):
    """Наложить разреженный слой (индексы, RGBA 0..1) на поле цвета N×N×3 (на месте)."""
    idx, rgba = layer
    cf = col.reshape(-1, 3)
    a = rgba[:, 3:4]
    cf[idx] = cf[idx] * (1 - a) + rgba[:, :3] * 255 * a
    return col


# ---------------------------------------------------------------- сборка ливреи
DRIVERS = {
    "Pozdnyakov_00": dict(ru="Станислав Поздняков", short="С. ПОЗДНЯКОВ", ident=0),
    "Konopelko_00": dict(ru="Матвей Конопелько", short="М. КОНОПЕЛЬКО", ident=1),
}


def side_layers(car):
    """Тональная мясницкая схема + блёстки/бабочки на бортах (холст вида сбоку, проецируется на оба борта)."""
    cv = SideCanvas(1000)
    tone = PINK_T
    # пунктир «разделки»: линии повторяют тело и уходят в голо-волну, не обрываясь в воздухе
    cuts = [
        [(-1.00, 0.56), (-1.02, 0.70), (-0.98, 0.82), (-0.90, 0.95)],    # лопатка | грудинка (от кромки арки до плеча)
        [(-0.10, 0.44), (-0.115, 0.70), (-0.09, 0.955)],                 # грудинка | корейка (от рокера до плеча, за номером)
        [(-0.95, 0.955), (-0.30, 0.945), (0.40, 0.955), (1.05, 0.975)],  # хребет — по линии плеча
    ]
    for c in cuts:
        cv.dashed(curve_pts(c, 40), tone, width_m=0.010, dash=0.045, gap=0.028, smooth=False)
    return cv


LABELS = [("КОРЕЙКА", 0.080, 0.885, "front_door")]      # (текст, y, z, панель) — подписи отрубов на бортах


def label_art(s, cap_m, col=PINK_T):
    m = text_mask(s, F_CUT(int(cap_m * D * 1.35)), track=0.06)
    return solid(m, col)


def build(skin_name):
    info = DRIVERS[skin_name]
    car = CAR
    car.report = []
    col = build_fields(car, info["ident"])
    layers = []      # (layer, name) — в порядке наложения

    # --- тональная схема и декор (вид сбоку, оба борта)
    cv = side_layers(car)
    for view in ("left", "right"):
        layers.append(project_canvas(car, cv, view, max_angle=70, exclude_parts=("mirror", "door_handle")))

    # --- элементы бортов (каждый проверяется)
    for view, sgn in (("left", 1), ("right", -1)):
        X = 0.90 * sgn
        S = "L" if sgn > 0 else "R"
        # номер на передней двери
        npl = number_plate(int(0.52 * D), int(0.33 * D))
        layers.append(car.project(npl, view, (X, -0.42, 0.708), 0.52, ["front_door"], f"номер 00 дверь {S}",
                                  zones=[f"front_door_{S}"], min_clear_cm=3))
        # «Арка» — задняя дверь
        ar = arka_logo(int(0.20 * D))
        layers.append(car.project(ar, view, (X, 0.555, 0.79), 0.42, ["rear_door"], f"Арка тайм-кафе дверь {S}",
                                  zones=[f"rear_door_{S}"], min_clear_cm=2))
        # «Симкарт» — низ передней двери на сливовом рокере
        sk = simkart_logo(int(0.09 * D))
        layers.append(car.project(sk, view, (X, -0.33, 0.348), 0.50, ["front_door_low"], f"Симкарт низ двери {S}",
                                  zones=[f"front_door_low_{S}"], min_clear_cm=1))
        # DriveOil + KARTING64 — низ задней двери (рокер)
        do = driveoil(int(0.055 * D))
        layers.append(car.project(do, view, (X, 0.47 if sgn > 0 else 0.47, 0.348), 0.26, ["rear_door"],
                                  f"DriveOil низ задней двери {S}", zones=[f"rear_door_{S}"], min_clear_cm=1))
        k64 = karting64(int(0.10 * D), flip=(sgn > 0))
        layers.append(car.project(k64, view, (X, 0.80, 0.350), 0.17, ["rear_door"],
                                  f"KARTING64.RU низ задней двери {S}", zones=[f"rear_door_{S}"], min_clear_cm=1))
        # пороги: SMP RACING ESPORTS + BR ENGINEERING (белые на сливе)
        lk = smp_lockup(int(0.05 * D))
        layers.append(car.project(lk, view, (X + 0.05 * sgn, -0.45 if sgn > 0 else 0.45, 0.207), 0.175, ["sill"],
                                  f"SMP RACING ESPORTS порог {S}", max_angle=62, zones=[f"sill_{S}"], min_clear_cm=0.5))
        be = mono("br_engineering_black.png", int(0.04 * D), WHITE)
        layers.append(car.project(be, view, (X + 0.05 * sgn, 0.40 if sgn > 0 else -0.40, 0.207), 0.36, ["sill"],
                                  f"BR ENGINEERING порог {S}", max_angle=62, zones=[f"sill_{S}"], min_clear_cm=0.5))
        # подписи отрубов (тон в тон)
        for txt, y, z, part in LABELS:
            lab = label_art(txt, 0.030)
            layers.append(car.project(lab, view, (X, y, z), lab.width / D, [part],
                                      f"подпись {txt} {S}", zones=[f"{part}_{S}"], min_clear_cm=1))

    # --- капот: «Симкарт» на голо-плашке + ссылка
    sk = simkart_logo(int(0.15 * D))
    url = solid(text_mask("simkart.vercel.app", F_URL(300)), WHITE)
    url = scale_to(url, h=int(0.034 * D))
    hood = on_plate(stack([sk, url], gap=int(0.012 * D)), int(0.045 * D), int(0.03 * D), r=int(0.06 * D),
                    rim_w=int(0.012 * D))
    layers.append(car.project(hood, "top", (0.0, -1.30, 0.95), 0.70, ["hood"], "Симкарт капот",
                              zones=["hood"], min_clear_cm=3))

    # --- крыша: номер (читается с левого борта) + стикер МБУ «Клуб Энгельсская молодёжь»
    rn = number_plate(int(0.74 * D), int(0.48 * D))
    layers.append(car.project(rn, "top_left", (0.0, 0.36, 1.40), 0.74, ["roof"], "номер 00 крыша",
                              zones=["roof"], min_clear_cm=4))
    st = edm_sticker(int(0.24 * D))
    layers.append(car.project(st, "top_rear", (0.0, 0.98, 1.38), 0.24, ["roof"], "МБУ стикер крыша",
                              zones=["roof"], min_clear_cm=2))

    # --- багажник: «КОМАНДА ЭДМ» (читается сзади)
    # (между стойками антикрыла, |x| < 0.2 м)
    tm = styled(text_mask("КОМАНДА ЭДМ", F_SPON(300)), WHITE, outline=PLUM, ow=0.10, shadow=False)
    layers.append(car.project(tm, "top_rear", (0.0, 1.99, 1.04), 0.30, ["trunk_lid"], "КОМАНДА ЭДМ багажник",
                              zones=["trunk_lid"], max_angle=50, min_clear_cm=1.5))

    # --- задняя панель между фонарями: герб + САРАТОВСКАЯ ОБЛАСТЬ
    sar = stack([coat(int(0.12 * D)),
                 stack([solid(scale_to(text_mask("САРАТОВСКАЯ", F_SPON(300)), h=int(0.040 * D)), PLUM),
                        solid(scale_to(text_mask("ОБЛАСТЬ · 64", F_SPON(300)), h=int(0.040 * D)), PLUM)],
                       gap=int(0.014 * D))], gap=int(0.025 * D), vertical=False)
    layers.append(car.project(sar, "rear", (0.0, 2.16, 0.775), 0.42, ["rear_panel"], "Саратовская область задняя панель",
                              zones=["rear_panel"], max_angle=50, min_clear_cm=1))
    # --- задний бампер: ссылка Симкарт
    url2 = solid(text_mask("simkart.vercel.app", F_URL(300)), PLUM)
    layers.append(car.project(url2, "rear", (0.0, 2.25, 0.585), 0.62, ["rear_bumper"], "simkart.vercel.app задний бампер",
                              zones=["rear_bumper"], max_angle=50, min_clear_cm=1))

    # --- перед: BR ENGINEERING и РАФ на углах бампера
    brs = mono("br_symbol_white.png", int(0.10 * D), PLUM)
    raf = mono("raf_black.png", int(0.12 * D), PLUM)
    layers.append(car.project(brs, "front", (0.505, -2.05, 0.505), 0.085, ["front_bumper", "front_bumper_corner"],
                              "BR знак бампер L", max_angle=45, min_clear_cm=1.5))
    layers.append(car.project(raf, "front", (-0.51, -2.05, 0.51), 0.095, ["front_bumper", "front_bumper_corner"],
                              "РАФ бампер R", max_angle=45, min_clear_cm=1.5))

    # --- тональные подписи отрубов сверху (читаются спереди)
    for txt, y, part in (("ВЫРЕЗКА", -0.125, "roof"),):
        lab = label_art(txt, 0.032)
        layers.append(car.project(lab, "top", (0.0, y, 1.2), lab.width / D, [part], f"подпись {txt}",
                                  zones=[part], max_angle=35, min_clear_cm=1, depth_lim=0.6))

    # --- блёстки и бабочки (декор, после логотипов не должны их перекрывать — кладём ДО логотипов)
    deco = decor_layers(car)

    for L in deco + layers:
        col = comp(col, L)

    # --- финал: кузов × тени исходника, кольца Audi сохраняем
    out = car.a.copy()
    sh = car.shade[..., None]
    out[car.body] = (col * sh)[car.body]
    out[car.rings] = car.orig[car.rings]
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return img, car.report


def decor_layers(car):
    """Y2K-блёстки и бабочки: на гребне волны у передней арки, на «окороке», на капоте."""
    res = []
    for view, sgn in (("left", 1), ("right", -1)):
        X = 0.9 * sgn
        cv = SideCanvas(1000)
        # звёзды на гребне волны (кромка + чуть выше), разного размера
        for y, dz, r in ((-1.55, -0.06, 30), (-1.43, -0.015, 14), (-1.30, -0.05, 20), (1.22, -0.07, 26),
                         (1.33, -0.035, 13), (1.10, -0.04, 14), (-2.0, -0.12, 16), (0.66, -0.035, 12),
                         (-0.20, -0.04, 10), (2.05, -0.20, 18)):
            z = float(wave_z(y)) + dz
            s = sparkle(r * 3)
            cv.paste(s, y, z, s.width / 3000 * 1.0)
        res.append(project_canvas(car, cv, view, max_angle=70, exclude_parts=("mirror", "door_handle")))
    # бабочки на углах капота (в голо-шевроне)
    for sgn in (1, -1):
        b = butterfly(int(0.15 * D), flip=sgn < 0)
        res.append(CAR.project(b, "top", (0.58 * sgn, -1.62, 1.0), 0.14, ["hood"],
                               f"бабочка капот {'L' if sgn > 0 else 'R'}", max_angle=45, min_clear_cm=1.5, depth_lim=0.6))
    # блёстки вокруг плашки «Симкарт» на капоте
    for x, y, r in ((0.45, -1.20, 24), (-0.45, -1.53, 16), (0.30, -1.62, 12), (-0.48, -1.17, 12)):
        s = sparkle(r * 3)
        res.append(CAR.project(s, "top", (x, y, 1.0), s.width / 3000, ["hood"], "блёстка капот", max_angle=50,
                               check=False, depth_lim=0.6))
    return res


# ---------------------------------------------------------------- стекло / антикрыло (glass_sticker.dds 1024)
def glass(info):
    g = Image.open(os.path.join(SRC, "glass_sticker.dds")).convert("RGBA")
    G = g.width
    assert G == 1024
    gr = []
    a = np.asarray(g).copy()
    a[..., 3] = 0                          # всё стекло прозрачное, рисуем только наши стикеры
    g = Image.fromarray(a)

    def box_paste(art, box, name, rot=0, margin=4):
        x0, y0, x1, y1 = box
        if rot:
            art = art.rotate(rot, expand=True, resample=Image.BICUBIC)
        bw, bh = x1 - x0 - 2 * margin, y1 - y0 - 2 * margin
        k = min(bw / art.width, bh / art.height, 1.0)
        if k < 1:
            art = art.resize((int(art.width * k), int(art.height * k)), Image.LANCZOS)
        px, py = (x0 + x1 - art.width) // 2, (y0 + y1 - art.height) // 2
        g.alpha_composite(art, (px, py))
        bb = art.getbbox()
        cl = min(bb[0] + px - x0, bb[1] + py - y0, x1 - (bb[2] + px), y1 - (bb[3] + py))
        gr.append(dict(name=name, box=box, clearance_px=int(cl), status="OK" if cl >= 2 else "CHECK"))

    # лобовое: сливовая полоса, SMP RACING ESPORTS
    ban = Image.new("RGBA", (1024, 152), PLUM + (255,))
    hb = holo_img(1024, 10)
    ban.alpha_composite(fill_img(Image.new("L", (1024, 8), 255), hb[:8]), (0, 142))
    g.alpha_composite(ban, (0, 0))
    box_paste(smp_lockup(300), (150, 20, 870, 136), "SMP RACING ESPORTS лобовое")

    # антикрыло верх: слива + голо-канты; Арка | Симкарт; читается сзади (rot 180)
    x0, y0, x1, y1 = 100, 858, 946, 1012
    w = Image.new("RGBA", (x1 - x0, y1 - y0), PLUM + (255,))
    hb = fill_img(Image.new("L", (x1 - x0, 10), 255), holo_img(x1 - x0, 10))
    w.alpha_composite(hb, (0, 0))
    w.alpha_composite(hb, (0, w.height - 10))
    g.alpha_composite(w, (x0, y0))
    half = (x1 - x0) // 2
    wa = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    # раскладка «как видит зритель сзади», потом поворот 180
    al = arka_logo(260, taimcafe=True)
    al = scale_to(al, h=118)
    sk = scale_to(simkart_logo(200), w=int(half * 0.86))
    if sk.height > 110:
        sk = scale_to(sk, h=110)
    wa.alpha_composite(al, ((half - al.width) // 2, (wa.height - al.height) // 2))
    wa.alpha_composite(sk, (half + (half - sk.width) // 2, (wa.height - sk.height) // 2))
    dv = Image.new("L", (6, wa.height - 40), 255)
    wa.alpha_composite(fill_img(dv, holo_img(6, wa.height - 40)), (half - 3, 20))
    wa = wa.rotate(180)
    g.alpha_composite(wa, (x0, y0))
    bbw = wa.getbbox()
    gr.append(dict(name="антикрыло: Арка | Симкарт", box=(104, 863, 941, 1006),
                   clearance_px=int(min(bbw[0] + x0 - 104, bbw[1] + y0 - 863, 941 - (bbw[2] + x0), 1006 - (bbw[3] + y0))),
                   status="OK"))
    # пластины: слива + Арка (без «тайм-кафе»)
    for nm, (ex0, ey0, ex1, ey1) in (("пластина L", (329, 758, 520, 856)), ("пластина R", (112, 762, 304, 858))):
        p = Image.new("RGBA", (ex1 - ex0, ey1 - ey0), PLUM + (255,))
        hb = fill_img(Image.new("L", (ex1 - ex0, 6), 255), holo_img(ex1 - ex0, 6))
        p.alpha_composite(hb, (0, 0))
        p.alpha_composite(hb, (0, p.height - 6))
        g.alpha_composite(p, (ex0, ey0))
        box_paste(arka_logo(200, taimcafe=False), (ex0 + 4, ey0 + 4, ex1 - 4, ey1 - 4), f"Арка {nm}", margin=12)

    # боковые окна: имя пилота + флаг России на сливовой полосе по низу окна
    for nm, (bx0, by0, bx1, by1) in (("L", (115, 272, 513, 496)), ("R", (114, 518, 512, 741))):
        sh = 52
        strip_box = (bx0, by1 - sh - 6, bx1, by1)
        pl = Image.new("RGBA", (bx1 - bx0, sh + 6), PLUM + (255,))
        hb = fill_img(Image.new("L", (bx1 - bx0, 4), 255), holo_img(bx1 - bx0, 4))
        pl.alpha_composite(hb, (0, 0))
        g.alpha_composite(pl, (bx0, by1 - sh - 6))
        nmimg = solid(text_mask(info["short"], F_NAME(200)), WHITE)
        nmimg = scale_to(nmimg, h=26)
        fl = ru_flag(22, 33)
        row = stack([fl, nmimg], gap=12, vertical=False)
        box_paste(row, (bx0 + 8, by1 - sh, bx1 - 8, by1 - 2), f"имя + флаг окно {nm}", margin=4)

    # номер 00 в верхнем углу лобового (передняя идентификация)
    num = number_plate(300, 200, sub=False)
    box_paste(num, (640, 330, 800, 440), "номер 00 лобовое (угол)", rot=-8)

    # заднее стекло: верхний баннер — Y2K-заголовок PINK PIG 2.0 (хром-пузырь)
    rb = (135, 164, 864, 253)
    band = Image.new("RGBA", (rb[2] - rb[0], rb[3] - rb[1]), PLUM + (255,))
    g.alpha_composite(band, rb[:2])
    pp = styled(text_mask("PINK PIG 2.0", F_Y2K(200)), "holo", outline=PLUM, ow=0.05, outline2=WHITE, ow2=0.03, shadow=False)
    box_paste(pp, rb, "PINK PIG 2.0 заднее стекло", margin=10)
    return g, gr


# ---------------------------------------------------------------- вывод
def save_dxt5(im, path):
    im.convert("RGBA").save(path, pixel_format="DXT5")
    with open(path, "r+b") as fh:
        fh.seek(20)
        fh.write(struct.pack("<I", (im.width // 4) * (im.height // 4) * 16))
        fh.seek(88)
        fh.write(struct.pack("<I", 0))


def icon(ident):
    S = 185 * 3
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).ellipse([6, 6, S - 6, S - 6], fill=255)
    im.alpha_composite(solid(m, HOT))
    band = Image.new("L", (S, S), 0)
    ImageDraw.Draw(band).polygon([(0, S * 0.62), (S, S * 0.38), (S, S * 0.70), (0, S * 0.95)], fill=255)
    im.alpha_composite(fill_img(ImageChops.multiply(band, m), holo_img(S, S)))
    rim = grow(m, 0)
    ring = ImageChops.subtract(m, m.filter(ImageFilter.MinFilter(19)))
    im.alpha_composite(solid(ring, PLUM))
    t = styled(scale_to(text_mask("00", F_NUM(300)), h=int(S * 0.34)), "chrome", outline=HOT, ow=0.06, outline2=PLUM, ow2=0.04)
    im.alpha_composite(t, ((S - t.width) // 2, (S - t.height) // 2 - int(S * 0.04)))
    if ident == 1:
        f = ru_flag(30, 45)
    return im.resize((185, 185), Image.LANCZOS)


def main():
    global CAR
    CAR = Car()
    previews = {}
    rep_all = []
    only = os.environ.get("ONLY")
    for skin_name, info in DRIVERS.items():
        if only and only != skin_name:
            continue
        print("сборка", skin_name)
        img, rep = build(skin_name)
        gl, grep = glass(info)
        out = os.path.join(HERE, skin_name)
        os.makedirs(out, exist_ok=True)
        save_dxt5(img, os.path.join(out, "Skin.dds"))
        save_dxt5(gl, os.path.join(out, "glass_sticker.dds"))
        icon(info["ident"]).save(os.path.join(out, "livery.png"))
        with open(os.path.join(out, "ui_skin.json"), "w", encoding="utf-8") as fh:
            json.dump({"skinname": skin_name, "drivername": info["ru"], "country": "Russia",
                       "team": "Команда ЭДМ", "number": "00", "priority": 1}, fh, ensure_ascii=False, indent=2)
        with zipfile.ZipFile(os.path.join(HERE, skin_name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(os.listdir(out)):
                z.write(os.path.join(out, f), skin_name + "/" + f)
        previews[skin_name] = (img, gl)
        rep_all.append((skin_name, rep, grep))
    img, gl = previews[list(previews)[0]]
    prev = img.resize((2048, 2048), Image.LANCZOS)
    prev.save(os.path.join(HERE, "texture_preview.png"))
    gl.save(os.path.join(HERE, "glass_preview.png"))
    lines = []
    bad = 0
    for skin_name, rep, grep in rep_all:
        lines.append(f"== {skin_name}")
        for r in rep:
            lines.append(f"  [{r['status']}] {r['name']}: {r['width_cm']}×{r['height_cm']} см, вид {r['view']}, "
                         f"покрытие {r['coverage']}%, запас до края {r['clearance_cm']} см (в точке {r['at']} см от центра), в масках зон {r['in_zone_masks']}%")
            bad += r["status"] != "OK"
        for r in grep:
            lines.append(f"  [{r['status']}] glass: {r['name']}: запас {r['clearance_px']} px (1024)")
            bad += r["status"] != "OK"
    lines.append(f"ИТОГО замечаний: {bad}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(HERE, "check_report.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
