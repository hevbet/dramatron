"""Концепция 3 «POP PIG» — Audi RS3 LMS (smp_audi_rs3_lms). Pink Pig 2.0 × поп-арт / комикс Y2K.

Идея: бабл-гам розовый кузов, «брюхо» и «окорок» второго цвета (кислотный лайм / электрик-циан у второй машины),
жирная чёрная комикс-обводка по линии кузова (порог → подъём к заднему крылу), белый блик-контур, точки Бен-Дэй
(полутон) вдоль линии, чёрные пунктиры «мясницкой схемы» поперёк кузова, подписи отрубов в комикс-пузырях,
оригинальный маскот-свин в гоночных очках (крыша, антикрыло).

Все поля цвета считаются по 3D-позиции каждого текселя (карта текстель→мир, geom.py) — линии непрерывны через
швы развёртки и щели дверей. Логотипы и надписи рисуются «как их видит зритель» (вид сбоку/сверху/спереди/сзади)
и проецируются — текст параллелен земле, пропорции сохранены. Для каждого элемента автоматически проверяется,
что он целиком лежит на своей панели (покрытие 100 %) и запас до края (см) ≥ порога → check_report.txt.

Запуск: python3 make_skin.py [папка исходного скина SMP01]
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
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geom  # noqa: E402

LIV = os.path.abspath(os.path.join(HERE, "..", ".."))
SCR = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad"
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCR, "rs3", "SMP01")
KN5 = os.path.join(SCR, "kn5", "smp_audi_rs3_lms.kn5")
FONTS = os.path.join(LIV, "fonts")
BRAND = os.path.join(LIV, "br03", "brand")
BX = os.path.join(LIV, "br03_pro", "brand_extra")
N = 4096
PPM = 1400                      # px арта на метр (текстура ≈ 795 px/м → ×1.75 запас)
MIN_CLEAR_CM = 1.5              # минимальный запас от края панели
FITTER = None

# ---------------------------------------------------------------- палитра
PINK = (255, 100, 180)          # бабл-гам
PINK_D = (236, 62, 150)         # горячий розовый: точки Бен-Дэй, пятачок
INK = (18, 14, 22)              # комикс-чёрный
WHITE = (255, 255, 255)
LIME = (196, 255, 20)
CYAN = (0, 214, 255)
YEL = (240, 209, 102)           # фирменный жёлтый «Арки»
RU = [(255, 255, 255), (0, 57, 166), (213, 43, 30)]


# ---------------------------------------------------------------- шрифты (выбор концепции)
def font(name, size, var=None, axes=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), max(6, int(size)))
    if var:
        f.set_variation_by_name(var)
    if axes:
        f.set_variation_by_axes(axes)
    return f


F_NUM = lambda s: font("Unbounded[wght].ttf", s, "Black")                 # номера
F_NAME = lambda s: font("Exo2-Italic[wght].ttf", s, "Black Italic")        # имена пилотов
F_SPON = lambda s: font("MontserratAlternates-BlackItalic.ttf", s)         # команда / регион
F_URL = lambda s: font("Exo2-Italic[wght].ttf", s, "ExtraBold Italic")     # ссылки
F_TECH = lambda s: font("Tektur[wdth,wght].ttf", s, axes=[100, 800])       # KARTING64.RU
F_CUT = lambda s: font("Podkova[wght].ttf", s, "ExtraBold")                # подписи отрубов
F_POP = lambda s: font("RubikBubbles-Regular.ttf", s)                      # Y2K-заголовок «ХРЮ!»


# ---------------------------------------------------------------- базовые слои
def solid(mask_l, col, alpha=1.0):
    t = Image.new("RGBA", mask_l.size, tuple(col[:3]) + (255,))
    t.putalpha(mask_l if alpha == 1.0 else mask_l.point(lambda v: int(v * alpha)))
    return t


def text_mask(s, f, track=0.0):
    if not track:
        bb = f.getbbox(s)
        m = Image.new("L", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), 0)
        ImageDraw.Draw(m).text((4 - bb[0], 4 - bb[1]), s, font=f, fill=255)
        return m.crop(m.getbbox())
    gap = f.size * track
    w = int(sum(f.getlength(c) for c in s) + gap * (len(s) - 1)) + 20
    asc, desc = f.getmetrics()
    m = Image.new("L", (w, asc + desc + 20), 0)
    d = ImageDraw.Draw(m)
    x = 10
    for c in s:
        d.text((x, 10), c, font=f, fill=255)
        x += f.getlength(c) + gap
    return m.crop(m.getbbox())


def text_h(s, f_maker, h, track=0.0):
    """Маска текста с высотой заглавных ≈ h px (кегль подбирается)."""
    probe = text_mask(s, f_maker(200), track)
    size = 200 * h / probe.height
    return text_mask(s, f_maker(size), track)


def grow(mask_l, r):
    if r <= 0:
        return mask_l
    a = np.asarray(mask_l) > 127
    dt = ndimage.distance_transform_edt(~a)
    return Image.fromarray((np.clip(r + 0.5 - dt, 0, 1) * 255).astype(np.uint8))


def pad(mask_l, p):
    out = Image.new("L", (mask_l.width + 2 * p, mask_l.height + 2 * p), 0)
    out.paste(mask_l, (p, p))
    return out


def comic(mask_l, fill, ow, outline=INK, shadow=None, sh=(0, 0)):
    """Заливка + жирный комикс-контур + жёсткая смещённая тень (поп-арт)."""
    p = ow + max(abs(sh[0]), abs(sh[1])) + 4
    m = pad(mask_l, p)
    ol = grow(m, ow)
    t = Image.new("RGBA", m.size, (0, 0, 0, 0))
    if shadow is not None:
        t.alpha_composite(solid(ImageChops.offset(ol, *sh), shadow))
    t.alpha_composite(solid(ol, outline))
    if isinstance(fill, Image.Image):
        f = fill.resize(m.size).convert("RGBA")
        f.putalpha(m)
        t.alpha_composite(f)
    else:
        t.alpha_composite(solid(m, fill))
    return t.crop(t.getbbox())


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
    w = sum(i.width for i in items) + gap * (len(items) - 1)
    h = max(i.height for i in items)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for i in items:
        out.alpha_composite(i, (x, (h - i.height) // 2))
        x += i.width + gap
    return out


def rrect(w, h, r, fill, outline=None, ow=0):
    t = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    ImageDraw.Draw(t).rounded_rectangle([ow // 2, ow // 2, w - 1 - ow // 2, h - 1 - ow // 2], radius=r,
                                        fill=fill + (255,) if fill else None,
                                        outline=(outline + (255,)) if outline else None, width=ow)
    return t


def panel(content, padx, pady, fill=INK, key=WHITE, kw=0, shadow=None, sh=0, r=None):
    """Комикс-плашка: заливка, тонкая обводка-«кант», жёсткая тень цвета акцента вниз-вправо."""
    w, h = content.width + 2 * padx, content.height + 2 * pady
    r = int(h * 0.18) if r is None else r
    out = Image.new("RGBA", (w + sh, h + sh), (0, 0, 0, 0))
    if shadow is not None and sh:
        out.alpha_composite(rrect(w, h, r, shadow), (sh, sh))
    out.alpha_composite(rrect(w, h, r, fill, key if kw else None, kw), (0, 0))
    out.alpha_composite(content, (padx, pady))
    return out


def brand_mask(name, h, folder=BRAND):
    m = Image.open(os.path.join(folder, name)).convert("L")
    m = m.crop(m.getbbox())
    return m.resize((max(1, round(m.width * h / m.height)), int(h)), Image.LANCZOS)


# ---------------------------------------------------------------- бренды
def arka_logo(h, taimcafe=True):
    """«арка»: фирменный жёлтый, тёмный контур и тень вниз-влево (как на афише)."""
    m = brand_mask("arka_word.png", h)
    t = comic(m, YEL, max(3, h // 20), INK, INK, (-max(2, h // 26), max(3, h // 14)))
    if not taimcafe:
        return t
    s = brand_mask("arka_taimcafe.png", max(10, int(h * 0.24)))
    ts = comic(s, YEL, max(2, s.height // 9), INK, INK, (-max(1, s.height // 12), max(2, s.height // 8)))
    return stack([t, ts], int(h * 0.04))


def simkart_logo(h, glow=True, tagline=False):
    """«Симкарт»: «Сим» серебро с градиентом, «карт» красный, красное свечение (для тёмного фона)."""
    red = (234, 50, 38)
    wm = Image.open(os.path.join(BRAND, "simkart_word_white.png")).convert("L")
    rm = Image.open(os.path.join(BRAND, "simkart_word_red.png")).convert("L")
    bb = ImageChops.lighter(wm, rm).getbbox()
    sz = (round((bb[2] - bb[0]) * h / (bb[3] - bb[1])), int(h))
    w = wm.crop(bb).resize(sz, Image.LANCZOS)
    r = rm.crop(bb).resize(sz, Image.LANCZOS)
    p = int(h * 0.3)
    size = (w.width + 2 * p, w.height + 2 * p)
    t = Image.new("RGBA", size, (0, 0, 0, 0))
    if glow:
        both = Image.new("L", size, 0)
        both.paste(ImageChops.lighter(w, r), (p, p))
        t.alpha_composite(solid(both.filter(ImageFilter.GaussianBlur(h * 0.12)), red, 0.65))
    grad = Image.linear_gradient("L").resize(w.size)
    sil = Image.merge("RGB", [grad.point(lambda v, c=c: int(255 - (255 - c) * v / 255)) for c in (205, 205, 216)]).convert("RGBA")
    sil.putalpha(w)
    t.alpha_composite(sil, (p, p))
    t.alpha_composite(solid(r, red), (p, p))
    t = t.crop(t.getbbox())
    if not tagline:
        return t
    tg = solid(brand_mask("simkart_tagline.png", max(8, int(h * 0.15))), red)
    return stack([t, tg], 0)


def smp_lockup(h, col=WHITE, acc=None):
    """SMP RACING / ESPORTS (белая часть + акцентная)."""
    wm = Image.open(os.path.join(BX, "smp_racing_esports_white.png")).convert("L")
    am = Image.open(os.path.join(BX, "smp_racing_esports_accent.png")).convert("L")
    bb = ImageChops.lighter(wm, am).getbbox()
    wm, am = wm.crop(bb), am.crop(bb)
    k = h / wm.height
    sz = (max(1, round(wm.width * k)), int(h))
    t = solid(wm.resize(sz, Image.LANCZOS), col)
    t.alpha_composite(solid(am.resize(sz, Image.LANCZOS), acc or col))
    return t


def br_logo(h, col=INK):
    return solid(brand_mask("br_engineering_black.png", h, BX), col)


def raf_logo(h, col=INK):
    return solid(brand_mask("raf_black.png", h, BX), col)


def mbu_logo(h):
    m = Image.open(os.path.join(BX, "edm_mbu_logo.png")).convert("RGBA")
    m = m.crop(m.getbbox())
    return m.resize((round(m.width * h / m.height), int(h)), Image.LANCZOS)


def saratov_coa(h):
    """Герб Саратовской области: лазоревый щит, три серебряные стерляди, золотая корона."""
    lay = {}
    for k in ("shield", "fish", "lines", "crown", "crown_lines"):
        lay[k] = Image.open(os.path.join(BX, f"saratov_coa_{k}.png")).convert("L")
    bb = ImageChops.lighter(lay["shield"], lay["crown"]).getbbox()
    k = h / (bb[3] - bb[1])
    sz = (max(1, round((bb[2] - bb[0]) * k)), int(h))
    L = {n: m.crop(bb).resize(sz, Image.LANCZOS) for n, m in lay.items()}
    t = Image.new("RGBA", sz, (0, 0, 0, 0))
    t.alpha_composite(solid(L["shield"], (24, 92, 190)))
    t.alpha_composite(solid(L["fish"], (238, 240, 246)))
    t.alpha_composite(solid(L["lines"], (24, 60, 120)))
    t.alpha_composite(solid(L["crown"], (236, 190, 60)))
    t.alpha_composite(solid(L["crown_lines"], (120, 80, 20)))
    # чёрный контур вокруг щита и короны, чтобы читался на любом фоне
    sil = ImageChops.lighter(L["shield"], L["crown"])
    o = max(2, int(h * 0.025))
    out = Image.new("RGBA", (sz[0] + 2 * o + 4, sz[1] + 2 * o + 4), (0, 0, 0, 0))
    out.alpha_composite(solid(grow(pad(sil, o + 2), o), INK))
    out.alpha_composite(t, (o + 2, o + 2))
    return out


def karting64_logo(h, col=INK):
    """KARTING64.RU: картинг с пилотом (верх фирменного знака) + надпись шрифтом Tektur."""
    m = Image.open(os.path.join(BX, "karting64_black.png")).convert("L")
    a = np.asarray(m) > 127
    rows = np.nonzero(a.any(1))[0]
    # верхняя часть — картинг: до первой «пустой» полосы над буквами
    prof = a.sum(1)
    top = rows[0]
    gap_rows = [r for r in range(top + 50, a.shape[0]) if prof[r] < 4]
    cut = gap_rows[0] if gap_rows else int(a.shape[0] * 0.45)
    kart = m.crop((0, top, m.width, cut))
    kart = kart.crop(kart.getbbox())
    kh = int(h * 0.62)
    kart = kart.resize((round(kart.width * kh / kart.height), kh), Image.LANCZOS)
    txt = text_h("KARTING64.RU", F_TECH, int(h * 0.36))
    out = Image.new("L", (max(kart.width, txt.width), kh + txt.height + int(h * 0.04)), 0)
    out.paste(kart, ((out.width - kart.width) // 2 + int(txt.width * 0.04), 0))
    out.paste(txt, ((out.width - txt.width) // 2, kh + int(h * 0.04)))
    return solid(out, col)


def driveoil_logo(h, plate=True):
    sys.path.insert(0, os.path.join(LIV, "br03_pro"))
    import lib
    return lib.driveoil_logo(h, plate=plate)


def ru_flag(w, ow=None):
    h = int(w * 2 / 3)
    ow = max(2, w // 24) if ow is None else ow
    t = Image.new("RGBA", (w + 2 * ow, h + 2 * ow), INK + (255,))
    d = ImageDraw.Draw(t)
    for i, c in enumerate(RU):
        d.rectangle([ow, ow + i * h / 3, ow + w - 1, ow + (i + 1) * h / 3 - 1], fill=c + (255,))
    return t


def saratov_flag(w, ow=None):
    """Флаг Саратовской области (как на BR03): белое поле 2/3 над красной полосой 1/3, чёрный кант."""
    h = int(w * 2 / 3)
    ow = max(2, w // 24) if ow is None else ow
    t = Image.new("RGBA", (w + 2 * ow, h + 2 * ow), INK + (255,))
    d = ImageDraw.Draw(t)
    d.rectangle([ow, ow, ow + w - 1, ow + h * 2 / 3 - 1], fill=WHITE + (255,))
    d.rectangle([ow, ow + h * 2 / 3, ow + w - 1, ow + h - 1], fill=(213, 43, 30, 255))
    return t


# ---------------------------------------------------------------- комикс-элементы
def bubble(txt, h, tail="bl", fill=WHITE):
    """Комикс-пузырь с подписью отруба: овал-«подушка», жирный контур, хвостик к отрубу."""
    tm = text_h(txt, F_CUT, int(h * 0.42), track=0.04)
    w = int(tm.width + h * 1.0)
    S = 3
    W, H = w * S, int(h * 1.45) * S
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    bh = h * S
    d.rounded_rectangle([0, 0, W - 1, bh], radius=bh // 2, fill=255)
    # хвостик
    tx = W * (0.22 if tail.endswith("l") else 0.78)
    sgn = -1 if tail.endswith("l") else 1
    d.polygon([(tx - bh * 0.22, bh * 0.8), (tx + bh * 0.22, bh * 0.8), (tx + sgn * bh * 0.30, H - 1)], fill=255)
    m = m.resize((w, int(h * 1.45)), Image.LANCZOS)
    ow = max(3, int(h * 0.07))
    t = comic(m, fill, ow)
    t.alpha_composite(solid(tm, INK), ((t.width - tm.width) // 2, int(ow + 4 + (h - tm.height) / 2)))
    if tail.startswith("t"):
        t = t.transpose(Image.FLIP_TOP_BOTTOM)
        # текст после переворота перерисовываем
        t2 = comic(m.transpose(Image.FLIP_TOP_BOTTOM), fill, ow)
        off = t2.height - (ow + 4 + h)
        t2.alpha_composite(solid(tm, INK), ((t2.width - tm.width) // 2, int(off + (h - tm.height) / 2) + ow // 2))
        t = t2
    return t


def burst(txt, h, fill, spikes=14):
    """Комикс-взрыв «ХРЮ!»: звезда с неровными лучами, текст пузырчатым шрифтом."""
    S = 3
    w = int(h * 1.5)
    m = Image.new("L", (w * S, h * S), 0)
    cx, cy = w * S / 2, h * S / 2
    rng = np.random.default_rng(3)
    pts = []
    for k in range(spikes * 2):
        a = math.pi * 2 * k / (spikes * 2)
        rr = (1.0 if k % 2 == 0 else 0.72) * (0.92 + 0.08 * rng.random())
        pts.append((cx + math.cos(a) * rr * w * S / 2, cy + math.sin(a) * rr * h * S / 2))
    ImageDraw.Draw(m).polygon(pts, fill=255)
    m = m.resize((w, h), Image.LANCZOS)
    t = comic(m, fill, max(3, h // 28))
    tm = text_h(txt, F_POP, int(h * 0.30))
    if tm.width > w * 0.62:
        tm = tm.resize((int(w * 0.62), int(tm.height * w * 0.62 / tm.width)), Image.LANCZOS)
    tt = comic(tm, WHITE, max(2, h // 40), INK, INK, (max(2, h // 50), max(2, h // 50)))
    t.alpha_composite(tt, ((t.width - tt.width) // 2, (t.height - tt.height) // 2))
    return t


def sparkle_mask(r):
    s = int(r * 4)
    m = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(m)
    c = s / 2
    pts = []
    for k in range(8):
        a = math.radians(k * 45)
        rr = r * 1.9 if k % 2 == 0 else r * 0.32
        pts.append((c + math.cos(a) * rr, c + math.sin(a) * rr))
    d.polygon(pts, fill=255)
    return m.crop(m.getbbox())


def pig_mascot(h, acc):
    """Оригинальный маскот: круглая голова свина, уши, гоночные очки на лбу (цвет акцента), хитрая улыбка."""
    S = 3
    H = h * S
    W = int(H * 1.12)
    t = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    ow = int(H * 0.03)
    k = lambda *v: [int(x) for x in v]
    hx0, hy0, hx1, hy1 = W * 0.12, H * 0.2, W * 0.88, H * 0.97
    hcx = (hx0 + hx1) / 2
    # уши (за головой)
    for sgn in (-1, 1):
        base = hcx + sgn * W * 0.24
        ear = [(base - W * 0.13, H * 0.36), (base + sgn * W * 0.20, H * 0.02), (base + W * 0.13, H * 0.30)]
        if sgn < 0:
            ear = [(base - W * 0.13, H * 0.30), (base + sgn * W * 0.20, H * 0.02), (base + W * 0.13, H * 0.36)]
        d.polygon(ear, fill=INK + (255,))
        inner = [((x - base) * 0.62 + base, (y - H * 0.2) * 0.62 + H * 0.24) for x, y in ear]
        sh = [(x + (base - x) * 0.0, y) for x, y in ear]
        d.polygon([((x - base) * 0.8 + base, (y - H * 0.25) * 0.8 + H * 0.27) for x, y in sh], fill=PINK + (255,))
        d.polygon(inner, fill=PINK_D + (255,))
    # голова
    d.ellipse(k(hx0 - ow, hy0 - ow, hx1 + ow, hy1 + ow), fill=INK + (255,))
    d.ellipse(k(hx0, hy0, hx1, hy1), fill=(255, 168, 212, 255))
    # блик на голове (комикс): короткая белая дуга
    d.arc(k(hx0 + W * 0.06, hy0 + H * 0.05, hx1 - W * 0.06, hy1 - H * 0.05), 200, 228, fill=(255, 225, 240, 255),
          width=int(ow * 0.8))
    # очки на лбу: ремешок + две линзы
    sy = H * 0.315
    d.rounded_rectangle(k(hx0 + W * 0.01, sy - H * 0.035, hx1 - W * 0.01, sy + H * 0.035), radius=int(H * 0.03),
                        fill=INK + (255,))
    for sgn in (-1, 1):
        cx = hcx + sgn * W * 0.15
        rr = H * 0.105
        d.ellipse(k(cx - rr - ow, sy - rr * 0.8 - ow, cx + rr + ow, sy + rr * 0.8 + ow), fill=INK + (255,))
        d.ellipse(k(cx - rr, sy - rr * 0.8, cx + rr, sy + rr * 0.8), fill=acc + (255,))
        d.ellipse(k(cx - rr * 0.55, sy - rr * 0.55, cx - rr * 0.05, sy - rr * 0.15), fill=WHITE + (255,))
    # глаза
    for sgn in (-1, 1):
        cx = hcx + sgn * W * 0.135
        ey = H * 0.565
        d.ellipse(k(cx - W * 0.07 - ow * 0.6, ey - H * 0.085 - ow * 0.6, cx + W * 0.07 + ow * 0.6, ey + H * 0.085 + ow * 0.6),
                  fill=INK + (255,))
        d.ellipse(k(cx - W * 0.07, ey - H * 0.085, cx + W * 0.07, ey + H * 0.085), fill=WHITE + (255,))
        px = cx + W * 0.022
        d.ellipse(k(px - W * 0.035, ey - H * 0.045, px + W * 0.035, ey + H * 0.055), fill=INK + (255,))
        d.ellipse(k(px - W * 0.005, ey - H * 0.03, px + W * 0.017, ey - H * 0.005), fill=WHITE + (255,))
        # брови — гоночный прищур (к переносице вниз)
        bx0, bx1 = cx - sgn * W * 0.085, cx + sgn * W * 0.06
        d.line([(bx0, ey - H * 0.092), (bx1, ey - H * 0.135)], fill=INK + (255,), width=int(ow * 1.1))
    # пятачок
    sx0, sy0, sx1, sy1 = hcx - W * 0.17, H * 0.645, hcx + W * 0.17, H * 0.825
    d.ellipse(k(sx0 - ow, sy0 - ow, sx1 + ow, sy1 + ow), fill=INK + (255,))
    d.ellipse(k(sx0, sy0, sx1, sy1), fill=PINK_D + (255,))
    for sgn in (-1, 1):
        nx = hcx + sgn * W * 0.06
        d.ellipse(k(nx - W * 0.03, H * 0.695, nx + W * 0.03, H * 0.775), fill=INK + (255,))
    # ухмылка с клыком (сдвинута вбок)
    d.arc(k(hcx - W * 0.04, H * 0.77, hcx + W * 0.24, H * 0.905), 20, 150, fill=INK + (255,), width=int(ow * 0.85))
    d.polygon([(hcx + W * 0.15, H * 0.898), (hcx + W * 0.20, H * 0.885), (hcx + W * 0.185, H * 0.85)], fill=WHITE + (255,))
    # румянец точками Бен-Дэй
    for sgn in (-1, 1):
        for i, (dx, dy) in enumerate([(0, 0), (0.04, 0), (0.02, 0.035), (-0.02, 0.035), (0.06, 0.035), (0.02, -0.035)]):
            cx = hcx + sgn * (W * 0.29 + dx * W)
            cy = H * 0.72 + dy * H
            rr = H * 0.012
            d.ellipse(k(cx - rr, cy - rr, cx + rr, cy + rr), fill=PINK_D + (255,))
    return t.resize((W // S, H // S), Image.LANCZOS)


def number_plate(txt, h, acc, w_ratio=1.55):
    """Номерной щит: белый, жирная чёрная рамка, цифры Unbounded Black, жёсткая тень цвета акцента."""
    w = int(h * w_ratio)
    ow = max(4, int(h * 0.05))
    sh = max(4, int(h * 0.07))
    plate = Image.new("RGBA", (w + sh, h + sh), (0, 0, 0, 0))
    plate.alpha_composite(rrect(w, h, int(h * 0.16), acc), (sh, sh))
    plate.alpha_composite(rrect(w, h, int(h * 0.16), WHITE, INK, ow), (0, 0))
    nm = text_h(txt, F_NUM, int(h * 0.62), track=0.02)
    if nm.width > w * 0.82:
        nm = nm.resize((int(w * 0.82), int(nm.height * w * 0.82 / nm.width)), Image.LANCZOS)
    plate.alpha_composite(solid(nm, INK), ((w - nm.width) // 2, (h - nm.height) // 2))
    return plate


def name_tag(short, h, acc):
    """Флаг России + фамилия пилота (белое на чёрной плашке)."""
    fl = ru_flag(int(h * 0.62))
    nm = solid(text_h(short, F_NAME, int(h * 0.52)), WHITE)
    content = row([fl, nm], int(h * 0.22))
    return panel(content, int(h * 0.3), int((h - content.height) / 2), INK, acc, max(2, int(h * 0.05)))


# ---------------------------------------------------------------- исходник: тени (AO) и стирание логотипов
def load_source():
    src = Image.open(os.path.join(SRC, "Skin.dds")).convert("RGB")
    a = np.asarray(src).astype(np.float32)
    mx = a.max(-1)
    logo = ndimage.binary_dilation(mx > 110, iterations=6)
    box = np.zeros_like(logo)
    box[715:795, 1945:2145] = True
    rings = ndimage.binary_dilation(box & (mx > 70), iterations=1)      # только сами кольца, без фона
    logo &= ~box
    _, (iy, ix) = ndimage.distance_transform_edt(logo, return_indices=True)
    a[logo] = a[iy[logo], ix[logo]]
    mx = a.max(-1)
    shade = np.clip(0.55 + 0.45 * np.clip(mx / 76.0, 0, 1.12), 0.42, 1.06).astype(np.float32)
    shade = ndimage.uniform_filter(shade, 3)
    return np.asarray(src).astype(np.float32), shade, rings


# ---------------------------------------------------------------- холст-кузов
class Body:
    def __init__(self, G, base):
        self.G = G
        self.A = np.empty((N, N, 3), np.float32)
        self.A[:] = base
        self.report = []
        self.fitter = FITTER

    def fill(self, alpha, col, where=None):
        al = np.clip(alpha, 0, 1).astype(np.float32)
        if where is not None:
            al = al * where
        self.A += (np.asarray(col, np.float32) - self.A) * al[..., None]

    def fill_field(self, alpha, rgb):
        al = np.clip(alpha, 0, 1).astype(np.float32)[..., None]
        self.A += (rgb - self.A) * al

    def put(self, name, art, view, center, width_m=None, height_m=None, parts=None, max_angle=60, side=None,
            min_clear=MIN_CLEAR_CM, fit=True, search=(0.06, 0.04), shrink_to=0.8):
        """Ставит арт проекцией. fit=True — авто-подбор позиции (±search м) и размера (до shrink_to),
        чтобы элемент целиком лежал на панели с запасом ≥ min_clear см."""
        if height_m is None:
            height_m = width_m * art.height / art.width
        extra = 0.0
        for attempt in range(4):
            c, h = center, height_m
            if fit:
                c, h, _ = self.fitter.fit(art, view, center, height_m, parts, side, max_angle,
                                          (min_clear + extra) / 100.0, search, shrink_to)
            idx, rgba, rep = self.G.project(art, view, c, h * art.width / art.height, max_angle, parts, side)
            if not fit or (rep["inside"] > 0.999 and rep["clear_cm"] >= min_clear):
                break
            extra += 0.6                     # проверка по z-буферу строже — уточняем и повторяем
        flat = self.A.reshape(-1, 3)
        al = rgba[:, 3:4]
        flat[idx] = flat[idx] * (1 - al) + rgba[:, :3] * 255.0 * al
        rep.update(name=name, min_clear=min_clear, ok=rep["inside"] > 0.999 and rep["clear_cm"] >= min_clear)
        self.report.append(rep)
        return rep


# ---------------------------------------------------------------- поля концепции (мировые координаты)
# Линия «брюха»: z = g(y). Спереди под фарами, по дверям — по нижнему излому кузова, затем
# комикс-подъём над задней аркой в «окорок» и вокруг кормы.
SW_Y = [-2.4, -1.95, -1.55, -1.05, -0.70, 0.30, 0.62, 0.95, 1.22, 1.45, 1.72, 1.98, 2.18, 2.5]
SW_Z = [0.60, 0.60, 0.56, 0.47, 0.425, 0.415, 0.44, 0.56, 0.74, 0.845, 0.86, 0.76, 0.665, 0.665]
G_SW = PchipInterpolator(SW_Y, SW_Z)

# Пунктиры «мясницкой схемы»: плоскости y = y0 + k·(z − 0.4)
CUTS = [(-0.95, -0.42), (1.36, -0.40)]


def build_fields(G):
    y, z, x = G.y, G.z, G.x
    f_sw = (z - G_SW(np.clip(y, -2.4, 2.5))).astype(np.float32)
    sd_sw = G.signed_dist(f_sw)
    cuts = []
    for y0, k in CUTS:
        f = (y - (y0 + k * (z - 0.4))).astype(np.float32)
        cuts.append(G.signed_dist(f))
    return dict(sd_sw=sd_sw, cuts=cuts)


def arch_outline(G):
    """Чёрный комикс-кант по кромке колёсных арок (через вид сбоку: дыры-арки → расстояние)."""
    k = 400
    y0, y1, z0, z1 = -2.4, 2.5, 0.0, 1.6
    W, H = int((y1 - y0) * k), int((z1 - z0) * k)
    sel = G.valid & (np.abs(G.x) > 0.55) & (np.abs(G.nrm[..., 0]) > 0.25)
    ys = ((G.y[sel] - y0) * k).astype(int)
    zs = ((z1 - G.z[sel]) * k).astype(int)
    cov = np.zeros((H, W), bool)
    m = (ys >= 0) & (ys < W) & (zs >= 0) & (zs < H)
    cov[zs[m], ys[m]] = True
    cov = ndimage.binary_closing(cov, iterations=3)
    lab, n = ndimage.label(~cov)
    arch = np.zeros_like(cov)
    for i in range(1, n + 1):
        comp = lab == i
        rr, cc = np.nonzero(comp)
        if rr.max() < H - 1 - int(0.12 * k):    # должна касаться земли (низ картинки)
            continue
        yc = cc.mean() / k + y0
        if (abs(yc - (-1.31)) < 0.3 or abs(yc - 1.36) < 0.3) and cc.max() - cc.min() < 0.95 * k:
            arch |= comp
    dist = ndimage.distance_transform_edt(~arch) / k        # м до арки (в виде сбоку)
    yy = ((G.y - y0) * k).astype(int).clip(0, W - 1)
    zz = ((z1 - G.z) * k).astype(int).clip(0, H - 1)
    d = dist[zz, yy]
    return np.where(G.valid & (np.abs(G.x) > 0.55), d, 9.0).astype(np.float32)


def halftone(G, sd_sw, cell=0.032, reach=0.30, below=False):
    """Точки Бен-Дэй вдоль линии брюха (над ней, в розовом), радиус убывает с расстоянием.
    Сетка точек — в мировых координатах доминирующей плоскости (борт: y,z; корма/нос: x,z)."""
    nx, ny = np.abs(G.nrm[..., 0]), np.abs(G.nrm[..., 1])
    side = nx >= ny
    u = np.where(side, G.y, G.x)
    v = G.z
    a = math.radians(45)
    uu = u * math.cos(a) + v * math.sin(a)
    vv = -u * math.sin(a) + v * math.cos(a)
    cu = (np.floor(uu / cell) + 0.5) * cell
    cv = (np.floor(vv / cell) + 0.5) * cell
    dd = np.hypot(uu - cu, vv - cv)
    s = -sd_sw if below else sd_sw
    t = np.clip((s - 0.035) / reach, 0, 1)
    rad = cell * 0.5 * (1 - t) ** 1.25 * (s > 0.03)
    al = np.clip((rad - dd) * geom.DENS + 0.5, 0, 1)
    return al.astype(np.float32)


# ---------------------------------------------------------------- сборка кузова
def make_body(G, src, shade, rings, car):
    acc = car["acc"]
    B = Body(G, PINK)
    F = car["fields"]
    sd = F["sd_sw"]
    body = G.valid
    D = geom.DENS
    # нижнее «брюхо» + «окорок» — второй цвет
    B.fill(np.clip(0.5 - sd * D, 0, 1), acc, body)
    # точки Бен-Дэй над линией
    B.fill(F["dots"] * 0.9, PINK_D, body)
    acc_d = tuple(int(c * 0.80) for c in acc)
    B.fill(F["dots_low"] * 0.8, acc_d, body)
    # белый блик-контур над чёрной линией
    w_o = 0.016
    hl0, hl1 = w_o + 0.010, w_o + 0.018
    B.fill(np.clip(np.minimum(sd - hl0, hl1 - sd) * D + 0.5, 0, 1), WHITE, body)
    # чёрная комикс-линия
    B.fill(np.clip((w_o - np.abs(sd)) * D + 0.5, 0, 1), INK, body)
    # кант колёсных арок
    B.fill(np.clip((0.022 - F["arch"]) * D * 2 + 0.5, 0, 1), INK, body & (G.z < 0.95))
    # пунктиры отрубов
    R = 0.9
    t = R * np.arctan2(G.z - 0.45, G.x)
    dash = ((t % 0.10) < 0.062).astype(np.float32)
    exclude = G.parts_mask(["mirror", "hidden", "door_handle", "front_splitter", "rear_diffuser_band"])
    for sdc in F["cuts"]:
        B.fill(np.clip((0.0065 - np.abs(sdc)) * D + 0.5, 0, 1) * dash, INK, body & ~exclude & (G.z > 0.30))
    # торец расширителя заднего крыла (ступенька, смотрит назад): сплошной комикс-чёрный —
    # линия «брюха» по нему шла рваным зигзагом
    step = G.parts_mask(["rear_panel"]) & (np.abs(G.x) > 0.745) & (G.y < 1.98)
    # …и «карман» угла бампера между торцом и фонарём (обращён назад)
    step |= (G.parts_mask(["rear_bumper_corner"]) & (np.abs(G.x) > 0.66) & (G.y < 2.06) & (G.z > 0.69)
             & (G.nrm[..., 1] > 0.45))
    B.fill(step.astype(np.float32), INK)
    # зеркала — цвет акцента с чёрным
    mir = G.parts_mask(["mirror"])
    B.fill(mir.astype(np.float32), acc)
    place_sponsors(B, car)
    # тени исходника (multiply) + кольца Audi оставляем
    out = B.A * shade[..., None]
    out[~G.valid] = src[~G.valid]
    out[rings] = src[rings]
    return np.clip(out, 0, 255).astype(np.uint8), B.report


def place_sponsors(B, car):
    acc = car["acc"]
    P = PPM
    sides = (("L", "left", 0.9), ("R", "right", -0.9))
    side_parts = ["front_door", "rear_door", "front_door_low", "sill", "front_fender", "rear_fender",
                  "front_bumper_corner", "rear_bumper_corner", "front_fender_top", "rear_shoulder", "body_misc"]
    for s, view, xc in sides:
        # 1. Номер на передней двери
        B.put(f"номер 00 дверь {s}", number_plate("00", int(0.30 * P), acc), view, (xc, -0.40, 0.655),
              height_m=0.31, parts=["front_door"], side=s, max_angle=50)
        # 2. «Арка» — главный партнёр — на задней двери
        B.put(f"Арка задняя дверь {s}", arka_logo(int(0.24 * P)), view, (xc, 0.585, 0.70),
              height_m=0.31, parts=["rear_door"], side=s, max_angle=55)
        # 3. Пузырь «ГРУДИНКА» между номером и щелью дверей
        B.put(f"пузырь ГРУДИНКА {s}", bubble("ГРУДИНКА", int(0.075 * P), "bl" if s == "L" else "br"), view,
              (xc, 0.03, 0.60), height_m=0.11, parts=["front_door"], side=s, max_angle=50)
        # 4. Ряд под дверью (второй цвет): DriveOil (топливо) + KARTING64.RU + РАФ
        B.put(f"DriveOil полоса под дверью {s}", driveoil_logo(int(0.060 * P)), view, (xc, -0.60, 0.352),
              height_m=0.078, parts=["front_door_low"], side=s, max_angle=60)
        B.put(f"KARTING64.RU полоса под дверью {s}", karting64_logo(int(0.085 * P)), view, (xc, -0.22, 0.352),
              height_m=0.085, parts=["front_door_low"], side=s, max_angle=60)
        B.put(f"РАФ полоса под дверью {s}", raf_logo(int(0.09 * P)), view, (xc, 0.08, 0.352),
              height_m=0.09, parts=["front_door_low"], side=s, max_angle=60)
        # 5. Порог: simkart.vercel.app
        B.put(f"simkart.vercel.app порог {s}", solid(text_h("simkart.vercel.app", F_URL, int(0.04 * P)), INK), view,
              (xc, 0.05, 0.215), height_m=0.045, parts=["sill"], side=s, max_angle=60)
        # 6. Угол переднего бампера: SMP RACING ESPORTS (серия)
        B.put(f"SMP RACING ESPORTS бампер {s}", smp_lockup(int(0.06 * P), INK, INK), view, (xc, -1.87, 0.44),
              width_m=0.17, parts=["front_bumper_corner"], side=s, max_angle=65)
        # 7. «ОКОРОК» — пузырь на заднем крыле
        B.put(f"пузырь ОКОРОК {s}", bubble("ОКОРОК", int(0.075 * P), "bl" if s == "L" else "br", WHITE), view,
              (xc, 1.80, 0.80), height_m=0.085, parts=["rear_fender"], side=s, max_angle=60, search=(0.12, 0.08))
        # 8. Герб области за задней аркой
        B.put(f"герб Саратовской обл. {s}", saratov_coa(int(0.12 * P)), view, (xc, 1.83, 0.58),
              height_m=0.12, parts=["rear_fender"], side=s, max_angle=60)
    # ---- верх
    B.put("Симкарт капот", panel(simkart_logo(int(0.12 * P), tagline=True), int(0.05 * P), int(0.035 * P), INK, WHITE,
                                 max(3, int(0.006 * P)), acc, int(0.018 * P)),
          "top", (0, -1.345, 0.94), width_m=0.80, parts=["hood"], max_angle=50)
    B.put("пузырь ЛОПАТКА капот", bubble("ЛОПАТКА", int(0.09 * P), "bl"), "top", (0.43, -1.62, 0.9),
          height_m=0.125, parts=["hood"], max_angle=55)
    B.put("номер 00 крыша", number_plate("00", int(0.36 * P), acc), "top_left", (0, 0.17, 1.4),
          height_m=0.37, parts=["roof"], max_angle=40)
    B.put("маскот крыша", pig_mascot(int(0.40 * P), acc), "top_rear", (0, 0.83, 1.39),
          height_m=0.38, parts=["roof"], max_angle=40)
    B.put("пузырь КОРЕЙКА крыша", bubble("КОРЕЙКА", int(0.08 * P), "bl"), "top", (0.0, -0.165, 1.38),
          height_m=0.10, parts=["roof"], max_angle=40)
    # ---- корма
    # крышка багажника: между стойками антикрыла (x = ±0.208 м) — герб + флаг области + «64»,
    # чтобы стойки не перечёркивали надпись при взгляде сзади
    r64 = stack([solid(text_h("РЕГИОН", F_SPON, int(0.020 * P), track=0.12), INK),
                 solid(text_h("64", F_SPON, int(0.060 * P)), INK)], int(0.008 * P))
    B.put("Саратовская обл. крышка багажника",
          row([saratov_flag(int(0.075 * P)), saratov_coa(int(0.11 * P)), r64], int(0.022 * P)), "top_rear",
          (0, 1.985, 1.05), width_m=0.30, parts=["trunk_lid"], max_angle=60, search=(0.0, 0.03), shrink_to=0.75)
    edm = stack([solid(text_h("КОМАНДА", F_SPON, int(0.030 * P), track=0.1), INK),
                 solid(text_h("ЭДМ", F_SPON, int(0.075 * P)), INK)], int(0.01 * P))
    B.put("Команда ЭДМ задняя панель", row([mbu_logo(int(0.12 * P)), edm], int(0.03 * P)), "rear",
          (0.0, 2.16, 0.77), height_m=0.12, parts=["rear_panel"], max_angle=55)
    sim_rear = panel(row([simkart_logo(int(0.055 * P)),
                          solid(text_h("simkart.vercel.app", F_URL, int(0.026 * P)), WHITE)], int(0.03 * P)),
                     int(0.03 * P), int(0.014 * P), INK, WHITE, max(3, int(0.004 * P)))
    B.put("Симкарт + ссылка задний бампер", sim_rear, "rear", (0.0, 2.25, 0.585), height_m=0.10,
          parts=["rear_bumper"], max_angle=60, search=(0.05, 0.05))
    B.put("САРАТОВСКАЯ ОБЛАСТЬ задний бампер",
          solid(text_h("САРАТОВСКАЯ ОБЛАСТЬ · РЕГИОН 64", F_SPON, int(0.03 * P)), INK), "rear",
          (0.0, 2.25, 0.49), height_m=0.027, parts=["rear_bumper"], max_angle=60, search=(0.0, 0.02))
    # ---- перед
    B.put("номер 00 перед", number_plate("00", int(0.10 * P), acc), "front", (-0.52, -1.95, 0.50),
          height_m=0.10, parts=["front_bumper", "front_bumper_corner"], max_angle=55)
    B.put("BR ENGINEERING перед", br_logo(int(0.03 * P)), "front", (0.52, -1.95, 0.50),
          width_m=0.24, parts=["front_bumper", "front_bumper_corner"], max_angle=55)


# ---------------------------------------------------------------- glass_sticker.dds
def make_glass(car, masks):
    acc = car["acc"]
    g = Image.open(os.path.join(SRC, "glass_sticker.dds")).convert("RGBA")
    G = Image.new("RGBA", g.size, (0, 0, 0, 0))
    S = g.width / 1024
    rep = []

    def area(mask, box, fill):
        lay = Image.new("RGBA", g.size, fill + (255,))
        m = Image.new("L", g.size, 0)
        m.paste(mask.crop(box), box)
        lay.putalpha(m)
        G.alpha_composite(lay)

    def put(name, art, cx, cy, zone_mask, angle=0, min_clear=4):
        if angle:
            art = art.rotate(angle, expand=True, resample=Image.BICUBIC)
        x0, y0 = int(round(cx - art.width / 2)), int(round(cy - art.height / 2))
        G.alpha_composite(art, (x0, y0))
        a = np.zeros((g.height, g.width), bool)
        al = np.asarray(art.getchannel("A")) > 64
        a[y0:y0 + art.height, x0:x0 + art.width] = al[:max(0, g.height - y0), :max(0, g.width - x0)]
        zm = np.asarray(zone_mask) > 127
        dt = ndimage.distance_transform_edt(zm)
        clear = float(dt[a].min()) if a.any() else 0
        inside = float((a & zm).sum()) / max(1, a.sum())
        rep.append(dict(name=name, view="glass", inside=inside, clear_cm=clear, parts=["glass_sticker"],
                        size_cm=(art.width, art.height), min_clear=min_clear, ok=inside > 0.999 and clear >= min_clear,
                        unit="px"))

    def fit(name, make, h0, zone_mask, min_clear=5, angle=0):
        """Подбирает высоту и центр так, чтобы элемент целиком лежал в зоне с запасом ≥ min_clear px."""
        zm = np.asarray(zone_mask) > 127
        dt = ndimage.distance_transform_edt(zm)
        ys, xs = np.nonzero(zm)
        h = h0
        while h >= 10:
            art = make(h)
            if angle:
                art = art.rotate(angle, expand=True, resample=Image.BICUBIC)
            py, px = np.nonzero(np.asarray(art.getchannel("A")) > 64)
            sel = np.linspace(0, len(px) - 1, min(len(px), 3000)).astype(int)
            py, px = py[sel] - art.height / 2, px[sel] - art.width / 2
            best = None
            for cy in range(ys.min(), ys.max() + 1, 3):
                for cx in range(xs.min(), xs.max() + 1, 3):
                    X = np.round(px + cx).astype(int); Y = np.round(py + cy).astype(int)
                    if X.min() < 0 or Y.min() < 0 or X.max() >= dt.shape[1] or Y.max() >= dt.shape[0]:
                        continue
                    v = dt[Y, X].min()
                    if best is None or v > best[0]:
                        best = (v, cx, cy)
            if best and best[0] >= min_clear + 2:
                put(name, art, best[1], best[2], zone_mask, 0, min_clear)
                return h
            h = int(h * 0.94)
        raise RuntimeError("не влезло: " + name)

    m = masks
    ws = m["Circle.059"]
    ext = m["Plane.033"]
    # зоны
    def zone(mask, box):
        z = Image.new("L", g.size, 0)
        z.paste(mask.crop(box), box)
        return z
    z_ws = zone(ws, (0, 0, 1024, 156))
    z_rw = zone(ws, (0, 156, 1024, 262))
    z_sl = zone(ws, (0, 262, 560, 505))
    z_sr = zone(ws, (0, 505, 560, 760))
    z_wt = zone(ext, (0, 856, 1024, 1024))
    z_el = zone(ext, (320, 750, 530, 858))
    z_er = zone(ext, (100, 750, 318, 858))
    # ---- лобовое: чёрная полоса, SMP RACING ESPORTS, искры
    area(ws, (0, 0, 1024, 156), INK)
    stripe = Image.new("L", g.size, 0)
    ImageDraw.Draw(stripe).rectangle([0, 132, 1024, 140], fill=255)
    G.alpha_composite(solid(ImageChops.multiply(stripe, z_ws), acc))
    put("SMP RACING ESPORTS лобовое", smp_lockup(92, WHITE, acc), 508, 70, z_ws, min_clear=6)
    for sx in (250, 766):
        put(f"искра лобовое {sx}", solid(sparkle_mask(10), acc), sx, 70, z_ws, min_clear=6)
    # ---- заднее стекло: полоса, номер, команда
    area(ws, (0, 156, 1024, 262), INK)
    put("номер 00 заднее стекло", number_plate("00", 70, acc, 1.6), 500, 208, z_rw, min_clear=4)
    put("POP PIG заднее стекло", comic(text_h("POP PIG", F_POP, 40), PINK, 3), 300, 208, z_rw, min_clear=4)
    put("РЕГИОН 64 заднее стекло", solid(text_h("САРАТОВ·64", F_SPON, 30), WHITE), 710, 208, z_rw, min_clear=4)
    # ---- боковые окна: флаг + фамилия
    fit("имя пилота окно L", lambda h: name_tag(car["short"], h, acc), 74, z_sl, min_clear=8)
    fit("имя пилота окно R", lambda h: name_tag(car["short"], h, acc), 74, z_sr, min_clear=8)
    # ---- антикрыло сверху (читается сзади: поворот 180)
    area(ext, (0, 856, 1024, 1024), INK)
    stripe = Image.new("L", g.size, 0)
    ImageDraw.Draw(stripe).rectangle([0, 994, 1024, 1010], fill=255)
    ImageDraw.Draw(stripe).rectangle([0, 856, 1024, 872], fill=255)
    G.alpha_composite(solid(ImageChops.multiply(stripe, z_wt), PINK))
    wing = row([arka_logo(84, taimcafe=False), pig_snout(70), simkart_logo(70)], 46)
    put("Арка · Симкарт антикрыло", wing, 522, 933, z_wt, angle=180, min_clear=6)
    # ---- концевые пластины: маскот + «ХРЮ!»
    for nm_, z, box in (("L", z_el, (320, 750, 530, 858)), ("R", z_er, (100, 750, 318, 858))):
        area(ext, box, PINK)
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2 + 2
        fit(f"маскот + ХРЮ! пластина {nm_}",
            lambda h: row([pig_mascot(h, acc), comic(text_h("ХРЮ!", F_POP, int(h * 0.42)), acc, max(2, h // 22))], h // 10),
            80, z, min_clear=5)
    # итог: прозрачность только там, где нарисовано; вне — исходная альфа (0)
    out = g.copy()
    out.alpha_composite(G)
    a = np.asarray(out).copy()
    painted = np.asarray(G.getchannel("A")) > 0
    a[..., 3] = np.where(painted, 255, np.asarray(g.getchannel("A")))
    return Image.fromarray(a, "RGBA"), rep


def pig_snout(h):
    w = int(h * 1.35)
    S = 3
    t = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    ow = int(h * S * 0.07)
    d.ellipse([0, 0, w * S - 1, h * S - 1], fill=INK + (255,))
    d.ellipse([ow, ow, w * S - 1 - ow, h * S - 1 - ow], fill=PINK_D + (255,))
    for cx in (0.34, 0.66):
        d.ellipse([w * S * cx - w * S * 0.09, h * S * 0.28, w * S * cx + w * S * 0.09, h * S * 0.72], fill=INK + (255,))
    return t.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------- вывод
def save_dxt5(im, path):
    im.convert("RGBA").save(path, pixel_format="DXT5")
    with open(path, "r+b") as fh:
        fh.seek(20)
        fh.write(struct.pack("<I", max(1, im.width // 4) * max(1, im.height // 4) * 16))
        fh.seek(88)
        fh.write(struct.pack("<I", 0))


def livery_icon(car, S=185):
    """Иконка livery.png: розовое/акцентное поле с комикс-линией, маскот и номер."""
    k = 4
    W = S * k
    t = Image.new("RGBA", (W, W), PINK + (255,))
    d = ImageDraw.Draw(t)
    d.polygon([(0, W * 0.72), (W, W * 0.48), (W, W), (0, W)], fill=car["acc"] + (255,))
    d.line([(0, W * 0.72), (W, W * 0.48)], fill=INK + (255,), width=int(W * 0.03))
    pig = pig_mascot(int(W * 0.62), car["acc"])
    t.alpha_composite(pig, (int(W * 0.04), int(W * 0.06)))
    plate = number_plate("00", int(W * 0.24), car["acc"] if car["acc"] != LIME else WHITE)
    t.alpha_composite(plate, (W - plate.width - int(W * 0.05), W - plate.height - int(W * 0.06)))
    return t.resize((S, S), Image.LANCZOS)


CARS = [
    dict(folder="Pozdnyakov_00", driver="Станислав Поздняков", short="С. ПОЗДНЯКОВ", acc=LIME),
    dict(folder="Konopelko_00", driver="Матвей Конопелько", short="М. КОНОПЕЛЬКО", acc=CYAN),
]


def main():
    only = os.environ.get("ONLY")
    global FITTER
    G = geom.Geo()
    FITTER = geom.Fitter(G)
    src, shade, rings = load_source()
    fields = build_fields(G)
    fields["arch"] = arch_outline(G)
    fields["dots"] = halftone(G, fields["sd_sw"])
    fields["dots_low"] = halftone(G, fields["sd_sw"], cell=0.026, reach=0.16, below=True)
    masks = geom.rasterize_glass_masks(KN5)
    lines = []
    for car in CARS:
        if only and car["folder"] != only:
            continue
        car["fields"] = fields
        skin, rep = make_body(G, src, shade, rings, car)
        glass, rep_g = make_glass(car, masks)
        out = os.path.join(HERE, car["folder"])
        os.makedirs(out, exist_ok=True)
        img = Image.fromarray(skin, "RGB")
        save_dxt5(img, os.path.join(out, "Skin.dds"))
        save_dxt5(glass, os.path.join(out, "glass_sticker.dds"))
        img.save(os.path.join(SCR, "pp", car["folder"] + "_skin.png"))
        glass.save(os.path.join(SCR, "pp", car["folder"] + "_glass.png"))
        if car is CARS[0]:
            img.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, "texture_preview.png"))
            bg = Image.new("RGBA", glass.size, (90, 90, 100, 255))
            bg.alpha_composite(glass)
            bg.convert("RGB").save(os.path.join(HERE, "glass_preview.png"))
        livery_icon(car).save(os.path.join(out, "livery.png"))
        with open(os.path.join(out, "ui_skin.json"), "w", encoding="utf-8") as fh:
            json.dump({"skinname": car["folder"], "drivername": car["driver"], "country": "Russia",
                       "team": "Команда ЭДМ", "number": "00", "priority": 1}, fh, ensure_ascii=False, indent=2)
        with zipfile.ZipFile(os.path.join(HERE, car["folder"] + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(os.listdir(out)):
                z.write(os.path.join(out, f), car["folder"] + "/" + f)
        lines.append(f"== {car['folder']} ==")
        bad = 0
        for r in rep + rep_g:
            unit = r.get("unit", "см")
            flag = "OK " if r["ok"] else "FAIL"
            bad += not r["ok"]
            lines.append(f"{flag} {r['name']:<42} покрытие {r['inside'] * 100:6.2f}%  запас {r['clear_cm']:5.1f} {unit}"
                         f"  (мин {r['min_clear']})  {r['view']:<9} {'/'.join(r['parts'])}")
        lines.append(f"итого: {len(rep) + len(rep_g)} элементов, ошибок: {bad}")
        print("\n".join(lines[-(len(rep) + len(rep_g) + 2):]))
    with open(os.path.join(HERE, "check_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
