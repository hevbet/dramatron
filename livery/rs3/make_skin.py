"""Скин «Castrol GT500 tribute #86» для Audi RS3 LMS (Assetto Corsa).

Трибьют ливреи Castrol TOM'S Supra GT500: белая база, красная и зелёная «волны».
Берёт Skin.dds и glass_sticker.dds из скина SMP01, стирает старые логотипы,
перекрашивает с сохранением запечённых теней и рисует новую графику.

Развёртка RS3: в центре крыша/багажник/капот (перед — внизу), по бокам — борта,
«вверх» кузова = к центру развёртки. Правая колонка — зеркало левой.

Запуск: python3 make_skin.py <папка исходного скина SMP01>
"""
import json
import math
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

SRC = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "castrol_gt500_86")
os.makedirs(OUT, exist_ok=True)
F_RACE = os.path.join(HERE, "..", "br03", "brand", "fonts", "Exo2-Italic[wght].ttf")

WHITE = np.array([242, 243, 240], float)
RED = (214, 36, 34)          # красный Castrol
GREEN = (0, 150, 72)         # зелёный Castrol
BLACK = (18, 18, 20)
TXT_WHITE = (250, 250, 250)

src = Image.open(os.path.join(SRC, "Skin.dds")).convert("RGB")
N = src.width
a = np.asarray(src).astype(float)
mx = a.max(-1)

# ---------------------------------------------------------------- 1. стираем логотипы (всё светлее кузова)
logo = mx > 110
rings = np.zeros_like(logo)
rings[720:790, 1950:2140] = True           # кольца Audi на багажнике оставляем
logo &= ~rings
logo = ndimage.binary_dilation(logo, iterations=6)
_, (iy, ix) = ndimage.distance_transform_edt(logo, return_indices=True)
a[logo] = a[iy[logo], ix[logo]]
mx = a.max(-1)

# ---------------------------------------------------------------- 2. белая база с тенями
body = (a[..., 2] > a[..., 0] + 12) & (mx > 18)
shade = np.clip(mx / 76.0, 0, 1.12)[..., None]
out = a.copy()
out[body] = (WHITE * shade)[body]
out[rings] = np.asarray(src).astype(float)[rings]
body_mask = Image.fromarray((body * 255).astype(np.uint8), "L")
shade_img = Image.fromarray((np.clip(shade[..., 0] / 1.12, 0, 1) * 255).astype(np.uint8), "L")

# слой графики рисуем отдельно, потом умножаем на тени и режем по кузову
art = Image.new("RGBA", (N, N), (0, 0, 0, 0))


def catmull(points, n=40):
    pts = [points[0]] + points + [points[-1]]
    res = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for k in range(n):
            t = k / n
            res.append(tuple(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t * t
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t ** 3) for j in range(2)))
    res.append(points[-1])
    return res


def band(layer, pts, widths, col):
    """Лента переменной толщины вдоль сплайна (толщина интерполируется по widths)."""
    d = ImageDraw.Draw(layer)
    curve = catmull(pts)
    n = len(curve)
    for i, (x, y) in enumerate(curve):
        t = i / (n - 1) * (len(widths) - 1)
        k = min(int(t), len(widths) - 2)
        r = (widths[k] + (widths[k + 1] - widths[k]) * (t - k)) / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (255,))


def arc_band(layer, cx, cy, rad, width, a0, a1, col):
    d = ImageDraw.Draw(layer)
    for k in range(200):
        ang = math.radians(a0 + (a1 - a0) * k / 199)
        taper = math.sin(math.pi * k / 199) ** 0.6
        r = width / 2 * max(0.25, taper)
        x, y = cx + rad * math.cos(ang), cy + rad * math.sin(ang)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (255,))


def mirror_pts(pts):
    return [(N - x, y) for x, y in pts]


def ribbon(layer, pts, widths, col, offset=0.0):
    """Лента вдоль сплайна со смещением по нормали «вверх» кузова (offset в пикселях, может зависеть от
    ширины: offset=callable(w) → смещение). Нормаль «вверх» = (dy, −dx) при движении спереди назад."""
    d = ImageDraw.Draw(layer)
    curve = catmull(pts, 60)
    n = len(curve)
    for i, (x, y) in enumerate(curve):
        j0, j1 = max(0, i - 2), min(n - 1, i + 2)
        dx, dy = curve[j1][0] - curve[j0][0], curve[j1][1] - curve[j0][1]
        ln = math.hypot(dx, dy) or 1
        nx, ny = dy / ln, -dx / ln
        t = i / (n - 1) * (len(widths) - 1)
        k = min(int(t), len(widths) - 2)
        w = widths[k] + (widths[k + 1] - widths[k]) * (t - k)
        off = offset(w) if callable(offset) else offset
        cx_, cy_ = x + nx * off, y + ny * off
        r = w / 2
        d.ellipse([cx_ - r, cy_ - r, cx_ + r, cy_ + r], fill=col + (255,))


def arc_pts(cx, cy, r, a0, a1, n=6):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]


def swoosh(layer, pts, widths):
    """Фирменная связка Castrol: широкая красная лента + зелёная кромка сверху через белый зазор."""
    ribbon(layer, pts, [max(4, w * 0.42) for w in widths], GREEN, offset=lambda w: w / 2 + 14 + w * 0.21 + 6)
    ribbon(layer, pts, widths, RED)


# ---------------------------------------------------------------- 3. борта: одна непрерывная линия
# правая колонка развёртки: «вниз» кузова = +x, «вперёд» = +y; арки: передняя (3560, 3270) R≈310, задняя (3710, 860) R≈290
FRONT_ARCH, REAR_ARCH = (3560, 3270), (3710, 860)
SIDE = ([(3470, 4060), (3420, 3860)]                                  # угол переднего бампера
        + arc_pts(*FRONT_ARCH, 395, 115, 245, 6)                      # огибает переднюю арку
        + [(3600, 2620), (3640, 2200), (3620, 1760), (3560, 1420)]    # вдоль низа дверей, плавно поднимаясь
        + arc_pts(*REAR_ARCH, 385, 130, 235, 5)                       # огибает заднюю арку
        + [(3770, 400), (3840, 250)])                                 # уходит в задний бампер
SIDE_W = [20, 60, 90, 110, 120, 125, 120, 115, 120, 150, 175, 175, 160, 140, 130, 120, 110, 100, 90, 60, 20]
for tr in ((lambda p: p), mirror_pts):
    swoosh(art, tr(SIDE), SIDE_W)

# ---------------------------------------------------------------- 4. сквозная линия капот → крыша → багажник, бамперы
HOOD = [(2050, 3700), (2330, 3640), (2560, 3420), (2600, 3120), (2470, 2840)]
ROOF = [(2445, 2400), (2445, 1800), (2445, 1180)]
TRUNK = [(2445, 1170), (2420, 950), (2330, 760)]
for tr in ((lambda p: p), mirror_pts):
    swoosh(art, tr(HOOD), [30, 90, 110, 90, 60])
    swoosh(art, tr(ROOF), [60, 60, 60])
    swoosh(art, tr(TRUNK), [60, 40, 12])
# бамперы: полосы сужаются к краям и продолжают бортовую линию
ribbon(art, [(1120, 250), (1600, 225), (2050, 220), (2500, 225), (2980, 250)], [20, 60, 70, 60, 20], GREEN)
ribbon(art, [(1120, 200), (1600, 170), (2050, 165), (2500, 170), (2980, 200)], [30, 80, 90, 80, 30], RED)
ribbon(art, [(1400, 3900), (2050, 3990), (2700, 3900)], [20, 70, 20], RED)
ribbon(art, [(1400, 3960), (2050, 4050), (2700, 3960)], [10, 34, 10], GREEN)

# тени на графику + обрезка по кузову
rgb = np.asarray(art.convert("RGB")).astype(float) * np.clip(shade, 0.35, 1.05)
alpha = np.minimum(np.asarray(art.getchannel("A")), np.asarray(body_mask))
m = alpha > 0
out[m] = out[m] * (1 - alpha[m, None] / 255) + rgb[m] * (alpha[m, None] / 255)
img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


# ---------------------------------------------------------------- 5. надписи
def font(size, weight="Black Italic"):
    f = ImageFont.truetype(F_RACE, size)
    f.set_variation_by_name(weight)
    return f


def text_layer(s, f, fill, stroke=0, stroke_fill=None):
    bb = f.getbbox(s, stroke_width=stroke)
    t = Image.new("RGBA", (bb[2] - bb[0] + 20, bb[3] - bb[1] + 20), (0, 0, 0, 0))
    ImageDraw.Draw(t).text((10 - bb[0], 10 - bb[1]), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return t


def fit(s, size, maxw, weight="Black Italic", **kw):
    while size > 10:
        t = text_layer(s, font(size, weight), **kw)
        if t.width <= maxw:
            return t
        size -= 2
    return t


def castrol(size, maxw):
    """Надпись Castrol: красный наклонный жирный с белой обводкой и зелёным подчёркиванием."""
    t = fit("Castrol", size, maxw, fill=RED, stroke=max(3, size // 14), stroke_fill=TXT_WHITE)
    out_ = Image.new("RGBA", (t.width, t.height + size // 3), (0, 0, 0, 0))
    ImageDraw.Draw(out_).rounded_rectangle([size // 5, t.height - size // 10, t.width - size // 6, t.height + size // 8],
                                           radius=size // 12, fill=GREEN)
    out_.alpha_composite(t, (0, 0))
    return out_


def number_plate(h):
    """Номер 86: чёрные цифры на белой табличке с чёрной рамкой, как на GT500."""
    f = font(int(h * 0.9))
    txt = text_layer("86", f, BLACK)
    w = int(txt.width + h * 0.35)
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(t).rounded_rectangle([2, 2, w - 3, h - 3], radius=h // 10, fill=TXT_WHITE, outline=BLACK, width=max(6, h // 28))
    t.alpha_composite(txt, ((w - txt.width) // 2, (h - txt.height) // 2))
    return t


def name_plate():
    """Табличка пилота: триколор + имя на белой плашке с чёрной рамкой."""
    txt = fit("МАТВЕЙ КОНОПЕЛЬКО", 40, 470, fill=BLACK)
    w, h = txt.width + 90, 66
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d_ = ImageDraw.Draw(t)
    d_.rounded_rectangle([1, 1, w - 2, h - 2], radius=10, fill=TXT_WHITE, outline=BLACK, width=3)
    for k, c in enumerate([(245, 245, 245), (0, 57, 166), (213, 43, 30)]):
        d_.rectangle([12, 12 + k * 14, 58, 26 + k * 14], fill=c)
    d_.rectangle([12, 12, 58, 54], outline=BLACK, width=2)
    t.alpha_composite(txt, (70, (h - txt.height) // 2))
    return t


def sponsor_row(names, h, maxw, fill=BLACK, sep="  ·  "):
    return fit(sep.join(names), h, maxw, "ExtraBold Italic", fill=fill)


def meme(s, h, maxw):
    """Мем-стикер: белая плашка, чёрный текст (строки через \\n), красная рамка."""
    lines = [fit(ln, h, maxw - 40, "ExtraBold Italic", fill=BLACK) for ln in s.split("\n")]
    tw, th = max(l.width for l in lines), sum(l.height for l in lines)
    t = Image.new("RGBA", (tw + 40, th + 18), (0, 0, 0, 0))
    ImageDraw.Draw(t).rounded_rectangle([1, 1, t.width - 2, t.height - 2], radius=12, fill=TXT_WHITE, outline=RED, width=4)
    y = 9
    for l in lines:
        t.alpha_composite(l, ((t.width - l.width) // 2, y))
        y += l.height
    return t


def place(layer, cx, cy, angle=0):
    if angle:
        layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
    full = Image.new("RGBA", img.size, (0, 0, 0, 0))
    full.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)))
    full.putalpha(ImageChops.multiply(full.getchannel("A"), body_mask))
    img.alpha_composite(full)


# борта: правая колонка читается снизу вверх (+90), левая — сверху вниз (−90); «вниз» кузова = +x
for side, ang in ((1, 90), (-1, -90)):
    X = (lambda x: x) if side == 1 else (lambda x: N - x)
    place(castrol(160, 600), X(3255), 1440, ang)                                     # задняя дверь, ниже окна
    place(meme("Работает на 2.0 TFSI\nи молитвах", 36, 400), X(3435), 1585, ang)     # под Castrol, до арки
    place(number_plate(290), X(3265), 2230, ang)                                     # передняя дверь, воздух сверху
    place(name_plate(), X(3445), 2230, ang)                                          # под номером
    place(sponsor_row(["SMP ESPORTS", "BILSTEIN", "EIBACH", "RECARO", "SPARCO", "BREMBO", "MOTUL", "APR"], 48, 1460),
          X(3920), 1980, ang)                                                        # порог

place(castrol(190, 700), 2050, 3170)                                   # капот (ниже, подальше от лобового)
place(number_plate(520), 2050, 1790)                                   # крыша
place(castrol(140, 560), 2050, 900, 180)                               # багажник, ближе к кольцам
# задняя стенка под кольцами (бывшая надпись SMP ESPORTS) — единственный мем сзади
def livery_banner(text, h, maxw):
    """Надпись в стиле ливреи: белый наклонный текст на красной ленте с заострёнными концами и зелёной кромкой."""
    txt = fit(text, int(h * 0.62), maxw - int(h * 1.4), fill=TXT_WHITE, stroke=3, stroke_fill=(120, 16, 16))
    w = txt.width + int(h * 1.4)
    S = 3
    t = Image.new("RGBA", (w * S, (h + h // 3) * S), (0, 0, 0, 0))
    d_ = ImageDraw.Draw(t)
    sk = h * 0.35 * S                                           # наклон как у шрифта
    d_.polygon([(0, h * S), (sk, 0), (w * S, 0), (w * S - sk, h * S)], fill=RED)                  # красная лента
    g0, g1 = (h + h // 12) * S, (h + h // 4) * S
    d_.polygon([(sk * 0.3, g1), (sk * 0.3 + (g1 - g0) * 0.35, g0), (w * S - sk * 0.9, g0),
                (w * S - sk * 0.9 - (g1 - g0) * 0.35, g1)], fill=GREEN)                           # зелёная кромка
    t = t.resize((w, h + h // 3), Image.LANCZOS)
    t.alpha_composite(txt, ((w - txt.width) // 2, (h - txt.height) // 2))
    return t


place(livery_banner("МАСЛО НЕ ЖРЁТ (ПОКА)", 112, 760), 2048, 530, 180)
def badge(text, bg, fg, h, maxw, accent=None):
    """Спонсорская наклейка под ливрею: скошенная плашка фирменного цвета, белая окантовка, наклонный текст."""
    txt = fit(text, int(h * 0.66), maxw - int(h * 1.1), fill=fg)
    w = txt.width + int(h * 1.1)
    S = 3
    t = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d_ = ImageDraw.Draw(t)
    sk = h * 0.32 * S
    d_.polygon([(0, h * S), (sk, 0), (w * S, 0), (w * S - sk, h * S)], fill=TXT_WHITE)            # белая окантовка
    b = 5 * S
    d_.polygon([(b * 1.6, h * S - b), (sk + b * 0.6, b), (w * S - b * 1.6, b), (w * S - sk - b * 0.6, h * S - b)], fill=bg)
    if accent:                                                                                   # фирменный скошенный акцент
        d_.polygon([(w * S - sk - b * 0.6 - h * 0.5 * S, h * S - b), (w * S - b * 1.6 - h * 0.5 * S + sk * 0.2, b),
                    (w * S - b * 1.6, b), (w * S - sk - b * 0.6, h * S - b)], fill=accent)
    t = t.resize((w, h), Image.LANCZOS)
    t.alpha_composite(txt, ((w - txt.width) // 2 - (int(h * 0.12) if accent else 0), (h - txt.height) // 2))
    return t


NVIDIA = lambda h, mw: badge("NVIDIA", (118, 185, 0), BLACK, h, mw)
INTEL = lambda h, mw: badge("intel", (0, 113, 197), TXT_WHITE, h, mw)
MOZA = lambda h, mw: badge("MOZA RACING", BLACK, TXT_WHITE, h, mw, accent=RED)

# крыша: над и под номером
place(NVIDIA(130, 560), 2050, 2250)
place(INTEL(130, 480), 2050, 1330)
# передняя дверь: перед номером MOZA, за номером intel
for side, ang in ((1, 90), (-1, -90)):
    X = (lambda x: x) if side == 1 else (lambda x: N - x)
    place(MOZA(80, 300), X(3270), 2590, ang)
    place(NVIDIA(64, 220), X(3270), 1905, ang)
# багажник под Castrol
place(MOZA(70, 380), 2050, 1080, 180)
# капот по бокам воздухозаборника
place(INTEL(56, 170), 1640, 3470, 0)
place(NVIDIA(56, 190), N - 1640, 3470, 0)

# передний бампер: мелкие спонсоры на нижних «клыках» губы и на боковых гранях
place(fit("BREMBO", 32, 240, fill=TXT_WHITE), 1760, 3950, -6)
place(fit("MOTUL", 32, 240, fill=TXT_WHITE), N - 1760, 3950, 6)
place(fit("SMP ESPORTS", 40, 360, "ExtraBold Italic", fill=BLACK), 1090, 3750)
place(fit("APR", 48, 200, fill=BLACK), N - 1090, 3750)
# задний бампер под полосами: спонсоры
place(sponsor_row(["SMP ESPORTS", "CASTROL", "NVIDIA", "INTEL", "MOZA", "APR"], 44, 1500), 2050, 95, 180)

img = img.convert("RGB")
img.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, "skin_preview.png"))


def save_dxt5(im, path):
    """Pillow пишет в заголовок неверные pitch и RGBBitCount — правим как в оригинальных DDS."""
    im.save(path, pixel_format="DXT5")
    with open(path, "r+b") as fh:
        fh.seek(20)
        fh.write(struct.pack("<I", max(1, im.width // 4) * max(1, im.height // 4) * 16))
        fh.seek(88)
        fh.write(struct.pack("<I", 0))


save_dxt5(img.convert("RGBA"), os.path.join(OUT, "Skin.dds"))

# ---------------------------------------------------------------- 6. полоса на лобовом стекле
glass = Image.open(os.path.join(SRC, "glass_sticker.dds")).convert("RGBA")
G = glass.width
bh = int(G * 150 / 1024)
gd = ImageDraw.Draw(glass)
gd.rectangle([0, 0, G, bh], fill=TXT_WHITE + (255,))
gd.rectangle([0, bh - int(G * 0.012), G, bh], fill=GREEN + (255,))
gd.rectangle([0, bh - int(G * 0.024), G, bh - int(G * 0.012)], fill=RED + (255,))
banner = castrol(int(bh * 0.62), int(G * 0.5))
glass.alpha_composite(banner, ((G - banner.width) // 2, max(0, (bh - banner.height) // 2 - int(G * 0.008))))
save_dxt5(glass, os.path.join(OUT, "glass_sticker.dds"))

# ---------------------------------------------------------------- 7. иконка и ui_skin.json
ic = Image.new("RGB", (185, 185), tuple(int(v) for v in WHITE))
di = ImageDraw.Draw(ic)
di.pieslice([-120, 60, 150, 330], 250, 360, fill=RED)
di.pieslice([40, -150, 320, 110], 70, 180, fill=GREEN)
di.text((92, 95), "86", font=font(90), fill=BLACK, anchor="mm")
ic.save(os.path.join(OUT, "livery.png"))

with open(os.path.join(OUT, "ui_skin.json"), "w", encoding="utf-8") as fh:
    json.dump({"skinname": "Castrol GT500 tribute #86", "drivername": "Матвей Конопелько", "country": "Russia",
               "team": "Castrol GT500 tribute", "number": "86", "priority": 1}, fh, ensure_ascii=False, indent=2)
print("готово:", OUT)
