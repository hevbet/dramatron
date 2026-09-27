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


# ---------------------------------------------------------------- 3. борта (рисуем правую колонку и зеркалим)
SIDE_RED = [(3420, 2900), (3470, 2350), (3480, 1600), (3420, 1180), (3250, 900), (3090, 650), (3060, 390), (3170, 190)]
SIDE_RED_W = [60, 150, 190, 190, 170, 140, 110, 60]
SIDE_GREEN = [(x + 170, y - 110) for x, y in SIDE_RED]
SIDE_GREEN_W = [30, 70, 90, 90, 80, 70, 50, 26]
for side in (1, -1):
    tr = (lambda p: p) if side == 1 else mirror_pts
    band(art, tr(SIDE_GREEN), SIDE_GREEN_W, GREEN)
    band(art, tr(SIDE_RED), SIDE_RED_W, RED)
    cx = 3560 if side == 1 else N - 3560
    a0, a1 = (100, 260) if side == 1 else (-80, 80)
    arc_band(art, cx, 3265, 380, 70, a0, a1, GREEN)     # «скобка» над передним колесом
    arc_band(art, cx, 3265, 470, 36, a0 + 10, a1 - 10, RED)

# ---------------------------------------------------------------- 4. капот, крыша, багажник, бамперы
HOOD = [(2050, 3660), (2350, 3600), (2580, 3380), (2610, 3080), (2540, 2830)]
band(art, [(x + 60, y - 60) for x, y in HOOD], [40, 60, 70, 60, 30], GREEN)
band(art, mirror_pts([(x + 60, y - 60) for x, y in HOOD]), [40, 60, 70, 60, 30], GREEN)
band(art, HOOD, [80, 120, 130, 110, 60], RED)
band(art, mirror_pts(HOOD), [80, 120, 130, 110, 60], RED)
d = ImageDraw.Draw(art)
for x0, x1, col in ((1590, 1670, GREEN), (1685, 1715, RED)):      # полосы по краям крыши
    d.rectangle([x0, 1180, x1, 2400], fill=col + (255,))
    d.rectangle([N - x1, 1180, N - x0, 2400], fill=col + (255,))
d.rectangle([1150, 180, 2950, 240], fill=RED + (255,))             # задний бампер
d.rectangle([1150, 255, 2950, 285], fill=GREEN + (255,))
band(art, [(1450, 3860), (2050, 3960), (2650, 3860)], [50, 70, 50], RED)   # передний бампер
band(art, [(1450, 3930), (2050, 4030), (2650, 3930)], [26, 36, 26], GREEN)

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
    t = Image.new("RGBA", (420, 64), (0, 0, 0, 0))
    d_ = ImageDraw.Draw(t)
    for k, c in enumerate([(245, 245, 245), (0, 57, 166), (213, 43, 30)]):
        d_.rectangle([6, 10 + k * 15, 50, 25 + k * 15], fill=c)
    d_.rectangle([6, 10, 50, 55], outline=BLACK, width=2)
    txt = fit("М. КОНОПЛЕНКО", 44, 350, fill=BLACK)
    t.alpha_composite(txt, (62, (64 - txt.height) // 2))
    return t


def place(layer, cx, cy, angle=0):
    if angle:
        layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
    full = Image.new("RGBA", img.size, (0, 0, 0, 0))
    full.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)))
    full.putalpha(ImageChops.multiply(full.getchannel("A"), body_mask))
    img.alpha_composite(full)


# борта: правая колонка читается снизу вверх (+90), левая — сверху вниз (−90)
for side, ang in ((1, 90), (-1, -90)):
    X = (lambda x: x) if side == 1 else (lambda x: N - x)
    place(castrol(170, 640), X(3205), 1400, ang)
    place(number_plate(300), X(3260), 2300, ang)
    place(name_plate(), X(3090), 1885, ang)

place(castrol(190, 700), 2050, 3040)                 # капот
place(number_plate(520), 2050, 1790)                 # крыша
place(castrol(150, 620), 2050, 990, 180)             # багажник (читается сзади)

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
    json.dump({"skinname": "Castrol GT500 tribute #86", "drivername": "Матвей Конопленко", "country": "Russia",
               "team": "Castrol GT500 tribute", "number": "86", "priority": 1}, fh, ensure_ascii=False, indent=2)
print("готово:", OUT)
