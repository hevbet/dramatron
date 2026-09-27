"""Скин «Арка × SimKart #64» для SMP Racing BR03 EVO (Assetto Corsa).

Берёт body_paint.dds из скина 00_SMPRacing_25_1, перекрашивает с сохранением
запечённых теней и наносит надписи по развёртке 4096×4096.

Запуск: python3 make_skin.py <путь к body_paint.dds исходного скина>
"""
import json
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

SRC = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "arka_simkart_64")
os.makedirs(OUT, exist_ok=True)

GRAPHITE = np.array([40, 43, 58], float)
MAGENTA = np.array([255, 43, 214], float)
CYAN = np.array([0, 229, 255], float)
VIOLET = (124, 58, 255)
WHITE = (242, 245, 255)
SILVER = (190, 198, 214)
BLACK = (10, 10, 14)

FB = "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"
FR = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

src = Image.open(SRC).convert("RGB")
N = src.width

# ---------------------------------------------------------------- 1. стираем старые логотипы
# (цвет заливки = фон под логотипом, дальше он перекрасится вместе со всем)
RED0, WHITE0, BLACK0 = (140, 5, 13), (214, 215, 214), (0, 0, 0)
erase = [
    ((150, 60, 420, 380), BLACK0),        # Yokohama / «Авто» вверху слева
    ((120, 510, 260, 650), RED0),         # эмблема на законцовке крыла
    ((480, 3300, 600, 3430), BLACK0),     # эмблема внизу
    ((160, 3560, 330, 3820), BLACK0),     # Yokohama внизу
    ((1300, 2850, 1370, 3010), BLACK0),   # BR03
    ((1990, 2870, 2100, 3000), BLACK0),   # орёл на носу
    ((2000, 3260, 2090, 3350), BLACK0),   # BR на носу
    ((1930, 1100, 2160, 1315), WHITE0),   # KRB на крыше
    ((1070, 880, 1245, 1380), WHITE0),    # ENGINEERING + BR слева
    ((1195, 1772, 1288, 1858), WHITE0),   # BR на номерной панели слева
    ((1670, 60, 2430, 148), WHITE0),      # ENGINEERING на крыле
]
mirror = []
for (x0, y0, x1, y1), c in erase:
    if x1 < 1900:
        mirror.append(((N - x1, y0, N - x0, y1), c))
# правая сторона не строго зеркальна — добавляем её логотипы явно
mirror += [((2800, 880, 2975, 1380), WHITE0),   # ENGINEERING + BR справа
           ((2752, 1782, 2842, 1878), WHITE0),  # BR на номерной панели справа
           ((2865, 1755, 2952, 1848), WHITE0),  # «E»
           ((2948, 1730, 3028, 1818), WHITE0),  # треугольник
           ((3600, 60, 3960, 400), BLACK0),     # Yokohama вверху справа
           ((3800, 530, 3970, 690), RED0),      # эмблема на законцовке справа
           ((3460, 3300, 3620, 3440), BLACK0),  # эмблема внизу справа
           ((3720, 3540, 3960, 3800), BLACK0)]  # Yokohama внизу справа
# заливаем каждую область ближайшим пикселем снаружи — так сохраняются тени и края деталей
hole = np.zeros((N, N), bool)
for (x0, y0, x1, y1), _ in erase + mirror:
    hole[y0:y1 + 1, x0:x1 + 1] = True
arr = np.asarray(src).copy()
_, (iy, ix) = ndimage.distance_transform_edt(hole, return_indices=True)
arr[hole] = arr[iy[hole], ix[hole]]
src = Image.fromarray(arr, "RGB")

# ---------------------------------------------------------------- 2. классы цветов и перекраска
a = np.asarray(src).astype(float)
r, g, b = a[..., 0], a[..., 1], a[..., 2]
mx, mn = a.max(-1), a.min(-1)
red = (r > 45) & (r > 2 * g + 25) & (r > b + 30)
blue = (b > 45) & (b > r + 30)
white = (mx > 60) & (mx - mn < 45) & ~red & ~blue

out = a.copy()
fr = np.clip(r / 145.0, 0, 1.15)[..., None]
fb = np.clip(b / 150.0, 0, 1.15)[..., None]
fw = np.clip(mx / 214.0, 0, 1.1)[..., None]
out[red] = (MAGENTA * fr)[red]
out[blue] = (CYAN * fb)[blue]
out[white] = (GRAPHITE * fw)[white]

# ретро-сетка по графиту
yy, xx = np.mgrid[0:N, 0:N]
grid = ((xx % 96) < 3) | ((yy % 96) < 3)
gm = grid & white
out[gm] = out[gm] * 0.6 + np.array(VIOLET, float) * 0.4 * fw[gm]

# белый пинстрайп на границе неона и графита
neon = red | blue
ring = ndimage.binary_dilation(neon, iterations=9) & ~neon & white
out[ring] = np.array(WHITE, float) * np.clip(fw[ring], 0.6, 1.0)

img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
white_mask = Image.fromarray((white * 255).astype(np.uint8), "L")


# ---------------------------------------------------------------- 3. надписи
def font(path, size):
    return ImageFont.truetype(path, size)


def text_layer(s, f, fill, stroke=0, stroke_fill=None):
    bbox = f.getbbox(s, stroke_width=stroke)
    w, h = bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(t).text((10 - bbox[0], 10 - bbox[1]), s, font=f, fill=fill,
                           stroke_width=stroke, stroke_fill=stroke_fill)
    return t


def place(layer, cx, cy, angle=0, clip=True):
    if angle:
        layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
    x, y = int(cx - layer.width / 2), int(cy - layer.height / 2)
    full = Image.new("RGBA", img.size, (0, 0, 0, 0))
    full.paste(layer, (x, y))
    if clip:
        al = Image.fromarray(np.minimum(np.asarray(full.getchannel("A")), np.asarray(white_mask)))
        full.putalpha(al)
    img.alpha_composite(full)


def sterlet_layer(L, color):
    """Стилизованная стерлядь — отсылка к гербу Саратова."""
    h = int(L * 0.32)
    t = Image.new("RGBA", (L + 4, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    cy = h / 2
    body = [(0, 0), (0.12, -0.07), (0.45, -0.10), (0.78, -0.05), (0.88, -0.02), (1.0, -0.14), (0.95, 0.0),
            (1.0, 0.12), (0.88, 0.02), (0.78, 0.05), (0.45, 0.10), (0.12, 0.06)]
    d.polygon([(2 + bx * L, cy + by * L) for bx, by in body], fill=color)
    for k in range(5):
        bx = 0.2 + k * 0.13
        d.polygon([(2 + (bx - 0.03) * L, cy - 0.08 * L), (2 + bx * L, cy - 0.13 * L),
                   (2 + (bx + 0.03) * L, cy - 0.08 * L)], fill=color)
    return t


def number_disc(d, fg, bg, ring, fsize):
    t = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    dr = ImageDraw.Draw(t)
    dr.ellipse([4, 4, d - 4, d - 4], fill=bg, outline=ring, width=max(6, d // 25))
    f = font(FB, fsize)
    dr.text((d / 2, d / 2 + fsize * 0.04), "64", font=f, fill=fg, anchor="mm")
    return t


# Боковины кокпита: буквы «вверх» = +x слева (поворот −90) и −x справа (поворот +90)
side = [
    ("САРАТОВСКАЯ ОБЛАСТЬ", font(FB, 44), SILVER, 0, 1296, 1000),
    ("SIMKART", font(FB, 200), WHITE, 5, 1200, 1000),
    ("АРКА", font(FB, 170), WHITE, 5, 1022, 1040),
    ("ЭНГЕЛЬССКИЙ ДОМ МОЛОДЁЖИ", font(FB, 42), SILVER, 0, 930, 1030),
]
for s, f, col, st, cx, cy in side:
    lay = text_layer(s, f, col, st, BLACK if st else None)
    place(lay, cx, cy, -90)
    place(lay, N - cx, cy, 90)

# «64» на номерных панелях у кокпита
lay = text_layer("64", font(FB, 160), WHITE, 6, (int(MAGENTA[0]), int(MAGENTA[1]), int(MAGENTA[2])))
place(lay, 1185, 1825, -90)
place(lay, N - 1195, 1835, 90)

# Крыша: номер в белом круге
place(number_disc(290, BLACK, WHITE + (255,), tuple(int(v) for v in MAGENTA), 190), 2045, 1130, 0)

# Антикрыло: верхняя грань и нижняя
place(text_layer("SIMKART  ×  АРКА", font(FB, 92), WHITE, 3, BLACK), 2045, 106, 0)
place(text_layer("ЭНГЕЛЬССКИЙ ДОМ МОЛОДЁЖИ · САРАТОВСКАЯ ОБЛАСТЬ", font(FB, 32), SILVER), 2045, 214, 0)

# Нос: стерлядь вместо орла, SIMKART вместо BR
place(sterlet_layer(300, SILVER + (255,)), 2045, 2935, 0, clip=False)
place(text_layer("SIMKART", font(FB, 58), WHITE), 2045, 3305, 0, clip=False)

img = img.convert("RGB")
img.save(os.path.join(HERE, "body_paint_preview.png"))


def save_dxt5(im, path):
    """Pillow пишет в заголовок неверные pitch и RGBBitCount — правим как в оригинальных DDS."""
    im.save(path, pixel_format="DXT5")
    with open(path, "r+b") as fh:
        fh.seek(20)
        fh.write(struct.pack("<I", max(1, im.width // 4) * max(1, im.height // 4) * 16))
        fh.seek(88)
        fh.write(struct.pack("<I", 0))


save_dxt5(img.convert("RGBA"), os.path.join(OUT, "body_paint.dds"))

# ---------------------------------------------------------------- 4. диски, иконка, ui_skin.json
save_dxt5(Image.new("RGBA", (4, 4), (22, 24, 32, 255)), os.path.join(OUT, "car_paint_rims.dds"))

ic = Image.new("RGB", (185, 185), (20, 22, 30))
di = ImageDraw.Draw(ic)
di.polygon([(0, 120), (185, 40), (185, 85), (0, 165)], fill=tuple(int(v) for v in MAGENTA))
di.polygon([(0, 165), (185, 85), (185, 110), (0, 185)], fill=tuple(int(v) for v in CYAN))
di.text((92, 88), "64", font=font(FB, 96), fill=WHITE, anchor="mm", stroke_width=4, stroke_fill=BLACK)
ic.save(os.path.join(OUT, "livery.png"))

with open(os.path.join(OUT, "ui_skin.json"), "w", encoding="utf-8") as fh:
    json.dump({"skinname": "Арка × SimKart #64", "country": "Russia", "drivername": "Стас",
               "team": "SimKart × Арка · Саратовская область", "number": "64", "priority": 1},
              fh, ensure_ascii=False, indent=2)
print("готово:", OUT)
