"""Скин «Арка | SimKart #64» для SMP Racing BR03 EVO (Assetto Corsa).

Левая половина развёртки (x < 2048) — в стиле тайм-кафе «Арка»,
правая — в стиле SimKart. Берёт body_paint.dds из скина 00_SMPRacing_25_1,
перекрашивает с сохранением запечённых теней и наносит логотипы
(векторизованы из референсов скриптом brand/trace.py).

Запуск: python3 make_skin.py <путь к body_paint.dds исходного скина>
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
BRAND = os.path.join(HERE, "brand")
OUT = os.path.join(HERE, "arka_simkart_64")
os.makedirs(OUT, exist_ok=True)

# --- палитры (сняты с референсов)
A_BASE = np.array([46, 46, 48], float)      # графит Арки
A_YELLOW = (240, 209, 102)
A_YELLOW_L = (247, 237, 178)
A_DARK = (26, 26, 28)
A_RED = (212, 88, 86)
A_TEAL = (43, 90, 107)
S_BASE = np.array([20, 20, 30], float)      # почти чёрный SimKart
S_RED = (234, 50, 38)
S_SILVER = (217, 216, 224)
WHITE = (242, 244, 250)
BLACK = (10, 10, 14)
AZURE = (30, 110, 190)                      # геральдическая лазурь
GOLD = (222, 178, 60)
HERALD_SILVER = (232, 236, 242)
# варианты герба: поле, кайма, стерляди, корона, центральный камень
COA_ARKA = dict(field=(46, 46, 48), border=A_YELLOW, fish=A_YELLOW, crown=A_YELLOW, gem=A_RED, pearl=A_YELLOW_L)
COA_SIMKART = dict(field=(20, 20, 30), border=S_RED, fish=S_SILVER, crown=S_SILVER, gem=S_RED, pearl=S_RED)

FONTS = os.path.join(HERE, "brand", "fonts")
F_RACE = os.path.join(FONTS, "Exo2-Italic[wght].ttf")    # один гоночный шрифт на всю машину

src = Image.open(SRC).convert("RGB")
N = src.width
HALF = N // 2

# ---------------------------------------------------------------- 1. стираем старые логотипы
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
mirror = [((N - x1, y0, N - x0, y1), c) for (x0, y0, x1, y1), c in erase if x1 < 1900]
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

# ---------------------------------------------------------------- 2. классы цветов и перекраска
a = arr.astype(float)
r, g, b = a[..., 0], a[..., 1], a[..., 2]
mx, mn = a.max(-1), a.min(-1)
red = (r > 45) & (r > 2 * g + 25) & (r > b + 30)
blue = (b > 45) & (b > r + 30)
white = (mx > 60) & (mx - mn < 45) & ~red & ~blue
fr = np.clip(r / 145.0, 0, 1.15)[..., None]
fb = np.clip(b / 150.0, 0, 1.15)[..., None]
fw = np.clip(mx / 214.0, 0, 1.1)[..., None]

L = (np.arange(N)[None, :] < HALF).repeat(N, 0)   # половина Арки
R = ~L                                           # половина SimKart


def arka_pattern():
    """Фирменная лента Арки: ряды «арка» жёлтым на тёмном, под наклоном."""
    word = Image.open(os.path.join(BRAND, "arka_word.png"))
    h = 90
    word = word.resize((int(word.width * h / word.height), h), Image.LANCZOS)
    tile = Image.new("RGB", word.size, A_YELLOW)
    big = Image.new("RGB", (int(N * 1.3), int(HALF * 1.8)), (30, 30, 32))
    y, row = 0, 0
    while y < big.height:
        x = -((row * 97) % word.width)
        while x < big.width:
            big.paste(tile, (x, y), word)
            x += word.width + 8
        y += h + 14
        row += 1
    big = big.rotate(-8, resample=Image.BICUBIC)   # лёгкий наклон, как на афише
    cx, cy = big.width // 2, big.height // 2
    band = big.crop((cx - N // 2, cy - HALF // 2, cx + N // 2, cy + HALF - HALF // 2))
    # на левых деталях развёртки «вверх» кузова = +x, текст читается сверху вниз (как было ENGINEERING)
    return np.asarray(band.rotate(-90, expand=True)).astype(float)


out = a.copy()
# --- Арка: графит с размытыми пятнами, жёлтый вместо красного, лента «арка» вместо синего
pat = np.zeros_like(a)
pat[:, :HALF] = arka_pattern()
m = white & L
out[m] = (A_BASE * fw)[m]
m = red & L
out[m] = (np.array(A_YELLOW, float) * fr)[m]
m = blue & L
out[m] = (pat * np.clip(fb, 0.55, 1.1))[m]
# --- SimKart: почти чёрный с красным оттенком, красный, серебро вместо синего
m = white & R
out[m] = (S_BASE * fw + np.array([24, 0, 0], float) * fw)[m]
m = red & R
out[m] = (np.array(S_RED, float) * fr)[m]
m = blue & R
out[m] = (np.array(S_SILVER, float) * fb)[m]

# окантовка на стыке цвета и базы: у Арки тёмная обводка + светло-жёлтая линия, у SimKart — красная
neon = red | blue
ring_in = ndimage.binary_dilation(neon, iterations=5) & ~neon & white
ring_out = ndimage.binary_dilation(neon, iterations=12) & ~neon & white & ~ring_in
out[ring_in & L] = np.array(A_DARK, float)
out[ring_out & L] = (np.array(A_YELLOW_L, float) * np.clip(fw, 0.6, 1.0))[ring_out & L]
m = (ring_in | ring_out) & R
out[m] = (np.array(S_RED, float) * np.clip(fw, 0.6, 1.0))[m]

# стык половин по оси машины: жёлтый кант | шахматка | красный кант
shade = np.where(white[..., None], fw, np.where(red[..., None], fr, np.where(blue[..., None], fb, 0.8)))
SW = [(-34, -24, A_YELLOW), (-24, -20, A_DARK), (20, 24, A_DARK), (24, 34, S_RED)]
for x0, x1, col in SW:
    out[:, HALF + x0:HALF + x1] = (np.array(col, float) * np.clip(shade[:, HALF + x0:HALF + x1], 0.5, 1.05))
cy_, cx_ = np.mgrid[0:N, HALF - 20:HALF + 20]
chk = (((cy_ // 20) + ((cx_ - (HALF - 20)) // 20)) % 2 == 0)[..., None]
cell = np.where(chk, np.array(WHITE, float), np.array(A_DARK, float))
out[:, HALF - 20:HALF + 20] = cell * np.clip(shade[:, HALF - 20:HALF + 20], 0.5, 1.05)

img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
white_mask = Image.fromarray((white * 255).astype(np.uint8), "L")


# ---------------------------------------------------------------- 3. графика
def font(path, size, weight="Black Italic"):
    f = ImageFont.truetype(path, size)
    if path == F_RACE:
        f.set_variation_by_name(weight)
    return f


def spaced(s, f, fill, track=0.12, stroke=0, stroke_fill=None):
    """Строка с разрядкой — классическая подпись на ливрее."""
    gap = f.size * track
    w = int(sum(f.getlength(ch) for ch in s) + gap * (len(s) - 1)) + 20 + 2 * stroke
    asc, desc = f.getmetrics()
    t = Image.new("RGBA", (w, asc + desc + 20 + 2 * stroke), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    x = 10 + stroke
    for ch in s:
        d.text((x, 10 + stroke), ch, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        x += f.getlength(ch) + gap
    return t.crop(t.getbbox())


def label(s, size, maxw, fill, weight="ExtraBold Italic"):
    """Подпись Exo 2 с разрядкой; кегль уменьшается, пока строка не влезет в maxw."""
    while size > 10:
        t = spaced(s, font(F_RACE, size, weight), fill)
        if t.width <= maxw:
            return t
        size -= 1
    return t


def race_number(fill, outline, shadow, size=150):
    """Гоночный номер: Exo 2 Black Italic, обводка и смещённая тень."""
    f = font(F_RACE, size)
    m = text_layer("64", f, (255, 255, 255)).getchannel("A")
    pad = int(size * 0.2)
    big = Image.new("L", (m.width + 2 * pad, m.height + 2 * pad), 0)
    big.paste(m, (pad, pad))
    ol = big.filter(ImageFilter.MaxFilter(max(3, int(size * 0.07)) | 1))
    sh = ImageChops.offset(ol, -int(size * 0.04), int(size * 0.06))
    t = Image.new("RGBA", big.size, (0, 0, 0, 0))
    t.alpha_composite(solid(sh, shadow))
    t.alpha_composite(solid(ol, outline))
    t.alpha_composite(solid(big, fill))
    return t


def load_mask(name, h):
    m = Image.open(os.path.join(BRAND, name)).convert("L")
    return m.resize((max(1, int(m.width * h / m.height)), h), Image.LANCZOS)


def solid(mask, col):
    t = Image.new("RGBA", mask.size, col + (255,))
    t.putalpha(mask)
    return t


def arka_mark(mask, fill=A_YELLOW):
    """Стиль логотипа Арки: жёлтая заливка, тёмная обводка, тень вниз-влево."""
    pad = int(mask.height * 0.16) + 4
    big = Image.new("L", (mask.width + 2 * pad, mask.height + 2 * pad), 0)
    big.paste(mask, (pad, pad))
    ow = min(59, max(3, int(mask.height * 0.05)) | 1)
    outline = big.filter(ImageFilter.MaxFilter(ow))
    sh = ImageChops.offset(outline, -int(mask.height * 0.035), int(mask.height * 0.06))
    t = Image.new("RGBA", big.size, (0, 0, 0, 0))
    t.alpha_composite(solid(sh, A_DARK))
    t.alpha_composite(solid(outline, A_DARK))
    t.alpha_composite(solid(big, fill))
    return t


def simkart_mark(h):
    """«Сим» серебром с градиентом, «карт» красным, красное свечение."""
    w = load_mask("simkart_word_white.png", h)
    rr = load_mask("simkart_word_red.png", h)
    pad = int(h * 0.3)
    size = (w.width + 2 * pad, w.height + 2 * pad)
    t = Image.new("RGBA", size, (0, 0, 0, 0))
    both = Image.new("L", size, 0)
    both.paste(ImageChops.lighter(w, rr), (pad, pad))
    t.alpha_composite(solid(both.filter(ImageFilter.GaussianBlur(h * 0.12)).point(lambda v: int(v * 0.7)), S_RED))
    grad = Image.linear_gradient("L").resize(w.size)
    sil = Image.merge("RGB", [grad.point(lambda v, c=c: int(255 - (255 - c) * v / 255)) for c in S_SILVER]).convert("RGBA")
    sil.putalpha(w)
    t.alpha_composite(sil, (pad, pad))
    t.alpha_composite(solid(rr, S_RED), (pad, pad))
    return t


def text_layer(s, f, fill, stroke=0, stroke_fill=None):
    bbox = f.getbbox(s, stroke_width=stroke)
    t = Image.new("RGBA", (bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20), (0, 0, 0, 0))
    ImageDraw.Draw(t).text((10 - bbox[0], 10 - bbox[1]), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return t


def place(layer, cx, cy, angle=0, clip=True):
    if angle:
        layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
    full = Image.new("RGBA", img.size, (0, 0, 0, 0))
    full.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)))
    if clip:
        full.putalpha(Image.fromarray(np.minimum(np.asarray(full.getchannel("A")), np.asarray(white_mask))))
    img.alpha_composite(full)


def sterlet(Lg, color):
    """Стерлядь: рыло слева (x=0), хвост справа."""
    h = int(Lg * 0.32)
    t = Image.new("RGBA", (Lg + 4, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    cy = h / 2
    body = [(0, 0), (0.12, -0.07), (0.45, -0.10), (0.78, -0.05), (0.88, -0.02), (1.0, -0.14), (0.95, 0.0),
            (1.0, 0.12), (0.88, 0.02), (0.78, 0.05), (0.45, 0.10), (0.12, 0.06)]
    d.polygon([(2 + bx * Lg, cy + by * Lg) for bx, by in body], fill=color)
    for k in range(5):
        bx = 0.2 + k * 0.13
        d.polygon([(2 + (bx - 0.03) * Lg, cy - 0.08 * Lg), (2 + bx * Lg, cy - 0.13 * Lg),
                   (2 + (bx + 0.03) * Lg, cy - 0.08 * Lg)], fill=color)
    return t


def coat_split(W):
    """Герб пополам: левая половина в цветах Арки, правая — SimKart."""
    left, right = coat_of_arms(W, **COA_ARKA), coat_of_arms(W, **COA_SIMKART)
    out = right.copy()
    out.paste(left.crop((0, 0, W // 2, left.height)), (0, 0))
    return out


def coat_of_arms(W, field=AZURE, border=GOLD, fish=HERALD_SILVER, crown=GOLD, gem=(200, 40, 50), pearl=HERALD_SILVER):
    """Герб Саратовской области (стилизация): лазоревый щит, три серебряные стерляди
    в вилообразный крест головами к центру, золотая кайма и корона."""
    S = 4                       # суперсэмплинг
    w = W * S
    h = int(w * 1.45)
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    top = int(h * 0.33)         # место под корону

    def shield(inset):
        x0, x1, y0 = inset, w - inset, top + inset
        yb = h - inset
        pts = [(x0, y0), (x1, y0), (x1, y0 + (yb - y0) * 0.62)]
        cx = w / 2
        for k in range(1, 21):  # скруглённый низ с остриём (французский щит)
            tt = k / 20
            pts.append((x1 - (x1 - cx) * tt, y0 + (yb - y0) * (0.62 + 0.38 * math.sin(tt * math.pi / 2))))
        pts += [(cx - (p[0] - cx), p[1]) for p in reversed(pts[3:-1])]
        pts.append((x0, y0 + (yb - y0) * 0.62))
        return pts

    d.polygon(shield(0), fill=A_DARK)
    d.polygon(shield(int(w * 0.03)), fill=border)
    d.polygon(shield(int(w * 0.075)), fill=field)
    # три стерляди: вилообразный крест, головы к центру
    cx, cy = w / 2, top + (h - top) * 0.44
    fish_im = sterlet(int(w * 0.42), fish + (255,))
    for ang in (210, 330, 90):          # вверх-влево, вверх-вправо, вниз (y вниз)
        rot = fish_im.rotate(-ang, expand=True, resample=Image.BICUBIC)
        dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        c = fish_im.width / 2 + w * 0.02   # центр рыбы смещён наружу, рыло у центра
        t.alpha_composite(rot, (int(cx + dx * c - rot.width / 2), int(cy + dy * c - rot.height / 2)))
    # корона (упрощённая императорская): обруч, две полусферы, центральная дуга, держава с крестом
    ow = S * 2
    cb = top - int(h * 0.012)                  # низ обруча
    ch = h * 0.1                               # радиус полусфер по высоте
    d.chord([w * 0.20, cb - ch * 2, w * 0.49, cb + ch * 0.1], 180, 360, fill=crown, outline=A_DARK, width=ow)
    d.chord([w * 0.51, cb - ch * 2, w * 0.80, cb + ch * 0.1], 180, 360, fill=crown, outline=A_DARK, width=ow)
    d.rounded_rectangle([w * 0.465, cb - ch * 1.08, w * 0.535, cb], radius=int(w * 0.02), fill=crown, outline=A_DARK, width=ow)
    orb_r = w * 0.045
    oy = cb - ch * 1.08 - orb_r * 0.8
    d.ellipse([w / 2 - orb_r, oy - orb_r, w / 2 + orb_r, oy + orb_r], fill=crown, outline=A_DARK, width=ow)
    cw = w * 0.022
    d.rectangle([w / 2 - cw / 2, oy - orb_r * 3.0, w / 2 + cw / 2, oy - orb_r * 0.8], fill=crown, outline=A_DARK, width=ow)
    d.rectangle([w / 2 - cw * 2, oy - orb_r * 2.4, w / 2 + cw * 2, oy - orb_r * 2.4 + cw], fill=crown, outline=A_DARK, width=ow)
    d.rounded_rectangle([w * 0.18, cb - h * 0.05, w * 0.82, cb], radius=int(w * 0.02), fill=crown, outline=A_DARK, width=ow)
    cr = w * 0.03
    for fx in (0.27, 0.38, 0.5, 0.62, 0.73):   # камни на обруче
        d.ellipse([w * fx - cr / 2, cb - h * 0.025 - cr / 2, w * fx + cr / 2, cb - h * 0.025 + cr / 2],
                  fill=gem if fx == 0.5 else pearl)
    return t.resize((W, int(h / S)), Image.LANCZOS)


# ЛЕВАЯ боковина — Арка (буквы «вверх» = +x → поворот −90)
place(arka_mark(load_mask("arka_word.png", 240)), 1115, 1000, -90)
place(arka_mark(load_mask("arka_taimcafe.png", 64)), 945, 1000, -90)
place(label("МОСКОВСКАЯ 56 · САРАТОВ", 30, 700, A_YELLOW_L), 1285, 1000, -90)

# ПРАВАЯ боковина — SimKart (буквы «вверх» = −x → поворот +90)
place(simkart_mark(150), N - 1165, 1010, 90)
place(solid(load_mask("simkart_tagline.png", 26), S_RED), N - 950, 1000, 90)
place(label("ЭНГЕЛЬССКИЙ ДОМ МОЛОДЁЖИ", 30, 700, S_SILVER), N - 1285, 1000, 90)
place(coat_of_arms(120, **COA_ARKA), 1130, 500, -90)
place(coat_of_arms(120, **COA_SIMKART), N - 1130, 500, 90)

# Номерные панели: слева — в стиле Арки, справа — SimKart
place(race_number(A_YELLOW, A_DARK, A_DARK, 125), 1185, 1825, -90)
place(race_number(WHITE, S_RED, A_DARK, 125), N - 1195, 1835, 90)

# Крыша: номер в круге, обод пополам — жёлтый / красный
disc = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
dd = ImageDraw.Draw(disc)
dd.pieslice([4, 4, 296, 296], 90, 270, fill=A_YELLOW)
dd.pieslice([4, 4, 296, 296], -90, 90, fill=S_RED)
dd.ellipse([24, 24, 276, 276], fill=WHITE)
dd.text((146, 150), "64", font=font(F_RACE, 165), fill=BLACK, anchor="mm")
place(disc, 2045, 1130)

# Антикрыло: верх — «арка» слева, «Симкарт» справа; низ — ЭДМ и регион
place(arka_mark(load_mask("arka_word.png", 78)), 1790, 106)
place(simkart_mark(62), 2310, 106)
place(label("ЭНГЕЛЬССКИЙ ДОМ МОЛОДЁЖИ", 30, 440, A_YELLOW_L), 1780, 214)
place(label("САРАТОВСКАЯ ОБЛАСТЬ · 64", 30, 440, S_SILVER), 2315, 214)

# Нос: герб Саратовской области вместо орла, под ним оба логотипа
place(coat_split(170), 2045, 2925, 0, clip=False)
place(arka_mark(load_mask("arka_word.png", 56)), 1960, 3305, 0, clip=False)
place(simkart_mark(40), 2145, 3305, 0, clip=False)

img = img.convert("RGB")
img.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, "body_paint_preview.png"))
prev = Image.new("RGBA", (1880, 900), (12, 12, 16, 255))
for k, c in enumerate([coat_of_arms(600, **COA_ARKA), coat_split(600), coat_of_arms(600, **COA_SIMKART)]):
    prev.alpha_composite(c, (20 + k * 620, 15))
prev.convert("RGB").save(os.path.join(HERE, "coat_of_arms_preview.png"))


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

ic = Image.new("RGB", (185, 185), A_DARK)
di = ImageDraw.Draw(ic)
di.rectangle([0, 0, 92, 185], fill=tuple(int(v) for v in A_BASE))
di.rectangle([93, 0, 185, 185], fill=tuple(int(v) for v in S_BASE))
di.polygon([(0, 150), (92, 120), (92, 140), (0, 170)], fill=A_YELLOW)
di.polygon([(93, 120), (185, 90), (185, 110), (93, 140)], fill=S_RED)
di.text((92, 80), "64", font=font(F_RACE, 96), fill=WHITE, anchor="mm", stroke_width=4, stroke_fill=BLACK)
ic.save(os.path.join(OUT, "livery.png"))

with open(os.path.join(OUT, "ui_skin.json"), "w", encoding="utf-8") as fh:
    json.dump({"skinname": "Арка | SimKart #64", "country": "Russia", "drivername": "Стас",
               "team": "Арка × SimKart · Саратовская область", "number": "64", "priority": 1},
              fh, ensure_ascii=False, indent=2)
print("готово:", OUT)
