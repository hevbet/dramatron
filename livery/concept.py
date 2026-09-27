"""Концепт ливреи «Арка × SimKart · Саратовская область 64» — вид сбоку и сверху."""
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import math

S = 2  # суперсэмплинг
W, H = 1600 * S, 1250 * S
OUT = "/home/user/dramatron/livery/concept_arka_simkart.png"

BG = (8, 9, 14)
BASE = (20, 22, 30)
CYAN = (0, 229, 255)
MAGENTA = (255, 43, 214)
VIOLET = (124, 58, 255)
WHITE = (240, 244, 255)
SILVER = (190, 198, 214)

FB = "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"
FR = "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"


def font(path, size):
    return ImageFont.truetype(path, int(size * S))


def p(x, y):
    return (x * S, y * S)


def catmull(points, n=24):
    out = []
    pts = [points[0]] + points + [points[-1]]
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(tuple(
                0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                       + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    out.append(points[-1])
    return out


def arc(cx, cy, r, a0, a1, n=40):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]


def gradient(size, c0, c1, horizontal=True):
    w, h = size
    g = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(g)
    steps = w if horizontal else h
    for i in range(steps):
        t = i / max(1, steps - 1)
        c = tuple(int(c0[j] + (c1[j] - c0[j]) * t) for j in range(3))
        if horizontal:
            d.line([(i, 0), (i, h)], fill=c)
        else:
            d.line([(0, i), (w, i)], fill=c)
    return g


def mask_poly(pts):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon([p(*q) for q in pts], fill=255)
    return m


def paste_clipped(canvas, layer, mask):
    canvas.paste(layer, (0, 0), mask)


def text_center(d, xy, s, f, fill, anchor="mm", **kw):
    d.text(p(*xy), s, font=f, fill=fill, anchor=anchor, **kw)


def glow(canvas, layer_rgba, radius=10):
    blur = layer_rgba.filter(ImageFilter.GaussianBlur(radius * S))
    canvas.alpha_composite(blur)
    canvas.alpha_composite(layer_rgba)


def sterlet(d, x, y, L, color, flip=False):
    """Стилизованная стерлядь (герб Саратова): длинное тело, острое рыло, хвост."""
    sgn = -1 if flip else 1
    body = [(0, 0), (0.12, -0.07), (0.45, -0.10), (0.78, -0.05), (0.88, -0.02),
            (1.0, -0.14), (0.95, 0.0), (1.0, 0.12), (0.88, 0.02), (0.78, 0.05),
            (0.45, 0.10), (0.12, 0.06)]
    d.polygon([p(x + sgn * bx * L, y + by * L) for bx, by in body], fill=color)
    for k in range(5):  # жучки на спине
        bx = 0.2 + k * 0.13
        d.polygon([p(x + sgn * (bx - 0.03) * L, y - 0.08 * L), p(x + sgn * bx * L, y - 0.13 * L),
                   p(x + sgn * (bx + 0.03) * L, y - 0.08 * L)], fill=color)


# ---------------------------------------------------------------- фон
img = Image.new("RGBA", (W, H), BG + (255,))
d = ImageDraw.Draw(img)
bgg = gradient((W, H), (18, 10, 32), BG, horizontal=False)
img.paste(bgg)
d = ImageDraw.Draw(img)

text_center(d, (800, 55), "АРКА × SIMKART  ·  САРАТОВСКАЯ ОБЛАСТЬ  ·  #64", font(FB, 34), WHITE)
text_center(d, (800, 95), "концепт ливреи для Assetto Corsa — вид сбоку и сверху", font(FR, 20), SILVER)

# =============================================================== ВИД СБОКУ
GY = 560  # линия пола
WR = (340, 1260)  # центры колёс
WY, R = 490, 70

upper = catmull([(1370, 520), (1385, 450), (1345, 405), (1080, 375), (960, 300), (800, 262),
                 (590, 265), (470, 305), (360, 370), (250, 372), (225, 395), (218, 470), (232, 525)])
bottom = [(232, 525), (262, 525)] + arc(WR[0], WY, R + 12, 180, 360) + [(418, 525), (1182, 525)] \
         + arc(WR[1], WY, R + 12, 180, 360) + [(1338, 525), (1370, 520)]
body_pts = upper + bottom
body_mask = mask_poly(body_pts)

glass_pts = catmull([(1000, 382), (955, 318), (800, 284), (610, 286), (500, 318), (420, 372)]) + [(700, 382)]
glass_mask = mask_poly(glass_pts)
paint_mask = ImageChops.subtract(body_mask, glass_mask)

# тень под машиной
sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ImageDraw.Draw(sh).ellipse([p(200, GY - 22), p(1400, GY + 22)], fill=(0, 0, 0, 200))
img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(14 * S)))
floor = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ImageDraw.Draw(floor).ellipse([p(250, GY - 6), p(1350, GY + 10)], fill=MAGENTA + (70,))
img.alpha_composite(floor.filter(ImageFilter.GaussianBlur(20 * S)))

# база: графит
base = gradient((W, H), (38, 40, 52), (12, 13, 18), horizontal=False).convert("RGBA")
img.paste(base, (0, 0), paint_mask)

# ретро-сетка на задней части (ретровейв)
grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
gd = ImageDraw.Draw(grid)
hx, hy = 700, 330
for k in range(-6, 12):
    gd.line([p(hx, hy), p(hx - 700 + k * 60, 620)], fill=VIOLET + (90,), width=2 * S)
for k in range(9):
    y = 360 + (k ** 1.6) * 7
    gd.line([p(150, y), p(900, y)], fill=VIOLET + (90,), width=2 * S)
fade = gradient((W, H), (255, 255, 255), (0, 0, 0)).convert("L")
fade = fade.point(lambda v: 255 if v > 150 else int(v * 1.7))
grid.putalpha(ImageChops.multiply(ImageChops.multiply(grid.getchannel("A"), fade), paint_mask))
img.alpha_composite(grid)

# неоновая волна: от переднего бампера вверх к корме
wave_top = catmull([(1380, 470), (1180, 455), (950, 438), (720, 412), (480, 392), (200, 395)])
wave_bot = catmull([(200, 425), (480, 425), (720, 448), (950, 472), (1180, 490), (1380, 500)])
wave_mask = ImageChops.multiply(mask_poly(wave_top + wave_bot), paint_mask)
wave = gradient((W, H), MAGENTA, CYAN).convert("RGBA")
wl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
wl.paste(wave, (0, 0), wave_mask)
glow(img, wl, 8)

# тонкие пинстрайпы над волной
pin = Image.new("RGBA", (W, H), (0, 0, 0, 0))
pd = ImageDraw.Draw(pin)
for off, col in ((-12, CYAN), (-22, MAGENTA)):
    line = catmull([(1380, 470 + off), (1180, 455 + off), (950, 438 + off), (720, 412 + off),
                    (480, 392 + off), (200, 395 + off)])
    pd.line([p(*q) for q in line], fill=col + (255,), width=3 * S)
pin.putalpha(ImageChops.multiply(pin.getchannel("A"), paint_mask))
glow(img, pin, 5)

# стекло
glass = gradient((W, H), (30, 40, 60), (6, 8, 14), horizontal=False).convert("RGBA")
img.paste(glass, (0, 0), glass_mask)
d = ImageDraw.Draw(img)
text_center(d, (560, 305), "arkacafe.ru", font(FB, 17), WHITE)
d.line([p(705, 290), p(705, 380)], fill=BASE, width=8 * S)  # стойка

# текстовые слои (обрезаем по кузову)
tx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
td = ImageDraw.Draw(tx)
# номер на двери
td.rounded_rectangle([p(735, 395), p(895, 470)], radius=14 * S, fill=(10, 10, 16, 255),
                     outline=CYAN + (255,), width=3 * S)
text_center(td, (815, 433), "64", font(FB, 64), WHITE, stroke_width=2 * S, stroke_fill=MAGENTA)
# SIMKART на переднем крыле / двери
text_center(td, (1110, 420), "SIMKART", font(FB, 44), WHITE, stroke_width=2 * S, stroke_fill=(0, 0, 0))
# АРКА на заднем крыле
text_center(td, (555, 405), "АРКА", font(FB, 48), WHITE, stroke_width=2 * S, stroke_fill=(0, 0, 0))
text_center(td, (555, 470), "компьютерный клуб · Московская 56", font(FB, 13), WHITE)
# порог
text_center(td, (800, 512), "ЭНГЕЛЬССКИЙ ДОМ МОЛОДЁЖИ  ·  САРАТОВСКАЯ ОБЛАСТЬ", font(FB, 14), SILVER)
# стерляди на переднем крыле
sterlet(td, 900, 492, 120, SILVER + (255,), flip=True)
text_center(td, (940, 492), "САРАТОВ · 64", font(FB, 13), SILVER, anchor="lm")
tx.putalpha(ImageChops.multiply(tx.getchannel("A"), paint_mask))
img.alpha_composite(tx)
d = ImageDraw.Draw(img)

# линии панелей, фары
d.line([p(1000, 382), p(1015, 520)], fill=(0, 0, 0), width=2 * S)
d.line([p(700, 382), p(690, 520)], fill=(0, 0, 0), width=2 * S)
d.rounded_rectangle([p(955, 400), p(985, 406)], radius=3 * S, fill=(60, 64, 80))
d.polygon([p(1330, 412), p(1378, 425), p(1382, 445), p(1340, 438)], fill=WHITE)
d.polygon([p(222, 392), p(250, 390), p(248, 415), p(221, 415)], fill=(255, 30, 60))
# антикрыло
d.polygon([p(210, 330), p(330, 330), p(330, 342), p(210, 346)], fill=(10, 10, 14))
d.line([p(290, 342), p(300, 372)], fill=(10, 10, 14), width=6 * S)
d.line([p(210, 330), p(330, 330)], fill=CYAN, width=2 * S)

# колёса
for cx in WR:
    d.ellipse([p(cx - R, WY - R), p(cx + R, WY + R)], fill=(12, 12, 14))
    d.ellipse([p(cx - 48, WY - 48), p(cx + 48, WY + 48)], fill=(28, 30, 38), outline=CYAN, width=3 * S)
    for a in range(0, 360, 36):
        d.line([p(cx, WY), p(cx + 44 * math.cos(math.radians(a)), WY + 44 * math.sin(math.radians(a)))],
               fill=(70, 74, 90), width=5 * S)
    d.ellipse([p(cx - 10, WY - 10), p(cx + 10, WY + 10)], fill=MAGENTA)

# =============================================================== ВИД СВЕРХУ
TY = 900  # ось симметрии
top_half = catmull([(230, TY), (235, TY - 95), (300, TY - 118), (700, TY - 125), (1100, TY - 122),
                    (1330, TY - 105), (1385, TY - 60), (1392, TY)])
top_pts = top_half + [(x, 2 * TY - y) for x, y in reversed(top_half)]
top_mask = mask_poly(top_pts)
roof_pts = catmull([(470, TY - 88), (560, TY - 100), (900, TY - 100), (990, TY - 90)]) + \
           catmull([(990, TY + 90), (900, TY + 100), (560, TY + 100), (470, TY + 88)])
ws_pts = [(990, TY - 90), (1080, TY - 100), (1080, TY + 100), (990, TY + 90)]
rw_pts = [(470, TY - 88), (390, TY - 96), (390, TY + 96), (470, TY + 88)]
glass2 = ImageChops.add(mask_poly(ws_pts), mask_poly(rw_pts))
paint2 = ImageChops.subtract(top_mask, glass2)

sh2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ImageDraw.Draw(sh2).polygon([p(x + 10, y + 14) for x, y in top_pts], fill=(0, 0, 0, 180))
img.alpha_composite(sh2.filter(ImageFilter.GaussianBlur(12 * S)))
img.paste(gradient((W, H), (30, 32, 42), (18, 19, 26)).convert("RGBA"), (0, 0), paint2)

# две продольные неоновые полосы (градиент по длине)
stripes = Image.new("L", (W, H), 0)
sd = ImageDraw.Draw(stripes)
for yc in (TY - 40, TY + 40):
    sd.rectangle([p(230, yc - 14), p(1400, yc + 14)], fill=255)
stripes = ImageChops.multiply(stripes, paint2)
sl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
sl.paste(gradient((W, H), MAGENTA, CYAN).convert("RGBA"), (0, 0), stripes)
glow(img, sl, 6)

# стекло
img.paste(gradient((W, H), (6, 8, 14), (30, 40, 60)).convert("RGBA"), (0, 0), glass2)
d = ImageDraw.Draw(img)
text_center(d, (1035, TY), "SIMKART", font(FB, 18), WHITE, direction=None)

t2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
t2d = ImageDraw.Draw(t2)
# крыша: номер в круге
t2d.ellipse([p(655, TY - 75), p(805, TY + 75)], fill=(245, 245, 250, 255), outline=MAGENTA + (255,), width=5 * S)
text_center(t2d, (730, TY + 2), "64", font(FB, 96), (10, 10, 16))
# капот: SIMKART
text_center(t2d, (1240, TY), "SIMKART", font(FB, 44), WHITE, stroke_width=2 * S, stroke_fill=(0, 0, 0))
# багажник: АРКА
text_center(t2d, (310, TY), "АРКА", font(FB, 40), WHITE, stroke_width=2 * S, stroke_fill=(0, 0, 0))
# крыша: стерляди по бокам от номера
sterlet(t2d, 520, TY - 45, 110, SILVER + (255,))
sterlet(t2d, 940, TY + 45, 110, SILVER + (255,), flip=True)
t2.putalpha(ImageChops.multiply(t2.getchannel("A"), paint2))
img.alpha_composite(t2)
d = ImageDraw.Draw(img)
d.polygon([p(1375, TY - 70), p(1392, TY - 40), p(1388, TY - 70)], fill=WHITE)
d.polygon([p(1375, TY + 70), p(1392, TY + 40), p(1388, TY + 70)], fill=WHITE)

# =============================================================== палитра
px = 230
for name, col in (("графит", BASE), ("циан", CYAN), ("маджента", MAGENTA), ("фиолет", VIOLET), ("серебро", SILVER)):
    d.rounded_rectangle([p(px, 1120), p(px + 44, 1164)], radius=8 * S, fill=col, outline=(80, 80, 100), width=S)
    hexs = "#%02X%02X%02X" % col
    d.text(p(px + 54, 1128), name, font=font(FB, 15), fill=WHITE)
    d.text(p(px + 54, 1148), hexs, font=font(FR, 13), fill=SILVER)
    px += 230

img = img.convert("RGB").resize((W // S, H // S), Image.LANCZOS)
img.save(OUT)
print(OUT)
