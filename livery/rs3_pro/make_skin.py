"""Скин Pozdnyakov_26 для Audi RS3 LMS — ливрея «ГРАНЬ» (как BR03 Pozdnyakov_64).

Графитовая база, сквозная красная лента с жёлтой кромкой по бортам, капоту, крыше и багажнику.
Опоры: тайм-кафе «Арка», «Симкарт», Саратовская область (флаг на номерах, герб, надпись на порогах),
команда ЭДМ, SMP Racing Esports, РАФ, DriveOil. Геометрия развёртки RS3 — как в ../rs3/make_skin.py.

Запуск: python3 make_skin.py <папка исходного скина SMP01>
"""
import json
import math
import os
import sys
import zipfile

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "br03_pro"))
import lib as blib  # noqa: E402  бренд-графика, шрифт, DriveOil, save_dxt5

SRC = sys.argv[1]
SKIN = "Pozdnyakov_26"          # правило: фамилия латиницей_номер
NUM = "26"
OUT = os.path.join(HERE, SKIN)
os.makedirs(OUT, exist_ok=True)
BX = os.path.join(HERE, "..", "br03_pro", "brand_extra")

hexc = blib.hexc
BASE = np.array(hexc("#1B1C22"), float)
DEEP, INK = hexc("#0F1014"), hexc("#15161B")
YELLOW, RED, AZURE, WHITE = hexc("#F0D166"), hexc("#E5342C"), hexc("#1694DC"), hexc("#F4F5F7")
CYAN = hexc("#00B5EF")

src = Image.open(os.path.join(SRC, "Skin.dds")).convert("RGB")
N = src.width
a = np.asarray(src).astype(float)
mx = a.max(-1)

# ---------------------------------------------------------------- 1. стираем логотипы SMP, оставляем кольца Audi
logo = mx > 110
rings = np.zeros_like(logo)
rings[720:790, 1950:2140] = True
logo &= ~rings
logo = ndimage.binary_dilation(logo, iterations=6)
_, (iy, ix) = ndimage.distance_transform_edt(logo, return_indices=True)
a[logo] = a[iy[logo], ix[logo]]
mx = a.max(-1)

# ---------------------------------------------------------------- 2. графитовая база с запечёнными тенями
body = (a[..., 2] > a[..., 0] + 12) & (mx > 18)
shade = np.clip(mx / 76.0, 0, 1.12)[..., None]
out = a.copy()
out[body] = (BASE * (0.55 + 0.45 * shade))[body]
out[rings] = np.asarray(src).astype(float)[rings]
body_mask = Image.fromarray((body * 255).astype(np.uint8), "L")
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


def ribbon(layer, pts, widths, col, offset=0.0):
    """Лента переменной ширины вдоль сплайна, со сдвигом по нормали «вверх» кузова."""
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


def mirror_pts(pts):
    return [(N - x, y) for x, y in pts]


def swoosh(layer, pts, widths):
    """Связка «ГРАНЬ»: красная лента, над ней через зазор жёлтая кромка и тонкая лазурная нить."""
    ribbon(layer, pts, [max(3, w * 0.16) for w in widths], AZURE, offset=lambda w: w / 2 + 12 + w * 0.36 + 10 + w * 0.08 + 8)
    ribbon(layer, pts, [max(4, w * 0.36) for w in widths], YELLOW, offset=lambda w: w / 2 + 12 + w * 0.18)
    ribbon(layer, pts, widths, RED)


# ---------------------------------------------------------------- 3. линии (та же геометрия, что одобрена на Audi)
FRONT_ARCH, REAR_ARCH = (3560, 3270), (3710, 860)
SIDE = ([(3470, 4060), (3420, 3860)]
        + arc_pts(*FRONT_ARCH, 395, 115, 245, 6)
        + [(3600, 2620), (3640, 2200), (3620, 1760), (3560, 1420)]
        + arc_pts(*REAR_ARCH, 385, 130, 235, 5)
        + [(3770, 400), (3840, 250)])
SIDE_W = [20, 60, 90, 110, 120, 125, 120, 115, 120, 150, 175, 175, 160, 140, 130, 120, 110, 100, 90, 60, 20]
HOOD = [(2050, 3700), (2330, 3640), (2560, 3420), (2600, 3120), (2470, 2840)]
ROOF = [(2445, 2400), (2445, 1800), (2445, 1180)]
TRUNK = [(2445, 1170), (2420, 950), (2330, 760)]
for tr in ((lambda p: p), mirror_pts):
    swoosh(art, tr(SIDE), SIDE_W)
    swoosh(art, tr(HOOD), [30, 90, 110, 90, 60])
    swoosh(art, tr(ROOF), [60, 60, 60])
    swoosh(art, tr(TRUNK), [60, 40, 12])
ribbon(art, [(1120, 250), (1600, 225), (2050, 220), (2500, 225), (2980, 250)], [20, 50, 60, 50, 20], YELLOW)
ribbon(art, [(1120, 200), (1600, 170), (2050, 165), (2500, 170), (2980, 200)], [30, 80, 90, 80, 30], RED)
ribbon(art, [(1400, 3900), (2050, 3990), (2700, 3900)], [20, 70, 20], RED)
ribbon(art, [(1400, 3960), (2050, 4050), (2700, 3960)], [10, 30, 10], YELLOW)

# шашка на переднем бампере под решёткой (как кромка сплиттера на BR03)
yy, xx = np.mgrid[0:N, 0:N]
chk = ((((xx - 2048) // 28) + ((yy - 4040) // 28)) % 2 == 0) & (yy >= 4040) & (xx > 1500) & (xx < 2600)
art_np = np.asarray(art).copy()
art_np[chk] = WHITE + (255,)
art = Image.fromarray(art_np, "RGBA")

rgb = np.asarray(art.convert("RGB")).astype(float) * np.clip(0.55 + 0.45 * shade, 0.35, 1.05)
alpha = np.minimum(np.asarray(art.getchannel("A")), np.asarray(body_mask))
m = alpha > 0
out[m] = out[m] * (1 - alpha[m, None] / 255) + rgb[m] * (alpha[m, None] / 255)
img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def place(layer, cx, cy, angle=0):
    if angle:
        layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
    full = Image.new("RGBA", img.size, (0, 0, 0, 0))
    full.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)))
    full.putalpha(ImageChops.multiply(full.getchannel("A"), body_mask))
    img.alpha_composite(full)


def T(s, size, maxw, fill=WHITE, weight="Black Italic", **kw):
    return blib.fit_text(s, size, maxw, fill, weight=weight, **kw)


def _L(n):
    return Image.open(os.path.join(BX, n)).convert("L")


def fit_h(im, h):
    return im.resize((max(1, int(im.width * h / im.height)), h), Image.LANCZOS)


def fit_w(im, w):
    return im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS)


def smp_lockup(h, mono=None):
    w_, a_ = _L("smp_racing_esports_white.png"), _L("smp_racing_esports_accent.png")
    im = Image.new("RGBA", w_.size, (0, 0, 0, 0))
    im.alpha_composite(blib.solid(w_, mono or WHITE))
    im.alpha_composite(blib.solid(a_, mono or CYAN))
    return fit_h(im, h)


def coat(h):
    sh = _L("saratov_coa_shield.png")
    im = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    im.alpha_composite(blib.solid(sh, (20, 30, 60)))
    im.alpha_composite(blib.solid(sh.filter(ImageFilter.MinFilter(31)), AZURE))
    im.alpha_composite(blib.solid(_L("saratov_coa_crown.png"), (222, 178, 60)))
    im.alpha_composite(blib.solid(_L("saratov_coa_crown_lines.png"), (20, 30, 60)))
    im.alpha_composite(blib.solid(_L("saratov_coa_fish.png"), (225, 230, 238)))
    im.alpha_composite(blib.solid(_L("saratov_coa_lines.png"), (20, 30, 60)))
    return fit_h(im, h)


def raf(h):
    return fit_h(blib.solid(_L("raf_black.png"), WHITE), h)


def mbu(h):
    return fit_h(Image.open(os.path.join(BX, "edm_mbu_logo.png")).convert("RGBA"), h)


def number_plate(H, with_name=True):
    """Номерная табличка — флаг Саратовской области (белый, нижняя треть красная): имя сверху,
    крупный номер, SMP в красной полосе."""
    W = int(H * 1.45)
    S = 2
    t = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    d.rounded_rectangle([0, 0, W * S - 1, H * S - 1], radius=H * S // 14, fill=WHITE + (255,), outline=DEEP + (255,), width=6 * S)
    red_top = int(H * S * 2 / 3)
    m = Image.new("L", t.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, W * S - 1, H * S - 1], radius=H * S // 14, fill=255)
    band = Image.new("RGBA", t.size, RED + (255,))
    bm = Image.new("L", t.size, 0)
    ImageDraw.Draw(bm).rectangle([0, red_top, W * S, H * S], fill=255)
    band.putalpha(ImageChops.multiply(m, bm))
    t.alpha_composite(band)
    ImageDraw.Draw(t).rounded_rectangle([0, 0, W * S - 1, H * S - 1], radius=H * S // 14, outline=DEEP + (255,), width=6 * S)
    t = t.resize((W, H), Image.LANCZOS)
    top = int(H * 0.06)
    if with_name:
        name = T("ПОЗДНЯКОВ СТАНИСЛАВ", int(H * 0.085), int(W * 0.72), INK, weight="ExtraBold Italic")
        fw, fh = int(H * 0.1), int(H * 0.07)
        nx = (W - (fw + 8 + name.width)) // 2
        ny = int(H * 0.05)
        dd = ImageDraw.Draw(t)
        for i, c in enumerate([(255, 255, 255), (0, 57, 166), (213, 43, 30)]):
            dd.rectangle([nx, ny + i * fh // 3, nx + fw - 1, ny + (i + 1) * fh // 3 - 1], fill=c + (255,))
        dd.rectangle([nx, ny, nx + fw - 1, ny + fh - 1], outline=INK + (255,), width=1)
        t.alpha_composite(name, (nx + fw + 8, ny + (fh - name.height) // 2))
        top = ny + fh + int(H * 0.02)
    num = T(NUM, int(H * 0.6), int(W * 0.8), INK, maxh=int(H * 2 / 3) - top - int(H * 0.04))
    t.alpha_composite(num, ((W - num.width) // 2, top + (int(H * 2 / 3) - top - num.height) // 2))
    s = smp_lockup(int(H * 0.2), mono=WHITE)
    t.alpha_composite(s, ((W - s.width) // 2, int(H * 2 / 3) + (H // 3 - s.height) // 2))
    return t


def team_lockup(h):
    k = T("— КОМАНДА —", int(h * 0.3), 10000, YELLOW, weight="Bold Italic", track=0.08)
    e = T("ЭДМ", h, 10000, WHITE)
    t = Image.new("RGBA", (max(k.width, e.width), k.height + e.height + h // 8), (0, 0, 0, 0))
    t.alpha_composite(k, ((t.width - k.width) // 2, 0))
    t.alpha_composite(e, ((t.width - e.width) // 2, k.height + h // 8))
    return t


def pepe_banner(h, maxw):
    """«ПЕПЕ ШНЕЙНЕ ВОТАФА» на красной ленте с жёлтой кромкой — для тех, кто едет сзади."""
    txt = T("ПЕПЕ ШНЕЙНЕ ВОТАФА", int(h * 0.6), maxw - int(h * 1.4), WHITE, stroke=3, stroke_fill=DEEP)
    w = txt.width + int(h * 1.4)
    S = 3
    t = Image.new("RGBA", (w * S, (h + h // 3) * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    sk = h * 0.35 * S
    d.polygon([(0, h * S), (sk, 0), (w * S, 0), (w * S - sk, h * S)], fill=RED)
    g0, g1 = (h + h // 12) * S, (h + h // 4) * S
    d.polygon([(sk * 0.3, g1), (sk * 0.3 + (g1 - g0) * 0.35, g0), (w * S - sk * 0.9, g0),
               (w * S - sk * 0.9 - (g1 - g0) * 0.35, g1)], fill=YELLOW)
    t = t.resize((w, h + h // 3), Image.LANCZOS)
    t.alpha_composite(txt, ((w - txt.width) // 2, (h - txt.height) // 2))
    return t


# ---------------------------------------------------------------- 4. борта
# правая колонка (левый борт) читается +90, левая колонка (правый борт) −90; «вниз» кузова = +x
for side, ang in ((1, 90), (-1, -90)):
    X = (lambda x: x) if side == 1 else (lambda x: N - x)
    # задняя дверь: главный спонсор борта; на левом борту — Симкарт, на правом — Арка (второй — на капоте/багажнике)
    if side == 1:
        place(blib.simkart_logo(118, tagline=True), X(3255), 1440, ang)
        place(T("SIMKART.VERCEL.APP", 24, 330, WHITE, weight="Bold Italic", track=0.05), X(3425), 1530, ang)
    else:
        place(blib.arka_logo(150, taimcafe=True), X(3270), 1440, ang)
    place(number_plate(300), X(3265), 2275, ang)                        # передняя дверь
    place(team_lockup(64), X(3270), 2640, ang)                          # перед номером — команда ЭДМ
    place(mbu(96), X(3460), 1745, ang)                                 # задняя дверь, за спонсором — МБУ
    place(T("САРАТОВСКАЯ ОБЛАСТЬ", 56, 1300, WHITE, track=0.08), X(3920), 1980, ang)   # порог

# ---------------------------------------------------------------- 5. капот, крыша, багажник, бамперы, стекло
place(blib.arka_logo(118), 2050, 3080)                                 # капот: оба главных спонсора
place(blib.simkart_logo(70), 2050, 3245)
place(blib.driveoil_logo(48, plate=False), 1640, 3470)                 # по бокам воздухозаборника
place(raf(110), N - 1640, 3470)
place(number_plate(470, with_name=False), 2050, 1790)                  # крыша: номер во флаге области
place(coat(250), 2050, 2245)                                           # герб с короной — перед номером
place(smp_lockup(90), 2050, 1330)                                      # SMP за номером
place(blib.simkart_logo(80), 2050, 900, 180)                           # багажник (читается сзади)
place(blib.arka_logo(70), 2050, 1075, 180)
place(pepe_banner(104, 820), 2048, 530, 180)                           # задняя стенка под кольцами
place(T("SMP RACING ESPORTS  ·  КОМАНДА ЭДМ  ·  DRIVEOIL", 40, 1500, WHITE, weight="ExtraBold Italic"), 2050, 95, 180)
place(T("САРАТОВ", 32, 240, WHITE), 1760, 3950, -6)                    # губа переднего бампера
place(T("64 РЕГИОН", 32, 240, WHITE), N - 1760, 3950, 6)
place(blib.driveoil_logo(40, plate=False), 1090, 3750)                  # боковины бампера
place(T("КОМАНДА ЭДМ", 40, 300, WHITE, weight="ExtraBold Italic"), N - 1090, 3750)

img = img.convert("RGB")
img.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, "skin_preview.png"))
blib.save_dxt5(img, os.path.join(OUT, "Skin.dds"))

# лобовое стекло: альфа-канал как в исходнике — прозрачно всё, кроме полосы
glass = Image.open(os.path.join(SRC, "glass_sticker.dds")).convert("RGBA")
G = glass.width
bh = int(G * 150 / 1024)
gd = ImageDraw.Draw(glass)
gd.rectangle([0, 0, G, bh], fill=tuple(int(v) for v in BASE) + (255,))
gd.rectangle([0, bh - int(G * 0.012), G, bh], fill=YELLOW + (255,))
gd.rectangle([0, bh - int(G * 0.024), G, bh - int(G * 0.012)], fill=RED + (255,))
ar = blib.arka_logo(int(bh * 0.55))
sk = blib.simkart_logo(int(bh * 0.42))
glass.alpha_composite(ar, (G // 2 - ar.width - int(G * 0.03), max(0, (bh - ar.height) // 2 - 6)))
glass.alpha_composite(sk, (G // 2 + int(G * 0.03), max(0, (bh - sk.height) // 2 - 6)))
blib.save_dxt5(glass, os.path.join(OUT, "glass_sticker.dds"))

ic = Image.new("RGB", (185, 185), tuple(int(v) for v in BASE))
di = ImageDraw.Draw(ic)
di.polygon([(0, 128), (185, 70), (185, 92), (0, 150)], fill=YELLOW)
di.polygon([(0, 154), (185, 96), (185, 118), (0, 176)], fill=RED)
di.text((92, 70), NUM, font=blib.font(92), fill=WHITE, anchor="mm", stroke_width=4, stroke_fill=DEEP)
ic.save(os.path.join(OUT, "livery.png"))

with open(os.path.join(OUT, "ui_skin.json"), "w", encoding="utf-8") as fh:
    json.dump({"skinname": SKIN, "drivername": "Станислав Поздняков", "country": "Russia",
               "team": "Команда ЭДМ · Арка · Симкарт", "number": NUM, "priority": 1}, fh, ensure_ascii=False, indent=2)

with zipfile.ZipFile(os.path.join(HERE, SKIN + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, f), SKIN + "/" + f)
print("готово:", OUT)
