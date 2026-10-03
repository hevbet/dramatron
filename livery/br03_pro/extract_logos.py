"""Векторизация логотипов организаторов и BR Engineering в чистые маски высокого разрешения.

Источники (передаются путями): скин Audi RS3 SMP01 (Skin.dds, glass_sticker.dds), исходная текстура
BR03 (body_paint.dds) и контурный логотип BR от пользователя. Результат — PNG-маски в brand_extra/.

Запуск: python3 extract_logos.py <SMP01 dir> <BR03 body_paint.dds> <br_outline.png>
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

SMP01, BR03, BR_OUTLINE = sys.argv[1:4]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brand_extra")
os.makedirs(OUT, exist_ok=True)
UP = 8


def trace(mask_bool, name, up=UP):
    """Маска → potrace → сглаженная маска высокого разрешения (L, белое = фигура), обрезанная по содержимому."""
    m = Image.fromarray((mask_bool * 255).astype(np.uint8))
    m = m.resize((m.width * up, m.height * up), Image.BICUBIC).filter(ImageFilter.GaussianBlur(up * 0.45))
    m = m.point(lambda v: 0 if v > 127 else 255)          # potrace: чёрное = фигура
    pbm, pgm = os.path.join(OUT, "_t.pbm"), os.path.join(OUT, "_t.pgm")
    m.convert("1").save(pbm)
    subprocess.run(["potrace", pbm, "-g", "-t", "10", "-a", "1.0", "-o", pgm], check=True)
    r = Image.eval(Image.open(pgm).convert("L"), lambda v: 255 - v).filter(ImageFilter.GaussianBlur(1.0))
    os.remove(pbm)
    os.remove(pgm)
    return r


def save_set(layers, name):
    """Сохраняет набор масок одного логотипа, обрезая все по общему bbox."""
    bb = None
    for l in layers.values():
        b = l.getbbox()
        bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
    for k, l in layers.items():
        l.crop(bb).save(os.path.join(OUT, f"{name}_{k}.png"))
    print(name, bb[2] - bb[0], bb[3] - bb[1])


def drop_small(m, min_px):
    lab, n = ndimage.label(m)
    s = ndimage.sum(m, lab, range(1, n + 1))
    return np.isin(lab, [i + 1 for i in range(n) if s[i] >= min_px])


# ---------------------------------------------------------------- SMP ESPORTS (борт Audi, вертикально)
sk = Image.open(os.path.join(SMP01, "Skin.dds")).convert("RGB").crop((3150, 1420, 3560, 2660)).rotate(-90, expand=True)
a = np.asarray(sk).astype(int)
r, g, b = a[..., 0], a[..., 1], a[..., 2]
grey = (r > 120) & (abs(r - g) < 30) & (abs(g - b) < 40)
blue = (b > 110) & (b > r + 50)
grey, blue = drop_small(grey, 30), drop_small(blue, 15)
save_set({"white": trace(grey, "e_w"), "accent": trace(blue, "e_b")}, "smp_esports")

# ---------------------------------------------------------------- SMP RACING (полоса на лобовом стекле Audi)
gl = Image.open(os.path.join(SMP01, "glass_sticker.dds")).convert("RGBA").crop((250, 0, 780, 140))
a = np.asarray(gl).astype(int)
r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
white = (r > 150) & (g > 150) & (b > 150) & (al > 100)
blue = (b > 120) & (b > r + 50) & (al > 100)
save_set({"white": trace(drop_small(white, 10), "r_w", 12), "accent": trace(drop_small(blue, 6), "r_b", 12)}, "smp_racing")

# ---------------------------------------------------------------- BR ENGINEERING (верх антикрыла BR03, чёрное на белом)
br = Image.open(BR03).convert("RGB").crop((1640, 60, 2440, 150))
a = np.asarray(br).astype(int)
dark = (a.max(-1) < 90)
save_set({"black": trace(drop_small(dark, 15), "b_k", 10)}, "br_engineering")

# ---------------------------------------------------------------- знак BR крупно (контурный рисунок → заливка по чётности вложенности)
o = Image.open(BR_OUTLINE).convert("LA")
la = np.asarray(o).astype(int)
lum = la[..., 0]
line = lum < 200                              # тонкие контуры
line = ndimage.binary_dilation(line, iterations=1)
free = ~line
lab, n = ndimage.label(free)
bg = lab[0, 0]
# глубина вложенности: BFS от фона через соседство регионов (через линию толщиной ≤ 4 px)
depth = {bg: 0}
frontier = [bg]
adj_struct = np.ones((3, 3), bool)
while frontier:
    nxt = []
    for reg in frontier:
        ring = ndimage.binary_dilation(lab == reg, structure=adj_struct, iterations=5) & free
        for nb in np.unique(lab[ring]):
            if nb and nb not in depth:
                depth[nb] = depth[reg] + 1
                nxt.append(nb)
    frontier = nxt
fill = np.zeros_like(free)
for reg, dpt in depth.items():
    if dpt % 2 == 1:
        fill |= lab == reg
fill = ndimage.binary_closing(fill | (line & ndimage.binary_dilation(fill, iterations=3)), iterations=2)
H = fill.shape[0]
sym = fill[: int(H * 0.86)]
txt = fill[int(H * 0.86):]
save_set({"white": trace(drop_small(sym, 200), "s", 4)}, "br_symbol")
save_set({"white": trace(drop_small(txt, 20), "s2", 6)}, "br_engineering_outline_text")


# ---------------------------------------------------------------- доводка: шов в «ESP ORTS», ровное кольцо, сборка «SMP RACING / ESPORTS»
def load(name):
    return np.asarray(Image.open(os.path.join(OUT, name)).convert("L")).astype(np.uint8)


def close_gap(arrs, x0, width, keep):
    """Вырезает колонки [x0+keep, x0+width) во всех масках (схлопывает лишний пробел)."""
    return [np.concatenate([a[:, :x0 + keep], a[:, x0 + width:]], axis=1) for a in arrs]


def fit_ring(white, bird_xmax):
    """Кольцо вокруг птицы: тонкие структуры в зоне птицы (≠ толстое тело птицы) → робастная подгонка
    окружности → стираем старые обрывки, рисуем ровное кольцо."""
    reg = white[:, :bird_xmax] > 127
    dist = ndimage.distance_transform_edt(reg)
    k = 12
    thick = ndimage.binary_dilation(dist > k, iterations=k + 3) & reg       # тело птицы
    thin = reg & ~thick
    thin = drop_small(thin, 150)
    ys, xs = np.nonzero(thin)
    keep = np.ones(len(xs), bool)
    for _ in range(6):                                                       # итеративно отбрасываем выбросы
        A = np.c_[2 * xs[keep], 2 * ys[keep], np.ones(keep.sum())]
        cx, cy, c = np.linalg.lstsq(A, xs[keep] ** 2 + ys[keep] ** 2, rcond=None)[0]
        rad = np.sqrt(c + cx ** 2 + cy ** 2)
        res = np.abs(np.hypot(xs - cx, ys - cy) - rad)
        keep = res < max(8, np.percentile(res[keep], 70) * 2)
    on_ring = np.zeros_like(reg)
    on_ring[ys[keep], xs[keep]] = True
    thick_px = np.median(dist[on_ring]) * 2 if on_ring.any() else 10
    yy, xx = np.mgrid[0:white.shape[0], 0:white.shape[1]]
    d = np.abs(np.hypot(xx - cx, yy - cy) - rad)
    out = white.copy()
    erase = ndimage.binary_dilation(thin, iterations=3)
    sub = out[:, :bird_xmax]
    sub[erase] = 0
    ring_new = np.clip((thick_px / 2 - d) * 1.5 + 0.5, 0, 1) * 255
    print("ring", int(cx), int(cy), int(rad), round(float(thick_px), 1))
    return np.maximum(out, ring_new.astype(np.uint8))


# SMP ESPORTS: схлопнуть шов, кольцо
ew, ea = load("smp_esports_white.png"), load("smp_esports_accent.png")
ew, ea = close_gap([ew, ea], 6195, 472, 80)
ew = fit_ring(ew, 1539)
Image.fromarray(ew).save(os.path.join(OUT, "smp_esports_white.png"))
Image.fromarray(ea).save(os.path.join(OUT, "smp_esports_accent.png"))

# SMP RACING: кольцо
rw, ra = load("smp_racing_white.png"), load("smp_racing_accent.png")
rw = fit_ring(rw, 1046)
Image.fromarray(rw).save(os.path.join(OUT, "smp_racing_white.png"))

# Сборка «SMP RACING / ESPORTS» (как официальный логотип): птица + SMP RACING сверху, ESPORTS голубым ниже
bird_w, bird_a = rw[:, :1046 + 30], ra[:, :1046 + 30]
smp_racing_txt = rw[:, 1107:]
esp = ew[:, 4291:]
esp = esp[:, :np.nonzero(esp.max(0))[0].max() + 1]
H = rw.shape[0]
txt_rows = np.nonzero(smp_racing_txt.max(1))[0]
t0, t1 = txt_rows.min(), txt_rows.max()
cap = t1 - t0
esp_rows = np.nonzero(esp.max(1))[0]
esp = esp[esp_rows.min():esp_rows.max() + 1]
esp_h = int(cap * 0.78)
esp_img = Image.fromarray(esp).resize((int(esp.shape[1] * esp_h / esp.shape[0]), esp_h), Image.LANCZOS)
gap = int(cap * 0.12)
canvas_h = max(H, t1 + gap + esp_h + 20)
W = 1107 + smp_racing_txt.shape[1]
white = Image.new("L", (W, canvas_h), 0)
accent = Image.new("L", (W, canvas_h), 0)
white.paste(Image.fromarray(bird_w), (0, 0))
accent.paste(Image.fromarray(bird_a), (0, 0))
white.paste(Image.fromarray(smp_racing_txt), (1107, 0))
# ESPORTS: начинается под «SMP», с учётом наклона шрифта сдвиг влево
accent.paste(esp_img, (1107 - int(esp_h * 0.12), t1 + gap))
save_set({"white": white, "accent": accent}, "smp_racing_esports")
