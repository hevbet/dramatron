"""Логотип РАФ и герб Саратовской области (без короны) → маски высокого разрешения в brand_extra/.

Герб: по линейному рисунку определяем щит (область внутри контура), стерляди (замкнутые области
внутри щита, кроме поля), линии контуров. Слои: field (поле), fish (тела рыб), lines (контуры/детали),
outline (кант щита).

Запуск: python3 extract_region_logos.py <раф.jpg> <герб.webp>
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

RAF, COA = sys.argv[1:3]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brand_extra")
os.makedirs(OUT, exist_ok=True)


def trace(mask_bool, up, turd=10):
    m = Image.fromarray((mask_bool * 255).astype(np.uint8))
    m = m.resize((m.width * up, m.height * up), Image.BICUBIC).filter(ImageFilter.GaussianBlur(up * 0.4))
    m = m.point(lambda v: 0 if v > 127 else 255)
    pbm, pgm = os.path.join(OUT, "_t.pbm"), os.path.join(OUT, "_t.pgm")
    m.convert("1").save(pbm)
    subprocess.run(["potrace", pbm, "-g", "-t", str(turd), "-a", "1.0", "-o", pgm], check=True)
    r = Image.eval(Image.open(pgm).convert("L"), lambda v: 255 - v).filter(ImageFilter.GaussianBlur(0.8))
    os.remove(pbm)
    os.remove(pgm)
    return r


def save_set(layers, name):
    bb = None
    for l in layers.values():
        b = l.getbbox()
        if b is None:
            continue
        bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
    for k, l in layers.items():
        l.crop(bb).save(os.path.join(OUT, f"{name}_{k}.png"))
    print(name, bb[2] - bb[0], bb[3] - bb[1])


# ---------------------------------------------------------------- РАФ
a = np.asarray(Image.open(RAF).convert("L")).astype(int)
save_set({"black": trace(a < 128, 8, 4)}, "raf")

# ---------------------------------------------------------------- Герб Саратовской области (с короной)
g = np.asarray(Image.open(COA).convert("L")).astype(int)
H, W = g.shape
lines = g < 140
free = ~ndimage.binary_dilation(lines, iterations=1)
lab, n = ndimage.label(free)
outside = lab == lab[5, 5]
top = int(H * 0.26)                                  # граница корона / щит
field = lab == lab[int(H * 0.6), int(W * 0.12)]      # поле щита (точка заведомо на поле)
shield = ~outside
shield[:top] = False
shield = ndimage.binary_fill_holes(shield)
edge = shield & ~ndimage.binary_erosion(shield, iterations=28)
rest = shield & ~ndimage.binary_dilation(field, iterations=1)
lab3, n3 = ndimage.label(rest)
fish = np.zeros_like(rest)
for i in range(1, n3 + 1):
    comp = lab3 == i
    area = comp.sum()
    if area > 1500 and (comp & edge).sum() < 0.05 * area:
        fish |= comp
fish = ndimage.binary_fill_holes(fish)
fish_lines = lines & ndimage.binary_dilation(fish, iterations=2)
# корона: всё выше top, что не снаружи; заливка = внутренние области, линии = контуры
crown_all = ~outside
crown_all[top:] = False
crown_all = ndimage.binary_fill_holes(crown_all)
crown_lines = lines & crown_all
UP = 3
save_set({
    "shield": trace(shield, UP),
    "fish": trace(fish, UP),
    "lines": trace(fish_lines, UP, 4),
    "crown": trace(crown_all, UP),
    "crown_lines": trace(crown_lines, UP, 4),
}, "saratov_coa")
