"""Тестовый скин Audi RS3 с сеткой координат: клетки 256 px (A1…P16) + мелкие подклетки 1–4,
сетка по всей текстуре (включая антикрыло). По скриншотам из игры видно, какое место текстуры где лежит на машине.

Запуск: python3 uv_test.py <папка исходного скина SMP01>
"""
import colorsys
import json
import os
import struct
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "uv_test", "UV_test_99")
os.makedirs(OUT, exist_ok=True)
N, C = 4096, 256
F = os.path.join(HERE, "..", "br03", "brand", "fonts", "Exo2-Italic[wght].ttf")
big = ImageFont.truetype(F, 110); big.set_variation_by_name("Black Italic")
small = ImageFont.truetype(F, 34); small.set_variation_by_name("Bold Italic")
cols = "ABCDEFGHIJKLMNOP"

a = np.asarray(Image.open(os.path.join(SRC, "Skin.dds")).convert("RGB")).astype(int)
body = (a[..., 2] > a[..., 0] + 12) & (a.max(-1) > 18)

img = Image.new("RGB", (N, N), (0, 0, 0))
d = ImageDraw.Draw(img)
for i in range(16):
    for j in range(16):
        h = (i * 0.618 + j * 0.13) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.5 if (i + j) % 2 else 0.85, 1.0)
        d.rectangle([i * C, j * C, i * C + C - 1, j * C + C - 1], fill=(int(r * 255), int(g * 255), int(b * 255)))
for k in range(0, N, 64):
    w = 7 if k % C == 0 else 2
    d.line([(k, 0), (k, N)], fill=(0, 0, 0), width=w)
    d.line([(0, k), (N, k)], fill=(0, 0, 0), width=w)
for i in range(16):
    for j in range(16):
        x0, y0 = i * C, j * C
        d.text((x0 + C / 2, y0 + C / 2), f"{cols[i]}{j + 1}", font=big, fill=(0, 0, 0), anchor="mm")
        for q, (dx, dy) in enumerate(((0.25, 0.18), (0.75, 0.18), (0.25, 0.85), (0.75, 0.85)), 1):
            d.text((x0 + C * dx, y0 + C * dy), str(q), font=small, fill=(0, 0, 0), anchor="mm")
arr = np.asarray(img).copy()
# сетка по всей текстуре: антикрыло и чёрные детали тоже в Skin.dds
img = Image.fromarray(arr)
img.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, "uv_test", "uv_test_preview.png"))


def save_dxt5(im, path):
    im.convert("RGBA").save(path, pixel_format="DXT5")
    with open(path, "r+b") as fh:
        fh.seek(20); fh.write(struct.pack("<I", (im.width // 4) * (im.height // 4) * 16))
        fh.seek(88); fh.write(struct.pack("<I", 0))


save_dxt5(img, os.path.join(OUT, "Skin.dds"))
with open(os.path.join(OUT, "ui_skin.json"), "w", encoding="utf-8") as fh:
    json.dump({"skinname": "UV_test_99", "drivername": "", "country": "", "team": "", "number": "99", "priority": 9}, fh)
with zipfile.ZipFile(os.path.join(HERE, "uv_test", "UV_test_99.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, f), "UV_test_99/" + f)
print("готово")
