"""Тестовый скин с сеткой координат: по скриншотам из игры видно, какая клетка развёртки где лежит на кузове.

Запуск: python3 uv_debug.py
"""
import colorsys
import os
import struct

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "uv_debug")
os.makedirs(OUT, exist_ok=True)
N, C = 4096, 256
F = os.path.join(HERE, "brand", "fonts", "Exo2-Italic[wght].ttf")
font = ImageFont.truetype(F, 120)
font.set_variation_by_name("Black Italic")
cols = "ABCDEFGHIJKLMNOP"

img = Image.new("RGB", (N, N))
d = ImageDraw.Draw(img)
for i in range(N // C):
    for j in range(N // C):
        h = (i * 0.618 + j * 0.13) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.55 if (i + j) % 2 else 0.85, 1.0)
        d.rectangle([i * C, j * C, i * C + C - 1, j * C + C - 1], fill=(int(r * 255), int(g * 255), int(b * 255)))
        d.text((i * C + C / 2, j * C + C / 2), f"{cols[i]}{j + 1}", font=font, fill=(0, 0, 0), anchor="mm")
for k in range(0, N, C // 4):
    w = 6 if k % C == 0 else 2
    d.line([(k, 0), (k, N)], fill=(0, 0, 0), width=w)
    d.line([(0, k), (N, k)], fill=(0, 0, 0), width=w)
img.save(os.path.join(HERE, "uv_debug_preview.png"))
im = img.convert("RGBA")
p = os.path.join(OUT, "body_paint.dds")
im.save(p, pixel_format="DXT5")
with open(p, "r+b") as fh:          # тот же фикс заголовка, что в make_skin.py
    fh.seek(20)
    fh.write(struct.pack("<I", (N // 4) * (N // 4) * 16))
    fh.seek(88)
    fh.write(struct.pack("<I", 0))
with open(os.path.join(OUT, "ui_skin.json"), "w", encoding="utf-8") as fh:
    fh.write('{"skinname": "UV test", "drivername": "", "country": "", "team": "", "number": "", "priority": 9}')
print("готово:", OUT)
