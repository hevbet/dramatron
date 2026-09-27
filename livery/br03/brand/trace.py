"""Векторизация логотипов из референсов (potrace) в чистые маски высокого разрешения."""
import subprocess, os
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

UP = 8      # апскейл перед трассировкой
OUT_SCALE = 2  # масштаб растеризации potrace


def trace(src, box, cond, name, keep_full=False):
    im = Image.open(src).convert("RGB").crop(box)
    a = np.asarray(im).astype(int)
    m = cond(a[..., 0], a[..., 1], a[..., 2])
    mask = Image.fromarray((m * 255).astype(np.uint8)).resize((im.width * UP, im.height * UP), Image.BICUBIC)
    mask = mask.filter(ImageFilter.GaussianBlur(UP * 0.45)).point(lambda v: 0 if v > 127 else 255)  # potrace: чёрное = фигура
    mask.convert("1").save(name + ".pbm")
    subprocess.run(["potrace", name + ".pbm", "-g", "-x", str(OUT_SCALE), "-t", "20", "-a", "1.1", "-o", name + ".pgm"], check=True)
    subprocess.run(["potrace", name + ".pbm", "-s", "-t", "20", "-a", "1.1", "-o", name + ".svg"], check=True)
    r = Image.open(name + ".pgm").convert("L")
    r = np.asarray(Image.eval(r, lambda v: 255 - v)) > 127
    lab, n = ndimage.label(r)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    for k in edge:  # выкидываем то, что касается края кропа (соседние элементы, рамки)
        r[lab == k] = False
    r = Image.fromarray((r * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    if keep_full:
        return r
    bb = r.getbbox()
    r.crop(bb).save(name + ".png")
    os.remove(name + ".pbm"); os.remove(name + ".pgm")
    print(name, r.crop(bb).size)


yellow = lambda r, g, b: ((r > 170) & (g > 140) & (r - b > 40)) | ((r > 225) & (g > 215) & (b > 140))
trace("arka_ref.png", (75, 240, 297, 358), yellow, "arka_word")
trace("arka_ref.png", (60, 362, 315, 419), yellow, "arka_taimcafe")
white = lambda r, g, b: (r > 150) & (g > 150) & (b > 150) & (abs(r - g) < 40)
red = lambda r, g, b: (r > 170) & (g < 110) & (b < 110)
trace("simkart_ref.png", (70, 190, 470, 330), white, "simkart_sim")
trace("simkart_ref.png", (380, 190, 840, 360), red, "simkart_kart")
trace("simkart_ref.png", (860, 60, 1110, 560), white, "simkart_track")
trace("simkart_ref.png", (860, 60, 1110, 560), red, "simkart_track_red")
trace("simkart_ref.png", (70, 150, 470, 180), red, "simkart_tagline")


# «Симкарт» целиком: «Сим» белым и «карт» красным в одних координатах
box = (70, 190, 840, 360)
w = trace("simkart_ref.png", box, white, "_w", keep_full=True)
r = trace("simkart_ref.png", box, red, "_r", keep_full=True)
bb = Image.fromarray(np.maximum(np.asarray(w), np.asarray(r))).getbbox()
w.crop(bb).save("simkart_word_white.png")
r.crop(bb).save("simkart_word_red.png")
print("simkart_word", w.crop(bb).size)
