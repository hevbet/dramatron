"""Библиотека отрисовки ливреи BR03 EVO: слои, ленты, паттерны, бренд-графика, надписи.

Все координаты — пиксели текстуры 4096×4096 (x вправо, y вниз).
Боковины: левая половина развёртки = правый борт машины (вверх кузова = +x, текст rotate −90),
правая половина = левый борт (вверх = −x, текст rotate +90); вперёд везде = +y.
"""
import math
import os
import struct

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

N = 4096
HERE = os.path.dirname(os.path.abspath(__file__))
BRAND = os.path.join(HERE, "..", "br03", "brand")
F_RACE = os.path.join(BRAND, "fonts", "Exo2-Italic[wght].ttf")


def hexc(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------- маски и тени
def load_mask():
    return np.asarray(Image.open(os.path.join(HERE, "uv_mask.png")).convert("L")) > 127


def load_ao(strength=0.7):
    """AO 0..1.1 из ao.png; strength смягчает (1 = как есть, 0 = без теней)."""
    ao = np.asarray(Image.open(os.path.join(HERE, "ao.png")).convert("L")).astype(np.float32) / 255 * 1.1
    return 1 - strength * (1 - np.clip(ao, 0.3, 1.08))


# ---------------------------------------------------------------- шрифты и текст
_font_cache = {}


def font(size, weight="Black Italic"):
    key = (int(size), weight)
    if key not in _font_cache:
        f = ImageFont.truetype(F_RACE, int(size))
        f.set_variation_by_name(weight)
        _font_cache[key] = f
    return _font_cache[key]


def text(s, size, fill, weight="Black Italic", stroke=0, stroke_fill=None, track=0.0):
    """Слой с текстом (обрезан по содержимому). track — разрядка в долях кегля."""
    f = font(size, weight)
    if not track:
        bb = f.getbbox(s, stroke_width=stroke)
        t = Image.new("RGBA", (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), (0, 0, 0, 0))
        ImageDraw.Draw(t).text((2 - bb[0], 2 - bb[1]), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        return t
    gap = f.size * track
    w = int(sum(f.getlength(c) for c in s) + gap * (len(s) - 1)) + 20 + 2 * stroke
    asc, desc = f.getmetrics()
    t = Image.new("RGBA", (w, asc + desc + 20 + 2 * stroke), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    x = 10 + stroke
    for c in s:
        d.text((x, 10 + stroke), c, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        x += f.getlength(c) + gap
    return t.crop(t.getbbox())


def fit_text(s, size, maxw, fill, maxh=None, **kw):
    """Уменьшает кегль, пока строка не влезет в maxw × maxh."""
    while size > 8:
        t = text(s, size, fill, **kw)
        if t.width <= maxw and (maxh is None or t.height <= maxh):
            return t
        size -= 2
    return t


def outlined(mask_l, fill, outline=None, ow=0, shadow=None, sh_off=(0, 0)):
    """Заливка маски + обводка + тень (как у логотипа Арки)."""
    pad = ow + max(abs(sh_off[0]), abs(sh_off[1])) + 4
    big = Image.new("L", (mask_l.width + 2 * pad, mask_l.height + 2 * pad), 0)
    big.paste(mask_l, (pad, pad))
    t = Image.new("RGBA", big.size, (0, 0, 0, 0))
    ol = big.filter(ImageFilter.MaxFilter(min(59, ow * 2 + 1))) if ow else big
    if shadow is not None:
        t.alpha_composite(solid(ImageChops.offset(ol, *sh_off), shadow))
    if outline is not None and ow:
        t.alpha_composite(solid(ol, outline))
    t.alpha_composite(solid(big, fill))
    return t


def solid(mask_l, col, alpha=255):
    t = Image.new("RGBA", mask_l.size, tuple(col[:3]) + (255,))
    t.putalpha(mask_l.point(lambda v: v * alpha // 255) if alpha != 255 else mask_l)
    return t


# ---------------------------------------------------------------- бренд-ассеты
def brand_mask(name, h):
    m = Image.open(os.path.join(BRAND, name)).convert("L")
    return m.resize((max(1, int(m.width * h / m.height)), h), Image.LANCZOS)


def arka_logo(h, fill=(240, 209, 102), dark=(26, 26, 28), taimcafe=False):
    """«арка» как на афише: жёлтая заливка, тёмная обводка и тень вниз-влево; опц. «· тайм-кафе ·» снизу."""
    m = brand_mask("arka_word.png", h)
    t = outlined(m, fill, dark, max(3, h // 22), dark, (-max(2, h // 28), max(3, h // 16)))
    if not taimcafe:
        return t
    s = brand_mask("arka_taimcafe.png", max(12, int(h * 0.27)))
    ts = outlined(s, fill, dark, max(2, s.height // 10), dark, (-2, 3))
    out = Image.new("RGBA", (max(t.width, ts.width), t.height + ts.height + h // 12), (0, 0, 0, 0))
    out.alpha_composite(t, ((out.width - t.width) // 2, 0))
    out.alpha_composite(ts, ((out.width - ts.width) // 2, t.height + h // 12 - h // 10))
    return out


def simkart_logo(h, silver=(217, 216, 224), red=(234, 50, 38), glow=True, tagline=False):
    """«Симкарт»: «Сим» серебро с градиентом, «карт» красный, красное свечение; опц. тэглайн."""
    w = brand_mask("simkart_word_white.png", h)
    r = brand_mask("simkart_word_red.png", h)
    pad = int(h * 0.3)
    size = (w.width + 2 * pad, w.height + 2 * pad)
    t = Image.new("RGBA", size, (0, 0, 0, 0))
    if glow:
        both = Image.new("L", size, 0)
        both.paste(ImageChops.lighter(w, r), (pad, pad))
        t.alpha_composite(solid(both.filter(ImageFilter.GaussianBlur(h * 0.12)).point(lambda v: int(v * 0.7)), red))
    grad = Image.linear_gradient("L").resize(w.size)
    sil = Image.merge("RGB", [grad.point(lambda v, c=c: int(255 - (255 - c) * v / 255)) for c in silver]).convert("RGBA")
    sil.putalpha(w)
    t.alpha_composite(sil, (pad, pad))
    t.alpha_composite(solid(r, red), (pad, pad))
    t = t.crop(t.getbbox())
    if not tagline:
        return t
    tg = solid(brand_mask("simkart_tagline.png", max(10, int(h * 0.16))), red)
    out = Image.new("RGBA", (max(t.width, tg.width), t.height + tg.height + h // 10), (0, 0, 0, 0))
    out.alpha_composite(t, (0, 0))
    out.alpha_composite(tg, (int(h * 0.1), t.height + h // 10 - int(h * 0.12)))
    return out


# ---------------------------------------------------------------- геометрия
def catmull(points, n=48):
    pts = [points[0]] + list(points) + [points[-1]]
    res = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for k in range(n):
            t = k / n
            res.append(tuple(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t * t
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t ** 3) for j in range(2)))
    res.append(tuple(points[-1]))
    return res


def ribbon_poly(points, widths, offset=0.0, n=48):
    """Полигон ленты переменной ширины вдоль сплайна; offset — сдвиг по левой нормали (dy, −dx)."""
    c = catmull(points, n)
    m = len(c)
    L, R = [], []
    for i, (x, y) in enumerate(c):
        j0, j1 = max(0, i - 1), min(m - 1, i + 1)
        dx, dy = c[j1][0] - c[j0][0], c[j1][1] - c[j0][1]
        ln = math.hypot(dx, dy) or 1
        nx, ny = dy / ln, -dx / ln
        t = i / (m - 1) * (len(widths) - 1)
        k = min(int(t), len(widths) - 2)
        w = widths[k] + (widths[k + 1] - widths[k]) * (t - k)
        o = offset(w) if callable(offset) else offset
        cx, cy = x + nx * o, y + ny * o
        L.append((cx + nx * w / 2, cy + ny * w / 2))
        R.append((cx - nx * w / 2, cy - ny * w / 2))
    return L + R[::-1]


def poly_mask(poly, ss=1):
    m = Image.new("L", (N * ss, N * ss), 0)
    ImageDraw.Draw(m).polygon([(x * ss, y * ss) for x, y in poly], fill=255)
    return m if ss == 1 else m.resize((N, N), Image.LANCZOS)


def mirror(points):
    return [(N - x, y) for x, y in points]


# ---------------------------------------------------------------- паттерны (возвращают L-маски N×N)
def checker_mask(cell, x0=0, y0=0, angle=0):
    yy, xx = np.mgrid[0:N, 0:N]
    if angle:
        a = math.radians(angle)
        u = (xx - x0) * math.cos(a) + (yy - y0) * math.sin(a)
        v = -(xx - x0) * math.sin(a) + (yy - y0) * math.cos(a)
    else:
        u, v = xx - x0, yy - y0
    return Image.fromarray(((((u // cell).astype(int) + (v // cell).astype(int)) % 2 == 0) * 255).astype(np.uint8))


def hatch_mask(period, width, angle):
    """Параллельные полосы: период, толщина, угол (градусы от оси x)."""
    yy, xx = np.mgrid[0:N, 0:N]
    a = math.radians(angle)
    u = xx * math.cos(a) + yy * math.sin(a)
    return Image.fromarray(((np.mod(u, period) < width) * 255).astype(np.uint8))


def halftone_mask(cell, density, angle=45):
    """Точки полутона; density — функция (x, y) → 0..1 радиус (массивы)."""
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    a = math.radians(angle)
    u = xx * math.cos(a) + yy * math.sin(a)
    v = -xx * math.sin(a) + yy * math.cos(a)
    cu = (np.floor(u / cell) + 0.5) * cell
    cv = (np.floor(v / cell) + 0.5) * cell
    cx = cu * math.cos(a) - cv * math.sin(a)
    cy = cu * math.sin(a) + cv * math.cos(a)
    r = np.clip(density(cx, cy), 0, 1) * cell * 0.62
    d = np.hypot(u - cu, v - cv)
    return Image.fromarray((np.clip((r - d) * 1.5 + 0.5, 0, 1) * 255).astype(np.uint8))


def word_ribbon_mask(word_png, h, gap, angle, row_gap, shift):
    """Сплошной паттерн из повторяющегося слова (например «арка»), повёрнутый на angle."""
    w = brand_mask(word_png, h)
    big = Image.new("L", (int(N * 1.5), int(N * 1.5)), 0)
    y, row = 0, 0
    while y < big.height:
        x = -((row * shift) % (w.width + gap))
        while x < big.width:
            big.paste(w, (x, y), w)
            x += w.width + gap
        y += h + row_gap
        row += 1
    big = big.rotate(angle, resample=Image.BICUBIC)
    c = big.width // 2
    return big.crop((c - N // 2, c - N // 2, c + N // 2, c + N // 2))


# ---------------------------------------------------------------- холст
class Canvas:
    def __init__(self, base_rgb):
        self.img = Image.new("RGBA", (N, N), tuple(base_rgb) + (255,))

    def fill(self, mask_l, col, alpha=255):
        """Заливает цветом по L-маске."""
        lay = Image.new("RGBA", (N, N), tuple(col[:3]) + (255,))
        lay.putalpha(mask_l if alpha == 255 else mask_l.point(lambda v: v * alpha // 255))
        self.img.alpha_composite(lay)

    def fill_rgb(self, rgb_img, mask_l):
        lay = rgb_img.convert("RGBA")
        lay.putalpha(mask_l)
        self.img.alpha_composite(lay)

    def place(self, layer, cx, cy, angle=0, clip=None):
        """Ставит RGBA-слой центром в (cx, cy), поворот angle (PIL, против часовой), опц. клип по L-маске."""
        if angle:
            layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
        full = Image.new("RGBA", (N, N), (0, 0, 0, 0))
        full.paste(layer, (int(round(cx - layer.width / 2)), int(round(cy - layer.height / 2))))
        if clip is not None:
            full.putalpha(ImageChops.multiply(full.getchannel("A"), clip))
        self.img.alpha_composite(full)
        return (int(round(cx - layer.width / 2)), int(round(cy - layer.height / 2)), layer.width, layer.height)

    def shade(self, ao):
        a = np.asarray(self.img.convert("RGB")).astype(np.float32) * ao[..., None]
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def save_dxt5(im, path):
    """Pillow пишет в заголовок DDS неверные pitch и RGBBitCount — правим как в оригинальных DDS."""
    im.convert("RGBA").save(path, pixel_format="DXT5")
    with open(path, "r+b") as fh:
        fh.seek(20)
        fh.write(struct.pack("<I", max(1, im.width // 4) * max(1, im.height // 4) * 16))
        fh.seek(88)
        fh.write(struct.pack("<I", 0))
