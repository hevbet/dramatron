"""Командная ливрея Audi RS3 LMS: «Pink Pig 2.0» × Y2K.

Розовая «свинка» в духе Porsche 917/20 Pink Pig (кузов поделён на «отрубы» пунктиром с подписями),
но пунктир и надписи — голографические/хромовые в стиле Y2K, плюс звёзды-блёстки.
Две машины команды: A — розовая база + голограмма, B — голографическая (сиреневая) база + розовые акценты.

Карта развёртки — livery/rs3_pro/uv_test/zones_rs3.md (Skin.dds) и антикрыло в glass_sticker.dds.
Запуск: python3 make_skin.py <папка исходного скина SMP01>
"""
import json
import math
import os
import struct
import sys
import zipfile

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

SRC = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
N = 4096

# ---------------------------------------------------------------- палитры
PIG = (244, 160, 184)          # поросячий розовый
HOT = (255, 63, 164)           # горячий розовый Y2K
LILAC = (196, 168, 255)
INK = (110, 18, 40)            # «мясницкая» тёмно-вишнёвая
STAMP = (92, 62, 170)          # фиолетовые чернила ветклейма
WHITE = (250, 250, 255)
HOLO = [(255, 120, 200), (200, 160, 255), (140, 210, 255), (150, 255, 220), (255, 245, 210), (255, 140, 210)]


def font(name, size, var=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), size)
    if var:
        f.set_variation_by_name(var)
    return f


RUSLAN = lambda s: font("RuslanDisplay-Regular.ttf", s)
UNB = lambda s: font("Unbounded[wght].ttf", s, "Black")
OSW = lambda s: font("Oswald[wght].ttf", s, "Bold")


# ---------------------------------------------------------------- исходник: маска кузова и тени
src = Image.open(os.path.join(SRC, "Skin.dds")).convert("RGB")
a = np.asarray(src).astype(float)
mx = a.max(-1)
logo = ndimage.binary_dilation(mx > 110, iterations=6)
rings = np.zeros_like(logo)
rings[720:790, 1950:2140] = True                     # кольца Audi на багажнике оставляем
logo &= ~rings
_, (iy, ix) = ndimage.distance_transform_edt(logo, return_indices=True)
a[logo] = a[iy[logo], ix[logo]]
mx = a.max(-1)
BODY = (a[..., 2] > a[..., 0] + 12) & (mx > 18)
SHADE = np.clip(0.55 + 0.45 * np.clip(mx / 76.0, 0, 1.12), 0.4, 1.06)[..., None]
BODY_L = Image.fromarray((BODY * 255).astype(np.uint8))
DT = ndimage.distance_transform_edt(BODY)
yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)


def holo_field(scale=1.0, phase=0.0):
    """Переливающийся голографический градиент по всей текстуре."""
    t = (xx * 0.00055 + yy * 0.00032) * scale + 0.08 * np.sin(yy * 0.004 + xx * 0.002) + phase
    t = (t % 1.0) * (len(HOLO) - 1)
    k = np.floor(t).astype(int)
    f = (t - k)[..., None]
    pal = np.array(HOLO, np.float32)
    return pal[k] * (1 - f) + pal[np.minimum(k + 1, len(HOLO) - 1)] * f


def chrome_fill(h):
    """Вертикальный хромовый градиент для текста высотой h."""
    stops = [(0, 255), (0.35, 230), (0.5, 140), (0.56, 245), (1, 200)]
    col = np.interp(np.linspace(0, 1, h), [s[0] for s in stops], [s[1] for s in stops])
    return col


class Skin:
    def __init__(self, base_rgb, base_holo=False):
        if base_holo:
            base = holo_field(0.55, 0.25) * 0.55 + np.array(LILAC, np.float32) * 0.45
        else:
            base = np.broadcast_to(np.array(base_rgb, np.float32), (N, N, 3)).copy()
        out = a.copy()
        out[BODY] = (base * SHADE)[BODY]
        out[rings] = np.asarray(src).astype(float)[rings]
        self.img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")

    def paste(self, layer, cx, cy, angle=0, clip=True):
        if angle:
            layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
        full = Image.new("RGBA", (N, N), (0, 0, 0, 0))
        full.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)))
        if clip:
            full.putalpha(ImageChops.multiply(full.getchannel("A"), BODY_L))
        self.img.alpha_composite(full)

    def fit(self, make, size, box, angle=0, margin=18, min_size=20):
        """Ставит слой make(size) в лучшее место внутри box (x0,y0,x1,y1) целиком на кузове;
        уменьшает размер, пока не влезет. Возвращает итоговый размер или None."""
        x0, y0, x1, y1 = box
        while size >= min_size:
            lay = make(size)
            if angle:
                lay = lay.rotate(angle, expand=True, resample=Image.BICUBIC)
            al = np.asarray(lay.getchannel("A")) > 40
            py, px = np.nonzero(al)
            if len(px) > 4000:
                sel = np.linspace(0, len(px) - 1, 4000).astype(int)
                py, px = py[sel], px[sel]
            px = px - lay.width / 2
            py = py - lay.height / 2
            best = None
            step = max(8, size // 6)
            for cy in range(int(y0), int(y1), step):
                for cx in range(int(x0), int(x1), step):
                    xs = (px + cx).astype(int); ys = (py + cy).astype(int)
                    if xs.min() < 0 or ys.min() < 0 or xs.max() >= N or ys.max() >= N:
                        continue
                    v = DT[ys, xs].min()
                    if v >= margin and (best is None or v > best[0]):
                        best = (v, cx, cy)
            if best:
                full = Image.new("RGBA", (N, N), (0, 0, 0, 0))
                full.paste(lay, (int(best[1] - lay.width / 2), int(best[2] - lay.height / 2)))
                self.img.alpha_composite(full)
                return size
            size = int(size * 0.9)
        print("не влезло:", box)
        return None

    def paint(self, mask_l, rgb_field):
        """Залить L-маску полем цвета (N×N×3), с тенями кузова."""
        m = np.asarray(ImageChops.multiply(mask_l, BODY_L)).astype(np.float32)[..., None] / 255
        cur = np.asarray(self.img.convert("RGB")).astype(np.float32)
        cur = cur * (1 - m) + rgb_field * SHADE * m
        self.img = Image.fromarray(np.clip(cur, 0, 255).astype(np.uint8)).convert("RGBA")


# ---------------------------------------------------------------- элементы
def text_layer(s, f, fill=(255, 255, 255), stroke=0, stroke_fill=None):
    bb = f.getbbox(s, stroke_width=stroke)
    t = Image.new("RGBA", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), (0, 0, 0, 0))
    ImageDraw.Draw(t).text((4 - bb[0], 4 - bb[1]), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return t


def chrome_text(s, f, outline=INK, ow=6, holo=False):
    """Надпись: хром (или голограмма) с тёмной обводкой и тенью."""
    m = text_layer(s, f).getchannel("A")
    pad = ow + 10
    big = Image.new("L", (m.width + 2 * pad, m.height + 2 * pad), 0)
    big.paste(m, (pad, pad))
    ol = big.filter(ImageFilter.MaxFilter(ow * 2 + 1))
    out = Image.new("RGBA", big.size, (0, 0, 0, 0))
    sh = ImageChops.offset(ol, ow // 2 + 3, ow // 2 + 4)
    out.alpha_composite(Image.merge("RGBA", (*[Image.new("L", big.size, c) for c in outline], sh.point(lambda v: v * 0.6))))
    out.alpha_composite(Image.merge("RGBA", (*[Image.new("L", big.size, c) for c in outline], ol)))
    if holo:
        hx, hy = np.mgrid[0:big.height, 0:big.width]
        t = ((hy / max(1, big.width)) * 1.6 + hx / max(1, big.height) * 0.4) % 1.0 * (len(HOLO) - 1)
        k = np.floor(t).astype(int); fr = (t - k)[..., None]
        pal = np.array(HOLO, np.float32)
        rgb = pal[k] * (1 - fr) + pal[np.minimum(k + 1, len(HOLO) - 1)] * fr
    else:
        c = chrome_fill(big.height)
        rgb = np.repeat(np.repeat(c[:, None, None], big.width, 1), 3, 2) * np.array([1.0, 0.97, 1.04])
    fill = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")
    fill.putalpha(big)
    out.alpha_composite(fill)
    return out


def stamp(lines, w, color=STAMP, angle=-8):
    """Ветеринарное клеймо: двойной овал и надпись, «чернила» с потёртостями."""
    h = int(w * 0.62)
    t = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(t)
    lw = max(4, w // 40)
    d.ellipse([lw, lw, w - lw, h - lw], outline=255, width=lw)
    d.ellipse([lw * 4, lw * 4, w - lw * 4, h - lw * 4], outline=255, width=max(2, lw // 2))
    fs = int(h * 0.2)
    ys = h / 2 - (len(lines) - 1) * fs * 0.6
    for i, s in enumerate(lines):
        f = OSW(int(fs * (1.25 if i == 1 else 0.85)))
        d.text((w / 2, ys + i * fs * 1.2), s, font=f, fill=255, anchor="mm")
    rng = np.random.default_rng(7)
    noise = (rng.random((h, w)) > 0.18).astype(np.uint8) * 255
    t = ImageChops.multiply(t, Image.fromarray(noise).filter(ImageFilter.GaussianBlur(1.2)))
    out = Image.new("RGBA", (w, h), color + (0,))
    out.putalpha(t.point(lambda v: int(v * 0.85)))
    return out.rotate(angle, expand=True, resample=Image.BICUBIC)


def sparkle(r, color=(255, 255, 255)):
    """Четырёхлучевая звезда-блёстка Y2K со свечением."""
    s = r * 4
    t = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(t)
    c = s / 2
    for ang in (0, 90):
        pts = []
        for k in range(4):
            a_ = math.radians(ang + k * 90)
            pts.append((c + math.cos(a_) * r * 1.6, c + math.sin(a_) * r * 1.6))
            a2 = math.radians(ang + k * 90 + 45)
            pts.append((c + math.cos(a2) * r * 0.28, c + math.sin(a2) * r * 0.28))
        d.polygon(pts, fill=255)
    glow = t.filter(ImageFilter.GaussianBlur(r * 0.35))
    out = Image.new("RGBA", (s, s), color + (0,))
    out.putalpha(ImageChops.lighter(t, glow.point(lambda v: int(v * 0.8))))
    return out


def dashed(points, dash=70, gap=45, width=20):
    """L-маска пунктирной ломаной (линии «разделки»)."""
    m = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(m)
    on, left = True, dash
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        pos = 0.0
        while pos < seg:
            step = min(left, seg - pos)
            if on:
                p0 = (x0 + (x1 - x0) * pos / seg, y0 + (y1 - y0) * pos / seg)
                p1 = (x0 + (x1 - x0) * (pos + step) / seg, y0 + (y1 - y0) * (pos + step) / seg)
                d.line([p0, p1], fill=255, width=width)
            pos += step
            left -= step
            if left <= 0:
                on = not on
                left = dash if on else gap
    return m


def pig_tail(size, color):
    """Закрученный хвостик-спираль."""
    t = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(t)
    c = size / 2
    pts = [(c + math.cos(th) * (8 + th * size * 0.035), c + math.sin(th) * (8 + th * size * 0.035)) for th in np.linspace(0, 4.2 * math.pi, 200)]
    d.line(pts, fill=255, width=max(6, size // 18), joint="curve")
    out = Image.new("RGBA", (size, size), color + (0,))
    out.putalpha(t)
    return out


def snout(w, base, dark):
    """Пятачок: овал и две ноздри."""
    h = int(w * 0.66)
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    d.ellipse([4, 4, w - 4, h - 4], fill=base + (255,), outline=dark + (255,), width=max(5, w // 30))
    for cx in (0.33, 0.67):
        d.ellipse([w * cx - w * 0.09, h * 0.3, w * cx + w * 0.09, h * 0.72], fill=dark + (255,))
    return t


# ---------------------------------------------------------------- раскладка «отрубов»
# Борта: левые колонки текстуры = правый борт (текст −90), правые = левый борт (+90). Вперёд = +y.
def side_cuts(sk, line_field, label, label_holo):
    for side, ang in ((1, -90), (-1, 90)):
        X = (lambda x: x) if side == 1 else (lambda x: N - x)
        lines = [
            [(X(330), 1010), (X(1040), 1010)],                     # граница окорока и дверей
            [(X(330), 2760), (X(1040), 2760)],                     # граница дверей и лопатки
            [(X(600), 1010), (X(600), 2760)],                      # корейка / грудинка
            [(X(250), 1000), (X(250), 2800)],                      # подбрюшина (порог)
        ]
        for pts in lines:
            sk.paint(dashed(pts, width=22), line_field)
        def L(t):
            return lambda sz: chrome_text(t, RUSLAN(sz), outline=label.ink, ow=max(4, sz // 18), holo=label.holo)
        bx = lambda a0, a1: (min(X(a0), X(a1)), max(X(a0), X(a1)))
        for text_, size, xr, yr in (("ОКОРОК", 150, (330, 1040), (240, 980)),
                                    ("КОРЕЙКА", 140, (600, 1040), (1050, 2740)),
                                    ("ГРУДИНКА", 120, (320, 600), (1050, 2740)),
                                    ("ЛОПАТКА", 120, (380, 1160), (2800, 3600))):
            xa, xb = bx(*xr)
            sk.fit(L(text_), size, (xa, yr[0], xb, yr[1]), ang)
        xa, xb = bx(100, 300)
        sk.fit(lambda sz: text_layer("ПОДБРЮШИНА", OSW(sz), fill=label.ink + (255,)), 54, (xa, 1000, xb, 2800), ang, margin=8)


def build(name, variant):
    if variant == "A":
        sk = Skin(PIG)
        line_field = holo_field(1.0)
        ink = INK
    else:
        sk = Skin(None, base_holo=True)
        line_field = np.broadcast_to(np.array(HOT, np.float32), (N, N, 3))
        ink = (90, 20, 70)

    def label(s, size, cx, cy, ang):
        sk.fit(lambda sz: chrome_text(s, RUSLAN(sz), outline=ink, ow=max(4, sz // 18), holo=(variant == "A")),
               size, (cx - 350, cy - 250, cx + 350, cy + 250), ang)
    label.ink, label.holo = ink, (variant == "A")

    def label_small(s, size, cx, cy, ang):
        sk.fit(lambda sz: text_layer(s, OSW(sz), fill=ink + (255,)), size, (cx - 300, cy - 120, cx + 300, cy + 120), ang, margin=8)

    side_cuts(sk, line_field, label, label_small)

    # капот, крыша, багажник, перед и зад
    sk.fit(lambda sz: chrome_text("ШЕЙКА", RUSLAN(sz), outline=ink, ow=max(4, sz // 18), holo=(variant == "A")), 150, (1800, 2820, 2300, 3120), 0, margin=6)
    sk.paste(stamp(["ВЕТКОНТРОЛЬ", "1 СОРТ", "ГОСТ · 64"], 400), 1640, 3230, 0)
    sk.fit(lambda sz: snout(sz, HOT if variant == "A" else PIG, ink), 260, (1960, 3640, 2140, 4040), 0, margin=6)
    label("ВЫРЕЗКА", 190, 2050, 1600, 0)
    sk.paste(stamp(["ВЕТКОНТРОЛЬ", "ВЫСШИЙ СОРТ", "PINK PIG RACING"], 640, angle=6), 2050, 2080, 0)
    sk.paint(dashed([(1600, 1220), (2520, 1220)], width=22), line_field)          # вырезка / хвостик
    label("ХВОСТИК", 120, 2050, 980, 180)
    sk.paste(pig_tail(260, HOT if variant == "A" else PIG), 2380, 900, 0)
    sk.fit(lambda sz: chrome_text("ХРЮ!", RUSLAN(sz), outline=ink, ow=max(4, sz // 18), holo=(variant == "A")), 150, (1900, 380, 2200, 560), 180)
    label_small("PINK PIG RACING · Y2K EDITION", 64, 2050, 200, 180)
    for cx in (1240, N - 1240):
        sk.fit(lambda sz: text_layer("ПЯТАЧОК", OSW(sz), fill=ink + (255,)), 70, (cx - 250, 3600, cx + 250, 4000), 0, margin=8)

    # блёстки Y2K
    rng = np.random.default_rng(64 if variant == "A" else 26)
    for _ in range(70):
        x, y = rng.uniform(150, N - 150), rng.uniform(150, N - 150)
        if BODY[int(y), int(x)]:
            sk.paste(sparkle(int(rng.uniform(14, 34))), x, y, 0)

    out = os.path.join(HERE, name)
    os.makedirs(out, exist_ok=True)
    img = sk.img.convert("RGB")
    img.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, name + "_preview.png"))
    save_dxt5(img, os.path.join(out, "Skin.dds"))
    glass(os.path.join(out, "glass_sticker.dds"), variant, ink)
    with open(os.path.join(out, "ui_skin.json"), "w", encoding="utf-8") as fh:
        json.dump({"skinname": name, "drivername": "", "country": "Russia", "team": "Pink Pig Racing",
                   "number": "", "priority": 1}, fh, ensure_ascii=False, indent=2)
    ic = img.resize((185, 185), Image.LANCZOS)
    ic.save(os.path.join(out, "livery.png"))
    with zipfile.ZipFile(os.path.join(HERE, name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out)):
            z.write(os.path.join(out, f), name + "/" + f)


def glass(path, variant, ink):
    """glass_sticker.dds: верхняя полоса лобового стекла + антикрыло (верх и пластины). Альфа-канал сохраняем."""
    g = Image.open(os.path.join(SRC, "glass_sticker.dds")).convert("RGBA")
    G = g.width
    s = G / 1024
    base = PIG if variant == "A" else LILAC
    d = ImageDraw.Draw(g)
    # лобовое стекло
    d.rectangle([0, 0, G, int(150 * s)], fill=base + (255,))
    ban = chrome_text("PINK PIG RACING", UNB(int(64 * s)), outline=ink, ow=int(5 * s), holo=True)
    g.alpha_composite(ban, ((G - ban.width) // 2, int(75 * s - ban.height / 2)))
    # верх антикрыла
    x0, y0, x1, y1 = [int(v * s) for v in (105, 860, 940, 1005)]
    d.rectangle([x0, y0, x1, y1], fill=base + (255,))
    wing = chrome_text("ХРЮ-ХРЮ · PINK PIG", RUSLAN(int(66 * s)), outline=ink, ow=int(5 * s), holo=True)
    if wing.width > x1 - x0 - 20:
        wing = wing.resize((x1 - x0 - 20, int(wing.height * (x1 - x0 - 20) / wing.width)), Image.LANCZOS)
    g.alpha_composite(wing, ((x0 + x1 - wing.width) // 2, (y0 + y1 - wing.height) // 2))
    # концевые пластины
    for ex0 in (115, 330):
        bx0, by0, bx1, by1 = [int(v * s) for v in (ex0, 762, ex0 + 185, 855)]
        d.rounded_rectangle([bx0, by0, bx1, by1], radius=int(14 * s), fill=base + (255,))
        sn = snout(int(120 * s), HOT if variant == "A" else PIG, ink)
        g.alpha_composite(sn, ((bx0 + bx1 - sn.width) // 2, (by0 + by1 - sn.height) // 2))
    save_dxt5(g, path)


def save_dxt5(im, path):
    im.convert("RGBA").save(path, pixel_format="DXT5")
    with open(path, "r+b") as fh:
        fh.seek(20)
        fh.write(struct.pack("<I", (im.width // 4) * (im.height // 4) * 16))
        fh.seek(88)
        fh.write(struct.pack("<I", 0))


if __name__ == "__main__":
    build("PinkPig_A", "A")
    build("PinkPig_B", "B")
    print("готово")
