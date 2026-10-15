"""SMP Racing BR03 — ливрея «Розовый пятачок / Butcher Chart Chrome» (семья скинов Audi RS3 команды ЭДМ).

Раскладка и спонсоры — утверждённые клиентом из «ГРАНЬ 64» (livery/br03_pro/make_skin.py): Арка, Симкартинг,
KARTING64, DriveOil, РАФ, BR, SMP Racing eSports, МБУ «Энгельсская молодёжь» + «КОМАНДА ЭДМ»,
«Саратовская область · 64», имя пилота на номерной табличке.  Стиль — как у RS3 butcher_chrome:
розовый перламутр (PIG), бордовые чернила INK с белыми хром-кантами вместо золота/лазури, голографическая
радуга внизу (пороги, юбки, кромка сплиттера, ободки фар), бордовые пунктирные «линии разделки» и подписи
частей туши (ПЯТАЧОК, ЛОПАТКА, КОРЕЙКА, ОКОРОК) шрифтом Podkova, номера — бордовые таблички с хром-рамкой.

Запуск:  python3 make_skin.py                      → оба скина
         python3 make_skin.py Myakotin_11           → один
Каждый скин: <SKIN>/body_paint.dds (4096 DXT5 + мипмапы), car_paint_rims.dds, livery.png, ui_skin.json,
<SKIN>.zip и preview_texture_<SKIN>.png (2048).  Координаты — пиксели текстуры 4096, маски деталей — zones/.
"""
import io
import json
import math
import os
import shutil
import struct
import sys
import zipfile

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import draw  # noqa: E402
from draw import S, M, C, BR, Canvas, place, text_mask, fit, brand, vline, offset_pts, poly_mask, hatch  # noqa: E402
import lib as blib  # noqa: E402

LIV = os.path.dirname(HERE)
FONTS = os.path.join(LIV, "fonts")
BX = os.path.join(LIV, "br03_pro", "brand_extra")          # логотипы высокого разрешения (только чтение)
OLD_RIMS = os.path.join(LIV, "br03_pro", "Pozdnyakov_64", "car_paint_rims.dds")

DRIVERS = {
    "Pozdnyakov_64": dict(num="64", first="Станислав", last="Поздняков"),
    "Myakotin_11": dict(num="11", first="Руслан", last="Мякотин"),
}

# ---------------------------------------------------------------- палитра RS3 butcher_chrome
PIG = (242, 158, 178)            # розовый перламутр
INK = (96, 18, 42)               # бордовые чернила мясника
INK_D = (40, 10, 22)             # глубокие чернила (обводки)
NIGHT = (16, 12, 20)             # ночная плашка Симкартинга
WHITE = (252, 250, 255)
ARKA_Y = (242, 201, 84)          # жёлтый «арки» (чуть насыщеннее брендового 240,209,102: студийный рендер
                                 # иначе принимает пастельный жёлтый за голограмму и красит в лайм)
SK_RED = (234, 50, 38)
HOLO = np.array([(255, 120, 200), (180, 140, 255), (110, 200, 255), (110, 245, 210), (255, 230, 150),
                 (255, 120, 200)], np.float32)
C.update(base=PIG, white=WHITE, ink=INK, deep=INK_D, red=INK, yellow=ARKA_Y)


def font(name, size, var=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), int(size))
    if var:
        f.set_variation_by_name(var)
    return f


F_NUM = lambda s: font("Unbounded[wght].ttf", s, "Black")                          # noqa: E731 номера
F_NAME = lambda s: font("SofiaSansCondensed-Italic[wght].ttf", s, "Black Italic")  # noqa: E731 имена
F_LABEL = lambda s: font("Podkova[wght].ttf", s, "ExtraBold")                      # noqa: E731 подписи частей
F_SPON = lambda s: font("Exo2-Italic[wght].ttf", s, "ExtraBold Italic")            # noqa: E731 ссылки


# ================================================================= 2D-графика (из RS3 butcher_chrome)
def tmask(s, f, track=0.0):
    if not track:
        bb = f.getbbox(s)
        m = Image.new("L", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), 0)
        ImageDraw.Draw(m).text((4 - bb[0], 4 - bb[1]), s, font=f, fill=255)
        return m
    gap = f.size * track
    w = int(sum(f.getlength(c) for c in s) + gap * len(s)) + 20
    asc, desc = f.getmetrics()
    m = Image.new("L", (w, asc + desc + 20), 0)
    d = ImageDraw.Draw(m)
    x = 10
    for c in s:
        d.text((x, 10), c, font=f, fill=255)
        x += f.getlength(c) + gap
    return m.crop(m.getbbox())


def solid(mask, col):
    t = Image.new("RGBA", mask.size, tuple(col[:3]) + (255,))
    t.putalpha(mask)
    return t


def padm(mask, p):
    b = Image.new("L", (mask.width + 2 * p, mask.height + 2 * p), 0)
    b.paste(mask, (p, p))
    return b


def grow(mask, r):
    if r <= 0:
        return mask
    a = np.asarray(mask) > 127
    d = ndimage.distance_transform_edt(~a)
    return Image.fromarray((np.clip(r + 0.5 - d, 0, 1) * 255).astype(np.uint8))


def chrome_rgb(w, h, tint=(1.0, 0.97, 1.04)):
    y = np.linspace(0, 1, h)[:, None]
    st = [(0, 252), (0.38, 232), (0.48, 150), (0.53, 96), (0.6, 205), (0.82, 246), (1, 200)]
    v = np.interp(y, [s[0] for s in st], [s[1] for s in st]) * np.ones((1, w))
    return Image.fromarray(np.clip(np.stack([v * tint[0], v * tint[1], v * tint[2]], -1), 0, 255).astype(np.uint8))


def holo_lookup(t):
    t = (np.asarray(t, np.float32) % 1.0) * (len(HOLO) - 1)
    k = np.floor(t).astype(int)
    f = (t - k)[..., None]
    return HOLO[k] * (1 - f) + HOLO[np.minimum(k + 1, len(HOLO) - 1)] * f


def holo_rgb(w, h, scale=1.0, phase=0.0):
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    return Image.fromarray(holo_lookup(((x * 1.3 + y * 0.45) * scale + phase) % 1.0).astype(np.uint8))


def fill_mask(mask, rgb_img):
    t = rgb_img.convert("RGBA").resize(mask.size)
    t.putalpha(mask)
    return t


def chrome_text(s, f, ow, outline=INK_D, track=0.0):
    """Хромовые буквы, голо-волосок по краю, тёмная обводка, мягкая тень (как цифры RS3)."""
    m = padm(tmask(s, f, track), ow * 2 + 8)
    ol = grow(m, ow)
    out = Image.new("RGBA", m.size, (0, 0, 0, 0))
    sh = ImageChops.offset(ol, max(2, ow // 2), max(3, ow // 2 + 1)).filter(ImageFilter.GaussianBlur(ow * 0.4))
    out.alpha_composite(solid(sh.point(lambda v: int(v * 0.45)), INK_D))
    out.alpha_composite(solid(ol, outline))
    out.alpha_composite(fill_mask(m, holo_rgb(m.width, m.height, 1.4)))
    inner = Image.fromarray(((ndimage.distance_transform_edt(np.asarray(m) > 127) > max(2, ow * 0.35)) * 255)
                            .astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    out.alpha_composite(fill_mask(inner, chrome_rgb(m.width, m.height)))
    return out.crop(out.getbbox())


def ink_text(s, f, col=INK, track=0.0):
    return solid(tmask(s, f, track), col)


def rrect(w, h, r, fill, outline=None, ow=0):
    t = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    ImageDraw.Draw(t).rounded_rectangle([0, 0, int(w) - 1, int(h) - 1], radius=r, fill=tuple(fill) + (255,),
                                        outline=(tuple(outline) + (255,)) if outline else None, width=ow)
    return t


def chrome_frame(w, h, r, bw, fill, ellipse=False):
    """Табличка: хромовый кант + тонкая голо-линия внутри + заливка (ellipse=True — диск)."""
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    def shape(k):
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        if ellipse:
            d.ellipse([k, k, w - 1 - k, h - 1 - k], fill=255)
        else:
            d.rounded_rectangle([k, k, w - 1 - k, h - 1 - k], radius=max(2, r - k), fill=255)
        return m
    out.alpha_composite(fill_mask(shape(0), chrome_rgb(w, h)))
    out.alpha_composite(fill_mask(shape(bw), holo_rgb(w, h, 1.2)))
    out.alpha_composite(solid(shape(bw + max(2, bw // 3)), fill))
    return out


def fit_h(img, h):
    return img.resize((max(1, round(img.width * h / img.height)), max(1, int(h))), Image.LANCZOS)


def fit_w(img, w):
    return img.resize((max(1, int(w)), max(1, round(img.height * w / img.width))), Image.LANCZOS)


def fit_box(img, w, h):
    k = min(w / img.width, h / img.height)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def stack(items, gap):
    w = max(i.width for i in items)
    h = sum(i.height for i in items) + gap * (len(items) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y = 0
    for i in items:
        out.alpha_composite(i, ((w - i.width) // 2, y))
        y += i.height + gap
    return out


def ru_flag(w, h):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, c in enumerate([(255, 255, 255), (0, 57, 166), (213, 43, 30)]):
        d.rectangle([0, i * h // 3, w - 1, (i + 1) * h // 3 - 1], fill=c + (255,))
    d.rectangle([0, 0, w - 1, h - 1], outline=INK_D + (255,), width=1)
    return im


def cut_label(s, h_px, col=INK):
    """Подпись части туши: Podkova ExtraBold (вывеска мясной лавки), h_px — высота прописных, подчёркивание ——◆——."""
    t = ink_text(s, F_LABEL(int(h_px * 1.45)), col, track=0.05)
    t = fit_h(t.crop(t.getbbox()), h_px)
    lw = max(2, h_px // 14)
    u = Image.new("RGBA", (t.width, lw * 7), (0, 0, 0, 0))
    d = ImageDraw.Draw(u)
    c = t.width / 2
    d.line([(0, lw * 3.5), (c - lw * 4, lw * 3.5)], fill=col + (255,), width=lw)
    d.line([(c + lw * 4, lw * 3.5), (t.width, lw * 3.5)], fill=col + (255,), width=lw)
    d.polygon([(c - lw * 3, lw * 3.5), (c, 0), (c + lw * 3, lw * 3.5), (c, lw * 7)], fill=col + (255,))
    return stack([t, u], int(h_px * 0.12))


# ================================================================= логотипы
def _L(n):
    return Image.open(os.path.join(BX, n)).convert("L")


def _two(n, wc, ac):
    im = Image.new("RGBA", _L(n + "_white.png").size, (0, 0, 0, 0))
    im.alpha_composite(blib.solid(_L(n + "_white.png"), wc))
    im.alpha_composite(blib.solid(_L(n + "_accent.png"), ac))
    return im


def _coat():
    """Герб Саратовской области (как в «ГРАНЬ 64»)."""
    sh = _L("saratov_coa_shield.png")
    im = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    im.alpha_composite(blib.solid(sh, (20, 30, 60)))
    im.alpha_composite(blib.solid(sh.filter(ImageFilter.MinFilter(31)), draw.hexc("#1694DC")))
    im.alpha_composite(blib.solid(_L("saratov_coa_crown.png"), (222, 178, 60)))
    im.alpha_composite(blib.solid(_L("saratov_coa_crown_lines.png"), (20, 30, 60)))
    im.alpha_composite(blib.solid(_L("saratov_coa_fish.png"), (225, 230, 238)))
    im.alpha_composite(blib.solid(_L("saratov_coa_lines.png"), (20, 30, 60)))
    return im


def _karting64(flip=False):
    """KARTING64 для розового фона: белая заливка карта с бордовым контуром, ссылка бордовым."""
    k = _L("karting64_black.png")
    cut = 1500
    im = Image.new("RGBA", (k.width, cut), (0, 0, 0, 0))
    for n, c in (("white", WHITE), ("blue", (0, 57, 166)), ("red", (213, 43, 30)), ("black", INK)):
        im.alpha_composite(blib.solid(_L(f"karting64_{n}.png").crop((0, 0, k.width, cut)), c))
    im = im.crop(im.getbbox())
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    t = blib.text("KARTING64.RU", 400, INK, track=0.04)
    t = t.resize((int(im.width * 0.86), int(t.height * im.width * 0.86 / t.width)), Image.LANCZOS)
    out = Image.new("RGBA", (im.width, im.height + t.height + im.height // 10), (0, 0, 0, 0))
    out.alpha_composite(im, (0, 0))
    out.alpha_composite(t, ((im.width - t.width) // 2, im.height + im.height // 10))
    return out


def simkart_plate(w, tagline=True, url=True):
    """«Симкартинг» (lib.simkart_logo, новое имя бренда) на ночной плашке с белым кантом — как на RS3."""
    logo = blib.simkart_logo(160, word="simkarting", glow=False)
    logo = fit_w(logo, w * 0.84)
    parts = [logo]
    if tagline:
        parts.append(blib.solid(blib.brand_mask("simkart_tagline.png", max(10, int(logo.height * 0.24))), SK_RED))
    if url:
        u = ink_text("SIMKARTING.RU", F_SPON(80), WHITE, track=0.06)
        parts.append(fit_h(u.crop(u.getbbox()), max(10, int(logo.height * 0.24))))
    content = stack(parts, max(3, int(logo.height * 0.12)))
    pad = int(w * 0.06)
    plate = rrect(content.width + 2 * pad, content.height + 2 * pad, int(pad * 0.9), NIGHT, WHITE, max(3, int(w / 160)))
    plate.alpha_composite(content, (pad, pad))
    return plate


ASSETS = draw.ASSETS
ASSETS.update({
    "smp_lockup": _two("smp_racing_esports", WHITE, draw.hexc("#00B5EF")),
    "br_eng": blib.solid(_L("br_engineering_black.png"), WHITE),
    "br_sym": blib.solid(_L("br_symbol_white.png"), INK),
    "br03": blib.solid(_L("br03_logo_white.png"), INK),
    "raf": blib.solid(_L("raf_black.png"), INK),
    "coat": _coat(),
    "mbu": Image.open(os.path.join(BX, "edm_mbu_logo.png")).convert("RGBA"),
    "driveoil": blib.driveoil_logo(160, plate=True),
    "karting64": _karting64(),
    "karting64_flip": _karting64(flip=True),
})


# ================================================================= DDS с мипмапами
def save_dxt5_mips(im, path):
    """DXT5 как в старом генераторе (Pillow + правка pitch / RGBBitCount в заголовке) плюс полная цепочка мипмапов."""
    im = im.convert("RGBA")
    levels = []
    w = im.width
    lv = im
    while True:
        enc = lv if lv.width >= 4 else lv.resize((4, 4), Image.BOX)
        buf = io.BytesIO()
        enc.save(buf, "DDS", pixel_format="DXT5")
        b = buf.getvalue()
        assert b[84:88] == b"DXT5"
        n = max(1, lv.width // 4) * max(1, lv.height // 4) * 16
        levels.append((b[:128], b[128:128 + n]))
        if lv.width == 1:
            break
        w = max(1, lv.width // 2)
        lv = im.resize((w, w), Image.LANCZOS) if w >= 4 else lv.resize((w, w), Image.BOX)
    hdr = bytearray(levels[0][0])
    flags = struct.unpack_from("<I", hdr, 8)[0] | 0x20000 | 0x80000           # MIPMAPCOUNT | LINEARSIZE
    struct.pack_into("<I", hdr, 8, flags)
    struct.pack_into("<I", hdr, 20, max(1, im.width // 4) * max(1, im.height // 4) * 16)
    struct.pack_into("<I", hdr, 28, len(levels))
    struct.pack_into("<I", hdr, 88, 0)
    struct.pack_into("<I", hdr, 108, 0x1000 | 0x8 | 0x400000)                  # TEXTURE | COMPLEX | MIPMAP
    with open(path, "wb") as fh:
        fh.write(bytes(hdr))
        for _, data in levels:
            fh.write(data)
    return len(levels)


# ================================================================= ливрея
YY, XX = np.indices((S, S), dtype=np.int32)
DT = ndimage.distance_transform_edt(M)


def Z(n):
    return draw.zmask(n)


def A(m):
    return Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))


def resample(pts, step=2.0):
    P = np.array([(p[0], p[1]) for p in pts], float)
    if len(P) >= 3:
        P = np.array(blib.catmull([tuple(p) for p in P], 24), float)
    L = np.hypot(*np.diff(P, axis=0).T)
    s = np.r_[0, np.cumsum(L)]
    t = np.arange(0, s[-1], step)
    return np.c_[np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], t


def line_mask(pts, w, dash=None, ss=2):
    """Линия ширины w по точкам (сглаженная), сплошная или пунктир dash=(штрих, зазор); маска L на всю текстуру."""
    P, s = resample(pts)
    x0, y0 = np.floor(P.min(0) - w - 4).astype(int)
    x1, y1 = np.ceil(P.max(0) + w + 4).astype(int)
    loc = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
    d = ImageDraw.Draw(loc)
    Q = (P - [x0, y0]) * ss
    r = w * ss / 2

    def seg(a, b):
        q = Q[a:b]
        if len(q) < 2:
            return
        d.line([tuple(p) for p in q], fill=255, width=int(round(w * ss)), joint="curve")
        for p in (q[0], q[-1]):
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
    if dash is None:
        seg(0, len(Q))
    else:
        per = dash[0] + dash[1]
        k = 0
        while k * per < s[-1]:
            a = np.searchsorted(s, k * per + w / 2)
            b = np.searchsorted(s, k * per + dash[0] - w / 2)
            seg(a, b + 1)
            k += 1
    loc = loc.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    full = Image.new("L", (S, S), 0)
    full.paste(loc, (x0, y0))
    return full


def holo_paint(cv, mask, phase=0.0, period=1500.0, name=None):
    """Голографическая радуга (палитра RS3) по маске; цвет бежит вдоль машины (+y) с лёгким наклоном."""
    m = np.asarray(mask, np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    m = m * M
    ys, xs = np.nonzero(m > 0)
    if len(ys) == 0:
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    t = (YY[y0:y1, x0:x1] + 0.35 * XX[y0:y1, x0:x1]) / period + phase
    col = holo_lookup(t)
    arr = np.array(cv.img).astype(np.float32)
    a = m[y0:y1, x0:x1, None]
    arr[y0:y1, x0:x1] = arr[y0:y1, x0:x1] * (1 - a) + col * a
    cv.img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if name:
        cv.log.append((name, A(m)))


def build(skin):
    drv = DRIVERS[skin]
    num = drv["num"]
    cv = Canvas()
    log_labels = []

    def zfill(z, color, cond=None, name=None):
        m = Z(z).astype(np.float32)
        if cond is not None:
            m = m * cond
        cv.paint_alpha(A(m), color, name)

    def keyed(pts, color=INK, key=WHITE, kw=7, name=None):
        """Линия INK с белым хром-кантом (вместо золотых/лазурных лезвий «ГРАНЬ 64»)."""
        kp = [(x, y, ww + 2 * kw if ww > 0 else 0) for x, y, ww in pts]
        cv.paint_alpha(vline(kp), key, None)
        cv.paint_alpha(vline(pts), color, name)

    def cut_line(pts, side_off, w=13, name=None, clip=None):
        """Пунктирная бордовая «линия разделки» + параллельный белый хром-кант (как на RS3)."""
        key = line_mask(offset_pts([(x, y, 0) for x, y in pts], side_off), 4)
        dash = line_mask(pts, w, dash=(44, 24))
        c = None if clip is None else clip.astype(np.float32)
        cv.paint_alpha(key, WHITE, None, clip=c)
        cv.paint_alpha(dash, INK, name, clip=c)

    def put_rgba(img, center, rot, name):
        cv.paint_rgba(place(img, center, rot, "RGBA"), name)

    def logo_rgba(nm, center, rot, h=None, w=None, name=None, mono=None):
        a = ASSETS[nm].copy()
        a = fit_w(a, w) if w else fit_h(a, h)
        if mono is not None:
            al = a.split()[3]
            a = Image.new("RGBA", a.size, mono + (255,))
            a.putalpha(al)
        put_rgba(a, center, rot, name or nm)
        return a.size

    def txt(s, wt, h, center, rot, color, name, tracking=0.0, outline=0, ocol=WHITE):
        m = fit(text_mask(s, wt, 200, tracking), h=h)
        if outline:
            pad = outline + 4
            big = Image.new("L", (m.width + 2 * pad, m.height + 2 * pad), 0)
            big.paste(m, (pad, pad))
            cv.paint_alpha(place(big.filter(ImageFilter.MaxFilter(2 * outline + 1)), center, rot), ocol, None)
            m = big
        cv.paint_alpha(place(m, center, rot), color, name)
        return m.size

    def label(s, h, center, rot, name, min_clear=14):
        """Подпись части туши; уменьшается, пока не ляжет целиком внутрь детали с запасом min_clear px."""
        while h >= 18:
            lab = cut_label(s, h)
            full = place(lab, center, rot, "RGBA")
            a = np.asarray(full.split()[3]) > 40
            if a.any() and DT[a].min() >= min_clear:
                cv.paint_rgba(full, name)
                log_labels.append((name, h))
                return h
            h = int(h * 0.9)
        print("label did not fit:", name)
        return 0

    # ============ 1. ЗАЛИВКИ ============
    # концевые пластины антикрыла: бордо + голо-полоса внизу (где был красный низ флага)
    zfill("rear_wing_endplate_R", INK)
    holo_paint(cv, Z("rear_wing_endplate_R") * (XX < 148), 0.1, name="holo_endplate_R")
    zfill("rear_wing_endplate_L", INK)
    holo_paint(cv, Z("rear_wing_endplate_L") * (XX > 3921), 0.1, name="holo_endplate_L")
    # крыло: верх и задняя кромка бордо, нижняя полоса — голограмма, нижний элемент — белый кант
    zfill("rear_wing_top", INK)
    zfill("rear_wing_rear_face", INK)
    holo_paint(cv, Z("rear_wing_lower_band"), 0.3, period=900.0, name="holo_wingband")
    cv.paint_alpha(vline([(1500, 560, 16), (2600, 560, 16)]), WHITE, "wing_lower_line")
    for z in ["mirror_R", "mirror_L", "splitter_endplate_R", "splitter_endplate_L"]:
        zfill(z, INK)
    for z, ph in (("sidepod_vane_R", 0.55), ("sidepod_vane_L", 0.55)):
        holo_paint(cv, Z(z), ph, name="holo_" + z)
    # ободки фар: голограмма с бордовым волоском по краю
    for z in ["headlight_bezel_carLeftLamp", "headlight_bezel_carRightLamp"]:
        m = Z(z)
        holo_paint(cv, m, 0.15, period=500.0)
        hair = m & (ndimage.distance_transform_edt(m) <= 5)
        cv.paint_alpha(A(hair.astype(np.float32)), INK_D, "bezel_" + z)
    # пороги: голо-полоса внизу (как «пояс» RS3) + бордовая штриховка над ней + белый кант по краю
    for side, z in (("R", "mid_skirt_R"), ("L", "mid_skirt_L")):
        zm = Z(z)
        holo_paint(cv, zm * (YY >= 1500), 0.0, name="holo_sill_" + side)
        cv.paint_alpha(A(zm * (YY >= 1500) * (YY < 1508)), WHITE, None)
        h = hatch([(1300, 1230), (2800, 1230), (2800, 1500), (1300, 1500)], 48, None,
                  lean=(1, -1) if side == "R" else (1, 1), fade=("y", 1260, 1500, 4, 26))
        cv.paint_alpha(h, INK, clip=zm.astype(np.float32))
    for z in ["front_skirt_R", "front_skirt_L"]:
        holo_paint(cv, Z(z), 0.35, name="holo_" + z)

    # ============ 2. СПЛИТТЕР: голо-кромка (низ носа) с тремя бордовыми окнами под спонсоров ============
    lip = (Z("splitter_front_edge_R") | Z("splitter_front_edge_L")).astype(np.float32)
    holo_paint(cv, lip * (YY >= 3852), 0.2, period=1100.0, name="holo_lip")
    cv.paint_alpha(A(lip * (YY >= 3852) * (YY < 3859)), WHITE, None)
    WIN = [(1360, 1690), (1878, 2218), (2406, 2736)]
    for (a, b) in WIN:
        wm = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        ImageDraw.Draw(wm).rounded_rectangle([a, 3900, b - 1, 3995], radius=14, fill=INK + (255,),
                                             outline=WHITE + (255,), width=4)
        cv.paint_rgba(wm, None)

    # ============ 3. ЛИНИИ: лезвия «ГРАНЬ 64» → бордо с белым кантом ============
    Y = {}
    Y["fin_front_R"] = [(842, 1332, 0), (880, 1382, 12), (948, 1440, 22), (1022, 1500, 28), (1118, 1560, 32), (1198, 1612, 34), (1284, 1650, 34)]
    Y["fin_front_L"] = [(3256, 1332, 0), (3218, 1382, 12), (3150, 1440, 22), (3074, 1500, 28), (2980, 1560, 32), (2900, 1612, 34), (2812, 1650, 34)]
    Y["arm_R"] = [(1450, 730, 0), (1488, 860, 16), (1530, 1000, 22), (1562, 1120, 22), (1586, 1240, 18), (1604, 1330, 0)]
    Y["arm_L"] = [(4094 - x, y, w) for x, y, w in Y["arm_R"]]
    Y["pod_shoulder_R"] = [(606, 1520, 0), (618, 1600, 12), (628, 1680, 16), (650, 1760, 18), (654, 1900, 18), (654, 2080, 14), (646, 2190, 0)]
    Y["pod_shoulder_L"] = [(3448, 1520, 0), (3438, 1600, 12), (3424, 1680, 16), (3420, 1760, 18), (3420, 1900, 18), (3420, 2080, 14), (3426, 2190, 0)]
    Y["fender_crest_R"] = [(540, 2998, 0), (540, 3060, 30), (540, 3300, 36), (540, 3400, 36)]
    Y["brow_out_R"] = [(540, 3392, 36), (420, 3448, 30), (276, 3500, 18), (214, 3540, 0)]
    Y["brow_in_R"] = [(540, 3392, 36), (662, 3448, 30), (806, 3500, 18), (846, 3540, 0)]
    Y["fender_crest_L"] = [(3518, 2998, 0), (3518, 3060, 30), (3518, 3300, 36), (3518, 3400, 36)]
    Y["brow_in_L"] = [(3518, 3392, 36), (3400, 3448, 30), (3254, 3500, 18), (3222, 3540, 0)]
    Y["brow_out_L"] = [(3518, 3392, 36), (3640, 3448, 30), (3796, 3500, 18), (3846, 3540, 0)]
    Y["nose_top_R"] = [(1836, 2030, 0), (1840, 2200, 16), (1866, 2400, 22), (1930, 2560, 24), (1915, 2680, 14), (1890, 2740, 0)]
    Y["nose_top_L"] = [(2204, 2030, 0), (2196, 2200, 16), (2160, 2400, 22), (2092, 2560, 24), (2106, 2680, 14), (2128, 2740, 0)]
    T2R = [(1095, 205, 30), (1030, 360, 34), (950, 540, 34), (870, 720, 30), (830, 880, 26), (835, 1020, 20), (818, 1160, 13), (770, 1290, 6), (720, 1390, 0)]
    T2cR = [(1010, 300, 0), (935, 450, 8), (860, 620, 12), (790, 800, 12), (780, 960, 10), (775, 1100, 6), (740, 1220, 0)]
    T2L = [(4081 - x, y, w) for x, y, w in T2R]
    T2cL = [(4081 - x, y, w) for x, y, w in T2cR]
    for nm, p in (("T2R", T2R), ("T2L", T2L)):
        keyed(p, kw=8, name="trail_" + nm)
    for k_, v in Y.items():
        keyed(v, kw=6, name="blade_" + k_)
    threads = {"arm_R": (Y["arm_R"], -24), "arm_L": (Y["arm_L"], 24), "fin_front_R": (Y["fin_front_R"], -30),
               "fin_front_L": (Y["fin_front_L"], 30)}
    for k_, (pts, off) in threads.items():
        p = offset_pts(pts, off)
        p = [(x, y, min(6, w)) for x, y, w in p]
        p[0] = (p[0][0], p[0][1], 0)
        p[-1] = (p[-1][0], p[-1][1], 0 if "arm" in k_ else 6)
        cv.paint_alpha(vline(p), WHITE, "thread_" + k_)

    # ============ 4. ПУНКТИРНЫЕ ЛИНИИ РАЗДЕЛКИ ============
    # хребет кожуха двигателя (вместо шашечных лент), с кантом на стороне борта
    cut_line([(1290, 420), (1318, 1000), (1314, 1630)], -16, name="cut_ridge_R", clip=Z("engine_cover_side_R"))
    cut_line([(2780, 420), (2768, 1000), (2786, 1630)], 16, name="cut_ridge_L", clip=Z("engine_cover_side_L"))
    # окорок: пунктир по бывшей жёлтой линии-компаньону над задней аркой
    cut_line([(x, y) for x, y, _ in T2cR], -14, name="cut_ham_R")
    cut_line([(x, y) for x, y, _ in T2cL], 14, name="cut_ham_L")
    # пятачок: поперечный рез на носу над кончиком
    cut_line([(1790, 3286), (2045, 3298), (2300, 3286)], -14, name="cut_snout", clip=Z("nose_front"))

    # ============ 5. ШТРИХОВКА НА ЗАДНИХ КРЫЛЬЯХ (бордо) ============
    def hatch_block(side, xb, yr, xm, P=52, w_rear=6, w_front=26):
        s = (XX - YY) if side == "R" else (XX + YY)
        s_rear = xm - yr[0] if side == "R" else xm + yr[0]
        s_front = xm - yr[1] if side == "R" else xm + yr[1]
        n = int(abs(s_rear - s_front) // P)
        out = np.zeros((S, S), np.float32)
        for i in range(n + 1):
            w = w_front + (w_rear - w_front) * i / max(1, n)
            if side == "R":
                a = s_front + i * P
                m = (s >= a) & (s < a + w)
            else:
                a = s_front - i * P
                m = (s <= a) & (s > a - w)
            out[m] = 1
        out *= ((XX >= xb[0]) & (XX <= xb[1]))
        return A(out)
    for k_, (sd, xb, yr, xm, z) in {"haunch_R": ("R", (430, 700), (1150, 1370), 565, "rear_haunch_R"),
                                    "haunch_L": ("L", (3381, 3651), (1150, 1370), 3516, "rear_haunch_L")}.items():
        cv.paint_alpha(hatch_block(sd, xb, yr, xm), INK, "hatch_" + k_, clip=Z(z).astype(np.float32))

    # ============ 6. НОМЕРНЫЕ ТАБЛИЧКИ (RS3: бордо, хром-рамка, хромовые цифры) + имя пилота ============
    def plate_img(W=382, H=280):
        bw = 10
        p = chrome_frame(W, H, 34, bw, INK)
        # верхняя строка: флаг + ФАМИЛИЯ ИМЯ
        fx, fy, fw, fh = 26, 24, 40, 27
        p.alpha_composite(ru_flag(fw, fh), (fx, fy))
        nm = ink_text(f"{drv['last']} {drv['first']}".upper(), F_NAME(60), WHITE, track=0.03)
        nm = nm.crop(nm.getbbox())
        nm = fit_box(nm, W - fx - fw - 14 - 26, 25)
        p.alpha_composite(nm, (fx + fw + 12, fy + (fh - nm.height) // 2))
        # нижняя полоса: голограмма с SMP Racing eSports бордовым
        band = Image.new("L", (W, H), 0)
        ImageDraw.Draw(band).rounded_rectangle([22, H - 64, W - 23, H - 22], radius=12, fill=255)
        p.alpha_composite(fill_mask(band, holo_rgb(W, H, 1.1, 0.2)))
        smp = ASSETS["smp_lockup"].copy()
        smp = fit_box(smp, W - 90, 30)
        al = smp.split()[3]
        smp = Image.new("RGBA", smp.size, INK + (255,))
        smp.putalpha(al)
        p.alpha_composite(smp, ((W - smp.width) // 2, H - 43 - smp.height // 2))
        # цифры
        dg = chrome_text(num, F_NUM(170), 6)
        dg = fit_box(dg, W * 0.78, H - 64 - 66 - 10)
        p.alpha_composite(dg, ((W - dg.width) // 2, 64 + (H - 64 - 66 - dg.height) // 2))
        return p

    def shear(img, k, sign):
        """Наклон как у утверждённого параллелограмма таблички: верх сдвинут на k px вперёд по борту."""
        W, H = img.size
        if sign > 0:
            return img.transform((W + k, H), Image.AFFINE, (1, k / H, -k, 0, 1, 0), Image.BICUBIC)
        return img.transform((W + k, H), Image.AFFINE, (1, -k / H, 0, 0, 1, 0), Image.BICUBIC)
    pl = plate_img()
    put_rgba(shear(pl, 60, +1), (256, 2551), -90, "plate_R")
    put_rgba(shear(pl, 60, -1), (4045 - 256, 2551), 90, "plate_L")
    # крыша: хромовый диск с бордовой заливкой и хромовым номером
    r = 180
    dsk = chrome_frame(2 * r, 2 * r, r, 14, INK, ellipse=True)
    dg = fit_box(chrome_text(num, F_NUM(200), 7), 1.25 * r, 0.78 * r)
    dsk.alpha_composite(dg, (r - dg.width // 2, r - dg.height // 2 + 4))
    put_rgba(dsk, (2045, 1130), 0, "roofdisc")

    # ============ 7. ТЕКСТ И ЛОГОТИПЫ (утверждённая раскладка) ============
    def arka(center, rot, h, name):
        w = fit(brand("arka_word"), h=h)
        pad = 24
        big = Image.new("L", (w.width + 2 * pad, w.height + 2 * pad), 0)
        big.paste(w, (pad, pad))
        ol = big.filter(ImageFilter.MaxFilter(9))
        sh = ImageChops.offset(ol, 8, 8)
        cv.paint_alpha(place(sh, center, rot), INK_D, None)
        cv.paint_alpha(place(ol, center, rot), INK_D, None)
        cv.paint_alpha(place(big, center, rot), ARKA_Y, name)
        return w.size

    # --- кили: Арка (R), Симкартинг (L)
    arka((1105, 1130), -90, 138, "arka_fin_R")
    cv.paint_alpha(place(fit(brand("arka_taimcafe"), h=34), (1100, 1385), -90), INK, "taimcafe_fin_R")
    put_rgba(simkart_plate(440), (2968, 1148), 90, "simkart_fin_L")
    # --- задние крылья: Симкартинг (R), Арка (L)
    put_rgba(simkart_plate(500), (478, 820), -90, "simkart_haunch_R")
    arka((3590, 820), 90, 118, "arka_haunch_L")
    # --- МБУ
    logo_rgba("mbu", (1204, 1439), -90, h=124, name="mbu_R")
    logo_rgba("mbu", (2877, 1500), 90, h=112, name="mbu_L")
    # --- KARTING64 + DriveOil за задним колесом (углы ±118° — как в утверждённой раскладке)
    logo_rgba("karting64", (392, 250), -118, w=210, name="karting64_R")
    logo_rgba("karting64_flip", (3652, 236), 118, w=210, name="karting64_L")
    logo_rgba("driveoil", (282, 196), -118, w=180, name="driveoil_R")
    logo_rgba("driveoil", (3746, 168), 118, w=186, name="driveoil_L")
    logo_rgba("driveoil", (234, 3667), -64, w=180, name="driveoil_lamp_R")
    logo_rgba("driveoil", (3818, 3665), 62, w=180, name="driveoil_lamp_L")
    # --- «бочки» за кокпитом: BR, ENGINEERING, SMP Racing eSports, РАФ (бордо)
    for side, rot in (("R", -90), ("L", 90)):
        X = (lambda x: x) if side == "R" else (lambda x: 4045 - x)
        dy = 0 if side == "R" else 18
        logo_rgba("br_sym", (X(1268), 1815 + dy), rot, h=84, name="br_hoop_" + side)
        cv.paint_alpha(place(fit(text_mask("ENGINEERING", 700, 200, 0.08), h=12), (X(1206), 1815 + dy), rot), INK,
                       "br_eng_hoop_" + side)
        logo_rgba("smp_lockup", (X(1140), 1815 + dy - (8 if side == "L" else 0)), rot, h=30, name="smp_hoop_" + side,
                  mono=INK)
        logo_rgba("raf", (X(1066), 1815 + dy), rot, h=64, name="raf_hoop_" + side)
    # --- РАФ на нижних боковинах, МБУ на носу
    logo_rgba("raf", (1329, 2938), -90, h=108, name="raf_R")
    logo_rgba("raf", (2769, 2936), 90, h=108, name="raf_L")
    logo_rgba("mbu", (2336, 2420), 90, h=210, name="mbu_nose")
    # --- КОМАНДА ЭДМ: понтоны и нос (бордо с белым кантом)
    for side, c1, c2, rot, hh in (("R", (545, 1770), (606, 1770), -90, 64), ("L", (3561, 1785), (3488, 1785), 90, 72)):
        txt("ЭДМ", 900, hh, c1, rot, INK, "team_EDM_" + side, outline=4)
        txt("— КОМАНДА —", 700, 26, c2, rot, INK, "team_komanda_" + side, 0.08)
    txt("— КОМАНДА —", 700, 22, (1772, 2400), -90, INK, "nose_komanda", 0.08)
    txt("ЭДМ", 900, 76, (1712, 2400), -90, INK, "nose_EDM", outline=4)
    # --- пороги и юбки (по голограмме — бордо)
    txt("ЭНГЕЛЬССКИЙ ДОМ МОЛОДЁЖИ", 800, 25, (1465, 1700), -90, INK, "midskirt_text_R", 0.05)
    txt("ЭНГЕЛЬССКИЙ ДОМ МОЛОДЁЖИ", 800, 25, (2590, 1700), 90, INK, "midskirt_text_L", 0.05)
    txt("САРАТОВСКАЯ ОБЛАСТЬ", 800, 26, (186, 3130), -94.5, INK, "skirt_text_R", 0.06)
    txt("САРАТОВСКАЯ ОБЛАСТЬ", 800, 26, (3891, 3104), 97, INK, "skirt_text_L", 0.06)
    # --- крыло
    logo_rgba("smp_lockup", (2045, 110), 0, h=80, name="smp_wing")
    txt("KARTING64.RU   ·   SIMKARTING.RU", 900, 34, (2045, 214), 180, WHITE, "wing_links", 0.03)
    # --- нос: герб, «Саратовская область», номер
    logo_rgba("coat", (2044, 2949), 0, h=268, name="coat_nose")
    txt("САРАТОВСКАЯ", 800, 17, (2044, 3125), 0, INK, "nose_text1", 0.04)
    txt("ОБЛАСТЬ", 800, 17, (2044, 3153), 0, INK, "nose_text2", 0.04)
    put_rgba(fit_h(chrome_text(num, F_NUM(160), 6), 88), (2045, 3222), 0, "nose_num")
    # --- сплиттер сверху: Арка + Симкартинг
    arka((1712, 3600), 0, 128, "arka_splitter")
    put_rgba(simkart_plate(340, tagline=False), (2394, 3612), 0, "simkart_splitter")
    # --- окна на кромке: SMP, KARTING64.RU, BR ENGINEERING (белым по бордо)
    logo_rgba("smp_lockup", (1525, 3948), 0, h=62, name="smp_lip_R")
    txt("KARTING64.RU", 900, 30, (2048, 3948), 0, WHITE, "k64_lip", 0.03)
    logo_rgba("br_eng", (2571, 3948), 0, w=280, name="breng_lip_L")
    txt("KARTING64.RU", 900, 21, (1095, 3880), -90, WHITE, "k64_splitter_R", 0.03)
    txt("KARTING64.RU", 900, 21, (3003, 3880), 90, WHITE, "k64_splitter_L", 0.03)
    # --- концевые пластины: Арка (R), номер (L)
    arka((218, 535), -90, 66, "arka_endplate_R")
    put_rgba(fit_h(chrome_text(num, F_NUM(160), 6), 70), (3846, 545), 90, "num_endplate_L")
    # --- BR03 на нижних боковинах перед задним колесом
    logo_rgba("br03", (1180, 3170), -90, h=62, name="br03_R")
    logo_rgba("br03", (2950, 3170), 90, h=62, name="br03_L")

    # ============ 8. ПОДПИСИ ЧАСТЕЙ ТУШИ (как на RS3) ============
    label("ПЯТАЧОК", 30, (2045, 3346), 0, "lab_pyatachok", min_clear=10)
    # наружная боковина переднего понтона над колесом (вертикальная): читается вдоль борта, ровно
    label("ЛОПАТКА", 38, (455, 3160), -90, "lab_lopatka_R")
    label("ЛОПАТКА", 38, (3603, 3160), 90, "lab_lopatka_L")
    label("КОРЕЙКА", 44, (1222, 640), -90, "lab_koreyka_R")
    label("КОРЕЙКА", 44, (2823, 640), 90, "lab_koreyka_L")
    # верх задних крыльев (смотрят вверх): читается сзади (rot 180), как подписи хвоста RS3
    label("ОКОРОК", 36, (670, 990), 180, "lab_okorok_R")
    label("ОКОРОК", 36, (3420, 990), 180, "lab_okorok_L")
    return cv, log_labels


def write_skin(skin, cv):
    drv = DRIVERS[skin]
    out = os.path.join(HERE, skin)
    os.makedirs(out, exist_ok=True)
    img = cv.img
    img.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, f"preview_texture_{skin}.png"))
    # защита от просвечивания на мип-уровнях: каждая деталь продлена наружу своим цветом
    arr = np.array(img)
    dist, (iy, ix) = ndimage.distance_transform_edt(~M, return_indices=True)
    pad = (~M) & (dist <= 16)
    arr[pad] = arr[iy[pad], ix[pad]]
    nl = save_dxt5_mips(Image.fromarray(arr), os.path.join(out, "body_paint.dds"))
    shutil.copyfile(OLD_RIMS, os.path.join(out, "car_paint_rims.dds"))     # тёмные диски, как в старом скине
    # иконка 185x185: розовый, бордовая полоса с голо-кантом, хромовый номер
    ic = Image.new("RGBA", (185, 185), PIG + (255,))
    di = ImageDraw.Draw(ic)
    di.polygon([(0, 132), (185, 84), (185, 120), (0, 168)], fill=INK + (255,))
    hb = Image.new("L", (185, 185), 0)
    ImageDraw.Draw(hb).polygon([(0, 168), (185, 120), (185, 132), (0, 180)], fill=255)
    ic.alpha_composite(fill_mask(hb, holo_rgb(185, 185, 1.0)))
    dg = fit_box(chrome_text(drv["num"], F_NUM(120), 5), 130, 84)
    ic.alpha_composite(dg, ((185 - dg.width) // 2, 70 - dg.height // 2))
    ic.convert("RGB").save(os.path.join(out, "livery.png"))
    with open(os.path.join(out, "ui_skin.json"), "w", encoding="utf-8") as fh:
        json.dump({"skinname": skin, "drivername": f"{drv['first']} {drv['last']}", "country": "Russia",
                   "team": "Команда ЭДМ", "number": drv["num"], "priority": 1}, fh, ensure_ascii=False, indent=2)
    zp = os.path.join(HERE, skin + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out)):
            z.write(os.path.join(out, f), skin + "/" + f)
    print(f"{skin}: body_paint.dds DXT5 4096, {nl} mip levels -> {out}, {zp}")


def report(cv):
    print("--- clearance (min px to island edge, fraction outside) ---")
    for name, alpha in cv.log:
        a = np.array(alpha) > 127
        if not a.any():
            print(name, "EMPTY")
            continue
        d = DT[a].min()
        o = (a & ~M).sum() / a.sum()
        if d < 12 or o > 0.002:
            ys, xs = np.nonzero(a)
            print("%-26s min=%5.1f out=%.3f bbox x%d-%d y%d-%d <<" % (name, d, o, xs.min(), xs.max(), ys.min(), ys.max()))


if __name__ == "__main__":
    skins = sys.argv[1:] or list(DRIVERS)
    for sk in skins:
        cv, labs = build(sk)
        print(sk, "labels:", labs)
        if sk == skins[0]:
            report(cv)
        write_skin(sk, cv)
