"""«Butcher Chart Chrome» — driver gear for smp_audi_rs3_lms (both AC driver-model generations).

Restyles the team's SMP01 driver textures in the livery's language: pearl flesh-pink base, maroon (INK) side
panels / cuffs / collar, a holographic stripe instead of the white diagonal, thin dashed maroon "butcher cut"
lines along real seams, partner logos only where the original texture already carries readable logos (same
place, same orientation, aspect kept, never mirrored), driver name + RU flag only on the chest where the
original carries horizontal text.

Recolouring keeps the baked fabric: every texture is split into its ORIGINAL colour groups (blue cloth, white
print, red print, grey leather, black ...). Inside each group a shading map = luminance / local mean of that group
(folds, seams, stitching), plus a softened low-frequency term, is applied to the new albedo. The alpha channel is
the original one, bit for bit: the DXT5 alpha blocks are copied from the original file for every mip level.

Run:  python3 make_gear.py      -> out/<skin>/*.dds, gear_preview_<skin>.png
"""
import io
import math
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
LIV = "/home/user/dramatron/livery"
SCRATCH = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad"
SRC = os.path.join(SCRATCH, "rs3", "SMP01")
# other copies of the same team textures; used ONLY to read the original alpha blocks of mip levels that are
# cut off in the SMP01 files (a twin is accepted only if its header and mip 0 are byte-identical to SMP01)
TWINS = [os.path.join(SCRATCH, "src", "00_SMPRacing_25_1"), os.path.join(SCRATCH, "smp08", "SMP08_AGTEAM"),
         os.path.join(SCRATCH, "smp07", "SMP07")]
FONTS = os.path.join(LIV, "fonts")
BRAND = os.path.join(LIV, "br03", "brand")
BX = os.path.join(LIV, "br03_pro", "brand_extra")
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(LIV, "br03_pro"))
import lib as blib  # noqa: E402  (arka_logo, simkart_logo)

DRIVERS = [("Pozdnyakov_23", ("Станислав", "Поздняков")), ("Konopelko_86", ("Матвей", "Конопелько"))]

# ---------------------------------------------------------------- palette (= make_skin.py)
PIG = np.array((242, 158, 178), np.float32)
INK = (96, 18, 42)
INK_D = (40, 10, 22)
NIGHT = (16, 12, 20)
WHITE = (252, 250, 255)
HOLO = np.array([(255, 120, 200), (180, 140, 255), (110, 200, 255), (110, 245, 210), (255, 230, 150),
                 (255, 120, 200)], np.float32)
CLOTH = np.array((240, 160, 184), np.float32)   # pearl flesh-pink cloth (PIG, a touch cooler for fabric)
PEARL = np.array((250, 240, 248), np.float32)   # pearl keylines / piping


# ---------------------------------------------------------------- fonts (= make_skin.py)
def font(name, size, var=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), int(size))
    if var:
        f.set_variation_by_name(var)
    return f


F_NAME = lambda s: font("SofiaSansCondensed-Italic[wght].ttf", s, "Black Italic")  # driver names
F_SPON = lambda s: font("Exo2-Italic[wght].ttf", s, "ExtraBold Italic")         # partner text


# ================================================================= 2D art helpers (= make_skin.py)
def text_mask(s, f):
    bb = f.getbbox(s)
    m = Image.new("L", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), 0)
    ImageDraw.Draw(m).text((4 - bb[0], 4 - bb[1]), s, font=f, fill=255)
    return m


def solid(mask, col):
    t = Image.new("RGBA", mask.size, tuple(int(c) for c in col[:3]) + (255,))
    t.putalpha(mask)
    return t


def holo_lookup(t):
    t = (np.asarray(t, np.float32) % 1.0) * (len(HOLO) - 1)
    k = np.floor(t).astype(int)
    f = (t - k)[..., None]
    return HOLO[k] * (1 - f) + HOLO[np.minimum(k + 1, len(HOLO) - 1)] * f


def holo_rgb(w, h, scale=1.0, phase=0.0):
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    t = ((x * 1.3 + y * 0.45) * scale + phase) % 1.0
    return Image.fromarray(holo_lookup(t).astype(np.uint8))


def chrome_rgb(w, h, tint=(1.0, 0.97, 1.04)):
    y = np.linspace(0, 1, h)[:, None]
    st = [(0, 252), (0.38, 232), (0.48, 150), (0.53, 96), (0.6, 205), (0.82, 246), (1, 200)]
    v = np.interp(y, [s[0] for s in st], [s[1] for s in st]) * np.ones((1, w))
    rgb = np.stack([v * tint[0], v * tint[1], v * tint[2]], -1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def fill_mask(mask, rgb_img):
    t = rgb_img.convert("RGBA").resize(mask.size)
    t.putalpha(mask)
    return t


def chrome_frame(w, h, r, bw, fill):
    """Rounded plate: chrome bevel border + thin holo inner line + fill (= make_skin.py)."""
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    out.alpha_composite(fill_mask(m, chrome_rgb(w, h)))
    m2 = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m2).rounded_rectangle([bw, bw, w - 1 - bw, h - 1 - bw], radius=max(2, r - bw), fill=255)
    out.alpha_composite(fill_mask(m2, holo_rgb(w, h, 1.2)))
    m3 = Image.new("L", (w, h), 0)
    k = bw + max(2, bw // 3)
    ImageDraw.Draw(m3).rounded_rectangle([k, k, w - 1 - k, h - 1 - k], radius=max(2, r - k), fill=255)
    out.alpha_composite(solid(m3, fill))
    return out


def fit_h(img, h):
    return img.resize((max(1, round(img.width * h / img.height)), max(1, int(round(h)))), Image.LANCZOS)


def fit_w(img, w):
    return img.resize((max(1, int(round(w))), max(1, round(img.height * w / img.width))), Image.LANCZOS)


def fit_box(img, w, h):
    k = min(w / img.width, h / img.height)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def stack(items, gap, align="c"):
    w = max(i.width for i in items)
    h = sum(i.height for i in items) + gap * (len(items) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y = 0
    for i in items:
        out.alpha_composite(i, ((w - i.width) // 2 if align == "c" else 0, y))
        y += i.height + gap
    return out


def row(items, gap):
    h = max(i.height for i in items)
    w = sum(i.width for i in items) + gap * (len(items) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for i in items:
        out.alpha_composite(i, (x, (h - i.height) // 2))
        x += i.width + gap
    return out


def ink_text(s, f, col):
    return solid(text_mask(s, f), col)


def ru_flag(w, h, border=INK_D):
    t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    for i, c in enumerate([(255, 255, 255), (0, 57, 166), (213, 43, 30)]):
        d.rectangle([0, i * h // 3, w - 1, (i + 1) * h // 3 - 1], fill=c + (255,))
    d.rectangle([0, 0, w - 1, h - 1], outline=border + (255,), width=max(1, h // 14))
    return t


# ================================================================= brand assets
def _L(path):
    return Image.open(path).convert("L")


def asset_arka(h):
    """«арка» wordmark: yellow fill, dark outline + drop shadow (the poster look, as on the car)."""
    return blib.arka_logo(int(h), fill=(242, 206, 92), dark=INK_D, taimcafe=False)


def asset_smp(h, white=WHITE, accent=(0, 181, 239)):
    """SMP RACING ESPORTS (white layer + accent layer of the brand artwork)."""
    w = _L(os.path.join(BX, "smp_racing_esports_white.png"))
    a = _L(os.path.join(BX, "smp_racing_esports_accent.png"))
    im = Image.new("RGBA", w.size, (0, 0, 0, 0))
    im.alpha_composite(solid(w, white))
    im.alpha_composite(solid(a, accent))
    return fit_h(im.crop(im.getbbox()), h)


def asset_mono(name, h, col):
    m = _L(os.path.join(BX, name))
    m = m.crop(m.getbbox())
    return fit_h(solid(m, col), h)


def asset_br_stacked(h, col):
    """BR ENGINEERING, stacked like the original chest mark: the BR symbol over the word ENGINEERING, both
    cut from the brand artwork (br_engineering_black.png = symbol + word on one line)."""
    m = _L(os.path.join(BX, "br_engineering_black.png"))
    m = m.crop(m.getbbox())
    a = np.asarray(m) > 127
    cols = np.flatnonzero(a.any(0))
    gaps = np.flatnonzero(np.diff(cols) > m.height * 0.12)   # first wide gap = symbol | word
    cut = cols[gaps[0]] + 1
    sym = m.crop((0, 0, cut, m.height))
    word = m.crop((cut, 0, m.width, m.height))
    sym, word = sym.crop(sym.getbbox()), word.crop(word.getbbox())
    word = fit_w(word, sym.width * 1.25)
    art = stack([solid(sym, col), solid(word, col)], int(sym.height * 0.10))
    return fit_h(art, h)


def asset_simkart_patch(length):
    """Симкарт wordmark on its night plate with a chrome frame (the car's door patch, no url)."""
    logo = blib.simkart_logo(int(length * 0.16), silver=(222, 220, 232), red=(236, 52, 40), glow=True)
    logo = fit_w(logo, int(length * 0.82))
    bw = max(3, int(length * 0.018))
    H = logo.height + int(length * 0.07) + 2 * bw
    plate = chrome_frame(int(length), H, int(H * 0.22), bw, NIGHT)
    plate.alpha_composite(logo, ((plate.width - logo.width) // 2, (plate.height - logo.height) // 2))
    return plate


def name_plate(driver, w, h):
    """Burgundy chrome-framed name strip: RU flag + «И. ФАМИЛИЯ» (same strip as the side-window names)."""
    first, last = driver
    S = 4
    W, H = w * S, h * S
    p = chrome_frame(W, H, int(H * 0.24), max(3, int(H * 0.075)), INK)
    fh = int(H * 0.46)
    fl = ru_flag(int(fh * 1.5), fh)
    tx = ink_text(f"{first[0]}. {last}".upper(), F_NAME(int(H * 0.62)), WHITE)
    tx = tx.crop(tx.getbbox())
    pad = int(H * 0.22)
    tx = fit_box(tx, W - fl.width - 3 * pad - int(H * 0.08), H * 0.50)
    x0 = pad + int(H * 0.04)
    p.alpha_composite(fl, (x0, (H - fl.height) // 2))
    tw = W - (x0 + fl.width + pad) - pad
    p.alpha_composite(tx, (x0 + fl.width + pad + (tw - tx.width) // 2, (H - tx.height) // 2))
    return p.resize((w, h), Image.LANCZOS)


def ink_plate(art, padx, pady, fill=INK, line=PEARL, lw=2, radius=None, skew=0.0):
    """Burgundy plate (rounded / optionally skewed like a price tag) with a pearl hairline, art centred."""
    w, h = art.width + 2 * padx, art.height + 2 * pady
    S = 4
    sk = int(abs(skew) * h)
    W, H = (w + sk) * S, h * S
    m_out = Image.new("L", (W, H), 0)
    m_in = Image.new("L", (W, H), 0)
    if skew:
        def quad(i):
            o = i * S
            if skew > 0:
                return [(sk * S + o, o), (W - 1 - o, o), (W - 1 - sk * S - o, H - 1 - o), (o, H - 1 - o)]
            return [(o, o), (W - 1 - sk * S - o, o), (W - 1 - o, H - 1 - o), (sk * S + o, H - 1 - o)]
        ImageDraw.Draw(m_out).polygon(quad(0), fill=255)
        ImageDraw.Draw(m_in).polygon(quad(lw * 1.2), fill=255)
    else:
        r = (radius if radius is not None else h // 4) * S
        ImageDraw.Draw(m_out).rounded_rectangle([0, 0, W - 1, H - 1], radius=r, fill=255)
        ImageDraw.Draw(m_in).rounded_rectangle([lw * S, lw * S, W - 1 - lw * S, H - 1 - lw * S],
                                               radius=max(1, r - lw * S), fill=255)
    t = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    t.alpha_composite(solid(m_out, line))
    t.alpha_composite(solid(m_in, fill))
    t = t.resize((w + sk, h), Image.LANCZOS)
    t.alpha_composite(art, ((t.width - art.width) // 2, (t.height - art.height) // 2))
    return t


# ================================================================= DDS i/o
def dds_header(path):
    with open(path, "rb") as fh:
        hdr = fh.read(128)
    assert hdr[:4] == b"DDS " and hdr[84:88] == b"DXT5", path
    _, flags, h, w, _, _, mips = struct.unpack("<7I", hdr[4:32])
    return hdr, w, h, max(1, mips) if flags & 0x20000 else 1


def level_dims(w, h, n):
    out = []
    for i in range(n):
        lw, lh = max(1, w >> i), max(1, h >> i)
        out.append((lw, lh, max(1, (lw + 3) // 4) * max(1, (lh + 3) // 4) * 16))
    return out


def original_alpha_blocks(name):
    """Original DXT5 alpha blocks (8 bytes per 4x4 block) for every mip level of texture `name`."""
    path = os.path.join(SRC, name)
    hdr, w, h, n = dds_header(path)
    dims = level_dims(w, h, n)
    data = open(path, "rb").read()
    sources = [data]
    for t in TWINS:          # full-chain twin: identical header + identical mip 0 bytes
        p = os.path.join(t, name)
        if os.path.exists(p) and os.path.getsize(p) > len(data):
            d2 = open(p, "rb").read()
            if d2[:128] == data[:128] and d2[128:128 + dims[0][2]] == data[128:128 + dims[0][2]]:
                sources.append(d2)
    blocks, off = [], 128
    for (lw, lh, nb) in dims:
        got = None
        for d in sources:
            if len(d) >= off + nb:
                got = np.frombuffer(d[off:off + nb], np.uint8).reshape(-1, 16)[:, :8].copy()
                break
        blocks.append(got)
        off += nb
    return blocks


def encode_dxt5(img_rgba):
    w, h = img_rgba.size
    if w < 4 or h < 4:        # tiny mips: pad to one 4x4 block by edge repeat
        img_rgba = img_rgba.resize((max(4, w), max(4, h)), Image.NEAREST)
    b = io.BytesIO()
    img_rgba.save(b, format="DDS", pixel_format="DXT5")
    return np.frombuffer(b.getvalue()[128:], np.uint8).reshape(-1, 16).copy()


def save_like_original(rgb, name, out_path):
    """DXT5 with the ORIGINAL header (same size, flags, mip count) and the ORIGINAL alpha blocks on every level.
    Colour of mip i = box-filtered mip i-1 of the new RGB."""
    hdr, w, h, n = dds_header(os.path.join(SRC, name))
    assert rgb.size == (w, h), (name, rgb.size, (w, h))
    a0 = Image.open(os.path.join(SRC, name)).convert("RGBA").getchannel("A")
    ab = original_alpha_blocks(name)
    out = bytearray(hdr)
    cur, cur_a = rgb.convert("RGB"), a0
    for i, (lw, lh, nb) in enumerate(level_dims(w, h, n)):
        if i:
            cur = cur.resize((lw, lh), Image.BOX)
            cur_a = cur_a.resize((lw, lh), Image.BOX)
        im = cur.copy()
        im.putalpha(cur_a)
        blk = encode_dxt5(im)
        assert blk.shape[0] * 16 == nb, (name, i, blk.shape, nb)
        if ab[i] is not None:
            blk[:, :8] = ab[i]
        out += blk.tobytes()
    with open(out_path, "wb") as fh:
        fh.write(out)
    return out_path


# ================================================================= recolour engine
def load_src(name):
    im = Image.open(os.path.join(SRC, name))
    a = np.asarray(im.convert("RGBA"))
    return a[..., :3].astype(np.float32), a[..., 3].copy()


def _down(x, f):
    H, W = x.shape
    return x.reshape(H // f, f, W // f, f).mean((1, 3))


def blur_n(V, M, sigma):
    """Gaussian blur of V over the pixels of mask M only (normalised convolution); big sigmas run downsampled."""
    M = M.astype(np.float32)
    f = 1
    while sigma / (f * 2) >= 5 and V.shape[0] % (f * 2) == 0 and V.shape[1] % (f * 2) == 0:
        f *= 2
    Vd, Md = (_down(V * M, f), _down(M, f)) if f > 1 else (V * M, M)
    num = ndimage.gaussian_filter(Vd, sigma / f, mode="nearest")
    den = ndimage.gaussian_filter(Md, sigma / f, mode="nearest")
    r = num / np.maximum(den, 1e-6)
    if f > 1:
        r = ndimage.zoom(r, f, order=1, mode="nearest")[:V.shape[0], :V.shape[1]]
    return r


def inpaint(V, M):
    """Fill pixels of mask M (old logos) from their surroundings (multi-scale normalised blur)."""
    out = V.copy()
    todo = M.copy()
    for s in (3, 8, 20, 50):
        f = blur_n(V, ~M, s)
        den = ndimage.gaussian_filter((~M).astype(np.float32), s)
        ok = todo & (den > 0.02)
        out[ok] = f[ok]
        todo &= ~ok
    out[todo] = np.median(V[~M])
    return out


def denoise_dark(V, r=4, sig_s=3.0, sig_r=4.0, passes=2, lo=45.0, hi=110.0):
    """Edge-preserving (bilateral, `passes` times) smoothing of the luminance, faded in where the original is dark.
    Near-black DXT texels carry only a few 5/6-bit levels, so V / local-mean turns their block noise and 1-level
    banding into posterised blotches and contour rings on a light new colour; seams and folds (steps of 10+ levels)
    stay."""
    H, W = V.shape
    B = V
    for _ in range(passes):
        P = np.pad(B, r, mode="edge")
        acc = np.zeros_like(V)
        wsum = np.zeros_like(V)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                N = P[r + dy:r + dy + H, r + dx:r + dx + W]
                w = math.exp(-(dx * dx + dy * dy) / (2 * sig_s ** 2)) * np.exp(-((N - B) ** 2) / (2 * sig_r ** 2))
                acc += w * N
                wsum += w
        B = acc / wsum
    t = np.clip((hi - ndimage.gaussian_filter(V, 3)) / (hi - lo), 0, 1)
    return V * (1 - t) + B * t


def shade_map(V, groups, params=None, sig_f=10, sig_m=40, sig_lo=150, c_f=1.0, c_m=0.6, g_lo=0.15,
              lo_clip=(0.85, 1.08), k=10.0, clip=(0.42, 1.45), flat=()):
    """Per original colour group (normalised only over the pixels of that group, so neighbouring prints / the
    background never light up the edges):
      fine detail   (V / mean_10px)            -> seams, stitching, small creases, full strength
      medium folds  (mean_10px / mean_40px)^0.6 -> the big wrinkles, softened
      large scale   (mean_150px / median)^0.15 -> a hint of the baked light (old design gradients removed)
    Groups listed in `flat` (flat background between UV islands) get no shading."""
    S = np.ones_like(V)
    params = params or {}
    for g in np.unique(groups):
        if g < 0 or g in flat:
            continue
        M = groups == g
        p = dict(sig_f=sig_f, sig_m=sig_m, sig_lo=sig_lo, c_f=c_f, c_m=c_m, g_lo=g_lo, lo_clip=lo_clip, k=k)
        p.update(params.get(int(g), {}))
        kk = p["k"]
        f = blur_n(V, M, p["sig_f"])
        m = blur_n(V, M, p["sig_m"])
        lo = blur_n(V, M, p["sig_lo"])
        med = float(np.median(V[M]))
        det = ((V + kk) / (f + kk)) ** p["c_f"] * ((f + kk) / (m + kk)) ** p["c_m"]
        lof = np.clip(((lo + kk) / (med + kk)) ** p["g_lo"], *p["lo_clip"])
        S[M] = (det * lof)[M]
    return np.clip(S, *clip)


def apply_shade(A, S, hi=0.55):
    """A = albedo (H,W,3), S = shading. Shadows multiply (slightly cooler = pearl), highlights lift to white."""
    s = S[..., None]
    d = np.minimum(s, 1.0)
    cool = 1 + (1 - d) * np.array([-0.05, -0.07, 0.06], np.float32)
    out = A * d * cool
    out = out + (255 - out) * np.clip(s - 1, 0, 1) * hi
    return np.clip(out, 0, 255)


def pearl_cloth(shape, seed=0, base=CLOTH):
    """Pearl pink with a soft low-frequency pearl drift (+-4 levels), so big panels are not one flat value."""
    H, W = shape
    rng = np.random.default_rng(seed)
    n = ndimage.gaussian_filter(rng.standard_normal((H // 16, W // 16)).astype(np.float32), 3)
    n = ndimage.zoom(n / (n.std() + 1e-6), 16, order=1)[:H, :W]
    A = np.empty((H, W, 3), np.float32)
    A[:] = base
    A += n[..., None] * np.array((2.5, 2.0, 4.0), np.float32)
    return A


def put(A, mask, col, alpha=1.0):
    """Paint colour `col` (tuple / (H,W,3) array) into albedo A through a soft mask."""
    m = (np.asarray(mask, np.float32) * alpha)[..., None]
    c = np.asarray(col, np.float32)
    A[:] = A * (1 - m) + c * m


def holo_field(shape, p0, direction, period, phase=0.0, across=None, across_k=0.0):
    """Holographic colour field: hue runs along `direction` (unit vector) with the given period in px."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = ((xx - p0[0]) * direction[0] + (yy - p0[1]) * direction[1]) / period + phase
    if across is not None:
        t = t + ((xx - p0[0]) * across[0] + (yy - p0[1]) * across[1]) * across_k
    return holo_lookup(t)


# ================================================================= geometry helpers
def poly_mask(shape, pts, ss=4):
    """Anti-aliased polygon coverage (0..1)."""
    H, W = shape
    pts = np.asarray(pts, np.float32)
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    out = np.zeros(shape, np.float32)
    if x1 <= x0 or y1 <= y0:
        return out
    m = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
    ImageDraw.Draw(m).polygon([((x - x0) * ss, (y - y0) * ss) for x, y in pts], fill=255)
    m = m.resize((x1 - x0, y1 - y0), Image.BOX)
    out[y0:y1, x0:x1] = np.asarray(m, np.float32) / 255
    return out


def catmull(pts, n=12):
    pts = np.asarray(pts, np.float32)
    P = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out, np.float32)


def resample(pts, step=2.0):
    pts = np.asarray(pts, np.float32)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    s = np.arange(0, seg[-1], step)
    return np.stack([np.interp(s, seg, pts[:, 0]), np.interp(s, seg, pts[:, 1])], 1)


def smooth(pts, win=9):
    pts = np.asarray(pts, np.float32)
    if len(pts) < win:
        return pts
    k = np.ones(win, np.float32) / win
    out = pts.copy()
    for c in range(2):
        v = np.r_[np.full(win, pts[0, c]), pts[:, c], np.full(win, pts[-1, c])]
        out[:, c] = np.convolve(v, k, "same")[win:-win]
    out[0], out[-1] = pts[0], pts[-1]
    return out


def offset(pts, d):
    """Offset a polyline by d px along its left normal (image coords: +d = to the right of travel when y down)."""
    pts = np.asarray(pts, np.float32)
    t = np.gradient(pts, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1])[:, None], 1e-6)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    return pts + n * d


def presnap(cost, anchors, r=4):
    """Move each anchor onto the cheapest pixel (the dark line) within r px (slight pull towards the anchor)."""
    out = []
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    for x, y in anchors:
        x, y = int(round(x)), int(round(y))
        w = cost[y - r:y + r + 1, x - r:x + r + 1] * (1 + 0.02 * (xx ** 2 + yy ** 2))
        k = np.unravel_index(np.argmin(w), w.shape)
        out.append((x + k[1] - r, y + k[0] - r))
    return out


def snap_seam(cost, anchors, margin=14, win=15, step=2.0, pre=0):
    """Follow the real seam: minimum-cost 8-connected path through `cost` (low on the dark seam groove) between
    consecutive anchor points; returns a smoothed, `step` px resampled polyline. pre>0: anchors are first moved
    onto the line (presnap); a small `win` keeps the corners of an outline sharp."""
    H, W = cost.shape
    if pre:
        anchors = presnap(cost, anchors, pre)
    path = []
    for (ax, ay), (bx, by) in zip(anchors[:-1], anchors[1:]):
        x0, x1 = int(max(0, min(ax, bx) - margin)), int(min(W, max(ax, bx) + margin + 1))
        y0, y1 = int(max(0, min(ay, by) - margin)), int(min(H, max(ay, by) + margin + 1))
        c = cost[y0:y1, x0:x1]
        h, w = c.shape
        idx = np.arange(h * w).reshape(h, w)
        rows, cols, wts = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ys0, ys1 = 0, h - dy
            xs0, xs1 = max(0, -dx), w - max(0, dx)
            a = idx[ys0:ys1, xs0:xs1].ravel()
            b = idx[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx].ravel()
            wgt = ((c[ys0:ys1, xs0:xs1] + c[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]) * 0.5
                   * math.hypot(dx, dy)).ravel()
            rows += [a, b]
            cols += [b, a]
            wts += [wgt, wgt]
        g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w))
        s = idx[int(round(ay)) - y0, int(round(ax)) - x0]
        e = idx[int(round(by)) - y0, int(round(bx)) - x0]
        _, pred = dijkstra(g.tocsr(), indices=s, return_predecessors=True)
        p, seg = e, []
        while p != s and p >= 0:
            seg.append((p % w + x0, p // w + y0))
            p = pred[p]
        seg.append((ax, ay))
        path += seg[::-1]
    return resample(smooth(resample(path, 1.0), win), step)


def seam_cost(V):
    hp = (V + 4) / (ndimage.gaussian_filter(V, 6) + 4)
    return np.clip(hp, 0.3, 2.0) ** 4 + 0.03


def dash_mask(shape, pts, width, on, off, phase=0.0, ss=4):
    """Anti-aliased dashed line (rounded dashes) along a polyline."""
    H, W = shape
    pts = np.asarray(pts, np.float32)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    total = seg[-1]
    x0, y0 = np.floor(pts.min(0) - width - 2).astype(int)
    x1, y1 = np.ceil(pts.max(0) + width + 2).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    out = np.zeros(shape, np.float32)
    if x1 <= x0 or y1 <= y0:
        return out
    m = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
    d = ImageDraw.Draw(m)
    r = width * ss / 2
    s = phase % (on + off) - (on + off)
    while s < total:
        a, b = max(0.0, s), min(total, s + on)
        if b - a > width * 0.5:
            ts = np.r_[a, seg[(seg > a) & (seg < b)], b]
            q = [((np.interp(t, seg, pts[:, 0]) - x0) * ss, (np.interp(t, seg, pts[:, 1]) - y0) * ss) for t in ts]
            if len(q) > 1:
                d.line(q, fill=255, width=int(round(width * ss)), joint="curve")
            for (qx, qy) in (q[0], q[-1]):
                d.ellipse([qx - r, qy - r, qx + r, qy + r], fill=255)
        s += on + off
    m = m.resize((x1 - x0, y1 - y0), Image.BOX)
    out[y0:y1, x0:x1] = np.asarray(m, np.float32) / 255
    return out


def line_mask(shape, pts, width, ss=4):
    """Anti-aliased solid polyline (round caps): exact coverage from the distance to the densely sampled line
    (a wide PIL polyline through hundreds of 1 px segments leaves little notches along its edge)."""
    from scipy.spatial import cKDTree
    H, W = shape
    pts = resample(np.asarray(pts, np.float32), 0.25)
    out = np.zeros(shape, np.float32)
    r = width / 2 + 1
    x0, y0 = np.floor(pts.min(0) - r).astype(int)
    x1, y1 = np.ceil(pts.max(0) + r).astype(int) + 1
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d, _ = cKDTree(pts).query(np.stack([xx.ravel(), yy.ravel()], 1), distance_upper_bound=r + 1)
    d = np.minimum(d, r + 1).reshape(xx.shape)
    out[y0:y1, x0:x1] = np.clip(width / 2 + 0.5 - d, 0, 1)
    return out


def drop_stubs(dash, frac=0.5, on=34, width=7):
    """Remove dash fragments left over after masking (a dash cut by a logo / piping / panel edge): connected
    pieces smaller than `frac` of a full dash."""
    lab, n = ndimage.label(dash > 0.3)
    if not n:
        return dash
    area = ndimage.sum(np.ones_like(dash), lab, range(1, n + 1))
    ok = np.r_[False, area >= frac * on * width]
    keep = dilate(ok[lab], 1)
    return dash * keep


def dilate(m, r):
    m = np.asarray(m) > 0.5 if m.dtype != bool else m
    return ndimage.distance_transform_edt(~m) <= r


def erode(m, r):
    m = np.asarray(m) > 0.5 if m.dtype != bool else m
    return ndimage.distance_transform_edt(m) > r


class Prints:
    """Logos / plates placed on the texture. Keeps an RGBA layer, the list of placements (for the report), and
    the union alpha (to keep dashes clear and to soften the cloth shading under prints)."""

    def __init__(self, shape):
        self.H, self.W = shape
        self.layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        self.log = []

    def place(self, name, art, center, angle=0.0, ref_box=None):
        """Paste art centred at `center`, rotated `angle` deg counter-clockwise (as seen in the texture).
        ref_box = bbox of the original logo this one replaces (for the report)."""
        if angle:
            big = art.resize((art.width * 2, art.height * 2), Image.LANCZOS)
            big = big.rotate(angle, resample=Image.BICUBIC, expand=True)
            art = big.resize((max(1, big.width // 2), max(1, big.height // 2)), Image.LANCZOS)
        x = int(round(center[0] - art.width / 2))
        y = int(round(center[1] - art.height / 2))
        cx0, cy0 = max(0, -x), max(0, -y)
        cx1, cy1 = min(art.width, self.W - x), min(art.height, self.H - y)
        self.layer.alpha_composite(art.crop((cx0, cy0, cx1, cy1)), (x + cx0, y + cy0))
        a = np.asarray(art.getchannel("A")) > 24
        ys, xs = np.nonzero(a)
        bb = (int(x + xs.min()), int(y + ys.min()), int(x + xs.max()), int(y + ys.max()))
        self.log.append(dict(name=name, center=(round(center[0]), round(center[1])), angle=round(angle, 2),
                             bbox=bb, replaces=ref_box))
        return bb

    def alpha(self):
        return np.asarray(self.layer.getchannel("A"), np.float32) / 255

    def rgb(self):
        return np.asarray(self.layer.convert("RGB"), np.float32)


def composite(A, S, prints, dash=None, dash_col=INK, print_soft=0.45):
    """Final colour = shade(albedo + dashes + prints). Prints get a softer copy of the cloth shading."""
    A = A.copy()
    if dash is not None:
        put(A, dash, dash_col)
    pa = prints.alpha()
    put(A, pa, prints.rgb())
    S_eff = S ** (1 - print_soft * pa)
    return apply_shade(A, S_eff)


def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ================================================================= 2016 suit (2016_Suit_DIFF.dds, 2048²)
# UV map (studied from the texture): torso wrap x 0..1360 / y 0..1090 (front panel between the side seams at
# x≈370 and x≈990, side/back stretch panels with diagonal quilting outside them); boot upper with laces
# x 350..1020 / y 830..1190; legs y 1100..2040 (feet up: the thigh logos are rotated ~180°); sleeves = the two
# long pieces on the right (x 1490..2000), wrist ends at the top of the upper one and at the bottom of the lower
# one; collar = the long strap x 1463..2031 / y 988..1069.
S16 = dict(
    # white diagonal band (fitted straight edges) and the original logos (bbox, angle of the reading direction,
    # measured from the glyph pixels: + = counter-clockwise in the texture)
    band=[(30, 675.0), (1346, 374.4), (1346, 426.4), (30, 875.9)],
    logos=dict(chest=((438, 389, 916, 492), 0.0), raf=((719, 204, 810, 289), 0.0), rskg=((709, 309, 799, 365), 0.0),
               legL=((131, 1539, 369, 1601), 172.0), legR=((1037, 1530, 1270, 1603), 191.9),
               sleeve=((1797, 1156, 1938, 1752), 84.1)),
    seamL=[(378, 405), (374, 470), (372, 560), (371, 640), (368, 720), (360, 780)],
    seamR=[(986, 402), (990, 480), (993, 580), (994, 680), (998, 770)],
    leg_up_L=[(28, 1334), (100, 1338), (200, 1362), (300, 1364), (400, 1393), (500, 1427), (570, 1452)],
    leg_up_R=[(815, 1458), (850, 1441), (900, 1421), (950, 1402), (1000, 1386), (1100, 1370), (1200, 1362),
              (1300, 1347), (1380, 1335)],
    leg_lo_L=[(110, 1749), (250, 1750), (400, 1751), (530, 1751)],
    leg_lo_R=[(830, 1751), (950, 1751), (1050, 1750), (1100, 1750), (1150, 1749), (1200, 1749), (1270, 1746)],
    sleeve_top=[(1562, 200), (1600, 300), (1633, 400), (1664, 500), (1662, 600), (1660, 700), (1670, 800),
                (1685, 870)],
    sleeve_bot=[(1698, 1150), (1675, 1200), (1667, 1300), (1654, 1400), (1648, 1500), (1634, 1600), (1621, 1650),
                (1580, 1750), (1552, 1800)],
    cuff_top=[(1490, 193), (1560, 181), (1620, 168), (1680, 150), (1725, 128), (1775, 106), (1830, 80), (1880, 52),
              (1925, 38), (2000, 30)],
    cuff_bot=[(1565, 1836), (1620, 1860), (1680, 1888), (1740, 1922), (1800, 1952), (1860, 1980), (1910, 1998),
              (2000, 2015)],
)


def suit16_prep():
    rgb, alpha = load_src("2016_Suit_DIFF.dds")
    H, W = rgb.shape[:2]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    V = rgb.max(-1)
    sat = (V - rgb.min(-1)) / (V + 1e-3)
    yy, xx = np.mgrid[0:H, 0:W]
    gray = (sat < 0.25) & (V > 40)
    red = (R > G + 25) & (R > B + 15)
    black = V < 10
    band = poly_mask((H, W), S16["band"]) > 0.5
    # original logos
    logo = np.zeros((H, W), bool)
    for k, ((x0, y0, x1, y1), _) in S16["logos"].items():
        sub = (gray | red)[y0 - 6:y1 + 7, x0 - 6:x1 + 7] & ~dilate(band, 3)[y0 - 6:y1 + 7, x0 - 6:x1 + 7]
        logo[y0 - 6:y1 + 7, x0 - 6:x1 + 7] |= sub
    logo = dilate(logo, 2)
    # original colour groups
    redm = ndimage.binary_fill_holes(ndimage.binary_closing(red & (yy > 400) & (yy < 900), iterations=3))
    lab, n = ndimage.label(redm)
    redm = lab == (np.argmax(ndimage.sum(redm, lab, range(1, n + 1))) + 1)
    # boot upper with laces / collar strap: the whole neutral (unsaturated) island, i.e. the grey leather AND its
    # dark shading (the shadow round the black boot opening, the dark folds of the strap: V < 40, so not `gray`)
    def neutral_island(box):
        m = ((sat < 0.4) | (V < 6)) & box
        lab, n = ndimage.label(m)
        m = lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)
        return ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=3)) & box
    boot = neutral_island((xx > 330) & (xx < 1040) & (yy > 780) & (yy < 1192))
    rect = gray & (xx > 390) & (xx < 1000) & (yy >= 1192)
    collar = neutral_island((xx > 1450) & (yy > 975) & (yy < 1080))
    blk = black & (ndimage.binary_opening(black, iterations=2))
    groups = np.zeros((H, W), np.int16)          # 0 cloth (blue + background + inpainted logos)
    groups[band & ~logo] = 1
    groups[redm] = 2
    groups[boot] = 3
    groups[rect] = 4
    groups[collar] = 5
    groups[blk] = 6
    # flat background between the UV islands: its own group, so island edges are not "lit" by the dark gap
    # (flat = bg colour and no texture in 7x7; the darkest parts of the old gradient are this flat colour too,
    #  with only the seams drawn on them - those seams stay in group 0 and keep their shading)
    m1 = ndimage.uniform_filter(V, 7)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(V * V, 7) - m1 * m1, 0))
    bg = (np.abs(rgb - np.array((2, 12, 27), np.float32)).max(-1) <= 3) & (sd < 0.6)
    bg = ndimage.binary_closing(ndimage.binary_opening(bg, iterations=3), iterations=2)
    groups[bg & (groups == 0)] = 7
    Vi = inpaint(denoise_dark(V), logo)
    S = shade_map(Vi, groups, params={6: dict(k=12.0)}, flat=(7,))
    # black parts (boot opening, ...) never get highlights: their rim (V 5..9 against a group mean of ~1) would
    # light up as a grey line along the soft shadow edge
    S[groups == 6] = np.minimum(S[groups == 6], 1.0)
    cost = seam_cost(V)
    return dict(rgb=rgb, alpha=alpha, V=V, Vi=Vi, S=S, groups=groups, band=band, red=redm, logo=logo, cost=cost,
                shape=(H, W), black=blk)


BOOT = (74, 17, 36)           # boot leather: dark burgundy
BLACKS = (24, 8, 14)          # black parts stay (almost) black, ink-tinted


def suit16_design(c):
    """Albedo, dashes and the driver-independent prints of the 2016 suit."""
    H, W = c["shape"]
    g = c["groups"]
    cloth = (g == 0) | (g == 4) | (g == 7)
    A = pearl_cloth((H, W), seed=16)
    rep = []
    # ---- maroon side/back panels: outside the two side seams of the front panel (seams followed on the cloth)
    sL = snap_seam(c["cost"], S16["seamL"])
    sR = snap_seam(c["cost"], S16["seamR"])
    polyL = [(0, 0), (350, 0), (350, 330), (372, 385)] + [tuple(p) for p in sL] + [(346, 830), (342, 900),
                                                                                   (340, 1092), (0, 1092)]
    polyR = [(1382, 0), (1045, 0), (1045, 320), (1000, 382)] + [tuple(p) for p in sR] + [(1003, 830), (1008, 900),
                                                                                         (1010, 1092), (1382, 1092)]
    panels = np.maximum(poly_mask((H, W), polyL), poly_mask((H, W), polyR))
    # ---- cuffs: sleeve wrist ends (52 px band along the island edge), the leg openings' ribbed trim
    ct = resample(catmull(S16["cuff_top"]), 2)
    cb = resample(catmull(S16["cuff_bot"]), 2)
    cuff_t = np.vstack([offset(ct, -30), offset(ct, 52)[::-1]])
    cuff_b = np.vstack([offset(cb, 30), offset(cb, -52)[::-1]])
    cuffs = np.maximum(poly_mask((H, W), cuff_t), poly_mask((H, W), cuff_b))
    oval = c["black"] & (np.mgrid[0:H, 0:W][1] > 520) & (np.mgrid[0:H, 0:W][1] < 880) & \
        (np.mgrid[0:H, 0:W][0] > 1300)
    ring = dilate(oval, 40) & ~oval
    # ---- collar strap, the shoulder strap piece next to the torso, the round tab in the right armhole
    tab = Image.new("L", (W, H), 0)
    ImageDraw.Draw(tab).rounded_rectangle([902, 214, 1028, 342], radius=34, fill=255)
    strap = poly_mask((H, W), [(1333, 398), (1495, 398), (1495, 712), (1333, 712)])
    trims = np.maximum.reduce([cuffs, ring.astype(np.float32), np.asarray(tab, np.float32) / 255, strap])
    put(A, panels * cloth, INK)
    put(A, trims * cloth, INK)
    put(A, g == 5, INK)
    # ---- holographic stripe instead of the white diagonal; pearl keylines on both long edges
    b = S16["band"]
    d = np.array(b[1]) - np.array(b[0])
    d = d / np.hypot(*d)
    holo = holo_field((H, W), b[0], d, period=820.0, phase=0.05, across=(-d[1], d[0]), across_k=1 / 900)
    band = poly_mask((H, W), b)
    put(A, band, holo)
    top = resample([b[0], b[1]], 2)
    bot = resample([b[3], b[2]], 2)
    kl = np.maximum(line_mask((H, W), offset(top, 6), 6), line_mask((H, W), offset(bot, -6), 6)) * band
    put(A, kl, PEARL)
    # ---- the red wedge -> burgundy with pearl hairlines
    red = c["red"]
    put(A, red, INK)
    ys, xs = np.nonzero(red)
    # straight top edge of the wedge (fit), straight lower edge near its tip
    tops = [(x, ys[xs == x].min()) for x in range(int(xs.min()) + 4, int(xs.max()) - 3, 6)]
    tx, ty = np.array(tops, np.float32).T
    kt = np.polyfit(tx, ty, 1)
    bots = [(x, ys[xs == x].max()) for x in range(1100, int(xs.max()) - 3, 4)]
    bx, by = np.array(bots, np.float32).T
    kb = np.polyfit(bx, by, 1)
    x0, x1 = float(xs.min()), float(xs.max())
    lt = resample([(x0, np.polyval(kt, x0)), (x1, np.polyval(kt, x1))], 2)
    lb = resample([(x0, np.polyval(kb, x0)), (x1, np.polyval(kb, x1))], 2)
    redd = dilate(red, 1).astype(np.float32)
    hl = np.maximum(line_mask((H, W), offset(lt, 4), 4), line_mask((H, W), offset(lb, -4), 4)) * redd
    put(A, hl, PEARL)
    # ---- boots, black parts
    put(A, g == 3, BOOT)
    put(A, g == 6, BLACKS)
    # ---- dashed butcher cut lines (on the pink only)
    pink = cloth & (panels < 0.5) & (trims < 0.5) & (band < 0.5) & ~dilate(red, 3)
    lines = []
    lines.append(offset(sL, -14))          # side seams, 14 px inside the pink front panel
    lines.append(offset(sR, 14))
    for k in ("leg_up_L", "leg_up_R", "leg_lo_L", "leg_lo_R", "sleeve_top", "sleeve_bot"):
        lines.append(snap_seam(c["cost"], S16[k]))
    dash = np.zeros((H, W), np.float32)
    for i, ln in enumerate(lines):
        dash = np.maximum(dash, dash_mask((H, W), ln, width=7, on=34, off=20, phase=i * 7))
    return dict(A=A, dash=dash, pink=pink, panels=panels, trims=trims, band=band, rep=rep)


def suit16_prints(prints, c):
    """Partner logos exactly on the original logo spots, same orientation (driver independent)."""
    L = S16["logos"]
    b = S16["band"]
    band_top = lambda x: b[0][1] + (x - b[0][0]) * (b[1][1] - b[0][1]) / (b[1][0] - b[0][0])
    # chest: SMP RACING -> SMP RACING ESPORTS (burgundy + brand cyan accent), above the holo stripe
    smp = asset_smp(84, white=INK, accent=(0, 160, 220))
    smp = fit_box(smp, 400, 84)
    cx = (L["chest"][0][0] + L["chest"][0][2]) / 2
    cy = min(418.0, band_top(cx + smp.width / 2) - 12 - smp.height / 2)
    prints.place("chest SMP RACING ESPORTS", smp, (cx, cy), 0.0, L["chest"][0])
    # RAF wreath -> РАФ (the federation mark itself, burgundy)
    raf = asset_mono("raf_black.png", 84, INK)
    bx = L["raf"][0]
    prints.place("chest РАФ", raf, ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), 0.0, bx)
    # thighs (texture upside down there): «арка» left, BR ENGINEERING right — same rotation as the old text
    bx, ang = L["legL"]
    arka = fit_box(asset_arka(80), 236, 74)
    prints.place("leg арка", arka, ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), ang, bx)
    bx, ang = L["legR"]
    br = fit_box(asset_mono("br_engineering_black.png", 30, INK), 238, 30)
    prints.place("leg BR ENGINEERING", br, ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), ang, bx)
    # sleeve: vertical SMP RACING -> Симкарт night patch, reading wrist->shoulder like the original
    bx, ang = L["sleeve"]
    sk = fit_box(asset_simkart_patch(560), 560, 132)
    prints.place("sleeve Симкарт", sk, ((bx[0] + bx[2]) / 2 + 4, (bx[1] + bx[3]) / 2), ang, bx)


def make_suit16(c, des, driver):
    H, W = c["shape"]
    prints = Prints((H, W))
    suit16_prints(prints, c)
    # name + RU flag: chest, on the original РСКГ (horizontal text) spot, between the РАФ wreath and SMP
    bx = S16["logos"]["rskg"][0]
    np_ = name_plate(driver, 232, 46)
    prints.place("chest name plate", np_, ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2 - 4), 0.0, bx)
    pa = prints.alpha()
    dash = drop_stubs(des["dash"] * des["pink"] * (1 - dilate(pa > 0.1, 12)))
    out = composite(des["A"], c["S"], prints, dash=dash)
    return to_img(out), prints.log


# ================================================================= old suit (DRIVER_Suit.dds, 2048², + boots)
# UV map: legs y 0..690 (white bands at both ends), boots x 610..1110 / y 560..1010 and x 1470..1940 / y 340..840,
# soles = the two dark ovals top right, sleeves left (x 0..640, y 820..2045, white ends), torso x 620..2048 /
# y 850..2048: white collar, white piping from collar to shoulder, chest band with BR03 + SMP RACING, BR mark
# under it, white side panels, white belt across.
SO = dict(
    logos=dict(br03=((1150, 1320, 1336, 1367), 0.0), smp=((1368, 1316, 1570, 1370), 0.0),
               br=((1206, 1384, 1298, 1476), 0.0)),
    boots=[(600, 555, 1140, 1048)], boot_seed=(1800, 700),
    sole_seeds=[(1800, 150), (1960, 150)], epaulettes=[(993, 1294), (1701, 1292)],
    # piping: the two straight collar -> shoulder lines (boxes) and the curved chest line under the BR mark
    # (columns x0..x1, searched in y0..y1: the lowest white run of each column, the BR mark sits just above it)
    piping=[(965, 1005, 1165, 1182), (1540, 1005, 1735, 1182)], chest_pipe=(1130, 1572, 1450, 1500),
    pipe_w=8.0,
    sleeve_seed=(300, 1100),     # the upper sleeve island; its corner reaches into the left boot box
)


def suit_old_prep():
    rgb, alpha = load_src("DRIVER_Suit.dds")
    H, W = rgb.shape[:2]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    V = rgb.max(-1)
    sat = (V - rgb.min(-1)) / (V + 1e-3)
    yy, xx = np.mgrid[0:H, 0:W]
    white = (sat < 0.25) & (V > 110)
    gray = (sat < 0.25) & (V > 40) & (V <= 110)
    bg = V < 12
    inbox = lambda b: (xx >= b[0]) & (xx < b[2]) & (yy >= b[1]) & (yy < b[3])
    # old logos: dark/colour pixels on the white chest band, white pixels of the BR mark on blue
    logo = np.zeros((H, W), bool)
    for k in ("br03", "smp"):
        logo |= inbox(SO["logos"][k][0]) & ~white
    logo |= inbox(SO["logos"]["br"][0]) & (white | gray)
    logo = dilate(logo, 2)
    # white structure: belt rows (white across the whole torso width), thin piping
    rows = np.flatnonzero((white[:, 650:2040].mean(1) > 0.93))
    rows = rows[(rows > 1500) & (rows < 1850)]
    belt = white & (yy >= rows.min()) & (yy <= rows.max()) & (xx >= 630)
    # piping -> smooth vector lines fitted to the original (the thresholded mask was stair-stepped and broken)
    wide = ndimage.binary_opening(white, iterations=6)           # collar, side panels, belt
    thin = white & ~wide
    pipes = []
    for b in SO["piping"]:                                       # straight collar -> shoulder lines
        ys, xs = np.nonzero(thin & inbox(b))
        k = np.polyfit(xs, ys, 1)
        keep = np.abs(ys - np.polyval(k, xs)) < 6
        k = np.polyfit(xs[keep], ys[keep], 1)
        px = np.arange(xs[keep].min() - 14, xs[keep].max() + 14.5, 1.0)
        pipes.append(np.stack([px, np.polyval(k, px)], 1))
    x0, x1, y0, y1 = SO["chest_pipe"]                            # curved chest line: quadratic
    cp = []
    for x in range(x0, x1):
        idx = np.flatnonzero(white[y0:y1, x])
        if len(idx):
            r = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)[-1]
            cp.append((x, y0 + (r[0] + r[-1]) / 2))
    cp = np.array(cp, np.float64)
    k = np.polyfit(cp[:, 0], cp[:, 1], 2)
    px = np.arange(x0 - 14, x1 + 14.5, 1.0)
    pipes.append(np.stack([px, np.polyval(k, px)], 1))
    keep_out = dilate(wide, 1) | bg
    pipe_line = np.maximum.reduce([line_mask((H, W), pp, SO["pipe_w"]) for pp in pipes]) * ~keep_out
    piping = dilate(pipe_line > 0.05, 3) & ~keep_out             # zone: old piping + its anti-aliased rim -> cloth
    logo &= ~piping
    # UV islands (separated by the black background): right boot, soles and shoulder epaulettes are islands of
    # their own; the left boot touches the torso island -> cut by its box
    isl, n = ndimage.label(~bg)
    comp = lambda x, y: ndimage.binary_fill_holes(ndimage.binary_closing(isl == isl[y, x], iterations=4))
    legs = isl == isl[300, 400]
    sl_lab, _ = ndimage.label(~bg & (V >= 90))                   # bright sleeve cloth (the boot is darker)
    sleeve = dilate((sl_lab == sl_lab[SO["sleeve_seed"][1], SO["sleeve_seed"][0]]) & (xx < 680), 2)
    boots = (inbox(SO["boots"][0]) & ~((xx > 860) & (yy > 850)) & ~legs & ~sleeve) | comp(*SO["boot_seed"])
    boots &= ~bg | comp(*SO["boot_seed"])
    soles = comp(*SO["sole_seeds"][0]) | comp(*SO["sole_seeds"][1])
    epaul = comp(*SO["epaulettes"][0]) | comp(*SO["epaulettes"][1])
    groups = np.zeros((H, W), np.int16)          # 0 blue cloth
    groups[white | (logo & inbox(SO["logos"]["br03"][0])) | (logo & inbox(SO["logos"]["smp"][0]))] = 1
    groups[logo & inbox(SO["logos"]["br"][0])] = 0
    groups[gray & ~logo & boots] = 2
    groups[gray & ~logo & ~boots] = 1      # grey = folds of the white print
    groups[bg & ~boots & ~soles] = 3
    groups[piping & (groups == 1)] = 0
    Vi = inpaint(V, logo)
    S = shade_map(Vi, groups, flat=(3,))
    # piping zone: cloth shading from around it (the old white line and its rim would print as stripes)
    S[piping] = blur_n(S, ~piping & (groups == 0) & ~bg, 5)[piping]
    return dict(rgb=rgb, alpha=alpha, V=V, S=S, groups=groups, belt=belt, piping=piping, pipe_line=pipe_line,
                boots=boots, soles=soles, epaul=epaul, white=white, bg=bg, shape=(H, W))


def boundary_line(mask, axis, lo, hi, start, step_dir, d_in=14, smooth_win=15, max_jump=8, min_len=40):
    """Polylines along the first mask->not-mask transition when scanning from `start` in `step_dir` (+1/-1) along
    rows (axis=1, for each y in lo..hi) or columns (axis=0, for each x), shifted d_in px past the transition.
    Split wherever the edge jumps (a different panel was hit); pieces shorter than min_len px are dropped."""
    pts = []
    for t in range(lo, hi, 2):
        line = mask[t, :] if axis == 1 else mask[:, t]
        idx = np.arange(start, len(line) if step_dir > 0 else -1, step_dir)
        v = line[idx]
        on = np.flatnonzero(v)
        if not len(on):
            pts.append(None)
            continue
        off = np.flatnonzero(~v[on[0]:])
        if not len(off):
            pts.append(None)
            continue
        e = idx[on[0] + off[0]] + step_dir * d_in
        pts.append((e, t) if axis == 1 else (t, e))
    segs, cur = [], []
    for p in pts:
        if p is None or (cur and math.hypot(p[0] - cur[-1][0], p[1] - cur[-1][1]) > max_jump):
            if cur:
                segs.append(cur)
            cur = [p] if p is not None else []
        else:
            cur.append(p)
    if cur:
        segs.append(cur)
    out = []
    for sgm in segs:
        a = np.array(sgm, np.float32)
        if len(a) > 3 and np.hypot(*np.diff(a, axis=0).T).sum() >= min_len:
            out.append(smooth(a, smooth_win))
    return out


def suit_old_design(c):
    H, W = c["shape"]
    g = c["groups"]
    A = pearl_cloth((H, W), seed=7)
    ink = (g == 1) & ~c["belt"] & ~c["piping"]
    put(A, ink, INK)
    put(A, c["pipe_line"], PEARL)
    # belt: holographic, pearl keylines on both edges
    yy, xx = np.mgrid[0:H, 0:W]
    belt = c["belt"]
    ys = np.flatnonzero(belt.any(1))
    holo = holo_field((H, W), (630, ys.min()), (1.0, 0.0), period=700.0, phase=0.12, across=(0, 1), across_k=1 / 400)
    put(A, belt, holo)
    kl = belt & ((yy <= ys.min() + 6) | (yy >= ys.max() - 6)) & ((yy - ys.min() >= 1) & (ys.max() - yy >= 1))
    put(A, kl, PEARL)
    # boots: burgundy leather, grey side stripes -> holo; soles black
    put(A, c["boots"] & (g == 0), BOOT)
    put(A, c["boots"] & (g == 2), holo_field((H, W), (0, 0), (0.7, 0.7), period=160.0))
    put(A, c["soles"], BLACKS)
    put(A, g == 3, BLACKS)
    put(A, c["epaul"] & (g == 0), INK)          # shoulder epaulettes
    # dashed cut lines, 14 px inside the pink next to the maroon side panels (torso) and leg bands
    pink = (g == 0) & ~c["boots"] & ~c["soles"] & ~c["epaul"] & ~c["piping"]
    side = (g == 1) & ~belt & ~c["piping"]
    lines = []
    bt = int(ys.min()) - 2
    for (y0, y1) in ((1395, bt - 4), (int(ys.max()) + 6, 1990)):
        lines.append(boundary_line(side, 1, y0, y1, 860, +1))       # left side panel, inner edge
        lines.append(boundary_line(side, 1, y0, y1, 1830, -1))      # right side panel, inner edge
    for (x0, x1) in ((90, 770), (835, 1460)):                       # legs: upper / lower maroon bands
        lines.append(boundary_line(side, 0, x0, x1, 0, +1))
        lines.append(boundary_line(side, 0, x0, x1, 700, -1))
    dash = np.zeros((H, W), np.float32)
    for i, ln in enumerate([q for grp in lines for q in grp]):
        dash = np.maximum(dash, dash_mask((H, W), resample(ln, 2), width=7, on=34, off=20, phase=i * 9))
    return dict(A=A, dash=dash, pink=erode(pink, 3))


def make_suit_old(c, des):
    H, W = c["shape"]
    prints = Prints((H, W))
    L = SO["logos"]
    # chest band (burgundy now): BR03 -> «арка», SMP RACING -> SMP RACING ESPORTS (brand colours)
    bx = L["br03"][0]
    prints.place("chest арка", fit_box(asset_arka(72), bx[2] - bx[0] - 10, 64),
                 ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), 0.0, bx)
    bx = L["smp"][0]
    prints.place("chest SMP RACING ESPORTS", fit_box(asset_smp(52), bx[2] - bx[0] - 6, 50),
                 ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), 0.0, bx)
    # BR mark (symbol over ENGINEERING) -> BR ENGINEERING stacked, burgundy on the pink
    bx = L["br"][0]
    prints.place("chest BR ENGINEERING", fit_box(asset_br_stacked(88, INK), 96, 88),
                 ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), 0.0, bx)
    pa = prints.alpha()
    dash = drop_stubs(des["dash"] * des["pink"] * (1 - dilate(pa > 0.1, 12)))
    return to_img(composite(des["A"], c["S"], prints, dash=dash)), prints.log


# ================================================================= 2016 gloves (2016_Gloves_DIFF.dds, 2048x1024)
# UV map: glove backs x 190..570 (left) and 1440..1830 (right) with the white SMP patch and the RU tricolour over
# a stitched gauntlet panel; palms (grey grip with dots) and the palm-side cuffs (zigzag stitching) in the middle;
# finger grips in the outer columns. Left/right halves are near mirror images (axis x≈1027.5), measured separately.
# Patch / gauntlet outlines: anchor points ON the thin dark outline line of the original (read off the texture),
# snapped to that line (snap_seam, pre-snapped anchors, sharp corners) -> closed outline polylines.
G16 = dict(
    panelL=((236, 264, 526, 527), -14.4),     # text angle measured from the glyphs (descends to the right)
    panelR=((1528, 263, 1819, 508), 12.6),
    patchL=[(236, 282), (277, 260), (298, 290), (360, 297), (420, 310), (480, 325), (520, 342), (526, 360),
            (516, 417), (504, 463), (490, 528), (453, 522), (387, 498), (320, 463), (261, 441), (255, 417),
            (243, 360), (236, 282)],
    patchR=[(1819, 282), (1778, 260), (1757, 289), (1695, 297), (1635, 310), (1575, 325), (1535, 342),
            (1526, 360), (1536, 417), (1548, 463), (1565, 527), (1602, 522), (1668, 498), (1735, 463), (1793, 441),
            (1798, 417), (1809, 360), (1819, 282)],
    # gauntlet panel: TL corner, top edge (4 pts to the TR corner), then clockwise / anticlockwise round the panel
    gauntL=[(247.5, 526), (320, 546), (370, 572), (430, 602), (482.5, 622.5), (484, 650), (490, 700), (502, 750),
            (520, 800), (546, 853), (520, 864), (480, 872), (420, 876), (370, 875), (320, 868), (270, 860),
            (230, 852), (190, 840), (205, 800), (230, 725), (245, 675), (250, 625), (249, 575), (247.5, 526)],
    gauntR=[(1569, 622.5), (1630, 592), (1730, 547.5), (1800, 523), (1799, 550), (1796, 600), (1796, 650),
            (1802.5, 700), (1820, 750), (1842, 800), (1864, 839), (1830, 855), (1780, 866), (1730, 873),
            (1680, 876), (1630, 874), (1580, 869), (1530, 862), (1508, 856), (1535, 800), (1555, 725), (1565, 675),
            (1569, 630), (1569, 622.5)],
    gaunt_top=5,                              # the first 5 anchors of gauntL / gauntR = the top edge, left -> right
    cuffL=[(603, 722), (640, 744), (700, 772), (760, 788), (820, 783), (880, 764), (937, 722), (958, 770),
           (1004, 968), (900, 990), (700, 996), (556, 976), (575, 850)],
    flag_x=((150, 620), (1430, 1900)),        # search windows of the two RU tricolours (y 520..830)
)
STITCH = (240, 170, 192)      # contrast thread on the burgundy panels (the original: grey thread on white / blue)


def mirror_pts(pts, axis=1027.5):
    return [(2 * axis - x, y) for x, y in pts]


def fit_flag(white, red, xr, yr=(520, 830)):
    """The printed RU tricolour is a rotated rectangle (white / blue / red, 3 x 33 px, ends cut square). Fit it:
    PCA direction of the white + red pixels, refined on the straight outer edges (top of white, bottom of red),
    ends from the red stripe (never shaded). Returns the 4 corners."""
    H, W = white.shape
    yy, xx = np.mgrid[0:H, 0:W]
    m = (white | red) & (yy > yr[0]) & (yy < yr[1]) & (xx >= xr[0]) & (xx < xr[1])
    lab, n = ndimage.label(m)
    m = np.isin(lab, 1 + np.flatnonzero(ndimage.sum(m, lab, range(1, n + 1)) >= 500))   # stripes, not grip dots
    ys, xs = np.nonzero(m)
    P = np.stack([xs, ys], 1).astype(np.float64)
    c = P.mean(0)
    d = np.linalg.svd(P - c, full_matrices=False)[2][0]
    d = d if d[0] > 0 else -d
    Wt, Rd = white[ys, xs], red[ys, xs]
    for _ in range(3):
        nrm = np.array([-d[1], d[0]])
        t, u = (P - c) @ d, (P - c) @ nrm
        slopes = []
        for sel, fn in ((Wt, np.min), (Rd, np.max)):
            tb, ub = [], []
            for t0 in np.arange(-170, 170, 4):
                k = sel & (t >= t0) & (t < t0 + 4)
                if k.sum() > 5:
                    tb.append(t0 + 2)
                    ub.append(fn(u[k]))
            tb, ub = np.array(tb), np.array(ub)
            keep = np.abs(ub - np.median(ub)) < 3
            slopes.append(np.polyfit(tb[keep], ub[keep], 1)[0])
        a = math.atan(float(np.mean(slopes)))
        d = np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])
    nrm = np.array([-d[1], d[0]])
    t, u = (P - c) @ d, (P - c) @ nrm
    t0, t1 = np.percentile(t[Rd], 0.3) - 0.5, np.percentile(t[Rd], 99.7) + 0.5
    u0, u1 = np.percentile(u[Wt], 0.3) - 0.5, np.percentile(u[Rd], 99.7) + 0.5
    return [tuple(c + t0 * d + u0 * nrm), tuple(c + t1 * d + u0 * nrm), tuple(c + t1 * d + u1 * nrm),
            tuple(c + t0 * d + u1 * nrm)]


def stitch_weight(V, region, outline, d0=3.0, d1=15.0, sig=2.5):
    """Baked stitch rows inside `region` (a band d0..d1 px from the outline line): pixels a little darker than their
    2.5 px neighbourhood (the thread), but not the near-black outline itself. 0..1."""
    dist = ndimage.distance_transform_edt(~(outline > 0.3))
    band = region & (dist >= d0) & (dist <= d1)
    loc = blur_n(V, region & (dist >= d0 - 0.5), sig)
    r = (V + 4) / (loc + 4)
    w = np.clip((0.95 - r) / 0.08, 0, 1) * np.clip((r - 0.55) / 0.1, 0, 1)
    return (w * band).astype(np.float32)


def gloves16_prep():
    rgb, alpha = load_src("2016_Gloves_DIFF.dds")
    H, W = rgb.shape[:2]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    V = rgb.max(-1)
    sat = (V - rgb.min(-1)) / (V + 1e-3)
    yy, xx = np.mgrid[0:H, 0:W]
    white = (sat < 0.2) & (V > 150)
    red = (R > G + 60) & (R > B + 40)
    gray = (sat < 0.35) & (V > 25) & (V <= 150)
    black = V < 25
    cost = seam_cost(V)
    snap = lambda an: snap_seam(cost, an, margin=6, win=5, step=1.0, pre=4)
    # white back patches: outline polygons; the text = dark pixels well inside (the stitch rows run 4..14 px in)
    out = {}
    for k in ("patchL", "patchR", "gauntL", "gauntR"):
        out[k] = snap(G16[k])
    top = {k: snap(G16[k][:G16["gaunt_top"]]) for k in ("gauntL",)}
    top["gauntR"] = snap(G16["gauntR"][:G16["gaunt_top"]])
    patch_cov = np.maximum(poly_mask((H, W), out["patchL"]), poly_mask((H, W), out["patchR"]))
    patch_line = np.maximum(line_mask((H, W), out["patchL"], 2.5), line_mask((H, W), out["patchR"], 2.5))
    patch_in = (patch_cov > 0.5) & (patch_line < 0.5)
    patch_side = {k: (poly_mask((H, W), out[p]) > 0.5) & patch_in for k, p in (("panelL", "patchL"),
                                                                              ("panelR", "patchR"))}
    text = dilate(erode(patch_in, 15) & (V < 150), 2)
    gaunt_cov = np.maximum(poly_mask((H, W), out["gauntL"]), poly_mask((H, W), out["gauntR"]))
    gaunt_line = np.maximum(line_mask((H, W), out["gauntL"], 2.5), line_mask((H, W), out["gauntR"], 2.5))
    # RU tricolours: fitted rectangles, anti-aliased coverage
    loose_white = (sat < 0.3) & (V > 100)
    flags = [fit_flag(loose_white, red, xr) for xr in G16["flag_x"]]
    tri_cov = np.maximum.reduce([poly_mask((H, W), f, ss=8) for f in flags])
    tri = tri_cov > 0.5
    grip = ndimage.binary_fill_holes(ndimage.binary_closing(gray & ~tri & ~patch_in, iterations=3))
    grip &= ~tri & ~(patch_cov > 0.5)
    groups = np.zeros((H, W), np.int16)       # 0 blue cloth
    groups[patch_in] = 1
    groups[grip] = 2
    groups[black & ~grip] = 3
    groups[tri] = 4
    Vi = inpaint(V, text)
    S = shade_map(Vi, groups, params={3: dict(k=14.0)})
    # baked stitch rows along the patch / gauntlet outlines (kept, re-coloured as contrast thread)
    st = np.maximum(stitch_weight(Vi, patch_in, patch_line),
                    stitch_weight(V, (gaunt_cov > 0.5) & (groups == 0), gaunt_line) * (1 - dilate(tri, 3)))
    return dict(rgb=rgb, alpha=alpha, V=V, S=S, groups=groups, tri=tri, tri_cov=tri_cov, flags=flags,
                patch_cov=patch_cov, patch_line=patch_line, patch_in=patch_in, patch_side=patch_side,
                gaunt_cov=gaunt_cov,
                gaunt_line=gaunt_line, gaunt_top=top, outlines=out, stitch=st, shape=(H, W))


def gloves16_design(c):
    H, W = c["shape"]
    g = c["groups"]
    A = pearl_cloth((H, W), seed=3)
    cuffs = np.maximum(poly_mask((H, W), G16["cuffL"]), poly_mask((H, W), mirror_pts(G16["cuffL"])))
    put(A, cuffs * (g == 0), INK)                       # palm-side cuffs (zigzag stitching)
    put(A, c["gaunt_cov"] * ((g == 0) | (g == 4)), INK)  # stitched gauntlet panels, exactly to their outline
    put(A, c["patch_cov"], INK)                         # back patches (were white)
    put(A, g == 2, (64, 16, 32))
    put(A, g == 3, BLACKS)
    put(A, np.maximum(c["patch_line"], c["gaunt_line"]) * (g != 3), INK_D)   # the dark outline lines stay dark
    put(A, c["stitch"], STITCH)
    # dashed cut line 14 px above each gauntlet panel (the wrist "cut"), parallel to its stitched top edge
    dash = np.zeros((H, W), np.float32)
    for k in ("gauntL", "gauntR"):
        tp = c["gaunt_top"][k]
        if tp[0, 0] > tp[-1, 0]:
            tp = tp[::-1]
        ln = offset(resample(tp, 2.0), -14)[4:-4]
        dash = np.maximum(dash, dash_mask((H, W), ln, width=6, on=26, off=16))
    cl = (g == 0) & (cuffs < 0.5) & (c["gaunt_cov"] < 0.5) & (c["patch_cov"] < 0.5)
    pink = erode(cl, 3)
    return dict(A=A, dash=dash, pink=pink, cuffs=cuffs)


def make_gloves16(c, des):
    H, W = c["shape"]
    prints = Prints((H, W))
    # back patches (now burgundy): left «арка», right SMP RACING ESPORTS, at the measured text angle
    for k, name in (("panelL", "glove арка"), ("panelR", "glove SMP RACING ESPORTS")):
        bx, ang = G16[k]
        ys, xs = np.nonzero(c["patch_side"][k])
        cx, cy = xs.mean(), ys.mean()
        art = fit_box(asset_arka(120), 230, 110) if "арка" in name else fit_box(asset_smp(80), 236, 76)
        prints.place(name, art, (cx, cy), ang, bx)
    pa = prints.alpha()
    dash = drop_stubs(des["dash"] * des["pink"] * (1 - dilate(pa > 0.1, 8)), on=26, width=6)
    # thread: keep the fold shading of the panel, not the darkening of the old grey thread
    S = c["S"].copy()
    st = c["stitch"]
    Sb = blur_n(S, (c["patch_cov"] > 0.5) | (c["gaunt_cov"] > 0.5), 3)
    S = S * (1 - st) + np.maximum(Sb, 0.85) * st
    g = c["groups"]
    S[g == 4] = blur_n(S, g == 0, 4)[g == 4]          # cloth shading under the flag edges (old flag shading out)
    out = composite(des["A"], S, prints, dash=dash)
    # RU tricolour: the original printed pixels inside the fitted rectangle; edge pixels anti-aliased against the
    # new cloth with the flag colour of the nearest core pixel (2 px in: the original's own anti-aliasing against
    # the blue cloth sits in the outer pixel rows, so those take the colour from inside -> no blue fringe)
    cov = c["tri_cov"]
    core = erode(cov >= 0.999, 2)
    _, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
    flag_rgb = np.where(core[..., None], c["rgb"], c["rgb"][iy, ix])
    out = out * (1 - cov[..., None]) + flag_rgb * cov[..., None]
    return to_img(out), prints.log


# ================================================================= old gloves (DRIVER_Gloves.dds, 512x256)
def make_gloves_old():
    rgb, alpha = load_src("DRIVER_Gloves.dds")
    H, W = rgb.shape[:2]
    V = rgb.max(-1)
    # flat-shaded greys: 5 background, 2 black palm, ~19 back of hand, ~42 palm pads, 7..15 seams/outlines
    groups = np.full((H, W), 0, np.int16)
    groups[V <= 3] = 1
    groups[(V >= 16) & (V <= 26)] = 2
    groups[V >= 30] = 3
    groups[(V >= 7) & (V <= 15)] = 4
    groups[(V >= 4) & (V <= 6)] = 0
    S = shade_map(V, groups, params={g: dict(sig_f=4, sig_m=12, sig_lo=40, k=4.0) for g in range(5)}, flat=(0,))
    A = np.zeros((H, W, 3), np.float32)
    A[:] = (30, 10, 18)
    put(A, groups == 2, CLOTH)
    put(A, groups == 3, INK)
    put(A, groups == 1, INK_D)
    put(A, groups == 4, (34, 9, 18))
    return to_img(apply_shade(A, S)), []


# ================================================================= helmet (HELMET_2012.dds, 2048x512)
# UV map: main shell strip x 565..2048 (chin bar in the middle with the vent, sides to the back at both ends),
# left column: visor pivot plates (red, screws), crown oval, rear spoiler, visor seal strip. Original logos: the two
# white SMP RACING plates on the cheeks. No number spot -> no number. Safety stickers, screws, vents untouched.
HM = dict(plates=dict(L=((960, 337, 1183, 403), -3.4), R=((1443, 336, 1667, 404), 2.5)),
          pin_y=(20, 42),
          # metal kept as is: visor pivot screws, chin-vent screws, side bolts; FIA homologation stickers
          keep_circles=[(81.5, 74, 43), (235, 77.5, 43), (1286, 203, 19), (1255, 245, 19), (750, 382, 27),
                        (1877, 382, 27)],
          keep_boxes=[(583, 395, 632, 442), (1934, 395, 2000, 442)])


def make_helmet():
    rgb, alpha = load_src("HELMET_2012.dds")
    H, W = rgb.shape[:2]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    V = rgb.max(-1)
    sat = (V - rgb.min(-1)) / (V + 1e-3)
    yy, xx = np.mgrid[0:H, 0:W]
    white = (sat < 0.2) & (V > 150)
    red = (R > G + 60) & (R > B + 40)
    blue = (B > R + 40) & (B > G + 20)
    plates = {}
    text = np.zeros((H, W), bool)
    for k, ((x0, y0, x1, y1), _) in HM["plates"].items():
        pm = white & (xx >= x0 - 8) & (xx <= x1 + 8) & (yy >= y0 - 8) & (yy <= y1 + 8)
        lab, n = ndimage.label(pm)
        big = lab == (np.argmax(ndimage.sum(pm, lab, range(1, n + 1))) + 1)
        reg = ndimage.binary_fill_holes(ndimage.binary_closing(big, iterations=3))
        plates[k] = reg
        text |= reg & ~white
    plate = plates["L"] | plates["R"]
    # thin red pinstripe across the top of the shell -> becomes the dashed cut line
    thin_red = red & ~ndimage.binary_opening(red, iterations=3)
    pin = thin_red & (yy >= HM["pin_y"][0]) & (yy <= HM["pin_y"][1]) & (xx >= 565)
    groups = np.full((H, W), -1, np.int16)    # -1 = keep original pixels (black, metal, stickers, vents)
    groups[white & ~plate] = 0
    groups[red & ~pin] = 1
    groups[blue & ~plate] = 2
    groups[plate] = 3
    groups[pin] = 0
    for (cx, cy, r) in HM["keep_circles"]:
        groups[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = -1
    for (x0, y0, x1, y1) in HM["keep_boxes"]:
        groups[(xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)] = -1
    Vi = inpaint(V, dilate(text, 1) | dilate(pin, 1))
    work = groups.copy()
    work[work < 0] = 9
    S = shade_map(Vi, work, params={g: dict(sig_f=6, sig_m=24, sig_lo=90, c_m=0.8) for g in range(4)}, flat=(9,))
    A = rgb.copy()
    SHELL = PIG + np.array((2, 0, 4), np.float32)
    # pearl drift along the strip (warm at the chin, cooler towards the back), like the car's pearl
    t = np.abs((xx - 1300) / 760.0).clip(0, 1)[..., None]
    shell = SHELL * (1 - t) + np.array((230, 150, 196), np.float32) * t
    put(A, groups == 0, shell)
    put(A, groups == 1, INK)
    holo = holo_field((H, W), (565, 300), (0.96, -0.28), period=520.0, phase=0.3)
    put(A, groups == 2, holo)
    put(A, groups == 3, INK)
    # pearl hairline around the plates (inner 3 px of the plate)
    rim = plate & ~erode(plate, 3)
    put(A, rim, PEARL)
    # dashed cut line on the old pinstripe
    ys, xs = np.nonzero(pin)
    k = np.polyfit(xs, ys, 1)
    ln = resample([(565, np.polyval(k, 565)), (2048, np.polyval(k, 2048))], 2)
    dash = dash_mask((H, W), ln, width=6, on=30, off=18) * ((groups == 0) | dilate(pin, 3))
    prints = Prints((H, W))
    for k_, name in (("L", "helmet арка"), ("R", "helmet SMP RACING ESPORTS")):
        bx, ang = HM["plates"][k_]
        reg = plates[k_]
        ys, xs = np.nonzero(erode(reg, 6))
        art = fit_box(asset_arka(80), 150, 50) if "арка" in name else fit_box(asset_smp(52), 160, 46)
        prints.place(name, art, (xs.mean(), ys.mean()), ang, bx)
    put(A, dash, INK)
    pa = prints.alpha()
    put(A, pa, prints.rgb())
    keep = groups < 0
    S_eff = S ** (1 - 0.6 * pa)
    out = apply_shade(A, S_eff, hi=0.6)
    out[keep & (pa < 0.01)] = rgb[keep & (pa < 0.01)]
    return to_img(out), prints.log


def make_helmet_glass():
    """Visor strip: the dark sun band at the top of the visor with the partner logo in the middle."""
    rgb, alpha = load_src("HELMET_2012_Glass.dds")
    H, W = rgb.shape[:2]
    V = rgb.max(-1)
    yy, xx = np.mgrid[0:H, 0:W]
    band = yy < 56
    logo = dilate((V > 110) & band, 2)
    Vi = inpaint(V, logo)
    out = rgb.copy()
    # band: same luminance profile, tinted deep burgundy; a thin holo line along its lower edge
    tint = np.array((1.25, 0.42, 0.72), np.float32)
    out[band] = np.clip(Vi[band][:, None] * tint, 0, 255)
    holo = holo_field((H, W), (0, 0), (1, 0), period=260.0)
    hl = band & (yy >= 50) & (yy <= 51)
    out[hl] = holo[hl] * 0.85
    prints = Prints((H, W))
    art = fit_box(asset_arka(64), 118, 34)
    prints.place("visor арка", art, ((196 + 315) / 2, (11 + 44) / 2), 0.0, (196, 11, 315, 44))
    pa = prints.alpha()[..., None]
    out = out * (1 - pa) + prints.rgb() * pa
    return to_img(out), prints.log


# ================================================================= checks + preview
TEXTURES = ["2016_Suit_DIFF.dds", "DRIVER_Suit.dds", "2016_Gloves_DIFF.dds", "DRIVER_Gloves.dds", "HELMET_2012.dds",
            "HELMET_2012_Glass.dds"]
SKIPPED = {"DRIVER_Suit2.dds": "not a colour layer (grey spec / maps layer of DRIVER_Suit) - left as is"}


def check_dds(name, path):
    """Same header (size, flags, mip count), complete chain, decoded alpha identical to the original (mip 0)."""
    hdr, w, h, n = dds_header(os.path.join(SRC, name))
    data = open(path, "rb").read()
    exp = 128 + sum(nb for _, _, nb in level_dims(w, h, n))
    o = np.asarray(Image.open(os.path.join(SRC, name)).convert("RGBA").getchannel("A"))
    im = Image.open(path)
    im.load()
    a = np.asarray(im.convert("RGBA").getchannel("A"))
    # alpha blocks byte-identical on every level that the original file holds completely
    ab_o = original_alpha_blocks(name)
    off, same_levels = 128, 0
    for i, (lw, lh, nb) in enumerate(level_dims(w, h, n)):
        blk = np.frombuffer(data[off:off + nb], np.uint8).reshape(-1, 16)[:, :8]
        if ab_o[i] is not None and np.array_equal(blk, ab_o[i]):
            same_levels += 1
        off += nb
    return dict(file=name, size=im.size, size_ok=im.size == (w, h), header_same=data[:128] == hdr, mips=n,
                length_ok=len(data) == exp, alpha_max_diff=int(np.abs(a.astype(int) - o.astype(int)).max()),
                alpha_levels_identical=f"{same_levels}/{n}", fourcc=data[84:88].decode())


def _font(sz, w="Bold"):
    f = ImageFont.truetype(os.path.join(FONTS, "Exo2-Italic[wght].ttf"), sz)
    f.set_variation_by_name(w + " Italic" if w != "Regular" else "Italic")
    return f


def preview(skin, results, checks):
    """Labelled contact sheet: original | restyle (RGB) | alpha of the written DDS + check line."""
    BG, FG, MUTE = (22, 18, 24), (240, 232, 238), (170, 150, 165)
    pink = tuple(int(v) for v in PIG)
    colw, aw, pad = 860, 200, 28
    rows = []
    for name in TEXTURES + list(SKIPPED):
        src = Image.open(os.path.join(SRC, name)).convert("RGBA")
        k = colw / src.width
        th = (colw, max(1, round(src.height * k)))
        if name in SKIPPED and name not in results:
            th = (max(1, round(src.width * 320 / src.height)), 320)
            rows.append((name, src.convert("RGB").resize(th, Image.LANCZOS), None, None, SKIPPED[name]))
            continue
        new = Image.open(os.path.join(OUT, skin, name)).convert("RGBA")
        ak = aw / src.width
        at = new.getchannel("A").resize((aw, max(1, round(src.height * ak))), Image.LANCZOS)
        rows.append((name, src.convert("RGB").resize(th, Image.LANCZOS), new.convert("RGB").resize(th, Image.LANCZOS),
                     at, checks[name]))
    W = pad * 4 + colw * 2 + aw
    H = 150 + sum(r[1].height + 92 for r in rows) + pad
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 26), f"{skin}  ·  driver gear  ·  Butcher Chart Chrome", font=_font(46, "Black"), fill=pink)
    d.text((pad, 88), "left: original SMP01 texture   ·   middle: restyle (RGB)   ·   right: alpha channel of the "
                      "written DDS (spec/mask, must equal the original)", font=_font(22, "Regular"), fill=MUTE)
    y = 150
    for name, o, n, a, info in rows:
        d.text((pad, y), name, font=_font(30, "Black"), fill=FG)
        if isinstance(info, dict):
            line = (f"{info['size'][0]}×{info['size'][1]}  DXT5  mips {info['mips']}  ·  header identical: "
                    f"{'yes' if info['header_same'] else 'NO'}  ·  alpha max diff {info['alpha_max_diff']}  ·  "
                    f"alpha blocks identical on {info['alpha_levels_identical']} levels")
        else:
            line = info
        d.text((pad + 520, y + 6), line, font=_font(22, "Regular"), fill=MUTE)
        y += 46
        sheet.paste(o, (pad, y))
        if n is not None:
            sheet.paste(n, (pad * 2 + colw, y))
            sheet.paste(a.convert("RGB"), (pad * 3 + colw * 2, y))
            d.rectangle([pad * 3 + colw * 2 - 1, y - 1, pad * 3 + colw * 2 + a.width, y + a.height],
                        outline=(90, 70, 85))
            d.text((pad * 3 + colw * 2, y + a.height + 6), "alpha (new file)", font=_font(18, "Regular"), fill=MUTE)
        else:
            d.text((pad * 2 + colw + 20, y + 20), "not restyled", font=_font(34, "Black"), fill=MUTE)
        y += o.height + 46
    sheet.save(os.path.join(HERE, f"gear_preview_{skin}.png"), optimize=True)


def main():
    print("2016 suit ...")
    c16 = suit16_prep()
    d16 = suit16_design(c16)
    print("old suit ...")
    co = suit_old_prep()
    do = suit_old_design(co)
    old_suit, log_o = make_suit_old(co, do)
    print("gloves ...")
    cg = gloves16_prep()
    dg = gloves16_design(cg)
    gl16, log_g = make_gloves16(cg, dg)
    glo, _ = make_gloves_old()
    print("helmet ...")
    hel, log_h = make_helmet()
    vis, log_v = make_helmet_glass()
    for skin, driver in DRIVERS:
        od = os.path.join(OUT, skin)
        os.makedirs(od, exist_ok=True)
        s16, log_s = make_suit16(c16, d16, driver)
        imgs = {"2016_Suit_DIFF.dds": s16, "DRIVER_Suit.dds": old_suit, "2016_Gloves_DIFF.dds": gl16,
                "DRIVER_Gloves.dds": glo, "HELMET_2012.dds": hel, "HELMET_2012_Glass.dds": vis}
        checks = {}
        for name, im in imgs.items():
            save_like_original(im, name, os.path.join(od, name))
            checks[name] = check_dds(name, os.path.join(od, name))
        preview(skin, imgs, checks)
        print(f"\n== {skin} ({driver[0]} {driver[1]})")
        for ch in checks.values():
            print("  ", ch)
        if skin == DRIVERS[0][0]:
            print("  placements:")
            for tex, lg in (("2016_Suit_DIFF", log_s), ("DRIVER_Suit", log_o), ("2016_Gloves_DIFF", log_g),
                            ("HELMET_2012", log_h), ("HELMET_2012_Glass", log_v)):
                for e in lg:
                    print(f"    {tex:<18} {e['name']:<28} centre {e['center']} rot {e['angle']:>7}°  bbox {e['bbox']}"
                          f"  (original logo bbox {e['replaces']})")
    print("\n  skipped:", SKIPPED)


if __name__ == "__main__":
    main()
