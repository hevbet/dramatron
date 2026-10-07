"""Designer images from zones.json: zones_overview.png, text_angle.png, density_map.png, islands.png,
text_test/Skin.png (+ .dds) = every safe_box stamped with its zone name at text_rotation (for in-game check)."""
import colorsys
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
Z = json.load(open(os.path.join(HERE, "zones.json")))
zones = Z["zones"]
zone_map = np.load(os.path.join(HERE, "zone_map.npy"))
ang_map = np.load(os.path.join(HERE, "ang_map.npy"))
D = np.load(os.path.join(HERE, "_analysis.npz"))
FONT = "/home/user/dramatron/livery/br03/brand/fonts/Exo2-Italic[wght].ttf"
SRC = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad/rs3/SMP01/Skin.dds"
OUT = 2048
R = zone_map.shape[0]
S = OUT / 4096


def font(sz, w=800):
    f = ImageFont.truetype(FONT, sz)
    try:
        f.set_variation_by_axes([w])
    except Exception:
        pass
    return f


def grid(dr, size, col=(255, 255, 255, 70)):
    c = size / 16
    f = font(int(c * 0.16))
    for i in range(17):
        dr.line([(i * c, 0), (i * c, size)], fill=col, width=1)
        dr.line([(0, i * c), (size, i * c)], fill=col, width=1)
    for i in range(16):
        for j in range(16):
            dr.text((i * c + 3, j * c + 2), f"{chr(65 + i)}{j + 1}", fill=(255, 255, 255, 110), font=f)


def zcolor(i, n, p):
    if p == "hidden":
        return (60, 60, 60)
    h = (i * 0.618034) % 1
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def stamp(label, w, h, rot, col=(0, 0, 0), bg=None):
    """upright text block sized to fit w x h AFTER rotation; returns RGBA rotated."""
    # dimensions of the upright block such that rotated bbox fits w x h
    r = math.radians(rot)
    c, s = abs(math.cos(r)), abs(math.sin(r))
    # choose upright block (bw, bh) with aspect from text; solve fit
    f = font(100)
    tw, th = f.getbbox(label)[2], 120
    asp = th / tw
    # rotated bbox: W = bw*c + bh*s, H = bw*s + bh*c with bh = asp*bw
    k = min(w / (c + asp * s + 1e-9), h / (s + asp * c + 1e-9))
    bw = max(int(k), 8)
    fs = max(int(100 * bw / tw * 0.95), 6)
    f = font(fs)
    bb = f.getbbox(label)
    im = Image.new("RGBA", (bb[2] + 8, bb[3] + 8), bg or (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((4, 2), label, fill=col, font=f)
    d.line([(4, im.height - 3), (im.width - 4, im.height - 3)], fill=(255, 0, 0, 255), width=max(2, fs // 12))  # baseline marker
    d.polygon([(im.width - 4, im.height // 2 - fs // 4), (im.width + 0, im.height // 2), (im.width - 4, im.height // 2 + fs // 4)], fill=(255, 0, 0, 255))
    return im.rotate(rot, expand=True, resample=Image.BICUBIC)


def overview():
    src = Image.open(SRC).convert("RGB").resize((OUT, OUT))
    base = Image.blend(src, Image.new("RGB", src.size, (0, 0, 0)), 0.55).convert("RGBA")
    zm = np.asarray(Image.fromarray(zone_map.astype(np.int16) + 1).resize((OUT, OUT), Image.NEAREST)).astype(int) - 1
    col = np.zeros((OUT, OUT, 4), np.uint8)
    for i, z in enumerate(zones):
        col[zm == i] = (*zcolor(i, len(zones), z["part"]), 170)
    # zone edges
    e = (zm != np.roll(zm, 1, 0)) | (zm != np.roll(zm, 1, 1))
    col[e & (zm >= 0)] = (255, 255, 255, 230)
    im = Image.alpha_composite(base, Image.fromarray(col))
    dr = ImageDraw.Draw(im)
    grid(dr, OUT)
    f = font(17, 700)
    for i, z in enumerate(zones):
        if z["safe_box"]:
            b = [v * S for v in z["safe_box"]]
            dr.rectangle(b, outline=(255, 255, 0, 255), width=2)
        cx, cy = [v * S for v in z["safe_center"]]
        rot = z["text_rotation"]
        lab = f"{z['id']}  r{rot:+.0f}"
        tw = dr.textlength(lab, font=f)
        dr.rectangle([cx - tw / 2 - 3, cy - 11, cx + tw / 2 + 3, cy + 11], fill=(0, 0, 0, 200))
        dr.text((cx - tw / 2, cy - 10), lab, fill=(255, 255, 255, 255), font=f)
    # legend
    f2 = font(22, 800)
    dr.rectangle([1500, 1880, 2044, 2044], fill=(0, 0, 0, 220))
    dr.text((1510, 1888), "RS3 LMS Skin.dds zone map (4096 -> shown 2048)", fill=(255, 255, 255), font=font(16, 700))
    dr.text((1510, 1912), "yellow box = safe_box (text/logo area)", fill=(255, 255, 0), font=font(16, 600))
    dr.text((1510, 1934), "rN = PIL rotate() angle for upright text", fill=(255, 255, 255), font=font(16, 600))
    dr.text((1510, 1956), "grey = hidden; cells A..P / 1..16 = UV-test grid", fill=(200, 200, 200), font=font(16, 600))
    dr.text((1510, 1980), "LEFT side of car = right half (x>2048)", fill=(255, 200, 120), font=font(16, 700))
    dr.text((1510, 2002), "RIGHT side of car = left half", fill=(255, 200, 120), font=font(16, 700))
    im.convert("RGB").save(os.path.join(HERE, "zones_overview.png"), optimize=True)


def text_angle():
    a = np.asarray(Image.fromarray(np.nan_to_num(ang_map, nan=999).astype(np.float32)).resize((OUT, OUT), Image.NEAREST))
    cov = a < 900
    h = ((a % 360) / 360.0)
    hsv = np.stack([h, np.full_like(h, 0.75), np.full_like(h, 0.9)], -1)
    rgb = np.asarray(Image.fromarray((hsv * 255).astype(np.uint8), "HSV").convert("RGB")).copy()
    rgb[~cov] = 25
    im = Image.fromarray(rgb).convert("RGBA")
    dr = ImageDraw.Draw(im)
    grid(dr, OUT, (255, 255, 255, 50))
    # arrows: text 'up' direction every 64 px
    step = 48
    for yy in range(step // 2, OUT, step):
        for xx in range(step // 2, OUT, step):
            if not cov[yy, xx]:
                continue
            r = math.radians(a[yy, xx])
            dx, dy = -math.sin(r), -math.cos(r)  # text up in image coords
            L = 16
            x1, y1 = xx + dx * L, yy + dy * L
            dr.line([(xx - dx * L * 0.6, yy - dy * L * 0.6), (x1, y1)], fill=(0, 0, 0, 255), width=2)
            px, py = -dy, dx
            dr.polygon([(x1 + dx * 6, y1 + dy * 6), (x1 + px * 5, y1 + py * 5), (x1 - px * 5, y1 - py * 5)], fill=(0, 0, 0, 255))
    # wheel legend
    f = font(16, 700)
    cx, cy, rr = 1960, 1960, 60
    for deg in range(0, 360, 2):
        r = math.radians(deg)
        c = tuple(int(v * 255) for v in colorsys.hsv_to_rgb(deg / 360, 0.75, 0.9))
        dr.line([(cx, cy), (cx - math.sin(r) * rr, cy - math.cos(r) * rr)], fill=c, width=3)
    dr.text((1800, 1880), "hue = rotate() angle", fill=(255, 255, 255), font=f)
    dr.text((1800, 1900), "arrow = text UP", fill=(255, 255, 255), font=f)
    for z in zones:
        if z["visibility"] == "hidden":
            continue
        cx2, cy2 = [v * S for v in z["safe_center"]]
        lab = f"{z['text_rotation']:+.0f}"
        dr.rectangle([cx2 - 20, cy2 - 10, cx2 + 20, cy2 + 10], fill=(255, 255, 255, 220))
        dr.text((cx2 - 17, cy2 - 9), lab, fill=(0, 0, 0), font=f)
    im.convert("RGB").save(os.path.join(HERE, "text_angle.png"), optimize=True)


def density_and_islands():
    dens = D["dens"]; an = D["aniso"]; idm = D["idm"]; lab = D["lab"]
    med = Z["density_median_px_per_m"]
    cov = idm >= 0
    t = np.where(cov, idm, 0)
    rel = np.where(cov, dens[t] / med, np.nan)
    a = np.where(cov, an[t], np.nan)
    rgb = np.zeros((R, R, 3), np.uint8) + 25
    ok = cov & (np.abs(rel - 1) <= 0.18) & (a <= 1.25)
    rgb[ok] = (60, 150, 80)
    rgb[cov & ~ok & (rel > 1.18)] = (230, 60, 60)        # more texels per metre (compressed on car)
    rgb[cov & ~ok & (rel < 0.82)] = (60, 110, 240)       # fewer texels (stretched on car)
    rgb[cov & ~ok & (np.abs(rel - 1) <= 0.18)] = (240, 200, 40)  # anisotropic shear
    im = Image.fromarray(rgb).resize((OUT, OUT), Image.NEAREST).convert("RGBA")
    dr = ImageDraw.Draw(im)
    grid(dr, OUT, (255, 255, 255, 50))
    f = font(18, 700)
    dr.text((20, 2000), f"green = uniform density ({med} px/m at 4096); red = denser, blue = stretched, yellow = sheared (>25%)",
            fill=(255, 255, 255), font=f)
    im.convert("RGB").save(os.path.join(HERE, "density_map.png"), optimize=True)
    # islands
    rgb = np.zeros((R, R, 3), np.uint8) + 25
    li = np.where(cov, lab[t], -1)
    for l in range(lab.max() + 1):
        h = (l * 0.618034) % 1
        rgb[li == l] = [int(v * 255) for v in colorsys.hsv_to_rgb(h, 0.6, 0.85)]
    e = (li != np.roll(li, 1, 0)) | (li != np.roll(li, 1, 1))
    rgb[e & (li >= 0)] = 255
    im = Image.fromarray(rgb).resize((OUT, OUT), Image.NEAREST).convert("RGBA")
    dr = ImageDraw.Draw(im)
    grid(dr, OUT, (255, 255, 255, 50))
    for isl in Z["islands"]:
        if isl["area_m2"] < 0.01:
            continue
        b = isl["bbox"]
        cx, cy = (b[0] + b[2]) / 2 * S, (b[1] + b[3]) / 2 * S
        top = next(iter(isl["parts"]), "")
        lab_ = f"#{isl['island']} {top}"
        tw = dr.textlength(lab_, font=f)
        dr.rectangle([cx - tw / 2 - 2, cy - 10, cx + tw / 2 + 2, cy + 11], fill=(0, 0, 0, 200))
        dr.text((cx - tw / 2, cy - 10), lab_, fill=(255, 255, 255), font=f)
    im.convert("RGB").save(os.path.join(HERE, "islands.png"), optimize=True)


def text_test():
    """Skin with every zone filled flat + its safe box stamped with the zone id at text_rotation."""
    os.makedirs(os.path.join(HERE, "text_test"), exist_ok=True)
    N = 4096
    im = Image.new("RGB", (N, N), (30, 30, 34))
    for i, z in enumerate(zones):
        m = Image.open(os.path.join(HERE, z["mask"]))
        im.paste(Image.new("RGB", (N, N), zcolor(i, len(zones), z["part"])), (0, 0), m)
    for z in zones:
        if not z["safe_box"] or z["visibility"] == "hidden":
            continue
        x0, y0, x1, y1 = z["safe_box"]
        ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=6)
        st = stamp(z["id"].upper(), x1 - x0 - 16, y1 - y0 - 16, z["text_rotation"])
        im.paste(st, ((x0 + x1) // 2 - st.width // 2, (y0 + y1) // 2 - st.height // 2), st)
    im.save(os.path.join(HERE, "text_test", "Skin.png"))
    # glass sticker test (wing + windscreen + windows)
    g = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    for gz in Z["glass_sticker_zones"]:
        x0, y0, x1, y1 = gz["bbox"]
        ImageDraw.Draw(g).rectangle([x0, y0, x1, y1], fill=(250, 250, 250, 255), outline=(255, 0, 0, 255), width=3)
        lab = gz["kind"].upper().replace("WING_", "").replace("ENDPLATE_", "EP ")
        st = stamp(lab, x1 - x0 - 10, y1 - y0 - 10, gz["text_rotation"])
        if gz.get("mirrored_fraction", 0) > 0.5:
            from PIL import ImageOps
            st = ImageOps.mirror(st)
        g.paste(st, ((x0 + x1) // 2 - st.width // 2, (y0 + y1) // 2 - st.height // 2), st)
    g.save(os.path.join(HERE, "text_test", "glass_sticker.png"))


if __name__ == "__main__":
    overview()
    text_angle()
    density_and_islands()
    text_test()
    import projector
    for v in projector.VIEWS:
        projector.elevation(v).save(os.path.join(HERE, f"elevation_{v}.png"))
    print("ok")
