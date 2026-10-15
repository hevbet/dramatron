"""Team banner «Команда ЭДМ» (4 cars: 2x BR03 + 2x Audi RS3): title over the 2400x800 render of
tools/team_render_mixed.py.  Copy of the banner B overlay (scratchpad team_page/B/overlay.py) in the same
team_render title style - «КОМАНДА ЭДМ» + small «РОЗОВЫЙ ПЯТАЧОК» with holo diamonds and dashed butcher-chart
rules - without the driver tags (no driver names on this banner).

python3 overlay_team.py RAW.png OUT_MASTER.png OUT.jpg PREVIEW.png [--no-sub] [--guides]
"""
import argparse
import io
import sys

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFilter

sys.path.insert(0, "/home/user/dramatron/livery/tools")
import team_render as tr  # noqa: E402  (fonts, holo strip, letter-spaced text)

PINK = (255, 150, 198, 255)            # team line (draw_title)
PEARL = (244, 232, 242, 255)           # subtitle
MAROON = (170, 34, 68, 255)            # dashed butcher-chart rule
SHADOW = (70, 0, 30)
TITLE_Y = 180          # title baseline (master px)


def soft_shadow(im, draw_fn, blur, dx, dy, alpha=170, rgb=SHADOW):
    """Blurred maroon drop shadow of whatever draw_fn(ImageDraw, dx, dy) draws (fill = mask value)."""
    W, H = im.size
    m = Image.new("L", (W, H), 0)
    draw_fn(ImageDraw.Draw(m), dx, dy, alpha)
    m = m.filter(ImageFilter.GaussianBlur(blur))
    layer = Image.new("RGBA", (W, H), rgb + (255,)); layer.putalpha(m)
    im.alpha_composite(layer)


def holo_diamond(im, cx, cy, r, phase=1.3):
    W, H = im.size
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon([(cx - r, cy), (cx, cy - r), (cx + r, cy), (cx, cy + r)], fill=255)
    hol = tr.holo_strip(W, H, phase).convert("RGBA"); hol.putalpha(m)
    im.alpha_composite(hol)


def dashes(d, x0, x1, y, dash, gap, h, fill, from_right=False):
    """Dashed rule from x0 to x1 (whole dashes only), anchored at x1 when from_right."""
    n = int((x1 - x0 + gap) // (dash + gap))
    for i in range(n):
        if from_right:
            b = x1 - i * (dash + gap); a = b - dash
        else:
            a = x0 + i * (dash + gap); b = a + dash
        d.rounded_rectangle([a, y - h / 2, b, y + h / 2], radius=h / 2, fill=fill)


def dark_fade(im, y0, y1, amax, top_down=True):
    """Dithered dark ramp over rows y0..y1 (alpha 0 -> amax towards the frame edge)."""
    W, _ = im.size
    h = y1 - y0
    t = np.linspace(0, 1, h)
    if top_down:
        t = t[::-1]
    a = (t ** 1.6) * amax
    a = a[:, None] + np.random.default_rng(3).random((h, W)) - 0.5
    over = Image.new("RGBA", (W, h), (8, 4, 8, 255))
    over.putalpha(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L"))
    im.alpha_composite(over, (0, y0))


def compose(raw, mode="top", guides=False, sub=True):
    im = raw.convert("RGBA")
    W, H = im.size                      # 2400 x 800 master
    k = W / 2400.0
    ff = lambda name, size, var=None: tr.find_font(name, size * k, var)
    cx = W / 2

    if mode == "top":
        # soft darkening of the top strip so the type sits on clean black-maroon
        dark_fade(im, 0, int(300 * k), 120)
        f_team = ff("MontserratAlternates-BlackItalic.ttf", 104)
        f_sub = ff("Exo2-Italic[wght].ttf", 56, "ExtraBold Italic")
        yT = int(196 * k)                       # title baseline
        yS = int(292 * k)                       # subtitle baseline
    else:
        f_team = ff("MontserratAlternates-BlackItalic.ttf", 112)
        f_sub = ff("Exo2-Italic[wght].ttf", 60, "ExtraBold Italic")
        yT = int(TITLE_Y * k)
        yS = int((TITLE_Y + 94) * k)
    d = ImageDraw.Draw(im)
    team = tr.TEAM_LINE[0]
    tb = f_team.getbbox(team, anchor="ls")
    tw = tb[2] - tb[0]
    tx = cx - tw / 2 - tb[0] - 6 * k             # italic: nudge left so the slanted block looks centred

    def team_sh(dd, dx, dy, a):
        dd.text((tx + dx, yT + dy), team, font=f_team, fill=a, anchor="ls")
    soft_shadow(im, team_sh, 6 * k, 4 * k, 6 * k, 190)
    d = ImageDraw.Draw(im)
    d.text((tx, yT), team, font=f_team, fill=PINK, anchor="ls")

    if not sub:
        return finish(im, guides, k)
    # subtitle with holo diamonds and maroon butcher-chart dashes on both sides
    sp = 6 * k
    sub = tr.TEAM_LINE[1]
    sw = tr._spaced_len(d, sub, f_sub, sp)
    sx = cx - sw / 2
    cap = f_sub.getbbox("Н", anchor="ls")
    cyS = yS + (cap[1] + cap[3]) / 2

    def sub_sh(dd, dx, dy, a):
        x = sx + dx
        for ch in sub:
            dd.text((x, yS + dy), ch, font=f_sub, fill=a, anchor="ls")
            x += dd.textlength(ch, font=f_sub) + sp
    soft_shadow(im, sub_sh, 5 * k, 3 * k, 4 * k, 200, (20, 0, 8))
    d = ImageDraw.Draw(im)
    tr._spaced(d, (sx, yS), sub, f_sub, PEARL, sp)
    r = 10 * k
    gx = 26 * k
    holo_diamond(im, sx - gx - r, cyS, r)
    holo_diamond(im, sx + sw + gx + r, cyS, r, 2.4)
    d = ImageDraw.Draw(im)
    dl = 120 * k                                 # length of each dashed rule
    dashes(d, sx - 2 * gx - 2 * r - dl, sx - 2 * gx - 2 * r, cyS, 30 * k, 14 * k, 5 * k, MAROON, from_right=True)
    dashes(d, sx + sw + 2 * gx + 2 * r, sx + sw + 2 * gx + 2 * r + dl, cyS, 30 * k, 14 * k, 5 * k, MAROON)
    # holographic hairline between the title and the subtitle
    hw = int(tw * 0.62); hh = max(2, int(3 * k))
    hl = tr.holo_strip(hw, hh, 0.4).convert("RGBA")
    hm = np.clip(1.0 - np.abs(np.linspace(-1, 1, hw)) ** 3, 0, 1)        # tapered ends
    hl.putalpha(Image.fromarray((np.repeat(hm[None, :], hh, 0) * 235).astype(np.uint8), "L"))
    im.alpha_composite(hl, (int(cx - hw / 2), int((yT + 15 * k + yS + cap[1]) / 2 - hh / 2)))

    return finish(im, guides, k)


def finish(im, guides, k):
    W, H = im.size
    if guides:
        g = ImageDraw.Draw(im)
        g.rectangle([200 * k, 60 * k, 2200 * k, 740 * k], outline=(0, 255, 0, 255), width=2)   # safe 1000x340 @1200
        g.rectangle([80 * k, 80 * k, W - 80 * k, H - 80 * k], outline=(255, 255, 0, 255), width=1)  # 40 px @1200
    return im.convert("RGB")


SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def save_jpeg(im, path, limit=900_000):
    for q in (97, 95, 93, 91, 89, 87, 85, 82, 80):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True, progressive=True, subsampling=0 if q >= 90 else 2,
                icc_profile=SRGB_ICC)
        if buf.tell() <= limit:
            with open(path, "wb") as fh:
                fh.write(buf.getvalue())
            return q, buf.tell()
    raise SystemExit("cannot fit the JPEG under the limit")


def preview(big, small, path):
    pad = 24
    W = pad * 3 + big.width + small.width
    H = pad * 2 + big.height + 40
    pv = Image.new("RGB", (W, H), (236, 236, 240))
    pv.paste(big, (pad, pad + 40)); pv.paste(small, (pad * 2 + big.width, pad + 40))
    d = ImageDraw.Draw(pv)
    f = tr.find_font("Exo2-Italic[wght].ttf", 22, "Bold Italic")
    d.text((pad, pad + 28), "1200 x 400 (100 %)", font=f, fill=(40, 40, 50), anchor="ls")
    d.text((pad * 2 + big.width, pad + 28), "600 x 200 (mobile)", font=f, fill=(40, 40, 50), anchor="ls")
    pv.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("master"); ap.add_argument("jpg"); ap.add_argument("preview")
    ap.add_argument("--mode", default="top"); ap.add_argument("--guides", action="store_true")
    ap.add_argument("--no-sub", action="store_true", help="title only, without «РОЗОВЫЙ ПЯТАЧОК»")
    a = ap.parse_args()
    raw = Image.open(a.raw).convert("RGB")
    if raw.size != (2400, 800):
        raw = raw.resize((2400, 800), Image.LANCZOS)
    master = compose(raw, a.mode, a.guides, not a.no_sub)
    master.save(a.master, optimize=True, icc_profile=SRGB_ICC)
    small = master.resize((1200, 400), Image.LANCZOS)
    q, n = save_jpeg(small, a.jpg)
    mob = small.resize((600, 200), Image.LANCZOS)
    preview(Image.open(a.jpg).convert("RGB"), mob, a.preview)
    print(f"master {master.size}, jpeg q{q} {n} bytes -> {a.jpg}")


if __name__ == "__main__":
    main()
