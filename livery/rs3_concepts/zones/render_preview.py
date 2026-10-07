"""Tiny software renderer: shows a Skin texture (and glass_sticker) on the kn5 model from fixed views.

python3 render_preview.py <skin.png|dds> <out_prefix> [glass_sticker.png|dds]
Writes <out_prefix>_sheet.jpg (left, right, front, rear, top, 3/4 views).
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import load  # noqa: E402

VIEWS = [("left", (1, 0, 0.12)), ("right", (-1, 0, 0.12)), ("front", (0, -1, 0.3)), ("rear", (0, 1, 0.35)),
         ("top", (0.0001, 0.0, 1)), ("front 3/4 L", (1, -1.1, 0.45)), ("rear 3/4 R", (-1, 1.1, 0.5)),
         ("rear 3/4 L high", (0.7, 1.0, 1.1))]


def get_tris():
    meshes = load()
    out = []
    for m in meshes:
        w = m["wpos"][m["idx"]]
        P = np.stack([w[..., 0], -w[..., 2], w[..., 1] + 0.072], -1)
        uv = m["uv"][m["idx"]]
        Tt = np.stack([uv[..., 0], 1 + uv[..., 1]], -1)
        nm = m["mat"]
        if "COCKPIT" in m["name"] or "/INT" in m["name"].upper():
            continue
        if nm == "skin":
            kind = 1
        elif nm in ("glass_sticker", "ext_sticker"):
            kind = 2
        elif "glass" in nm.lower() or nm in ("air",):
            continue
        else:
            kind = 0
        out.append((P, Tt, np.full(len(P), kind)))
    P = np.concatenate([o[0] for o in out]); Tt = np.concatenate([o[1] for o in out]); Kd = np.concatenate([o[2] for o in out])
    return P, Tt, Kd


def render(P, Tt, Kd, tex, gtex, d, W=900, H=600, scale=None):
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.97 else np.array([0, -1.0, 0])
    r = np.cross(up, d); r /= np.linalg.norm(r); u = np.cross(d, r)
    ctr = np.array([0, 0.07, 0.75])
    V = P - ctr
    sc = scale or min(W, H) / 3.2 if abs(d[2]) < 0.97 else H / 5.0
    X = V @ r * sc + W / 2; Y = -(V @ u) * sc + H / 2; Z = V @ d
    zb = np.full((H, W), -1e9); img = np.zeros((H, W, 3), np.float32) + 40
    # normals for shading
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.3, -0.4, 0.85]); light /= np.linalg.norm(light)
    sh = 0.55 + 0.45 * np.abs(n @ light)
    th, tw = tex.shape[:2]; gh, gw = gtex.shape[:2]
    facing = (n @ d) > 0
    order = np.nonzero(facing | (Kd == 2))[0]
    for t in order:
        xs, ys = X[t], Y[t]
        x0 = max(int(xs.min()), 0); x1 = min(int(xs.max()) + 1, W - 1)
        y0 = max(int(ys.min()), 0); y1 = min(int(ys.max()) + 1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(den) < 1e-9:
            continue
        l1 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
        l2 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
        l3 = 1 - l1 - l2
        m = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not m.any():
            continue
        zz = l1 * Z[t, 0] + l2 * Z[t, 1] + l3 * Z[t, 2]
        py, px = (gy[m] - 0.5).astype(int), (gx[m] - 0.5).astype(int)
        zz = zz[m]
        k = Kd[t]
        if k == 0:
            col = np.full((len(px), 3), 70 * sh[t])
            ok = zz > zb[py, px]
        else:
            tu = l1[m] * Tt[t, 0, 0] + l2[m] * Tt[t, 1, 0] + l3[m] * Tt[t, 2, 0]
            tv = l1[m] * Tt[t, 0, 1] + l2[m] * Tt[t, 1, 1] + l3[m] * Tt[t, 2, 1]
            if k == 1:
                c = tex[np.clip((tv * th).astype(int), 0, th - 1), np.clip((tu * tw).astype(int), 0, tw - 1)]
                col = c[:, :3] * sh[t]
                ok = zz > zb[py, px]
            else:
                c = gtex[np.clip((tv * gh).astype(int), 0, gh - 1), np.clip((tu * gw).astype(int), 0, gw - 1)]
                col = c[:, :3] * sh[t]
                ok = (zz > zb[py, px] - 0.003) & (c[:, 3] > 128)
        img[py[ok], px[ok]] = col[ok]
        zb[py[ok], px[ok]] = zz[ok]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    tex = np.asarray(Image.open(sys.argv[1]).convert("RGB").resize((2048, 2048))).astype(np.float32)
    gt = np.asarray(Image.open(sys.argv[3]).convert("RGBA")).astype(np.float32) if len(sys.argv) > 3 else np.zeros((4, 4, 4), np.float32)
    P, Tt, Kd = get_tris()
    keep = Kd >= 0
    ims = []
    from PIL import ImageDraw
    for name, d in VIEWS:
        im = render(P, Tt, Kd, tex, gt, d)
        ImageDraw.Draw(im).text((10, 10), name, fill=(255, 255, 0))
        ims.append(im)
    sheet = Image.new("RGB", (1800, 600 * ((len(ims) + 1) // 2)))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % 2) * 900, (i // 2) * 600))
    sheet.save(sys.argv[2] + "_sheet.jpg", quality=88)
    for (name, _), im in zip(VIEWS, ims):
        im.save(sys.argv[2] + "_" + name.replace(" ", "_").replace("/", "") + ".jpg", quality=90)
