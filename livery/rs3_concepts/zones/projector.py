"""World-space projection of artwork onto Skin.dds — paint in a side/top/front/rear VIEW, get a texture layer.
Crosses UV seams automatically (rear door is split in 2 islands, front fender in 3, etc.).

    import sys; sys.path.insert(0, ".../rs3_concepts/zones"); import projector as pj
    layer = pj.project(art_rgba, "left", center=(0.0, 0.25, 0.62), width_m=1.6)   # PIL RGBA 4096
    skin.alpha_composite(layer)

Views (art is drawn exactly as the viewer sees it, upright):
    left   : camera at +x (car's LEFT side). image right = car rear (+y), image up = z
    right  : camera at -x. image right = car front (-y), image up = z
    front  : camera ahead. image right = car's left (+x), up = z
    rear   : camera behind. image right = car's right (-x), up = z
    top    : looking down, readable from the FRONT (image up = car rear, image right = car left)   [hood, roof]
    top_rear : looking down, readable from BEHIND (image up = car front, image right = car right) [trunk, roof]
    top_left : looking down, readable from the car's LEFT side (image up = car right, image right = rear) [roof]
center = world point (x, y, z) in metres of the art centre (only the in-plane coords matter);
width_m = art width in metres (height follows the art aspect).
max_angle = only texels whose normal is within this angle of the view direction receive art
(e.g. 75 for wrap-around stripes, 45 for crisp logos).  parts = optional list of part names to restrict to
(see zones.json 'part_names'), e.g. ["front_door", "rear_door"].
"""
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
_PM = None

VIEWS = {
    "left": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "right": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    "front": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "rear": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "top": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "top_rear": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "top_left": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
}


def _load(res=4096):
    global _PM
    if _PM is None or _PM["res"] != res:
        d = np.load(os.path.join(HERE, "posmap.npz"))
        k = res / d["pos"].shape[0]
        pos = d["pos"].astype(np.float32)
        nrm = d["nrm"].astype(np.float32)
        if k != 1:
            pos = np.stack([ndimage.zoom(pos[..., c], k, order=1) for c in range(3)], -1)
            nrm = np.stack([ndimage.zoom(nrm[..., c], k, order=1) for c in range(3)], -1)
        part = ndimage.zoom(d["part"], k, order=0) if k != 1 else d["part"]
        valid = ndimage.zoom((d["tri"] >= 0).astype(np.uint8), k, order=0) > 0 if k != 1 else d["tri"] >= 0
        nl = np.linalg.norm(nrm, axis=-1, keepdims=True)
        nrm = nrm / np.maximum(nl, 1e-6)
        names = json.load(open(os.path.join(HERE, "zones.json")))["part_names"]
        _PM = dict(res=res, pos=pos, nrm=nrm, part=part, valid=valid, names=names)
    return _PM


def project(art, view, center, width_m, max_angle=70, parts=None, res=4096, feather=0.0):
    pm = _load(res)
    r, u, d = (np.array(v, float) for v in VIEWS[view])
    art = art.convert("RGBA")
    a = np.asarray(art).astype(np.float32) / 255.0
    H, W = a.shape[:2]
    height_m = width_m * H / W
    c = np.asarray(center, float)
    rel = pm["pos"] - c
    sa = rel @ r; sb = rel @ u
    px = (sa / width_m + 0.5) * W - 0.5
    py = (0.5 - sb / height_m) * H - 0.5
    m = pm["valid"] & ((pm["nrm"] @ d) > np.cos(np.radians(max_angle)))
    if parts is not None:
        ids = [pm["names"].index(p) for p in parts if p in pm["names"]]
        m &= np.isin(pm["part"], ids)
    m &= (px > -1) & (px < W) & (py > -1) & (py < H)
    out = np.zeros((res, res, 4), np.float32)
    if m.any():
        pre = a.copy(); pre[..., :3] *= pre[..., 3:4]
        coords = np.stack([py[m], px[m]])
        vals = [ndimage.map_coordinates(pre[..., ch], coords, order=1, mode="constant", cval=0.0) for ch in range(4)]
        al = vals[3]
        if feather > 0:   # fade near the angular limit to avoid hard cut on curved surfaces
            cosv = (pm["nrm"][m] @ d)
            lim = np.cos(np.radians(max_angle))
            al_f = np.clip((cosv - lim) / max(feather, 1e-3), 0, 1)
            vals = [v * al_f for v in vals]; al = vals[3]
        rgb = np.stack(vals[:3], -1) / np.maximum(al[:, None], 1e-6)
        out[m, :3] = rgb; out[m, 3] = al
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGBA")


def elevation(view, px_per_m=400, max_angle=70):
    """Diagram of the car as seen from `view` built from texels: colour = flatness (cos to view dir),
    white lines = UV island seams, grid every 10 cm (labels in metres of the in-plane world axes)."""
    from PIL import ImageDraw, ImageFont
    pm = _load(2048)
    d0 = np.load(os.path.join(HERE, "posmap.npz"))
    cov = d0["covered"]; isl = d0["island"]
    r, u, d = (np.array(v, float) for v in VIEWS[view])
    cosv = pm["nrm"] @ d
    m = cov & (cosv > np.cos(np.radians(max_angle)))
    pos = pm["pos"]
    sa, sb, dep = pos @ r, pos @ u, pos @ d
    lo_a, hi_a, lo_b, hi_b = -2.4, 2.4, -2.4, 2.4
    if view in ("left", "right", "front", "rear"):
        lo_b, hi_b = 0.0, 1.6
    if view in ("front", "rear"):
        lo_a, hi_a = -1.1, 1.1
    if view.startswith("top") and view != "top_left":
        lo_a, hi_a = -1.1, 1.1
    if view == "top_left":
        lo_b, hi_b = -1.1, 1.1
    W, H = int((hi_a - lo_a) * px_per_m), int((hi_b - lo_b) * px_per_m)
    X = ((sa - lo_a) * px_per_m).astype(int); Y = ((hi_b - sb) * px_per_m).astype(int)
    ok = m & (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    zb = np.full((H, W), -1e9); col = np.zeros((H, W, 3), np.uint8) + 20
    seam = np.zeros((H, W), bool)
    il = np.where(cov, isl, -1)
    edge = np.zeros_like(cov)
    for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        edge |= il != np.roll(il, sh, ax)
    edge &= cov
    order = np.argsort(dep[ok])
    xs, ys, ds, cs, es = X[ok][order], Y[ok][order], dep[ok][order], cosv[ok][order], edge[ok][order]
    zb[ys, xs] = ds
    fl = np.clip((cs - 0.5) / 0.5, 0, 1)
    c = np.stack([255 * (1 - fl), 90 + 140 * fl, 90 + 60 * fl], -1).astype(np.uint8)
    col[ys, xs] = c
    seam[ys[es], xs[es]] = True
    filled = zb > -1e8
    # close splat holes: copy nearest painted pixel into gaps up to 3 px
    dist, (iy, ix) = ndimage.distance_transform_edt(~filled, return_indices=True)
    hole = (~filled) & (dist <= 3) & ndimage.binary_closing(filled, iterations=4)
    col[hole] = col[iy[hole], ix[hole]]
    seam = ndimage.binary_dilation(seam)
    col[seam] = 255
    im = Image.fromarray(col)
    dr = ImageDraw.Draw(im)
    f = ImageFont.load_default()
    for k in range(int(lo_a * 10), int(hi_a * 10) + 1):
        xx = (k / 10 - lo_a) * px_per_m
        dr.line([(xx, 0), (xx, H)], fill=(70, 70, 70) if k % 5 else (130, 130, 130), width=1)
        if k % 5 == 0:
            dr.text((xx + 2, H - 12), f"{k / 10:+.1f}", fill=(255, 255, 0), font=f)
    for k in range(int(lo_b * 10), int(hi_b * 10) + 1):
        yy = (hi_b - k / 10) * px_per_m
        dr.line([(0, yy), (W, yy)], fill=(70, 70, 70) if k % 5 else (130, 130, 130), width=1)
        if k % 5 == 0:
            dr.text((2, yy + 2), f"{k / 10:+.1f}", fill=(255, 255, 0), font=f)
    ax = {"left": "x-axis: world y (front<-  ->rear), up: z",
          "right": "x-axis: world -y (rear<-  ->front), up: z",
          "front": "x-axis: world x (car right<- ->car left), up: z",
          "rear": "x-axis: world -x (car left<- ->car right), up: z",
          "top": "x-axis: world x, up: world +y (rear)", "top_rear": "x-axis: world -x, up: world -y (front)",
          "top_left": "x-axis: world y, up: world -x"}[view]
    dr.text((6, 4), f"view={view}: {ax}. green=flat to viewer, red=curving away, white=UV seam. grid 10 cm", fill=(255, 255, 255), font=f)
    return im
