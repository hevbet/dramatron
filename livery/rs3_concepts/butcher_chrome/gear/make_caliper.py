#!/usr/bin/env python3
"""Pink brake calipers for the «Butcher Chart Chrome» skins.

Recolours the caliper texture embedded in the .kn5 (caliper.dds, 256x256 DXT5 with 9 mip levels)
and writes it as a skin-folder override, together with the caliper's detail/maps texture:
    out/<skin>/caliper.dds                (256x256 DXT5, 9 mips, ORIGINAL alpha = 0 on every texel)
    out/<skin>/caliper_detail.dds         (4x4, 3 mips: one white texel, written as DXT5)
    identical files for Pozdnyakov_00 and Konopelko_00

How it works
  * the original is a grey baked-AO map (R=G=B), no lettering at all; the shading of each texel is kept:
    body colour = gradient map of s(L) through the livery palette only (s a gentle curve of the original
    grey L): deep AO -> INK_D, mid -> INK (burgundy), lit faces -> PIG (the body's flesh pink), the few
    hottest texels a soft pearl sheen;
  * UV islands are taken from the caliper mesh and split into: caliper body (pink), brake pads (steel
    grey, they sit inside the caliper) and the tiny bleed screws (silver);
  * texels outside the islands get the colour of the nearest island texel (clean mip padding);
  * the DDS keeps the original 128-byte header (DXT5, 256x256, 9 mips) and gets a full mip chain.

Alpha and caliper_detail.dds: the material is ksPerPixelMultiMap with useDetail=1, detailUVMultiplier=0
and txDetail = txMaps = caliper_detail.dds (one flat grey 45/255 texel). AC multiplies the diffuse by
that detail colour where the diffuse alpha is black:
    diffuse.rgb *= lerp(detail.rgb, 1, diffuse.a)
The original alpha is 0 everywhere, so with the stock detail every caliper texel is multiplied by 0.18
and renders near-black whatever its colour. The alpha of caliper.dds is kept EXACTLY as in the original
(0, all levels); instead the skin also overrides caliper_detail.dds with a white texel (detail x1.0), so
the pink shows. Side effect (the same texel is txMaps): the spec / gloss / reflection multipliers of the
caliper material go from 0.18 to 1.0, i.e. ksSpecular 0.2, ksSpecularEXP 50, fresnelMaxLevel 0.1 as set
in the material - a satin painted caliper instead of a matte black one. Only the caliper material uses
caliper_detail.dds.

Usage: python3 make_caliper.py [--out DIR] [--png PATH]
"""
import argparse
import io
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = "/home/user/dramatron/livery/tools"
sys.path.insert(0, TOOLS)
import kn5  # noqa: E402
import render_rs3  # noqa: E402

KN5 = render_rs3.DEFAULT_KN5
TEX = "caliper.dds"
SKINS = ("Pozdnyakov_00", "Konopelko_00")

# palette (butcher_chrome/make_skin.py): PIG (242,158,178) body, INK (96,18,42), INK_D (40,10,22)
PIG = np.array((242, 158, 178), np.float32)
INK = np.array((96, 18, 42), np.float32)
INK_D = np.array((40, 10, 22), np.float32)
PEARL = np.array((250, 240, 248), np.float32)
CAL_STOPS = ((0.10, INK_D), (0.40, INK), (1.00, PIG))   # shading s -> body colour (palette only)
DETAIL = "caliper_detail.dds"
PAD = np.array((1.00, 0.98, 1.00), np.float32)  # brake pads: neutral steel (x grey value)
HW = np.array((0.97, 0.97, 1.00), np.float32)   # bleed screws: silver (x grey value)
PAD_TEXELS = 4                                  # island edge padding (texels); the rest is one flat colour


# ---------------------------------------------------------------- source
def load_original():
    m = kn5.load(KN5, keep_tex=True)
    data = dict(m["textures"])[TEX]
    img = Image.open(io.BytesIO(data))
    img.load()
    mat = [x for x in m["materials"] if x["samplers"].get("txDiffuse") == TEX]
    return data, img.convert("RGBA"), mat


def caliper_mesh():
    """One caliper (front left) in world space; all 4 share the same UVs."""
    _, mats, meshes = render_rs3.load_scene(KN5, interior=False)
    cal = [x for x in meshes if x["mat"] == "caliper"]
    assert len(cal) == 4, len(cal)
    for x in cal[1:]:
        assert np.allclose(x["uv"], cal[0]["uv"]) and np.array_equal(x["idx"], cal[0]["idx"])
    lf = [x for x in cal if "SUSP_LF" in x["name"]][0]
    return lf, cal


def islands(mesh, size, ss=4):
    """Label raster (size x size) of UV islands + per-island class: 0 body, 1 pad, 2 hardware."""
    P, T, UV = mesh["pos"], mesh["idx"], mesh["uv"]
    key = np.round(np.mod(UV, 1) * 8192).astype(np.int64)
    _, vid = np.unique(key[:, 0] * 100000 + key[:, 1], return_inverse=True)
    vid = vid.ravel()
    TT = vid[T]
    parent = np.arange(vid.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for t in TT:
        r0 = find(t[0])
        for v in t[1:]:
            rv = find(v)
            if rv != r0:
                parent[rv] = r0
    root = np.array([find(v) for v in range(len(parent))])
    tri_isl = np.unique(root[TT[:, 0]], return_inverse=True)[1].ravel()
    n_isl = tri_isl.max() + 1
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    cls, info = np.zeros(n_isl, np.int64), []
    for k in range(n_isl):
        s = tri_isl == k
        vv = np.unique(T[s].ravel())
        ar = area[s].sum() * 1e4                        # cm2
        xe = np.ptp(P[vv, 0])                            # thickness across the disc
        cls[k] = 2 if ar < 1.0 else (1 if xe < 0.0105 else 0)
        info.append((k, int(s.sum()), round(ar, 2), round(float(xe), 4), int(cls[k])))
    S = size * ss
    lab = Image.new("I", (S, S), 0)
    d = ImageDraw.Draw(lab)
    for t in range(len(T)):
        pts = [(float(np.mod(u, 1) * S), float(np.mod(v, 1) * S)) for u, v in UV[T[t]]]
        d.polygon(pts, fill=int(tri_isl[t]) + 1, outline=int(tri_isl[t]) + 1)
    lab = np.asarray(lab)[ss // 2::ss, ss // 2::ss].astype(np.int64)     # texel centres
    cover = lab > 0
    # every texel -> nearest island (padding)
    _, (iy, ix) = ndimage.distance_transform_edt(~cover, return_indices=True)
    near = lab[iy, ix] - 1
    return cover, near, cls, info, (iy, ix)


# ---------------------------------------------------------------- recolour
def shade_curve(L):
    """Original grey (0..231) -> shading factor; keeps AO order, lifts the deep AO a little."""
    return 0.10 + 0.90 * np.clip(L / 205.0, 0, 1.15) ** 0.7


def recolour(src, cover, near, cls, idx):
    a = np.asarray(src).astype(np.float32)
    L = a[..., :3].mean(-1)
    # padding: outside the islands (dilated by 1 texel, the original bake already has a thin halo)
    # take the grey of the nearest island texel
    inside = ndimage.binary_dilation(cover, iterations=1)
    iy, ix = idx
    Lp = np.where(inside, L, L[iy, ix])
    c = cls[near]
    s = shade_curve(Lp)
    xs = [p for p, _ in CAL_STOPS]
    body = np.stack([np.interp(s, xs, [col[k] for _, col in CAL_STOPS]) for k in range(3)], -1)
    s = s[..., None]
    hl = np.clip(s - 1.0, 0, None) / 0.09                     # the few hottest texels: soft pearl sheen
    body = body + (PEARL - body) * 0.35 * np.clip(hl, 0, 1)
    pad = PAD * (14.0 + 0.55 * Lp)[..., None]
    hw = HW * np.clip(70.0 + 0.8 * Lp, 0, 235)[..., None]
    rgb = np.where((c == 0)[..., None], body, np.where((c == 1)[..., None], pad, hw))
    # beyond a few texels of padding: one flat muted pink (the mean caliper body colour), so far mips
    # stay pink instead of fading to the black background of the original
    far = ndimage.distance_transform_edt(~cover) > PAD_TEXELS
    rgb[far] = rgb[cover & (c == 0)].mean(0) * 0.8
    return np.clip(rgb, 0, 255)


# ---------------------------------------------------------------- DDS with mips
def dxt5_blocks(im):
    """DXT5 block data of one level (Pillow BC3 encoder); levels below 4x4 are edge-padded to 4x4."""
    w, h = im.size
    if w < 4 or h < 4:
        a = np.asarray(im)
        a = np.pad(a, ((0, max(0, 4 - h)), (0, max(0, 4 - w)), (0, 0)), mode="edge")
        im = Image.fromarray(a, "RGBA")
    bio = io.BytesIO()
    im.save(bio, format="DDS", pixel_format="DXT5")
    raw = bio.getvalue()
    assert raw[84:88] == b"DXT5", raw[84:88]
    n = max(1, im.width // 4) * max(1, im.height // 4) * 16
    body = raw[128:]
    assert len(body) == n, (len(body), n)
    return body


def write_dds(path, rgba_f, alpha, header):
    """rgba_f: float RGB (H,W,3); alpha: uint8 (H,W).  header: original 128-byte DDS header (reused)."""
    hh = struct.unpack_from("<31I", header, 4)
    H, W, mips = hh[2], hh[3], max(1, hh[6])
    assert header[84:88] == b"DXT5" and rgba_f.shape[:2] == (H, W)
    base = np.dstack([np.clip(rgba_f, 0, 255), alpha.astype(np.float32)])
    out = bytearray(header)
    for lv in range(mips):
        w, h = max(1, W >> lv), max(1, H >> lv)
        # box filter straight from level 0 (alpha is uniform or untouched, so no premultiply needed)
        lvl = base.reshape(h, H // h, w, W // w, 4).mean((1, 3))
        out += dxt5_blocks(Image.fromarray(np.clip(lvl + 0.5, 0, 255).astype(np.uint8), "RGBA"))
    with open(path + ".tmp", "wb") as fh:
        fh.write(out)
    os.replace(path + ".tmp", path)
    return len(out)


def copy_alpha_blocks(orig, path):
    """Same header, same level layout: copy the 8-byte DXT5 alpha half of every block from the original."""
    out = bytearray(open(path, "rb").read())
    assert len(out) == len(orig) and out[:128] == orig[:128]
    for o in range(128, len(out), 16):
        out[o:o + 8] = orig[o:o + 8]
    open(path, "wb").write(bytes(out))


def read_dds_levels(path):
    """Decode every mip level of a DXT5 DDS (Pillow reads the top level only -> re-wrap each level)."""
    return read_dds_levels_bytes(open(path, "rb").read())


def read_dds_levels_bytes(data):
    hh = struct.unpack_from("<31I", data, 4)
    H, W, mips = hh[2], hh[3], max(1, hh[6])
    off, levels = 128, []
    for lv in range(mips):
        w, h = max(1, W >> lv), max(1, H >> lv)
        n = max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 16
        hdr = bytearray(data[:128])
        struct.pack_into("<I", hdr, 12, max(4, h)); struct.pack_into("<I", hdr, 16, max(4, w))
        struct.pack_into("<I", hdr, 28, 1)
        im = Image.open(io.BytesIO(bytes(hdr) + data[off:off + n])).convert("RGBA")
        levels.append(np.asarray(im)[:h, :w])
        off += n
    assert off == len(data), (off, len(data))
    return levels


def detail_header(w=4, h=4, mips=3):
    """DXT5 header for the small caliper_detail.dds (same size and mip count as the kn5 original, which is
    uncompressed A8R8G8B8): linear size of one 4x4 block, RGBBitCount 0."""
    hd = bytearray(128)
    hd[0:4] = b"DDS "
    struct.pack_into("<7I", hd, 4, 124, 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000, h, w,
                     max(1, w // 4) * max(1, h // 4) * 16, 0, mips)
    struct.pack_into("<2I", hd, 76, 32, 0x4)                 # pixel format: FOURCC
    hd[84:88] = b"DXT5"
    struct.pack_into("<I", hd, 108, 0x1000 | 0x8 | 0x400000)  # TEXTURE | COMPLEX | MIPMAP
    return bytes(hd)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"), help="writes <out>/<skin>/caliper.dds")
    ap.add_argument("--skins", default=",".join(SKINS))
    ap.add_argument("--png", default=None, help="also save the flat level-0 RGB as PNG")
    a = ap.parse_args()

    data, src, mats = load_original()
    header = data[:128]
    hh = struct.unpack_from("<31I", header, 4)
    print(f"original {TEX}: {hh[3]}x{hh[2]} {header[84:88].decode()} mips={hh[6]} bytes={len(data)}; "
          f"alpha min/max {np.asarray(src)[..., 3].min()}/{np.asarray(src)[..., 3].max()}")
    for m in mats:
        print(f"material {m['name']} ({m['shader']}): useDetail={m['props'].get('useDetail')} "
              f"detailUVMultiplier={m['props'].get('detailUVMultiplier')} samplers={m['samplers']}")
    lf, _ = caliper_mesh()
    cover, near, cls, info, idx = islands(lf, src.width)
    names = {0: "body", 1: "pad", 2: "hardware"}
    for k, nt, ar, xe, c in info:
        print(f"  island {k:2d}: {nt:3d} tris {ar:7.2f} cm2  thickness {xe:.4f} m -> {names[c]}")
    rgb = recolour(src, cover, near, cls, idx)
    alpha = np.asarray(src)[..., 3]                           # the original alpha, untouched (0 everywhere)
    m = kn5.load(KN5, keep_tex=True)
    det_src = Image.open(io.BytesIO(dict(m["textures"])[DETAIL]))
    det_src.load()
    dh = struct.unpack_from("<31I", dict(m["textures"])[DETAIL], 4)
    det_hdr = detail_header(det_src.width, det_src.height, max(1, dh[6]))
    det_rgb = np.full((det_src.height, det_src.width, 3), 255.0, np.float32)            # detail x1.0
    det_a = np.asarray(det_src.convert("RGBA"))[..., 3]                                  # original alpha (255)
    if a.png:
        Image.fromarray(np.clip(rgb + 0.5, 0, 255).astype(np.uint8)).save(a.png)
    for skin in [s for s in a.skins.split(",") if s]:
        d = os.path.join(a.out, skin)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, TEX)
        n = write_dds(p, rgb, alpha, header)
        copy_alpha_blocks(data, p)                             # alpha halves byte-identical to the original
        lv = read_dds_levels(p)
        h2 = struct.unpack_from("<31I", open(p, "rb").read(128), 4)
        err = np.abs(lv[0][..., :3].astype(int) - np.clip(rgb + 0.5, 0, 255).astype(int)).mean()
        same_a = all(np.array_equal(x[..., 3], y[..., 3]) for x, y in zip(lv, read_dds_levels_bytes(data)))
        print(f"wrote {p}: {n} bytes (orig {len(data)}), {h2[3]}x{h2[2]}, mips {h2[6]} "
              f"[{' '.join(f'{x.shape[1]}' for x in lv)}], alpha {lv[0][..., 3].min()}..{lv[0][..., 3].max()} "
              f"(identical to the original on every level: {same_a}), mean |DXT5 - target| {err:.2f}")
        pd = os.path.join(d, DETAIL)
        nd = write_dds(pd, det_rgb, det_a, det_hdr)
        dl = read_dds_levels(pd)
        print(f"wrote {pd}: {nd} bytes, {dl[0].shape[1]}x{dl[0].shape[0]}, mips {len(dl)}, "
              f"texel {tuple(int(v) for v in dl[0][0, 0])} on every level: "
              f"{all((x == dl[0][0, 0]).all() for x in dl)}")


if __name__ == "__main__":
    main()
