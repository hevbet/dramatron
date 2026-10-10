#!/usr/bin/env python3
"""Pink wheel nuts (lug nuts) for the «Butcher Chart Chrome» skins.

Writes skin-folder overrides of the two rim textures embedded in the .kn5:
    out/<skin>/rim_d.dds      1024x1024 DXT5, 11 mips  (rim material, the static wheel)
    out/<skin>/rim_blur.dds   1024x1024 DXT5, 11 mips  (rimblur material, the spinning wheel)
    identical files for Pozdnyakov_23 and Konopelko_86

Where the nuts are
  The 5 lug nuts are part of the rim mesh (WHEEL_xx/RIM_xx/Plane.0xx, material «rim»). Each nut is made of
  separate UV islands: hex body (~62 tris), domed cap (108 tris), flange/seat ring (88 tris) and a few
  one-triangle slivers. All 5 nuts share the same three islands in the top-left corner of rim_d.dds
  (hex ~x10-121/y3-135, cap ~x21-92/y154-225, flange ~x138-233/y7-101). The islands are found
  from the mesh (UV islands whose geometry is small and sits within 7 cm of the wheel axis), rasterised and
  checked: no texel used by any other part of the rim lies within 2 texels of them. All 4 rims share
  the same UVs. There is no centre-lock nut: the dark disc in the hub is the axle seen through the bore.

How they are recoloured (rim_d.dds)
  The stock nut is a flat grey bake (R=G=B, lit faces L=66, deep AO ~20). Each nut texel keeps its
  shading: colour = NUT_TEX * L / L_REF (a multiply in luminance, L_REF = the lit-face grey), so faces,
  AO and edges keep the original relative brightness. Unused texels next to the islands (6 texels of
  padding and the small hole in the middle of the hex island, never closer than 3 texels to another island) take
  the colour of the nearest nut texel, so filtering and mips do not pull a black halo into the pink. Every
  other texel is untouched.

The detail multiply (why the alpha stays as it is)
  rim is ksPerPixelMultiMap with useDetail=1, detailUVMultiplier=0 and txDetail = car_paint_rims.dds,
  one constant texel 226/255 = 0.886 grey. AC computes  diffuse.rgb *= lerp(detail.rgb, 1, diffuse.a).
  The rim_d alpha is 0 on every texel, so the whole rim, nuts included, is multiplied by 0.886 - a mild,
  colour-neutral darkening (unlike the caliper, whose detail texel is 45/255 and kills any colour).
  So no alpha change is needed: the alpha blocks are copied byte for byte from the original on every level.
  The pink is chosen so that it lands on target AFTER the x0.886: NUT_TEX = NUT / max(NUT) * 255, which
  shows in game as NUT_TEX * 0.886 = (226, 128, 151) on a fully lit face. car_paint_rims.dds is not touched.

Rim blur (rim_blur.dds)
  rimblur is ksPerPixelMultiMap_AT with useDetail=0 (no multiply). Same layout as rim_d, but the face is a
  blurred bake; the nuts are not separate geometry there - they are the 5 dark blurred blobs around the
  bore (uv radius ~40-110 px, mapped onto the hub of RIM_BLUR_xx/Plane.016 at the nuts' radius). Each blob
  is read as a mix of the radially symmetric hub background bg(r) (angular median) and the grey nut:
  coverage c = (bg - B) / (bg - N_REF), N_REF = grey of the blob cores. The grey nut is swapped for the
  pink nut at the same scale as on the static rim: there a nut grey that shows as g in game becomes
  NUT_TEX * g / L_REF (the 0.886 cancels), and rimblur has no detail multiply, so
      new = bg * (1 - c) + c * N_REF * NUT_TEX / L_REF
  only inside the nut annulus; the bore rings, the hub and everything else stay as they were.

DDS: original 128-byte headers (DXT5, 1024x1024, 11 mips). The stock mips are not a box filter of level 0,
so they are kept wherever nothing changed. rim_d: mip texels whose footprint lies in the nut zone are rebuilt
from the new level 0 (box filter), mixed texels of the deep mips get original + box-filtered change.
rim_blur: mips 0-4 are recoloured level by level with the same coverage model (re-filtering level 0 would
leave the stock mips' sharper grey cores inside pink halos), deeper mips get the box-filtered change.
Only 4x4 blocks whose colour changed are re-encoded, and only where the result is closer to the target than
the stock block AND no texel of another part of the wheel moves: a DXT block shares one 4-colour palette
between its 16 texels, so on the small mips a block holding nut texels next to the rim lip / barrel could
snap the lip texel to the pink palette and still win on the mean error (seen at mip 5-7). On every level a
texel whose footprint touches a non-nut island (rim_blur: a used texel that is not recoloured) must decode
within KEEP_TOL grey levels of the segment [stock, target]; its unused bilinear neighbours must not change
hue. Pillow's BC3 encode is tried first, then a small constrained BC1 encoder; failing both, the stock block
stays (the nut texels in it then stay grey on that mip). All other blocks are byte-identical to the
original, and the alpha half of every block is the original one on every level.

Rim colour (RIM_COLOUR, default graphite)
  The client liked a preview where car_paint_rims.dds was overridden with (70,72,76): in game that gives
  rim_d.rgb * (70,72,76)/255, but it darkened the nuts too. Instead car_paint_rims.dds stays at 226 and every
  NON-nut texel of rim_d is multiplied by RIM_COLOUR / 226 = (0.310, 0.319, 0.336), on every mip level (the
  stock mips times the factor, so the baked AO / facets are kept): in game the rim is then exactly the
  graphite preview, while the nut texels (and their padding) stay the pink of above. rim_blur gets the same
  factor (it has no detail multiply, so the factor is relative to its stock in-game look as well) under the
  pink blobs: new = f * bg * (1 - c) + pink * c.
  Encoding: a "graphite base" DDS is made first (every level = stock level * factor, Pillow BC3 colour halves,
  original alpha halves); the nut recolour above is then written on top of that base exactly as before
  (base = "stock" for write_levels), so the rim texels around the nuts are protected against the graphite
  base and no pink can bleed into them, and the nut blocks get the same constrained encode.
  --rim-colour stock (or 226,226,226) reproduces the old white-rim output.

Per-skin rims (SKIN_RIMS): a plain re-run writes both skins - Pozdnyakov_23 graphite, Konopelko_86 stock white;
--rim-colour forces one colour on every skin listed.

Usage: python3 make_bolts.py [--out DIR] [--skins A,B] [--debug DIR] [--rim-colour R,G,B|graphite|stock]
"""
import argparse
import colorsys
import io
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = "/home/user/dramatron/livery/tools"
sys.path.insert(0, TOOLS)
import kn5  # noqa: E402
import render_rs3  # noqa: E402

KN5 = render_rs3.DEFAULT_KN5
SKINS = ("Pozdnyakov_23", "Konopelko_86")
SKIN_RIMS = {"Pozdnyakov_23": "graphite", "Konopelko_86": "stock"}   # per-driver rims (a plain re-run makes both)
RIM, BLUR, DETAIL = "rim_d.dds", "rim_blur.dds", "car_paint_rims.dds"

# palette (butcher_chrome/make_skin.py): PIG (242,158,178) is the body's flesh pink
PIG = (242, 158, 178)
SAT_GAIN = 1.25                        # anodised metal: same hue as PIG, saturation 0.35 -> 0.43
_h, _s, _v = colorsys.rgb_to_hsv(*(c / 255.0 for c in PIG))
NUT = np.array(colorsys.hsv_to_rgb(_h, min(1.0, _s * SAT_GAIN), _v), np.float32) * 255   # wanted in-game pink
NUT_TEX = NUT / NUT.max() * 255.0      # texel colour of a fully lit face (in game: x detail 0.886)

GRAPHITE = (70, 72, 76)                # client-approved in-game rim colour (as a car_paint_rims.dds texel)
RIM_COLOUR = GRAPHITE                  # in-game rim tint; the non-nut texels get RIM_COLOUR / detail texel (226)

PAD = 6                                # texels of padding around the nut islands (filtering / mips)
KEEP_OFF = 3                           # padding never comes closer than this to another island
HOLE_MAX = 800                         # unused holes inside nut islands up to this size (texels) are filled
NUT_R_MAX = 0.07                       # nut parts: every vertex within 7 cm of the wheel axis ...
NUT_AREA_MAX = 20.0                    # ... and island area below 20 cm2 (the hub face itself is 229 cm2)
BLOB_R = (36.0, 120.0)                 # rim_blur: nut annulus around the texture centre (uv px)
BLOB_FEATHER = 8.0                     # soft edge of that annulus (px)
BLOB_NOISE = 2.0                       # darkening below this (grey levels) is bake / DXT noise, left alone
BLUR_LEVELS = 5                        # rim_blur mips 0..4 (hub >= 7 px) are recoloured level by level
KEEP_TOL = 2                           # DXT: a texel of a non-nut part may end up at most this many grey levels
                                       # (any channel) off the segment [stock, target] (see write_levels)
RING_LUM_TOL = 24                      # ... its unused bilinear neighbours: hue within KEEP_TOL, brightness this


def rgb3(c):
    return tuple(int(round(float(v))) for v in c)


# ---------------------------------------------------------------- source
def load_textures():
    m = kn5.load(KN5, keep_tex=True)
    tex = dict(m["textures"])
    mats = {x["name"]: x for x in m["materials"]}
    return tex, mats


def decode_levels(data):
    """Every mip level of a DXT5 DDS as uint8 RGBA arrays (Pillow reads the top level only -> re-wrap each)."""
    hh = struct.unpack_from("<31I", data, 4)
    H, W, mips = hh[2], hh[3], max(1, hh[6])
    off, levels = 128, []
    for lv in range(mips):
        w, h = max(1, W >> lv), max(1, H >> lv)
        n = max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 16
        hdr = bytearray(data[:128])
        struct.pack_into("<I", hdr, 12, max(4, h))
        struct.pack_into("<I", hdr, 16, max(4, w))
        struct.pack_into("<I", hdr, 20, n)
        struct.pack_into("<I", hdr, 28, 1)
        im = Image.open(io.BytesIO(bytes(hdr) + data[off:off + n])).convert("RGBA")
        levels.append(np.asarray(im)[:h, :w].copy())
        off += n
    assert off == len(data), (off, len(data))
    return levels


# ---------------------------------------------------------------- geometry -> UV islands
def wheel_meshes():
    _, _, meshes = render_rs3.load_scene(KN5, interior=False)
    rims = [x for x in meshes if x["mat"] == "rim"]
    blurs = [x for x in meshes if x["mat"] == "rimblur"]
    assert len(rims) == 4, [x["name"] for x in rims]
    for x in rims[1:]:
        assert np.allclose(x["uv"], rims[0]["uv"]) and np.array_equal(x["idx"], rims[0]["idx"]), x["name"]
    return rims, blurs


def uv_islands(mesh):
    """Triangle -> island id; vertices are welded when position AND uv match (UV seams split islands)."""
    P, T, UV = mesh["pos"], mesh["idx"], mesh["uv"]
    key = np.c_[np.round(P * 1e5), np.round(UV * 1e5)].astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    TT = vid.ravel()[T]
    n = TT.max() + 1
    r = np.r_[TT[:, 0], TT[:, 1]]
    c = np.r_[TT[:, 1], TT[:, 2]]
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    return np.unique(lab[TT[:, 0]], return_inverse=True)[1].ravel()


def nut_islands(mesh):
    """Classify the rim's UV islands; returns (tri_is_nut bool array, report rows)."""
    P, T, UV = mesh["pos"], mesh["idx"], mesh["uv"]
    lo, hi = P.min(0), P.max(0)
    cy, cz = (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2           # wheel axis (along x)
    rad = np.hypot(P[:, 1] - cy, P[:, 2] - cz)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2 * 1e4     # cm2
    isl = uv_islands(mesh)
    nut = np.zeros(isl.max() + 1, bool)
    rows = []
    for k in range(isl.max() + 1):
        s = isl == k
        vv = np.unique(T[s].ravel())
        ar = area[s].sum()
        nut[k] = rad[vv].max() < NUT_R_MAX and ar < NUT_AREA_MAX
        uv = np.mod(UV[vv], 1) * 1024
        kind = ("other" if not nut[k] else "sliver" if ar < 0.1 else "hex body" if s.sum() < 70
                else "flange" if s.sum() < 100 else "cap")
        rows.append((k, int(s.sum()), ar, rad[vv].min(), rad[vv].max(), np.abs(P[vv, 0]).min(),
                     np.abs(P[vv, 0]).max(), uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max(), kind))
    return nut[isl], rows


def raster(mesh, tri_sel, size, ss=4):
    """Texel coverage of the selected triangles: (any sub-sample covered, texel centre covered)."""
    S = size * ss
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    UV = mesh["uv"]
    for t in np.nonzero(tri_sel)[0]:
        pts = [(float(np.mod(u, 1) * S), float(np.mod(v, 1) * S)) for u, v in UV[mesh["idx"][t]]]
        d.polygon(pts, fill=255, outline=255)
    a = np.asarray(im) > 0
    anyc = a.reshape(size, ss, size, ss).any((1, 3))
    cen = a[ss // 2::ss, ss // 2::ss]
    return anyc, cen


# ---------------------------------------------------------------- rim_d recolour
def box(a, f):
    """Box-filter an (H, W, C) array by an integer factor f (power of two)."""
    H, W = a.shape[:2]
    h, w = max(1, H // f), max(1, W // f)
    return a.reshape(h, H // h, w, W // w, -1).mean((1, 3))


def recolour_rim(lv0, nut_any, nut_cen, other_any):
    """Returns (new float RGB level 0, changed texels, nut zone, stats)."""
    rgb = lv0[..., :3].astype(np.float32)
    L = rgb @ np.array([0.25, 0.5, 0.25], np.float32)               # grey bake: R=B (5 bit), G (6 bit)
    keep_off = ndimage.binary_dilation(other_any, iterations=KEEP_OFF)
    assert not (nut_any & ndimage.binary_dilation(other_any, iterations=2)).any(), "nut texels shared"
    vals = L[nut_cen]
    l_ref = float(np.percentile(vals, 99))
    # texels whose centre lies on a nut keep their own grey; edge / padding texels take the nearest centre's
    _, (iy, ix) = ndimage.distance_transform_edt(~nut_cen, return_indices=True)
    Lf = np.where(nut_cen, L, L[iy, ix])
    # nut zone: texels whose nearest used texel (of any island) is a nut texel, off the other islands
    used = nut_any | other_any
    dist, (jy, jx) = ndimage.distance_transform_edt(~used, return_indices=True)
    zone = nut_any[jy, jx] & ~keep_off
    # padding + the small unused hole inside the hex body island (its centre, under the cap); the flange's big
    # hex-shaped hole stays as it is apart from the padding
    holes, nh = ndimage.label(ndimage.binary_fill_holes(nut_any) & ~nut_any)
    small = np.isin(holes, [i + 1 for i, n in enumerate(ndimage.sum(holes > 0, holes, range(1, nh + 1)))
                            if n <= HOLE_MAX])
    region = (nut_any | small | (dist <= PAD)) & zone
    s = np.clip(Lf / l_ref, 0.0, 1.0)
    new = rgb.copy()
    new[region] = NUT_TEX[None, :] * s[region][:, None]
    st = dict(l_ref=l_ref, l_min=float(vals.min()), l_med=float(np.median(vals)), n_cen=int(nut_cen.sum()),
              n_any=int(nut_any.sum()), n_region=int(region.sum()))
    return new, region, zone, st


def rim_levels(orig_lv, new0, region, zone, tint=1.0):
    """Float RGB of every mip level. Level k texels whose footprint touches the changed texels: rebuilt from
    the new level 0 (box filter) when the whole footprint lies in the nut zone, otherwise (deep mips, where a
    texel also covers the rim) tinted original + box-filtered change. Everything else: the original mip times
    tint (the rim colour factor, per channel; 1 = stock)."""
    tint = np.broadcast_to(np.asarray(tint, np.float32), (3,))
    o0 = orig_lv[0][..., :3].astype(np.float32) * tint
    new0 = np.where(region[..., None], new0, o0)
    delta = new0 - o0
    out = [new0]
    reg, zon = region[..., None].astype(np.float32), zone[..., None].astype(np.float32)
    for k in range(1, len(orig_lv)):
        f = 1 << k
        ok = orig_lv[k][..., :3].astype(np.float32) * tint
        touch = box(reg, f)[..., 0] > 0
        inside = box(zon, f)[..., 0] > 0.9999
        nk = ok.copy()
        rb, dl = touch & inside, touch & ~inside
        nk[rb] = box(new0, f)[rb]
        nk[dl] = ok[dl] + box(delta, f)[dl]
        out.append(nk)
    return out


# ---------------------------------------------------------------- rim_blur recolour
def blur_used(blurs, size):
    """Texels of rim_blur.dds used by the rimblur meshes (front left wheel; all 4 share the UVs)."""
    lf = [x for x in blurs if "_LF/" in x["name"]]
    cov = np.zeros((size, size), bool)
    for x in lf:
        cov |= raster(x, np.ones(len(x["idx"]), bool), size)[0]
    return cov


def blur_coverage(B, scale):
    """Nut coverage of one level of the blurred hub: (darkness below the radial background, background,
    annulus weight, annulus mask). scale = level size / 1024."""
    H, W = B.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cx = cy = (W - 1) / 2.0                     # the bore is centred on the texture (uv 0.5, 0.5)
    r = np.hypot(xx - cx, yy - cy)
    r0, r1, fe = BLOB_R[0] * scale, BLOB_R[1] * scale, max(0.75, BLOB_FEATHER * scale)
    ann = (r > r0 - fe) & (r < r1 + fe)
    # radially symmetric hub background: per 1 px ring the 90th percentile over the angle (the blobs cover up
    # to ~60 % of a ring, so a median would sit inside them), smoothed over a few rings
    rb = np.round(r).astype(np.int64)
    ks = np.arange(max(0, int(r0 - fe) - 4), int(r1 + fe) + 5)
    prof = np.array([np.percentile(B[rb == k], 90) if (rb == k).any() else np.nan for k in ks])
    ok = ~np.isnan(prof)
    ks, prof = ks[ok], prof[ok]
    sw = 3 if scale >= 0.5 else 1
    prof = np.convolve(np.pad(prof, sw, mode="edge"), np.ones(2 * sw + 1) / (2 * sw + 1), mode="valid")
    bg = np.interp(r, ks, prof).astype(np.float32)
    w_ann = np.clip(np.minimum(r - r0, r1 - r) / fe + 1.0, 0, 1)
    dark = np.where(ann, np.clip(bg - B, 0, None), 0)
    return dark, bg, w_ann, ann


def blur_levels(orig_lv, used0, l_ref, tint=1.0):
    """Float RGB of every mip level of rim_blur + the changed texels of level 0 + stats.
    Each original level 0..BLUR_LEVELS-1 is recoloured on its own (the stock mips are sharper than a box
    filter of level 0, so re-filtering would leave grey cores in pink halos); deeper levels get the
    box-filtered change of the last recoloured level."""
    B0 = orig_lv[0][..., :3].astype(np.float32).mean(-1)
    dark0, bg0, _, ann0 = blur_coverage(B0, 1.0)
    core = dark0 > 0.5 * dark0.max()
    n_ref = float(np.percentile(B0[core & ann0], 5))          # grey of the blob cores (full nut coverage)
    # grey -> pink at the static rim's scale. Static rim in game: 0.886 * NUT_TEX * L/L_REF, its grey was
    # 0.886 * L, so in-game grey g -> NUT_TEX * g / L_REF (the 0.886 cancels); rimblur has no detail multiply,
    # so its texel grey is already the in-game grey.
    pink = NUT_TEX * (n_ref / l_ref)
    tint = np.broadcast_to(np.asarray(tint, np.float32), (3,))
    out, m0 = [], None
    usedf = used0[..., None].astype(np.float32)
    for k, lv in enumerate(orig_lv):
        rgb = lv[..., :3].astype(np.float32)
        base = rgb * tint                                             # rim colour (graphite) everywhere ...
        if k < BLUR_LEVELS:
            dark, bg, w_ann, _ = blur_coverage(rgb.mean(-1), rgb.shape[0] / orig_lv[0].shape[0])
            c = np.clip((dark - BLOB_NOISE) / np.maximum(bg - n_ref - BLOB_NOISE, 1.0), 0, 1) * w_ann
            m = (c > 1e-3) & (box(usedf, 1 << k)[..., 0] > 0)
            new = base.copy()
            # ... and under the blobs  tint bg (1-c) + pink c,  for B = bg(1-c) + n_ref c
            new[m] = base[m] + c[m][:, None] * (pink[None, :] - tint[None, :] * n_ref)
            if k == 0:
                m0 = m
            last, last_d = k, new - base
        else:
            new = base + box(last_d, 1 << (k - last))
        out.append(new)
    st = dict(n_ref=n_ref, pink=pink, n_changed=int(m0.sum()), bg_at_blobs=float(np.median(bg0[core & ann0])))
    return out, m0, st


def protected_levels(used0, levels, orig_lv, only_unchanged=False):
    """Per mip level the texels a DXT re-encode must not drag along, as classes (uint8):
      2  footprint touches used0 (a level-0 mask; with only_unchanged just those whose target is the stock
         colour): within KEEP_TOL of [stock, target] in every channel
      1  unused 1-texel ring around them with an unchanged target (bilinear filtering blends it into the edge of
         the part): no hue shift (chroma within KEEP_TOL), brightness within RING_LUM_TOL
      0  free (nut texels, empty space)"""
    out = []
    for k, lv in enumerate(orig_lv):
        same = np.all(np.clip(levels[k] + 0.5, 0, 255).astype(np.uint8) == lv[..., :3], -1)
        core = box(used0[..., None].astype(np.float32), 1 << k)[..., 0] > 0
        if only_unchanged:
            core &= same
        ring = ndimage.binary_dilation(core, np.ones((3, 3), bool)) & same & ~core
        out.append(np.where(core, 2, np.where(ring, 1, 0)).astype(np.uint8))
    return out


# ---------------------------------------------------------------- DDS out (changed blocks only)
def encode_level(rgba):
    h, w = rgba.shape[:2]
    if w < 4 or h < 4:
        rgba = np.pad(rgba, ((0, max(0, 4 - h)), (0, max(0, 4 - w)), (0, 0)), mode="edge")
    bio = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(bio, format="DDS", pixel_format="DXT5")
    raw = bio.getvalue()
    assert raw[84:88] == b"DXT5"
    return np.frombuffer(raw[128:], np.uint8).reshape(-1, 16)


def decode_level(header, w, h, blocks):
    hdr = bytearray(header)
    struct.pack_into("<I", hdr, 12, max(4, h))
    struct.pack_into("<I", hdr, 16, max(4, w))
    struct.pack_into("<I", hdr, 20, len(blocks))
    struct.pack_into("<I", hdr, 28, 1)
    return np.asarray(Image.open(io.BytesIO(bytes(hdr) + blocks)).convert("RGBA"))[:h, :w]


def block_err(a, tgt, bw, bh):
    """Mean |a - tgt| per 4x4 block (flattened like the DXT block order)."""
    h, w = tgt.shape[:2]
    e = np.abs(a - tgt.astype(np.float32)).mean(-1)
    e = np.pad(e, ((0, bh * 4 - h), (0, bw * 4 - w)))
    n = np.pad(np.ones((h, w)), ((0, bh * 4 - h), (0, bw * 4 - w)))
    return (e.reshape(bh, 4, bw, 4).sum((1, 3)) / n.reshape(bh, 4, bw, 4).sum((1, 3))).ravel()


def keep_dev(dec, stock, tgt):
    """Per texel: how far (max channel, grey levels) the decoded colour lies off the segment [stock, target].
    0 for the stock colour, the target and anything on the way between them."""
    dec, stock, tgt = (np.asarray(x, np.float32) for x in (dec, stock, tgt))
    seg = tgt - stock
    t = np.clip(((dec - stock) * seg).sum(-1) / np.maximum((seg * seg).sum(-1), 1e-9), 0.0, 1.0)
    return np.abs(dec - (stock + t[..., None] * seg)).max(-1)


def keep_excess(dec, stock, tgt, cls):
    """Grey levels by which a decoded texel breaks the limit of its protection class (0 = fine)."""
    dec, stock, tgt = (np.asarray(x, np.float32) for x in (dec, stock, tgt))
    strict = np.maximum(keep_dev(dec, stock, tgt) - KEEP_TOL, 0.0)
    m = lambda x: x.mean(-1, keepdims=True)
    hue = np.maximum(keep_dev(dec - m(dec), stock - m(stock), tgt - m(tgt)) - KEEP_TOL, 0.0)
    md, ms, mt = dec.mean(-1), stock.mean(-1), tgt.mean(-1)
    lum = np.maximum(np.maximum(np.minimum(ms, mt) - md, md - np.maximum(ms, mt)) - RING_LUM_TOL, 0.0)
    return np.where(cls == 2, strict, np.where(cls == 1, hue + lum, 0.0))


# ---- a small constrained BC1 colour-block encoder (only for the blocks where Pillow's encode is refused)
_C565_MAX = np.array([31, 63, 31])


def _expand565(e):
    r, g, b = e[..., 0], e[..., 1], e[..., 2]
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1)


def _palette(E):
    """(N, 2, 3) endpoints in 5:6:5 units -> (N, 4, 3) decoded palette. BC3 colour blocks always decode in
    4-colour mode: c0, c1, (2 c0 + c1) / 3, (c0 + 2 c1) / 3 (integer, as Pillow; D3D may differ by 1)."""
    p0, p1 = _expand565(E[:, 0]), _expand565(E[:, 1])
    return np.stack([p0, p1, (2 * p0 + p1) // 3, (p0 + 2 * p1) // 3], 1)


def _block_cost(E, T, S, keep, valid):
    """Cost of N endpoint pairs for one block: squared error to the target over the valid texels, plus a
    dominating penalty for every protected texel (keep = class, see protected_levels) beyond its limit."""
    P = _palette(E).astype(np.float32)[:, :, None, :]                     # N, 4, 1, 3
    err = ((P - T) ** 2).sum(-1)                                           # N, 4, 16
    ex = keep_excess(P, S, T, keep)                                        # N, 4, 16
    c = err + 1e6 * ex
    idx = c.argmin(1)                                                      # N, 16
    cb = np.take_along_axis(c, idx[:, None], 1)[:, 0]
    eb = np.take_along_axis(ex, idx[:, None], 1)[:, 0]
    total = np.where(valid, cb, 0.0).sum(-1)
    ok = ~(valid & (eb > 0)).any(-1)
    return total, idx, ok


def _moves():
    m = []
    for e in (0, 1):
        for ch in range(3):
            for d in (-4, -2, -1, 1, 2, 4):
                x = np.zeros((2, 3), int)
                x[e, ch] = d
                m.append(x)
    for ch in range(3):
        for d in (-2, -1, 1, 2):
            x = np.zeros((2, 3), int)
            x[:, ch] = d                                                   # both ends together
            m.append(x)
            y = np.zeros((2, 3), int)
            y[0, ch], y[1, ch] = d, -d                                     # spread / squeeze
            m.append(y)
    return np.array(m)


_MOVES = _moves()


def _to565(rgb):
    return np.clip(np.round(np.asarray(rgb, np.float32) * _C565_MAX / 255.0), 0, _C565_MAX).astype(int)


def _ends_of(block):
    c0, c1 = struct.unpack_from("<HH", bytes(block[8:12]))
    return np.array([[(c >> 11) & 31, (c >> 5) & 63, c & 31] for c in (c0, c1)])


def _pack(E, idx):
    """Endpoints + indices -> 8-byte colour half, with c0 > c1 (4-colour mode for any decoder)."""
    c0, c1 = (int(e[0]) << 11 | int(e[1]) << 5 | int(e[2]) for e in E)
    idx = np.asarray(idx).copy()
    if c0 < c1:
        c0, c1, idx = c1, c0, np.array([1, 0, 3, 2])[idx]
    elif c0 == c1:
        idx[:] = 0
    word = sum(int(i) << (2 * t) for t, i in enumerate(idx))
    return np.frombuffer(struct.pack("<HHI", c0, c1, word), np.uint8)


def _anchor_pairs(cols):
    """Endpoint pairs (5:6:5) that put any two of the given colours exactly on two of the 4 palette positions
    (0, 1/3, 2/3, 1): a coarse global search over the colinear BC1 palettes that fit the block."""
    cols = np.unique(np.round(cols), axis=0)
    i, j = np.triu_indices(len(cols), 1)
    A, B = cols[i], cols[j]
    out = [np.stack([_to565(cols), _to565(cols)], 1)]
    pos = (0.0, 1 / 3, 2 / 3, 1.0)
    for ta in pos:
        for tb in pos:
            if ta == tb:
                continue
            step = (B - A) / (tb - ta)                                     # c1 - c0
            c0 = A - ta * step
            out.append(np.stack([_to565(np.clip(c0, 0, 255)), _to565(np.clip(c0 + step, 0, 255))], 1))
    return np.unique(np.concatenate(out), axis=0)


def encode_block_constrained(T, S, keep, valid, starts, n_climb=6):
    """Best BC1 colour half for one block with the protected texels as hard limits: coarse search over anchor
    pairs (target and stock colours of the block), then hill-climbs on the two 5:6:5 endpoints from the best
    few and from `starts`. Returns (8 colour bytes, feasible)."""
    v = valid.astype(bool)
    cand = _anchor_pairs(np.concatenate([T[v], S[v & (keep > 0)]]))
    cs, _, _ = _block_cost(cand, T, S, keep, v)
    seeds = [cand[j] for j in np.argsort(cs)[:n_climb]] + [np.asarray(E, int) for E in starts]
    best = None
    for E in seeds:
        E = np.clip(np.asarray(E, int), 0, _C565_MAX)
        cost = _block_cost(E[None], T, S, keep, v)[0][0]
        for _ in range(400):
            C = np.clip(E[None] + _MOVES, 0, _C565_MAX)
            cs, _, _ = _block_cost(C, T, S, keep, v)
            j = int(cs.argmin())
            if cs[j] >= cost - 1e-6:
                break
            E, cost = C[j], cs[j]
        if best is None or cost < best[0]:
            best = (cost, E)
    _, idx, ok = _block_cost(best[1][None], T, S, keep, v)
    return _pack(best[1], idx[0]), bool(ok[0])


def _blocks(a, bw, bh):
    """(h, w, ...) -> (bh * bw, 16, ...) in DXT block order, texel t = 4 y + x; edge-padded."""
    h, w = a.shape[:2]
    pad = [(0, bh * 4 - h), (0, bw * 4 - w)] + [(0, 0)] * (a.ndim - 2)
    a = np.pad(a, pad, mode="edge")
    rest = a.shape[2:]
    return a.reshape((bh, 4, bw, 4) + rest).swapaxes(1, 2).reshape((bh * bw, 16) + rest)


def write_levels(path, orig, levels, keep):
    """orig: original DDS bytes; levels: float RGB per mip level; keep: per level the protection classes of
    protected_levels (2: texel of a non-nut part, 1: its unused bilinear neighbour, 0: free). 4x4 blocks
    without any colour change keep their original 16 bytes. A changed block gets a new colour half and the
    ORIGINAL alpha half only if
      - every protected texel in it stays within its limit (class 2: KEEP_TOL of the segment [stock, target]
        in every channel; class 1: no hue shift) - a DXT block is shared by 16 texels: without this, a block
        where the nut texels gain a lot could snap a neighbouring rim-lip texel to the pink palette and still
        win on the mean error (mip 5-7 of rim_d), and
      - the block's mean error to the target is lower than the stock block's (on the tiniest mips the change
        is a fraction of a grey level and BC3 re-encoding error would dominate).
    Pillow's BC3 encode is tried first; where it breaks the first rule a constrained encode (protected texels
    as hard limits, squared error to the target otherwise) is tried; where that fails too, the stock block
    stays. Returns per level (blocks re-encoded by Pillow, by the constrained encoder, refused -> stock)."""
    hh = struct.unpack_from("<31I", orig, 4)
    mips = max(1, hh[6])
    lv = decode_levels(orig)
    assert len(levels) == len(lv) == len(keep) == mips
    out = bytearray(orig[:128])
    off, stats = 128, []
    for k in range(mips):
        h, w = lv[k].shape[:2]
        tgt = np.clip(levels[k] + 0.5, 0, 255).astype(np.uint8)
        st = lv[k][..., :3]
        chg = np.any(tgt != st, -1)
        bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
        nb = bw * bh
        ob = np.frombuffer(orig[off:off + nb * 16], np.uint8).reshape(-1, 16).copy()
        n_pil = n_con = n_ref = 0
        if chg.any():
            valid = _blocks(np.pad(np.ones((h, w), bool), ((0, bh * 4 - h), (0, bw * 4 - w))), bw, bh)
            kp = np.where(valid, _blocks(keep[k], bw, bh), 0)
            Tb = _blocks(tgt, bw, bh).astype(np.float32)
            Sb = _blocks(st, bw, bh).astype(np.float32)
            cp = _blocks(chg, bw, bh).any(1) & valid.any(1)
            e_old = block_err(st.astype(np.float32), tgt, bw, bh)

            def judge(cand):
                dec = decode_level(orig[:128], w, h, cand.tobytes())[..., :3].astype(np.float32)
                ex = _blocks(keep_excess(dec, st, tgt, keep[k]), bw, bh)
                return ~(valid & (ex > 0)).any(1), block_err(dec, tgt, bw, bh) < e_old

            enc = encode_level(np.dstack([tgt, lv[k][..., 3]]))
            cand = ob.copy()
            cand[cp, 8:] = enc[cp, 8:]
            safe, better = judge(cand)
            use = cp & safe & better
            ob[use, 8:] = enc[use, 8:]
            n_pil = int(use.sum())
            retry = np.nonzero(cp & ~safe)[0]
            if len(retry):
                cand = ob.copy()
                for i in retry:
                    col, ok = encode_block_constrained(Tb[i], Sb[i], kp[i], valid[i],
                                                       [_ends_of(ob[i]), _ends_of(enc[i])])
                    if ok:
                        cand[i, 8:] = col
                safe, better = judge(cand)
                use2 = np.zeros(nb, bool)
                use2[retry] = safe[retry] & better[retry]
                ob[use2, 8:] = cand[use2, 8:]
                n_con = int(use2.sum())
                n_ref = len(retry) - n_con
        stats.append((n_pil, n_con, n_ref))
        out += ob.tobytes()
        off += nb * 16
    assert off == len(orig) and len(out) == len(orig)
    with open(path + ".tmp", "wb") as fh:
        fh.write(out)
    os.replace(path + ".tmp", path)
    return stats


def tinted_base(orig, tint):
    """The stock DDS with every level's colour multiplied by tint (per channel): colour halves re-encoded by
    Pillow's BC3 from (stock level * tint), alpha halves and header byte-identical to the original. With
    tint 1 the original bytes. Returns (DDS bytes, decoded levels, float tinted levels)."""
    tint = np.broadcast_to(np.asarray(tint, np.float32), (3,))
    lv = decode_levels(orig)
    G = [x[..., :3].astype(np.float32) * tint for x in lv]
    if np.allclose(tint, 1.0):
        return orig, lv, G
    out, off = bytearray(orig[:128]), 128
    for k, x in enumerate(lv):
        h, w = x.shape[:2]
        nb = max(1, (w + 3) // 4) * max(1, (h + 3) // 4)
        ob = np.frombuffer(orig[off:off + nb * 16], np.uint8).reshape(-1, 16).copy()
        enc = encode_level(np.dstack([np.clip(G[k] + 0.5, 0, 255).astype(np.uint8), x[..., 3]]))
        ob[:, 8:] = enc[:, 8:]
        out += ob.tobytes()
        off += nb * 16
    out = bytes(out)
    return out, decode_levels(out), G


def snap_to_base(levels, G, base_lv):
    """Texels whose target is just the tinted stock colour (no nut change) take the decoded base colour, so
    write_levels sees them as unchanged and protects them against it (only the nut blocks are re-encoded)."""
    out = []
    for F, g, b in zip(levels, G, base_lv):
        nut = np.abs(F - g).max(-1) > 1e-3
        out.append(np.where(nut[..., None], F, b[..., :3].astype(np.float32)))
    return out


def parse_colour(s):
    if s in ("graphite", "default"):
        return GRAPHITE
    if s in ("stock", "white", "none"):
        return None
    c = tuple(int(v) for v in s.split(","))
    assert len(c) == 3 and all(0 <= v <= 255 for v in c), s
    return c


def colour_check(path, levels, nut_mask0, rim_mask0):
    """Level-0 check of the written file: mean / max abs error of the rim texels to the graphite target, their
    max chroma (pink bleed would show here), and the nut texels' error to the pink target."""
    a = decode_levels(open(path, "rb").read())[0][..., :3].astype(np.float32)
    t = levels[0]
    e = np.abs(a - t).max(-1)
    chroma = lambda x: x.max(-1) - x.min(-1)
    r = dict(rim_mean_err=round(float(e[rim_mask0].mean()), 2), rim_max_err=round(float(e[rim_mask0].max()), 1),
             rim_max_chroma=round(float(chroma(a[rim_mask0]).max()), 1),
             rim_target_max_chroma=round(float(chroma(t[rim_mask0]).max()), 1))
    if nut_mask0.any():
        r.update(nut_mean_err=round(float(e[nut_mask0].mean()), 2), nut_max_err=round(float(e[nut_mask0].max()), 1),
                 nut_mean_rgb=rgb3(a[nut_mask0].mean(0)))
    return r


def verify(path, orig, levels, keep):
    data = open(path, "rb").read()
    a, b = decode_levels(data), decode_levels(orig)
    same_alpha = all(np.array_equal(data[o:o + 8], orig[o:o + 8]) for o in range(128, len(data), 16))
    tgt0 = np.clip(levels[0] + 0.5, 0, 255).astype(np.uint8)
    chg = np.any(tgt0 != b[0][..., :3], -1)
    h, w = chg.shape
    blk = chg.reshape(h // 4, 4, w // 4, 4).any((1, 3))                # 4x4 blocks holding changed texels
    outside = ~np.repeat(np.repeat(blk, 4, 0), 4, 1)
    err = np.abs(a[0][..., :3][chg].astype(np.float32) - levels[0][chg]).mean() if chg.any() else 0.0
    # every level: texels of non-nut parts stay on [stock, target] within KEEP_TOL, their unused neighbours
    # keep their hue (protected_levels)
    kd, kx = [], 0.0
    for k in range(len(a)):
        t = np.clip(levels[k] + 0.5, 0, 255).astype(np.uint8)
        kd.append(float(keep_dev(a[k][..., :3], b[k][..., :3], t)[keep[k] == 2].max(initial=0.0)))
        kx = max(kx, float(keep_excess(a[k][..., :3], b[k][..., :3], t, keep[k]).max()))
    hh = struct.unpack_from("<31I", data, 4)
    return dict(bytes=len(data), size=f"{hh[3]}x{hh[2]}", fourcc=data[84:88].decode(), mips=len(a),
                same_header=data[:128] == orig[:128], alpha_blocks_identical=same_alpha,
                alpha_minmax=(int(a[0][..., 3].min()), int(a[0][..., 3].max())),
                rest_of_level0_identical=bool(np.array_equal(a[0][outside], b[0][outside])),
                mean_dxt_err_changed=round(float(err), 2),
                non_nut_max_dev_per_level=[round(x, 1) for x in kd],
                protected_ok_all_levels=kx == 0.0)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"), help="writes <out>/<skin>/rim_d.dds, rim_blur.dds")
    ap.add_argument("--skins", default=",".join(SKINS))
    ap.add_argument("--debug", default=None, help="also save masks / level-0 PNGs into this folder")
    ap.add_argument("--no-blur", action="store_true", help="skip rim_blur.dds")
    ap.add_argument("--rim-colour", default=None,
                    help="in-game rim colour as a car_paint_rims texel for every skin listed: R,G,B | graphite "
                         f"({GRAPHITE}) | stock (white rim, old output); default: per skin, SKIN_RIMS")
    a = ap.parse_args()
    skins = [s for s in a.skins.split(",") if s]
    by_col = {}
    for skin in skins:
        by_col.setdefault(a.rim_colour or SKIN_RIMS.get(skin, "graphite"), []).append(skin)
    for col_name, col_skins in by_col.items():
        print(f"=== {', '.join(col_skins)}: rim colour {col_name}")
        run(a, parse_colour(col_name), col_skins)


def run(a, rim_col, skins):

    tex, mats = load_textures()
    rim_m = mats["rim"]
    p = rim_m["props"]
    det = np.asarray(Image.open(io.BytesIO(tex[DETAIL])).convert("RGBA"))
    print(f"material rim ({rim_m['shader']}): useDetail={p.get('useDetail')} "
          f"detailUVMultiplier={p.get('detailUVMultiplier')} samplers={rim_m['samplers']}")
    print(f"  {DETAIL}: {det.shape[1]}x{det.shape[0]}, texel {tuple(int(v) for v in det[0, 0])} "
          f"(constant: {bool((det == det[0, 0]).all())}) -> diffuse x {det[0, 0, 0] / 255:.3f} where rim_d alpha is 0")
    bm = mats["rimblur"]
    print(f"material rimblur ({bm['shader']}): useDetail={bm['props'].get('useDetail')} samplers={bm['samplers']}")
    tint = np.ones(3, np.float32) if rim_col is None else np.array(rim_col, np.float32) / det[0, 0, :3]
    print(f"rim colour: {'stock' if rim_col is None else rim_col} -> non-nut texels x {np.round(tint, 3).tolist()} "
          f"(car_paint_rims.dds untouched)")
    print(f"pink: PIG {PIG} -> anodised (sat x{SAT_GAIN}) {rgb3(NUT)}, texel {rgb3(NUT_TEX)} -> in game "
          f"x{det[0, 0, 0] / 255:.3f} = {rgb3(NUT_TEX * det[0, 0, 0] / 255)}")

    rims, blurs = wheel_meshes()
    lf = [x for x in rims if "_LF/" in x["name"]][0]
    tri_nut, rows = nut_islands(lf)
    print(f"{lf['name']}: {len(lf['idx'])} tris, {len(rows)} UV islands (4 rims share UVs)")
    for k, nt, ar, r0, r1, x0, x1, u0, u1, v0, v1, kind in sorted(rows, key=lambda r: (r[-1] == "other", r[7], r[9])):
        if kind != "other" or ar > 100:
            print(f"  island {k:2d} {kind:8s} {nt:4d} tris {ar:8.2f} cm2  r {r0:.3f}-{r1:.3f} m  |x| {x0:.3f}-{x1:.3f}"
                  f"  uv x {u0:6.1f}-{u1:6.1f} y {v0:6.1f}-{v1:6.1f}")
    kinds = {}
    for r in rows:
        kinds[r[-1]] = kinds.get(r[-1], 0) + 1
    print("  nut islands:", {k: v for k, v in kinds.items() if k != "other"}, f"; other islands: {kinds.get('other', 0)}")

    rim_bytes = tex[RIM]
    rim_lv = decode_levels(rim_bytes)
    size = rim_lv[0].shape[0]
    nut_any, nut_cen = raster(lf, tri_nut, size)
    other_any, _ = raster(lf, ~tri_nut, size)
    shared = nut_any & other_any
    gap = ndimage.distance_transform_edt(~other_any)[nut_any].min()
    print(f"  nut texels: {nut_any.sum()} (centre-covered {nut_cen.sum()}), shared with other parts: {shared.sum()}, "
          f"closest other-part texel: {gap:.1f} texels; nut bbox x {nut_any.any(0).nonzero()[0][[0, -1]].tolist()} "
          f"y {nut_any.any(1).nonzero()[0][[0, -1]].tolist()}")
    assert shared.sum() == 0
    new_rim, region, zone, st = recolour_rim(rim_lv[0], nut_any, nut_cen, other_any)
    print(f"  rim_d nut grey: min {st['l_min']:.0f} median {st['l_med']:.0f} L_REF (p99) {st['l_ref']:.1f}; "
          f"texels recoloured incl. padding: {st['n_region']} (other texels of level 0 untouched)")
    # protected on every level: texels whose footprint touches any non-nut island (rim lip, spokes, barrel ...)
    rim_new = rim_levels(rim_lv, new_rim, region, zone, tint)
    rim_base, rim_base_lv, rim_G = tinted_base(rim_bytes, tint)
    rim_tgt = snap_to_base(rim_new, rim_G, rim_base_lv)
    jobs = [(RIM, rim_bytes, rim_base, rim_tgt, protected_levels(other_any, rim_tgt, rim_base_lv), nut_cen,
             other_any & ~nut_any)]

    if not a.no_blur:
        blur_bytes = tex[BLUR]
        used = blur_used(blurs, size)
        blur_orig = decode_levels(blur_bytes)
        blur_lv, bmask, bst = blur_levels(blur_orig, used, st["l_ref"], tint)
        print(f"rim_blur: blob core grey {bst['n_ref']:.0f} on hub {bst['bg_at_blobs']:.0f} -> nut pink "
              f"{rgb3(bst['pink'])}; level-0 texels changed {bst['n_changed']} (inside the hub used by the "
              f"rimblur meshes: {bool((bmask <= used).all())})")
        # rimblur: the hub is one island with the nut blobs on it -> protected = used texels left unchanged
        blur_base, blur_base_lv, blur_G = tinted_base(blur_bytes, tint)
        blur_tgt = snap_to_base(blur_lv, blur_G, blur_base_lv)
        jobs.append((BLUR, blur_bytes, blur_base, blur_tgt,
                     protected_levels(used, blur_tgt, blur_base_lv, only_unchanged=True),
                     bmask, used & ~ndimage.binary_dilation(bmask, iterations=2)))

    if a.debug:
        os.makedirs(a.debug, exist_ok=True)
        for name, _, _, levels, _, _, _ in jobs:
            Image.fromarray(np.clip(levels[0] + 0.5, 0, 255).astype(np.uint8)).save(
                os.path.join(a.debug, name.replace(".dds", "_new.png")))
        dbg = np.zeros((size, size, 3), np.uint8)
        dbg[zone] = (40, 20, 30)
        dbg[other_any] = (90, 90, 90)
        dbg[region] = (120, 40, 70)
        dbg[nut_any] = (255, 140, 170)
        dbg[shared] = (255, 255, 0)
        Image.fromarray(dbg).save(os.path.join(a.debug, "rim_d_islands.png"))

    for skin in skins:
        d = os.path.join(a.out, skin)
        os.makedirs(d, exist_ok=True)
        for name, orig, base, levels, keep, nut_m, rim_m in jobs:
            pth = os.path.join(d, name)
            nb = write_levels(pth, base, levels, keep)
            data = open(pth, "rb").read()
            same_alpha = all(data[o:o + 8] == orig[o:o + 8] for o in range(128, len(data), 16))
            print(f"wrote {pth}: nut 4x4 blocks per level (Pillow, constrained, refused -> base) {nb}; "
                  f"{verify(pth, base, levels, keep)}; vs stock: header identical {data[:128] == orig[:128]}, "
                  f"alpha identical {same_alpha}, size {len(data) == len(orig)}; "
                  f"level 0 {colour_check(pth, levels, nut_m, rim_m)}")


if __name__ == "__main__":
    main()
