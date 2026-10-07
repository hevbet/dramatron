"""Texel -> world map at full Skin.dds resolution (4096), float32 (the shared posmap.npz is 2048/float16).
Cached in the scratchpad; built from rs3_concepts/zones/_analysis.npz + part_tri.npy (same data as posmap.py)."""
import os
import sys

import numpy as np
from scipy import ndimage

ZONES = "/home/user/dramatron/livery/rs3_concepts/zones"
CACHE = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad/bc/posmap4096.npz"
sys.path.insert(0, ZONES)


def build(res=4096, pad_px=10):
    from analyze import raster
    D = np.load(os.path.join(ZONES, "_analysis.npz"))
    P, T, Nt = D["P"], D["T"], D["N"]
    part_tri = np.load(os.path.join(ZONES, "part_tri.npy"))
    idm, cnt, bary = raster(T, res)
    cov = idm >= 0
    t = np.where(cov, idm, 0)
    pos = np.einsum("hwk,hwkc->hwc", bary, P[t]).astype(np.float32)
    del bary
    nrm = Nt[t].astype(np.float32)
    dist, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
    pad = (~cov) & (dist <= pad_px)
    pos[pad] = pos[iy[pad], ix[pad]]
    nrm[pad] = nrm[iy[pad], ix[pad]]
    tri = np.where(cov, idm, -1)
    tri[pad] = tri[iy[pad], ix[pad]]
    valid = cov | pad
    part = np.where(valid, part_tri[np.maximum(tri, 0)], -1).astype(np.int16)
    pos[~valid] = 0
    nrm[~valid] = 0
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, pos=pos, nrm=nrm.astype(np.float16), part=part, covered=cov)


def load():
    if not os.path.exists(CACHE):
        print("building 4096 posmap (one-off, ~1-2 min)...")
        build()
    d = np.load(CACHE)
    return d["pos"], d["nrm"].astype(np.float32), d["part"], d["covered"]


if __name__ == "__main__":
    build()
