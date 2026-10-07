"""Texel -> world map for Skin.dds (2048 res): posmap.npz with pos (H,W,3) float16 metres, nrm (H,W,3) float16,
tri (H,W) int32 (-1 = empty), island (H,W) int16, part (H,W) int16 (index into zones.json 'part_names').
Uncovered texels within 6 px of an island are filled from the nearest covered texel (seam bleed padding),
so projected artwork does not show dark seams after mip-mapping."""
import json
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import raster  # noqa: E402

RES = 2048
D = np.load(os.path.join(HERE, "_analysis.npz"))
P, T, N, lab = D["P"], D["T"], D["N"], D["lab"]
part_tri = np.load(os.path.join(HERE, "part_tri.npy"))
idm, cnt, bary = raster(T, RES)
cov = idm >= 0
t = np.where(cov, idm, 0)
pos = np.einsum("hwk,hwkc->hwc", bary, P[t])
nrm = N[t]
# pad
dist, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
pad = (~cov) & (dist <= 6)
pos[pad] = pos[iy[pad], ix[pad]]; nrm[pad] = nrm[iy[pad], ix[pad]]
tri = np.where(cov, idm, -1); tri_p = tri.copy(); tri_p[pad] = tri[iy[pad], ix[pad]]
valid = cov | pad
pos[~valid] = 0; nrm[~valid] = 0
isl = np.where(valid, lab[np.maximum(tri_p, 0)], -1).astype(np.int16)
prt = np.where(valid, part_tri[np.maximum(tri_p, 0)], -1).astype(np.int16)
np.savez_compressed(os.path.join(HERE, "posmap.npz"), pos=pos.astype(np.float16), nrm=nrm.astype(np.float16),
                    tri=tri_p.astype(np.int32), covered=cov, island=isl, part=prt)
print("posmap", RES, "covered", cov.mean().round(3), "padded", pad.sum())
