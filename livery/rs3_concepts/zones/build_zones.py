"""Zone map of Audi RS3 LMS Skin.dds built from _analysis.npz (run analyze.py first).

Outputs (this folder): zones.json, masks/*.png (4096, white = zone), zones_overview.png, text_angle.png,
density_map.png, islands.png, part_tri.npy (per-triangle part id, used by the preview renderer).
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import tex_dir, pil_angle  # noqa: E402

D = np.load(os.path.join(HERE, "_analysis.npz"))
P, T, N, mid, J = D["P"], D["T"], D["N"], D["mid"], D["J"]
names = [str(s) for s in D["names"]]
lab, area3, dens, aniso = D["lab"], D["area3"], D["dens"], D["aniso"]
vis, proj, views = D["vis"], D["proj"], [str(v) for v in D["views"]]
idm = D["idm"]
R = idm.shape[0]
FULL = 4096
K = FULL // R
FONT = "/home/user/dramatron/livery/br03/brand/fonts/Exo2-Italic[wght].ttf"

C = P.mean(1)                       # centroids
x, y, z = C[:, 0], C[:, 1], C[:, 2]
nx, ny, nz = N[:, 0], N[:, 1], N[:, 2]
mesh = np.array([names[i].split("|")[1] for i in mid])
visfrac = np.clip(vis / np.maximum(proj, 1e-6), 0, 1)     # per view
vmax = visfrac.max(0)
best_view = np.array(views)[visfrac.argmax(0)]

# island areas (for small-detail detection)
isl_area = np.bincount(lab, weights=area3)
tri_isl_area = isl_area[lab]

# ------------------------------------------------------------------ part classification per triangle
part = np.full(len(P), "body_misc", dtype=object)
side = np.where(x > 0.06, "L", np.where(x < -0.06, "R", "C")).astype(object)
sidey = np.abs(nx) > 0.5

is_door = (mesh == "DOOR_L") | (mesh == "DOOR_R")
is_hood = mesh == "FRONT_HOOD"
is_rb = mesh == "REAR_BUMPER"
is_body = mesh == "Circle"

# doors (front doors are a separate mesh)
part[is_door] = "front_door"
part[is_door & (z > 0.97) & (y < -0.4)] = "mirror"
part[is_door & (tri_isl_area < 0.02) & (np.abs(nx) > 0.9)] = "door_handle"
part[is_door & (z < 0.42) & (part == "front_door")] = "front_door_low"

# hood
part[is_hood] = "hood"
part[is_hood & (ny < -0.45)] = "nose"            # front lip above the grille
part[is_hood & (ny > 0.3)] = "hood_underside"

# rear bumper
part[is_rb] = "rear_bumper"
part[is_rb & (np.abs(nx) > 0.6)] = "rear_bumper_corner"
part[is_rb & (z < 0.42) & (ny > 0.5)] = "rear_diffuser_band"

# body shell
b = is_body
part[b & (z > 1.3) & (nz > 0.8)] = "roof"
gh = b & (z > 1.0) & (part != "roof") & (np.abs(x) > 0.25) & (y > -1.0) & (y < 1.75)
part[gh & (y < -0.25)] = "a_pillar"
part[gh & (y >= -0.25) & (y < 0.55)] = "b_pillar"
part[gh & (y >= 0.55)] = "c_pillar"
part[gh & (z > 1.27)] = "roof_rail"
part[b & (y > 1.35) & (z > 0.84) & (nz > 0.5) & (np.abs(x) < 0.75)] = "trunk_lid"
part[b & (y > 1.85) & (ny > 0.6) & (z <= 0.98)] = "rear_panel"
sideb = b & (part == "body_misc") & (z < 1.08)
part[sideb & (y < -1.72)] = "front_bumper"
part[sideb & (y < -1.72) & (np.abs(nx) > 0.55)] = "front_bumper_corner"
part[sideb & (y < -1.72) & (z < 0.27)] = "front_splitter"
part[sideb & (y >= -1.72) & (y < -0.84)] = "front_fender"
part[sideb & (y >= -1.72) & (y < -0.84) & (nz > 0.6)] = "front_fender_top"
part[sideb & (y >= -0.84) & (y < 1.0)] = "rear_door"
part[sideb & (y >= 1.0)] = "rear_fender"
part[sideb & (y >= 1.0) & (np.abs(nx) < 0.5) & (np.abs(x) < 0.65)] = "trunk_lid"
part[sideb & (y >= 1.0) & (nz > 0.55) & (z > 0.85)] = "rear_shoulder"
part[b & (z < 0.3) & (y > -0.95) & (y < 1.0) & (np.abs(x) > 0.7)] = "sill"
part[b & (tri_isl_area < 0.02) & (np.abs(nx) > 0.9) & (z > 0.7) & (z < 0.9)] = "door_handle"
part[b & (vmax < 0.2)] = "hidden"
part[vmax < 0.12] = "hidden"
side[np.isin(part, ["roof", "trunk_lid", "rear_panel", "rear_bumper", "nose", "hood", "front_bumper",
                    "front_splitter", "rear_diffuser_band", "hood_underside"])] = "C"
part = part.astype(str)
side = side.astype(str)

# ------------------------------------------------------------------ text orientation per triangle
facing = np.where(nz > 0.62, "top", np.where(nz < -0.62, "bottom", np.where(np.abs(nx) >= np.abs(ny), "side",
                  np.where(ny < 0, "front", "rear"))))
# reading rule per part: side parts -> world up (z); front parts -> readable from the front (up = rear/up blend);
# rear parts -> readable from behind (up = forward/up blend).  For a surface whose normal lies in the y-z plane
# the projection of (0,+-1,1) equals the projection of z on vertical faces and of +-y on horizontal faces.
FRONT_READ = ["hood", "roof", "front_fender_top", "nose", "front_bumper", "front_splitter", "hood_underside"]
REAR_READ = ["trunk_lid", "rear_panel", "rear_bumper", "rear_diffuser_band", "rear_fender"]
up_t = np.zeros_like(N); up_t[:, 2] = 1.0
fr = np.isin(part, FRONT_READ) & (side == "C") | (part == "roof") | (part == "front_fender_top")
rr_ = np.isin(part, REAR_READ) & (side == "C")
up_t[fr] = (0, 1, 1)
up_t[rr_] = (0, -1, 1)
upp = up_t - np.einsum("ij,ij->i", up_t, N)[:, None] * N
ang = pil_angle(tex_dir(J, upp))
# alternative orientation for the roof: read from the car's LEFT side (helicopter/TV) and from behind
alt_left = pil_angle(tex_dir(J, np.tile([-1.0, 0, 0], (len(P), 1)) - nx[:, None] * N))
alt_back = pil_angle(tex_dir(J, np.tile([0, -1.0, 0], (len(P), 1)) - (-ny)[:, None] * N))
# world forward (-y) in tex space
fwd = tex_dir(J, np.tile([0, -1.0, 0], (len(P), 1)) + ny[:, None] * N)
upz = tex_dir(J, np.tile([0, 0, 1.0], (len(P), 1)) - nz[:, None] * N)

# ------------------------------------------------------------------ texel-level maps (R x R)
cov = idm >= 0
tid = np.where(cov, idm, 0)
PARTS = sorted(set(part))
pid = {p: i for i, p in enumerate(PARTS)}
part_map = np.where(cov, np.array([pid[p] for p in part])[tid], -1)
side_map = np.where(cov, np.array([{"L": 0, "R": 1, "C": 2}[s] for s in side])[tid], -1)
ang_map = np.where(cov, ang[tid], np.nan)
dens_map = np.where(cov, dens[tid], np.nan)
aniso_map = np.where(cov, aniso[tid], np.nan)
isl_map = np.where(cov, lab[tid], -1)
np.save(os.path.join(HERE, "part_tri.npy"), np.array([pid[p] for p in part]))

DMED = float(np.median(dens[part != "hidden"]))
stretch = cov & ((np.abs(dens_map / DMED - 1) > 0.18) | (aniso_map > 1.25))
# local angle change (curvature/seam of orientation): gradient of the angle field
a_rad = np.radians(np.nan_to_num(ang_map))
cx_, sx_ = np.cos(a_rad), np.sin(a_rad)
gx = np.hypot(ndimage.sobel(cx_, 0), ndimage.sobel(cx_, 1)) + np.hypot(ndimage.sobel(sx_, 0), ndimage.sobel(sx_, 1))


def largest_rect(m):
    """largest axis-aligned all-True rectangle in boolean array m -> (x0,y0,x1,y1) inclusive-exclusive."""
    h, w = m.shape
    hist = np.zeros(w, int)
    best = (0, None)
    for r in range(h):
        hist = np.where(m[r], hist + 1, 0)
        st = []
        for i in range(w + 1):
            cur = hist[i] if i < w else 0
            start = i
            while st and st[-1][1] >= cur:
                s, hh = st.pop()
                a = hh * (i - s)
                if a > best[0]:
                    best = (a, (s, r - hh + 1, i, r + 1))
                start = s
            st.append((start, cur))
    return best[1]


def circ_mean(a, w):
    r = np.radians(a)
    m = math.degrees(math.atan2((np.sin(r) * w).sum(), (np.cos(r) * w).sum()))
    dev = np.degrees(np.abs((r - math.radians(m) + np.pi) % (2 * np.pi) - np.pi))
    return m, dev


VIS_LABEL = {"side": "side", "top": "top", "front": "front", "rear": "rear"}
os.makedirs(os.path.join(HERE, "masks"), exist_ok=True)
zones = []
zone_map = -np.ones((R, R), int)
MIN_AREA = 0.004  # m^2
for p in PARTS:
    for s in ("L", "R", "C"):
        tm = (part == p) & (side == s)
        if not tm.any() or area3[tm].sum() < MIN_AREA:
            continue
        zm = (part_map == pid[p]) & (side_map == {"L": 0, "R": 1, "C": 2}[s])
        if zm.sum() < 20:
            continue
        zid = f"{p}_{s}" if s != "C" else p
        zi = len(zones)
        zone_map[zm] = zi
        A = float(area3[tm].sum())
        w = area3[tm] * vmax[tm]
        rot, dev = circ_mean(ang[tm], w)
        spread = float(np.average(dev, weights=w + 1e-12))
        fac = {f: float(area3[tm & (facing == f)].sum() / A) for f in ("side", "top", "front", "rear")}
        vcls = max(fac, key=fac.get) if p != "hidden" else "hidden"
        if p not in ("hidden",) and np.average(vmax[tm], weights=area3[tm]) < 0.35:
            vcls = "hidden"
        # safe area: zone texels, eroded, not stretched, orientation within 12 deg of the zone mean, not hidden
        rr = np.radians(np.nan_to_num(ang_map) - rot)
        orient_ok = np.abs(np.degrees((rr + np.pi) % (2 * np.pi) - np.pi)) < 12
        vis_ok = np.where(cov, vmax[tid] > 0.45, False)
        good = zm & ~stretch & orient_ok & vis_ok & (gx < 0.6)
        good = ndimage.binary_closing(good, iterations=2) & zm
        er = ndimage.binary_erosion(good, iterations=6)  # 6 px @1024 = 24 px @4096 (~3 cm)
        ys_, xs_ = np.nonzero(zm)
        bbox = [int(xs_.min() * K), int(ys_.min() * K), int((xs_.max() + 1) * K), int((ys_.max() + 1) * K)]
        safe = None
        if er.any():
            y0, y1 = ys_.min(), ys_.max() + 1
            x0, x1 = xs_.min(), xs_.max() + 1
            rect = largest_rect(er[y0:y1, x0:x1])
            if rect:
                safe = [int((rect[0] + x0) * K), int((rect[1] + y0) * K), int((rect[2] + x0) * K), int((rect[3] + y0) * K)]
        if safe is None or (safe[2] - safe[0]) * (safe[3] - safe[1]) < 48 * 48:
            safe = None
        if safe:
            sc = [(safe[0] + safe[2]) // 2, (safe[1] + safe[3]) // 2]
        else:
            cy_, cx2 = ndimage.center_of_mass(zm)
            sc = [int(cx2 * K), int(cy_ * K)]
        # local rotation at safe centre (median over safe box)
        dens_z = dens[tm]
        dok = bool(np.average(np.abs(dens_z / DMED - 1) < 0.18, weights=area3[tm]) > 0.9 and
                   np.average(aniso[tm] < 1.25, weights=area3[tm]) > 0.9)
        isl = sorted(set(int(i) for i in lab[tm]))
        cent = (C[tm] * area3[tm, None]).sum(0) / A
        # text 'length' direction in world for a box: run along text baseline (tex 'right' after rotation)
        r_ = math.radians(rot)
        base_tex = np.array([math.cos(r_), -math.sin(r_)])        # text x-axis after PIL rotate (CCW)
        Jm = np.average(J[tm], axis=0, weights=area3[tm] + 1e-12)
        bw = Jm @ base_tex
        bw = bw / (np.linalg.norm(bw) + 1e-12)
        notes = []
        if spread > 15:
            notes.append(f"orientation varies across zone (mean dev {spread:.0f} deg) - curved; use safe_box")
        if not dok:
            notes.append("texel density / stretch varies - avoid fine text outside safe_box")
        if vcls == "hidden":
            notes.append("mostly hidden/inside - base colour only")
        z = dict(id=zid, part=p, side={"L": "left", "R": "right", "C": "centre"}[s], visibility=vcls,
                 facing_fraction={k: round(v, 2) for k, v in fac.items()},
                 area_m2=round(A, 3), bbox=bbox, safe_box=safe, safe_center=sc,
                 safe_box_size_cm=([round((safe[2] - safe[0]) / DMED * 100), round((safe[3] - safe[1]) / DMED * 100)] if safe else None),
                 text_rotation=round(rot, 1), text_rotation_spread=round(spread, 1),
                 text_baseline_world=[round(float(v), 2) for v in bw],
                 density_ok=dok, density_px_per_m=round(float(np.median(dens_z))),
                 world_centroid=[round(float(v), 3) for v in cent], islands=isl, notes=notes,
                 mask=f"masks/{zid}.png")
        if p == "roof":
            z["text_rotation_alt"] = {"read_from_left_side": round(circ_mean(alt_left[tm], w)[0], 1),
                                      "read_from_behind": round(circ_mean(alt_back[tm], w)[0], 1)}
        zones.append(z)
        # 4096 mask: rasterize the zone triangles directly at full res
        im = Image.new("L", (FULL, FULL), 0)
        dr = ImageDraw.Draw(im)
        for t in np.nonzero(tm)[0]:
            dr.polygon([tuple(v) for v in (T[t] * FULL)], fill=255)
        im.save(os.path.join(HERE, "masks", f"{zid}.png"), optimize=True)
        print(f"{zid:28s} vis={vcls:6s} rot={rot:7.1f} spread={spread:5.1f} A={A:.3f} safe={safe}")

np.save(os.path.join(HERE, "zone_map.npy"), zone_map)
np.save(os.path.join(HERE, "ang_map.npy"), ang_map)

# ------------------------------------------------------------------ islands listing
isl_list = []
for l in range(lab.max() + 1):
    s = lab == l
    A = float(area3[s].sum())
    t = T[s].reshape(-1, 2) * FULL
    pc = {}
    for p_, a_ in zip(part[s], area3[s]):
        pc[p_] = pc.get(p_, 0) + a_
    sd = sorted(set(side[s]))
    isl_list.append(dict(island=l, bbox=[int(t[:, 0].min()), int(t[:, 1].min()), int(t[:, 0].max()), int(t[:, 1].max())],
                         area_m2=round(A, 4), side=sd, mesh=sorted(set(mesh[s])),
                         parts={k: round(v / A, 2) for k, v in sorted(pc.items(), key=lambda kv: -kv[1]) if v / A > 0.02},
                         density_px_per_m=round(float(np.median(dens[s]))),
                         stretched_fraction=round(float(area3[s & ((np.abs(dens / DMED - 1) > 0.18) | (aniso > 1.25))].sum() / A), 3)))

# ------------------------------------------------------------------ glass_sticker / ext_sticker zones
from analyze import load, tri_table, jacobians  # noqa: E402
meshes = load()
gz = []
for mat, mlist in (("glass_sticker", None), ("ext_sticker", None)):
    Pg, Tg, Ng, mg, nmg = tri_table(meshes, {mat})
    Jg, okg = jacobians(Pg, Tg)
    Cg = Pg.mean(1)
    ag = 0.5 * np.linalg.norm(np.cross(Pg[:, 1] - Pg[:, 0], Pg[:, 2] - Pg[:, 0]), axis=1)
    for mi, nm in enumerate(nmg):
        sel = (mg == mi) & okg
        # split into connected pieces by uv proximity (islands)
        from analyze import islands as _isl
        li = _isl(Pg[sel], Tg[sel])
        idx = np.nonzero(sel)[0]
        for l in range(li.max() + 1):
            tt = idx[li == l]
            A = ag[tt].sum()
            if A < 0.002:
                continue
            nmean = (Ng[tt] * ag[tt, None]).sum(0); nmean /= np.linalg.norm(nmean)
            cg = (Cg[tt] * ag[tt, None]).sum(0) / A
            tb = Tg[tt].reshape(-1, 2) * 1024
            bbox = [int(tb[:, 0].min()), int(tb[:, 1].min()), int(np.ceil(tb[:, 0].max())), int(np.ceil(tb[:, 1].max()))]
            if mat == "ext_sticker" and abs(nmean[2]) > 0.6:
                kind, viewer, upw = "wing_top", np.array([0, 0.6, 1.0]), np.array([0, -1.0, 1.0])
                how = "rear-wing top surface: readable from behind / above (text up = car forward)"
            elif mat == "ext_sticker":
                kind = "wing_endplate_" + ("left" if cg[0] > 0 else "right")
                viewer, upw = np.array([np.sign(cg[0]), 0, 0]), np.array([0, 0, 1.0])
                how = "rear-wing endplate outer face: upright, seen from the side"
            elif cg[1] < -0.2:
                kind = "windscreen_banner" if abs(cg[0]) < 0.3 else "windscreen_corner_" + ("left" if cg[0] > 0 else "right")
                viewer, upw = np.array([0, -1.0, 0.5]), np.array([0, 0, 1.0])
                how = "windscreen, readable from the front (outside)"
            elif cg[1] > 1.0:
                kind = "rear_window" + ("" if abs(cg[0]) < 0.25 else "_" + ("left" if cg[0] > 0 else "right"))
                viewer, upw = np.array([0, 1.0, 0.6]), np.array([0, 0, 1.0])
                how = "rear window, readable from behind (outside)"
            else:
                kind = "side_window_" + ("left" if cg[0] > 0 else "right")
                viewer, upw = np.array([np.sign(cg[0]), 0, 0]), np.array([0, 0, 1.0])
                how = "rear side window, readable from outside"
            if any(g["bbox"] == bbox and g["kind"] == kind for g in gz):
                continue        # double-sided duplicate geometry
            Nt = Ng[tt]
            flip = np.einsum("ij,j->i", Nt, viewer) < 0
            Nv = np.where(flip[:, None], -Nt, Nt)
            up = upw - np.einsum("ij,j->i", Nv, upw)[:, None] * Nv
            a_ = pil_angle(tex_dir(Jg[tt], up))
            rot, dev = circ_mean(a_, ag[tt])
            Rw = Jg[tt][:, :, 0]; Uw = -Jg[tt][:, :, 1]
            mir = float(np.average(np.einsum("ij,ij->i", np.cross(Rw, Uw), Nv) < 0, weights=ag[tt]))
            gz.append(dict(id=f"{mat}:{kind}", texture="glass_sticker.dds (1024)", mesh=nm.split("|")[0], kind=kind,
                           bbox=bbox, area_m2=round(float(A), 3), world_centroid=[round(float(v), 3) for v in cg],
                           text_rotation=round(rot, 1), text_rotation_spread=round(float(np.mean(dev)), 1),
                           mirrored_fraction=round(mir, 2), how=how,
                           notes=("UV is MIRRORED as seen by the viewer: flip artwork horizontally (ImageOps.mirror) "
                                  "before rotating" if mir > 0.5 else "")))
            print("STICKER", gz[-1])

json.dump(dict(texture="Skin.dds 4096x4096", grid="cells A..P (x) / 1..16 (y), 256 px",
               coord_note="tx = u*4096, ty = (1+v)*4096; +x world = car LEFT, +y = rear",
               rotation_note="text_rotation = PIL Image.rotate(angle, expand=True) angle applied to upright artwork "
                             "(positive = CCW). Side panels: text upright and parallel to ground; hood/roof: readable "
                             "from the front; trunk/rear: readable from behind.",
               density_median_px_per_m=round(DMED), part_names=PARTS, zones=zones, glass_sticker_zones=gz, islands=isl_list),
          open(os.path.join(HERE, "zones.json"), "w"), ensure_ascii=False, indent=1)
print("zones", len(zones), "DMED", DMED, "stretch frac", stretch.sum() / cov.sum())
