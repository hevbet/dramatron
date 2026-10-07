"""Concept 4 «Pro Pink TCR» — Audi RS3 LMS (smp_audi_rs3_lms), two team cars.

The most "real motorsport" take on «Pink Pig 2.0 × Y2K»: a works-TCR two-tone (hot magenta upper body, deep
plum lower body) split by a sharp geometric chrome BLADE that follows the door crease and then kinks up over the
rear wide-body flare in one straight cut; a holographic hairline rides above the blade and wraps the whole car.
The Pink-Pig butcher chart appears only as an accent: dashed cut lines + vintage cut labels on the roof and the
rear wing. Strong sponsor hierarchy like a works car. Car 2 (Konopelko) is the colour-inverted twin.

Everything that crosses panels is painted in WORLD space through a texel->3D position map (4096), so lines
are continuous across door gaps / UV seams, and every logo is projected from a side/top/front/rear view, so
text is level with the ground and never distorted. Each decal is checked automatically: full coverage (not
clipped), clearance (cm) to the panel edge / colour blade / other decals, surface tilt, and that the painted
texels lie inside the zone mask(s) of rs3_concepts/zones/masks. Report -> check_report.txt.

Run:  python3 make_skin.py      -> Pozdnyakov_00/, Konopelko_00/, *.zip, texture_preview.png, check_report.txt
"""
import json
import math
import os
import struct
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import art as A  # noqa: E402
import posmap4k  # noqa: E402

LIV = "/home/user/dramatron/livery"
SCRATCH = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad"
SRC = os.path.join(SCRATCH, "rs3", "SMP01")
ZONES = os.path.join(LIV, "rs3_concepts", "zones")
N = 4096
PPM = 900                     # artwork pixels per metre (texture density is ~795 texel/m)

DRIVERS = {
    "Pozdnyakov_00": dict(sur="Поздняков", first="Станислав", full="Станислав Поздняков", invert=False),
    "Konopelko_00": dict(sur="Конопелько", first="Матвей", full="Матвей Конопелько", invert=True),
}

# ================================================================= source skin: logo erase, body mask, AO
print("loading source skin + posmap ...")
src = Image.open(os.path.join(SRC, "Skin.dds")).convert("RGB")
A0 = np.asarray(src).astype(np.float32)
mx = A0.max(-1)
logo = ndimage.binary_dilation(mx > 110, iterations=6)
rings = np.zeros_like(logo)
rings[720:790, 1950:2140] = True                      # Audi rings on the trunk stay (baked)
logo &= ~rings
_, (iy, ix) = ndimage.distance_transform_edt(logo, return_indices=True)
AS = A0.copy()
AS[logo] = AS[iy[logo], ix[logo]]
mx = AS.max(-1)
BODY_COL = (AS[..., 2] > AS[..., 0] + 12) & (mx > 18)
SHADE = np.clip(0.55 + 0.45 * np.clip(mx / 76.0, 0, 1.12), 0.42, 1.06)

POS, NRM, PART, COV = posmap4k.load()
ZJ = json.load(open(os.path.join(ZONES, "zones.json")))
PNAMES = ZJ["part_names"]
PAINT = (BODY_COL | (~COV & (PART >= 0))) & ~rings
_, (sy, sx) = ndimage.distance_transform_edt(~(COV & BODY_COL), return_indices=True)
SHADE = np.where(COV & BODY_COL, SHADE, SHADE[sy, sx]).astype(np.float32)
del sy, sx, iy, ix
POSF = POS.reshape(-1, 3)
NRMF = NRM.reshape(-1, 3)
PARTF = PART.reshape(-1)
COVF = COV.reshape(-1)
PIDX = np.flatnonzero(PAINT.reshape(-1))          # texels we paint
P_IDX = {n: np.flatnonzero(PARTF == i) for i, n in enumerate(PNAMES)}

# ================================================================= the colour split ("blade") in world (y, z)
# measured door crease (front-door / lower-strip seam): z = 0.387 @ y=-0.84 ... 0.465 @ y=0.48, then the wide-body
# flare rises to z~0.82 @ y=1.16. The cut follows the crease, then kinks up in ONE straight line over the flare.
SPLIT = [(-2.6, 0.470), (-1.80, 0.452), (-0.95, 0.381), (-0.84, 0.387), (0.40, 0.466),
         (1.16, 0.862), (1.40, 0.885), (2.7, 0.885)]
BL0, BL1 = 0.000, 0.034        # chrome blade: from the cut up to +3.4 cm (perpendicular, in the side plane)
HL0, HL1 = 0.046, 0.056        # holographic hairline above the blade
KEEP = 0.025                   # decals keep 2.5 cm of air from the blade system


def _dense(poly, step=0.001):
    pts = []
    for (y0, z0), (y1, z1) in zip(poly[:-1], poly[1:]):
        n = max(2, int(math.hypot(y1 - y0, z1 - z0) / step))
        t = np.linspace(0, 1, n, endpoint=False)
        pts.append(np.stack([y0 + (y1 - y0) * t, z0 + (z1 - z0) * t], -1))
    pts.append(np.array([poly[-1]]))
    return np.concatenate(pts)


def split_z(y):
    ys, zs = zip(*SPLIT)
    return np.interp(y, ys, zs)


print("blade distance field ...")
_tree = cKDTree(_dense(SPLIT))
SD = np.full(N * N, 9.0, np.float32)              # signed distance (m) to the cut in the (y,z) plane, + = above
_p = POSF[PIDX]
_near = np.abs(_p[:, 2] - split_z(_p[:, 1])) < 0.25
_d, _ = _tree.query(_p[_near][:, 1:3], distance_upper_bound=0.3)
_d = np.minimum(_d, 0.3).astype(np.float32)
_s = np.sign(_p[_near][:, 2] - split_z(_p[_near][:, 1])).astype(np.float32)
SD[PIDX[_near]] = _d * _s
SD[PIDX[~_near]] = np.where(_p[~_near][:, 2] > split_z(_p[~_near][:, 1]), 9.0, -9.0)
del _p, _near, _d, _s
TEXEL_M = 1.0 / 795.0

# ================================================================= decal machinery
VIEWS = {
    "left": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "right": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    "front": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "rear": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "top": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "top_rear": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "top_left": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
}
_mask_cache = {}


def zone_mask(name):
    if name not in _mask_cache:
        p = os.path.join(ZONES, "masks", name + ".png")
        _mask_cache[name] = (np.asarray(Image.open(p).convert("L")) > 127).reshape(-1) if os.path.exists(p) else None
    return _mask_cache[name]


class Car:
    def __init__(self, name, upper, lower):
        self.name, self.upper, self.lower = name, np.array(upper, np.float32), np.array(lower, np.float32)
        self.can = np.zeros((N * N, 3), np.float32)
        self.occ = np.zeros(N * N, bool)          # texels already carrying a decal (overlap check)
        self.report = []

    # ------------------------------------------------------------- base
    def paint_base(self):
        sd = SD[PIDX]
        p = POSF[PIDX]
        aa = TEXEL_M * 0.9

        def ss(e0, e1):           # anti-aliased band indicator for sd in [e0, e1]
            return np.clip((sd - e0) / aa + 0.5, 0, 1) * np.clip((e1 - sd) / aa + 0.5, 0, 1)
        up = np.clip(sd / aa + 0.5, 0, 1)[:, None]
        col = self.lower * (1 - up) + self.upper * up
        # chrome blade: bright-dark-bright gradient across its width (reads as polished metal)
        t = np.clip((sd - BL0) / (BL1 - BL0), 0, 1)
        st = [(0, 236), (0.12, 252), (0.42, 214), (0.55, 120), (0.66, 176), (0.88, 246), (1, 210)]
        v = np.interp(t, [s[0] for s in st], [s[1] for s in st])
        chrome = np.stack([v * 1.0, v * 0.97, v * 1.03], -1)
        b = ss(BL0, BL1)[:, None]
        col = col * (1 - b) + chrome * b
        # holographic hairline (hue drifts along the car)
        h = ss(HL0, HL1)[:, None]
        holo = A.holo_lookup(p[:, 1] * 0.55 + p[:, 2] * 0.8)
        col = col * (1 - h) + holo * h
        # thin dark keyline under the blade (separates chrome from the lower colour)
        k = ss(-0.006, 0.0)[:, None]
        col = col * (1 - k) + np.array(A.PLUM_D, np.float32) * k
        self.can[PIDX] = col

    def paint_field(self, mask_idx, rgb):
        self.can[mask_idx] = rgb

    # ------------------------------------------------------------- decals
    def decal(self, name, art, view, center, width_m, parts, side=None, paint_angle=80, check_angle=50,
              min_clear=0.015, blade_keep=True, zones=None, dry=False):
        r, u, d = (np.array(v, np.float32) for v in VIEWS[view])
        c = np.asarray(center, np.float32)
        art = art.convert("RGBA")
        W = max(8, int(round(width_m * PPM)))
        H = max(8, int(round(art.height * W / art.width)))
        art = art.resize((W, H), Image.LANCZOS)
        height_m = width_m * H / W
        cand = np.concatenate([P_IDX[p] for p in parts])
        if side is not None:
            sx_ = POSF[cand, 0]
            cand = cand[(sx_ > 0.05) if side == "L" else (sx_ < -0.05)]
        rel = POSF[cand] - c
        sa, sb, dep = rel @ r, rel @ u, rel @ d
        cosn = NRMF[cand] @ d
        # --- coverage / clearance check on a 4 mm grid around the art
        cell, padm = 0.004, 0.08
        gw, gh = int((width_m + 2 * padm) / cell) + 1, int((height_m + 2 * padm) / cell) + 1
        gx = ((sa + width_m / 2 + padm) / cell).astype(int)
        gy = ((height_m / 2 + padm - sb) / cell).astype(int)
        good = (cosn > math.cos(math.radians(check_angle))) & (np.abs(dep) < 0.12) & COVF[cand]
        if blade_keep:
            sdv = SD[cand]
            good &= ~((sdv > -KEEP) & (sdv < HL1 + KEEP))
        good &= ~self.occ[cand]
        ing = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        grid = np.zeros((gh, gw), bool)
        sel = ing & good
        grid[gy[sel], gx[sel]] = True
        bad = np.zeros((gh, gw), bool)
        selb = ing & ~good & (cosn > 0) & COVF[cand]
        bad[gy[selb], gx[selb]] = True
        grid = ndimage.binary_closing(grid, iterations=2) & ~bad
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = False
        clear = ndimage.distance_transform_edt(grid) * cell
        aw, ah = int(round(width_m / cell)), int(round(height_m / cell))
        alpha_small = np.asarray(art.getchannel("A").resize((aw, ah), Image.BOX)) > 20
        ox, oy = int(round(padm / cell)), int(round(padm / cell))
        sub = clear[oy:oy + ah, ox:ox + aw][:alpha_small.shape[0], :alpha_small.shape[1]]
        am = alpha_small[:sub.shape[0], :sub.shape[1]]
        clr = float(sub[am].min()) - cell / 2 if am.any() else 0.0
        if os.environ.get("DBG"):
            vis = np.zeros((gh, gw, 3), np.uint8); vis[grid] = (0, 120, 0); vis[bad] = (160, 0, 0)
            v2 = vis[oy:oy + ah, ox:ox + aw]; v2[am[:v2.shape[0], :v2.shape[1]]] += np.array((90, 0, 120), np.uint8)
            Image.fromarray(vis).resize((gw * 2, gh * 2)).save(os.path.join(SCRATCH, "pp4", f"dbg_{name}.png"))
        clipped = int((sub[am] <= 0).sum())
        tilt = float(np.degrees(np.arccos(np.clip(np.median(cosn[(np.abs(sa) < width_m / 2) & (np.abs(sb) < height_m / 2) & (cosn > 0)]) if True else 1, -1, 1))))
        if dry:
            return clipped == 0 and clr >= min_clear - 1e-6
        # --- paint
        px = (sa / width_m + 0.5) * W - 0.5
        py = (0.5 - sb / height_m) * H - 0.5
        m = (cosn > math.cos(math.radians(paint_angle))) & (np.abs(dep) < 0.12) & (px > -1) & (px < W) & (py > -1) & (py < H)
        idx = cand[m]
        a = np.asarray(art).astype(np.float32) / 255.0
        pre = a[..., :3] * a[..., 3:4]
        coords = np.stack([py[m], px[m]])
        al = ndimage.map_coordinates(a[..., 3], coords, order=1, mode="constant", cval=0.0)
        rgb = np.stack([ndimage.map_coordinates(pre[..., ch], coords, order=1, mode="constant", cval=0.0)
                        for ch in range(3)], -1) * 255.0
        self.can[idx] = self.can[idx] * (1 - al[:, None]) + rgb
        painted = idx[al > 0.04]
        self.occ[painted] = True
        # dilate occupancy a little (air between decals)
        # --- zone-mask containment (texture space)
        zn = zones or [p + ("_" + side if side else "") for p in parts]
        zm = None
        for z in zn:
            mk = zone_mask(z)
            if mk is None and side:
                mk = zone_mask(z.rsplit("_", 1)[0])
            if mk is not None:
                zm = mk if zm is None else (zm | mk)
        pc = painted[COVF[painted]]
        outside = int((~zm[pc]).sum()) if zm is not None else -1
        ok = clipped == 0 and outside == 0 and clr >= min_clear - 1e-6
        self.report.append(dict(name=name, view=view, size_cm=(round(width_m * 100, 1), round(height_m * 100, 1)),
                                clearance_cm=round(clr * 100, 1), clipped_cells=clipped, outside_zone_texels=outside,
                                tilt_deg=round(tilt, 1), zones=zn, ok=bool(ok)))
        return ok

    def sides(self, name, make_art, y, z, width_m, parts, mirror_art=False, **kw):
        """Same decal on both sides: drawn as seen from each side (left view: image right = rear)."""
        res = []
        for s, view, xs in (("L", "left", 1), ("R", "right", -1)):
            art = make_art(s)
            res.append(self.decal(f"{name}_{s}", art, view, (xs * 0.9, y, z), width_m, parts, side=s, **kw))
        return all(res)

    def fit(self, name, art, view, depth_c, region, wmax, parts, side=None, prefer=None, min_clear=0.02,
            check_angle=50, blade_keep=True, wmin=0.05, depth_tol=0.12, commit=True, **kw):
        """Find the LARGEST size (<= wmax) and a position inside `region` = (a0, a1, b0, b1) (view-plane metres:
        a along image-right, b along image-up) where the artwork keeps `min_clear` of air everywhere; then paint it.
        depth_c = world coordinate along the view direction of the surface (x for sides, z for top...)."""
        r, u, d = (np.array(v, np.float32) for v in VIEWS[view])
        cand = np.concatenate([P_IDX[p] for p in parts])
        if side is not None:
            sx_ = POSF[cand, 0]
            cand = cand[(sx_ > 0.05) if side == "L" else (sx_ < -0.05)]
        P = POSF[cand]
        sa, sb, dep = P @ r, P @ u, P @ d - depth_c
        cosn = NRMF[cand] @ d
        good = (cosn > math.cos(math.radians(check_angle))) & (np.abs(dep) < depth_tol) & COVF[cand]
        if blade_keep:
            sdv = SD[cand]
            good &= ~((sdv > -KEEP) & (sdv < HL1 + KEEP))
        good &= ~self.occ[cand]
        cell = 0.006
        a0, a1, b0, b1 = region
        gw, gh = int((a1 - a0) / cell) + 1, int((b1 - b0) / cell) + 1
        gx = ((sa - a0) / cell).astype(int)
        gy = ((b1 - sb) / cell).astype(int)
        ing = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        grid = np.zeros((gh, gw), bool)
        grid[gy[ing & good], gx[ing & good]] = True
        bad = np.zeros((gh, gw), bool)
        sel = ing & ~good & (cosn > 0) & COVF[cand]
        bad[gy[sel], gx[sel]] = True
        grid = ndimage.binary_closing(grid, iterations=2) & ~bad
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = False
        okc = ndimage.distance_transform_edt(grid) * cell >= (min_clear + 1.5 * cell)  # safety vs. the 4 mm verification grid
        from scipy.signal import fftconvolve
        notok = (~okc).astype(np.float32)
        art = art.convert("RGBA")
        pa = np.array(prefer if prefer else ((a0 + a1) / 2, (b0 + b1) / 2))
        w = wmax
        while w >= wmin:
            aw = max(2, int(round(w / cell)))
            ah = max(2, int(round(art.height * aw / art.width)))
            if ah < gh and aw < gw:
                am = (np.asarray(art.getchannel("A").resize((aw, ah), Image.BOX)) > 10).astype(np.float32)
                am = ndimage.binary_dilation(am, iterations=1).astype(np.float32)
                conv = fftconvolve(notok, am[::-1, ::-1], mode="valid")
                vy, vx = np.nonzero(conv < 0.5)
                if len(vx):
                    ca = a0 + (vx + aw / 2) * cell
                    cb = b1 - (vy + ah / 2) * cell
                    order = np.argsort((ca - pa[0]) ** 2 + (cb - pa[1]) ** 2)
                    for k in order[:6]:       # verify on the fine grid before committing
                        c = r * ca[k] + u * cb[k] + d * depth_c
                        args = dict(side=side, min_clear=min_clear, check_angle=check_angle, blade_keep=blade_keep, **kw)
                        if self.decal(name, art, view, c, aw * cell, parts, dry=True, **args):
                            if not commit:
                                return aw * cell
                            return self.decal(name, art, view, c, aw * cell, parts, **args)
            w *= 0.96
        if not commit:
            return 0.0
        self.report.append(dict(name=name, ok=False, error="does not fit"))
        print("DOES NOT FIT:", name)
        return False

    def fit_sides(self, name, make_art, region_y, region_z, wmax, parts, prefer=None, depth=0.88, **kw):
        """Fit on both sides. region_y/region_z in car coordinates (y: + = rear); art drawn as seen from that side."""
        jobs = []
        for s, view, xs in (("L", "left", 1), ("R", "right", -1)):
            y0, y1 = region_y
            if view == "left":       # image right = +y
                reg = (y0, y1, region_z[0], region_z[1])
                pf = prefer
            else:                    # image right = -y
                reg = (-y1, -y0, region_z[0], region_z[1])
                pf = (-prefer[0], prefer[1]) if prefer else None
            jobs.append((s, view, reg, pf, make_art(s)))
        # same size on both sides: the smaller of the two maxima
        w = min(self.fit(f"{name}_{s}", art, view, depth, reg, wmax, parts, side=s, prefer=pf, commit=False, **kw)
                for s, view, reg, pf, art in jobs)
        for s, view, reg, pf, art in jobs:
            self.fit(f"{name}_{s}", art, view, depth, reg, max(w, 0.05) * 1.0001, parts, side=s, prefer=pf, **kw)

    # ------------------------------------------------------------- output
    def image(self):
        img = self.can.reshape(N, N, 3) * SHADE[..., None]
        out = AS.copy()
        out[PAINT] = img[PAINT]
        out[rings] = A0[rings]
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ================================================================= livery content
def build_body(car, drv, inv):
    up_dark = inv                       # upper colour is plum on car 2
    ink_up = A.WHITE if up_dark else A.PLUM_D       # text colour on the upper colour
    W = A.WHITE
    # ---------- side: front door race number (white blade plate, plum Unbounded Black digits)
    car.fit_sides("number", lambda s: A.number_plate(620, 330), (-0.84, 0.10), (0.40, 0.92), 0.62,
                  ["front_door"], prefer=(-0.42, 0.66), min_clear=0.025)
    # ---------- rear door upper (above the blade): тайм-кафе «Арка» — main partner #1
    car.fit_sides("arka", lambda s: A.arka(400), (0.22, 1.25), (0.40, 1.00), 0.50, ["rear_door"],
                  prefer=(0.42, 0.78))
    # ---------- rear door lower (plum under the blade): «Симкарт» + url — main partner #2
    if not inv:   # wordmark straight on the plum
        sk_art = lambda s: A.simkart(300)
    else:         # on magenta the red «карт» would vanish -> brand night plate
        sk_art = lambda s: A.simkart_plate(700, url=False)
    car.fit_sides("simkart", sk_art, (0.22, 1.25), (0.22, 0.85), 0.60, ["rear_door"], prefer=(0.75, 0.40))
    # ---------- front bumper corner (ahead of the wheel): SMP RACING ESPORTS (series)
    car.fit_sides("smp_bumper", lambda s: A.smp(200), (-2.3, -1.5), (0.25, 0.85), 0.30,
                  ["front_bumper_corner", "front_fender"], prefer=(-1.9, 0.6), depth=0.85, depth_tol=0.2)
    # ---------- lower door strip = partner strip: DriveOil (fuel) + KARTING64 logo (artwork only, no lettering)
    car.fit_sides("driveoil", lambda s: A.driveoil(120), (-0.84, -0.30), (0.22, 0.47), 0.32, ["front_door_low"],
                  prefer=(-0.60, 0.33), min_clear=0.012)
    car.fit_sides("karting64", lambda s: A.karting64(300, flip=(s == "L")), (-0.25, 0.30), (0.22, 0.47), 0.42,
                  ["front_door_low"], prefer=(0.0, 0.33), min_clear=0.010)
    # ---------- sill: BR ENGINEERING
    car.fit_sides("br_sill", lambda s: A.mono("br_engineering_black.png", 100, W), (-1.0, 1.1), (0.10, 0.32), 0.70,
                  ["sill"], prefer=(0.0, 0.21), min_clear=0.008)
    # ---------- sill, ahead of BR ENGINEERING (under the front door): karting64.ru — set exactly like the
    # simkart.vercel.app link (F_SPON = Exo 2 ExtraBold Italic, white, no tracking). Region stops at y=-0.47 so
    # there is plenty of air to BR ENGINEERING further back on the sill; level with the ground by projection.
    car.fit_sides("k64_url", lambda s: A.text("karting64.ru", A.F_SPON(300), W), (-1.0, -0.47), (0.10, 0.32), 0.34,
                  ["sill"], prefer=(-0.72, 0.21), min_clear=0.010)
    # ---------- Y2K sparkles riding the blade where it kinks up (rear door, behind Арка)
    car.fit_sides("sparkle_a", lambda s: A.sparkle(80), (0.62, 1.25), (0.62, 1.0), 0.10, ["rear_door"],
                  prefer=(0.95, 0.88), min_clear=0.012)
    car.fit_sides("sparkle_b", lambda s: A.sparkle(80), (0.62, 1.25), (0.62, 1.0), 0.055, ["rear_door"],
                  prefer=(1.08, 0.93), min_clear=0.01)
    # ---------- behind the rear wheel (plum haunch): РАФ (national federation)
    car.fit_sides("raf", lambda s: A.mono("raf_black.png", 300, W), (1.70, 2.40), (0.30, 0.85), 0.16,
                  ["rear_fender"], prefer=(1.92, 0.58), depth=0.85, depth_tol=0.2, check_angle=40, min_clear=0.015)


def dashed_contour(w, h, margin, col, dash, gap, lw, n=4.0):
    """Closed dashed superellipse (|x|^n + |y|^n = 1) — a butcher-chart 'cut' outline that never ends abruptly."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rx, ry = w / 2 - margin, h / 2 - margin
    t = np.linspace(0, 2 * np.pi, 4000)
    x = w / 2 + rx * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / n)
    y = h / 2 + ry * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / n)
    seg = np.hypot(np.diff(x), np.diff(y))
    L = np.concatenate([[0], np.cumsum(seg)])
    period = dash + gap
    k = max(1, int(round(L[-1] / period)))          # whole number of dashes -> seamless loop
    period = L[-1] / k
    for i in range(k):
        s0, s1 = i * period, i * period + period * dash / (dash + gap)
        sel = (L >= s0) & (L <= s1)
        pts = list(zip(x[sel], y[sel]))
        if len(pts) > 1:
            d.line(pts, fill=tuple(col) + (255,), width=lw, joint="curve")
    return img


def roof_art(cut_col, label_col):
    """Roof (seen from the car's LEFT): number plate inside a dashed butcher 'cut' with the label ВЫРЕЗКА."""
    S = 900                                   # px per metre of this artwork
    w, h = int(1.06 * S), int(0.80 * S)
    img = dashed_contour(w, h, 14, cut_col, dash=0.045 * S, gap=0.028 * S, lw=int(0.012 * S))
    plate = A.number_plate(int(0.62 * S), int(0.33 * S))
    img.alpha_composite(plate, ((w - plate.width) // 2, int(0.13 * S)))
    lab = A.cut_label("ВЫРЕЗКА", int(0.07 * S), label_col)
    img.alpha_composite(lab, ((w - lab.width) // 2, int(0.13 * S) + plate.height + int(0.04 * S)))
    return img


def build_top(car, drv, inv):
    W = A.WHITE
    cut_col = A.MAG_L if inv else A.PLUM_D
    # ---------- roof: number (reads from the left, like TCR roof numbers) inside the butcher cut
    car.fit("roof_number_cut", roof_art(cut_col, cut_col), "top_left", 1.38, (-0.30, 1.25, -0.62, 0.62), 1.10,
            ["roof"], prefer=(0.45, 0.0), min_clear=0.03, depth_tol=0.15, blade_keep=False)
    # ---------- hood: «Симкарт» night plate with url — main partner #2 (seen from the front / in mirrors)
    car.fit("hood_simkart", A.simkart_plate(900), "top", 0.87, (-0.70, 0.70, -1.75, -0.85), 0.86,
            ["hood"], prefer=(0.0, -1.22), min_clear=0.03, depth_tol=0.2, check_angle=40, blade_keep=False)
    # ---------- trunk lid: «Команда ЭДМ» + МБУ «Клуб Энгельсская молодёжь» (team)
    # The rear-wing uprights stand on the trunk at |x| = 0.19..0.21 m (kn5 Plane.010), so from behind they would
    # cut a wide lockup in three. Team name goes BETWEEN the uprights; the МБУ sticker sits outboard on both sides.
    car.fit("trunk_edm", A.team_edm(300, col=(W if inv else A.PLUM_D), sticker=False), "top_rear", 1.0,
            (-0.165, 0.165, -2.15, -1.6), 0.30, ["trunk_lid"], prefer=(0.0, -1.84), min_clear=0.02, depth_tol=0.2,
            check_angle=22, blade_keep=True)
    # The trunk-lid pins (3D, not in the skin) stand at |x|~0.56-0.60 m: the outboard limit 0.52 keeps them off the
    # МБУ sticker, which is now enlarged (wmax 16 -> 22 cm) so it reads clearly from behind / above.
    for sgn, tag in ((1, "a"), (-1, "b")):
        reg = (0.25, 0.52) if sgn > 0 else (-0.52, -0.25)
        car.fit(f"trunk_mbu_{tag}", A.mbu(300), "top_rear", 1.0, (reg[0], reg[1], -2.15, -1.6), 0.22,
                ["trunk_lid"], prefer=(sgn * 0.385, -1.84), min_clear=0.02, depth_tol=0.2, check_angle=22,
                blade_keep=True)
    # ---------- rear panel between the lights: Саратовская область · 64 (region)
    car.fit("rear_region64", A.region64(300, col=W), "rear", 2.15, (-0.6, 0.6, 0.55, 1.0), 0.50,
            ["rear_panel"], prefer=(0.0, 0.80), min_clear=0.015, depth_tol=0.25, check_angle=45)
    # ---------- rear bumper: website
    car.fit("rear_url", A.text("simkart.vercel.app", A.F_SPON(300), W), "rear", 2.2, (-0.8, 0.8, 0.3, 0.8), 0.95,
            ["rear_bumper"], prefer=(0.0, 0.58), min_clear=0.015, depth_tol=0.25, check_angle=40)


def paint_parts(car, inv):
    """Mirror caps in the lower colour (classic works-car detail)."""
    lower = A.MAG if inv else A.PLUM
    for p in ("mirror",):
        idx = P_IDX[p]
        idx = idx[PAINT.reshape(-1)[idx]]
        car.can[idx] = np.array(lower, np.float32)


# ================================================================= glass_sticker.dds (1024): windscreen, rear window, side windows, wing
def _glass_cov():
    """UV coverage masks of the glass_sticker / ext_sticker meshes rasterised from the kn5 (1024)."""
    sys.path.insert(0, os.path.join(LIV, "tools"))
    import kn5
    d = kn5.load(os.path.join(SCRATCH, "kn5", "smp_audi_rs3_lms.kn5"))
    mats = d["materials"]
    cov = {}
    for m in d["meshes"]:
        mn = mats[m["mat"]]["name"]
        if mn not in ("glass_sticker", "ext_sticker"):
            continue
        im = Image.new("L", (1024, 1024), 0)
        dr = ImageDraw.Draw(im)
        for t in m["idx"]:
            dr.polygon([(m["uv"][i, 0] % 1 * 1024, m["uv"][i, 1] % 1 * 1024) for i in t], fill=255)
        cov[mn] = np.maximum(cov.get(mn, 0), np.asarray(im))
    return {k: v > 127 for k, v in cov.items()}


GCOV = None
GZ = {  # glass zones (1024 px): box (x0, y0, x1, y1)
    "windscreen": (118, 16, 899, 146), "rear_window": (135, 164, 864, 253),
    "side_L": (115, 272, 513, 496), "side_R": (114, 518, 512, 741),
    "wing_top": (104, 863, 941, 1006), "endplate_L": (331, 762, 516, 851), "endplate_R": (115, 766, 300, 854),
}


def glass(car, drv, inv):
    global GCOV
    if GCOV is None:
        GCOV = _glass_cov()
    g = Image.open(os.path.join(SRC, "glass_sticker.dds")).convert("RGBA")
    g = Image.new("RGBA", g.size, (0, 0, 0, 0))            # clean sheet: everything clear glass ...
    up, lo = (A.PLUM, A.MAG) if inv else (A.MAG, A.PLUM)
    band = A.PLUM_D if not inv else A.MAG
    cov_all = GCOV["glass_sticker"] | GCOV["ext_sticker"]
    rep = []

    def fill_box(box, col, mask=None):
        x0, y0, x1, y1 = box
        m = np.zeros((1024, 1024), bool)
        m[y0:y1, x0:x1] = True
        if mask is not None:
            m &= mask
        lay = Image.new("RGBA", (1024, 1024), tuple(col) + (255,))
        lay.putalpha(Image.fromarray((m * 255).astype(np.uint8)))
        g.alpha_composite(lay)

    def put(name, im, zone, cx, cy, rot=0, margin=6):
        if rot:
            im = im.rotate(rot, expand=True, resample=Image.BICUBIC)
        x, y = int(round(cx - im.width / 2)), int(round(cy - im.height / 2))
        lay = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        lay.paste(im, (x, y))
        al = np.asarray(lay.getchannel("A")) > 10
        x0, y0, x1, y1 = GZ[zone]
        zm = np.zeros((1024, 1024), bool)
        zm[y0:y1, x0:x1] = True
        zm &= cov_all
        inside = ndimage.distance_transform_edt(zm)
        clr = float(inside[al].min()) if al.any() else 0
        out = int((al & ~zm).sum())
        ok = out == 0 and clr >= margin
        rep.append(dict(name="glass:" + name, zone=zone, clearance_px=round(clr, 1), outside_px=out, ok=bool(ok)))
        g.alpha_composite(lay)

    # ---- windscreen banner (series): band + chrome blade underline + holo hairline, SMP RACING ESPORTS
    g.alpha_composite(Image.new("RGBA", (1024, 150), tuple(band) + (255,)), (0, 0))
    ch = A.chrome_rgb(1024, 9).convert("RGBA")
    g.alpha_composite(ch, (0, 130))
    g.alpha_composite(A.holo_rgb(1024, 3, 3.0).convert("RGBA"), (0, 124))
    put("smp_banner", A.smp(78), "windscreen", 508, 66)
    put("raf_banner", A.mono("raf_black.png", 92, A.WHITE), "windscreen", 205, 70)
    put("br_banner", A.mono("br_symbol_white.png", 64, A.WHITE), "windscreen", 812, 70)
    # ---- rear window banner (seen by following cars): «Симкарт» + url
    fill_box(GZ["rear_window"], A.PLUM_D, GCOV["glass_sticker"])
    g.alpha_composite(A.chrome_rgb(864 - 135, 6).convert("RGBA"), (135, 247))
    sk = A.fit_h(A.simkart(64), 70)
    put("rw_simkart", sk, "rear_window", 400, 206)
    put("rw_url", A.fit_h(A.text("simkart.vercel.app", A.F_SPON(120), A.WHITE), 28), "rear_window", 712, 206)
    # ---- side windows (rear door glass): driver name + Russian flag on a blade plate at the window bottom
    for zone in ("side_L", "side_R"):
        x0, y0, x1, y1 = GZ[zone]
        tag = A.name_tag(drv["sur"], drv["first"], 260)
        plate = A.blade_plate(tag.width + 34, tag.height + 22, band, fw=3)
        plate.alpha_composite(tag, (17, 11))
        cy = y1 - 52
        cx = x0 + 30 + plate.width / 2 if zone == "side_L" else x1 - 30 - plate.width / 2
        put("name_" + zone, plate, zone, cx, cy, margin=8)
    # ---- rear wing top (text up = car forward -> rotate 180): тайм-кафе «Арка» + butcher accent
    wx0, wy0, wx1, wy1 = GZ["wing_top"]
    fill_box((wx0 - 4, wy0 - 4, wx1 + 4, wy1 + 4), up, ndimage.binary_dilation(GCOV["ext_sticker"], iterations=3))
    ww, wh = wx1 - wx0, wy1 - wy0
    wing = Image.new("RGBA", (ww, wh), (0, 0, 0, 0))
    cut = A.MAG_L if inv else A.PLUM_D
    dr = ImageDraw.Draw(wing)
    for xd in (int(ww * 0.31), int(ww * 0.69)):          # two dashed 'cut' lines across the wing chord
        for yy in range(12, wh - 12, 22):
            dr.line([(xd, yy), (xd, min(wh - 12, yy + 13))], fill=cut + (255,), width=5)
    wing = wing.rotate(180)
    g.alpha_composite(wing, (wx0, wy0))
    put("wing_arka", A.fit_box(A.arka(300, taimcafe=False), ww * 0.33, wh - 30), "wing_top", (wx0 + wx1) / 2, (wy0 + wy1) / 2, rot=180, margin=8)
    for nm_, txt, xc in (("wing_lbl_1", "ХВОСТИК", wx1 - ww * 0.155), ("wing_lbl_2", "ТАЙМ-КАФЕ", wx0 + ww * 0.155)):
        put(nm_, A.fit_box(A.cut_label(txt, 60, cut), ww * 0.24, 60), "wing_top", xc, (wy0 + wy1) / 2, rot=180)
    # ---- endplates: Saratov region — white/red flag field, coat of arms + 64
    for zone in ("endplate_L", "endplate_R"):
        x0, y0, x1, y1 = GZ[zone]
        epm = ndimage.binary_dilation(GCOV["ext_sticker"], iterations=3)
        fill_box((x0 - 4, y0 - 4, x1 + 4, y1 + 4), (255, 255, 255), epm)
        fill_box((x0 - 4, y0 + int((y1 - y0) * 0.64), x1 + 4, y1 + 4), (206, 32, 42), epm)
        c = A.coat(46)
        n = A.fit_h(A.text("64", A.F_TECH(200), A.PLUM_D), 40)
        blk = A.row([c, n], 8)
        put("ep_64_" + zone, blk, zone, (x0 + x1) / 2, y0 + (y1 - y0) * 0.34, margin=4)
    # keep glass clear outside the painted areas: alpha 0 where nothing was drawn
    return g, rep


def build(name):
    drv = DRIVERS[name]
    inv = drv["invert"]
    upper, lower = (A.PLUM, A.MAG) if inv else (A.MAG, A.PLUM)
    car = Car(name, upper, lower)
    car.paint_base()
    paint_parts(car, inv)
    car.occ[(np.arange(N * N) // N >= 700) & (np.arange(N * N) // N < 800) & (np.arange(N * N) % N >= 1930) & (np.arange(N * N) % N < 2160)] = True  # Audi rings stay free
    build_body(car, drv, inv)
    build_top(car, drv, inv)
    return car


def livery_icon(inv):
    """185x185 AC livery swatch: upper colour, chrome blade, lower colour (car-2 inverted)."""
    up, lo = (A.PLUM, A.MAG) if inv else (A.MAG, A.PLUM)
    S = 185 * 4
    im = Image.new("RGBA", (S, S), up + (255,))
    d = ImageDraw.Draw(im)
    d.polygon([(0, S * 0.62), (S * 0.45, S * 0.62), (S * 0.78, S * 0.30), (S, S * 0.30), (S, S), (0, S)], fill=lo + (255,))
    for off, col in ((-34, None), (-60, "holo")):
        pts = [(0, S * 0.62 + off), (S * 0.45 + off * 0.4, S * 0.62 + off), (S * 0.78 + off * 0.4, S * 0.30 + off), (S, S * 0.30 + off)]
        if col == "holo":
            d.line(pts, fill=A.MAG_L + (255,) if not inv else (200, 170, 255, 255), width=8)
        else:
            d.line(pts, fill=(235, 235, 245, 255), width=26)
    return im.resize((185, 185), Image.LANCZOS).convert("RGB")


def save_dxt5(im, path):
    im.convert("RGBA").save(path, pixel_format="DXT5")
    with open(path, "r+b") as fh:
        fh.seek(20)
        fh.write(struct.pack("<I", (im.width // 4) * (im.height // 4) * 16))
        fh.seek(88)
        fh.write(struct.pack("<I", 0))


if __name__ == "__main__":
    names = sys.argv[1:] or list(DRIVERS)
    lines = []
    for nm in names:
        car = build(nm)
        img = car.image()
        out = os.path.join(HERE, nm)
        os.makedirs(out, exist_ok=True)
        save_dxt5(img, os.path.join(out, "Skin.dds"))
        img.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, f"texture_preview_{nm}.png"))
        gl, grep_ = glass(car, DRIVERS[nm], DRIVERS[nm]["invert"])
        car.report += grep_
        save_dxt5(gl, os.path.join(out, "glass_sticker.dds"))
        gl.save(os.path.join(HERE, f"glass_preview_{nm}.png"))
        drv = DRIVERS[nm]
        icon = livery_icon(drv["invert"])
        icon.save(os.path.join(out, "livery.png"))
        with open(os.path.join(out, "ui_skin.json"), "w", encoding="utf-8") as fh:
            json.dump({"skinname": nm, "drivername": drv["full"], "country": "Russia", "team": "Команда ЭДМ",
                       "number": "00", "priority": 1}, fh, ensure_ascii=False, indent=2)
        with zipfile.ZipFile(os.path.join(HERE, nm + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
            for f in ("Skin.dds", "glass_sticker.dds", "livery.png", "ui_skin.json"):
                z.write(os.path.join(out, f), nm + "/" + f)
        if nm == "Pozdnyakov_00":
            img.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, "texture_preview.png"))
        lines.append(f"== {nm}")
        for r in car.report:
            lines.append(("OK   " if r["ok"] else "FAIL ") + json.dumps(r, ensure_ascii=False))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(HERE, "check_report.txt"), "w").write(txt + "\n")
