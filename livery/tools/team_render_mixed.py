#!/usr/bin/env python3
"""team_render.py studio with different cars in one scene: every car has its own kn5, skin folder and
placement (e.g. two SMP BR03 in front, two Audi RS3 LMS behind).  team_render.py itself is not modified.

    python3 team_render_mixed.py --out DIR \
        --car BR03_KN5,skin_dir,x,z,yaw --car RS3_KN5,skin_dir,x,z,yaw ... \
        [--az 0 --eye-h 1.5 --dist 14 --ty 0.55 --margins l,r,t,b] [--width 2400 --height 800] [--ss 2] \
        [--jobs 2] [--fast] [--name team4] [--back-ev EV]

A kn5 path may be the word «br03» or «rs3» (the default kn5 of each car).  x, z in metres (x = left, z =
front), yaw in degrees (+ turns the nose towards +x); camera az in degrees from +z towards +x, eye-h camera
height, dist horizontal distance from the cars' centre, ty target height.  Writes DIR/<name>.png.

How: every kn5 is loaded as a team_render.Model (the BR03 through br03_cfg: «body» -> «skin», other
materials prefixed «b3:»), the cars are placed into ONE combined model with a union material table, and that
model goes through team_render.Layout with a single identity instance; the per-triangle car index is then
set to the real car so every car keeps its own livery, holo phase and title data.  All the shading, studio,
floor and post code is team_render's own.
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import br03_cfg  # noqa: E402
import team_render as tr  # noqa: E402

br03_cfg.patch_team_render(tr)
KN5 = {"br03": br03_cfg.BR03_KN5, "rs3": tr.rr.DEFAULT_KN5}


def combine(models, placements):
    """One placed model (floor_y 0, identity instance) from per-car (model, x, z, yaw) + union materials."""
    names = []
    for m in models:
        for n in m.mat_names:
            if n not in names:
                names.append(n)
    cm = SimpleNamespace()
    P, VN, FN, UV, T, TM, EM, OK, UA, LENS, EXT, CAR = [], [], [], [], [], [], [], [], [], [], [], []
    off = 0
    remaps = []
    for i, (m, (x, z, yaw)) in enumerate(zip(models, placements)):
        R = tr.yaw_matrix(yaw)
        remap = np.array([names.index(n) for n in m.mat_names], np.int32)
        remaps.append(remap)
        P.append(m.P @ R + np.array([x, -m.floor_y, z]))
        VN.append((m.VN @ R.astype(np.float32)).astype(np.float32))
        FN.append((m.FN @ R.astype(np.float32)).astype(np.float32))
        UV.append(m.UV); T.append(m.T + off); TM.append(remap[m.TM]); EM.append(m.EMIS)
        OK.append(m.tri_ok); UA.append(m.uv_area); LENS.append(m.lens); EXT.append(m.ext_vertices + off)
        CAR.append(np.full(len(m.T), i, np.int32))
        off += len(m.P)
    cm.P = np.concatenate(P); cm.VN = np.concatenate(VN); cm.FN = np.concatenate(FN); cm.UV = np.concatenate(UV)
    cm.T = np.concatenate(T); cm.TM = np.concatenate(TM); cm.EMIS = np.concatenate(EM)
    cm.tri_ok = np.concatenate(OK); cm.uv_area = np.concatenate(UA); cm.lens = np.concatenate(LENS)
    cm.ext_vertices = np.concatenate(EXT); cm.car = np.concatenate(CAR)
    cm.floor_y = 0.0
    cm.path, cm.interior = models[0].path, models[0].interior
    cm.mat_names = names
    cm.is_glass = np.array([n in tr.rr.GLASS_MATS for n in names])
    cm.skin_id = names.index("skin")
    cm.emit_rgb = models[0].emit_rgb
    prm = {k: [] for k in tr.MAT_DEFAULT}
    for n in names:
        d = dict(tr.MAT_DEFAULT); d.update(tr.MATS.get(n, {}))
        for k in prm:
            prm[k].append(d[k])
    cm.prm = {k: np.array(v, np.float32) for k, v in prm.items()}
    return cm, remaps


def union_livery(lv, remap, n):
    """Re-index a Livery built on its own model to the union material table."""
    kind, pyr, flat = ["flat"] * n, [None] * n, [tr.srgb_to_lin((40, 40, 42))] * n
    for j, u in enumerate(remap):
        kind[u], pyr[u], flat[u] = lv.kind[j], lv.pyr[j], lv.flat[j]
    lv.kind, lv.pyr, lv.flat = kind, pyr, flat
    return lv


class MixedFactory(tr.Factory):
    def __init__(self, model, liveries, layout, log):
        super().__init__(model, liveries, 1, None, False, log)
        self.L = layout

    def layout(self, cars):
        return self.L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--car", action="append", required=True, metavar="kn5,skin,x,z,yaw")
    ap.add_argument("--name", default="team4")
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--eye-h", type=float, default=1.5)
    ap.add_argument("--dist", type=float, default=14.0)
    ap.add_argument("--ty", type=float, default=0.55)
    ap.add_argument("--tx", type=float, default=None, help="target x (default: centre of the cars)")
    ap.add_argument("--tz", type=float, default=None, help="target z (default: centre of the cars)")
    ap.add_argument("--margins", default="0.08,0.08,0.40,0.12")
    ap.add_argument("--width", type=int, default=2400); ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--back-ev", type=float, default=0.0, help="exposure (EV) of the cars behind (index >= 2)")
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--refl", type=float, default=0.55)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cars = []
    for spec in a.car:
        k, skin, x, z, yaw = spec.split(",")
        cars.append((KN5.get(k.lower(), k), skin, float(x), float(z), float(yaw)))
    models = {}
    for k, *_ in cars:
        if k not in models:
            models[k] = tr.Model(k, interior=True)
            print(f"model {os.path.basename(k)}: {len(models[k].T)} tris, {time.time() - t0:.0f}s", flush=True)
    sub = [models[c[0]] for c in cars]
    place = [(c[2], c[3], c[4]) for c in cars]
    cm, remaps = combine(sub, place)
    liveries = []
    for (k, skin, *_), m, remap in zip(cars, sub, remaps):
        lv = tr.Livery(m, skin, ks_detail=True)
        print(f"  {lv.info['folder']}: #{lv.info.get('number')}; overrides: "
              + ", ".join(sorted(os.path.basename(p) for p in lv.overrides.values())), flush=True)
        liveries.append(union_livery(lv, remap, len(cm.mat_names)))
    L = tr.Layout(cm, [(0.0, 0.0, 0.0)], jobs=a.jobs, log=lambda s: print(s, flush=True), fast=a.fast)
    L.CAR = cm.car.copy()
    L.n = len(cars)
    L.cars = place
    L.car_centres = [np.array([x, 0.62, z]) for x, z, _ in place]
    tr.PRESETS[a.name] = dict(az=a.az, eye_h=a.eye_h, dist=a.dist, ty=a.ty, cars=tuple(place),
                              margins=tuple(float(v) for v in a.margins.split(",")), dof=0.0, back_ev=0.0,
                              exposure=1.0, floor_alb=0.16, refl=a.refl)
    tr.CAPTIONS[a.name] = "Команда"
    if a.tx is not None or a.tz is not None:
        cx = float(np.mean([p[0] for p in place])); cz = float(np.mean([p[1] for p in place]))
        dx = (a.tx if a.tx is not None else cx) - cx; dz = (a.tz if a.tz is not None else cz) - cz
        # view_setup aims at the mean of the cars: shift the target through a phantom offset of all cars
        tr.PRESETS[a.name]["cars"] = tuple((x + dx, z + dz, y) for x, z, y in place)
    args = SimpleNamespace(width=a.width, height=a.height, ss=a.ss, gap=None, stagger=None, dof=None,
                           back_ev=None, zoom=a.zoom, exposure=a.exposure, title=False, camera=None, eye=None,
                           target=None)
    if a.back_ev:
        orig = tr.ShadeCtx.__init__

        def init(self, model, livs, LL, studio, car_gain=None, lod_bias=tr.LOD_BIAS):
            g = np.ones(LL.n, np.float32)
            g[2:] = 2.0 ** a.back_ev
            orig(self, model, livs, LL, studio, g, lod_bias)
        tr.ShadeCtx.__init__ = init
    fac = MixedFactory(cm, liveries, L, lambda s: print(s, flush=True))
    im, raw, bbox = tr.render_view(fac, a.name, args, log=lambda s: print(s, flush=True))
    fp = os.path.join(a.out, a.name + ".png")
    raw.save(fp)
    print(f"bbox (px) {tuple(round(v) for v in bbox)}; done in {time.time() - t0:.0f}s -> {fp}", flush=True)


if __name__ == "__main__":
    main()
