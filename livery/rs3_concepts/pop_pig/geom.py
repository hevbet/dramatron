"""Геометрия для «Pop Pig»: карта текстель → мир (4096, float32), проекция арта из вида, поля и проверки.

Мир (модель kn5): x = ЛЕВЫЙ борт, y = назад, z = вверх. Плотность текстуры ≈ 795 px/м (равномерная).
Кэш карты (≈350 МБ) лежит в scratchpad, строится один раз за ~1–2 мин из zones/_analysis.npz.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

ZONES = "/home/user/dramatron/livery/rs3_concepts/zones"
CACHE = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad/pp/posmap4096.npz"
N = 4096
DENS = 795.0                      # px текстуры на метр
sys.path.insert(0, ZONES)

PART_NAMES = json.load(open(os.path.join(ZONES, "zones.json")))["part_names"]


def _build(res=N, pad_px=10):
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


class Geo:
    def __init__(self):
        if not os.path.exists(CACHE):
            print("строю карту текстель→мир 4096 (один раз)…")
            _build()
        d = np.load(CACHE)
        self.pos = d["pos"]
        nrm = d["nrm"].astype(np.float32)
        nl = np.linalg.norm(nrm, axis=-1, keepdims=True)
        self.nrm = nrm / np.maximum(nl, 1e-6)
        self.part = d["part"]
        self.covered = d["covered"]
        self.valid = self.part >= 0
        self.x, self.y, self.z = (self.pos[..., i] for i in range(3))
        self.pid = {n: i for i, n in enumerate(PART_NAMES)}

    def parts_mask(self, names):
        ids = [self.pid[n] for n in names]
        return np.isin(self.part, ids)

    # ---------------------------------------------------------- поля
    def signed_dist(self, f):
        """f — скалярное поле в метрах на текстуре. Возвращает знаковое расстояние до f=0 в МЕТРАХ по поверхности
        (через градиент в пространстве текстуры: на швах берём меньшую из односторонних разностей)."""
        f = f.astype(np.float32)
        def g(axis):
            fw = np.abs(np.diff(f, axis=axis, append=np.take(f, [-1], axis=axis)))
            bw = np.abs(np.diff(f, axis=axis, prepend=np.take(f, [0], axis=axis)))
            return np.minimum(fw, bw)
        gr = np.hypot(g(0), g(1)) * DENS             # |∇f| на метр поверхности
        return f / np.maximum(gr, 1e-4)

    # ---------------------------------------------------------- проекция
    VIEWS = {
        "left": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        "right": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
        "front": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        "rear": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        "top": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "top_rear": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        "top_left": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        "top_right": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    }

    def _flat(self):
        if not hasattr(self, "_V"):
            self._V = np.flatnonzero(self.valid.ravel())
            self._P = self.pos.reshape(-1, 3)[self._V]
            self._Nn = self.nrm.reshape(-1, 3)[self._V]
            self._pt = self.part.ravel()[self._V]
        return self._V, self._P, self._Nn, self._pt

    def project(self, art, view, center, width_m, max_angle=60, parts=None, side=None):
        """art (PIL RGBA, нарисован как его видит зритель) → (индексы текселей, rgba float (K,4), отчёт проверки).
        side: 'L' / 'R' — ограничить бортом (x>0 / x<0)."""
        V, P, Nn, pt = self._flat()
        r, u, d = (np.array(v, np.float32) for v in self.VIEWS[view])
        a = np.asarray(art.convert("RGBA")).astype(np.float32) / 255.0
        H, W = a.shape[:2]
        ppm = W / width_m
        height_m = H / ppm
        c = np.asarray(center, np.float32)
        rel = P - c
        px = ((rel @ r) / width_m + 0.5) * W - 0.5
        py = (0.5 - (rel @ u) / height_m) * H - 0.5
        ok = (Nn @ d) > np.cos(np.radians(max_angle))
        if parts is not None:
            ok &= np.isin(pt, [self.pid[n] for n in parts])
        if side == "L":
            ok &= P[:, 0] > 0.05
        elif side == "R":
            ok &= P[:, 0] < -0.05
        ok &= (rel @ d) > -0.6          # только ближняя к зрителю сторона
        # z-буфер (ячейка ≈ 1 см): берём только поверхность, ближайшую к зрителю
        dep = rel @ d
        cs = max(1, int(ppm * 0.01))
        M0 = int(0.06 * ppm)
        gx = np.floor((px + M0) / cs).astype(np.int64)
        gy = np.floor((py + M0) / cs).astype(np.int64)
        GW, GH = (W + 2 * M0) // cs + 2, (H + 2 * M0) // cs + 2
        inz = ok & (gx >= 0) & (gx < GW) & (gy >= 0) & (gy < GH)
        zb = np.full(GW * GH, -1e9, np.float32)
        cell = gy[inz] * GW + gx[inz]
        np.maximum.at(zb, cell, dep[inz])
        front = np.zeros_like(ok)
        front[np.flatnonzero(inz)] = dep[inz] >= zb[cell] - 0.04
        ok = ok & (front | ~inz)
        inb = ok & (px > -1) & (px < W) & (py > -1) & (py < H)
        idx = V[inb]
        pre = a.copy()
        pre[..., :3] *= pre[..., 3:4]
        coords = np.stack([py[inb], px[inb]])
        vals = [ndimage.map_coordinates(pre[..., ch], coords, order=1, mode="constant", cval=0.0) for ch in range(4)]
        al = vals[3]
        rgb = np.stack(vals[:3], -1) / np.maximum(al[:, None], 1e-6)
        rgba = np.concatenate([rgb, al[:, None]], 1)
        keep = al > 0.002
        idx, rgba = idx[keep], rgba[keep]
        # ---- проверка: принимающая поверхность в пространстве арта
        M = int(0.06 * ppm)                       # поле вокруг арта: проверяем реальный край панели
        R = np.zeros((H + 2 * M, W + 2 * M), bool)
        ix = np.round(px[ok]).astype(np.int64) + M
        iy = np.round(py[ok]).astype(np.int64) + M
        m = (ix >= 0) & (ix < W + 2 * M) & (iy >= 0) & (iy < H + 2 * M)
        R[iy[m], ix[m]] = True
        k = max(1, int(np.ceil(ppm / DENS)) + 1)
        R = ndimage.binary_closing(R, iterations=k)
        Rfull = R
        R = R[M:M + H, M:M + W]
        A = a[..., 3] > 0.25
        if A.any():
            dt = ndimage.distance_transform_edt(np.pad(Rfull, 1, constant_values=False))[1 + M:1 + M + H, 1 + M:1 + M + W]
            clear_cm = float(dt[A].min()) / ppm * 100
            inside = float((A & R).sum()) / float(A.sum())
        else:
            clear_cm, inside = 0.0, 0.0
        tp = self.part.ravel()[idx[rgba[:, 3] > 0.25]]
        touched = sorted({PART_NAMES[p] for p in np.unique(tp) if p >= 0})
        return idx, rgba, dict(view=view, inside=inside, clear_cm=clear_cm, parts=touched, ppm=ppm,
                               size_cm=(round(width_m * 100, 1), round(height_m * 100, 1)))


def rasterize_glass_masks(kn5_path, res=1024):
    """Точные UV-маски glass_sticker / ext_sticker (L, 255 = есть геометрия)."""
    sys.path.insert(0, "/home/user/dramatron/livery/tools")
    import kn5
    from PIL import ImageDraw
    m = kn5.load(kn5_path)
    mats = m["materials"]
    out = {}
    for me in m["meshes"]:
        mn = mats[me["mat"]]["name"] if isinstance(me["mat"], int) else me["mat"]
        if mn not in ("glass_sticker", "ext_sticker"):
            continue
        im = Image.new("L", (res, res), 0)
        d = ImageDraw.Draw(im)
        uv = me["uv"]
        tx = (uv[:, 0] % 1) * res
        ty = (uv[:, 1] % 1) * res
        for t in me["idx"]:
            d.polygon([(tx[i], ty[i]) for i in t], fill=255)
        out[me["name"].split("/")[-1]] = im
    return out


def _recv_key(view, parts, side, max_angle):
    return (view, tuple(parts) if parts else None, side, max_angle)


class Fitter:
    """Быстрый подбор места: карта «принимающей поверхности» в плоскости вида (res px/м) + расстояние до края."""

    def __init__(self, G, res=500):
        self.G, self.res, self.cache = G, res, {}

    def recv(self, view, parts, side, max_angle):
        key = _recv_key(view, parts, side, max_angle)
        if key in self.cache:
            return self.cache[key]
        G = self.G
        V, P, Nn, pt = G._flat()
        r, u, d = (np.array(v, np.float32) for v in G.VIEWS[view])
        ok = (Nn @ d) > np.cos(np.radians(max_angle))
        if parts is not None:
            ok &= np.isin(pt, [G.pid[n] for n in parts])
        if side == "L":
            ok &= P[:, 0] > 0.05
        elif side == "R":
            ok &= P[:, 0] < -0.05
        sa, sb = P[ok] @ r, P[ok] @ u
        a0, b1 = sa.min() - 0.1, sb.max() + 0.1
        W = int((sa.max() + 0.1 - a0) * self.res) + 1
        H = int((b1 - (sb.min() - 0.1)) * self.res) + 1
        R = np.zeros((H, W), bool)
        R[((b1 - sb) * self.res).astype(int), ((sa - a0) * self.res).astype(int)] = True
        R = ndimage.binary_closing(R, iterations=2)
        dt = ndimage.distance_transform_edt(R) / self.res        # м до края
        self.cache[key] = (dt, a0, b1)
        return self.cache[key]

    def fit(self, art, view, center, height_m, parts, side, max_angle, min_clear_m, search_m=(0.06, 0.04),
            shrink_to=0.75):
        """→ (center, height_m, clearance_m). Ищет ближайший к center вариант с запасом ≥ min_clear (+3 мм),
        при необходимости уменьшает элемент (не меньше shrink_to от исходного)."""
        r, u, d = (np.array(v, np.float32) for v in self.G.VIEWS[view])
        c = np.asarray(center, np.float32)
        ca, cb = float(c @ r), float(c @ u)
        dt, a0, b1 = self.recv(view, parts, side, max_angle)
        al = np.asarray(art.getchannel("A")) > 64
        k = 1.0
        best_any = None
        while k >= shrink_to - 1e-6:
            h = height_m * k
            w = h * art.width / art.height
            ys, xs = np.nonzero(al)
            sel = np.linspace(0, len(xs) - 1, min(len(xs), 4000)).astype(int)
            pa = (xs[sel] / art.width - 0.5) * w
            pb = (0.5 - ys[sel] / art.height) * h
            cands = []
            for da in np.arange(-search_m[0], search_m[0] + 1e-9, 0.01):
                for db in np.arange(-search_m[1], search_m[1] + 1e-9, 0.01):
                    A = ((ca + da + pa - a0) * self.res).astype(int)
                    B = ((b1 - (cb + db + pb)) * self.res).astype(int)
                    if A.min() < 0 or B.min() < 0 or A.max() >= dt.shape[1] or B.max() >= dt.shape[0]:
                        continue
                    cl = dt[B, A].min()
                    cands.append((cl, math_hypot(da, db), da, db))
            good = [cc for cc in cands if cc[0] >= min_clear_m + 0.007]
            if good:
                cl, _, da, db = min(good, key=lambda t: t[1])
                newc = c + r * da + u * db
                return tuple(newc), h, cl
            if cands:
                b = max(cands)
                if best_any is None or b[0] > best_any[0]:
                    best_any = (b[0], c + r * b[2] + u * b[3], h)
            k -= 0.05
        return tuple(best_any[1]), best_any[2], best_any[0]


def math_hypot(a, b):
    return (a * a + b * b) ** 0.5
