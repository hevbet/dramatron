"""Analytical UV analysis of smp_audi_rs3_lms.kn5 -> per-triangle + per-texel data (cached in .npz).

Frame (raw kn5 vertex coords == Blender world): x = lateral (+x = car LEFT), y = longitudinal (+y = REAR), z = up.
Texture pixel coords: tx = u * N, ty = (1 + v) * N  (v negative), y axis pointing DOWN in the image.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kn5x  # noqa: E402

KN5 = os.environ.get("RS3_KN5") or \
    "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad/kn5/smp_audi_rs3_lms.kn5"
R = 1024  # analysis raster

FRONT_AXLE, REAR_AXLE, WHEEL_Z = -1.314, 1.348, 0.397


def load():
    mats, meshes, nodes = kn5x.load(KN5)
    return meshes


def tri_table(meshes, matnames):
    P, T, N, mid, names = [], [], [], [], []
    for k, m in enumerate(meshes):
        if m["mat"] not in matnames:
            continue
        i = m["idx"]
        P.append(m["pos"][i]); N.append(m["nrm"][i].mean(1))
        uv = m["uv"][i]
        T.append(np.stack([uv[..., 0], 1.0 + uv[..., 1]], -1))
        mid.append(np.full(len(i), len(names))); names.append(m["name"].split("/")[-1] + "|" + m["name"].split("/")[2])
    P = np.concatenate(P); T = np.concatenate(T); N = np.concatenate(N); mid = np.concatenate(mid)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return P, T, N, mid, names


def jacobians(P, T):
    """J (3x2): world displacement per unit texture (tex coords in [0,1], y down)."""
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    d1, d2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    D = np.stack([d1, d2], -1)  # (n,2,2) columns = d1,d2
    det = D[:, 0, 0] * D[:, 1, 1] - D[:, 0, 1] * D[:, 1, 0]
    ok = np.abs(det) > 1e-14
    Dinv = np.zeros_like(D)
    Dinv[ok, 0, 0] = D[ok, 1, 1] / det[ok]; Dinv[ok, 1, 1] = D[ok, 0, 0] / det[ok]
    Dinv[ok, 0, 1] = -D[ok, 0, 1] / det[ok]; Dinv[ok, 1, 0] = -D[ok, 1, 0] / det[ok]
    E = np.stack([e1, e2], -1)  # (n,3,2)
    J = E @ Dinv
    return J, ok


def tex_dir(J, w):
    """texture-space direction (unit, tex y down) whose image on the surface best matches world vector(s) w."""
    pinv = np.linalg.pinv(J)  # (n,2,3)
    d = np.einsum("nij,nj->ni", pinv, w)
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)


def pil_angle(d):
    """PIL Image.rotate(angle) angle that turns an upright text block so its 'up' points along tex dir d."""
    return np.degrees(np.arctan2(-d[:, 0], -d[:, 1]))


def islands(P, T):
    """union-find over triangles sharing a vertex with the same uv AND same position."""
    n = len(P)
    key = np.round(np.concatenate([T.reshape(-1, 2) * 1e5, P.reshape(-1, 3) * 1e4], 1)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(n, 3)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    first = {}
    for t in range(n):
        for v in inv[t]:
            if v in first:
                ra, rb = find(t), find(first[v])
                if ra != rb:
                    parent[ra] = rb
            else:
                first[v] = t
    lab = np.array([find(t) for t in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def raster(T, res, values=None):
    """rasterize triangles in texture space. returns id map (-1 empty), coverage count, barycentric per texel."""
    idm = -np.ones((res, res), np.int32)
    cnt = np.zeros((res, res), np.int16)
    bary = np.zeros((res, res, 3), np.float32)
    Tp = T * res - 0.5  # pixel centers at integer coords
    for t in range(len(T)):
        a, b, c = Tp[t]
        x0 = max(int(np.floor(min(a[0], b[0], c[0]))), 0); x1 = min(int(np.ceil(max(a[0], b[0], c[0]))), res - 1)
        y0 = max(int(np.floor(min(a[1], b[1], c[1]))), 0); y1 = min(int(np.ceil(max(a[1], b[1], c[1]))), res - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / den
        l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / den
        l3 = 1 - l1 - l2
        eps = -1e-6
        m = (l1 >= eps) & (l2 >= eps) & (l3 >= eps)
        if not m.any():
            continue
        yy, xx = ys[m], xs[m]
        idm[yy, xx] = t
        cnt[yy, xx] += 1
        bary[yy, xx] = np.stack([l1[m], l2[m], l3[m]], -1)
    return idm, cnt, bary


# ------------------------------------------------------------------ occlusion: painter's ID render
def visibility(P_skin, all_tris, views, res=1400):
    """all_tris: (M,3,3) occluder triangles (skin first, same order). returns per-view visible pixel counts
    and projected pixel areas for skin triangles."""
    from PIL import Image, ImageDraw
    ns = len(P_skin)
    out_vis, out_proj = [], []
    lo, hi = all_tris.reshape(-1, 3).min(0), all_tris.reshape(-1, 3).max(0)
    ctr = (lo + hi) / 2
    for d in views:
        d = np.asarray(d, float); d /= np.linalg.norm(d)  # direction FROM car TO camera
        up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.95 else np.array([0, -1.0, 0])
        r = np.cross(up, d); r /= np.linalg.norm(r); u = np.cross(d, r)
        V = all_tris - ctr
        sx = V @ r; sy = V @ u; dep = V @ d  # larger dep = closer to camera
        scale = res / 5.2
        X = sx * scale + res / 2; Y = -sy * scale + res / 2
        order = np.argsort(dep.mean(1))  # far first
        im = Image.new("RGB", (res, res), (0, 0, 0)); dr = ImageDraw.Draw(im)
        for t in order:
            idv = t + 1
            col = (idv & 255, (idv >> 8) & 255, (idv >> 16) & 255)
            dr.polygon([(X[t, 0], Y[t, 0]), (X[t, 1], Y[t, 1]), (X[t, 2], Y[t, 2])], fill=col)
        a = np.asarray(im).astype(np.int64)
        ids = a[..., 0] + (a[..., 1] << 8) + (a[..., 2] << 16) - 1
        vis = np.bincount(ids[ids >= 0], minlength=len(all_tris))[:ns]
        e1 = np.stack([X[:ns, 1] - X[:ns, 0], Y[:ns, 1] - Y[:ns, 0]], -1)
        e2 = np.stack([X[:ns, 2] - X[:ns, 0], Y[:ns, 2] - Y[:ns, 0]], -1)
        proj = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        out_vis.append(vis); out_proj.append(proj)
    return np.array(out_vis), np.array(out_proj)


VIEWS = {
    "left": (1, 0, 0), "right": (-1, 0, 0), "front": (0, -1, 0), "rear": (0, 1, 0), "top": (0, 0, 1),
    "left_hi": (1, 0, 0.6), "right_hi": (-1, 0, 0.6), "front_hi": (0, -1, 0.7), "rear_hi": (0, 1, 0.7),
    "fl": (1, -1, 0.35), "fr": (-1, -1, 0.35), "rl": (1, 1, 0.35), "rr": (-1, 1, 0.35),
    "left_lo": (1, 0, -0.25), "right_lo": (-1, 0, -0.25),
}

if __name__ == "__main__":
    meshes = load()
    P, T, N, mid, names = tri_table(meshes, {"skin"})
    print("skin tris", len(P), names)
    J, ok = jacobians(P, T)
    area3 = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    d1, d2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    areaT = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    # singular values of J -> metres per unit tex; density px/m at 4096
    sv = np.linalg.svd(J, compute_uv=False)
    dens = 4096.0 / np.sqrt(np.maximum(sv[:, 0] * sv[:, 1], 1e-18))
    aniso = sv[:, 0] / np.maximum(sv[:, 1], 1e-12)
    # mirror test: text right=(1,0) up=(0,-1) in tex
    Rw = J[:, :, 0]; Uw = -J[:, :, 1]
    mirrored = np.einsum("ij,ij->i", np.cross(Rw, Uw), N) < 0
    lab = islands(P, T)
    print("islands", lab.max() + 1)
    # occluders: all opaque-ish meshes
    occ = []
    for m in meshes:
        nm = m["mat"].lower()
        if m["mat"] == "skin" or "glass" in nm or "sticker" in nm or "air" == nm or "lights_glass" in nm:
            continue
        occ.append(m["wpos"][m["idx"]] @ np.diag([1, 1, 1]))
    # convert occluder world (AC frame) -> blender frame
    occ = np.concatenate(occ)
    occ = np.stack([occ[..., 0], -occ[..., 2], occ[..., 1] + 0.072], -1)
    allt = np.concatenate([P, occ])
    vis, proj = visibility(P, allt, list(VIEWS.values()))
    idm, cnt, bary = raster(T, R)
    np.savez_compressed(os.path.join(HERE, "_analysis.npz"), P=P, T=T, N=N, mid=mid, names=np.array(names), J=J, ok=ok,
                        area3=area3, areaT=areaT, dens=dens, aniso=aniso, mirrored=mirrored, lab=lab, vis=vis,
                        proj=proj, idm=idm, cnt=cnt, bary=bary, views=np.array(list(VIEWS.keys())))
    print("done; overlap texels", int((cnt > 1).sum()), "covered", int((cnt > 0).sum()))
