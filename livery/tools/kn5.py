"""Минимальный парсер .kn5 (Assetto Corsa): текстуры, материалы, меши с UV.

Использование как модуль: m = load(path) → dict(textures, materials, meshes).
"""
import struct
import sys

import numpy as np


class R:
    def __init__(self, b):
        self.b, self.p = b, 0

    def i(self):
        v = struct.unpack_from("<i", self.b, self.p)[0]; self.p += 4; return v

    def u8(self):
        v = self.b[self.p]; self.p += 1; return v

    def f(self, n=1):
        v = struct.unpack_from(f"<{n}f", self.b, self.p); self.p += 4 * n; return v if n > 1 else v[0]

    def s(self):
        n = self.i(); v = self.b[self.p:self.p + n].decode("utf-8", "replace"); self.p += n; return v

    def raw(self, n):
        v = self.b[self.p:self.p + n]; self.p += n; return v


def load(path, keep_tex=False):
    r = R(open(path, "rb").read())
    assert r.raw(6) == b"sc6969"
    ver = r.i()
    if ver > 5:
        r.i()
    tex = []
    for _ in range(r.i()):
        r.i(); name = r.s(); size = r.i(); data = r.raw(size)
        tex.append((name, data if keep_tex else size))
    mats = []
    for _ in range(r.i()):
        name = r.s(); shader = r.s(); r.u8(); r.u8()
        if ver > 4:
            r.i()
        props = {}
        for _ in range(r.i()):
            pn = r.s(); pv = r.f(); r.raw(36); props[pn] = pv
        samplers = {}
        for _ in range(r.i()):
            sn = r.s(); r.i(); samplers[sn] = r.s()
        mats.append(dict(name=name, shader=shader, props=props, samplers=samplers))
    meshes = []

    def node(path_):
        cls = r.i(); name = r.s(); nch = r.i(); r.u8()
        full = path_ + "/" + name
        if cls == 1:
            r.raw(64)
        elif cls == 2:
            r.u8(); r.u8(); r.u8()
            nv = r.i(); v = np.frombuffer(r.raw(nv * 44), dtype="<f4").reshape(nv, 11)
            ni = r.i(); idx = np.frombuffer(r.raw(ni * 2), dtype="<u2")
            mid = r.i(); r.i(); r.f(2); r.f(3); r.f(); r.u8()
            meshes.append(dict(name=full, pos=v[:, 0:3], uv=v[:, 6:8], idx=idx.reshape(-1, 3), mat=mid))
        elif cls == 3:
            r.u8(); r.u8(); r.u8()
            for _ in range(r.i()):
                r.s(); r.raw(64)
            nv = r.i(); v = np.frombuffer(r.raw(nv * 76), dtype="<f4").reshape(nv, 19)
            ni = r.i(); idx = np.frombuffer(r.raw(ni * 2), dtype="<u2")
            mid = r.i(); r.i(); r.raw(8)
            meshes.append(dict(name=full, pos=v[:, 0:3], uv=v[:, 6:8], idx=idx.reshape(-1, 3), mat=mid))
        for _ in range(nch):
            node(full)

    node("")
    return dict(version=ver, textures=tex, materials=mats, meshes=meshes)


if __name__ == "__main__":
    m = load(sys.argv[1])
    print("version", m["version"], "textures", len(m["textures"]), "materials", len(m["materials"]), "meshes", len(m["meshes"]))
