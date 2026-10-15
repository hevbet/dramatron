"""BR03 (smp_br03_v2024.kn5) support for the RS3 renderers, without touching them.

The renderers key everything on material names of the Audi RS3 kn5 (skin, glass, ture, ...).  This module
renames the BR03 materials when its kn5 is loaded: the painted body «body» becomes «skin» (pearl paint,
clearcoat, holo detection, skin-texture override = body_paint.dds), every other material gets the prefix
«b3:» so it can never collide with an RS3 material, and registers the prefixed names in the material
tables of render_rs3 / team_render (textured with the textures embedded in the kn5, window glass as glass).
RS3 materials and RS3 output are unaffected: no RS3 name starts with «b3:».

Why the plain render_rs3.py --kn5 <br03> crashed: the BR03 has no material called glass / lights_glass, so
the glass pass gets an empty triangle list and `gid_gl[np.maximum(tidg, 0)]` indexes an empty array
(IndexError: index 0 is out of bounds for axis 0 with size 0).  With the renaming the BR03 windows are glass.
"""
import os

BR03_KN5 = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad/br03src/smp_br03_v2024.kn5"
P = "b3:"
BODY = "body"
GLASS = {"EXT_Window_Alpha": ((18, 20, 24), 0.55)}
SKIP = {"damage_glass", "int_glass"}
AT = {"lod_grill"}
# team_render material parameters borrowed from the RS3 material of the same kind
LIKE = {"tire": "ture", "rim_br03": "rim", "rim_details": "rim", "br_disk": "brakedisc", "chrom": "chrome",
        "mirror": "mirror", "lamps": "lights", "carbone": "ext_carbon", "carbon_sidepods": "ext_carbon",
        "plasti_mate": "ext_plastic", "nads": "pl_black", "dno": "ext_carbon", "Black": "black",
        "ext_details": "ext_plastic", "ext": "ext_metal", "front_susp": "ext_metal", "rear_details": "ext_metal",
        "Material #25": "screw", "front_hub": "screw", "brake_duct": "ext_plastic", "lod_grill": "reshotka"}
# inner panel skins that sit (almost) on the body: AC hides them by back-face culling, the renderers do not
# cull, so they lose depth ties instead (pushed back 4 mm) - otherwise they speckle through the paint
BIAS = {"plasti_mate": -0.004}
SPEC = {"chrom": 0.6, "lamps": 0.5, "rim_br03": 0.35, "body": 0.35, "mirror": 0.6}


def is_br03(path):
    return os.path.basename(str(path)).lower().startswith("smp_br03")


def new_name(n):
    return "skin" if n == BODY else P + n


def _rename(mats, meshes):
    for m in mats:
        if m["name"] == "rim_br03":
            # car_paint_rims.dds is a flat 4x4 colour: its x10 tiling is a constant detail colour
            m["props"] = dict(m.get("props", {}), detailUVMultiplier=0.0)
        m["name"] = new_name(m["name"])
    for m in meshes:
        m["mat"] = new_name(m["mat"])
    return mats, meshes


def all_names():
    from kn5 import load
    return [new_name(m["name"]) for m in load(BR03_KN5)["materials"]]


def patch_render_rs3(rr, names=None):
    """Material tables + a renaming load_scene in render_rs3 (idempotent)."""
    if getattr(rr, "_br03_patched", False):
        return
    rr._br03_patched = True
    names = names or all_names()
    for n in names:
        if not n.startswith(P):
            continue
        base = n[len(P):]
        if base in SKIP:
            rr.SKIP_MATS.add(n)
        elif base in GLASS:
            rr.GLASS_MATS[n] = GLASS[base]
        else:
            rr.TEX_OK.add(n)
        if base in AT:
            rr.AT_MATS.add(n)
        if base in SPEC:
            rr.SPEC[n] = SPEC[base]
        if base in BIAS:
            rr.DEPTH_BIAS[n] = BIAS[base]
        if base in LIKE and LIKE[base] in rr.FLAT:
            rr.FLAT[n] = rr.FLAT[LIKE[base]]
    orig = rr.load_scene

    def load_scene(path, interior=True):
        tex, mats, meshes = orig(path, interior)
        if is_br03(path):
            mats, meshes = _rename(mats, meshes)
        return tex, mats, meshes
    rr.load_scene = load_scene


def patch_team_render(tr, names=None):
    """patch_render_rs3 + team_render's material parameters and renaming load_geometry."""
    if getattr(tr, "_br03_patched", False):
        return
    tr._br03_patched = True
    names = names or all_names()
    patch_render_rs3(tr.rr, names)
    for n in names:
        if n.startswith(P) and n[len(P):] in LIKE:
            tr.MATS[n] = dict(tr.MATS.get(LIKE[n[len(P):]], {}))
    orig = tr.load_geometry

    def load_geometry(path, interior=True):
        tex, mats, meshes = orig(path, interior)
        if is_br03(path):
            mats, meshes = _rename(mats, meshes)
        return tex, mats, meshes
    tr.load_geometry = load_geometry
