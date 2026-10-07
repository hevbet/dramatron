"""Adds world coordinates of every safe box + a curated 'spots' list (sponsor / number / name placements)
to zones.json. Run after build_zones.py and posmap.py."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "zones.json")))
pm = np.load(os.path.join(HERE, "posmap.npz"))
pos, cov = pm["pos"].astype(np.float32), pm["covered"]
k = pos.shape[0] / 4096
Z = {z["id"]: z for z in J["zones"]}
G = {g["kind"]: g for g in J["glass_sticker_zones"]}


def world_of_box(b):
    x0, y0, x1, y1 = [int(v * k) for v in b]
    sub, c = pos[y0:y1, x0:x1], cov[y0:y1, x0:x1]
    if not c.any():
        return None
    p = sub[c]
    return dict(center=[round(float(v), 3) for v in p.mean(0)], min=[round(float(v), 3) for v in p.min(0)],
                max=[round(float(v), 3) for v in p.max(0)])


for z in J["zones"]:
    if z["safe_box"]:
        z["safe_box_world"] = world_of_box(z["safe_box"])

AUDI_RINGS = [1950, 720, 2140, 790]
J["keep_clear"] = [dict(what="Audi rings on the trunk (baked, keep)", box=AUDI_RINGS),
                   dict(what="front-door / rear-door shut line runs at world y~0.28 (texture: between rows 7 and 8 on each side)",
                        box=None)]


def sz(b):
    return [b[2] - b[0], b[3] - b[1]]


def spot(name, use, src, box, rot, notes="", tex="Skin.dds", world=None, project=None):
    d = dict(spot=name, use=use, texture=tex, zone=src, box=box, box_px=sz(box) if box else None, text_rotation=rot,
             notes=notes)
    if world:
        d["world"] = world
    if project:
        d["projector"] = project
    return d


S = []
for s, sign in (("L", 1), ("R", -1)):
    fd = Z[f"front_door_{s}"]
    S.append(spot(f"door_number_{s}", "race number (main) / number plate", fd["id"], fd["safe_box"], fd["text_rotation"],
                  f"~{fd['safe_box_size_cm'][1]}x{fd['safe_box_size_cm'][0]} cm (long x tall) on the flat front door",
                  world=fd.get("safe_box_world")))
    rd = Z[f"rear_door_{s}"]
    S.append(spot(f"rear_door_logo_{s}", "main sponsor logo (2nd door)", rd["id"], rd["safe_box"], rd["text_rotation"],
                  "rear door is split in 2 UV islands (diagonal seam toward the rear arch); box stays in the upper "
                  "island. For a logo bigger than the box use projector.project(view='%s')" % ("left" if s == "L" else "right"),
                  world=rd.get("safe_box_world")))
    S.append(spot(f"side_band_{s}", "full-length sponsor band across both doors (crosses UV seams)", "front_door+rear_door",
                  None, None, "draw the band as seen from the side and project; y from -0.84 (front door front edge) "
                              "to 0.95 (rear arch), z 0.42-0.82 is flat (cos>0.9) except the flare step at y~0.85",
                  project=dict(view="left" if s == "L" else "right", center=[0.9 * sign, 0.05, 0.62], width_m=1.75,
                               max_angle=80)))
    sl = Z[f"sill_{s}"]
    S.append(spot(f"sill_{s}", "website / series / long thin wordmark", sl["id"], sl["safe_box"], sl["text_rotation"],
                  f"thin strip ~{sl['safe_box_size_cm'][1]}x{sl['safe_box_size_cm'][0]} cm", world=sl.get("safe_box_world")))
    fl = Z[f"front_door_low_{s}"]
    S.append(spot(f"door_low_strip_{s}", "small sponsor row under the number", fl["id"], fl["safe_box"], fl["text_rotation"],
                  "separate UV island below the door crease (z<0.42)", world=fl.get("safe_box_world")))
    rf = Z[f"rear_fender_{s}"]
    S.append(spot(f"rear_fender_{s}", "small logo above the rear arch / behind the rear wheel", rf["id"], rf["safe_box"],
                  rf["text_rotation"], "curved; keep logos < 30 cm", world=rf.get("safe_box_world")))
    ft = Z[f"front_fender_top_{s}"]
    S.append(spot(f"front_fender_top_{s}", "flag / small logo on top of front wing (seen from front & top)", ft["id"],
                  ft["safe_box"], ft["text_rotation"], "", world=ft.get("safe_box_world")))
    S.append(spot(f"front_fender_side_{s}", "driver flag / small sponsor above the front arch", f"front_fender_{s}", None,
                  None, "front fender is split in 3 UV islands with rotated layouts -> ONLY via projector",
                  project=dict(view="left" if s == "L" else "right", center=[0.9 * sign, -1.15, 0.78], width_m=0.35,
                               max_angle=60)))
    S.append(spot(f"side_window_{s}", "driver name + flag (rear side window sticker)", "glass_sticker:side_window_" +
                  ("left" if s == "L" else "right"), G["side_window_" + ("left" if s == "L" else "right")]["bbox"],
                  G["side_window_" + ("left" if s == "L" else "right")]["text_rotation"],
                  "glass_sticker.dds 1024; keep alpha=0 where glass must stay clear", tex="glass_sticker.dds"))
for zid, use, note in (("hood", "main sponsor (hood)", "vent hole cuts the hood at the front-centre; box is the flat area in front of the windscreen"),
                       ("roof", "roof number (TV)", "default reads from the front; text_rotation_alt gives read-from-left (-90) and read-from-behind (180)"),
                       ("trunk_lid", "sponsor on trunk lid (read from behind)", "keep clear of Audi rings box (keep_clear)"),
                       ("rear_panel", "small sponsor between the tail lights", ""),
                       ("rear_bumper", "website / sponsor on rear bumper", ""),
                       ("nose", "thin strip above the grille (front lip)", "only ~3 cm tall in texture -> tiny text only")):
    z = Z[zid]
    S.append(spot(zid, use, zid, z["safe_box"], z["text_rotation"], note, world=z.get("safe_box_world")))
for kind, use in (("windscreen_banner", "windscreen banner (series / main sponsor)"),
                  ("wing_top", "rear-wing top (read from behind/above)"),
                  ("wing_endplate_left", "left endplate logo"), ("wing_endplate_right", "right endplate logo"),
                  ("rear_window", "rear window top banner (read from behind)")):
    g = G[kind]
    S.append(spot(kind, use, g["id"], g["bbox"], g["text_rotation"], g["how"], tex="glass_sticker.dds"))
J["spots"] = S
json.dump(J, open(os.path.join(HERE, "zones.json"), "w"), ensure_ascii=False, indent=1)
for s in S:
    print(f"{s['spot']:26s} {s['texture']:18s} box={s['box']} rot={s['text_rotation']} {s.get('world', {}) and s['world']['center']}")
