#!/usr/bin/env python3
"""Preview sheet for the pink wheel nuts -> gear/bolts_preview.png

  1. flat textures: rim_d.dds nut islands and the rim_blur.dds hub, original (kn5) vs new (gear/out/<skin>/)
  2. hub close-ups of all 4 wheels, rendered with tools/render_rs3.py --fit --ks-detail (the in-game x0.886
     detail multiply) from a test skin = copy of butcher_chrome/Konopelko_00 + the new rim_d.dds / rim_blur.dds
  3. front left: original vs new in game, the pixels that changed, the raw texel colour (no detail multiply)
  4. the spinning wheel (rimblur meshes, which render_rs3 normally skips) before / after, and the whole
     front left / rear right wheels in game

Usage: python3 bolts_preview.py [--work DIR] [--skip-render] [--skin Konopelko_00]
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bolts as MB  # noqa: E402

BC = os.path.dirname(HERE)
RENDER = "/home/user/dramatron/livery/tools/render_rs3.py"
WORK = os.path.join(os.path.dirname(os.path.dirname(MB.KN5)), "bolts_pv")     # scratchpad, next to kn5/
FONTS = "/home/user/dramatron/livery/fonts"

BG = (24, 18, 22)
FG = (246, 236, 240)
DIM = (176, 150, 160)
PIG = (242, 158, 178)

HUB = 0.105          # half size of the hub close-up box (m)
WHEELS = {           # name: (view, wheel axis y, z, outer face side)
    "front left": ("side_left", 0.3245, 1.3135, 1),
    "rear left": ("side_left", 0.3245, -1.3475, 1),
    "front right": ("side_right", 0.3245, 1.3135, -1),
    "rear right": ("side_right", 0.3245, -1.3475, -1),
}

BLUR_RUN = """import sys
sys.path.insert(0, %r)
import render_rs3 as r
r.SKIP_MATS.discard("rimblur"); r.SKIP_MATS.add("rim")      # the spinning wheel instead of the static one
r.TEX_OK.add("rimblur"); r.SPEC["rimblur"] = r.SPEC["rim"]
sys.argv = [%r] + sys.argv[1:]
r.main()
""" % (os.path.dirname(RENDER), RENDER)


def hub_box(name):
    _, y, z, s = WHEELS[name]
    x0, x1 = (0.85, 0.98) if s > 0 else (-0.98, -0.85)
    return (x0, y - HUB, z - HUB, x1, y + HUB, z + HUB)


def wheel_box(name):
    _, y, z, s = WHEELS[name]
    x0, x1 = (0.70, 0.98) if s > 0 else (-0.98, -0.70)
    return (x0, 0.0, z - 0.325, x1, 0.65, z + 0.325)


def font(name, size, var=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), int(size))
    if var:
        f.set_variation_by_name(var)
    return f


F_T = lambda s: font("Podkova[wght].ttf", s, "ExtraBold")
F_L = lambda s: font("Exo2-Italic[wght].ttf", s, "SemiBold Italic")


def render_cmd(skin, out, view, w, h, fit, detail, blur=False):
    head = [sys.executable, "-c", BLUR_RUN] if blur else [sys.executable, RENDER]
    cmd = head + [skin, out, "--views", view, "--width", str(w), "--height", str(h), "--jobs", "1",
                  "--no-interior", "--fit=" + ",".join(f"{v:.4f}" for v in fit)]
    if detail:
        cmd.append("--ks-detail")
    return cmd


def run_all(jobs, par=4):
    """jobs: [(key, cmd, png)] -> rendered in parallel (one view per process)."""
    todo = list(jobs)
    while todo:
        batch, todo = todo[:par], todo[par:]
        procs = [(k, subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)) for k, c, _ in batch]
        for k, p in procs:
            _, err = p.communicate()
            if p.returncode:
                raise SystemExit(f"render {k} failed:\n{err.decode()[-2000:]}")
            print(f"rendered {k}", flush=True)


def make_skin_copy(src, dst, files):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for n in (MB.RIM, MB.BLUR):                       # the "before" skin must show the stock rim textures
        p = os.path.join(dst, n)
        if os.path.exists(p):
            os.remove(p)
    for f in files:
        shutil.copy(f, os.path.join(dst, os.path.basename(f)))


def label(d, xy, text, f, fill=FG, bg=(0, 0, 0)):
    x, y = xy
    w = d.textlength(text, font=f)
    d.rectangle([x, y, x + w + 14, y + f.size + 10], fill=bg)
    d.text((x + 7, y + 3), text, font=f, fill=fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--skin", default="Konopelko_00")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "bolts_preview.png"))
    a = ap.parse_args()
    W = a.work
    os.makedirs(W, exist_ok=True)
    new = {n: os.path.join(HERE, "out", a.skin, n) for n in (MB.RIM, MB.BLUR)}
    test, base = os.path.join(W, "test_skin"), os.path.join(W, "base_skin")
    if not a.skip_render:
        make_skin_copy(os.path.join(BC, a.skin), test, new.values())
        make_skin_copy(os.path.join(BC, a.skin), base, [])

    jobs, P = [], {}

    def job(key, skin, view, size, fit, detail, blur=False):
        out = os.path.join(W, "pv_" + key)
        P[key] = os.path.join(out, view + ".png")
        jobs.append((key, render_cmd(skin, out, view, size, size, fit, detail, blur), P[key]))

    for name, (view, *_r) in WHEELS.items():
        k = name.replace(" ", "_")
        job("hub_" + k, test, view, 900, hub_box(name), True)
    job("hub_front_left_orig", base, "side_left", 900, hub_box("front left"), True)
    job("hub_front_left_raw", test, "side_left", 900, hub_box("front left"), False)
    job("blur_orig", base, "side_left", 900, hub_box("front left"), False, True)
    job("blur_new", test, "side_left", 900, hub_box("front left"), False, True)
    job("wheel_front_left", test, "side_left", 900, wheel_box("front left"), True)
    job("wheel_rear_right", test, "side_right", 900, wheel_box("rear right"), True)
    job("wheel_front_left_orig", base, "side_left", 900, wheel_box("front left"), True)
    if not a.skip_render:
        run_all(jobs)
    R = {k: Image.open(p).convert("RGB") for k, p in P.items()}

    # ------------------------------------------------------------ flat textures
    tex, _ = MB.load_textures()
    lv = {n: (MB.decode_levels(tex[n])[0][..., :3], MB.decode_levels(open(new[n], "rb").read())[0][..., :3])
          for n in (MB.RIM, MB.BLUR)}
    crops = [(lv[MB.RIM][0], (0, 0, 244, 244), "rim_d.dds nuts, original (kn5)"),
             (lv[MB.RIM][1], (0, 0, 244, 244), "rim_d.dds nuts, new"),
             (lv[MB.BLUR][0], (380, 380, 644, 644), "rim_blur.dds hub, original"),
             (lv[MB.BLUR][1], (380, 380, 644, 644), "rim_blur.dds hub, new")]

    # ------------------------------------------------------------ changed pixels (in game, front left)
    o = np.asarray(R["wheel_front_left_orig"], np.int16)
    n = np.asarray(R["wheel_front_left"], np.int16)
    chg = np.abs(o - n).max(-1) > 6
    diff = np.where(chg[..., None], np.array(PIG, np.uint8), (np.asarray(R["wheel_front_left_orig"]) * 0.35)
                    .astype(np.uint8))
    R["diff"] = Image.fromarray(diff.astype(np.uint8))
    print(f"front left wheel in game: {chg.mean() * 100:.2f} % of the pixels changed by > 6 levels")

    # ------------------------------------------------------------ sheet
    PAD, T = 24, 432
    SW = PAD * 2 + 4 * T + 3 * PAD
    SH = 2500
    sheet = Image.new("RGB", (SW, SH), BG)
    d = ImageDraw.Draw(sheet)
    y = 18
    d.text((PAD, y), "БОЛТЫ КОЛЁС  ·  Butcher Chart Chrome", font=F_T(46), fill=PIG)
    y += 66
    d.text((PAD, y), f"rim_d.dds + rim_blur.dds, 1024×1024 DXT5, 11 mips, original headers and alpha — same files "
                     f"for Pozdnyakov_23 and Konopelko_00; all 4 rims share one UV", font=F_L(22), fill=DIM)
    y += 46

    def row(items, y, hdr=None):
        if hdr:
            d.text((PAD, y), hdr, font=F_T(27), fill=FG)
            y += 42
        for i, (im, lab) in enumerate(items):
            x = PAD + i * (T + PAD)
            sheet.paste(im.resize((T, T), Image.LANCZOS), (x, y))
            label(d, (x + 6, y + 6), lab, F_L(20))
        return y + T + 22

    y = row([(Image.fromarray(img).crop(box).resize((T, T), Image.NEAREST), lab) for img, box, lab in crops], y)

    # notes + swatches
    nut_game = MB.NUT_TEX * 226 / 255
    sw = [(MB.PIG, f"PIG {MB.PIG}  body pink (make_skin.py)"),
          (tuple(int(round(v)) for v in MB.NUT), f"anodised: PIG hue, saturation x{MB.SAT_GAIN}  {MB.rgb3(MB.NUT)}"),
          (MB.rgb3(MB.NUT_TEX), f"texel, lit face  {MB.rgb3(MB.NUT_TEX)}"),
          (MB.rgb3(nut_game), f"in game (x0.886 detail)  {MB.rgb3(nut_game)}")]
    for i, (c, t) in enumerate(sw):
        yy = y + i * 44
        d.rectangle([PAD, yy, PAD + 52, yy + 34], fill=tuple(int(v) for v in c))
        d.text((PAD + 66, yy + 3), t, font=F_L(21), fill=FG)
    notes = [
        "nuts = 5 x (hex body + domed cap + flange) of the rim mesh, UV islands",
        "in the top-left corner of rim_d.dds; no other rim part within 18 texels",
        "colour = pink x original grey / 66: the baked shading is kept",
        "",
        "rim: ksPerPixelMultiMap, useDetail=1, detailUVMultiplier=0,",
        "txDetail car_paint_rims.dds = one 226 grey texel, rim_d alpha 0",
        "-> AC: diffuse x 0.886 everywhere (colour-neutral), so the alpha",
        "stays bit for bit; the pink is pre-scaled for it. car_paint_rims",
        "untouched. rimblur: useDetail=0; its 5 dark nut blobs -> pink at",
        "the same scale. Every other 4x4 DXT block is byte-identical.",
    ]
    for i, t in enumerate(notes):
        d.text((PAD + 2 * (T + PAD) - 40, y + i * 29), t, font=F_L(20), fill=DIM if i > 3 else FG)
    y += max(4 * 44, len(notes) * 29) + 24

    y = row([(R["hub_" + k.replace(" ", "_")], k) for k in WHEELS], y,
            "hub close-ups, all 4 wheels (test skin = Konopelko_00 + new rim textures, render_rs3 --fit --ks-detail)")
    y = row([(R["hub_front_left_orig"], "original, in game"), (R["hub_front_left"], "new, in game"),
             (R["hub_front_left_raw"], "new, texel colour (no detail)"), (R["diff"], "pixels changed (wheel)")], y,
            "front left: before / after, and what changed")
    y = row([(R["blur_orig"], "spinning (rim_blur), original"), (R["blur_new"], "spinning (rim_blur), new"),
             (R["wheel_front_left"], "front left, in game"), (R["wheel_rear_right"], "rear right, in game")], y,
            "rim blur (rimblur meshes, shown at speed) and the whole wheel")
    sheet = sheet.crop((0, 0, SW, y))
    sheet.save(a.out)
    print(f"wrote {a.out} {sheet.size}")


if __name__ == "__main__":
    main()
