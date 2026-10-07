#!/usr/bin/env python3
"""Preview sheet for the pink calipers -> gear/caliper_preview.png

  1. flat caliper.dds: original (embedded in the kn5) vs new (gear/out/Pozdnyakov_00/caliper.dds)
  2. wheel close-ups (front/rear, both sides) rendered with tools/render_rs3.py --ks-detail from a test skin
     = copy of butcher_chrome/Pozdnyakov_00 + the new caliper.dds + caliper_detail.dds
  3. full side_left / side_right of the same test skin
  4. the AC detail multiply (render_rs3 --ks-detail): original / pink with the stock detail / final

Usage: python3 caliper_preview.py [--work DIR] [--skip-render]
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
import make_caliper as MC  # noqa: E402

BC = os.path.dirname(HERE)
SKIN = os.path.join(BC, "Pozdnyakov_00")
RENDER = "/home/user/dramatron/livery/tools/render_rs3.py"
WORK = "/tmp/claude-0/-home-user-dramatron/2c81f51a-52a4-58df-8bc9-3f535a44fd37/scratchpad/gear_cal"
FONTS = "/home/user/dramatron/livery/fonts"

BG = (24, 18, 22)
FG = (246, 236, 240)
DIM = (176, 150, 160)
PIG = (242, 158, 178)
INK = (96, 18, 42)

WHEELS = {   # name: (view, fit box x0,y0,z0,x1,y1,z1)
    "front left": ("side_left", (0.70, 0.0, 0.97, 0.98, 0.66, 1.66)),
    "rear left": ("side_left", (0.70, 0.0, -1.68, 0.98, 0.66, -1.01)),
    "front right": ("side_right", (-0.98, 0.0, 0.97, -0.70, 0.66, 1.66)),
    "rear right": ("side_right", (-0.98, 0.0, -1.68, -0.70, 0.66, -1.01)),
}


def font(name, size, var=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), int(size))
    if var:
        f.set_variation_by_name(var)
    return f


F_T = lambda s: font("Podkova[wght].ttf", s, "ExtraBold")
F_L = lambda s: font("Exo2-Italic[wght].ttf", s, "SemiBold Italic")


def render(skin, out, view, w, h, fit=None, detail=False):
    cmd = [sys.executable, RENDER, skin, out, "--views", view, "--width", str(w), "--height", str(h), "--jobs", "1"]
    if fit:
        cmd.append("--fit=" + ",".join(str(v) for v in fit))
    if detail:
        cmd.append("--ks-detail")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return Image.open(os.path.join(out, view + ".png")).convert("RGB")


def make_skin_copy(dst, files):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(SKIN, dst)
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
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "caliper_preview.png"))
    a = ap.parse_args()
    W = a.work
    os.makedirs(W, exist_ok=True)
    final = os.path.join(HERE, "out", "Pozdnyakov_00", MC.TEX)
    detail = os.path.join(HERE, "out", "Pozdnyakov_00", MC.DETAIL)
    test, test_stock = os.path.join(W, "test_skin"), os.path.join(W, "test_skin_stockdetail")

    if not a.skip_render:
        make_skin_copy(test, [final, detail])
        make_skin_copy(test_stock, [final])
    R = {}
    for name, (view, fit) in WHEELS.items():
        out = os.path.join(W, "pv_" + name.replace(" ", "_"))
        R[name] = (Image.open(os.path.join(out, view + ".png")).convert("RGB") if a.skip_render
                   else render(test, out, view, 900, 820, fit, True))
    for view in ("side_left", "side_right"):
        out = os.path.join(W, "pv_full")
        R[view] = (Image.open(os.path.join(out, view + ".png")).convert("RGB") if a.skip_render
                   else render(test, out, view, 1600, 900, None, True))
    fl = WHEELS["front left"]
    det = [("original, no detail", SKIN, False), ("original, in game", SKIN, True),
           ("pink, stock detail (x0.18)", test_stock, True), ("final: + white detail", test, True)]
    for k, (lab, skin, dflag) in enumerate(det):
        out = os.path.join(W, f"pv_detail{k}")
        R[lab] = (Image.open(os.path.join(out, fl[0] + ".png")).convert("RGB") if a.skip_render
                  else render(skin, out, fl[0], 600, 547, fl[1], dflag))

    # ------------------------------------------------------------ flat textures
    _, src, _ = MC.load_original()
    before = src.convert("RGB")
    after = Image.fromarray(MC.read_dds_levels(final)[0][..., :3])

    # ------------------------------------------------------------ sheet
    PAD, SW = 24, 1848
    sheet = Image.new("RGB", (SW, 3000), BG)
    d = ImageDraw.Draw(sheet)
    y = PAD
    d.text((PAD, y), "СУППОРТЫ  ·  Butcher Chart Chrome", font=F_T(46), fill=PIG)
    y += 64
    d.text((PAD, y), "caliper.dds 256×256 DXT5, 9 mips, alpha = original + caliper_detail.dds 4×4 white — same "
           "files for Pozdnyakov_00 and Konopelko_00; all 4 calipers share one UV", font=F_L(22), fill=DIM)
    y += 46

    # row 1: flat before / after + swatches / notes
    T = 520
    for k, (im, lab) in enumerate(((before, "original (kn5), grey AO, alpha 0"), (after, "new, original alpha (0)"))):
        x = PAD + k * (T + PAD)
        sheet.paste(im.resize((T, T), Image.NEAREST), (x, y))
        label(d, (x, y + T - 40), lab, F_L(22))
    x0 = PAD + 2 * (T + PAD) + 10
    yy = y
    d.text((x0, yy), "colour", font=F_T(30), fill=FG); yy += 44
    for col, txt in ((PIG, f"PIG  {PIG}  lit faces (= car body)"), (INK, f"INK  {INK}  mid AO"),
                     ((40, 10, 22), "INK_D  (40, 10, 22)  deep AO creases")):
        d.rectangle([x0, yy, x0 + 64, yy + 40], fill=col, outline=(255, 255, 255))
        d.text((x0 + 80, yy + 6), txt, font=F_L(22), fill=FG)
        yy += 54
    yy += 10
    notes = [
        "body: gradient map of the original AO, palette only",
        "brake pads (inside the caliper): neutral steel grey",
        "bleed screws (tiny top parts): silver",
        "no lettering in the original texture -> none added",
        "",
        "alpha: material ksPerPixelMultiMap, useDetail=1, detailUVMultiplier=0,",
        "txDetail = txMaps = caliper_detail.dds (one grey 45 texel).",
        "AC: diffuse.rgb *= lerp(detail, 1, diffuse.alpha); alpha is 0 everywhere",
        "-> with the stock detail every texel x0.18 (black / dark burgundy).",
        "caliper.dds keeps the original alpha bit for bit; the skin also",
        "overrides caliper_detail.dds with white (x1.0) -> the pink shows.",
        "txMaps side effect: spec/gloss/refl 0.18 -> 1.0 (satin paint).",
    ]
    for t in notes:
        d.text((x0, yy), t, font=F_L(21), fill=FG if t and not t.startswith(("AC", "->", "cal", "over", "txMaps",
                                                                            "txDetail", "alpha")) else DIM)
        yy += 30
    y = max(y + T, yy) + PAD

    # row 2: wheel close-ups
    d.text((PAD, y), "wheel close-ups (test skin = Pozdnyakov_00 + caliper.dds + caliper_detail.dds, render_rs3 --fit --ks-detail)", font=F_T(30),
           fill=FG)
    y += 46
    cw = (SW - PAD * 5) // 4
    ch = int(cw * 820 / 900)
    for k, name in enumerate(WHEELS):
        x = PAD + k * (cw + PAD)
        sheet.paste(R[name].resize((cw, ch), Image.LANCZOS), (x, y))
        label(d, (x + 6, y + 6), name, F_L(22))
    y += ch + PAD

    # row 3: full sides
    fw = (SW - PAD * 3) // 2
    fh = int(fw * 900 / 1600)
    for k, view in enumerate(("side_left", "side_right")):
        x = PAD + k * (fw + PAD)
        sheet.paste(R[view].resize((fw, fh), Image.LANCZOS), (x, y))
        label(d, (x + 6, y + 6), view, F_L(22))
    y += fh + PAD

    # row 4: detail multiply
    d.text((PAD, y), "AC detail multiply (render_rs3 --ks-detail), front left", font=F_T(30), fill=FG)
    y += 46
    dw = (SW - PAD * 5) // 4
    for k, (lab, _, _) in enumerate(det):
        x = PAD + k * (dw + PAD)
        im = R[lab].crop((100, 120, 440, 460)).resize((dw, dw), Image.LANCZOS)
        sheet.paste(im, (x, y))
        label(d, (x + 6, y + 6), lab, F_L(22))
    y += dw + PAD
    sheet = sheet.crop((0, 0, SW, y))
    sheet.save(a.out)
    print("wrote", a.out, sheet.size)


if __name__ == "__main__":
    main()
