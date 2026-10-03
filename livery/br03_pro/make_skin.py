"""Скин Pozdnyakov_64 для SMP Racing BR03 EVO — ливрея «ГРАНЬ 64».

Три опоры: тайм-кафе «Арка», «Симкарт», Саратовская область (флаг, герб, 64 — код региона).
Команда ЭДМ, организаторы SMP Racing Esports, BR Engineering, РАФ, спонсор DriveOil.
Дизайн выбран этапом из трёх концепций и двух судей; координаты — пиксели текстуры 4096,
маски деталей развёртки — в zones/.

Запуск: python3 make_skin.py   → папка Pozdnyakov_64/ и архив Pozdnyakov_64.zip
"""
import json
import os
import sys
import zipfile

from draw import *
import draw
import lib as blib
from scipy import ndimage

SKIN = "Pozdnyakov_64"          # правило: фамилия латиницей_номер
OUT = os.path.join(HERE, SKIN)
os.makedirs(OUT, exist_ok=True)
BX = os.path.join(HERE, "brand_extra")


# ---------------------------------------------------------------- логотипы в высоком разрешении
def _L(n):
    return Image.open(os.path.join(BX, n)).convert("L")


def _mono(mask, col):
    return blib.solid(mask, col)


def _two(n, wc, ac):
    w, a = _L(n + "_white.png"), _L(n + "_accent.png")
    im = Image.new("RGBA", w.size, (0, 0, 0, 0))
    im.alpha_composite(_mono(w, wc))
    im.alpha_composite(_mono(a, ac))
    return im


def _coat():
    """Официальный герб Саратовской области: лазоревый щит, серебряные стерляди, золотая корона."""
    sh = _L("saratov_coa_shield.png")
    im = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    im.alpha_composite(_mono(sh, (20, 30, 60)))
    im.alpha_composite(_mono(sh.filter(ImageFilter.MinFilter(31)), hexc("#1694DC")))
    im.alpha_composite(_mono(_L("saratov_coa_crown.png"), (222, 178, 60)))
    im.alpha_composite(_mono(_L("saratov_coa_crown_lines.png"), (20, 30, 60)))
    im.alpha_composite(_mono(_L("saratov_coa_fish.png"), (225, 230, 238)))
    im.alpha_composite(_mono(_L("saratov_coa_lines.png"), (20, 30, 60)))
    return im


WHITE_ = hexc("#F4F5F7")
draw.ASSETS.update({
    "smp_lockup": _two("smp_racing_esports", WHITE_, hexc("#00B5EF")),
    "br_eng": _mono(_L("br_engineering_black.png"), WHITE_),
    "br_sym": _mono(_L("br_symbol_white.png"), WHITE_),
    "br03": _mono(_L("br03_logo_white.png"), WHITE_),
    "raf": _mono(_L("raf_black.png"), WHITE_),
    "coat": _coat(),
    "mbu": Image.open(os.path.join(BX, "edm_mbu_logo.png")).convert("RGBA"),
    "driveoil": blib.driveoil_logo(160, plate=False),
})

cv = Canvas()
yy, xx = np.mgrid[0:S, 0:S]
C.update({k: hexc(v) for k, v in dict(fade='#5C5D66', bez0='#0E6FB0', bez1='#5CC0F2', teal='#1FA3A8', coral='#D45554',
                                      silver='#C9CCD3', ruw='#FFFFFF', rub='#0039A6', rur='#D52B1E', frog='#5BAA3C').items()})

def Z(n): return zmask(n)
def A(m): return Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
def zfill(z, color, cond=None, name=None):
    m = Z(z).astype(np.float32)
    if cond is not None: m = m * cond
    cv.paint_alpha(A(m), C[color], name)

def screen(alpha_f, color):
    """screen blend colour with per-pixel alpha (float array)"""
    a = np.clip(alpha_f, 0, 1)[..., None] * M[..., None]
    b = np.array(cv.img).astype(np.float32) / 255
    c = np.array(color, np.float32) / 255
    s = 1 - (1 - b) * (1 - c)
    cv.img = Image.fromarray(((b * (1 - a) + s * a) * 255).clip(0, 255).astype(np.uint8))

# ============ 1. FLAT FILLS ============
# rear wing endplates: Saratov flag, white 2/3 over red 1/3 (car-down = small x on R, large x on L)
zfill('rear_wing_endplate_R', 'white'); zfill('rear_wing_endplate_R', 'red', cond=(xx < 148))
zfill('rear_wing_endplate_L', 'white'); zfill('rear_wing_endplate_L', 'red', cond=(xx > 3921))
# wing lower band (mirrored from behind => colour only): Saratov flag white over red
band = Z('rear_wing_lower_band').astype(np.float32)
cv.paint_alpha(A(band * (yy < 342)), C['white'], 'wingband_white')
cv.paint_alpha(A(band * (yy >= 342)), C['red'], 'wingband_red')
# wing lower element: yellow line
cv.paint_alpha(vline([(1500, 560, 24), (2600, 560, 24)]), C['yellow'], 'wing_lower_line')
for z in ['mirror_R', 'mirror_L', 'splitter_endplate_R', 'splitter_endplate_L']: zfill(z, 'red')
for z in ['sidepod_vane_R', 'sidepod_vane_L']: zfill(z, 'yellow')
# headlight bezels: azure "eye" (distance-from-edge gradient + dark hairline), both identical
for z in ['headlight_bezel_carLeftLamp', 'headlight_bezel_carRightLamp']:
    m = Z(z)
    d = ndimage.distance_transform_edt(m)
    t = np.clip((d - 5) / 25, 0, 1)
    col = (np.array(C['bez0'], np.float32)[None, None] * (1 - t[..., None]) + np.array(C['bez1'], np.float32)[None, None] * t[..., None])
    arr = np.array(cv.img).astype(np.float32)
    arr[m] = col[m]
    hair = m & (d <= 5)
    arr[hair] = C['deep']
    cv.img = Image.fromarray(arr.astype(np.uint8)); cv.log.append(('bezel_' + z, A(m.astype(np.float32))))
# sills
for side, z in (('R', 'mid_skirt_R'), ('L', 'mid_skirt_L')):
    zm = Z(z)
    zfill(z, 'red', cond=(yy >= 1500))
    h = hatch([(1300, 1230), (2800, 1230), (2800, 1500), (1300, 1500)], 48, None, lean=(1, -1) if side == 'R' else (1, 1), fade=('y', 1260, 1500, 4, 26))
    cv.paint_alpha(h, C['red'], clip=zm.astype(np.float32))
for z in ['front_skirt_R', 'front_skirt_L']: zfill(z, 'red')

# ============ 2. SPLITTER: lip checker (symmetric about x=2048) + dissolving checker on top ============
SQ = 41; X0 = 2048 - SQ / 2
lip = (Z('splitter_front_edge_R') | Z('splitter_front_edge_L')).astype(np.float32)
ck = ((np.floor((xx - X0) / SQ) + np.floor((yy - 3852) / SQ)) % 2 == 0) & (yy >= 3852) & (yy < 4020)
ck = ck.astype(np.float32) * lip
WIN = [(1528, 1888), (2208, 2568)]
for (a, b) in WIN:
    wm = Image.new('L', (S, S), 0); ImageDraw.Draw(wm).rounded_rectangle([a, 3902, b - 1, 3993], radius=12, fill=255)
    ck *= (np.array(wm) == 0)
cv.paint_alpha(A(ck), C['white'], 'lip_checker')
top = (Z('splitter_top_R') | Z('splitter_top_L')).astype(np.float32)
SQ2 = 41
k = np.floor((3846 - yy) / SQ2)                      # row index going up from the lip
cyc = 3846 - (k + 0.5) * SQ2
cxc = X0 + (np.floor((xx - X0) / SQ2) + 0.5) * SQ2
sc = np.where(k < 1, 1.0, np.clip(1 - (k - 1) / 3.0, 0, 1))  # row0 full, gone by row 4
half = sc * SQ2 / 2
par = ((np.floor((xx - X0) / SQ2) + k) % 2 == 0)
fadeck = par & (np.abs(xx - cxc) < half) & (np.abs(yy - cyc) < half) & (yy < 3846) & (k >= 0)
fadeck = fadeck.astype(np.float32) * top
fadeck[3440:3846, 1890:2210] = 0                       # nose-tip keep-clean
cv.paint_alpha(A(fadeck), C['fade'], 'splitter_fadecheck')
# nose spine checker column (stops before nose-tip keep-clean)
cv.paint_alpha(checker_rect(2014, 3352, 2082, 3454, 34), C['white'], 'nose_checker')

# ============ 3. ENGINE-COVER RIDGE CHECKER TAPES ============
cv.paint_alpha(checker_path([(1277, 410), (1318, 1300), (1312, 1640)], 18, 2), C['white'], 'tapeR')
cv.paint_alpha(checker_path([(2782, 410), (2768, 1300), (2786, 1640)], 18, 2), C['white'], 'tapeL')

# ============ 4. LINES ============
Y = {}
Y['fin_front_R'] = [(842, 1332, 0), (880, 1382, 12), (948, 1440, 22), (1022, 1500, 28), (1118, 1560, 32), (1198, 1612, 34), (1284, 1650, 34)]
Y['fin_front_L'] = [(3256, 1332, 0), (3218, 1382, 12), (3150, 1440, 22), (3074, 1500, 28), (2980, 1560, 32), (2900, 1612, 34), (2812, 1650, 34)]
Y['arm_R'] = [(1450, 730, 0), (1488, 860, 16), (1530, 1000, 22), (1562, 1120, 22), (1586, 1240, 18), (1604, 1330, 0)]
Y['arm_L'] = [(4094 - x, y, w) for x, y, w in Y['arm_R']]
Y['pod_shoulder_R'] = [(606, 1520, 0), (618, 1600, 12), (628, 1680, 16), (650, 1760, 18), (654, 1900, 18), (654, 2080, 14), (646, 2190, 0)]
Y['pod_shoulder_L'] = [(3448, 1520, 0), (3438, 1600, 12), (3424, 1680, 16), (3420, 1760, 18), (3420, 1900, 18), (3420, 2080, 14), (3426, 2190, 0)]
Y['fender_crest_R'] = [(540, 2998, 0), (540, 3060, 30), (540, 3300, 36), (540, 3400, 36)]
Y['brow_out_R'] = [(540, 3392, 36), (420, 3448, 30), (276, 3500, 18), (214, 3540, 0)]
Y['brow_in_R'] = [(540, 3392, 36), (662, 3448, 30), (806, 3500, 18), (846, 3540, 0)]
Y['fender_crest_L'] = [(3518, 2998, 0), (3518, 3060, 30), (3518, 3300, 36), (3518, 3400, 36)]
Y['brow_in_L'] = [(3518, 3392, 36), (3400, 3448, 30), (3254, 3500, 18), (3222, 3540, 0)]
Y['brow_out_L'] = [(3518, 3392, 36), (3640, 3448, 30), (3796, 3500, 18), (3846, 3540, 0)]
Y['nose_top_R'] = [(1836, 2030, 0), (1840, 2200, 16), (1866, 2400, 22), (1930, 2560, 24), (1915, 2680, 14), (1890, 2740, 0)]
Y['nose_top_L'] = [(2204, 2030, 0), (2196, 2200, 16), (2160, 2400, 22), (2092, 2560, 24), (2106, 2680, 14), (2128, 2740, 0)]
T2R = [(1095, 205, 30), (1030, 360, 34), (950, 540, 34), (870, 720, 30), (830, 880, 26), (835, 1020, 20), (818, 1160, 13), (770, 1290, 6), (720, 1390, 0)]
T2cR = [(1010, 300, 0), (935, 450, 8), (860, 620, 12), (790, 800, 12), (780, 960, 10), (775, 1100, 6), (740, 1220, 0)]
T2L = [(4081 - x, y, w) for x, y, w in T2R]
T2cL = [(4081 - x, y, w) for x, y, w in T2cR]
# red tail trail with dark keyline, then yellow companion
for nm, p in (('T2R', T2R), ('T2L', T2L)):
    kp = [(x, y, w + 8 if w > 0 else 0) for x, y, w in p]
    cv.paint_alpha(vline(kp), C['deep'], None)
    cv.paint_alpha(vline(p), C['red'], 'trail_' + nm)
for nm, p in (('T2cR', T2cR), ('T2cL', T2cL)):
    cv.paint_alpha(vline(p), C['yellow'], 'trail_' + nm)
for k_, v in Y.items(): cv.paint_alpha(vline(v), C['yellow'], 'blade_' + k_)
threads = {'arm_R': (Y['arm_R'], -24), 'arm_L': (Y['arm_L'], 24), 'fin_front_R': (Y['fin_front_R'], -30), 'fin_front_L': (Y['fin_front_L'], 30)}
for k_, (pts, off) in threads.items():
    p = offset_pts(pts, off); p = [(x, y, min(8, w)) for x, y, w in p]
    p[0] = (p[0][0], p[0][1], 0); p[-1] = (p[-1][0], p[-1][1], 0 if 'arm' in k_ else 8)
    cv.paint_alpha(vline(p), C['azure'], 'thread_' + k_)

# ============ 5. HATCH BLOCKS (parallelograms of whole bars, width ramps rear->front) ============
def hatch_block(side, xb, yr, xm, P=52, w_rear=6, w_front=26):
    """side R: s=x-y (front = low s); L: s=x+y (front = high s). xb: car-height band (x0,x1);
    yr: (y_rear,y_front) measured at x=xm. Whole bars only."""
    s = (xx - yy) if side == 'R' else (xx + yy)
    s_rear = xm - yr[0] if side == 'R' else xm + yr[0]
    s_front = xm - yr[1] if side == 'R' else xm + yr[1]
    n = int(abs(s_rear - s_front) // P)
    out = np.zeros((S, S), np.float32)
    for i in range(n + 1):
        w = w_front + (w_rear - w_front) * i / max(1, n)
        if side == 'R': a = s_front + i * P; m = (s >= a) & (s < a + w)
        else: a = s_front - i * P; m = (s <= a) & (s > a - w)
        out[m] = 1
    out *= ((xx >= xb[0]) & (xx <= xb[1]))
    return A(out)
HB = {
    'haunch_R': ('R', (430, 700), (1150, 1370), 565, 'rear_haunch_R'),
    'haunch_L': ('L', (3381, 3651), (1150, 1370), 3516, 'rear_haunch_L'),
    'sidepod_R': ('R', (480, 620), (1930, 2180), 550, 'sidepod_side_R'),
    'sidepod_L': ('L', (3455, 3595), (1930, 2180), 3525, 'sidepod_side_L'),
}
for k_, (sd, xb, yr, xm, z) in HB.items():
    cv.paint_alpha(hatch_block(sd, xb, yr, xm, w_front=24 if 'lower' in k_ else 26), C['red'], 'hatch_' + k_, clip=Z(z).astype(np.float32))

# ============ 6. BRAND GLOWS (only behind the two fin heroes and the splitter Симкарт) ============
DT = ndimage.distance_transform_edt(M)
def zone_feather(z, r=60): return np.clip(ndimage.distance_transform_edt(Z(z)) / r, 0, 1)
def blob(cx, cy, rx, ry): return np.exp(-2.2 * (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
fR = zone_feather('engine_cover_side_R'); fL = zone_feather('engine_cover_side_L')
screen(0.30 * blob(2905, 1160, 130, 460) * fL, C['red'])

# ============ 7. PLATES ============
def number_plate(poly, redcond, name):
    pm = np.array(poly_mask(poly)).astype(np.float32) / 255
    cv.paint_alpha(A(pm), C['white'], 'plate_' + name)
    cv.paint_alpha(A(pm * redcond), C['red'], None)
PR = [(116, 2330), (396, 2390), (396, 2772), (116, 2712)]
PL = [(4045 - x, y) for x, y in PR]
number_plate(PR, (xx < 209), 'R'); number_plate(PL, (xx > 4045 - 209), 'L')
n64 = fit(text_mask('64', 900, 300), h=144)
cv.paint_alpha(place(n64, (303, 2561), -90), C['ink'], 'door64_R')
cv.paint_alpha(place(n64, (4045 - 303, 2561), 90), C['ink'], 'door64_L')

def logo_rgba(nm, center, rot, h=None, w=None, name=None, mono=None):
    a = asset(nm); a = a.resize((int(w), int(round(a.height * w / a.width))) if w else (int(round(a.width * h / a.height)), int(h)), Image.LANCZOS)
    if mono is not None:
        al = a.split()[3]; a = Image.new('RGBA', a.size, C[mono] + (255,)); a.putalpha(al)
    cv.paint_rgba(place(a, center, rot, 'RGBA'), name or nm); return a.size
# SMP mono-white strip in the red band of each plate
logo_rgba('smp_lockup', (162, 2531), -90, h=44, name='smp_plate_R', mono='white')
logo_rgba('smp_lockup', (4045 - 162, 2531), 90, h=44, name='smp_plate_L', mono='white')

def driver_plate(center, rot, name):
    W, H = 330, 104
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=10, fill=C['deep'] + (255,), outline=C['white'] + (255,), width=3)
    fx, fw, fh = 16, 56, 38; fy = (H - fh) // 2
    for i, c in enumerate([(255, 255, 255), (0, 57, 166), (213, 43, 30)]):
        d.rectangle([fx, fy + i * fh // 3, fx + fw - 1, fy + (i + 1) * fh // 3 - 1], fill=c + (255,))
    x0 = fx + fw + 12; x1 = W - 3 - 12; avail = x1 - x0
    l1 = fit(text_mask('ПОЗДНЯКОВ', 800, 200), h=24); l2 = fit(text_mask('СТАНИСЛАВ', 800, 200), h=24)
    mx = max(l1.width, l2.width)
    if mx > 0.92 * avail:
        k = 0.92 * avail / mx; l1 = fit(l1, w=l1.width * k); l2 = fit(l2, w=l2.width * k)
    gap = 14; tot = l1.height + gap + l2.height; y0 = (H - tot) // 2
    cx = (x0 + x1) / 2
    for t, yy0 in ((l1, y0), (l2, y0 + l1.height + gap)):
        im.paste(Image.new('RGBA', t.size, C['white'] + (255,)), (int(round(cx - t.width / 2)), yy0), t)
    cv.paint_rgba(place(im, center, rot, 'RGBA'), name)
driver_plate((1177, 564), -90, 'driverplate_R')
driver_plate((2905, 564), 90, 'driverplate_L')
# roof disc
cv.paint_alpha(disc(2045, 1130, 180), C['yellow'], 'roofdisc'); cv.paint_alpha(disc(2045, 1130, 168), C['ink']); cv.paint_alpha(disc(2045, 1130, 156), C['white'])
cv.paint_alpha(place(fit(text_mask('64', 900, 300), h=150), (2045, 1130), 0), C['ink'], 'roof64')

# ============ 8. TEXT & LOGOS ============
def txt(s, wt, h, center, rot, color, name, tracking=0.0, outline=0, ocol='deep'):
    m = fit(text_mask(s, wt, 200, tracking), h=h)
    if outline:
        pad = outline + 4; big = Image.new('L', (m.width + 2 * pad, m.height + 2 * pad), 0); big.paste(m, (pad, pad))
        cv.paint_alpha(place(big.filter(ImageFilter.MaxFilter(2 * outline + 1)), center, rot), C[ocol], None); m = big
    cv.paint_alpha(place(m, center, rot), C[color], name); return m.size
def mask_col(m, center, rot, color, name): cv.paint_alpha(place(m, center, rot), C[color], name)
def arka(center, rot, h, name):
    w = fit(brand('arka_word'), h=h)
    pad = 24; big = Image.new('L', (w.width + 2 * pad, w.height + 2 * pad), 0); big.paste(w, (pad, pad))
    ol = big.filter(ImageFilter.MaxFilter(9)); sh = ImageChops.offset(ol, 8, 8)
    mask_col(sh, center, rot, 'deep', None); mask_col(ol, center, rot, 'deep', None); mask_col(big, center, rot, 'yellow', name)
    return w.size
def simkart(center, rot, w, name, glow=0.0):
    Aw = Image.open(BR + 'simkart_word_white.png').convert('L'); Br = Image.open(BR + 'simkart_word_red.png').convert('L')
    bb = ImageChops.lighter(Aw, Br).getbbox(); Aw = fit(Aw.crop(bb), w=w); Br = fit(Br.crop(bb), w=w)
    if glow:
        pad = 40
        g = Image.new('L', (Aw.width + 2 * pad, Aw.height + 2 * pad), 0); g.paste(ImageChops.lighter(Aw, Br), (pad, pad))
        g = g.filter(ImageFilter.GaussianBlur(14))
        screen(np.array(place(g, center, rot)).astype(np.float32) / 255 * glow, C['red'])
    # 'Сим' white->silver vertical gradient in logo space
    gr = np.linspace(0, 1, Aw.height)[:, None] * np.ones((1, Aw.width))
    top = np.array(C['white'], np.float32); bot = np.array(C['silver'], np.float32)
    rgb = (top[None, None] * (1 - gr[..., None]) + bot[None, None] * gr[..., None]).astype(np.uint8)
    simg = Image.fromarray(rgb, 'RGB').convert('RGBA'); simg.putalpha(Aw)
    cv.paint_rgba(place(simg, center, rot, 'RGBA'), name + '_sim')
    mask_col(Br, center, rot, 'red', name + '_kart'); return Aw.size

# --- fin heroes
print('arka hero', arka((1143, 1067), -90, 218, 'arka_fin_R'))
mask_col(fit(brand('arka_taimcafe'), h=50), (984, 1068), -90, 'yellow', 'taimcafe_fin_R')
print('simkart hero', simkart((2905, 1160), 90, 800, 'simkart_fin_L', glow=0.30))
mask_col(fit(brand('simkart_tagline'), w=540), (3017, 1029), 90, 'red', 'tagline_fin_L')
# --- haunch secondaries
print('simkart haunch', simkart((498, 820), -90, 520, 'simkart_haunch_R'))
print('arka haunch', arka((3551, 820), 90, 150, 'arka_haunch_L'))
# --- МБУ
logo_rgba('mbu', (1204, 1439), -90, h=124, name='mbu_R'); logo_rgba('mbu', (3132, 1262), 90, h=112, name='mbu_L')
# --- DriveOil
logo_rgba('driveoil', (379, 254), -90, w=270, name='driveoil_R'); logo_rgba('driveoil', (3654, 235), 90, w=280, name='driveoil_L')
logo_rgba('driveoil', (234, 3667), -64, w=190, name='driveoil_lamp_R'); logo_rgba('driveoil', (3818, 3665), 62, w=190, name='driveoil_lamp_L')
# --- BR ENGINEERING on бочка panels (centred low) + BR ENGINEERING lockup in lip window L
for side, cs, ce, rot in (('R', (1128, 1815), (1058, 1815), -90), ('L', (4045 - 1128, 1833), (4045 - 1058, 1833), 90)):
    logo_rgba('br_sym', cs, rot, h=96, name='br_hoop_' + side)
    t = fit(text_mask('ENGINEERING', 700, 200, 0.08), h=13); mask_col(t, ce, rot, 'white', 'br_eng_hoop_' + side)
# --- РАФ on rear lower side panels + nose top L
logo_rgba('raf', (1329, 2938), -90, h=108, name='raf_R'); logo_rgba('raf', (2769, 2936), 90, h=108, name='raf_L')
logo_rgba('raf', (2336, 2420), 0, h=136, name='raf_nose')
# --- team lockups: sidepods (side rule) + nose top R (front view)
for side, c1, c2, rot in (('R', (467, 1761), (545, 1761), -90), ('L', (3561, 1756), (3485, 1756), 90)):
    txt('ЭДМ', 900, 84, c1, rot, 'white', 'team_EDM_' + side)
    txt('— КОМАНДА —', 700, 26, c2, rot, 'yellow', 'team_komanda_' + side, 0.08)
print('nose komanda', txt('— КОМАНДА —', 700, 22, (1694, 2370), 0, 'yellow', 'nose_komanda', 0.08))
print('nose EDM', txt('ЭДМ', 900, 76, (1694, 2446), 0, 'white', 'nose_EDM'))
# --- sills
txt('ЭНГЕЛЬССКИЙ ДОМ МОЛОДЁЖИ', 800, 28, (1465, 1800), -90, 'white', 'midskirt_text_R', 0.05)
txt('ЭНГЕЛЬССКИЙ ДОМ МОЛОДЁЖИ', 800, 28, (2590, 1800), 90, 'white', 'midskirt_text_L', 0.05)
txt('САРАТОВСКАЯ ОБЛАСТЬ', 800, 26, (186, 3130), -94.5, 'white', 'skirt_text_R', 0.06)
txt('САРАТОВСКАЯ ОБЛАСТЬ', 800, 26, (3891, 3104), 97, 'white', 'skirt_text_L', 0.06)
# --- wing
logo_rgba('smp_lockup', (2045, 110), 0, h=80, name='smp_wing')
txt('ПЕПЕ ШНЕЙНЕ ВОТАФА', 900, 50, (2045, 214), 180, 'white', 'meme')
def frog(h):
    S4 = 4; W_, H_ = int(h * 1.35) * S4, h * S4
    t = Image.new('RGBA', (W_, H_), (0, 0, 0, 0)); d = ImageDraw.Draw(t)
    G, GD, LIP = (92, 160, 60), (40, 80, 30), (150, 60, 50); ow = int(H_ * 0.03)
    d.ellipse([W_ * 0.02, H_ * 0.28, W_ * 0.98, H_ * 0.98], fill=G, outline=GD, width=ow)
    for ex in (0.30, 0.68):
        cx, cy, r = W_ * ex, H_ * 0.32, H_ * 0.26
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=G, outline=GD, width=ow)
        d.ellipse([cx - r * 0.78, cy - r * 0.62, cx + r * 0.78, cy + r * 0.72], fill=(250, 250, 250), outline=GD, width=ow)
        d.ellipse([cx - r * 0.12, cy - r * 0.25, cx + r * 0.42, cy + r * 0.35], fill=(20, 20, 24))
        d.chord([cx - r * 0.82, cy - r * 0.9, cx + r * 0.82, cy + r * 0.1], 180, 360, fill=G)
        d.line([(cx - r * 0.8, cy - r * 0.38), (cx + r * 0.8, cy - r * 0.38)], fill=GD, width=ow)
    d.chord([W_ * 0.16, H_ * 0.62, W_ * 0.84, H_ * 0.9], 0, 180, fill=LIP, outline=GD, width=ow)
    d.line([(W_ * 0.16, H_ * 0.76), (W_ * 0.84, H_ * 0.76)], fill=GD, width=ow)
    return t.resize((W_ // S4, H_ // S4), Image.LANCZOS)
cv.paint_rgba(place(frog(64), (1600, 214), 180, 'RGBA'), 'frog_lo')
cv.paint_rgba(place(frog(64).transpose(Image.FLIP_LEFT_RIGHT), (2490, 214), 180, 'RGBA'), 'frog_hi')
# --- nose front: arms, region text, front 64, checker (painted above)
logo_rgba('coat', (2044, 2949), 0, h=268, name='coat_nose')
txt('САРАТОВСКАЯ', 800, 32, (2044, 3127), 0, 'white', 'nose_text1', 0.06)
txt('ОБЛАСТЬ', 800, 32, (2044, 3175), 0, 'white', 'nose_text2', 0.06)
print('nose64', txt('64', 900, 88, (2045, 3282), 0, 'white', 'nose64', outline=5, ocol='yellow'))
# --- splitter top sponsors (front view) — above the dissolving checker
arka((1712, 3600), 0, 128, 'arka_splitter')
print('simsplit', simkart((2394, 3612), 0, 330, 'simkart_splitter', glow=0.25))
# --- lip windows: SMP (car-right) + BR ENGINEERING (car-left)
logo_rgba('smp_lockup', (1708, 3948), 0, h=70, name='smp_lip_R')
logo_rgba('br_eng', (2388, 3948), 0, w=330, name='breng_lip_L')

# ============ 9. ДОПОЛНЕНИЯ: BR03, ссылка Симкарта ============
# BR03 на нижней боковине перед задним колесом (как у ADR), читается вдоль борта
logo_rgba('br03', (1180, 3170), -90, h=62, name='br03_R')
logo_rgba('br03', (2950, 3170), 90, h=62, name='br03_L')
# simkart.vercel.app — под тэглайном на крыле L и под «Симкарт» на заднем крыле R
txt('SIMKART.VERCEL.APP', 700, 22, (3060, 1000), 90, 'white', 'link_fin_L', 0.08)
txt('SIMKART.VERCEL.APP', 700, 22, (436, 820), -90, 'white', 'link_haunch_R', 0.08)



# ============ 10. ВЫВОД ============
img = cv.img
img.resize((2048, 2048), Image.LANCZOS).save(os.path.join(HERE, 'preview_texture.png'))
# защита от просвечивания на мип-уровнях: продлеваем каждую деталь наружу её же цветом
arr = np.array(img)
dist, (iy, ix) = ndimage.distance_transform_edt(~M, return_indices=True)
pad = (~M) & (dist <= 16)
arr[pad] = arr[iy[pad], ix[pad]]
img = Image.fromarray(arr)
blib.save_dxt5(img, os.path.join(OUT, 'body_paint.dds'))
blib.save_dxt5(Image.new('RGBA', (4, 4), C['base'] + (255,)), os.path.join(OUT, 'car_paint_rims.dds'))

# иконка ливреи 185×185: графит, жёлтый и красный клинья, белый 64 и полоска флага области
ic = Image.new('RGB', (185, 185), C['base'])
di = ImageDraw.Draw(ic)
di.polygon([(0, 128), (185, 70), (185, 92), (0, 150)], fill=C['yellow'])
di.polygon([(0, 154), (185, 96), (185, 118), (0, 176)], fill=C['red'])
di.text((92, 70), '64', font=blib.font(92), fill=C['white'], anchor='mm', stroke_width=4, stroke_fill=C['deep'])
ic.save(os.path.join(OUT, 'livery.png'))

with open(os.path.join(OUT, 'ui_skin.json'), 'w', encoding='utf-8') as fh:
    json.dump({'skinname': SKIN, 'drivername': 'Станислав Поздняков', 'country': 'Russia',
               'team': 'Команда ЭДМ · Арка · Симкарт', 'number': '64', 'priority': 1},
              fh, ensure_ascii=False, indent=2)

zp = os.path.join(HERE, SKIN + '.zip')
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, f), SKIN + '/' + f)
print('готово:', OUT)
print('--- clearance (min px to island edge, fraction outside) ---')
for name, alpha in cv.log:
    a = np.array(alpha) > 127
    if not a.any(): print(name, 'EMPTY'); continue
    d = DT[a].min(); o = (a & ~M).sum() / a.sum()
    ys, xs = np.nonzero(a)
    flag = ' <<' if (d < 20 or o > 0.002) else ''
    print('%-26s min=%5.1f out=%.3f bbox x%d-%d y%d-%d%s' % (name, d, o, xs.min(), xs.max(), ys.min(), ys.max(), flag))
