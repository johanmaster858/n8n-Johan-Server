"""Procedural stained-glass window for "Breathing Room".

The window is a pointed (equilateral) arch, 10 m wide and 15 m high.  Window
coordinates are in metres: u in [-5, 5] (left -> right as seen from inside the
cathedral), v in [0, 15] (sill -> apex).

The design is drawn once as a map of glass-piece IDs, mirrored so that it is
exactly symmetric, then leaded, coloured and painted.  Everything else in the
film (the window as seen from inside, the coloured light columns and the image
projected on the floor) samples the transmission texture produced here, so all
of them always match.

Iconography (original design): the Virgin Mary standing on a crescent moon
inside a golden mandorla of rays, two kneeling angels with raised, outstretched
wings on either side, a twelve-petal rose window in the arch head, a ruby
border with gold pearls and a deep-blue quarry background.
"""
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.measure import label

S = 320                       # master drawing: pixels per metre
WM, HM = 10.0, 15.0           # window size in metres
W, H = int(WM * S), int(HM * S)
VS = HM - math.sqrt(10.0 ** 2 - 5.0 ** 2)   # springing line of the arch (6.34 m)

ROSE_C = (0.0, 11.35)
ROSE_R = 2.2
HALO_C = (0.0, 7.92)          # Mary's head/halo centre (centre of block row 5)

OUTSIDE, STONE, LEAD = 0, 1, 2

# Linear-light transmission colours.  Several tones per class; each glass piece
# picks one and is jittered, like hand-made antique glass.
PALETTE = {
    'bg_blue':      [(0.030, 0.070, 0.50), (0.040, 0.090, 0.58), (0.025, 0.060, 0.42), (0.050, 0.110, 0.62)],
    'bg_ruby':      [(0.80, 0.05, 0.06), (0.70, 0.03, 0.08)],
    'border_ruby':  [(0.82, 0.05, 0.06), (0.72, 0.03, 0.06), (0.90, 0.10, 0.08)],
    'pearl':        [(1.00, 0.80, 0.30), (1.00, 0.74, 0.22)],
    'rose_ruby':    [(0.88, 0.06, 0.07), (0.78, 0.04, 0.08)],
    'rose_blue':    [(0.06, 0.15, 0.80), (0.05, 0.12, 0.70)],
    'rose_gold':    [(1.00, 0.72, 0.14), (1.00, 0.80, 0.28)],
    'rose_white':   [(1.00, 0.97, 0.86)],
    'rose_green':   [(0.10, 0.55, 0.28), (0.08, 0.48, 0.24)],
    'ray_gold':     [(1.00, 0.70, 0.12), (1.00, 0.64, 0.10)],
    'ray_pale':     [(1.00, 0.88, 0.50), (1.00, 0.92, 0.62)],
    'mandorla_rim': [(0.85, 0.06, 0.06), (0.75, 0.04, 0.07)],
    'mantle':       [(0.07, 0.18, 0.96), (0.06, 0.15, 0.90), (0.09, 0.21, 1.00)],
    'mantle_trim':  [(1.00, 0.76, 0.20)],
    'gown':         [(0.92, 0.07, 0.07), (0.84, 0.05, 0.08)],
    'flesh':        [(1.00, 0.86, 0.74), (0.98, 0.82, 0.70)],
    'wimple':       [(0.96, 0.96, 0.92)],
    'halo_disc':    [(1.00, 0.80, 0.28), (1.00, 0.74, 0.20)],
    'halo_ring':    [(0.86, 0.08, 0.06), (0.80, 0.05, 0.05)],
    'moon':         [(0.95, 0.95, 0.85), (1.00, 0.90, 0.60)],
    'cloud_white':  [(0.92, 0.94, 0.98), (0.86, 0.90, 0.97)],
    'cloud_blue':   [(0.45, 0.62, 0.95), (0.55, 0.70, 0.98)],
    'robe':         [(0.97, 0.95, 0.88), (0.93, 0.93, 0.90), (0.98, 0.93, 0.82)],
    'robe_trim':    [(1.00, 0.72, 0.16)],
    'hair':         [(1.00, 0.66, 0.14), (0.96, 0.58, 0.10)],
    'wing_covert':  [(1.00, 0.74, 0.16), (1.00, 0.66, 0.12), (1.00, 0.80, 0.30)],
    'wing_feather': [(1.00, 0.95, 0.80), (0.98, 0.93, 0.86), (1.00, 0.90, 0.66)],
    'wing_tip_ruby': [(0.88, 0.06, 0.08), (0.80, 0.04, 0.10)],
    'wing_tip_blue': [(0.09, 0.20, 0.95), (0.08, 0.18, 0.88)],
    'star':         [(1.00, 0.84, 0.30), (1.00, 0.78, 0.22)],
}

CLASS_OF = {}                 # piece id -> class name
_next = [10]


def new_id(cls):
    i = _next[0]
    _next[0] += 1
    CLASS_OF[i] = cls
    return i


def px(u, v):
    return ((u + WM / 2) * S, (HM - v) * S)


def poly(pts):
    return [px(u, v) for u, v in pts]


def catmull(points, n=16, closed=False):
    """Centripetal-ish Catmull-Rom through (u, v) control points."""
    pts = [np.asarray(p, float) for p in points]
    if closed:
        pts = [pts[-1]] + pts + [pts[0], pts[1]]
    else:
        pts = [2 * pts[0] - pts[1]] + pts + [2 * pts[-1] - pts[-2]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(pts[-2])
    return [tuple(p) for p in out]


def circle(cx, cy, r, n=160, a0=0.0, a1=2 * math.pi):
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n))
            for k in range(n + 1)]


def mirror(pts):
    return [(-u, v) for u, v in pts]


def sym(half):
    """Close a left-half outline (top centre -> bottom centre) into a symmetric polygon."""
    return list(half) + mirror(half[::-1])


def star(cx, cy, r_out, r_in, n=8, rot=math.pi / 2):
    pts = []
    for k in range(2 * n):
        r = r_out if k % 2 == 0 else r_in
        a = rot + math.pi * k / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def leaf(p0, p1, width, n=24):
    """Pointed-oval (vesica) shape from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    side_a, side_b = [], []
    for k in range(n + 1):
        t = k / n
        w = width * 0.5 * math.sin(math.pi * t) ** 0.8
        side_a.append(tuple(p0 + d * t + nrm * w))
        side_b.append(tuple(p0 + d * t - nrm * w))
    return side_a + side_b[::-1]


# ----------------------------------------------------------------------------
# Geometry fields over the master grid
# ----------------------------------------------------------------------------
_uu = (np.arange(W) + 0.5) / S - WM / 2
_vv = HM - (np.arange(H) + 0.5) / S
U, V = np.meshgrid(_uu, _vv)


def arch_distance(u, v):
    """Signed distance (m) to the arch outline, positive inside."""
    d_side = 5.0 - np.abs(u)
    d_arc = np.minimum(10.0 - np.hypot(u - 5.0, v - VS), 10.0 - np.hypot(u + 5.0, v - VS))
    d = np.where(v <= VS, d_side, d_arc)
    return np.minimum(d, v)


def arch_inside(u, v):
    return arch_distance(u, v) > 0


def build(seed=7):
    rng = np.random.default_rng(seed)
    CLASS_OF.clear()
    _next[0] = 10

    dist = arch_distance(U, V)
    inside = dist > 0

    ids = np.zeros((H, W), np.int32)

    # -- background: diamond quarries ------------------------------------
    q = 0.62
    ia = np.floor((U + V) / q).astype(np.int64)
    ib = np.floor((V - U) / q).astype(np.int64)
    key = (ia + 1000) * 4000 + (ib + 1000)
    uniq, inv = np.unique(key[inside], return_inverse=True)
    base = _next[0]
    for k in range(len(uniq)):
        CLASS_OF[base + k] = 'bg_blue'
    _next[0] += len(uniq)
    ids[inside] = base + inv

    # -- border band + stone rim -----------------------------------------
    rim, band = 0.10, 0.44
    in_band = inside & (dist < band)
    # arc-length coordinate along the outline of the nearest edge
    theta_l = math.pi - np.arctan2(V - VS, U - 5.0)     # left arc, from springing
    s = np.where(V <= VS, V, VS + 10.0 * theta_l)
    s = np.where(U > 0, np.where(V <= VS, V, VS + 10.0 * (math.pi - np.arctan2(V - VS, -U - 5.0))), s)
    bottom = V < np.minimum(5.0 - np.abs(U), band + 0.0)
    seg_len = 0.62
    seg = np.floor(s / seg_len).astype(np.int64)
    seg = np.where(bottom, 5000 + np.floor((U + 5.0) / seg_len).astype(np.int64), seg)
    seg = np.where(U < 0, seg, seg + 20000)
    uniq, inv = np.unique(seg[in_band], return_inverse=True)
    base = _next[0]
    for k in range(len(uniq)):
        CLASS_OF[base + k] = 'border_ruby'
    _next[0] += len(uniq)
    ids[in_band] = base + inv
    ids[inside & (dist < rim)] = STONE

    img = Image.fromarray(ids, mode='I')
    dr = ImageDraw.Draw(img)

    def fill(pts, cls_or_id):
        i = cls_or_id if isinstance(cls_or_id, int) else new_id(cls_or_id)
        dr.polygon(poly(pts), fill=i)
        return i

    folds = []

    def stroke(pts, width_m, value=LEAD):
        if value == LEAD and width_m >= 0.028:
            folds.append(list(pts))
        w = max(1, int(round(width_m * S)))
        dr.line(poly(pts), fill=value, width=w, joint='curve')
        r = w / 2.0
        for (u, v) in (pts[0], pts[-1]):
            x, y = px(u, v)
            dr.ellipse([x - r, y - r, x + r, y + r], fill=value)

    # gold pearls along the middle of the border band (left side; mirrored later)
    mid = (rim + band) / 2
    # straight side
    t = 0.31
    while t < VS:
        fill(circle(-5.0 + mid, t, 0.085, 28), 'pearl')
        t += seg_len
    # arc (pearls centred in the same arc-length segments as the border pieces)
    kseg = int(math.floor(VS / seg_len))
    while True:
        sc = (kseg + 0.5) * seg_len
        kseg += 1
        if sc < VS + 0.05:
            continue
        th = (sc - VS) / 10.0
        if th > math.pi / 3 - 0.03:
            break
        r = 10.0 - mid
        cu, cv = 5.0 - r * math.cos(th), VS + r * math.sin(th)
        fill(circle(cu, cv, 0.085, 28), 'pearl')
    # bottom
    t = -5.0 + 0.31 + seg_len
    while t < 0:
        fill(circle(t, mid, 0.085, 28), 'pearl')
        t += seg_len

    # small ruby jewels at some background quarry corners
    for a in range(-40, 40):
        for b in range(-40, 40):
            if (a + b) % 2:
                continue
            cu, cv = (a - b) * q / 2, (a + b) * q / 2
            if cu > 0.01 or cv < 0.3:
                continue
            if arch_distance(np.array(cu), np.array(cv)) > band + 0.15:
                fill(circle(cu, cv, 0.075, 20), 'bg_ruby')

    # -- bottom cloud band ---------------------------------------------------
    k = 0
    for row, (cv, r) in enumerate([(0.95, 0.30), (0.66, 0.34)]):
        cu = -4.55 + (0.22 if row else 0.0)
        while cu < 0.3:
            fill(circle(cu, cv, r, 40), 'cloud_white' if (k % 2) else 'cloud_blue')
            cu += 0.44
            k += 1

    # -- mandorla (glory of rays) behind Mary ------------------------------
    mand_top, mand_bot, mand_hw = 9.05, 1.0, 2.25
    mc_v = (mand_top + mand_bot) / 2
    hh = (mand_top - mand_bot) / 2
    cc = (hh * hh - mand_hw * mand_hw) / (2 * mand_hw)
    rr = mand_hw + cc
    in_mand = (np.hypot(U - cc, V - mc_v) < rr) & (np.hypot(U + cc, V - mc_v) < rr)
    d_mand = np.minimum(rr - np.hypot(U - cc, V - mc_v), rr - np.hypot(U + cc, V - mc_v))
    ray_o = (0.0, 5.3)
    ang = np.arctan2(V - ray_o[1], U - ray_o[0])
    nray = 36
    ray_k = np.floor((ang + math.pi) / (2 * math.pi) * nray + 0.5).astype(np.int64) % nray
    rad = np.hypot(U - ray_o[0], V - ray_o[1])
    ring = (rad > 2.45).astype(np.int64)
    rim_w = 0.15
    rim_k = np.floor((ang + math.pi) / (2 * math.pi) * 30).astype(np.int64)
    arr = np.array(img)
    base_ray = _next[0]
    for kk in range(nray):
        for rg in range(2):
            CLASS_OF[base_ray + kk * 2 + rg] = 'ray_gold' if kk % 2 == 0 else 'ray_pale'
    _next[0] += nray * 2
    base_rim = _next[0]
    for kk in range(30):
        CLASS_OF[base_rim + kk] = 'mandorla_rim'
    _next[0] += 30
    m_ray = in_mand & (d_mand >= rim_w)
    arr[m_ray] = (base_ray + ray_k * 2 + ring)[m_ray]
    m_rim = in_mand & (d_mand < rim_w)
    arr[m_rim] = (base_rim + rim_k)[m_rim]
    img = Image.fromarray(arr, mode='I')
    dr = ImageDraw.Draw(img)

    # -- rose window -----------------------------------------------------------
    rcx, rcy = ROSE_C
    fill(circle(rcx, rcy, ROSE_R, 200), STONE)
    # jewel ring
    for kk in range(24):
        a0 = 2 * math.pi * kk / 24 + math.pi / 2
        a1 = 2 * math.pi * (kk + 1) / 24 + math.pi / 2
        pts = circle(rcx, rcy, 2.06, 12, a0, a1)[::-1] + circle(rcx, rcy, 1.84, 12, a0, a1)
        fill(pts, 'rose_gold' if kk % 2 == 0 else 'rose_ruby')
    # sector backgrounds
    for kk in range(12):
        a0 = 2 * math.pi * (kk - 0.5) / 12 + math.pi / 2
        a1 = 2 * math.pi * (kk + 0.5) / 12 + math.pi / 2
        pts = [(rcx, rcy)] + circle(rcx, rcy, 1.84, 24, a0, a1)
        fill(pts, 'rose_blue')
    # lancet petals + gold roundels
    for kk in range(12):
        a = 2 * math.pi * kk / 12 + math.pi / 2
        ca, sa_ = math.cos(a), math.sin(a)
        pts = []
        prof = []
        for j in range(33):
            x = 0.86 + (1.76 - 0.86) * j / 32
            if x < 1.40:
                hw = 0.10 + 0.13 * (x - 0.86) / 0.54
            else:
                tt = (x - 1.40) / 0.36
                hw = 0.23 * math.sqrt(max(0.0, 1 - tt ** 1.6))
            prof.append((x, hw))
        for x, hw in prof:
            pts.append((rcx + x * ca - hw * sa_, rcy + x * sa_ + hw * ca))
        for x, hw in prof[::-1]:
            pts.append((rcx + x * ca + hw * sa_, rcy + x * sa_ - hw * ca))
        fill(pts, 'rose_ruby')
        fill(circle(rcx + 1.47 * ca, rcy + 1.47 * sa_, 0.12, 24), 'rose_gold')
        fill(circle(rcx + 1.08 * ca, rcy + 1.08 * sa_, 0.07, 18), 'rose_green')
    # tracery spokes
    for kk in range(12):
        a = 2 * math.pi * (kk + 0.5) / 12 + math.pi / 2
        stroke([(rcx + 0.8 * math.cos(a), rcy + 0.8 * math.sin(a)),
                (rcx + 1.86 * math.cos(a), rcy + 1.86 * math.sin(a))], 0.07, STONE)
    stroke(circle(rcx, rcy, 1.85, 200), 0.05, STONE)
    # oculus
    fill(circle(rcx, rcy, 0.82, 120), STONE)
    fill(circle(rcx, rcy, 0.74, 120), 'rose_blue')
    fill(star(rcx, rcy, 0.70, 0.30, 8), 'rose_gold')
    fill(circle(rcx, rcy, 0.20, 60), 'rose_white')

    # apex quatrefoil
    qc = (0.0, 14.05)
    fill(circle(qc[0], qc[1], 0.40, 80), STONE)
    for kk in range(4):
        a = math.pi / 4 + kk * math.pi / 2
        fill(circle(qc[0] + 0.14 * math.cos(a), qc[1] + 0.14 * math.sin(a), 0.19, 40), 'rose_ruby')
    fill(circle(qc[0], qc[1], 0.09, 30), 'rose_gold')

    # background stars (left half; mirrored later)
    for (cu, cv, r) in [(-2.72, 8.30, 0.16), (-2.35, 12.35, 0.15), (-1.80, 1.36, 0.11),
                        (-2.27, 3.30, 0.14), (-2.20, 4.60, 0.12)]:
        fill(star(cu, cv, r, r * 0.42, 8), 'star')

    # -- angels (left one; right one is the mirror image) -----------------------
    def wing(lead_pts, trail_pts, n_feathers, covert_w=0.34, slant=0.22, tip_colours=('wing_tip_ruby', 'wing_tip_blue')):
        L = np.array(catmull(lead_pts, 20))
        T = np.array(catmull(trail_pts, 20))
        # resample both edges to the same number of points by arc length
        def resample(P, n):
            d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
            t = np.linspace(0, d[-1], n)
            return np.c_[np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])]
        n = 400
        L, T = resample(L, n), resample(T, n)

        def P(s, w):
            s = min(max(s, 0.0), 1.0)
            i = s * (n - 1)
            i0 = int(math.floor(i))
            i1 = min(i0 + 1, n - 1)
            f = i - i0
            l = L[i0] * (1 - f) + L[i1] * f
            t = T[i0] * (1 - f) + T[i1] * f
            return tuple(l * (1 - w) + t * w)

        # whole wing base (feather colour) so no gaps show
        fill([tuple(p) for p in L] + [tuple(p) for p in T[::-1]], 'wing_feather')
        # flight feathers: slanted bands with rounded, overshooting tips
        for kf in range(n_feathers - 1, -1, -1):
            s0 = kf / n_feathers
            s1 = (kf + 1.25) / n_feathers
            pts = []
            m = 14
            for j in range(m + 1):          # along side 0 (from covert line to tip)
                w = covert_w * 0.8 + (1.0 - covert_w * 0.8) * j / m
                pts.append(P(s0 + slant * w, w))
            for j in range(9):              # rounded tip
                a = math.pi * j / 8
                sm = (s0 + s1) / 2 + slant
                hw = (s1 - s0) / 2
                pts.append(P(sm - hw * math.cos(a), 1.0 + 0.10 * math.sin(a)))
            for j in range(m, -1, -1):      # back along side 1
                w = covert_w * 0.8 + (1.0 - covert_w * 0.8) * j / m
                pts.append(P(s1 + slant * w, w))
            fill(pts, 'wing_feather')
            # coloured tip
            tip = []
            for j in range(7):
                w = 0.74 + 0.26 * j / 6
                tip.append(P(s0 + slant * w, w))
            for j in range(9):
                a = math.pi * j / 8
                sm = (s0 + s1) / 2 + slant
                hw = (s1 - s0) / 2
                tip.append(P(sm - hw * math.cos(a), 1.0 + 0.10 * math.sin(a)))
            for j in range(6, -1, -1):
                w = 0.74 + 0.26 * j / 6
                tip.append(P(s1 + slant * w, w))
            fill(tip, tip_colours[kf % len(tip_colours)])
            # quill (lead) along the feather
            stroke([P(s0 + (s1 - s0) * 0.5 + slant * w, w) for w in np.linspace(covert_w, 0.74, 8)], 0.022)
        # coverts: scalloped band along the leading edge
        nc = int(n_feathers * 1.6)
        for kc in range(nc):
            s0 = kc / nc
            s1 = (kc + 1.15) / nc
            pts = [P(s0, 0.0)]
            for j in range(10):
                a = math.pi * j / 9
                sm = (s0 + s1) / 2
                hw = (s1 - s0) / 2
                pts.append(P(sm - hw * math.cos(a), covert_w + 0.08 * math.sin(a)))
            pts.append(P(s1, 0.0))
            fill(pts, 'wing_covert')

    # raised back wing, sweeping up along the arch
    wing([(-3.60, 5.95), (-4.10, 6.95), (-4.26, 8.20), (-4.00, 9.45), (-3.45, 10.55), (-2.98, 11.12)],
         [(-3.30, 6.10), (-3.30, 6.95), (-3.14, 8.10), (-2.90, 9.20), (-2.68, 10.25), (-2.98, 11.12)],
         9)
    # front wing, reaching up and over towards the Virgin
    wing([(-3.00, 6.55), (-2.78, 7.60), (-2.34, 8.52), (-1.78, 9.18), (-1.42, 9.42)],
         [(-2.78, 6.25), (-2.38, 7.00), (-1.95, 7.75), (-1.62, 8.52), (-1.42, 9.42)],
         7, covert_w=0.38, slant=0.18, tip_colours=('wing_tip_blue', 'wing_tip_ruby'))

    # angel halo
    fill(circle(-3.28, 6.55, 0.47, 90), 'halo_ring')
    fill(circle(-3.28, 6.55, 0.40, 90), 'halo_disc')

    # cloud under the angel
    for j, (cu, cv, r) in enumerate([(-4.15, 1.72, 0.30), (-3.70, 1.62, 0.34), (-3.22, 1.66, 0.33),
                                     (-2.78, 1.74, 0.28), (-2.45, 1.86, 0.20), (-3.92, 2.02, 0.22),
                                     (-3.40, 2.0, 0.24), (-2.92, 2.02, 0.22)]):
        fill(circle(cu, cv, r, 40), 'cloud_white' if j % 2 == 0 else 'cloud_blue')

    # robe
    robe = [(-3.14, 6.06), (-2.96, 5.84), (-2.86, 5.30), (-2.90, 4.62), (-2.84, 3.92), (-2.60, 3.14),
            (-2.46, 2.40), (-2.60, 2.02), (-3.20, 1.95), (-3.90, 2.00), (-4.38, 2.14), (-4.36, 2.95),
            (-4.14, 3.92), (-3.94, 4.82), (-3.78, 5.52), (-3.52, 5.94), (-3.30, 6.07)]
    fill(catmull(robe, 10, closed=True), 'robe')
    # belt
    fill([(-3.98, 4.60), (-2.90, 4.72), (-2.90, 4.58), (-3.99, 4.45)], 'robe_trim')
    # hem trim
    fill(catmull([(-4.37, 2.30), (-3.9, 2.16), (-3.2, 2.10), (-2.58, 2.18), (-2.52, 2.02), (-3.2, 1.95),
                  (-3.9, 2.0), (-4.38, 2.14)], 8, closed=True), 'robe_trim')
    # robe folds (lead)
    stroke(catmull([(-3.55, 4.50), (-3.70, 3.60), (-3.85, 2.70), (-3.95, 2.15)], 8), 0.03)
    stroke(catmull([(-3.20, 4.55), (-3.25, 3.60), (-3.20, 2.70), (-3.15, 2.10)], 8), 0.03)
    stroke(catmull([(-3.35, 5.80), (-3.45, 5.20), (-3.50, 4.62)], 8), 0.03)
    # sleeve and raised arm
    sleeve = [(-3.28, 5.86), (-2.98, 6.02), (-2.60, 6.13), (-2.50, 5.92), (-2.78, 5.62), (-3.12, 5.40)]
    fill(catmull(sleeve, 8, closed=True), 'robe')
    fill([(-2.66, 6.10), (-2.52, 6.14), (-2.44, 5.93), (-2.58, 5.88)], 'robe_trim')
    # praying hands pointing towards the Virgin
    fill(leaf((-2.52, 5.98), (-2.20, 6.46), 0.20), 'flesh')
    # neck, hair, face
    fill([(-3.22, 6.06), (-3.04, 6.08), (-3.02, 6.30), (-3.20, 6.30)], 'flesh')
    fill(catmull([(-3.52, 6.72), (-3.30, 6.84), (-3.08, 6.78), (-3.00, 6.62), (-3.10, 6.44),
                  (-3.20, 6.30), (-3.40, 6.18), (-3.58, 6.24), (-3.60, 6.46)], 10, closed=True), 'hair')
    face = [(-3.16, 6.70), (-3.02, 6.66), (-2.96, 6.54), (-2.90, 6.47), (-2.95, 6.43), (-2.96, 6.36),
            (-3.00, 6.28), (-3.08, 6.22), (-3.18, 6.24), (-3.20, 6.42)]
    fill(catmull(face, 8, closed=True), 'flesh')

    # -- the Virgin -------------------------------------------------------------
    # crescent moon
    arr = np.array(img)
    moon = (np.hypot(U, V - 1.72) < 1.30) & (np.hypot(U, V - 2.02) > 1.22) & (V < 1.9)
    arr[moon] = new_id('moon')
    img = Image.fromarray(arr, mode='I')
    dr = ImageDraw.Draw(img)

    # halo behind the head
    hx, hy = HALO_C
    fill(circle(hx, hy, 0.93, 160), 'halo_ring')
    for kk in range(16):
        a0 = 2 * math.pi * kk / 16
        a1 = 2 * math.pi * (kk + 1) / 16
        fill([(hx, hy)] + circle(hx, hy, 0.83, 10, a0, a1), 'halo_disc')

    mantle_half = [(0.0, 8.40), (-0.28, 8.34), (-0.43, 8.14), (-0.48, 7.86), (-0.48, 7.56), (-0.56, 7.30),
                   (-0.82, 7.08), (-1.02, 6.84), (-1.10, 6.36), (-1.10, 5.70), (-1.15, 4.80),
                   (-1.25, 3.80), (-1.36, 2.80), (-1.46, 1.95), (-1.52, 1.46), (-1.32, 1.34),
                   (-1.00, 1.38), (-0.70, 1.33), (-0.30, 1.36), (0.0, 1.32)]
    fill(sym(catmull(mantle_half, 8)), 'mantle')
    gown_half = [(0.0, 7.46), (-0.20, 7.40), (-0.28, 6.90), (-0.36, 5.90), (-0.44, 4.70),
                 (-0.56, 3.30), (-0.70, 2.00), (-0.78, 1.40), (-0.40, 1.34), (0.0, 1.32)]
    gown = sym(catmull(gown_half, 8))
    fill(gown, 'gown')
    # gold trim along the mantle opening
    trim = catmull([(-0.20, 7.40), (-0.28, 6.90), (-0.36, 5.90), (-0.44, 4.70), (-0.56, 3.30),
                    (-0.70, 2.00), (-0.78, 1.40)], 8)
    trim_id = new_id('mantle_trim')
    stroke(trim, 0.085, trim_id)
    # hem trim of the mantle
    hem_id = new_id('mantle_trim')
    stroke(catmull([(-1.50, 1.47), (-1.32, 1.36), (-1.00, 1.40), (-0.78, 1.38)], 8), 0.07, hem_id)
    # mantle folds and gown folds (lead lines)
    for fold in ([(-0.72, 7.20), (-0.82, 6.90), (-0.86, 5.60), (-0.96, 4.10), (-1.08, 2.70), (-1.16, 1.36)],
                 [(-0.40, 5.60), (-0.62, 5.20), (-0.72, 3.90), (-0.86, 2.60), (-0.98, 1.34)],
                 [(-1.12, 6.10), (-1.02, 5.60), (-0.98, 4.90), (-1.00, 4.25)],
                 [(-0.12, 5.62), (-0.14, 5.40), (-0.18, 4.00), (-0.24, 2.60), (-0.30, 1.30)]):
        stroke(catmull(fold, 8), 0.03)
    stroke([(-1.13, 4.30), (-0.50, 4.12)], 0.03)
    stroke([(-1.31, 2.55), (-0.64, 2.42)], 0.03)
    stroke([(-0.52, 3.80), (0.0, 3.72)], 0.03)
    stroke([(-0.70, 2.10), (0.0, 2.02)], 0.03)
    # forearms (gown sleeves) meeting at the praying hands
    fill(catmull([(-0.50, 5.52), (-0.30, 5.72), (-0.14, 5.86), (-0.10, 5.66), (-0.28, 5.48), (-0.46, 5.34)],
                 6, closed=True), 'gown')
    fill([(-0.17, 5.84), (0.17, 5.84), (0.17, 5.70), (-0.17, 5.70)], 'mantle_trim')
    fill(sym([(0.0, 6.66), (-0.07, 6.54), (-0.12, 6.28), (-0.14, 5.98), (-0.12, 5.82), (0.0, 5.80)]), 'flesh')
    # wimple and face
    fill(circle(hx, hy - 0.04, 0.36, 90), 'wimple')
    fill(sym([(0.0, 7.62), (-0.20, 7.60), (-0.22, 7.40), (0.0, 7.36)]), 'wimple')
    face_pts = [(hx + 0.225 * math.cos(a), hy - 0.02 + 0.30 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 90)]
    fill(face_pts, 'flesh')
    # re-assert veil top over the wimple
    veil_top = [(0.0, 8.40), (-0.28, 8.34), (-0.43, 8.14), (-0.48, 7.86), (-0.40, 7.95), (-0.24, 8.16), (0.0, 8.22)]
    fill(sym(catmull(veil_top, 8)), 'mantle')

    arr = np.array(img)

    # -- mirror the left half onto the right -----------------------------------
    cx = W // 2
    left = arr[:, :cx]
    central = np.unique(np.r_[arr[:, cx - 2:cx + 2].ravel()])
    central_set = set(int(c) for c in central)
    offset = 10_000_000
    mirrored = left[:, ::-1].copy()
    keep = np.isin(mirrored, list(central_set | {OUTSIDE, STONE, LEAD}))
    mirrored = np.where(keep, mirrored, mirrored + offset)
    arr = np.concatenate([left, mirrored], axis=1)
    for i, c in list(CLASS_OF.items()):
        if i not in central_set:
            CLASS_OF[i + offset] = c
    CLASS_OF[OUTSIDE] = 'outside'
    CLASS_OF[STONE] = 'stone'
    CLASS_OF[LEAD] = 'lead'

    # -- split regions cut by explicit lead strokes into separate pieces --------
    pieces = label(arr, background=-1, connectivity=1)
    # class per piece
    flat_p = pieces.ravel()
    flat_i = arr.ravel()
    npieces = int(flat_p.max()) + 1
    piece_src = np.zeros(npieces, np.int64)
    piece_src[flat_p] = flat_i
    cls_names = sorted(set(CLASS_OF.values()))
    cls_index = {c: k for k, c in enumerate(cls_names)}
    src_ids = np.unique(flat_i)
    id_to_cls = {int(i): cls_index[CLASS_OF.get(int(i), 'bg_blue')] for i in src_ids}
    piece_cls = np.array([id_to_cls[int(piece_src[p])] for p in range(npieces)], np.int32)

    # -- lead lines: region boundaries + explicit strokes -----------------------
    bnd = np.zeros((H, W), bool)
    dx = pieces[:, 1:] != pieces[:, :-1]
    dy = pieces[1:, :] != pieces[:-1, :]
    bnd[:, 1:] |= dx
    bnd[:, :-1] |= dx
    bnd[1:, :] |= dy
    bnd[:-1, :] |= dy
    is_lead_src = (arr == LEAD)
    edt = ndimage.distance_transform_edt(~(bnd | is_lead_src))
    lead_half = 0.017 * S
    lead = np.clip(lead_half + 0.5 - edt, 0.0, 1.0)
    lead = np.maximum(lead, is_lead_src.astype(np.float64))

    # -- colour every piece -------------------------------------------------------
    col = np.zeros((npieces, 3), np.float32)
    for p in range(npieces):
        cname = cls_names[piece_cls[p]]
        if cname in ('outside', 'stone', 'lead'):
            continue
        tones = PALETTE[cname]
        c = np.array(tones[rng.integers(len(tones))], np.float32)
        c = c * rng.uniform(0.80, 1.12)
        # slight hue drift toward a neighbour tone
        c2 = np.array(tones[rng.integers(len(tones))], np.float32)
        c = c * 0.8 + c2 * 0.2 * rng.uniform(0.9, 1.1)
        col[p] = np.clip(c, 0, 1)
    T = col[pieces]

    # glass texture: low-frequency density + fine streaks, like antique glass
    def noise(scale_m, octaves=3, seed_=0):
        r = np.random.default_rng(seed_)
        acc = np.zeros((H, W), np.float32)
        amp = 1.0
        tot = 0.0
        for o in range(octaves):
            sm = max(2, int(H / (scale_m * S) * (2 ** o)))
            g = r.standard_normal((sm, max(2, int(sm * W / H)))).astype(np.float32)
            g = np.array(Image.fromarray(g).resize((W, H), Image.BICUBIC))
            acc += amp * g
            tot += amp
            amp *= 0.5
        return acc / tot
    n1 = noise(0.45, 3, 11)
    n2 = noise(0.12, 2, 12)
    streak = np.array(Image.fromarray(np.random.default_rng(5).standard_normal((60, 900)).astype(np.float32))
                      .resize((W, H), Image.BICUBIC))
    T *= (1.0 + 0.14 * n1 + 0.06 * n2 + 0.05 * streak)[..., None]

    # matting: painted shading that darkens glass near the leads
    mat = np.clip(edt / (0.10 * S), 0, 1)
    T *= (0.64 + 0.36 * mat ** 0.8)[..., None]

    # (c) painted shadow along drapery folds, for volume
    shade = Image.new('L', (W, H), 0)
    sd = ImageDraw.Draw(shade)
    for f in folds:
        sd.line(poly([(u + 0.05, v) for u, v in f]), fill=255, width=int(0.10 * S), joint='curve')
    shade_arr = np.asarray(shade, np.float32) / 255.0
    shade_arr = np.maximum(shade_arr, shade_arr[:, ::-1])
    shade_arr = ndimage.gaussian_filter(shade_arr, 0.05 * S)
    T *= (1.0 - 0.50 * shade_arr)[..., None]

    # -- grisaille paint: faces, hands, hair, feathers ---------------------------
    paint = Image.new('L', (W, H), 0)
    pd = ImageDraw.Draw(paint)

    def pline(pts, w_m, val=235):
        pd.line(poly(pts), fill=val, width=max(1, int(w_m * S)), joint='curve')

    # Virgin: downcast eyes, brows, nose, mouth
    for sgn in (-1, 1):
        pline(catmull([(sgn * 0.14, 7.93), (sgn * 0.09, 7.905), (sgn * 0.04, 7.925)], 6), 0.018)
        pline(catmull([(sgn * 0.16, 8.00), (sgn * 0.10, 8.03), (sgn * 0.035, 8.01)], 6), 0.014, 200)
    pline([(0.012, 7.99), (0.018, 7.84), (0.0, 7.81)], 0.012, 180)
    pline(catmull([(-0.045, 7.74), (0.0, 7.73), (0.045, 7.74)], 6), 0.014, 210)
    pline(catmull([(-0.13, 7.66), (0.0, 7.60), (0.13, 7.66)], 6), 0.010, 120)
    # praying fingers
    for k in (-0.045, 0.0, 0.045):
        pline([(k, 6.02), (k * 0.6, 6.50)], 0.008, 150)
    # left angel: eye, brow, mouth, hair curls
    pline(catmull([(-3.06, 6.525), (-3.01, 6.515), (-2.975, 6.53)], 6), 0.016)
    pline(catmull([(-3.08, 6.58), (-3.02, 6.60), (-2.97, 6.585)], 6), 0.012, 190)
    pline([(-2.985, 6.37), (-2.95, 6.375)], 0.012, 200)
    for (cu, cv) in [(-3.45, 6.6), (-3.35, 6.72), (-3.5, 6.35), (-3.25, 6.76), (-3.42, 6.45)]:
        pd.arc([*px(cu - 0.06, cv + 0.06), *px(cu + 0.06, cv - 0.06)], 200, 480, fill=170,
               width=int(0.012 * S))
    pline([(-2.47, 6.02), (-2.24, 6.40)], 0.008, 150)
    paint_arr = np.asarray(paint, np.float32) / 255.0
    # mirror paint (left half authored)
    paint_arr = np.maximum(paint_arr, paint_arr[:, ::-1])
    paint_arr = ndimage.gaussian_filter(paint_arr, 0.8)
    T *= (1.0 - 0.88 * paint_arr)[..., None]

    # stone and outside are opaque; leads are opaque
    cls_px = piece_cls[pieces]
    stone_mask = (cls_px == cls_index['stone'])
    outside_mask = (cls_px == cls_index['outside'])
    T[stone_mask | outside_mask] = 0
    T *= (1.0 - lead)[..., None]
    T = np.clip(T, 0, 1)

    return {
        'T': T,                                   # H x W x 3 linear transmission
        'lead': lead.astype(np.float32),
        'stone': stone_mask.astype(np.float32),
        'inside': inside.astype(np.float32),
        'pieces': pieces,
        'piece_cls': piece_cls,
        'cls_names': cls_names,
    }


def downsample(a, f):
    h, w = a.shape[:2]
    a = a[: h - h % f, : w - w % f]
    sh = (h // f, f, w // f, f) + a.shape[2:]
    return a.reshape(sh).mean(axis=(1, 3))


def to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_assets(out_dir, factor=2):
    """Build the design and write the textures used by the renderer."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    d = build()
    T = downsample(d['T'], factor).astype(np.float32)
    stone = downsample(d['stone'], factor).astype(np.float32)
    inside = downsample(d['inside'], factor).astype(np.float32)
    lead = downsample(d['lead'], factor).astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, 'window_design.npz'), T=T.astype(np.float16), stone=stone.astype(np.float16),
                        inside=inside.astype(np.float16), lead=lead.astype(np.float16), px_per_m=S / factor)
    # a viewable version of the design (as if backlit)
    img = to_srgb(T * 1.25)
    bg = np.array([0.10, 0.10, 0.11])
    img = img * inside[..., None] + bg * (1 - inside[..., None])
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(os.path.join(out_dir, 'stained_glass_design.png'))
    return T, stone, inside


if __name__ == '__main__':
    import sys
    save_assets(sys.argv[1] if len(sys.argv) > 1 else 'build')
