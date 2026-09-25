"""Everything that changes over time, as pure functions of t (seconds).

Frames are independent: each one is computed from t = frame / 30 exactly, so
any frame can be rendered on its own and in any order.
"""
import math

import numpy as np

import scene as S
from assets import WORDS, CORE_WORDS

FPS = 30
DURATION = 40.0
NFRAMES = int(round(FPS * DURATION))

# ------------------------------------------------------------------ easing --


def clamp(x, a=0.0, b=1.0):
    return a if x < a else (b if x > b else x)


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp(x)
    return x * x * x * (x * (x * 6 - 15) + 10)


def ramp(t, t0, t1):
    return clamp((t - t0) / (t1 - t0))


def lerp(a, b, x):
    return a + (b - a) * x


# ------------------------------------------------------------- rotations --


def quat_axis_angle(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = math.sin(ang / 2)
    return np.array([math.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_slerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0:
        b = -b
        d = -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def bezier(p0, p1, p2, p3, s):
    u = 1 - s
    return (u * u * u) * p0 + (3 * u * u * s) * p1 + (3 * u * s * s) * p2 + (s * s * s) * p3


# ------------------------------------------------------------------ camera --


def cam_from_yaw_pitch(pos, yaw_deg, pitch_deg, fov_y_deg=44.0, aspect=16 / 9):
    ps, ys = math.radians(pitch_deg), math.radians(yaw_deg)
    f = np.array([math.sin(ys) * math.cos(ps), math.sin(ps), -math.cos(ys) * math.cos(ps)])
    u = np.array([-math.sin(ys) * math.sin(ps), math.cos(ps), math.cos(ys) * math.sin(ps)])
    r = np.cross(f, u)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    ty = math.tan(math.radians(fov_y_deg) / 2)
    C = np.zeros(14)
    C[0:3] = pos
    C[3:6] = r
    C[6:9] = u
    C[9:12] = f
    C[12] = ty * aspect
    C[13] = ty
    return C


def look_angles(pos, target):
    d = np.asarray(target, float) - np.asarray(pos, float)
    yaw = math.degrees(math.atan2(d[0], -d[2]))
    pitch = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
    return yaw, pitch


PATCH_Z = S.HS - S.D + 7.5          # centre of the floor projection (z)
FOV = 50.0
FOV_TOP = 44.0


def nave_camera(t):
    """Position and look-at target while we watch the window from the nave."""
    # 0-11 s almost still; 11-17 s a slow push in (towards the blocked window);
    # then the camera keeps creeping forward while the light returns.
    a = 0.25 * smoother(t / 11.0) if t < 11.0 else 0.25 + 0.55 * smoother((t - 11.0) / 6.5) if t < 17.5 else \
        0.80 + 0.20 * smoother((t - 17.5) / 8.5)
    z = lerp(31.0, 26.0, a)
    y = lerp(3.2, 3.7, a)
    pos = np.array([0.03 * math.sin(t * 0.35), y + 0.02 * math.sin(t * 0.5 + 1.0), z])
    tgt = np.array([0.0, lerp(7.3, 6.9, a), 0.0])
    return pos, tgt

# camera keys during the rise: (t, pos, yaw, pitch)
_P0, _T0 = nave_camera(26.0)
_Y0, _PI0 = look_angles(_P0, _T0)
# The light columns fill the slab 1.77 <= y + z <= 16.77 (m) in front of the wall.
# Dolly forward at floor level into the light, then crane straight up out of
# it, turning around so the image on the floor ends up upright.
RISE_T0, RISE_T1 = 25.4, 32.6
YAW_T0, YAW_T1 = 27.2, 32.8
RISE_KEYS = [
    (25.4, None, None, None),
    (27.7, np.array([0.0, 2.7, 12.6]), None, 15.0),          # into the beams, looking up at the glass
    (29.4, np.array([-0.4, 6.2, 9.9]), None, -34.0),         # rising inside the light, turning down
    (30.9, np.array([-0.25, 13.8, 9.45]), None, -76.0),      # out of the top of the columns
    (32.6, np.array([0.0, 20.6, PATCH_Z]), None, -90.0),     # straight down on the floor
]


def _rise_start():
    p, tg = nave_camera(RISE_T0)
    y, pi = look_angles(p, tg)
    return p, pi


def trapezoid(x, a=0.22, b=0.30):
    """Progress for a move that accelerates over a, cruises, and brakes over b."""
    x = clamp(x)
    vmax = 1.0 / (1.0 - a / 2 - b / 2)
    if x < a:
        return vmax * x * x / (2 * a)
    if x <= 1 - b:
        return vmax * (a / 2 + (x - a))
    y = 1 - x
    return 1.0 - vmax * y * y / (2 * b)


def _catmull(p0, p1, p2, p3, s):
    s2, s3 = s * s, s * s * s
    return 0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s3)


def camera(t):
    """Camera array C (see scene.cam_ray) for time t."""
    if t < RISE_T0:
        pos, tgt = nave_camera(t)
        yaw, pitch = look_angles(pos, tgt)
        return cam_from_yaw_pitch(pos, yaw, pitch, FOV)
    # the turn that makes the image on the floor upright, spread over the whole move
    yaw = 180.0 * smooth((t - YAW_T0) / (YAW_T1 - YAW_T0))
    if t >= RISE_T1:
        a = smooth((t - RISE_T1) / (36.0 - RISE_T1))
        pos = np.array([0.0, lerp(20.6, 19.5, a), PATCH_Z])
        return cam_from_yaw_pitch(pos, yaw, -90.0, FOV_TOP)
    p0, pi0 = _rise_start()
    keys = [(RISE_KEYS[0][0], p0, pi0)] + [(k[0], k[1], k[3]) for k in RISE_KEYS[1:]]
    tt = RISE_T0 + (RISE_T1 - RISE_T0) * trapezoid((t - RISE_T0) / (RISE_T1 - RISE_T0))
    for i in range(len(keys) - 1):
        if keys[i][0] <= tt <= keys[i + 1][0]:
            break
    k0 = keys[max(i - 1, 0)]
    k1 = keys[i]
    k2 = keys[i + 1]
    k3 = keys[min(i + 2, len(keys) - 1)]
    s = (tt - k1[0]) / (k2[0] - k1[0])
    pos = _catmull(k0[1], k1[1], k2[1], k3[1], s)
    pitch = float(_catmull(np.array(k0[2]), np.array(k1[2]), np.array(k2[2]), np.array(k3[2]), s))
    fov = lerp(FOV, FOV_TOP, smooth((tt - RISE_T0) / (RISE_T1 - RISE_T0)))
    return cam_from_yaw_pitch(pos, yaw, float(np.clip(pitch, -90.0, 89.0)), fov)


# ------------------------------------------------------------------- cells --


def _cells():
    cells = []
    for r in range(S.NROWS):
        for c in range(S.NCOLS):
            u0 = -5.0 + c * S.B
            v0 = r * S.B
            n = 0
            for j in range(12):
                for i in range(12):
                    if S.arch_inside(u0 + (i + 0.5) / 12 * S.B, v0 + (j + 0.5) / 12 * S.B):
                        n += 1
            if n / 144.0 > 0.02:
                cells.append((r, c, u0 + S.B / 2, v0 + S.B / 2))
    return cells


CELLS = _cells()
NB = len(CELLS)
CENTER = [k for k, c in enumerate(CELLS) if c[0] == 4 and c[1] == 3][0]   # Mary's praying hands

rng = np.random.default_rng(2024)

# ---- fill order: from the edges inward, the centre ("BUSY") closes last ----
_dist = np.array([math.hypot(CELLS[k][2] - CELLS[CENTER][2], CELLS[k][3] - CELLS[CENTER][3]) for k in range(NB)])
_key = -_dist + rng.uniform(-1.6, 1.6, NB)
_key[CENTER] = 1e9
FILL_ORDER = list(np.argsort(_key))
assert FILL_ORDER[-1] == CENTER

T_FILL0, T_FILL_SPAN = 3.0, 7.35
LAUNCH = np.zeros(NB)
LAND = np.zeros(NB)
for n, k in enumerate(FILL_ORDER):
    x = n / NB
    LAUNCH[k] = T_FILL0 + T_FILL_SPAN * x ** (1 / 1.9)
    LAND[k] = LAUNCH[k] + (2.15 - 1.52 * x ** 0.8)

# ---- words ----
WORD = np.zeros(NB, int)
_first = [0, 1, 2, 3]       # BUSY, EFFICIENCY, URGENT, DEADLINE fly in first
for n, k in enumerate(FILL_ORDER):
    if n < 4:
        WORD[k] = _first[n]
    elif rng.random() < 0.62:
        WORD[k] = rng.integers(CORE_WORDS)
    else:
        WORD[k] = rng.integers(CORE_WORDS, len(WORDS))
WORD[CENTER] = 0            # the one that will be let go of is "BUSY"

# ---- start positions and spins ----
START = np.zeros((NB, 3))
Q0 = np.zeros((NB, 4))
_cam0 = np.array([0.0, 2.4, 31.0])
# the first four cubes enter from just beside the camera so their words can be read
_hero_starts = [np.array([4.7, 1.6, 26.2]), np.array([-4.9, 5.3, 25.8]),
                np.array([4.9, 5.7, 25.6]), np.array([-4.7, 1.2, 26.0])]
HERO = set()
for n, k in enumerate(FILL_ORDER):
    if n < 4:
        START[k] = _hero_starts[n]
        ang = math.radians(rng.uniform(14, 26))
        HERO.add(k)
        LAND[k] = LAUNCH[k] + 3.3 - 0.15 * n
    else:
        side = 1.0 if rng.random() < 0.5 else -1.0
        START[k] = _cam0 + np.array([side * rng.uniform(1.6, 5.5), rng.uniform(-1.8, 4.0), rng.uniform(0.5, 3.5)])
        ang = math.radians(rng.uniform(70, 200))
    Q0[k] = quat_axis_angle(rng.normal(size=3), ang)

# ---- release (chain reaction) ----
T_BUSY = 14.0               # the first block starts to loosen
_dmax = _dist.max()
_d1 = np.sort(_dist)[1]
RELEASE = np.zeros(NB)
for k in range(NB):
    if k == CENTER:
        RELEASE[k] = T_BUSY
    else:
        x = (_dist[k] - _d1) / (_dmax - _d1)
        RELEASE[k] = 19.0 + 5.1 * x ** 0.78 + rng.uniform(0.0, 0.22)
SEED = rng.integers(1, 10_000, NB)
FALL_AXIS = np.array([quat_axis_angle((1.0, rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3)), 1.0)[1:] for _ in range(NB)])
FALL_SPIN = rng.uniform(0.5, 1.3, NB) * np.where(rng.random(NB) < 0.5, -1, 1)
FALL_DRIFT = rng.uniform(-0.25, 0.25, NB)


def removal_timing(k):
    """(loosen start, slide start, fall start, dissolve start, dissolve end) for cube k."""
    r = RELEASE[k]
    if k == CENTER:
        return r, r + 1.0, r + 1.8, r + 2.35, r + 4.3
    return r, r + 0.30, r + 0.70, r + 0.95, r + 2.15


def settled_pos(k):
    r, c, u, v = CELLS[k]
    return np.array([u, S.HS + v, -S.D / 2])


def block_state(k, t):
    """Returns (visible, moving, centre, R, dissolve, alpha, loosen) for cube k at time t."""
    end = settled_pos(k)
    if t < LAUNCH[k]:
        return False, False, end, np.eye(3), 0.0, 0.0, 0.0
    if t < LAND[k]:
        tau = (t - LAUNCH[k]) / (LAND[k] - LAUNCH[k])
        p0 = START[k]
        p3 = end
        if k in HERO:
            # drift into view slowly, then glide to the window
            s = 0.5 - 0.5 * math.cos(math.pi * tau)
            p1 = p0 + np.array([-0.45 * p0[0], 0.2, -6.0])
            p2 = p3 + np.array([0.0, 0.0, 6.0])
        else:
            s = 1 - (1 - tau) ** 2.3
            p1 = p0 + (p3 - p0) * 0.35 + np.array([0.0, 1.2, 0.0])
            p2 = p3 + np.array([0.0, 0.0, 3.2])
        pos = bezier(p0, p1, p2, p3, s)
        qr = quat_slerp(Q0[k], np.array([1.0, 0, 0, 0]), smooth(s / 0.85))
        return True, True, pos, quat_to_mat(qr), 0.0, 1.0, 0.0
    lo, sl, fa, d0, d1 = removal_timing(k)
    if t < lo:
        # settled, with a tiny recoil right after landing
        dt = t - LAND[k]
        z_off = 0.05 * math.sin(math.pi * min(dt / 0.13, 1.0)) * math.exp(-dt / 0.09) if dt < 0.5 else 0.0
        return True, False, end + np.array([0, 0, z_off]), np.eye(3), 0.0, 1.0, 0.0
    if t < sl:
        # loosening: a tremble that grows, light starts to leak around it
        a = (t - lo) / (sl - lo)
        amp = 0.004 + 0.018 * a * a
        jit = np.array([math.sin(t * 71.0 + k), math.sin(t * 53.0 + 2 * k), 0.0]) * amp
        rot = quat_to_mat(quat_axis_angle((math.sin(t * 37 + k), math.cos(t * 41 + k), 0.3), 0.6 * amp))
        return True, False, end + jit, rot, 0.0, 1.0, a
    if t < fa:
        a = smooth((t - sl) / (fa - sl))
        pos = end + np.array([0.0, 0.0, (S.D + 0.12) * a])
        rot = quat_to_mat(quat_axis_angle((1.0, 0.0, 0.0), 0.10 * a))
        return True, True, pos, rot, 0.0, 1.0, 1.0
    # falling and dissolving into light
    tf = t - fa
    p_out = end + np.array([0.0, 0.0, S.D + 0.12])
    g = 2.4 if k == CENTER else 3.0
    pos = p_out + np.array([FALL_DRIFT[k] * tf, -0.5 * g * tf * tf, 0.55 * tf + 0.25 * tf * tf])
    ang = 0.10 + FALL_SPIN[k] * (0.9 * tf + 0.35 * tf * tf)
    rot = quat_to_mat(quat_axis_angle(FALL_AXIS[k], ang))
    diss = smooth(ramp(t, d0, d1)) * 1.02
    alpha = 1.0 - smooth(ramp(t, d0, d1 - 0.15))
    if diss >= 1.0:
        return False, False, pos, rot, 1.0, 0.0, 1.0
    return True, True, pos, rot, diss, alpha, 1.0


def blocks(t):
    """Packed cube array for the kernels + index lists + per-cell cover/leak."""
    BLK = np.zeros((NB, 17))
    visible, moving = [], []
    cellcov = np.zeros((S.NROWS, S.NCOLS))
    leak = np.zeros((S.NROWS, S.NCOLS))
    for k in range(NB):
        vis, mov, c, R, diss, alpha, loosen = block_state(k, t)
        BLK[k, 0:3] = c
        BLK[k, 3:12] = R.ravel()
        BLK[k, 12] = S.B / 2 * 0.998
        BLK[k, 13] = diss
        BLK[k, 14] = SEED[k]
        BLK[k, 15] = WORD[k]
        BLK[k, 16] = alpha
        r, cc = CELLS[k][0], CELLS[k][1]
        if vis:
            visible.append(k)
            if mov:
                moving.append(k)
            else:
                cellcov[r, cc] = 1.0
                leak[r, cc] = 0.035 * loosen
    return BLK, np.array(visible, np.int64), np.array(moving, np.int64), cellcov, leak


# cover texture (40 px/m) helpers
_CVH = int(round(S.NROWS * S.B * 40))
_CVW = 400
_cu = -5.0 + (np.arange(_CVW) + 0.5) / 40.0
_cv = S.NROWS * S.B - (np.arange(_CVH) + 0.5) / 40.0
_CU, _CVV = np.meshgrid(_cu, _cv)
_COL = np.clip(np.floor((_CU + 5.0) / S.B).astype(int), 0, S.NCOLS - 1)
_ROW = np.clip(np.floor(_CVV / S.B).astype(int), 0, S.NROWS - 1)
_EDGE = np.minimum(np.minimum(_CU - (-5.0 + _COL * S.B), (-5.0 + (_COL + 1) * S.B) - _CU),
                   np.minimum(_CVV - _ROW * S.B, (_ROW + 1) * S.B - _CVV))


def cover_texture(cellcov, leak):
    cov = cellcov[_ROW, _COL]
    lk = leak[_ROW, _COL]
    f = np.where(lk > 0, np.clip((_EDGE - lk + 0.0125) / 0.025, 0, 1), 1.0)
    return (cov * f).astype(np.float32)


# ------------------------------------------------------------ particles --
NPART = 1400


def _particle_setup():
    from scene import vnoise3
    prng = np.random.default_rng(99)
    P = {}
    for k in range(NB):
        h = S.B / 2
        face = prng.integers(0, 6, NPART)
        loc = prng.uniform(-h, h, (NPART, 3))
        ax = face // 2
        sg = np.where(face % 2 == 0, -1.0, 1.0)
        loc[np.arange(NPART), ax] = sg * h
        # the dissolve threshold at that surface point (same noise as the shader)
        thr = np.array([0.55 * vnoise3(l[0] * 4.2 + 7.0, l[1] * 4.2, l[2] * 4.2, int(SEED[k]))
                        + 0.45 * vnoise3(l[0] * 13.0, l[1] * 13.0 + 3.0, l[2] * 13.0, int(SEED[k]) + 1) for l in loc])
        vel = prng.normal(0, 0.22, (NPART, 3)) + np.array([0.0, 0.18, 0.05])
        life = prng.uniform(1.3, 2.9, NPART)
        size = prng.uniform(0.010, 0.026, NPART)
        phase = prng.uniform(0, 2 * math.pi, NPART)
        warm = prng.uniform(0, 1, NPART)
        P[k] = (loc, thr, vel, life, size, phase, warm)
    return P


_PART = None


def particles(t):
    """Light particles released by dissolving cubes: positions, colour weights, sizes."""
    global _PART
    if _PART is None:
        _PART = _particle_setup()
    pts, col, rad = [], [], []
    for k in range(NB):
        lo, sl, fa, d0, d1 = removal_timing(k)
        if t < d0 or t > d1 + 3.2:
            continue
        loc, thr, vel, life, size, phase, warm = _PART[k]
        # time at which dissolve(t) reaches each threshold (inverse of smooth ramp)
        x = np.clip(thr / 1.02, 0, 1)
        # invert smoothstep numerically
        s = x.copy()
        for _ in range(6):
            f = s * s * (3 - 2 * s) - x
            fp = 6 * s * (1 - s) + 1e-6
            s = np.clip(s - f / fp, 0, 1)
        t_rel = d0 + s * (d1 - d0)
        age = t - t_rel
        alive = (age > 0) & (age < life)
        if not alive.any():
            continue
        idx = np.nonzero(alive)[0]
        out = np.zeros((len(idx), 3))
        for n, i in enumerate(idx):
            vis, mov, c, R, diss, alpha, lo_ = block_state(k, t_rel[i])
            out[n] = c + R @ loc[i]
        a = age[idx]
        tau = 0.9
        drift = vel[idx] * (tau * (1 - np.exp(-a / tau)))[:, None]
        drift[:, 1] += 0.10 * a * a
        swirl = 0.08 * np.stack([np.sin(a * 2.1 + phase[idx]), np.sin(a * 1.7 + 2 * phase[idx]),
                                 np.cos(a * 1.9 + phase[idx])], 1) * a[:, None]
        p = out + drift + swirl
        fade = (1 - a / life[idx]) ** 1.6 * np.minimum(1.0, a / 0.08)
        flick = 0.75 + 0.25 * np.sin(a * 13.0 + phase[idx] * 3)
        w = warm[idx]
        cc = np.stack([np.ones_like(w), 0.80 + 0.12 * w, 0.52 + 0.3 * w], 1) * (fade * flick)[:, None]
        pts.append(p)
        col.append(cc)
        rad.append(size[idx])
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(pts), np.concatenate(col), np.concatenate(rad)


# ------------------------------------------------------------------ dust --
NDUST = 26000
_drng = np.random.default_rng(7)
_du = _drng.uniform(-5.4, 5.4, NDUST)
_dv = _drng.uniform(-0.3, 15.3, NDUST)
_ds = _drng.uniform(0.0, 1.0, NDUST)
_DVEL = _drng.normal(0, 0.035, (NDUST, 3)) + np.array([0.0, 0.012, 0.0])
_DPH = _drng.uniform(0, 2 * math.pi, (NDUST, 3))
_DFREQ = _drng.uniform(0.6, 2.4, NDUST)
_DSIZE = _drng.lognormal(math.log(0.006), 0.45, NDUST)
_DBRIGHT = _drng.lognormal(0.0, 0.6, NDUST)


def dust(t):
    s = _ds * (_dv + S.HS)                     # keep motes above the floor
    base = np.stack([_du, _dv + S.HS - s, s - S.D], 1)
    p = base + _DVEL * t + 0.05 * np.sin(t * 0.3 + _DPH)
    # wrap back into the beam volume
    u = p[:, 0]
    v = p[:, 1] + p[:, 2] + S.D - S.HS
    u = (u + 5.4) % 10.8 - 5.4
    v = (v + 0.3) % 15.6 - 0.3
    ss = np.clip(p[:, 2] + S.D, 0.0, None)
    ss = np.minimum(ss, v + S.HS)
    y = v + S.HS - ss
    p = np.stack([u, np.maximum(y, 0.02), ss - S.D], 1)
    tw = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(t * _DFREQ * 2.2 + _DPH[:, 0])) ** 6
    return p, tw * _DBRIGHT, _DSIZE


# ------------------------------------------------------------ light & grade --


def params(t):
    """Scalar parameters for the renderer at time t."""
    P = np.zeros(S.NPARAM)
    reveal = 1.0 if t >= 12.5 else 0.0
    P[S.P_REVEAL] = reveal
    fade_in = smooth(ramp(t, 0.0, 1.6))
    P[S.P_LSUN] = (5.2 if reveal < 0.5 else 4.5) * fade_in
    P[S.P_LGLASS] = (4.0 if reveal < 0.5 else 2.3) * fade_in
    P[S.P_AMB] = 0.0035
    P[S.P_AMBR], P[S.P_AMBG], P[S.P_AMBB] = 0.75, 0.9, 1.25
    P[S.P_TIME] = t
    P[S.P_FOG] = 0.024
    P[S.P_PHASEG] = 0.45
    P[S.P_BOUNCE] = 1.0
    P[S.P_WSKY] = 0.35
    P[S.P_SPILL] = 0.16
    # the first light is colourless and cold
    P[S.P_SUNR], P[S.P_SUNG], P[S.P_SUNB] = 0.93, 0.97, 1.05
    return P


_CELL_W = None


def light_level(t):
    """Fraction of the window's light that reaches the room (1 = all open)."""
    global _CELL_W
    if _CELL_W is None:
        # share of the window area per cell
        w = np.zeros((S.NROWS, S.NCOLS))
        for (r, c, u, v) in CELLS:
            n = 0
            for j in range(10):
                for i in range(10):
                    if S.arch_inside(u - S.B / 2 + (i + 0.5) / 10 * S.B, v - S.B / 2 + (j + 0.5) / 10 * S.B):
                        n += 1
            w[r, c] = n
        _CELL_W = w / w.sum()
    _, _, _, cellcov, _ = blocks(t)
    return float(((1.0 - cellcov) * _CELL_W).sum())


def adaptation(t):
    """Eye adaptation: after the light is gone the eye slowly opens up."""
    if t < 10.0 or t > 27.0:
        return 1.0
    acc, wsum = 0.0, 0.0
    for k in range(12):
        tk = t - 0.22 * k
        wk = math.exp(-k * 0.22 / 1.1)
        acc += wk * light_level(tk)
        wsum += wk
    lv = acc / wsum
    return 1.0 + 2.1 * smooth((0.10 - lv) / 0.10)


def grade(t):
    """Exposure / colour-grade controls for post-processing."""
    cold = 1.0 - smooth(ramp(t, 14.5, 25.0))
    g = dict(
        exposure=adaptation(t),
        bloom=0.10 + 0.08 * (1 - smooth(ramp(t, 10.0, 12.0))),
        saturation=lerp(1.10, 0.62, cold),
        tint=(lerp(1.015, 0.93, cold), lerp(1.0, 0.98, cold), lerp(0.975, 1.08, cold)),
        lift=(lerp(0.004, 0.0, cold), lerp(0.003, 0.002, cold), lerp(0.002, 0.006, cold)),
        vignette=0.30,
        white=0.0,
    )
    if t >= 32.0:
        # the light swells until it envelops the whole frame
        a = smooth(ramp(t, 33.1, 35.0))
        g['exposure'] = lerp(1.0, 7.5, a ** 1.4)
        g['bloom'] = lerp(0.12, 0.55, a)
        w = smooth(ramp(t, 34.0, 35.4)) * 0.72 + smooth(ramp(t, 35.2, 35.95)) * 0.22
        g['white'] = w
        g['saturation'] = lerp(g['saturation'], 0.55, smooth(ramp(t, 33.8, 35.2)))
        g['vignette'] = lerp(0.30, 0.0, a)
    return g
