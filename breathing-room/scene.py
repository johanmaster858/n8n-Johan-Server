"""Cathedral scene: geometry, materials and numba ray-casting kernels.

World units are metres.  x -> right, y -> up, z -> towards the back of the nave.
The inner face of the front wall is the plane z = 0.  The stained glass sits at
the back of a window recess, in the plane z = -D; the word cubes fill that
recess.  Sunlight travels along SUN = (0, -sin45, cos45): it comes in through
the window at a 45 degree diagonal and falls onto the floor of the nave.

Every lit point is shaded by projecting it back along the sun direction onto
the glass plane and sampling the stained-glass texture there, multiplied by
the "cover" map (cubes sitting in the window) and a depth map of moving cubes.
The window as seen from inside, the coloured light columns and the image on
the floor therefore all come from the same design.
"""
import math

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------- constants --
B = 10.0 / 7.0            # cube edge = window width / 7 columns
D = B                     # depth of the window recess (cubes sit inside it)
HS = 3.2                  # height of the bottom of the glass above the floor
VS = 15.0 - math.sqrt(75.0)   # springing line of the pointed arch (window coords)
XW = 9.5                  # side walls at x = +-XW
HC = 27.0                 # vault height
ZB = 50.0                 # back wall
PILLAR_X = 7.9
PILLAR_R = 0.72
PILLAR_Z = np.array([5.0, 11.0, 17.0, 23.0, 29.0, 35.0, 41.0, 47.0])
S2 = 0.7071067811865476
LX, LY, LZ = 0.0, S2, -S2          # unit vector pointing towards the sun
NCOLS, NROWS = 7, 11

# material ids
M_NONE, M_FLOOR, M_WALL, M_SIDE, M_PILLAR, M_RECESS, M_GLASS, M_BLOCK, M_CEIL = 0, 1, 2, 3, 4, 5, 6, 7, 8

# indices into the per-frame parameter vector P
P_REVEAL = 0        # 0 = white glass, 1 = stained glass revealed
P_LSUN = 1          # sunlight radiance
P_LGLASS = 2        # radiance of the glass as seen from inside
P_AMB = 3           # ambient level
P_TIME = 4
P_PIXANG = 5        # angular size of one pixel (for mip selection)
P_DUSTK = 6
P_FOG = 7           # volumetric density
P_PHASEG = 8
P_BOUNCE = 9
P_WSKY = 10         # window sky-light strength
P_AMBR, P_AMBG, P_AMBB = 11, 12, 13
P_SUNR, P_SUNG, P_SUNB = 14, 15, 16
P_WHITE_SOFT = 17
P_SPILL = 18
P_FILL = 19
NPARAM = 24


# ------------------------------------------------------------------ helpers --
@njit(inline='always', fastmath=True, cache=True)
def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@njit(inline='always', fastmath=True, cache=True)
def smoothstep(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


@njit(inline='always', cache=True)
def hash3(ix, iy, iz, seed):
    h = (ix * 374761393 + iy * 668265263 + iz * 1274126177 + seed * 144665 + 1013904223) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF) / 65535.0


@njit(fastmath=True, cache=True)
def vnoise3(x, y, z, seed):
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    ix = int(ix)
    iy = int(iy)
    iz = int(iz)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)
    a = hash3(ix, iy, iz, seed)
    b = hash3(ix + 1, iy, iz, seed)
    c = hash3(ix, iy + 1, iz, seed)
    d = hash3(ix + 1, iy + 1, iz, seed)
    e = hash3(ix, iy, iz + 1, seed)
    f = hash3(ix + 1, iy, iz + 1, seed)
    g = hash3(ix, iy + 1, iz + 1, seed)
    h = hash3(ix + 1, iy + 1, iz + 1, seed)
    x1 = a + (b - a) * ux
    x2 = c + (d - c) * ux
    x3 = e + (f - e) * ux
    x4 = g + (h - g) * ux
    y1 = x1 + (x2 - x1) * uy
    y2 = x3 + (x4 - x3) * uy
    return y1 + (y2 - y1) * uz


@njit(inline='always', fastmath=True, cache=True)
def bilin3(tex, x, y):
    """Bilinear sample of an (H, W, C>=3) texture at texel coords, zero outside."""
    h = tex.shape[0]
    w = tex.shape[1]
    x -= 0.5
    y -= 0.5
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    r = 0.0
    g = 0.0
    b = 0.0
    for j in range(2):
        yy = y0 + j
        if yy < 0 or yy >= h:
            continue
        wy = fy if j == 1 else 1.0 - fy
        for i in range(2):
            xx = x0 + i
            if xx < 0 or xx >= w:
                continue
            wgt = wy * (fx if i == 1 else 1.0 - fx)
            r += wgt * tex[yy, xx, 0]
            g += wgt * tex[yy, xx, 1]
            b += wgt * tex[yy, xx, 2]
    return r, g, b


@njit(inline='always', fastmath=True, cache=True)
def bilin_ch(tex, x, y, ch):
    h = tex.shape[0]
    w = tex.shape[1]
    x -= 0.5
    y -= 0.5
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    acc = 0.0
    for j in range(2):
        yy = y0 + j
        if yy < 0 or yy >= h:
            continue
        wy = fy if j == 1 else 1.0 - fy
        for i in range(2):
            xx = x0 + i
            if xx < 0 or xx >= w:
                continue
            acc += wy * (fx if i == 1 else 1.0 - fx) * tex[yy, xx, ch]
    return acc


@njit(inline='always', fastmath=True, cache=True)
def bilin2d(tex, x, y):
    h = tex.shape[0]
    w = tex.shape[1]
    x -= 0.5
    y -= 0.5
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    acc = 0.0
    for j in range(2):
        yy = y0 + j
        if yy < 0 or yy >= h:
            continue
        wy = fy if j == 1 else 1.0 - fy
        for i in range(2):
            xx = x0 + i
            if xx < 0 or xx >= w:
                continue
            acc += wy * (fx if i == 1 else 1.0 - fx) * tex[yy, xx]
    return acc


@njit(inline='always', fastmath=True, cache=True)
def wrap_bilin3(tex, x, y):
    """Bilinear sample with wrap-around (tiling textures)."""
    h = tex.shape[0]
    w = tex.shape[1]
    x -= 0.5
    y -= 0.5
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    x0 %= w
    y0 %= h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    r = ((tex[y0, x0, 0] * (1 - fx) + tex[y0, x1, 0] * fx) * (1 - fy)
         + (tex[y1, x0, 0] * (1 - fx) + tex[y1, x1, 0] * fx) * fy)
    g = ((tex[y0, x0, 1] * (1 - fx) + tex[y0, x1, 1] * fx) * (1 - fy)
         + (tex[y1, x0, 1] * (1 - fx) + tex[y1, x1, 1] * fx) * fy)
    b = ((tex[y0, x0, 2] * (1 - fx) + tex[y0, x1, 2] * fx) * (1 - fy)
         + (tex[y1, x0, 2] * (1 - fx) + tex[y1, x1, 2] * fx) * fy)
    return r, g, b


# ------------------------------------------------------- window geometry --
@njit(inline='always', fastmath=True, cache=True)
def arch_inside(u, v):
    if v < 0.0 or u <= -5.0 or u >= 5.0:
        return False
    if v <= VS:
        return True
    return ((u - 5.0) ** 2 + (v - VS) ** 2 < 100.0) and ((u + 5.0) ** 2 + (v - VS) ** 2 < 100.0)


@njit(inline='always', fastmath=True, cache=True)
def recess_inside(x, y, z):
    """Cross-section of the recess at depth z in [-D, 0] (splayed sill)."""
    if x <= -5.0 or x >= 5.0:
        return False
    v = y - HS
    if v <= -D - z:
        return False
    if v <= VS:
        return True
    return ((x - 5.0) ** 2 + (v - VS) ** 2 < 100.0) and ((x + 5.0) ** 2 + (v - VS) ** 2 < 100.0)


@njit(inline='always', fastmath=True, cache=True)
def to_window(px, py, pz):
    """Project a point back along the sun direction onto the glass plane."""
    s = pz + D
    return px, py + s - HS, s


# --------------------------------------------------------- light lookup --
@njit(fastmath=True, cache=True)
def cover_at(CV, u, v):
    # cover texture: 40 px/m, u in [-5, 5], v in [0, NROWS*B]
    return bilin2d(CV, (u + 5.0) * 40.0, (NROWS * B - v) * 40.0)


@njit(fastmath=True, cache=True)
def dyn_vis(DD, DA, DI, u, v, s, exclude):
    """Visibility through moving cubes (30 px/m map over u in [-6,6], v in [-2,17])."""
    x = (u + 6.0) * 30.0 - 0.5
    y = (17.0 - v) * 30.0 - 0.5
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    h = DD.shape[0]
    w = DD.shape[1]
    vis = 0.0
    for j in range(2):
        yy = y0 + j
        wy = fy if j == 1 else 1.0 - fy
        for i in range(2):
            xx = x0 + i
            wgt = wy * (fx if i == 1 else 1.0 - fx)
            if yy < 0 or yy >= h or xx < 0 or xx >= w:
                vis += wgt
                continue
            if DI[yy, xx] >= 0 and DI[yy, xx] != exclude and s > DD[yy, xx] + 0.02:
                vis += wgt * (1.0 - DA[yy, xx])
            else:
                vis += wgt
    return vis


@njit(fastmath=True, cache=True)
def sun_at(P, F, CV, DD, DA, DI, px, py, pz, exclude):
    """Colour of sunlight arriving at a point (before the cosine term)."""
    u, v, s = to_window(px, py, pz)
    if s < 0.0 or v < -0.05 or v > 15.05 or u < -5.05 or u > 5.05:
        return 0.0, 0.0, 0.0
    fx = (u + 5.0) * 80.0
    fy = (15.0 - v) * 80.0
    r, g, b = bilin3(F, fx, fy)
    wht = bilin_ch(F, fx, fy, 3)
    ws = clamp01((s - 3.0) / 14.0) * 0.85
    if ws > 0.0:
        r2 = bilin_ch(F, fx, fy, 4)
        g2 = bilin_ch(F, fx, fy, 5)
        b2 = bilin_ch(F, fx, fy, 6)
        r = r * (1.0 - ws) + r2 * ws
        g = g * (1.0 - ws) + g2 * ws
        b = b * (1.0 - ws) + b2 * ws
    rv = P[P_REVEAL]
    r = r * rv + wht * (1.0 - rv) * P[P_SUNR]
    g = g * rv + wht * (1.0 - rv) * P[P_SUNG]
    b = b * rv + wht * (1.0 - rv) * P[P_SUNB]
    c = 1.0 - cover_at(CV, u, v)
    if c <= 0.0:
        return 0.0, 0.0, 0.0
    c *= dyn_vis(DD, DA, DI, u, v, s, exclude)
    k = P[P_LSUN] * c
    return r * k, g * k, b * k


@njit(fastmath=True, cache=True)
def indirect(P, VPL, WVPL, px, py, pz, nx, ny, nz):
    """Diffuse bounce from the lit floor patch + sky light through the window."""
    er = 0.0
    eg = 0.0
    eb = 0.0
    for k in range(VPL.shape[0]):
        wx = VPL[k, 0] - px
        wy = VPL[k, 1] - py
        wz = VPL[k, 2] - pz
        r2 = wx * wx + wy * wy + wz * wz
        inv = 1.0 / math.sqrt(r2 + 1e-9)
        cr = (nx * wx + ny * wy + nz * wz) * inv
        ce = -wy * inv
        if cr <= 0.0 or ce <= 0.0:
            continue
        f = cr * ce / max(r2, 2.5)
        er += VPL[k, 3] * f
        eg += VPL[k, 4] * f
        eb += VPL[k, 5] * f
    for k in range(WVPL.shape[0]):
        wx = WVPL[k, 0] - px
        wy = WVPL[k, 1] - py
        wz = WVPL[k, 2] - pz
        r2 = wx * wx + wy * wy + wz * wz
        inv = 1.0 / math.sqrt(r2 + 1e-9)
        cr = (nx * wx + ny * wy + nz * wz) * inv
        ce = -wz * inv
        if cr <= 0.0 or ce <= 0.0:
            continue
        f = cr * ce / max(r2, 4.0)
        er += WVPL[k, 3] * f
        eg += WVPL[k, 4] * f
        eb += WVPL[k, 5] * f
    k = P[P_BOUNCE]
    return (er * k + P[P_AMB] * P[P_AMBR], eg * k + P[P_AMB] * P[P_AMBG], eb * k + P[P_AMB] * P[P_AMBB])


# ---------------------------------------------------------- stone albedo --
@njit(inline='always', fastmath=True, cache=True)
def wrap_bilin1(tex, x, y):
    h = tex.shape[0]
    w = tex.shape[1]
    x -= 0.5
    y -= 0.5
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    x0 %= w
    y0 %= h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    return ((tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy)
            + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy)


@njit(inline='always', fastmath=True, cache=True)
def joint_cover(dist, jw, fw):
    """Anti-aliased coverage of a joint of half-width jw at distance dist."""
    if fw > jw:
        return (jw / fw) * smoothstep(2.0 * fw, 0.0, dist)
    return smoothstep(jw + fw, jw - fw, dist)


@njit(fastmath=True, cache=True)
def floor_albedo(NT, x, z, fw):
    sz = 1.25
    gx = x / sz + 101.5
    gz = z / sz + 0.0
    ix = math.floor(gx)
    iz = math.floor(gz)
    fx = gx - ix
    fz = gz - iz
    dj = min(min(fx, 1.0 - fx), min(fz, 1.0 - fz)) * sz
    jc = joint_cover(dj, 0.007, fw)
    ii = int(ix)
    jj = int(iz)
    tone = 0.80 + 0.36 * hash3(ii, jj, 0, 7)
    warm = hash3(ii, jj, 1, 9) - 0.5
    n = wrap_bilin1(NT, x * 85.0, z * 85.0)
    n2 = wrap_bilin1(NT, x * 11.0 + 37.0, z * 11.0 + 91.0)
    k = tone * (0.84 + 0.26 * n) * (0.88 + 0.24 * n2) * (1.0 - 0.6 * jc)
    return 0.47 * k * (1.0 + 0.06 * warm), 0.44 * k, 0.40 * k * (1.0 - 0.06 * warm)


@njit(fastmath=True, cache=True)
def wall_albedo(NT, a, y, fw, seed):
    ch = 0.78
    gy = y / ch
    row = math.floor(gy)
    fy = gy - row
    r = int(row)
    L = 1.45 + 0.7 * hash3(r, 3, seed, 5)
    gx = (a + 13.0 * hash3(r, 1, seed, 3)) / L
    col = math.floor(gx)
    fx = gx - col
    dj = min(min(fx, 1.0 - fx) * L, min(fy, 1.0 - fy) * ch)
    jc = joint_cover(dj, 0.008, fw)
    c = int(col)
    tone = 0.88 + 0.20 * hash3(c, r, seed, 11)
    warm = hash3(c, r, seed, 13) - 0.5
    n = wrap_bilin1(NT, a * 60.0 + seed * 17.0, y * 60.0)
    n2 = wrap_bilin1(NT, a * 5.0 + 51.0, y * 2.0 + seed * 3.0)
    k = tone * (0.86 + 0.22 * n) * (0.80 + 0.40 * n2) * (1.0 - 0.35 * jc)
    return 0.40 * k * (1.0 + 0.05 * warm), 0.375 * k, 0.345 * k * (1.0 - 0.05 * warm)


# ------------------------------------------------------------ cube hit --
@njit(fastmath=True, cache=True)
def hit_block(BLK, i, ox, oy, oz, dx, dy, dz, tmax):
    """Ray vs. oriented cube with dissolve holes. Returns t, face axis, sign, local hit."""
    cx = BLK[i, 0]
    cy = BLK[i, 1]
    cz = BLK[i, 2]
    h = BLK[i, 12]
    # local frame: columns of R are the local axes in world space
    rx = ox - cx
    ry = oy - cy
    rz = oz - cz
    lo0 = BLK[i, 3] * rx + BLK[i, 6] * ry + BLK[i, 9] * rz
    lo1 = BLK[i, 4] * rx + BLK[i, 7] * ry + BLK[i, 10] * rz
    lo2 = BLK[i, 5] * rx + BLK[i, 8] * ry + BLK[i, 11] * rz
    ld0 = BLK[i, 3] * dx + BLK[i, 6] * dy + BLK[i, 9] * dz
    ld1 = BLK[i, 4] * dx + BLK[i, 7] * dy + BLK[i, 10] * dz
    ld2 = BLK[i, 5] * dx + BLK[i, 8] * dy + BLK[i, 11] * dz
    tn = -1e30
    tf = 1e30
    an = -1
    sn = 0.0
    af = -1
    sf = 0.0
    for a in range(3):
        o = lo0 if a == 0 else (lo1 if a == 1 else lo2)
        d = ld0 if a == 0 else (ld1 if a == 1 else ld2)
        if abs(d) < 1e-12:
            if o < -h or o > h:
                return -1.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0
            continue
        t1 = (-h - o) / d
        t2 = (h - o) / d
        s1 = -1.0
        s2 = 1.0
        if t1 > t2:
            t1, t2 = t2, t1
            s1, s2 = s2, s1
        if t1 > tn:
            tn = t1
            an = a
            sn = s1
        if t2 < tf:
            tf = t2
            af = a
            sf = s2
    if tn > tf or tf < 1e-4 or tn > tmax:
        return -1.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0
    diss = BLK[i, 13]
    seed = int(BLK[i, 14])
    # entry face
    if tn > 1e-4:
        lx = lo0 + ld0 * tn
        ly = lo1 + ld1 * tn
        lz = lo2 + ld2 * tn
        if diss <= 0.0:
            return tn, an, sn, lx, ly, lz, 0.0
        n = 0.55 * vnoise3(lx * 4.2 + 7.0, ly * 4.2, lz * 4.2, seed) + 0.45 * vnoise3(lx * 13.0, ly * 13.0 + 3.0, lz * 13.0, seed + 1)
        if n > diss:
            return tn, an, sn, lx, ly, lz, n - diss
    # exit face seen from inside (through a dissolved hole)
    if tf < tmax:
        lx = lo0 + ld0 * tf
        ly = lo1 + ld1 * tf
        lz = lo2 + ld2 * tf
        n = 0.55 * vnoise3(lx * 4.2 + 7.0, ly * 4.2, lz * 4.2, seed) + 0.45 * vnoise3(lx * 13.0, ly * 13.0 + 3.0, lz * 13.0, seed + 1)
        if diss <= 0.0 or n > diss:
            return tf, af, -sf, lx, ly, lz, n - diss if diss > 0.0 else 0.0
    return -1.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0


# ------------------------------------------------------------ primary ray --
@njit(fastmath=True, cache=True)
def trace(ox, oy, oz, dx, dy, dz, BLK, cand, ncand):
    """Returns (t, material, nx, ny, nz, index, a, b, c, d) of the nearest hit."""
    tb = 1e9
    mat = M_NONE
    nx = 0.0
    ny = 0.0
    nz = 0.0
    idx = -1
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    a4 = 0.0
    # floor
    if dy < -1e-9:
        t = -oy / dy
        if t > 1e-5 and t < tb:
            hz = oz + t * dz
            hx = ox + t * dx
            if hz >= 0.0 and hz <= ZB and hx > -XW and hx < XW:
                tb = t
                mat = M_FLOOR
                nx, ny, nz = 0.0, 1.0, 0.0
    # vault
    if dy > 1e-9:
        t = (HC - oy) / dy
        if t > 1e-5 and t < tb:
            tb = t
            mat = M_CEIL
            nx, ny, nz = 0.0, -1.0, 0.0
    # side walls
    if dx > 1e-9:
        t = (XW - ox) / dx
        if t > 1e-5 and t < tb:
            tb = t
            mat = M_SIDE
            nx, ny, nz = -1.0, 0.0, 0.0
    elif dx < -1e-9:
        t = (-XW - ox) / dx
        if t > 1e-5 and t < tb:
            tb = t
            mat = M_SIDE
            nx, ny, nz = 1.0, 0.0, 0.0
    # back wall
    if dz > 1e-9:
        t = (ZB - oz) / dz
        if t > 1e-5 and t < tb:
            tb = t
            mat = M_SIDE
            nx, ny, nz = 0.0, 0.0, -1.0
    # pillars
    for k in range(PILLAR_Z.shape[0]):
        for sgn in range(2):
            pxc = PILLAR_X if sgn == 0 else -PILLAR_X
            pzc = PILLAR_Z[k]
            fx = ox - pxc
            fz = oz - pzc
            A = dx * dx + dz * dz
            if A < 1e-12:
                continue
            Bq = 2.0 * (fx * dx + fz * dz)
            Cq = fx * fx + fz * fz - PILLAR_R * PILLAR_R
            disc = Bq * Bq - 4.0 * A * Cq
            if disc < 0.0:
                continue
            sq = math.sqrt(disc)
            t = (-Bq - sq) / (2.0 * A)
            if t > 1e-5 and t < tb:
                hy = oy + t * dy
                if hy > 0.0 and hy < HC:
                    tb = t
                    mat = M_PILLAR
                    nx = (fx + t * dx) / PILLAR_R
                    ny = 0.0
                    nz = (fz + t * dz) / PILLAR_R
                    idx = k * 2 + sgn
    # front wall and the window recess
    if dz < -1e-9:
        t0 = -oz / dz
        if t0 > 1e-5 and t0 < tb:
            hx = ox + t0 * dx
            hy = oy + t0 * dy
            if recess_inside(hx, hy, 0.0):
                # through the opening: glass or recess surfaces
                tg = (-D - oz) / dz
                gx = ox + tg * dx
                gy = oy + tg * dy
                if arch_inside(gx, gy - HS):
                    tb = tg
                    mat = M_GLASS
                    nx, ny, nz = 0.0, 0.0, 1.0
                    a1 = gx
                    a2 = gy - HS
                else:
                    lo = t0
                    hi = tg
                    for _ in range(18):
                        mid = 0.5 * (lo + hi)
                        if recess_inside(ox + mid * dx, oy + mid * dy, oz + mid * dz):
                            lo = mid
                        else:
                            hi = mid
                    t = 0.5 * (lo + hi)
                    px_ = ox + t * dx
                    py_ = oy + t * dy
                    pz_ = oz + t * dz
                    v = py_ - HS
                    # which surface?
                    d_side = 5.0 - abs(px_)
                    d_sill = v + D + pz_
                    d_arc = 1e9
                    if v > VS:
                        d1 = 10.0 - math.sqrt((px_ - 5.0) ** 2 + (v - VS) ** 2)
                        d2 = 10.0 - math.sqrt((px_ + 5.0) ** 2 + (v - VS) ** 2)
                        d_arc = min(d1, d2)
                        d_side = 1e9
                    tb = t
                    mat = M_RECESS
                    if d_sill <= d_side and d_sill <= d_arc:
                        nx, ny, nz = 0.0, S2, S2
                    elif d_side <= d_arc:
                        nx = 1.0 if px_ < 0.0 else -1.0
                        ny = 0.0
                        nz = 0.0
                    else:
                        cxa = 5.0 if px_ < 0.0 else -5.0
                        ex = cxa - px_
                        ey = VS - v
                        ln = math.sqrt(ex * ex + ey * ey) + 1e-9
                        nx = ex / ln
                        ny = ey / ln
                        nz = 0.0
            else:
                hxw = ox + t0 * dx
                hyw = oy + t0 * dy
                if hxw > -XW and hxw < XW and hyw > 0.0 and hyw < HC:
                    tb = t0
                    mat = M_WALL
                    nx, ny, nz = 0.0, 0.0, 1.0
    # cubes
    for c in range(ncand):
        i = cand[c]
        t, ax, sg, lx, ly, lz, edge = hit_block(BLK, i, ox, oy, oz, dx, dy, dz, tb)
        if t > 0.0 and t < tb:
            tb = t
            mat = M_BLOCK
            idx = i
            # world normal = R[:, ax] * sg
            nx = BLK[i, 3 + ax] * sg
            ny = BLK[i, 6 + ax] * sg
            nz = BLK[i, 9 + ax] * sg
            a1 = lx
            a2 = ly
            a3 = lz
            a4 = float(ax)
    return tb, mat, nx, ny, nz, idx, a1, a2, a3, a4


@njit(inline='always', fastmath=True, cache=True)
def word_sample(WM, WS, w, tu, tv):
    n = WM.shape[2]
    mx = tu * n - 0.5
    my = tv * n - 0.5
    ix = int(math.floor(mx))
    iy = int(math.floor(my))
    if ix < 0 or iy < 0 or ix >= n - 1 or iy >= n - 1:
        return 0.0, 1.0
    fx = mx - ix
    fy = my - iy
    m = ((WM[w, iy, ix] * (1 - fx) + WM[w, iy, ix + 1] * fx) * (1 - fy)
         + (WM[w, iy + 1, ix] * (1 - fx) + WM[w, iy + 1, ix + 1] * fx) * fy)
    sh = ((WS[w, iy, ix] * (1 - fx) + WS[w, iy, ix + 1] * fx) * (1 - fy)
          + (WS[w, iy + 1, ix] * (1 - fx) + WS[w, iy + 1, ix + 1] * fx) * fy)
    return m, sh


@njit(fastmath=True, cache=True)
def shade(P, ox, oy, oz, dx, dy, dz, t, mat, nx, ny, nz, idx, a1, a2, a3, a4,
          BLK, G0, G1, G2, G3, F, CV, DD, DA, DI, VPL, WVPL, SP, NT, WORDM, WORDS, WORDM1, WORDS1, BNOISE):
    if mat == M_NONE:
        return 0.0, 0.0, 0.0
    px = ox + t * dx
    py = oy + t * dy
    pz = oz + t * dz
    if mat == M_GLASS:
        # mip level from the pixel footprint on the glass
        foot = t * P[P_PIXANG] / max(abs(dz), 0.2)
        lod = math.log2(max(foot * 160.0, 1.0))
        u = a1
        v = a2
        rr = 0.0
        gg = 0.0
        bb = 0.0
        wh = 0.0
        st = 0.0
        l0 = int(math.floor(lod))
        fl = lod - l0
        for k in range(2):
            lv = l0 + k
            if lv > 3:
                lv = 3
            wgt = (1.0 - fl) if k == 0 else fl
            if wgt <= 0.0:
                continue
            ppm = 160.0 / (2.0 ** lv)
            tex = G0 if lv == 0 else (G1 if lv == 1 else (G2 if lv == 2 else G3))
            fx = (u + 5.0) * ppm
            fy = (15.0 - v) * ppm
            r_, g_, b_ = bilin3(tex, fx, fy)
            rr += wgt * r_
            gg += wgt * g_
            bb += wgt * b_
            wh += wgt * bilin_ch(tex, fx, fy, 3)
            st += wgt * bilin_ch(tex, fx, fy, 4)
        rv = P[P_REVEAL]
        L = P[P_LGLASS]
        er, eg, eb = indirect(P, VPL, WVPL, px, py, pz + 0.05, 0.0, 0.0, 1.0)
        sa = 0.30 * st
        cr = (rr * rv + wh * (1.0 - rv) * P[P_SUNR]) * L + sa * er
        cg = (gg * rv + wh * (1.0 - rv) * P[P_SUNG]) * L + sa * eg
        cb = (bb * rv + wh * (1.0 - rv) * P[P_SUNB]) * L + sa * eb
        return cr, cg, cb
    # albedo
    ar = 0.4
    ag = 0.4
    ab = 0.4
    emr = 0.0
    emg = 0.0
    emb = 0.0
    exclude = -1
    cosi = abs(nx * dx + ny * dy + nz * dz)
    fw = t * P[P_PIXANG] / max(cosi, 0.08) * 0.7
    if mat == M_FLOOR:
        ar, ag, ab = floor_albedo(NT, px, pz, fw)
    elif mat == M_WALL:
        ar, ag, ab = wall_albedo(NT, px + 40.0, py, fw, 1)
    elif mat == M_SIDE:
        ar, ag, ab = wall_albedo(NT, pz + px, py, fw, 2)
        ar *= 0.85
        ag *= 0.85
        ab *= 0.85
    elif mat == M_CEIL:
        ar, ag, ab = 0.22, 0.21, 0.20
    elif mat == M_PILLAR:
        ang = math.atan2(nz, nx)
        ar, ag, ab = wall_albedo(NT, ang * PILLAR_R + idx * 2.7, py, fw, 3 + idx)
        # clustered shafts of a Gothic pier: perturb the normal around the axis
        fl = math.sin(ang * 12.0)
        tx_ = -nz
        tz_ = nx
        nx = nx + 0.45 * fl * tx_
        nz = nz + 0.45 * fl * tz_
        ln = math.sqrt(nx * nx + nz * nz)
        nx /= ln
        nz /= ln
        g_ = 0.82 + 0.18 * (0.5 + 0.5 * math.cos(ang * 12.0))
        ar *= g_
        ag *= g_
        ab *= g_
    elif mat == M_RECESS:
        ar, ag, ab = wall_albedo(NT, px + pz * 0.7 + 11.0, py + pz * 0.7, fw, 4)
    elif mat == M_BLOCK:
        exclude = idx
        h = BLK[idx, 12]
        ax = int(a4 + 0.5)
        lx = a1
        ly = a2
        lz = a3
        lc = lx if ax == 0 else (ly if ax == 1 else lz)
        sgf = 1.0 if lc > 0.0 else -1.0
        if ax == 2:
            fu = sgf * lx
            fv = -ly
        elif ax == 0:
            fu = -sgf * lz
            fv = -ly
        else:
            fu = lx
            fv = sgf * lz
        tu = (fu / h + 1.0) * 0.5
        tv = (fv / h + 1.0) * 0.5
        # concrete-like noise
        nb = BNOISE[int(clamp01(tv) * 255.0), int(clamp01(tu) * 255.0)]
        base = 0.30 * (0.88 + 0.24 * nb)
        ar = base * 0.97
        ag = base * 0.99
        ab = base * 1.03
        w = int(BLK[idx, 15])
        if w >= 0 and ax == 2:        # engraved on the front and back faces only
            texpp = fw / (2.0 * h) * WORDM.shape[2]
            lod = math.log2(max(texpp, 1.0)) * 0.5
            wl1 = clamp01(lod)
            m0, sh0 = word_sample(WORDM, WORDS, w, tu, tv)
            m = m0
            sh = sh0
            if wl1 > 0.0:
                m1, sh1 = word_sample(WORDM1, WORDS1, w, tu, tv)
                m = m0 * (1.0 - wl1) + m1 * wl1
                sh = sh0 * (1.0 - wl1) + sh1 * wl1
            if m > 0.0:
                lw = 0.86 * sh
                ar = ar * (1.0 - m) + lw * m
                ag = ag * (1.0 - m) + lw * m
                ab = ab * (1.0 - m) + lw * 1.02 * m
                # the white paint of the engraving catches the faintest light
                emr += m * sh * 0.0105
                emg += m * sh * 0.0110
                emb += m * sh * 0.0125
        # bevel highlight on edges
        e = min(h - abs(fu), h - abs(fv)) / h
        if e < 0.05:
            k = 1.0 + 0.6 * (1.0 - e / 0.05)
            ar *= k
            ag *= k
            ab *= k
        # dissolve edge glow
        diss = BLK[idx, 13]
        if diss > 0.0:
            n = 0.55 * vnoise3(lx * 4.2 + 7.0, ly * 4.2, lz * 4.2, int(BLK[idx, 14])) + 0.45 * vnoise3(lx * 13.0, ly * 13.0 + 3.0, lz * 13.0, int(BLK[idx, 14]) + 1)
            ed = n - diss
            if ed < 0.09:
                q = 1.0 - ed / 0.09
                gk = (q * q * q) * 2.2 * (0.35 + 0.65 * BLK[idx, 16])
                emr += gk * 1.0
                emg += gk * 0.72
                emb += gk * 0.34
    # direct sunlight through the window
    ndl = nx * LX + ny * LY + nz * LZ
    sr = 0.0
    sg = 0.0
    sb = 0.0
    if ndl > 0.0:
        sr, sg, sb = sun_at(P, F, CV, DD, DA, DI, px, py, pz, exclude)
        sr *= ndl
        sg *= ndl
        sb *= ndl
    er, eg, eb = indirect(P, VPL, WVPL, px + nx * 0.02, py + ny * 0.02, pz + nz * 0.02, nx, ny, nz)
    if mat == M_FLOOR or (mat == M_PILLAR and py < 3.0):
        # SP: blurred floor light, 8 px/m over u in [-12, 12], v in [-10, 25]
        su = px
        sv = pz + D - HS + py
        kx = (su + 12.0) * 8.0
        ky = (25.0 - sv) * 8.0
        spr, spg, spb = bilin3(SP, kx, ky)
        ks = P[P_SPILL] * (1.0 if mat == M_FLOOR else 0.5)
        # inside the sunlit patch the direct light dominates; keep its colours pure
        if mat == M_FLOOR and su > -5.0 and su < 5.0 and sv > 0.0 and sv < 15.0:
            ks *= 0.35
        er += spr * ks
        eg += spg * ks
        eb += spb * ks
    if mat == M_BLOCK:
        # soft fill from the lit nave behind the camera
        fd = nx * 0.0 + ny * 0.2873 + nz * 0.9578
        if fd > 0.0:
            fk = P[P_FILL] * (0.35 + 0.65 * fd)
            er += fk * 0.95
            eg += fk * 0.98
            eb += fk * 1.05
    cr = ar * (sr + er) + emr
    cg = ag * (sg + eg) + emg
    cb = ab * (sb + eb) + emb
    return cr, cg, cb


@njit(fastmath=True, cache=True)
def cam_ray(C, sx, sy, W, H):
    """Ray direction for pixel coords (sx, sy) with pixel centres at +0.5."""
    ndx = (2.0 * sx / W - 1.0) * C[12]
    ndy = (1.0 - 2.0 * sy / H) * C[13]
    dx = C[9] + ndx * C[3] + ndy * C[6]
    dy = C[10] + ndx * C[4] + ndy * C[7]
    dz = C[11] + ndx * C[5] + ndy * C[8]
    ln = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dx / ln, dy / ln, dz / ln


@njit(parallel=True, fastmath=True, cache=True)
def render_primary(C, P, W, H, BLK, TILE_N, TILE_I, TS, G0, G1, G2, G3, F, CV, DD, DA, DI,
                   VPL, WVPL, SP, NT, WORDM, WORDS, WORDM1, WORDS1, BNOISE, out, depth, objid):
    ox = C[0]
    oy = C[1]
    oz = C[2]
    for y in prange(H):
        ty = y // TS
        for x in range(W):
            tx = x // TS
            n = TILE_N[ty, tx]
            dx, dy, dz = cam_ray(C, x + 0.5, y + 0.5, W, H)
            t, mat, nx, ny, nz, idx, a1, a2, a3, a4 = trace(ox, oy, oz, dx, dy, dz, BLK, TILE_I[ty, tx], n)
            r, g, b = shade(P, ox, oy, oz, dx, dy, dz, t, mat, nx, ny, nz, idx, a1, a2, a3, a4,
                            BLK, G0, G1, G2, G3, F, CV, DD, DA, DI, VPL, WVPL, SP, NT, WORDM, WORDS, WORDM1, WORDS1, BNOISE)
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b
            depth[y, x] = t if mat != M_NONE else 1e4
            objid[y, x] = mat * 1000 + (idx + 1)


@njit(parallel=True, fastmath=True, cache=True)
def refine_edges(C, P, W, H, BLK, TILE_N, TILE_I, TS, G0, G1, G2, G3, F, CV, DD, DA, DI,
                 VPL, WVPL, SP, NT, WORDM, WORDS, WORDM1, WORDS1, BNOISE, out, depth, objid):
    """Supersample pixels on geometric edges (object id or depth discontinuities)."""
    ox = C[0]
    oy = C[1]
    oz = C[2]
    offx = np.array([0.125, 0.625, 0.875, 0.375])
    offy = np.array([0.375, 0.125, 0.625, 0.875])
    src = out.copy()
    for y in prange(1, H - 1):
        ty = y // TS
        for x in range(1, W - 1):
            o = objid[y, x]
            dc = depth[y, x]
            edge = False
            for j in range(-1, 2):
                for i in range(-1, 2):
                    if objid[y + j, x + i] != o:
                        edge = True
                    else:
                        dn = depth[y + j, x + i]
                        if abs(dn - dc) > 0.04 * dc + 0.05:
                            edge = True
            if not edge:
                continue
            tx = x // TS
            n = TILE_N[ty, tx]
            ar = src[y, x, 0]
            ag = src[y, x, 1]
            ab = src[y, x, 2]
            for k in range(4):
                dx, dy, dz = cam_ray(C, x + offx[k], y + offy[k], W, H)
                t, mat, nx, ny, nz, idx, a1, a2, a3, a4 = trace(ox, oy, oz, dx, dy, dz, BLK, TILE_I[ty, tx], n)
                r, g, b = shade(P, ox, oy, oz, dx, dy, dz, t, mat, nx, ny, nz, idx, a1, a2, a3, a4,
                                BLK, G0, G1, G2, G3, F, CV, DD, DA, DI, VPL, WVPL, SP, NT, WORDM, WORDS, WORDM1, WORDS1, BNOISE)
                ar += r
                ag += g
                ab += b
            out[y, x, 0] = ar / 5.0
            out[y, x, 1] = ag / 5.0
            out[y, x, 2] = ab / 5.0


# ------------------------------------------------------------ volumetrics --
@njit(fastmath=True, cache=True)
def slab_interval(ox, oy, oz, dx, dy, dz, tmax):
    """Clip a ray to the region that can receive light through the window."""
    t0 = 0.0
    t1 = tmax
    # |x| <= 5.2
    if abs(dx) > 1e-9:
        ta = (-5.2 - ox) / dx
        tb = (5.2 - ox) / dx
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
    elif abs(ox) > 5.2:
        return 1.0, 0.0
    # 0 <= y + z + D - HS <= 15.2  (window v range)
    k = dy + dz
    o = oy + oz + D - HS
    if abs(k) > 1e-9:
        ta = (-0.2 - o) / k
        tb = (15.2 - o) / k
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
    elif o < -0.2 or o > 15.2:
        return 1.0, 0.0
    # z >= -D
    if dz < -1e-9:
        t1 = min(t1, (-D - oz) / dz)
    return t0, t1


@njit(parallel=True, fastmath=True, cache=True)
def render_volume(C, P, W, H, VT, CV, DD, DA, DI, depth_full, scale, NOISE3, out, vdepth):
    """Single-scattering of sunlight through the window (half resolution)."""
    ox = C[0]
    oy = C[1]
    oz = C[2]
    g = P[P_PHASEG]
    tm = P[P_TIME]
    nsteps = 56
    sigma = P[P_FOG]
    rv = P[P_REVEAL]
    ns = NOISE3.shape[0]
    for y in prange(H):
        for x in range(W):
            sx = (x + 0.5) * scale
            sy = (y + 0.5) * scale
            dx, dy, dz = cam_ray(C, sx, sy, W * scale, H * scale)
            # nearest full-res depth in the footprint
            fy0 = int(y * scale)
            fx0 = int(x * scale)
            dmin = 1e9
            for j in range(int(scale)):
                for i in range(int(scale)):
                    yy = min(fy0 + j, depth_full.shape[0] - 1)
                    xx = min(fx0 + i, depth_full.shape[1] - 1)
                    dmin = min(dmin, depth_full[yy, xx])
            vdepth[y, x] = dmin
            t0, t1 = slab_interval(ox, oy, oz, dx, dy, dz, min(dmin, 200.0))
            ar = 0.0
            ag = 0.0
            ab = 0.0
            if t1 > t0:
                # Henyey-Greenstein phase for light travelling along -L
                cosang = dx * LX + dy * LY + dz * LZ
                gb = -0.25
                ph = 0.78 * (1.0 - g * g) / (4.0 * math.pi * (1.0 + g * g - 2.0 * g * cosang) ** 1.5) \
                    + 0.22 * (1.0 - gb * gb) / (4.0 * math.pi * (1.0 + gb * gb - 2.0 * gb * cosang) ** 1.5)
                dt = (t1 - t0) / nsteps
                jit = ((x * 0.7548776662 + y * 0.56984029) % 1.0)
                for k in range(nsteps):
                    tt = t0 + (k + jit) * dt
                    px = ox + tt * dx
                    py = oy + tt * dy
                    pz = oz + tt * dz
                    if py < 0.0:
                        continue
                    u, v, s = to_window(px, py, pz)
                    fx = (u + 5.0) * 20.0
                    fy = (15.0 - v) * 20.0
                    r, gg, b = bilin3(VT, fx, fy)
                    wh = bilin_ch(VT, fx, fy, 3)
                    r = r * rv + wh * (1.0 - rv) * P[P_SUNR]
                    gg = gg * rv + wh * (1.0 - rv) * P[P_SUNG]
                    b = b * rv + wh * (1.0 - rv) * P[P_SUNB]
                    if r + gg + b <= 1e-6:
                        continue
                    c = 1.0 - cover_at(CV, u, v)
                    if c <= 0.0:
                        continue
                    c *= dyn_vis(DD, DA, DI, u, v, s, -1)
                    # drifting dust density
                    qx = px * 0.55 + tm * 0.05 + 64.0
                    qy = py * 0.55 - tm * 0.03 + 64.0
                    qz = pz * 0.55 + 64.0
                    fqx = math.floor(qx)
                    fqy = math.floor(qy)
                    fqz = math.floor(qz)
                    wx = qx - fqx
                    wy = qy - fqy
                    wz = qz - fqz
                    ix = int(fqx) % ns
                    iy = int(fqy) % ns
                    iz = int(fqz) % ns
                    jx = (ix + 1) % ns
                    jy = (iy + 1) % ns
                    jz = (iz + 1) % ns
                    c00 = NOISE3[iz, iy, ix] * (1 - wx) + NOISE3[iz, iy, jx] * wx
                    c01 = NOISE3[iz, jy, ix] * (1 - wx) + NOISE3[iz, jy, jx] * wx
                    c10 = NOISE3[jz, iy, ix] * (1 - wx) + NOISE3[jz, iy, jx] * wx
                    c11 = NOISE3[jz, jy, ix] * (1 - wx) + NOISE3[jz, jy, jx] * wx
                    nval = (c00 * (1 - wy) + c01 * wy) * (1 - wz) + (c10 * (1 - wy) + c11 * wy) * wz
                    dens = 0.55 + 0.9 * nval
                    # thicker haze low in the nave
                    dens *= 1.0 + 0.6 * math.exp(-py * 0.25)
                    w = c * dens * dt
                    ar += r * w
                    ag += gg * w
                    ab += b * w
                k2 = sigma * ph * P[P_LSUN]
                ar *= k2
                ag *= k2
                ab *= k2
            out[y, x, 0] = ar
            out[y, x, 1] = ag
            out[y, x, 2] = ab


@njit(parallel=True, fastmath=True, cache=True)
def upsample_add(vol, vdepth, depth_full, scale, out):
    """Depth-aware upsampling of the half-res volumetric buffer, added to out."""
    H = out.shape[0]
    W = out.shape[1]
    h = vol.shape[0]
    w = vol.shape[1]
    for y in prange(H):
        for x in range(W):
            fx = (x + 0.5) / scale - 0.5
            fy = (y + 0.5) / scale - 0.5
            x0 = int(math.floor(fx))
            y0 = int(math.floor(fy))
            ax = fx - x0
            ay = fy - y0
            r = 0.0
            g = 0.0
            b = 0.0
            ws = 0.0
            for j in range(2):
                for i in range(2):
                    xx = min(max(x0 + i, 0), w - 1)
                    yy = min(max(y0 + j, 0), h - 1)
                    dd = depth_full[y, x]
                    rel = abs(vdepth[yy, xx] - dd) / (0.05 * dd + 0.05)
                    wgt = ((ax if i == 1 else 1.0 - ax) * (ay if j == 1 else 1.0 - ay) + 1e-3) * math.exp(-rel * rel)
                    r += wgt * vol[yy, xx, 0]
                    g += wgt * vol[yy, xx, 1]
                    b += wgt * vol[yy, xx, 2]
                    ws += wgt
            if ws < 1e-6:
                # no compatible neighbour: fall back to the nearest sample
                xx = min(max(int(fx + 0.5), 0), w - 1)
                yy = min(max(int(fy + 0.5), 0), h - 1)
                out[y, x, 0] += vol[yy, xx, 0]
                out[y, x, 1] += vol[yy, xx, 1]
                out[y, x, 2] += vol[yy, xx, 2]
            else:
                out[y, x, 0] += r / ws
                out[y, x, 1] += g / ws
                out[y, x, 2] += b / ws


# ------------------------------------------------------ point sprites --
@njit(fastmath=True, cache=True)
def splat_points(C, W, H, pts, col, rad, depth, out):
    """Additive soft point sprites with depth test. pts: (N,3), col: (N,3), rad: (N,) world size."""
    ox = C[0]
    oy = C[1]
    oz = C[2]
    for k in range(pts.shape[0]):
        cr = col[k, 0]
        cg = col[k, 1]
        cb = col[k, 2]
        if cr + cg + cb <= 1e-7:
            continue
        rx = pts[k, 0] - ox
        ry = pts[k, 1] - oy
        rz = pts[k, 2] - oz
        zc = rx * C[9] + ry * C[10] + rz * C[11]
        if zc < 0.3:
            continue
        xc = rx * C[3] + ry * C[4] + rz * C[5]
        yc = rx * C[6] + ry * C[7] + rz * C[8]
        sx = (xc / zc / C[12] + 1.0) * 0.5 * W
        sy = (1.0 - yc / zc / C[13]) * 0.5 * H
        pr = rad[k] / zc / C[13] * 0.5 * H          # radius in pixels
        if pr < 0.6:
            pr = 0.6
        if pr > 40.0:
            pr = 40.0
        if sx < -pr - 2 or sy < -pr - 2 or sx > W + pr + 2 or sy > H + pr + 2:
            continue
        dist = math.sqrt(rx * rx + ry * ry + rz * rz)
        inv = 0.63662 / (pr * pr)
        r = int(pr * 2.0 + 1.5)
        cx = int(sx)
        cy = int(sy)
        for j in range(cy - r, cy + r + 1):
            if j < 0 or j >= H:
                continue
            for i in range(cx - r, cx + r + 1):
                if i < 0 or i >= W:
                    continue
                if depth[j, i] < dist - 0.05:
                    continue
                ddx = (i + 0.5 - sx)
                ddy = (j + 0.5 - sy)
                q = (ddx * ddx + ddy * ddy) / (pr * pr)
                if q > 4.0:
                    continue
                wgt = math.exp(-2.0 * q) * inv
                out[j, i, 0] += cr * wgt
                out[j, i, 1] += cg * wgt
                out[j, i, 2] += cb * wgt


@njit(parallel=True, fastmath=True, cache=True)
def light_points(P, pts, F, CV, DD, DA, DI, out_col):
    """Sunlight colour at each point (for dust motes)."""
    for k in prange(pts.shape[0]):
        r, g, b = sun_at(P, F, CV, DD, DA, DI, pts[k, 0], pts[k, 1], pts[k, 2], -1)
        out_col[k, 0] = r
        out_col[k, 1] = g
        out_col[k, 2] = b


# ------------------------------------------------ moving-cube shadow map --
@njit(fastmath=True, cache=True)
def raster_tri(DD, DA, DI, x0, y0, x1, y1, x2, y2, depth, alpha, idx):
    minx = int(max(0, math.floor(min(x0, x1, x2))))
    maxx = int(min(DD.shape[1] - 1, math.ceil(max(x0, x1, x2))))
    miny = int(max(0, math.floor(min(y0, y1, y2))))
    maxy = int(min(DD.shape[0] - 1, math.ceil(max(y0, y1, y2))))
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(area) < 1e-9:
        return
    for yy in range(miny, maxy + 1):
        for xx in range(minx, maxx + 1):
            px = xx + 0.5
            py = yy + 0.5
            w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
            w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
            w2 = 1.0 - w0 - w1
            if w0 < -0.02 or w1 < -0.02 or w2 < -0.02:
                continue
            if depth < DD[yy, xx]:
                DD[yy, xx] = depth
                DA[yy, xx] = alpha
                DI[yy, xx] = idx


@njit(fastmath=True, cache=True)
def build_dyn_map(BLK, moving, DD, DA, DI):
    DD[:, :] = 1e9
    DA[:, :] = 0.0
    DI[:, :] = -1
    for m in range(moving.shape[0]):
        i = moving[m]
        cx = BLK[i, 0]
        cy = BLK[i, 1]
        cz = BLK[i, 2]
        h = BLK[i, 12]
        alpha = BLK[i, 16]
        if alpha <= 0.01:
            continue
        # 8 corners projected along the sun onto the glass plane
        us = np.empty(8)
        vs = np.empty(8)
        smin = 1e9
        k = 0
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                for c in (-1.0, 1.0):
                    wx = cx + h * (a * BLK[i, 3] + b * BLK[i, 4] + c * BLK[i, 5])
                    wy = cy + h * (a * BLK[i, 6] + b * BLK[i, 7] + c * BLK[i, 8])
                    wz = cz + h * (a * BLK[i, 9] + b * BLK[i, 10] + c * BLK[i, 11])
                    u, v, s = to_window(wx, wy, wz)
                    us[k] = (u + 6.0) * 30.0
                    vs[k] = (17.0 - v) * 30.0
                    smin = min(smin, s)
                    k += 1
        # the 6 faces (as 12 triangles); corner index = a*4 + b*2 + c
        faces = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
        for f in faces:
            raster_tri(DD, DA, DI, us[f[0]], vs[f[0]], us[f[1]], vs[f[1]], us[f[2]], vs[f[2]], smin, alpha, i)
            raster_tri(DD, DA, DI, us[f[0]], vs[f[0]], us[f[2]], vs[f[2]], us[f[3]], vs[f[3]], smin, alpha, i)
