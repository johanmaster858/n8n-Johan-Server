"""The JAGA TECH emblem: vectorised from "logo final 111.jpeg" and brought to life.

36.0 s  the dazzling light collapses into darkness
36.4 s  a cold halo of light rises behind the emblem: a dark, enigmatic
        silhouette with stark volumetric shadows
38.0 s  the circuitry and the letters J-A-G-A / TECH ignite (electric blue
        and gold neon), then pulse
39.8 s  cut to black
"""
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'assets', 'logo final 111.jpeg')

T_COLLAPSE = 36.0
T_EMERGE = 36.45
T_IGNITE = 38.0
T_CUT = 39.8

# --------------------------------------------------------- segmentation --
# Polygons in the 1280 x 1280 reference image (x, y).  The emblem is one
# connected shape; these regions split it into the letters J-A-G-A that are
# hidden in the face, the circuit traces, and the remaining face lines.
J_POLY = [(300, 250), (418, 36), (486, 36), (486, 300), (468, 345), (440, 398), (300, 398)]
TOPBAR_POLY = [(486, 36), (790, 36), (790, 100), (486, 100)]          # J's crown, fades into the hair
A1_POLY = [(486, 300), (505, 228), (578, 228), (613, 330), (613, 447), (428, 447), (441, 398), (468, 345)]
G_POLY = [(428, 447), (613, 447), (613, 642), (395, 642), (395, 520)]
A2_POLY = [(296, 700), (348, 686), (478, 638), (534, 638), (534, 668), (560, 740), (576, 830), (596, 900), (296, 760)]
BUS_POLY = [(596, 900), (576, 830), (620, 860), (880, 1010), (880, 1075), (840, 1075), (560, 905)]
CIRCUIT_SHAPES = [
    ('circle', (642, 839, 27)), ('rect', (626, 852, 656, 892)),
    ('poly', [(700, 744), (734, 744), (734, 842), (770, 858), (770, 958), (736, 958), (736, 870), (700, 852)]),
    ('circle', (821, 860, 27)), ('rect', (806, 776, 836, 850)),
    ('circle', (815, 954, 27)), ('rect', (816, 966, 836, 996)),
]
LOGO_BOX = (296, 36, 978, 1240)      # x0, y0, x1, y1 of the emblem in the reference


def _poly_mask(shape, polys, scale, ox, oy, ss=4):
    h, w = shape
    m = np.zeros((h * ss, w * ss), np.uint8)
    for kind, data in polys:
        if kind == 'poly':
            pts = np.array([[(x - ox) * scale * ss, (y - oy) * scale * ss] for x, y in data], np.int32)
            cv2.fillPoly(m, [pts], 255)
        elif kind == 'circle':
            cx, cy, r = data
            cv2.circle(m, (int((cx - ox) * scale * ss), int((cy - oy) * scale * ss)), int(r * scale * ss), 255, -1)
        elif kind == 'rect':
            x0, y0, x1, y1 = data
            cv2.rectangle(m, (int((x0 - ox) * scale * ss), int((y0 - oy) * scale * ss)),
                          (int((x1 - ox) * scale * ss), int((y1 - oy) * scale * ss)), 255, -1)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def trace_contours():
    """Vectorise the emblem: smooth sub-pixel contours of the black shapes."""
    img = cv2.imread(SRC, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # upsample and smooth before thresholding: removes JPEG noise, keeps geometry
    up = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 3.0)
    bw = (up < 128).astype(np.uint8) * 255
    contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < 4 * 4 * 40:
            continue
        a = cv2.approxPolyDP(c, 1.2, True)[:, 0, :].astype(np.float64) / 4.0
        out.append(a)
    return out, hier


def write_svg(path, contours):
    x0, y0, x1, y1 = LOGO_BOX
    parts = []
    for c in contours:
        d = 'M ' + ' L '.join('%.2f %.2f' % (p[0] - x0, p[1] - y0) for p in c) + ' Z'
        parts.append(d)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d">\n'
           '  <path fill="#000" fill-rule="evenodd" d="%s"/>\n</svg>\n') % (x1 - x0, y1 - y0, ' '.join(parts))
    with open(path, 'w') as f:
        f.write(svg)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def sstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Emblem:
    def __init__(self, W, H):
        self.W, self.H = W, H
        contours, _ = trace_contours()
        self.contours = contours
        x0, y0, x1, y1 = LOGO_BOX
        self.logo_h = int(H * 0.80)
        self.scale = self.logo_h / (y1 - y0)
        lw = int((x1 - x0) * self.scale)
        self.logo_w = lw
        # composition: emblem centred, a little higher than centre
        self.cx = W / 2 + (615 - (x0 + x1) / 2) * 0.0
        self.cy = H / 2
        # rasterise the vector contours at 4x supersampling (even-odd fill)
        ss = 4
        m = np.zeros((self.logo_h * ss, lw * ss), np.uint8)
        pts = [np.round((c - [x0, y0]) * self.scale * ss).astype(np.int32) for c in contours]
        cv2.fillPoly(m, pts, 255)
        # even-odd: holes are separate contours; use drawContours with hierarchy by xor
        m2 = np.zeros_like(m)
        for p in pts:
            tmp = np.zeros_like(m)
            cv2.fillPoly(tmp, [p], 255)
            m2 ^= tmp
        L = cv2.resize(m2, (lw, self.logo_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        self.L = L
        shp = L.shape
        sc = self.scale
        blue = np.maximum.reduce([
            _poly_mask(shp, [('poly', J_POLY)], sc, x0, y0),
            _poly_mask(shp, [('poly', A1_POLY)], sc, x0, y0),
            _poly_mask(shp, [('poly', G_POLY)], sc, x0, y0),
            _poly_mask(shp, [('poly', A2_POLY)], sc, x0, y0),
        ])
        # the J's crown flows into the hair and fades out
        xs = (np.arange(lw) / sc + x0)
        fade = np.clip((760 - xs) / 170.0, 0, 1)[None, :]
        crown = _poly_mask(shp, [('poly', TOPBAR_POLY)], sc, x0, y0) * fade
        blue = np.maximum(blue, crown)
        # TECH: everything below the emblem's lowest stroke
        ys = (np.arange(self.logo_h) / sc + y0)[:, None]
        tech = (ys > 1068).astype(np.float32) * np.ones((1, lw), np.float32)
        gold = np.maximum(_poly_mask(shp, CIRCUIT_SHAPES, sc, x0, y0, ss=4),
                          _poly_mask(shp, [('poly', BUS_POLY)], sc, x0, y0))
        self.m_blue = np.clip(blue, 0, 1) * L
        self.m_tech = tech * L
        self.m_gold = np.clip(gold - blue, 0, 1) * L * (1 - tech)
        self.m_face = np.clip(L - self.m_blue - self.m_tech - self.m_gold, 0, 1)
        # inner distance (pixels) -> neon tube profile and bevel
        inside = (L > 0.5).astype(np.uint8)
        self.d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5).astype(np.float32)
        # geodesic-ish coordinate along the circuit for travelling pulses:
        # distance from the bus line, measured inside the gold mask
        gm = (self.m_gold > 0.3).astype(np.uint8)
        seed = (_poly_mask(shp, [('poly', BUS_POLY)], sc, x0, y0) > 0.5).astype(np.uint8)
        self.geo = self._geodesic(gm, seed)
        # pad centres (for LED flashes)
        self.pads = [((cx - x0) * sc, (cy - y0) * sc, r * sc) for kind, (cx, cy, r) in
                     [s for s in CIRCUIT_SHAPES if s[0] == 'circle']]
        # full-frame placement offsets
        self.ox = int(round(self.cx - lw / 2))
        self.oy = int(round(self.cy - self.logo_h / 2))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.xx, self.yy = xx, yy
        # halo centre: behind the head
        self.hx = self.ox + (700 - x0) * sc
        self.hy = self.oy + (420 - y0) * sc
        rng = np.random.default_rng(12)
        self.fog = [cv2.GaussianBlur(rng.standard_normal((H // 8, W // 8)).astype(np.float32), (0, 0), s)
                    for s in (3.0, 7.0)]

    @staticmethod
    def _geodesic(mask, seed, iters=400):
        dist = np.full(mask.shape, 1e6, np.float32)
        cur = seed.copy()
        dist[cur > 0] = 0
        k = np.ones((3, 3), np.uint8)
        for i in range(1, iters):
            nxt = cv2.dilate(cur, k) & mask
            new = (nxt > 0) & (dist > 1e5)
            if not new.any():
                break
            dist[new] = i
            cur = nxt | cur
        dist[dist > 1e5] = 0
        return dist

    # ----------------------------------------------------------------- frame --
    def place(self, layer, s=1.0):
        """Put a logo-sized layer into the frame, scaled by s about the frame centre."""
        H, W = self.H, self.W
        M = np.array([[s, 0, (1 - s) * W / 2 + s * self.ox], [0, s, (1 - s) * H / 2 + s * self.oy]], np.float32)
        return cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def radial_rays(self, light, cx, cy, decay=0.975, strength=1.0):
        """Volumetric light shafts: accumulate light outward from (cx, cy) in polar space."""
        H, W = light.shape[:2]
        R = int(math.hypot(max(cx, W - cx), max(cy, H - cy))) + 2
        nr, na = 512, 1440
        polar = cv2.warpPolar(np.ascontiguousarray(light, np.float32), (nr, na), (float(cx), float(cy)), R,
                              cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS)
        polar = np.nan_to_num(polar, nan=0.0, posinf=0.0, neginf=0.0)
        acc = np.zeros_like(polar)
        run = np.zeros_like(polar[:, 0])
        for r in range(nr):
            run = run * decay + polar[:, r] * (1 - decay)
            acc[:, r] = run
        out = cv2.warpPolar(acc, (W, H), (float(cx), float(cy)), R,
                            cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0) * strength

    def render(self, t, fi):
        H, W = self.H, self.W
        img = np.zeros((H, W, 3), np.float32)

        # ---- 36.0-36.45: the dazzling light collapses into the dark ----------
        if t < T_EMERGE:
            # the light implodes towards the centre in ~8 frames, flares once, and is gone
            d = np.sqrt((self.xx - W / 2) ** 2 + (self.yy - H / 2) ** 2)
            a = (t - T_COLLAPSE) / 0.24
            glow = np.zeros_like(d)
            if a < 1.0:
                r = math.hypot(W, H) * 0.62 * (1 - a) ** 1.7 + 2.0
                edge = 10.0 + 160.0 * (1 - a)
                inside = np.clip((r - d) / edge + 0.5, 0, 1)
                glow = inside * (2.2 - 0.8 * a) + np.exp(-((d - r) / (edge * 0.6)) ** 2) * 1.2 * a
            b = (t - T_COLLAPSE - 0.24) / 0.08
            if -0.4 < b < 1.0:
                glow = glow + max(0.0, 1.0 - abs(b - 0.3) / 0.7) * 3.0 * np.exp(-(d / 22.0) ** 2)
            img = glow[..., None] * np.array([1.0, 0.95, 0.86], np.float32)
            return self.finish(img, t, fi, bloom=0.30)

        # time-dependent controls
        emerge = sstep(T_EMERGE, T_IGNITE - 0.1, t)             # silhouette rises
        ign = t - T_IGNITE
        power = 0.0
        if ign >= 0:
            # neon strike: a few stutters, then full power
            if ign < 0.28:
                flick = [0.0, 0.03, 0.06, 0.10, 0.13, 0.17, 0.21, 0.28]
                on = [1, 0, 1, 0.2, 1, 0.4, 1, 1]
                k = max(i for i in range(len(flick)) if flick[i] <= ign)
                power = on[k] * (0.55 + 0.45 * ign / 0.28)
            else:
                beat = 0.5 + 0.5 * math.cos(2 * math.pi * 1.25 * (ign - 0.28))
                power = 0.78 + 0.22 * beat ** 2
        surge = math.exp(-max(ign, 0) / 0.22) * (1.0 if ign >= 0.24 else 0.0) if ign >= 0 else 0.0
        push = 0.955 + 0.045 * sstep(T_EMERGE, T_CUT, t)

        # ---- the void: faint drifting fog -------------------------------------
        fog = (0.6 * self.fog[0] + 0.4 * self.fog[1])
        sh = int((t - T_EMERGE) * 6)
        fog = np.roll(fog, sh, axis=1)
        fog = cv2.resize(fog, (W, H), interpolation=cv2.INTER_CUBIC)
        fog = np.clip(0.5 + 0.45 * fog, 0, 1)

        # ---- backlight: a cold halo behind the head (the Virgin's halo, reborn)
        L = self.place(self.L, push)
        hx = W / 2 + push * (self.hx - W / 2)
        hy = H / 2 + push * (self.hy - H / 2)
        d = np.sqrt((self.xx - hx) ** 2 + (self.yy - hy) ** 2)
        R0 = 215.0 * push
        disc = np.exp(-(d / R0) ** 2 * 1.6)
        ring = np.exp(-((d - R0 * 1.18) / 9.0) ** 2) + 0.35 * np.exp(-((d - R0 * 1.18) / 40.0) ** 2)
        cold = np.array([0.62, 0.74, 1.0], np.float32)
        elec = np.array([0.10, 0.36, 1.0], np.float32)
        gold = np.array([1.0, 0.62, 0.12], np.float32)
        pw = min(power, 1.0)
        back_col = cold * (1 - pw) + elec * pw
        rise = emerge ** 1.6
        # once the emblem powers up, the halo sinks into a deep electric blue
        back = (disc * 0.9 * (1.0 - 0.62 * pw) + ring * 0.55 * (1.0 + 0.4 * pw)) * rise * (1.0 + 0.9 * surge)
        unocc = back * (1.0 - L)
        # volumetric shafts streaming past the emblem: stark shadows in the haze
        rays = self.radial_rays(unocc * fog, hx, hy, decay=0.985)
        img += rays[..., None] * back_col * (1.9 + 0.6 * pw)
        img += (unocc * 0.55)[..., None] * back_col
        if pw > 0:
            img += (rays * 0.9 * pw)[..., None] * gold * 0.35
        img += (fog * 0.010 * rise * (1 + 2 * pw))[..., None] * back_col

        # ---- the emblem: a dark body, its edges caught by the halo ------------
        d_in = self.place(self.d_in, push)
        rim = np.exp(-d_in / 1.6) * L
        img = img * (1 - L[..., None])
        # edges nearest the halo catch the most light
        near = np.exp(-(d / (R0 * 1.9)) ** 2)
        img += (rim * rise * (0.10 + 0.55 * near) * (1 - 0.6 * pw))[..., None] * cold

        # ---- ignition: neon letters (electric blue) and circuitry (gold) ------
        if power > 0:
            mb = self.place(self.m_blue + self.m_tech, push)
            mg = self.place(self.m_gold, push)
            mf = self.place(self.m_face, push)
            tube = np.exp(-np.maximum(d_in - 0.8, 0) / 2.4)          # bright tube along the edges
            body = np.clip(d_in / 9.0, 0, 1)                         # glowing gas inside the stroke
            geo = self.place(self.geo, push)
            ph = (geo / (16.0 * push) - (t - T_IGNITE) * 2.6) % 1.0
            pulse = np.exp(-((ph - 0.5) / 0.10) ** 2)
            p = power * (1.0 + 1.2 * surge)
            blue_core = np.array([0.55, 0.80, 1.0], np.float32)
            blue_gas = np.array([0.03, 0.22, 1.0], np.float32)
            blue_glow = np.array([0.02, 0.20, 1.0], np.float32)
            gold_core = np.array([1.0, 0.86, 0.50], np.float32)
            gold_gas = np.array([1.0, 0.42, 0.02], np.float32)
            gold_glow = np.array([1.0, 0.38, 0.02], np.float32)
            nb_core = mb * tube
            nb_gas = mb * (0.25 + 0.75 * body)
            img += (nb_core * p * 2.6)[..., None] * blue_core
            img += (nb_gas * p * 1.15)[..., None] * blue_gas
            glow_b = cv2.GaussianBlur(mb, (0, 0), 7) * 1.1 + cv2.GaussianBlur(mb, (0, 0), 26) * 0.9
            img += (glow_b * p * (1 - 0.7 * mb))[..., None] * blue_glow
            ng = mg * (0.6 + 0.9 * pulse)
            img += (ng * tube * p * 2.8)[..., None] * gold_core
            img += (ng * (0.3 + 0.7 * body) * p * 1.3)[..., None] * gold_gas
            glow_g = cv2.GaussianBlur(ng, (0, 0), 6) * 1.2 + cv2.GaussianBlur(ng, (0, 0), 22) * 0.9
            img += (glow_g * p * (1 - 0.6 * mg))[..., None] * gold_glow
            # the face lines only catch a thin cold electric edge
            img += (mf * tube * p * 0.35)[..., None] * blue_glow
            # LED pads flare with each passing pulse
            for (px, py, pr) in self.pads:
                X = W / 2 + push * (self.ox + px - W / 2)
                Y = H / 2 + push * (self.oy + py - H / 2)
                dd = np.sqrt((self.xx - X) ** 2 + (self.yy - Y) ** 2)
                fl = 0.55 + 0.45 * math.sin((t - T_IGNITE) * 7.0 + px * 0.05) ** 2
                img += (np.exp(-(dd / (pr * push * 0.9)) ** 2) * p * 2.4 * fl)[..., None] * gold_core
                img += (np.exp(-(dd / (pr * push * 3.0)) ** 2) * p * 0.5 * fl)[..., None] * gold_glow
            # shockwave of the ignition
            if 0.2 <= ign < 1.1:
                a = (ign - 0.2) / 0.9
                rr = 60 + a * 1300
                sw = np.exp(-((d - rr) / (18 + 50 * a)) ** 2) * (1 - a) ** 2
                img += (sw * 0.9)[..., None] * blue_glow
            # holographic scanlines, very faint
            scan = 0.5 + 0.5 * np.sin(self.yy * 1.3 + t * 26.0)
            img *= (1.0 - 0.05 * scan * power)[..., None]
        return self.finish(img, t, fi, bloom=0.14 + 0.10 * min(power, 1.0), aberration=0.6 + 3.5 * surge,
                           vignette=0.55)

    def finish(self, img, t, fi, bloom=0.2, aberration=0.0, vignette=0.35):
        import post
        H, W = self.H, self.W
        b = post.bloom(img)
        x = img * (1 - bloom) + b * bloom * 1.3
        x = post.aces(x)
        if aberration > 0.05:
            s = aberration
            M1 = np.float32([[1 + s / W, 0, -s / 2], [0, 1 + s / W, -s * H / W / 2]])
            M2 = np.float32([[1 - s / W, 0, s / 2], [0, 1 - s / W, s * H / W / 2]])
            x[..., 0] = cv2.warpAffine(x[..., 0], M1, (W, H), borderMode=cv2.BORDER_REPLICATE)
            x[..., 2] = cv2.warpAffine(x[..., 2], M2, (W, H), borderMode=cv2.BORDER_REPLICATE)
        vig = post.vignette_mask(H, W)
        x = x * (1 - vignette * vig)[..., None]
        x = post.to_srgb(np.clip(x, 0, 1)).astype(np.float32)
        x = post.grain(x, fi, 0.012)
        return post.to_uint8(x, fi)


_EMBLEM = {}


def render_frame(t, w, h, fi):
    if t >= T_CUT:
        return np.zeros((h, w, 3), np.uint8)
    key = (w, h)
    if key not in _EMBLEM:
        _EMBLEM[key] = Emblem(w, h)
    return _EMBLEM[key].render(t, fi)


if __name__ == '__main__':
    import sys
    c, _ = trace_contours()
    write_svg(os.path.join(HERE, 'output', 'jaga_tech_emblem_traced.svg'), c)
    print('contours', len(c))
