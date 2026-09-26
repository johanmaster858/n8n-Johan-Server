"""The JAGA TECH emblem as a dying ember: vectorised from "logo final 111.jpeg",
it kindles out of the amber light, glows golden-orange like the last of a sunset,
breathes, sheds a few sparks, then cools to deep red and goes dark.

36.0 s  amber light gathers; the emblem smoulders into view from below
37.3 s  fully alight: golden core, orange edges, warm halo, drifting sparks
38.7 s  it begins to cool (gold -> orange -> deep red)
40.0 s  black
"""
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'assets', 'logo final 111.jpeg')
LOGO_BOX = (296, 36, 978, 1240)   # x0, y0, x1, y1 of the emblem in the 1280 x 1280 reference


def trace_contours():
    """Smooth sub-pixel contours of the black shapes (4x upsample, blur, threshold)."""
    img = cv2.imread(SRC, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    up = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 3.0)
    bw = (up < 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < 4 * 4 * 40:
            continue
        out.append(cv2.approxPolyDP(c, 1.2, True)[:, 0, :].astype(np.float64) / 4.0)
    return out


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fsstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ember_ramp(h):
    """Heat 0..1 -> linear RGB: deep red, orange, gold, pale gold."""
    h = np.clip(h, 0, 1)[..., None]
    c0 = np.array([0.30, 0.02, 0.0], np.float32)
    c1 = np.array([1.0, 0.24, 0.02], np.float32)
    c2 = np.array([1.0, 0.56, 0.12], np.float32)
    c3 = np.array([1.0, 0.86, 0.55], np.float32)
    a = sstep(0.0, 0.35, h)
    b = sstep(0.35, 0.7, h)
    c = sstep(0.7, 1.0, h)
    return c0 * (1 - a) + c1 * a + (c2 - c1) * b + (c3 - c2) * c


class EmberEmblem:
    def __init__(self, W, H):
        self.W, self.H = W, H
        contours = trace_contours()
        x0, y0, x1, y1 = LOGO_BOX
        self.logo_h = int(H * 0.5)
        sc = self.logo_h / (y1 - y0)
        lw = int((x1 - x0) * sc)
        ss = 4
        m = np.zeros((self.logo_h * ss, lw * ss), np.uint8)
        for c in contours:
            tmp = np.zeros_like(m)
            cv2.fillPoly(tmp, [np.round((c - [x0, y0]) * sc * ss).astype(np.int32)], 255)
            m ^= tmp
        L = cv2.resize(m, (lw, self.logo_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        # place in the frame, a little above centre
        self.ox = int(round(W / 2 - lw / 2 + (615 - (x0 + x1) / 2) * 0))
        self.oy = int(round(H * 0.46 - self.logo_h / 2))
        full = np.zeros((H, W), np.float32)
        full[self.oy:self.oy + self.logo_h, self.ox:self.ox + lw] = L
        self.L = full
        inside = (full > 0.5).astype(np.uint8)
        d = cv2.distanceTransform(inside, cv2.DIST_L2, 5).astype(np.float32)
        self.core = np.clip(d / max(float(d.max()) * 0.55, 1.0), 0, 1)   # 0 at edges -> 1 in thick strokes
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.yy, self.xx = yy, xx
        # vertical coordinate inside the logo 0 (bottom) .. 1 (top) for the kindling front
        self.v = np.clip((self.oy + self.logo_h - yy) / self.logo_h, 0, 1)
        rng = np.random.default_rng(36)
        self.noise = [cv2.resize(cv2.GaussianBlur(rng.standard_normal((H // s, W // s)).astype(np.float32), (0, 0), 1.2),
                                 (W, H), interpolation=cv2.INTER_CUBIC) for s in (6, 14, 40)]
        # sparks leave from points on the strokes
        ys, xs = np.nonzero(inside)
        pick = rng.choice(len(xs), 140, replace=False)
        self.spx, self.spy = xs[pick].astype(np.float32), ys[pick].astype(np.float32)
        self.spt = rng.uniform(36.9, 39.2, 140)
        self.spv = rng.uniform(40, 120, 140)
        self.sps = rng.uniform(0.6, 1.6, 140)
        self.sph = rng.uniform(0, 6.28, 140)
        self.halo_c = (W / 2, H * 0.46)

    def heat(self, t):
        """Per-pixel heat field of the emblem at time t."""
        n0, n1, n2 = self.noise
        # smoulders into view from the bottom, like embers catching
        front = fsstep(36.0, 37.4, t) * 1.35
        kindle = sstep(-0.18, 0.0, front - self.v - 0.12 * n1)
        # cracked-charcoal texture that slowly shifts
        shift = 0.5 + 0.5 * np.sin(t * 1.3 + n2 * 3.0)
        crack = 0.72 + 0.28 * np.tanh(2.2 * (n0 * 0.7 + n1 * 0.5 + 0.3 * shift))
        breath = 1.0 + 0.06 * math.sin(t * 5.1) + 0.035 * math.sin(t * 11.7 + 1.0)
        base = (0.45 + 0.55 * self.core) * crack * breath
        cool = fsstep(38.6, 39.85, t)
        peak = 0.95 * (1 - 0.55 * cool) - 0.35 * cool * (1 - self.core)
        return np.clip(base * kindle * peak, 0, 1) * (1 - fsstep(39.55, 39.95, t)), kindle

    def render(self, t, background):
        """background: linear RGB float frame (the amber field). Returns linear RGB."""
        W, H = self.W, self.H
        h, kindle = self.heat(t)
        L = self.L
        emit = ember_ramp(h) * (h[..., None] ** 1.2) * 3.2 * L[..., None]
        # dark charcoal where not yet (or no longer) glowing
        char = np.array([0.006, 0.004, 0.003], np.float32)
        # the emblem appears as it catches: charcoal only just ahead of the kindling front
        front = fsstep(36.0, 37.4, t) * 1.35
        near = sstep(-0.32, 0.0, front - self.v - 0.12 * self.noise[1]) * (1 - fsstep(39.2, 39.75, t))
        occ = (L * near)[..., None]
        img = background * (1 - occ) + char * occ + emit
        # glow: layered blurs of the emission
        small = cv2.resize(emit, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(small, (0, 0), 3) * 0.6 + cv2.GaussianBlur(small, (0, 0), 12) * 0.55 + cv2.GaussianBlur(small, (0, 0), 40) * 0.7
        img += cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([1.0, 0.6, 0.3], np.float32)
        # sparks drifting up from the strokes
        sp = np.zeros((H, W), np.float32)
        for i in range(len(self.spx)):
            age = t - self.spt[i]
            if age < 0 or age > self.sps[i]:
                continue
            x = self.spx[i] + 14 * math.sin(age * 3 + self.sph[i]) + age * 10
            y = self.spy[i] - self.spv[i] * age - 30 * age * age
            b = (1 - age / self.sps[i]) ** 1.5 * (1 - fsstep(39.3, 39.9, t))
            if 2 <= x < W - 2 and 2 <= y < H - 2:
                cv2.circle(sp, (int(x), int(y)), 2, float(b * 2.5), -1, lineType=cv2.LINE_AA)
        if sp.max() > 0:
            spg = sp + cv2.GaussianBlur(sp, (0, 0), 4) * 1.5
            img += spg[..., None] * np.array([1.0, 0.55, 0.18], np.float32)
        return img
