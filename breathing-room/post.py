"""Post-processing: bloom, tone mapping, grading, film grain and the titles."""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, 'fonts', 'ShipporiMincho-Regular.ttf')

# (start, end, text, style) -- subtle Mincho-style titles
TEXTS = [
    (3.55, 6.75, 'Every day, a little faster.', 'low'),
    (7.15, 10.35, 'More efficiently.', 'low'),
    (11.25, 14.15, 'Before I knew it, the light had stopped coming in.', 'low'),
    (14.65, 17.95, 'I’ll let go of just one thing.', 'low'),
    (19.55, 23.45, 'Light enters through the space that opens up.', 'low'),
    (27.55, 31.65, 'That light was richer than before.', 'low'),
    (33.25, 35.90, 'Abundance lives in the space between.', 'final'),
]
FADE_IN, FADE_OUT = 0.85, 0.85


def bloom(img):
    h, w = img.shape[:2]
    acc = np.zeros_like(img)
    for f, sig, wt in ((2, 2.5, 0.28), (4, 4.0, 0.26), (8, 6.0, 0.24), (16, 8.0, 0.22)):
        small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sig)
        acc += wt * cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return acc


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


_vig_cache = {}


def vignette_mask(h, w):
    key = (h, w)
    if key not in _vig_cache:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 * 0.75 + ((yy - h / 2) / (h / 2)) ** 2 * 0.95
        _vig_cache[key] = r2.astype(np.float32)
    return _vig_cache[key]


def grade_image(hdr, g):
    x = hdr * g['exposure']
    b = bloom(x)
    x = x * (1 - g['bloom']) + b * g['bloom']
    x = aces(x * 1.0)
    # saturation (luma preserving)
    luma = (x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722)[..., None]
    x = luma + (x - luma) * g['saturation']
    x = x * np.array(g['tint'], np.float32) + np.array(g['lift'], np.float32)
    if g.get('white', 0) > 0:
        wv = g['white']
        x = x * (1 - wv) + wv * np.array([1.0, 0.975, 0.91], np.float32)
    h, w = x.shape[:2]
    x = x * (1 - g['vignette'] * vignette_mask(h, w))[..., None]
    return to_srgb(np.clip(x, 0, 1)).astype(np.float32)


# ------------------------------------------------------------------ titles --
_text_cache = {}


def _render_text(text, style, W, H):
    key = (text, style, W, H)
    if key in _text_cache:
        return _text_cache[key]
    size = int(round(H * (0.041 if style != 'final' else 0.046)))
    font = ImageFont.truetype(FONT, size)
    tracking = size * 0.06
    widths = [font.getlength(ch) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    pad = int(size * 1.5)
    tw = int(total + 2 * pad)
    th = int(size * 2.2 + 2 * pad)
    img = Image.new('L', (tw, th), 0)
    dr = ImageDraw.Draw(img)
    x = pad
    base_y = pad + size * 0.35
    for ch, wch in zip(text, widths):
        dr.text((x, base_y), ch, fill=255, font=font)
        x += wch + tracking
    m = np.asarray(img, np.float32) / 255.0
    halo = cv2.GaussianBlur(m, (0, 0), size * 0.28)
    halo = np.clip(halo * 2.2, 0, 1)
    cy = int(H * (0.905 if style == 'low' else 0.5))
    x0 = int(W / 2 - tw / 2)
    y0 = int(cy - th / 2)
    _text_cache[key] = (m, halo, x0, y0)
    return _text_cache[key]


def draw_titles(img, t):
    H, W = img.shape[:2]
    for (t0, t1, text, style) in TEXTS:
        if t < t0 or t > t1:
            continue
        a = min(1.0, (t - t0) / FADE_IN, (t1 - t) / FADE_OUT)
        a = a * a * (3 - 2 * a)
        m, halo, x0, y0 = _render_text(text, style, W, H)
        # a slow upward drift while it breathes in and out
        drift = int(round((1.0 - (t - t0) / (t1 - t0)) * H * 0.006))
        y0 += drift
        h, w = m.shape
        xa, ya = max(x0, 0), max(y0, 0)
        xb, yb = min(x0 + w, W), min(y0 + h, H)
        mm = m[ya - y0:yb - y0, xa - x0:xb - x0][..., None]
        hh = halo[ya - y0:yb - y0, xa - x0:xb - x0][..., None]
        reg = img[ya:yb, xa:xb]
        if style == 'final':
            ink = np.array([0.27, 0.21, 0.15], np.float32)
            glow = np.array([1.0, 0.98, 0.93], np.float32)
            reg = reg * (1 - 0.55 * a * hh) + glow * (0.55 * a * hh)
            reg = reg * (1 - 0.92 * a * mm) + ink * (0.92 * a * mm)
        else:
            ink = np.array([0.95, 0.92, 0.86], np.float32)
            wsum = float(hh.sum()) + 1e-6
            lum = float((reg.mean(axis=2, keepdims=True) * hh).sum()) / wsum
            k = min(0.82, max(0.30, (lum - 0.16) / (lum + 1e-3)))
            reg = reg * (1 - k * a * hh)
            reg = reg * (1 - 0.95 * a * mm) + ink * (0.95 * a * mm)
        img[ya:yb, xa:xb] = reg
    return img


def grain(img, frame, strength=0.006):
    rng = np.random.default_rng(1000 + frame)
    h, w = img.shape[:2]
    n = rng.standard_normal((h // 2, w // 2)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    luma = img.mean(axis=2, keepdims=True)
    amp = strength * (0.6 + 0.8 * (1 - luma))
    return img + n[..., None] * amp


def to_uint8(img, frame):
    rng = np.random.default_rng(5000 + frame)
    d = rng.uniform(-0.5, 0.5, img.shape[:2])[..., None] / 255.0
    return np.clip((img + d) * 255.0 + 0.5, 0, 255).astype(np.uint8)
