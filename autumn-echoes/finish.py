"""Finishing: assemble the rendered scenes with transitions, the amber dissolve, the ember
emblem, the Spanish titles and film grain, and pipe the frames to ffmpeg.

    python finish.py [--preview i0 i1 step]   (writes PNGs to build/preview instead of encoding)
"""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from logo import EmberEmblem, fsstep
from render import RANGES

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS, N = 1080, 1920, 30, 1200
FONT = os.path.join(HERE, 'fonts', 'CormorantGaramond[wght].ttf')
FFMPEG = os.environ.get('FFMPEG', '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')

# on-screen words (Spanish), bottom centre
TEXTS = [
    (0.45, 5.0, 'A veces, nos llenamos de cosas...'),
    (6.0, 11.0, '...buscando una felicidad\nque no se toca.'),
    (14.0, 18.0, 'Mi felicidad no se mide en bienes,'),
    (19.0, 23.0, 'se mide en recuerdos y anhelos.'),
    (26.0, 31.0, 'Al final, solo somos\nlas historias que vivimos.'),
]
FADE_IN, FADE_OUT = 0.7, 0.7

ORDER = ['forest', 'hand', 'boots', 'campfire', 'aerial']
# how each overlap blends: glow = a bloom of warm light at the midpoint
TRANSITION = {('forest', 'hand'): 0.9, ('hand', 'boots'): 0.35, ('boots', 'campfire'): 0.25, ('campfire', 'aerial'): 0.45}

SRGB2LIN = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)


def load(scene, i):
    p = os.path.join(HERE, 'frames', scene, '%05d.png' % i)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return SRGB2LIN[im[..., ::-1]]


def to_srgb(x):
    return np.power(np.clip(x, 0, 1), 1 / 2.2)


def blur_small(img, sigma, f=4):
    small = cv2.resize(img, (W // f, H // f), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(sigma / f, 0.3))
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def light_dissolve(A, B, s, glow):
    e = s * s * (3 - 2 * s)
    out = A * (1 - e) + B * e
    peak = math.sin(math.pi * s) ** 1.5
    if glow > 0 and peak > 0.01:
        g = blur_small(np.maximum(A, B), 60, 8) * np.array([1.0, 0.78, 0.5], np.float32)
        k = glow * peak
        out = out + g * k
    return out


def scene_frame(i):
    live = [s for s in ORDER if RANGES[s][0] <= i <= RANGES[s][1]]
    if len(live) == 1:
        return load(live[0], i)
    a, b = live[0], live[1]
    s0, s1 = RANGES[b][0], RANGES[a][1]
    s = (i - s0 + 0.5) / (s1 - s0 + 1)
    return light_dissolve(load(a, i), load(b, i), s, TRANSITION[(a, b)])


# ------------------------------------------------------------ amber and logo --
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.sqrt(((_xx - W / 2) / (W * 0.62)) ** 2 + ((_yy - H * 0.46) / (H * 0.5)) ** 2)


def amber_field(t):
    """A warm, glowing amber vignette (linear RGB)."""
    breathe = 1.0 + 0.03 * math.sin(t * 2.1)
    inner = np.array([1.1, 0.58, 0.17], np.float32)
    mid = np.array([0.72, 0.22, 0.035], np.float32)
    outer = np.array([0.03, 0.008, 0.002], np.float32)
    a = np.clip(_r / 0.55, 0, 1)[..., None] ** 1.1
    rb = np.clip((_r - 0.3) / 1.1, 0, 1)[..., None]
    b = rb * rb * (3 - 2 * rb)
    col = inner * (1 - a) + mid * a
    col = col * (1 - b) + outer * b
    return col * breathe + motes(t)


_mrng = np.random.default_rng(77)
_MOTES = [(_mrng.uniform(0.1, 0.9) * W, _mrng.uniform(0.2, 1.1) * H, _mrng.uniform(12, 40), _mrng.uniform(30, 90), _mrng.uniform(0, 6.28),
           _mrng.uniform(0.3, 1.0)) for _ in range(70)]


def motes(t):
    """Soft out-of-focus embers drifting up through the amber light (32.5-38 s)."""
    k = fsstep(32.5, 34.0, t) * (1 - fsstep(36.8, 38.2, t))
    out = np.zeros((H // 4, W // 4), np.float32)
    if k <= 0:
        return 0.0
    for (x0, y0, r, v, ph, b) in _MOTES:
        x = (x0 + 25 * math.sin(t * 0.7 + ph)) / 4
        y = (y0 - v * (t - 32.0)) / 4
        if -20 < y < H / 4 + 20:
            cv2.circle(out, (int(x), int(y)), max(int(r / 4), 1), float(b * (0.6 + 0.4 * math.sin(t * 3 + ph))), -1, lineType=cv2.LINE_AA)
    out = cv2.GaussianBlur(out, (0, 0), 1.5)
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
    return out[..., None] * np.array([1.0, 0.62, 0.25], np.float32) * 0.35 * k


_emblem = None


def emblem():
    global _emblem
    if _emblem is None:
        _emblem = EmberEmblem(W, H)
    return _emblem


def finish_linear(i):
    t = i / FPS
    if t < 36.0:
        img = scene_frame(min(i, RANGES['aerial'][1]))
    else:
        img = np.zeros((H, W, 3), np.float32)
    # 32-36 s: the landscape blurs and melts into a warm amber glow
    if t >= 32.0:
        s = fsstep(32.0, 36.0, t)
        if t < 36.0:
            img = blur_small(img, 2 + 90 * s ** 1.5) if s > 0.02 else img
            warm = img * np.array([1.12, 0.92, 0.62], np.float32) * (1 + 0.4 * s)
            img = img * (1 - s) + warm * s
        amb = amber_field(t)
        k = fsstep(32.6, 36.0, t) ** 1.3
        # after 36 s the amber light gathers behind the emblem and slowly dies away
        dim = (1.0 - 0.86 * fsstep(36.0, 37.5, t)) * (1.0 - fsstep(38.6, 39.8, t))
        if t < 36.0:
            img = img * (1 - k) + amb * k
        else:
            img = amb * dim
        if t >= 35.8:
            img = emblem().render(t, img)
    # fade in from black at the start
    if t < 0.9:
        img = img * fsstep(0.0, 0.9, t)
    return img


# ------------------------------------------------------------------ titles --
_text_cache = {}


def _wrap(text, font, maxw):
    words = text.split(' ')
    lines, cur = [], ''
    for w in words:
        trial = (cur + ' ' + w).strip()
        if font.getlength(trial) <= maxw or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    # balance two lines
    if len(lines) == 2:
        best = None
        for k in range(1, len(words)):
            l1, l2 = ' '.join(words[:k]), ' '.join(words[k:])
            m = max(font.getlength(l1), font.getlength(l2))
            if m <= maxw and (best is None or m < best[0]):
                best = (m, [l1, l2])
        if best:
            lines = best[1]
    return lines


def text_layers(text):
    if text in _text_cache:
        return _text_cache[text]
    size = int(round(H * 0.0375))
    font = ImageFont.truetype(FONT, size)
    font.set_variation_by_name(b'SemiBold')
    lines = text.split('\n') if '\n' in text else _wrap(text, font, W * 0.9)
    lh = size * 1.22
    pad = int(size * 1.6)
    tw = int(max(font.getlength(l) for l in lines)) + 2 * pad
    th = int(lh * len(lines) + size * 0.4) + 2 * pad
    img = Image.new('L', (tw, th), 0)
    dr = ImageDraw.Draw(img)
    for k, l in enumerate(lines):
        x = (tw - font.getlength(l)) / 2
        dr.text((x, pad + k * lh), l, fill=255, font=font)
    m = np.asarray(img, np.float32) / 255.0
    glow = np.clip(cv2.GaussianBlur(m, (0, 0), size * 0.3) * 1.8, 0, 1)
    shadow = cv2.GaussianBlur(np.roll(m, int(size * 0.05), axis=0), (0, 0), size * 0.1)
    backdrop = np.clip(cv2.GaussianBlur(m, (0, 0), size * 0.9) * 2.4, 0, 1)
    x0 = int(W / 2 - tw / 2)
    y0 = int(H * 0.80 - th / 2)
    _text_cache[text] = (m, glow, shadow, backdrop, x0, y0)
    return _text_cache[text]


def draw_titles(img, t):
    """img: sRGB float."""
    for (t0, t1, text) in TEXTS:
        if t < t0 or t > t1:
            continue
        a = min(1.0, (t - t0) / FADE_IN, (t1 - t) / FADE_OUT)
        a = a * a * (3 - 2 * a)
        m, glow, shadow, backdrop, x0, y0 = text_layers(text)
        drift = (1.0 - (t - t0) / (t1 - t0)) * H * 0.004
        y0 = int(round(y0 + drift))
        h, w = m.shape
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
        m, glow, shadow, backdrop = (q[ya - y0:yb - y0, xa - x0:xb - x0] for q in (m, glow, shadow, backdrop))
        x0, y0, h, w = xa, ya, yb - ya, xb - xa
        reg = img[y0:y0 + h, x0:x0 + w]
        # darken behind the words in proportion to how bright the background is
        wsum = float(backdrop.sum()) + 1e-6
        lum = float((reg.mean(axis=2) * backdrop).sum()) / wsum
        kb = np.clip((lum - 0.25) * 1.1, 0.12, 0.55)
        reg = reg * (1 - kb * a * backdrop[..., None])
        reg = reg * (1 - 0.6 * a * shadow[..., None])
        # soft warm glow, then the letters
        gcol = np.array([1.0, 0.82, 0.55], np.float32)
        reg = 1 - (1 - reg) * (1 - 0.22 * a * glow[..., None] * gcol)
        ink = np.array([1.0, 0.96, 0.88], np.float32)
        reg = reg * (1 - a * m[..., None]) + ink * (a * m[..., None])
        img[y0:y0 + h, x0:x0 + w] = reg
    return img


def grain(img, i, strength=0.012):
    rng = np.random.default_rng(1000 + i)
    n = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    luma = img.mean(axis=2, keepdims=True)
    return img + n[..., None] * strength * (0.5 + 0.9 * (1 - luma)) * (0.35 + 0.65 * np.sqrt(np.clip(luma, 0, 1) * 4).clip(0, 1))


def final_frame(i):
    t = i / FPS
    img = to_srgb(finish_linear(i))
    img = draw_titles(img, t)
    img = grain(img, i)
    rng = np.random.default_rng(5000 + i)
    img = img + rng.uniform(-0.5, 0.5, img.shape[:2])[..., None] / 255.0
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def main():
    if '--preview' in sys.argv:
        k = sys.argv.index('--preview')
        a, b, st = int(sys.argv[k + 1]), int(sys.argv[k + 2]), int(sys.argv[k + 3])
        out = os.path.join(HERE, 'build', 'preview')
        os.makedirs(out, exist_ok=True)
        for i in range(a, b + 1, st):
            cv2.imwrite(os.path.join(out, '%05d.png' % i), final_frame(i)[..., ::-1])
            print('preview', i, flush=True)
        return
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'build', 'video.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [FFMPEG, '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p',
           '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-movflags', '+faststart', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(3) as pool:
        for n, fr in enumerate(pool.imap(final_frame, range(N), chunksize=4)):
            p.stdin.write(fr.tobytes())
            if n % 60 == 0:
                print('frame', n, flush=True)
    p.stdin.close()
    p.wait()
    print('wrote', out)


if __name__ == '__main__':
    main()
