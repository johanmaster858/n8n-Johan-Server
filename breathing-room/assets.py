"""Textures derived from the stained-glass design, plus noise and the word atlas."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, 'fonts')

WORDS = ['BUSY', 'EFFICIENCY', 'URGENT', 'DEADLINE', 'FASTER', 'OVERTIME', 'ASAP', 'SCHEDULE',
         'PRODUCTIVITY', 'MULTITASK', 'HURRY', 'MORE', 'NOW', 'OPTIMIZE', 'RUSH', 'TARGETS']
CORE_WORDS = 4          # the first four are the ones named in the brief


def _down2(a):
    h, w = a.shape[:2]
    a = a[: h - h % 2, : w - w % 2]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def _box_down(a, f):
    h, w = a.shape[:2]
    a = a[: h - h % f, : w - w % f]
    sh = (h // f, f, w // f, f) + a.shape[2:]
    return a.reshape(sh).mean(axis=(1, 3))


def window_textures(design_npz):
    """Mip chain for the window display and pre-filtered light textures.

    Returns G0..G3 (H, W, 5) = [T r, g, b, white-glass mask, stone mask] at
    160/80/40/20 px per metre, F (1200, 800, 4) at 80 px/m for surface light,
    VT (300, 200, 4) at 20 px/m for the volumetric light columns.
    """
    d = np.load(design_npz)
    T = d['T'].astype(np.float32)
    stone = d['stone'].astype(np.float32)
    inside = d['inside'].astype(np.float32)
    white = np.clip(inside - stone, 0, 1)
    G0 = np.concatenate([T, white[..., None], stone[..., None]], axis=2).astype(np.float32)
    mips = [G0]
    for _ in range(3):
        mips.append(_down2(mips[-1]).astype(np.float32))

    # light on surfaces: sharp enough to keep the leading visible on the floor
    T80r = _box_down(T, 2)
    T80 = ndimage.gaussian_filter(T80r, sigma=(1.6, 1.6, 0))
    T80soft = ndimage.gaussian_filter(T80r, sigma=(5.5, 5.5, 0))
    w80 = _box_down(white, 2)
    # the first, colourless light is a soft white shape
    w80s = ndimage.gaussian_filter(w80, sigma=5.0)
    F = np.concatenate([T80, w80s[..., None], T80soft], axis=2).astype(np.float32)

    T20 = _box_down(T, 8)
    T20 = ndimage.gaussian_filter(T20, sigma=(0.8, 0.8, 0))
    w20 = ndimage.gaussian_filter(_box_down(white, 8), sigma=1.5)
    VT = np.concatenate([T20, w20[..., None]], axis=2).astype(np.float32)
    return mips, F, VT


def tile_noise(n, sigmas, seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros((n, n), np.float32)
    amp = 1.0
    for s in sigmas:
        g = rng.standard_normal((n, n)).astype(np.float32)
        g = ndimage.gaussian_filter(g, s, mode='wrap')
        g /= g.std() + 1e-9
        acc += amp * g
        amp *= 0.55
    acc -= acc.min()
    acc /= acc.max()
    return acc.astype(np.float32)


def noise3d(n=32, seed=3):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, n, n)).astype(np.float32)
    g = ndimage.gaussian_filter(g, 2.0, mode='wrap')
    g -= g.min()
    g /= g.max()
    return g.astype(np.float32)


def word_atlas(size=256):
    font_path = os.path.join(FONT_DIR, 'Oswald.ttf')
    masks = np.zeros((len(WORDS), size, size), np.float32)
    shades = np.ones((len(WORDS), size, size), np.float32)
    ss = 4
    S = size * ss
    for k, word in enumerate(WORDS):
        target_cap = 0.25 * S
        font = ImageFont.truetype(font_path, 100)
        try:
            font.set_variation_by_name('SemiBold')
        except Exception:
            pass
        bbox = font.getbbox('H')
        cap = bbox[3] - bbox[1]
        fs = 100 * target_cap / cap
        font = ImageFont.truetype(font_path, int(fs))
        try:
            font.set_variation_by_name('SemiBold')
        except Exception:
            pass
        tb = font.getbbox(word)
        tw = tb[2] - tb[0]
        if tw > 0.80 * S:
            fs *= 0.80 * S / tw
            font = ImageFont.truetype(font_path, int(fs))
            try:
                font.set_variation_by_name('SemiBold')
            except Exception:
                pass
            tb = font.getbbox(word)
        img = Image.new('L', (S, S), 0)
        dr = ImageDraw.Draw(img)
        x = (S - (tb[2] - tb[0])) / 2 - tb[0]
        y = (S - (tb[3] - tb[1])) / 2 - tb[1]
        dr.text((x, y), word, fill=255, font=font)
        m = np.asarray(img, np.float32) / 255.0
        # engraved: the upper-left inner walls of the letters are in shadow
        sh_px = int(0.012 * S)
        shifted = np.zeros_like(m)
        shifted[sh_px:, sh_px:] = m[:-sh_px, :-sh_px]
        wall = np.clip(m - shifted, 0, 1)
        shade = 1.0 - 0.55 * ndimage.gaussian_filter(wall, ss * 0.6)
        m = _box_down(m, ss)
        shade = _box_down(shade, ss)
        masks[k] = m
        shades[k] = shade
    return masks, shades


def build_all(design_npz):
    mips, F, VT = window_textures(design_npz)
    NT = tile_noise(512, [2.0, 6.0, 18.0], 21)
    BN = tile_noise(256, [1.0, 3.0, 9.0], 22)
    N3 = noise3d(32, 3)
    WM, WS = word_atlas(256)
    WM1 = np.stack([_box_down(m, 4) for m in WM]).astype(np.float32)
    WS1 = np.stack([_box_down(m, 4) for m in WS]).astype(np.float32)
    return dict(G=mips, F=F, VT=VT, NT=NT, BN=BN, N3=N3, WM=WM, WS=WS, WM1=WM1, WS1=WS1)
