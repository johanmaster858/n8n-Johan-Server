"""Procedural soundtrack for "Autumn Echoes", locked to the picture's timeline.

    python audio.py build/soundtrack.wav

0-6    heavy, muted drone in D minor; a still forest (distant chickadee)
6-14   the wind rises, leaves rustle, the goods dissolve into glittering chimes;
       the harmony opens from minor to D major; a flurry sweeps past
14-22  memories: nylon guitar over a warm pad; footsteps crunch in dry leaves;
       a campfire crackles by the lake and a loon calls across the water
22-32  the lift: strings and pad swell, a soaring melody over the hills
32-36  the chord glows and melts into warmth
36-40  an ember hum and one deep resonant bell; silence
"""
import math
import os
import sys

import numpy as np
from numba import njit
from scipy import signal
from scipy.io import wavfile

SR = 48000
DUR = 40.0
N = int(DUR * SR)
rng = np.random.default_rng(2024)


def secs(t):
    return int(round(t * SR))


def hz(note):
    return 440.0 * 2 ** ((note - 69) / 12)


NOTE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}


def n(name):
    """'F#4' -> MIDI"""
    p = name[:-1]
    o = int(name[-1])
    return 12 * (o + 1) + NOTE[p]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ramp(t, a, b):
    return np.clip((t - a) / (b - a), 0, 1)


class Bus:
    def __init__(self):
        self.l = np.zeros(N + SR * 8)
        self.r = np.zeros(N + SR * 8)

    def add(self, t0, x, gain=1.0, pan=0.0):
        i = secs(t0)
        if i < 0:
            x = x[-i:]
            i = 0
        k = min(len(x), len(self.l) - i)
        if k <= 0:
            return
        a = (pan + 1) * math.pi / 4
        self.l[i:i + k] += x[:k] * gain * math.cos(a)
        self.r[i:i + k] += x[:k] * gain * math.sin(a)

    def add_stereo(self, t0, l, r, gain=1.0):
        i = secs(t0)
        k = min(len(l), len(self.l) - i)
        self.l[i:i + k] += l[:k] * gain
        self.r[i:i + k] += r[:k] * gain


@njit(cache=True)
def svf(x, fc, q, mode):
    y = np.zeros_like(x)
    lp = 0.0
    bp = 0.0
    for i in range(x.shape[0]):
        f = 2.0 * math.sin(math.pi * min(fc[i], 18000.0) / 48000.0)
        hp = x[i] - lp - bp / q
        bp += f * hp
        lp += f * bp
        if mode == 0:
            y[i] = lp
        elif mode == 1:
            y[i] = bp
        else:
            y[i] = hp
    return y


@njit(cache=True)
def ks_pluck(period, nsamp, decay, seedbuf):
    """Karplus-Strong string: period in samples (float), fractional delay by linear interpolation."""
    L = int(period) + 2
    buf = np.zeros(L)
    for i in range(L):
        buf[i] = seedbuf[i % seedbuf.shape[0]]
    out = np.zeros(nsamp)
    frac = period - int(period)
    idx = 0
    last = 0.0
    for i in range(nsamp):
        j = idx % L
        j1 = (idx + 1) % L
        s = buf[j] * (1 - frac) + buf[j1] * frac
        out[i] = s
        v = decay * 0.5 * (s + last)
        last = s
        buf[(idx + int(period)) % L] = v
        idx += 1
    return out


def noise(k):
    return rng.standard_normal(k)


def pink(k):
    w = rng.standard_normal(k)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return signal.lfilter(b, a, w) * 3.5


def bandpass(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, hi], btype='band', fs=SR, output='sos'), x)


def lowpass(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc, btype='low', fs=SR, output='sos'), x)


def highpass(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc, btype='high', fs=SR, output='sos'), x)


def env_adsr(k, a, d, s, r, hold=None):
    t = np.arange(k) / SR
    dur = k / SR
    rel_start = dur - r if hold is None else hold
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    e = e * np.clip((dur - t) / max(r, 1e-4), 0, 1) if hold is None else e * np.where(t > rel_start, np.exp(-(t - rel_start) / r), 1.0)
    return e


def fade(x, fin=0.01, fout=0.05):
    k = len(x)
    a = np.ones(k)
    m = min(secs(fin), k)
    a[:m] = np.linspace(0, 1, m)
    m2 = min(secs(fout), k)
    if m2:
        a[-m2:] *= np.linspace(1, 0, m2)
    return x * a


def make_ir(rt60, dur, lp_hz, seed, predelay=0.02):
    r = np.random.default_rng(seed)
    k = secs(dur)
    t = np.arange(k) / SR
    env = np.exp(-6.91 * t / rt60)
    irs = []
    for ch in range(2):
        x = r.standard_normal(k) * env
        x = lowpass(x, lp_hz)
        x[:secs(0.01)] *= np.linspace(0, 1, secs(0.01))
        x = np.concatenate([np.zeros(secs(predelay)), x])
        irs.append(x / np.sqrt((x ** 2).sum()))
    return irs


def reverb(bus, rt60, dur, lp, seed, wet, predelay=0.02):
    irl, irr = make_ir(rt60, dur, lp, seed, predelay)
    return signal.fftconvolve(bus.l, irl)[:len(bus.l)] * wet, signal.fftconvolve(bus.r, irr)[:len(bus.r)] * wet


# ------------------------------------------------------------ instruments --
def pad_voice(freq, dur, bright=1.0, detune=7.0, attack=1.2, release=1.5):
    """Warm analog-style pad: detuned saws through a gentle low-pass."""
    k = secs(dur)
    t = np.arange(k) / SR
    out = np.zeros(k)
    for dc in (-detune, 0.0, detune * 0.6):
        f = freq * 2 ** (dc / 1200)
        ph = rng.uniform(0, 1)
        saw = 2 * ((f * t + ph) % 1.0) - 1
        out += saw
    fc = np.full(k, min(900 + 1400 * bright, 9000.0)) * (1 + 0.15 * np.sin(2 * math.pi * 0.13 * t))
    out = svf(out / 3, fc, 0.8, 0)
    e = np.clip(t / attack, 0, 1) ** 1.5 * np.clip((dur - t) / release, 0, 1)
    return out * e


def strings(freq, dur, attack=1.8, release=2.0, bright=1.0):
    k = secs(dur)
    t = np.arange(k) / SR
    out = np.zeros(k)
    for dc in (-9, -3, 4, 10):
        f = freq * 2 ** (dc / 1200) * (1 + 0.0025 * np.sin(2 * math.pi * 5.1 * t + rng.uniform(0, 6)))
        ph = np.cumsum(f) / SR
        out += 2 * ((ph + rng.uniform(0, 1)) % 1.0) - 1
    out = svf(out / 4, np.full(k, 1600 + 1800 * bright), 0.7, 0)
    out = svf(out, np.full(k, 180.0), 0.7, 2)
    e = smooth(t / attack) * np.clip((dur - t) / release, 0, 1)
    return out * e


def guitar(freq, dur=3.5, bright=0.6, pick=0.5):
    period = SR / freq
    seed = rng.standard_normal(int(period) + 2)
    seed = lowpass(seed, 1500 + 5000 * bright, 1)
    seed -= seed.mean()
    x = ks_pluck(period, secs(dur), 0.9965, seed)
    # pick position comb and body resonance
    d = max(1, int(period * pick * 0.5))
    x = x - 0.5 * np.concatenate([np.zeros(d), x[:-d]])
    body = bandpass(x, 90, 250, 1) * 0.6 + bandpass(x, 400, 3500, 1)
    return fade(body * 0.8, 0.002, 0.3)


def keys(freq, dur=4.0, bright=0.5):
    """Soft felt-piano-like tone."""
    k = secs(dur)
    t = np.arange(k) / SR
    out = np.zeros(k)
    for h, a, tau in ((1, 1.0, 2.2), (2, 0.42 * bright, 1.3), (3, 0.18 * bright, 0.8), (4, 0.08 * bright, 0.5), (0.5, 0.12, 2.5)):
        out += a * np.sin(2 * math.pi * freq * h * t * (1 + 0.0004 * h * h)) * np.exp(-t / tau)
    return out * np.clip(t / 0.004, 0, 1) * np.clip((dur - t) / 0.3, 0, 1)


def bell(freq, dur=3.0, bright=1.0):
    k = secs(dur)
    t = np.arange(k) / SR
    out = np.zeros(k)
    for ratio, amp, tau in ((1.0, 1.0, 1.4), (2.0, 0.35, 0.9), (2.76, 0.45 * bright, 0.55), (5.40, 0.22 * bright, 0.25), (8.93, 0.1 * bright, 0.12)):
        if freq * ratio > 16000:
            continue
        out += amp * np.sin(2 * math.pi * freq * ratio * t + rng.uniform(0, 6.28)) * np.exp(-t / (tau * dur / 3.0))
    return out * np.clip(t / 0.003, 0, 1)


# ------------------------------------------------------------------ foley --
def crackle_grains(dur, rate, lo=1500, hi=9000, amp_sigma=0.8, decay=(0.0006, 0.004)):
    """A cloud of tiny clicks: dry leaves, crunch, fire pops."""
    k = secs(dur)
    out = np.zeros(k)
    count = rng.poisson(rate * dur)
    for _ in range(count):
        i = rng.integers(0, max(k - secs(0.02), 1))
        L = secs(rng.uniform(*decay) * 5)
        g = rng.standard_normal(L) * np.exp(-np.arange(L) / (SR * rng.uniform(*decay)))
        out[i:i + L] += g * min(rng.lognormal(0, amp_sigma), 3.5)
    return bandpass(out, lo, hi, 2)


def footstep_crunch(strength=1.0):
    dur = 0.32
    k = secs(dur)
    t = np.arange(k) / SR
    c = crackle_grains(dur, 900, 1200, 9000, 0.7) * np.exp(-t / 0.09) * np.clip(t / 0.01, 0, 1)
    thump = np.sin(2 * math.pi * 95 * t) * np.exp(-t / 0.035) * 0.9
    body = lowpass(noise(k), 600) * np.exp(-t / 0.05) * 0.5
    return (c * 0.9 + thump + body) * strength


def loon(dur=2.6):
    """A distant common loon wail."""
    k = secs(dur)
    t = np.arange(k) / SR
    f = np.where(t < 0.55, 640 + 360 * smooth(t / 0.55), 1000 + 20 * np.sin(2 * math.pi * 5.5 * t))
    f = np.where(t > 1.35, 1000 + 380 * smooth((t - 1.35) / 0.18), f)
    f = np.where(t > 1.9, 1380 - 400 * smooth((t - 1.9) / 0.5), f)
    ph = 2 * math.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    e = smooth(t / 0.25) * np.clip((dur - t) / 0.5, 0, 1) * (1 - 0.35 * smooth((t - 1.25) / 0.1) * (1 - smooth((t - 1.4) / 0.1)))
    return x * e


def chickadee():
    """'fee-bee'"""
    out = []
    for f0, f1, d in ((3950, 3900, 0.34), (3420, 3380, 0.3)):
        k = secs(d)
        t = np.arange(k) / SR
        f = f0 + (f1 - f0) * t / d
        x = np.sin(2 * math.pi * np.cumsum(f) / SR) * np.sin(math.pi * t / d) ** 0.6
        out += [x, np.zeros(secs(0.05))]
    return np.concatenate(out)


def whoosh(dur, f0, f1, q=1.2):
    k = secs(dur)
    t = np.arange(k) / SR
    fc = f0 * (f1 / f0) ** smooth(t / dur)
    x = svf(pink(k), fc, q, 1)
    return x * np.sin(math.pi * np.clip(t / dur, 0, 1)) ** 1.5


# ------------------------------------------------------------------- score --
def chord(names):
    return [hz(n(x)) for x in names]


def build():
    dry = Bus()      # foley, close
    hall = Bus()     # music and distant sounds -> large reverb
    air = Bus()      # wind / ambience (light reverb)
    T = np.arange(N) / SR

    # ---------------- 0-6: heavy stillness
    for f, g in zip(chord(['D2', 'A2', 'F3', 'D3']), (0.5, 0.35, 0.22, 0.3)):
        hall.add(0.0, pad_voice(f, 9.0, bright=0.15, attack=2.5, release=2.5), g * 0.22)
    hall.add(1.1, keys(hz(n('D3')), 4.5, 0.3), 0.16, -0.1)
    hall.add(3.3, keys(hz(n('F3')), 4.0, 0.3), 0.13, 0.1)
    hall.add(4.9, keys(hz(n('E3')), 3.0, 0.3), 0.1, 0.0)
    hall.add(2.3, chickadee(), 0.018, 0.6)
    hall.add(4.4, chickadee(), 0.014, -0.5)

    # ---------------- wind: follows the gust in the forest
    wk = N
    wt = T
    speed = 0.22 + 3.6 * smooth(ramp(wt, 5.9, 7.4)) + 0.9 * np.exp(-((wt - 9.2) ** 2) / 2.0) - 0.8 * smooth(ramp(wt, 12.2, 13.6))
    speed *= 1 - 0.75 * smooth(ramp(wt, 13.6, 14.6))           # memories: a light breeze
    speed += 0.6 * smooth(ramp(wt, 22.0, 24.0)) * (1 - smooth(ramp(wt, 32.0, 35.5)))  # altitude
    base = pink(wk)
    fc = 250 + 420 * speed + 180 * np.sin(2 * math.pi * 0.21 * wt) + 90 * np.sin(2 * math.pi * 0.53 * wt)
    wl = svf(base, fc, 0.9, 1)
    wr = svf(pink(wk), fc * 1.07, 0.9, 1)
    amp = (0.05 + 0.2 * np.clip(speed / 3.0, 0, 1.3)) * (1 - smooth(ramp(wt, 35.0, 36.5)))
    air.add_stereo(0.0, wl * amp * 0.36, wr * amp * 0.36)
    # the gust arriving, and the flurry swirling past the lens
    air.add(5.8, whoosh(2.8, 250, 1800, 1.4), 0.08, -0.6)
    air.add(11.9, whoosh(1.6, 500, 3200, 1.6), 0.065, -0.8)
    air.add(12.6, whoosh(1.3, 700, 3800, 1.6), 0.06, 0.8)
    air.add(22.1, whoosh(2.6, 180, 1400, 1.0), 0.14, 0.0)   # rising above the trees
    # leaves rustling in the gust
    rust_env = np.clip(speed - 0.4, 0, 3) / 3
    gusty = 0.55 + 0.45 * np.sin(2 * math.pi * 0.7 * T[:secs(DUR)] + 1.3) * np.sin(2 * math.pi * 0.23 * T[:secs(DUR)])
    rustle = crackle_grains(DUR, 260, 2500, 7500, 0.8) * rust_env[:secs(DUR)] * gusty * (1 - smooth(ramp(T[:secs(DUR)], 13.5, 14.5)))
    dry.add(0.0, rustle, 0.06, 0.2)

    # ---------------- 6-14: dissolve into light; minor opens to major
    for f, g in zip(chord(['Bb2', 'F3', 'D4']), (0.3, 0.22, 0.16)):
        hall.add(6.2, pad_voice(f, 3.6, bright=0.4, attack=1.5, release=1.2), g * 0.24)
    for f, g in zip(chord(['D2', 'A2', 'D3', 'F#3', 'A3', 'E4']), (0.4, 0.3, 0.25, 0.22, 0.18, 0.1)):
        hall.add(8.6, pad_voice(f, 6.4, bright=0.8, attack=1.6, release=1.4), g * 0.26)
    for f, g in zip(chord(['D3', 'A3', 'F#4']), (0.3, 0.25, 0.2)):
        hall.add(9.0, strings(f, 5.8, attack=2.2, release=1.2), g * 0.2)
    # glittering chimes as the goods turn to light (density follows the dissolve)
    pent = [n(x) for x in ('D6', 'E6', 'F#6', 'A6', 'B6', 'D7', 'E7')]
    for _ in range(95):
        tt = rng.uniform(6.5, 12.2)
        dens = math.exp(-((tt - 9.4) ** 2) / 4.0)
        if rng.uniform() > 0.25 + 0.75 * dens:
            continue
        hall.add(tt, bell(hz(rng.choice(pent)), rng.uniform(0.8, 1.8), 0.6), rng.uniform(0.006, 0.02), rng.uniform(-0.8, 0.8))
    # a rising arpeggio carries the flurry upward
    arp = ['D4', 'F#4', 'A4', 'D5', 'F#5', 'A5', 'D6']
    for k, x in enumerate(arp):
        hall.add(11.7 + k * 0.2, keys(hz(n(x)), 2.6, 0.6), 0.1, -0.6 + 0.2 * k)

    # ---------------- 14-22: memories
    prog = [(14.0, ['G2', 'D3', 'B3', 'D4']), (15.6, ['D2', 'A2', 'F#3', 'D4']), (17.2, ['B1', 'F#2', 'D3', 'F#3']),
            (18.8, ['A1', 'E2', 'C#3', 'E3']), (20.4, ['G1', 'D2', 'B2', 'D3']), (21.4, ['A1', 'E2', 'A2', 'C#3'])]
    for k, (t0, notes) in enumerate(prog):
        d = (prog[k + 1][0] if k + 1 < len(prog) else 22.4) - t0 + 1.2
        for f in chord(notes):
            hall.add(t0, pad_voice(f, d, bright=0.5, attack=0.9, release=1.1), 0.05)
    # nylon guitar: arpeggios and a simple melody
    arps = {14.0: ['G3', 'D4', 'G4', 'B4'], 15.6: ['D3', 'A3', 'D4', 'F#4'], 17.2: ['B2', 'F#3', 'B3', 'D4'],
            18.8: ['A2', 'E3', 'A3', 'C#4'], 20.4: ['G2', 'D3', 'G3', 'B3'], 21.4: ['A2', 'E3', 'A3', 'C#4']}
    for t0, notes in arps.items():
        for k, x in enumerate(notes * 2):
            tt = t0 + k * 0.2
            if tt > 22.3:
                break
            hall.add(tt, guitar(hz(n(x)), 3.0, 0.5), 0.3, -0.3 + 0.15 * (k % 4))
    melody = [(14.5, 'B4', 1.0), (15.1, 'A4', 0.6), (15.7, 'F#4', 0.9), (16.4, 'A4', 0.8), (17.3, 'D5', 1.3), (18.4, 'C#5', 0.6),
              (18.9, 'A4', 1.2), (20.1, 'B4', 0.8), (20.6, 'D5', 1.0), (21.5, 'E5', 1.2)]
    for tt, x, d in melody:
        hall.add(tt, guitar(hz(n(x)), 3.5, 0.7, 0.3), 0.4, 0.15)
    # footsteps in dry leaves (heel strikes of the walk)
    for tt in (16.45, 17.025, 17.6, 18.175, 18.75, 19.325, 19.9):
        g = 0.75 if 16.9 < tt < 20.0 else 0.35
        dry.add(tt, footstep_crunch(rng.uniform(0.8, 1.1)), 0.22 * g, 0.25 if round((tt - 16.45) / 0.575) % 2 else -0.15)
    # campfire by the lake
    fire_t0, fire_t1 = 19.2, 23.2
    fd = fire_t1 - fire_t0
    fk = secs(fd)
    ft = np.arange(fk) / SR
    fenv = smooth(ft / 0.8) * np.clip((fd - ft) / 1.2, 0, 1)
    bed = lowpass(pink(fk), 700) * (0.6 + 0.4 * np.abs(svf(noise(fk), np.full(fk, 3.0), 0.7, 0)) * 4)
    pops = crackle_grains(fd, 45, 900, 7000, 1.1, (0.0008, 0.006))
    dry.add(fire_t0, (bed * 0.15 + pops * 0.9) * fenv, 0.22, -0.05)
    hall.add(20.3, loon(), 0.035, 0.55)
    # water lapping at the shore
    lap = lowpass(noise(fk), 300) * (0.5 + 0.5 * np.sin(2 * math.pi * 0.7 * ft) ** 2)
    air.add(fire_t0, lap * fenv * 0.06, 1.0, -0.3)

    # ---------------- 22-32: the lift
    lift = [(22.0, ['D2', 'A2', 'D3', 'F#3', 'A3']), (23.6, ['C#2', 'A2', 'E3', 'A3']), (25.2, ['B1', 'F#2', 'D3', 'F#3', 'B3']),
            (26.8, ['G1', 'D2', 'B2', 'D3', 'G3']), (28.4, ['F#1', 'D2', 'A2', 'D3', 'F#3']), (29.8, ['E2', 'B2', 'E3', 'G3']),
            (31.0, ['A1', 'E2', 'A2', 'D3', 'E3']), (32.0, ['D2', 'A2', 'D3', 'F#3', 'A3', 'E4'])]
    for k, (t0, notes) in enumerate(lift):
        d = (lift[k + 1][0] if k + 1 < len(lift) else 36.8) - t0 + 1.4
        br = 0.6 + 0.4 * min(1.0, (t0 - 22) / 6)
        for f in chord(notes):
            hall.add(t0, pad_voice(f, d, bright=br, attack=0.8, release=1.3), 0.1)
            hall.add(t0, strings(f * 2, d, attack=1.2, release=1.5, bright=br), 0.09)
    soar = [(22.6, 'F#5', 1.4), (24.0, 'E5', 1.2), (25.3, 'F#5', 0.9), (26.2, 'A5', 1.6), (27.9, 'G5', 0.7), (28.6, 'F#5', 1.3),
            (30.0, 'E5', 1.0), (31.0, 'D5', 0.5), (31.5, 'E5', 0.7), (32.3, 'F#5', 3.2)]
    for tt, x, d in soar:
        f = hz(n(x))
        hall.add(tt, strings(f, d + 1.2, attack=0.35, release=1.1, bright=1.2), 0.2, 0.1)
        hall.add(tt, keys(f, d + 1.6, 0.7), 0.12, -0.1)
    for k in range(20):
        tt = 22.1 + k * 0.5
        x = ['D4', 'A4', 'F#4', 'A4', 'D5', 'A4'][k % 6]
        hall.add(tt, guitar(hz(n(x)), 2.6, 0.55), 0.16, -0.4 + 0.08 * (k % 6))

    # ---------------- 32-40: warmth, ember, silence
    for f in chord(['D3', 'A3', 'F#4', 'A4', 'D5']):
        hall.add(33.4, pad_voice(f, 4.8, bright=0.35, attack=1.2, release=3.0), 0.035)
    ek = secs(4.2)
    et = np.arange(ek) / SR
    hum = (np.sin(2 * math.pi * hz(n('D2')) * et) + 0.5 * np.sin(2 * math.pi * hz(n('A2')) * et) + 0.25 * np.sin(2 * math.pi * hz(n('D3')) * et))
    hum *= smooth(et / 1.0) * np.clip((3.9 - et) / 1.6, 0, 1)
    hall.add(35.8, hum, 0.06)
    hall.add(37.0, bell(hz(n('D3')), 4.0, 0.5), 0.1)
    hall.add(37.0, bell(hz(n('A3')), 3.2, 0.4), 0.05)
    embers = crackle_grains(3.8, 14, 1500, 8000, 1.0, (0.0006, 0.004)) * np.clip((3.6 - np.arange(secs(3.8)) / SR) / 1.2, 0, 1)
    dry.add(36.0, embers, 0.12)

    # ---------------- mix
    hl, hr = reverb(hall, 3.6, 5.5, 5200, 11, 0.55, 0.03)
    al, ar = reverb(air, 1.4, 2.5, 7000, 12, 0.25)
    dl, dr = reverb(dry, 0.9, 1.6, 6000, 13, 0.18)
    L = hall.l * 0.7 + hl + air.l + al + dry.l + dl
    R = hall.r * 0.7 + hr + air.r + ar + dry.r + dr
    L, R = L[:N], R[:N]
    t = np.arange(N) / SR
    master = np.clip(t / 0.8, 0, 1) * np.clip((DUR - 0.05 - t) / 1.2, 0, 1)
    L, R = L * master, R * master
    # loudness: the climax (the lift) sits around -16 dBFS RMS
    ref = slice(secs(24), secs(32))
    rms = math.sqrt(float((L[ref] ** 2 + R[ref] ** 2).mean() / 2))
    L, R = L * (0.16 / rms), R * (0.16 / rms)
    # look-ahead peak limiter at -1 dBFS
    L, R = limit(L, R, 0.89)
    return np.stack([L, R], axis=1)


@njit(cache=True)
def _release(pk, rel):
    out = np.empty_like(pk)
    e = 0.0
    for i in range(pk.shape[0]):
        e = pk[i] if pk[i] > e else e * rel + pk[i] * (1 - rel)
        out[i] = e
    return out


def limit(L, R, ceiling):
    from scipy.ndimage import maximum_filter1d
    la = secs(0.004)
    m = np.maximum(np.abs(L), np.abs(R))
    pk = maximum_filter1d(m, size=2 * la + 1)
    env = _release(pk, math.exp(-1 / (SR * 0.12)))
    g = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    g = np.convolve(g, np.ones(la) / la, mode='same')
    return np.clip(L * g, -ceiling, ceiling), np.clip(R * g, -ceiling, ceiling)


def _unused():
    return np.stack([L, R], axis=1)


def main(out):
    x = build()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    wavfile.write(out, SR, (x * 32767).astype(np.int16))
    print('wrote', out, x.shape)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'soundtrack.wav'))
