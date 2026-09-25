"""Procedural soundtrack for "Breathing Room", locked to the same timeline as the picture.

    python audio.py out.wav
"""
import math
import os
import sys

import numpy as np
from numba import njit
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import timeline as TL     # noqa: E402

SR = 48000
DUR = 40.0
N = int(DUR * SR)
rng = np.random.default_rng(31)


def secs(t):
    return int(round(t * SR))


def hz(note):
    """MIDI note -> Hz."""
    return 440.0 * 2 ** ((note - 69) / 12)


class Bus:
    def __init__(self):
        self.l = np.zeros(N + SR * 6)
        self.r = np.zeros(N + SR * 6)

    def add(self, t0, x, gain=1.0, pan=0.0):
        """Mix mono signal x at time t0 (s) with equal-power pan in [-1, 1]."""
        i = secs(t0)
        if i < 0:
            x = x[-i:]
            i = 0
        n = min(len(x), len(self.l) - i)
        if n <= 0:
            return
        a = (pan + 1) * math.pi / 4
        self.l[i:i + n] += x[:n] * gain * math.cos(a)
        self.r[i:i + n] += x[:n] * gain * math.sin(a)

    def add_stereo(self, t0, l, r, gain=1.0):
        i = secs(t0)
        n = min(len(l), len(self.l) - i)
        self.l[i:i + n] += l[:n] * gain
        self.r[i:i + n] += r[:n] * gain


@njit(cache=True)
def svf(x, fc, q, mode):
    """State-variable filter with per-sample cutoff. mode 0=low 1=band 2=high."""
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


def exp_env(n, tau, attack=0.002):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / tau)


def noise(n):
    return rng.standard_normal(n)


def pink(n):
    w = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return signal.lfilter(b, a, w) * 3.5


def bandpass(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], btype='band', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def lowpass(x, fc, order=2):
    sos = signal.butter(order, fc, btype='low', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def highpass(x, fc, order=2):
    sos = signal.butter(order, fc, btype='high', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def additive(freq, dur, harm=12, rolloff=1.0, detune_cents=0.0, phase=None):
    n = secs(dur)
    t = np.arange(n) / SR
    f = freq * 2 ** (detune_cents / 1200)
    out = np.zeros(n)
    for k in range(1, harm + 1):
        if f * k > 15000:
            break
        ph = rng.uniform(0, 2 * math.pi) if phase is None else phase
        out += np.sin(2 * math.pi * f * k * t + ph) / (k ** rolloff)
    return out


def bell(freq, dur=3.0, bright=1.0):
    n = secs(dur)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for ratio, amp, tau in ((1.0, 1.0, 1.4), (2.0, 0.35, 0.9), (2.76, 0.45 * bright, 0.55),
                            (5.40, 0.22 * bright, 0.25), (8.93, 0.10 * bright, 0.12)):
        if freq * ratio > 16000:
            continue
        out += amp * np.sin(2 * math.pi * freq * ratio * t + rng.uniform(0, 6.28)) * np.exp(-t / (tau * dur / 3.0))
    return out * np.clip(t / 0.003, 0, 1)


def fade(x, fin=0.01, fout=0.05):
    n = len(x)
    a = np.ones(n)
    k = min(secs(fin), n)
    a[:k] = np.linspace(0, 1, k)
    k2 = min(secs(fout), n)
    if k2:
        a[-k2:] *= np.linspace(1, 0, k2)
    return x * a


def make_ir(rt60, dur, lowpass_hz, seed):
    r = np.random.default_rng(seed)
    n = secs(dur)
    t = np.arange(n) / SR
    env = np.exp(-6.91 * t / rt60)
    irs = []
    for ch in range(2):
        x = r.standard_normal(n) * env
        x = lowpass(x, lowpass_hz)
        x[:secs(0.012)] *= np.linspace(0, 1, secs(0.012))
        irs.append(x / np.sqrt((x ** 2).sum()))
    return irs


def reverb(bus, rt60, dur, lp, seed, wet_gain):
    irl, irr = make_ir(rt60, dur, lp, seed)
    wl = signal.fftconvolve(bus.l, irl)[:len(bus.l)]
    wr = signal.fftconvolve(bus.r, irr)[:len(bus.r)]
    return wl * wet_gain, wr * wet_gain


def screen_pan(p, t):
    C = TL.camera(t)
    rel = p - C[0:3]
    z = rel @ C[9:12]
    if z < 0.1:
        return float(np.clip(np.sign(rel @ C[3:6]), -1, 1)) * 0.9
    x = rel @ C[3:6] / z / C[12]
    return float(np.clip(x, -1, 1)) * 0.85


def build():
    dry = Bus()          # close, intimate
    hall = Bus()         # sent to the cathedral reverb
    tech = Bus()         # sent to the dark, short reverb of the emblem

    # ------------------------------------------------ 0-11 s: cold white light
    # glassy, colourless tones that fade as the window is covered
    tt = np.arange(secs(12.0)) / SR
    lv = np.array([TL.light_level(t) for t in np.arange(0, 12.0, 0.05)])
    lvi = np.interp(tt, np.arange(0, 12.0, 0.05), lv)
    glass = np.zeros_like(tt)
    for f, a in ((hz(76), 0.5), (hz(83), 0.35), (hz(88), 0.18)):
        glass += a * np.sin(2 * math.pi * f * tt) * (0.8 + 0.2 * np.sin(2 * math.pi * 0.23 * tt + f))
    glass *= np.clip(tt / 2.5, 0, 1) ** 2 * lvi ** 1.3
    hall.add(0.0, glass, 0.12)
    # room air
    air = lowpass(pink(N), 1800.0)
    room_env = np.ones(N)
    room_env[:secs(1.5)] = np.linspace(0, 1, secs(1.5))
    room_env[secs(10.8):secs(11.6)] = np.linspace(1, 0, secs(11.6) - secs(10.8))
    room_env[secs(11.6):secs(14.2)] = 0
    room_env[secs(14.2):secs(16.0)] = np.linspace(0, 1, secs(16.0) - secs(14.2))
    room_env[secs(35.95):] = 0
    dry.add(0.0, air * room_env, 0.006)

    # clock: accelerating tick-tock, 3 -> 10.9 s
    t = 3.05
    k = 0
    while t < 10.9:
        rate = 1.1 + 8.5 * ((t - 3.0) / 7.9) ** 1.8
        n = secs(0.06)
        click = bandpass(noise(n), 1800 if k % 2 else 2600, 6000) * exp_env(n, 0.004, 0.0005)
        ping = np.sin(2 * math.pi * (2100 if k % 2 else 2500) * np.arange(n) / SR) * exp_env(n, 0.012, 0.0005)
        g = 0.10 + 0.10 * (t - 3.0) / 7.9
        dry.add(t, click * 0.7 + ping * 0.25, g, pan=-0.25 if k % 2 else 0.25)
        t += 1.0 / rate
        k += 1

    # anxious drone and a high beating cluster, building to the closure
    n = secs(8.2)
    tt = np.arange(n) / SR
    grow = (tt / 8.2) ** 1.6
    drone = additive(hz(33), 8.2, harm=24, rolloff=1.0, phase=0.0) + additive(hz(33) * 1.004, 8.2, harm=24, rolloff=1.0, phase=0.0)
    drone = lowpass(drone, 380.0) * grow
    hall.add(2.9, fade(drone, 1.0, 0.08), 0.05)
    cluster = (np.sin(2 * math.pi * hz(94) * tt) + np.sin(2 * math.pi * hz(95) * tt) * 0.8) * grow ** 1.5
    hall.add(2.9, fade(cluster, 1.0, 0.05), 0.016)

    # every cube: a whoosh past the camera, then a stone thud in the window
    for k in range(TL.NB):
        t0, t1 = TL.LAUNCH[k], TL.LAND[k]
        dur = t1 - t0
        n = secs(dur + 0.05)
        tt = np.arange(n) / SR
        s = np.clip(tt / dur, 0, 1)
        # loudest as it passes the camera (early in its flight)
        amp = np.exp(-((s - 0.12) / 0.10) ** 2) + 0.2 * np.exp(-((s - 0.4) / 0.3) ** 2)
        fc = 300 + 1500 * np.exp(-((s - 0.12) / 0.12) ** 2) + 200 * (1 - s)
        w = svf(noise(n), fc, 2.6, 1) * amp
        p0 = TL.START[k]
        pan = float(np.clip(p0[0] / 5.0, -1, 1)) * 0.8
        dry.add(t0, fade(w, 0.01, 0.05), 0.030 + 0.020 * (k / TL.NB), pan=pan)
        # thud: pitch-dropping sine + stone click
        n2 = secs(1.2)
        tt2 = np.arange(n2) / SR
        f = 70 + 60 * np.exp(-tt2 / 0.05)
        th = np.sin(2 * math.pi * np.cumsum(f) / SR) * exp_env(n2, 0.16, 0.002)
        th += lowpass(noise(n2), 2500.0) * exp_env(n2, 0.02, 0.001) * 0.6
        pan2 = screen_pan(TL.settled_pos(k), t1)
        last = (k == TL.CENTER)
        hall.add(t1, th, 0.16 if not last else 0.0, pan=pan2)

    # the final cube closes the window: a heavy slam, then silence
    t_close = TL.LAND[TL.CENTER]
    n = secs(3.0)
    tt = np.arange(n) / SR
    f = 42 + 50 * np.exp(-tt / 0.08)
    slam = np.sin(2 * math.pi * np.cumsum(f) / SR) * exp_env(n, 0.55, 0.002)
    slam += lowpass(noise(n), 900.0) * exp_env(n, 0.25, 0.001) * 0.8
    slam += bandpass(noise(n), 1500, 5000) * exp_env(n, 0.03, 0.0005) * 0.5
    hall.add(t_close, slam, 0.55)
    dry.add(t_close, slam, 0.25)

    # ------------------------------------------ 14-19 s: letting go of one thing
    lo, sl, fa, d0, d1 = TL.removal_timing(TL.CENTER)
    n = secs(sl - lo + 0.3)
    tt = np.arange(n) / SR
    jitter = lowpass(noise(n), 14.0)
    jitter /= np.abs(jitter).max() + 1e-9
    creak = bandpass(noise(n), 120, 700) * (0.4 + 0.6 * np.abs(jitter)) * (tt / (sl - lo)) ** 1.5
    creak += np.sin(2 * math.pi * 55 * tt) * 0.3 * (tt / (sl - lo))
    hall.add(lo, fade(creak, 0.3, 0.2), 0.10)
    n = secs(fa - sl + 0.2)
    tt = np.arange(n) / SR
    scrape = svf(noise(n), 500 + 900 * tt / (fa - sl), 2.5, 1) * np.clip(tt / 0.2, 0, 1)
    hall.add(sl, fade(scrape, 0.05, 0.15), 0.12)
    # the fall
    n = secs(1.8)
    tt = np.arange(n) / SR
    fall = svf(noise(n), 1600 * np.exp(-tt / 0.9) + 250, 1.2, 1) * np.sin(np.pi * np.clip(tt / 1.8, 0, 1))
    hall.add(fa, fall, 0.06)
    # a breath of air as the first coloured light comes in
    n = secs(2.6)
    tt = np.arange(n) / SR
    breath = svf(pink(n), 500 + 900 * np.sin(np.pi * tt / 2.6), 0.9, 1) * np.sin(np.pi * tt / 2.6) ** 2
    hall.add(sl + 0.2, breath, 0.07)
    # turning into light: crystalline grains + a few bells
    for i in range(160):
        tg = d0 + (d1 - d0) * rng.random() ** 0.8
        f = rng.choice([hz(n_) for n_ in (86, 88, 90, 93, 95, 98, 100, 102)])
        n = secs(0.12)
        g = np.sin(2 * math.pi * f * np.arange(n) / SR) * exp_env(n, 0.03, 0.002)
        hall.add(tg, g, 0.02, pan=rng.uniform(-0.3, 0.3))
    for tb, note in ((d0 + 0.05, 74), (d0 + 0.6, 81), (d0 + 1.2, 78), (d0 + 1.8, 86)):
        hall.add(tb, bell(hz(note), 4.0, 0.8), 0.09, pan=rng.uniform(-0.2, 0.2))

    # ------------------------------------------------- warm pad from 15.2 s on
    chords = [
        (15.2, 28.6, [50, 57, 62, 66, 69]),            # D
        (28.4, 30.8, [47, 54, 59, 62, 66, 73]),        # Bm(add9)
        (30.6, 33.0, [43, 50, 55, 59, 62, 69]),        # G(add9)
        (32.8, 36.0, [50, 57, 62, 66, 69, 74, 78, 81]),  # D, blooming
    ]
    pad_l = np.zeros(N)
    pad_r = np.zeros(N)
    for (a, b, notes) in chords:
        dur = b - a
        n = secs(dur)
        tt = np.arange(n) / SR
        env = np.clip(tt / (3.5 if a < 16 else 1.2), 0, 1) ** 2 * np.clip((dur - tt) / 1.0, 0, 1)
        for j, m in enumerate(notes):
            v = sum(additive(hz(m), dur, harm=10, rolloff=1.6, detune_cents=dc) for dc in (-5, 0, 6))
            v = lowpass(v, 2200.0 + 400 * j)
            # slow swell towards the light
            swell = np.clip((np.arange(n) / SR + a - 15.0) / 18.0, 0.15, 1.0)
            v = v * env * swell / (1 + 0.35 * j)
            pn = ((j % 2) * 2 - 1) * 0.35
            i0 = secs(a)
            pad_l[i0:i0 + n] += v * math.cos((pn + 1) * math.pi / 4)
            pad_r[i0:i0 + n] += v * math.sin((pn + 1) * math.pi / 4)
    # crescendo into the dazzling light, cut at 36 s
    tt = np.arange(N) / SR
    cres = 1.0 + 0.35 * np.clip((tt - 25.5) / 2.5, 0, 1) + 1.6 * np.clip((tt - 32.5) / 3.3, 0, 1) ** 2
    cres[secs(36.0):] = 0
    hall.add_stereo(0.0, pad_l * cres, pad_r * cres, 0.045)
    dry.add_stereo(0.0, pad_l * cres, pad_r * cres, 0.018)

    # ------------------------------------------- 19-26 s: the chain reaction
    scale = [74, 76, 78, 81, 83, 86, 88, 90, 93]
    for k in range(TL.NB):
        if k == TL.CENTER:
            continue
        lo, sl, fa, d0, d1 = TL.removal_timing(k)
        pan = screen_pan(TL.settled_pos(k), lo)
        note = scale[int(rng.integers(len(scale)))]
        hall.add(sl, bell(hz(note), 3.0, 0.6), 0.05, pan=pan)
        for i in range(10):
            tg = d0 + (d1 - d0) * rng.random()
            f = hz(note + 12) * rng.choice([1.0, 1.5, 2.0])
            n = secs(0.08)
            g = np.sin(2 * math.pi * f * np.arange(n) / SR) * exp_env(n, 0.02, 0.002)
            hall.add(tg, g, 0.006, pan=pan)
    # shimmering air as the light fills the nave
    n = secs(20.0)
    tt = np.arange(n) / SR
    sh = highpass(noise(n), 6000.0) * (0.5 + 0.5 * np.sin(2 * math.pi * 7.0 * tt)) * np.clip((tt - 3.0) / 6.0, 0, 1)
    sh[secs(20.0) - secs(4.0):] *= 1.0
    sh *= np.clip((20.0 - tt) / 0.02, 0, 1)
    hall.add(16.0, sh, 0.004)

    # ------------------------------------------- 26-33 s: rising through the light
    n = secs(7.0)
    tt = np.arange(n) / SR
    rise = svf(pink(n), 300 + 2500 * (tt / 7.0) ** 1.5, 1.0, 1) * np.sin(np.pi * tt / 7.0) ** 1.5
    hall.add(25.6, rise, 0.030)
    # 33-36 s: the swell of light (reverse-cymbal-like air)
    n = secs(3.0)
    tt = np.arange(n) / SR
    sw = highpass(noise(n), 2500.0) * (tt / 3.0) ** 3
    dry.add(33.0, sw * np.clip((3.0 - tt) / 0.004, 0, 1), 0.05)

    # ---------------------------------------------- 36 s: collapse into shadow
    n = secs(4.0)
    tt = np.arange(n) / SR
    f = 30 + 55 * np.exp(-tt / 0.25)
    boom = np.sin(2 * math.pi * np.cumsum(f) / SR) * exp_env(n, 0.9, 0.003)
    boom += lowpass(noise(n), 300.0) * exp_env(n, 0.35, 0.002) * 0.7
    tech.add(36.0, boom, 0.34)
    # dark drone with a low beating and a faint electric hum
    n = secs(3.6)
    tt = np.arange(n) / SR
    dark = (np.sin(2 * math.pi * 36.7 * tt) + np.sin(2 * math.pi * 73.4 * tt) * 0.6 + np.sin(2 * math.pi * 73.9 * tt) * 0.5)
    dark *= np.clip(tt / 1.2, 0, 1)
    hum = sum(np.sin(2 * math.pi * 50 * h * tt) / h for h in (1, 2, 3, 5))
    wind = lowpass(pink(n), 250.0)
    tech.add(36.2, fade(dark * 0.5 + hum * 0.05 + wind * 0.5, 0.2, 0.01), 0.10)
    # heartbeat before the power comes
    for tb in (36.9, 37.15, 37.62, 37.85):
        n = secs(0.5)
        tt = np.arange(n) / SR
        hb = np.sin(2 * math.pi * (55 + 30 * np.exp(-tt / 0.03)) * tt) * exp_env(n, 0.09, 0.004)
        tech.add(tb, hb, 0.30 if tb in (36.9, 37.62) else 0.18)
    # data ticks: the intelligence waking up
    for i in range(40):
        tg = 36.8 + 1.2 * rng.random()
        n = secs(0.015)
        g = np.sin(2 * math.pi * rng.uniform(2500, 6000) * np.arange(n) / SR) * exp_env(n, 0.003, 0.0003)
        tech.add(tg, g, 0.02 * (tg - 36.6), pan=rng.uniform(-0.7, 0.7))

    # ---------------------------------------------- 38 s: ignition
    flick = [0.0, 0.03, 0.06, 0.10, 0.13, 0.17, 0.21, 0.28]
    on = [1, 0, 1, 0.2, 1, 0.4, 1, 1]
    n = secs(0.3)
    tt = np.arange(n) / SR
    gate = np.zeros(n)
    for i in range(len(flick)):
        a = secs(flick[i])
        b = secs(flick[i + 1]) if i + 1 < len(flick) else n
        gate[a:b] = on[i]
    buzz = signal.square(2 * math.pi * 120 * tt) * 0.5 + np.sign(np.sin(2 * math.pi * 60 * tt)) * 0.3
    buzz = lowpass(buzz, 3500.0) + bandpass(noise(n), 2000, 8000) * 0.4
    tech.add(38.0, buzz * lowpass(gate, 400.0), 0.22)
    # the hit
    n = secs(3.0)
    tt = np.arange(n) / SR
    f = 38 + 90 * np.exp(-tt / 0.06)
    hit = np.sin(2 * math.pi * np.cumsum(f) / SR) * exp_env(n, 0.7, 0.002)
    hit = np.tanh(hit * 2.2) * 0.8
    hit += lowpass(noise(n), 4000.0) * exp_env(n, 0.12, 0.001) * 0.6
    zap_f = 3200 * np.exp(-tt / 0.08) + 180
    zap = np.sin(2 * math.pi * np.cumsum(zap_f) / SR) * exp_env(n, 0.18, 0.001)
    tech.add(38.26, hit, 0.36)
    tech.add(38.26, zap, 0.10)
    # pulsing power, locked to the emblem's heartbeat (1.25 Hz)
    n = secs(1.6)
    tt = np.arange(n) / SR
    beat = 0.5 + 0.5 * np.cos(2 * math.pi * 1.25 * tt)
    bass = sum(additive(hz(38), 1.6, harm=18, rolloff=1.0, detune_cents=dc) for dc in (-7, 0, 7))
    bass = svf(bass, 180 + 900 * beat ** 2, 1.8, 0) * (0.55 + 0.45 * beat ** 2)
    neon = (signal.square(2 * math.pi * 120 * tt) * 0.3)
    neon = lowpass(neon, 2000.0) * 0.25
    high = (np.sin(2 * math.pi * hz(86) * tt) + 0.7 * np.sin(2 * math.pi * hz(93) * tt)) * 0.12
    blips = np.zeros(n)
    for i in range(int(1.6 * 2.6)):
        a = secs(i / 2.6 + 0.1)
        m = secs(0.05)
        if a + m < n:
            blips[a:a + m] += np.sin(2 * math.pi * hz(98 + (i % 3) * 5) * np.arange(m) / SR) * exp_env(m, 0.012, 0.001)
    power = bass * 0.9 + neon + high + blips * 0.25
    power *= np.clip(tt / 0.05, 0, 1)
    tech.add(38.28, power, 0.22)
    dry.add(38.28, power, 0.08)

    # ------------------------------------------------------------- mixdown
    hl, hr = reverb(hall, 3.8, 5.0, 5000.0, 1, 0.55)
    tl, tr = reverb(tech, 1.6, 2.5, 2500.0, 2, 0.35)
    L = dry.l + hall.l * 0.35 + hl + tech.l * 0.8 + tl
    R = dry.r + hall.r * 0.35 + hr + tech.r * 0.8 + tr
    L = L[:N]
    R = R[:N]
    # 11-14 s is darkness and silence: let only the slam's tail breathe out
    t = np.arange(N) / SR
    g = np.ones(N)
    g[(t > 11.3) & (t < 14.0)] = np.clip(1 - (t[(t > 11.3) & (t < 14.0)] - 11.3) / 1.3, 0, 1) ** 2
    # hard cut to black at 39.8 s (5 ms fade to avoid a click)
    cut = secs(39.8)
    g[cut:cut + secs(0.005)] *= np.linspace(1, 0, secs(0.005))
    g[cut + secs(0.005):] = 0
    L *= g
    R *= g
    # master: soft limit to -1 dBFS
    peak = max(np.abs(L).max(), np.abs(R).max())
    L, R = L / peak * 1.25, R / peak * 1.25
    L, R = np.tanh(L) / np.tanh(1.25) * 0.89, np.tanh(R) / np.tanh(1.25) * 0.89
    return np.stack([L, R], 1)


def main(out):
    x = build()
    wavfile.write(out, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    print('wrote', out, x.shape)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'build', 'soundtrack.wav'))
