// A worn leather hiking boot as a signed-distance field.
// Local frame: sole bottom at y = 0, toe towards +Z, X lateral (right boot: +X is outside).
import * as THREE from 'three';
import { V3, capsule, ellipsoid, smin, smax, polygonize } from '../core/sdf.js';

// 2D footprint (x, z) -> signed distance, smooth blend of heel and forefoot discs
function footprint(x, z, grow = 0) {
  const heel = Math.hypot(x, z + 0.088) - (0.041 + grow);
  const ball = Math.hypot((x - 0.004) / 1.0, (z - 0.092) / 1.12) * 1.0 - (0.052 + grow);
  const arch = Math.hypot(x * 1.35 + 0.006, 0) - (0.038 + grow);
  const archBand = Math.max(arch, Math.abs(z - 0.005) - 0.07);
  return smin(smin(heel, ball, 0.07), archBand, 0.04);
}

export function buildBoot({ cell = 0.0016, mirror = false } = {}) {
  const toeSpring = (z) => (z > 0.1 ? (z - 0.1) * (z - 0.1) * 2.6 : 0);
  const sgnX = mirror ? -1 : 1;

  // --- parts (each: sdf, region)
  const parts = [];
  // outsole with a slight toe spring and a heel block
  const sole = (x, y, z) => {
    const yy = y - toeSpring(z);
    const d2 = footprint(x * sgnX, z, 0.004);
    const slab = Math.max(d2, Math.abs(yy - 0.016) - 0.016);
    return slab - 0.003;
  };
  parts.push([sole, 'sole']);
  // stitched welt band just above the sole
  const welt = (x, y, z) => {
    const yy = y - toeSpring(z);
    const d2 = footprint(x * sgnX, z, 0.0055);
    return Math.max(d2, Math.abs(yy - 0.036) - 0.0045) - 0.0015;
  };
  parts.push([welt, 'welt']);
  // leather upper: toe box, vamp, heel counter, shaft
  const toe = ellipsoid(V3(0.002 * sgnX, 0.052, 0.098), V3(0.05, 0.036, 0.086));
  const vamp = capsule(V3(0, 0.062, 0.07), V3(0, 0.1, -0.02), 0.046, 0.049);
  const heel = ellipsoid(V3(0, 0.074, -0.083), V3(0.043, 0.062, 0.047));
  const shaft = capsule(V3(0, 0.08, -0.058), V3(0, 0.172, -0.05), 0.051, 0.048);
  const upperRaw = (x, y, z) => {
    let d = smin(toe(x, y, z), vamp(x, y, z), 0.03);
    d = smin(d, heel(x, y, z), 0.03);
    d = smin(d, shaft(x, y, z), 0.025);
    // keep the upper inside the footprint near the sole
    const fp = footprint(x * sgnX, z, 0.0);
    d = smax(d, fp - 0.004 - Math.max(y - 0.04, 0) * 0.35, 0.01);
    return Math.max(d, 0.036 + toeSpring(z) - y); // sits on the welt
  };
  parts.push([upperRaw, 'upper']);
  // toe cap overlay (slightly proud)
  const toeCap = (x, y, z) => Math.max(toe(x, y, z) - 0.0018, 0.07 - z, 0.036 + toeSpring(z) - y);
  parts.push([toeCap, 'toecap']);
  // padded collar
  const collar = (x, y, z) => {
    const px = x, pz = z + 0.05, py = y - 0.168;
    const q = Math.hypot(px, pz) - 0.049;
    return Math.hypot(q, py) - 0.0125;
  };
  parts.push([collar, 'collar']);
  // tongue rising above the laces
  const tongue = capsule(V3(0, 0.1, 0.03), V3(0, 0.19, -0.015), 0.03, 0.028);
  parts.push([(x, y, z) => Math.max(tongue(x, y, z), Math.abs(x) - 0.03), 'tongue']);
  // laces: crossing cords up the front, with eyelets and speed hooks
  const laces = [];
  const hooks = [];
  const front = (y) => new THREE.Vector3(0, y, 0.066 - (y - 0.075) * 1.02); // front line of the vamp/shaft
  for (let i = 0; i < 6; i++) {
    const y0 = 0.08 + i * 0.019, y1 = y0 + 0.019;
    const a = front(y0), b = front(y1);
    const off = (p, s) => p.clone().add(new THREE.Vector3(s * 0.024, 0, 0.012));
    laces.push(capsule(off(a, -1), off(b, 1), 0.0026));
    laces.push(capsule(off(a, 1), off(b, -1), 0.0026));
    for (const s of [-1, 1]) hooks.push(ellipsoid(off(a, s * 1.12).add(new THREE.Vector3(0, 0, -0.002)), V3(0.0045, 0.0045, 0.0035)));
  }
  parts.push([(x, y, z) => laces.reduce((d, f) => Math.min(d, f(x, y, z)), 1e9), 'lace']);
  parts.push([(x, y, z) => hooks.reduce((d, f) => Math.min(d, f(x, y, z)), 1e9), 'metal']);
  // a knot and loops at the top
  const k0 = front(0.195).add(V3(0, 0, 0.016));
  parts.push([ellipsoid(k0, V3(0.008, 0.006, 0.006)), 'lace']);
  parts.push([capsule(k0, k0.clone().add(V3(0.03, -0.018, 0.012)), 0.0026), 'lace']);
  parts.push([capsule(k0, k0.clone().add(V3(-0.026, -0.03, 0.01)), 0.0026), 'lace']);

  const sdf = (x, y, z) => {
    let d = 1e9;
    for (const [f] of parts) d = Math.min(d, f(x, y, z));
    return d;
  };
  const REG = {
    sole: { c: [0.09, 0.065, 0.05], r: 0.85, m: 0, b: 0.6 },
    welt: { c: [0.26, 0.15, 0.08], r: 0.6, m: 0, b: 0.4 },
    upper: { c: [0.3, 0.215, 0.155], r: 0.55, m: 0, b: 1.0 },
    toecap: { c: [0.22, 0.16, 0.12], r: 0.42, m: 0, b: 0.8 },
    collar: { c: [0.2, 0.12, 0.07], r: 0.75, m: 0, b: 0.7 },
    tongue: { c: [0.36, 0.22, 0.12], r: 0.7, m: 0, b: 0.8 },
    lace: { c: [0.55, 0.3, 0.15], r: 0.8, m: 0, b: 0.3 },
    metal: { c: [0.42, 0.36, 0.26], r: 0.35, m: 0.9, b: 0.1 },
  };
  const attrs = (x, y, z) => {
    let best = 1e9, reg = 'upper';
    for (const [f, r] of parts) {
      const d = f(x, y, z);
      if (d < best - 1e-5) { best = d; reg = r; }
    }
    let R = REG[reg];
    // a pale midsole stripe and dark lugs along the outsole edge
    if (reg === 'sole') {
      const yy = y - (z > 0.1 ? (z - 0.1) * (z - 0.1) * 2.6 : 0);
      if (yy > 0.022) R = { c: [0.42, 0.38, 0.32], r: 0.7, m: 0, b: 0.3 };
      else if (yy < 0.012 && Math.sin(z * 260) > 0.2) R = { c: [0.05, 0.04, 0.035], r: 0.9, m: 0, b: 0.6 };
    }
    // scuffs: lighter, rougher leather at the toe and heel
    let scuff = 0;
    if (reg === 'upper' || reg === 'toecap') scuff = Math.max(0, Math.min(1, (z - 0.13) / 0.06)) * 0.45 + Math.max(0, Math.min(1, (-z - 0.1) / 0.03)) * 0.3;
    return { color: R.c.map((v) => Math.pow(v, 2.2)), aRM: [R.r, R.m, R.b, scuff] };
  };
  const bounds = new THREE.Box3(V3(-0.075, -0.004, -0.15), V3(0.075, 0.215, 0.205));
  const geometry = polygonize(sdf, bounds, cell, { attrs, project: 2 });
  return { geometry };
}
