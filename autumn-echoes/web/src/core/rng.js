// Seeded randomness and noise used everywhere, so every frame is reproducible.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Rng {
  constructor(seed = 1) {
    this.r = mulberry32(seed);
  }
  next() {
    return this.r();
  }
  range(a, b) {
    return a + (b - a) * this.r();
  }
  int(n) {
    return Math.floor(this.r() * n);
  }
  pick(arr) {
    return arr[this.int(arr.length)];
  }
  normal() {
    const u = Math.max(1e-9, this.r());
    const v = this.r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
}

// ---------------------------------------------------------------- noise --
// Hash-based value noise and a 3D simplex noise (Gustavson), deterministic.

function hash3i(x, y, z, seed) {
  let h = (x * 374761393 + y * 668265263 + z * 1274126177 + seed * 144665 + 1013904223) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h & 0xffff) / 65535;
}

const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);

export function vnoise2(x, y, seed = 0) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const u = fade(fx), v = fade(fy);
  const a = hash3i(ix, iy, 0, seed), b = hash3i(ix + 1, iy, 0, seed);
  const c = hash3i(ix, iy + 1, 0, seed), d = hash3i(ix + 1, iy + 1, 0, seed);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

export function vnoise3(x, y, z, seed = 0) {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = x - ix, fy = y - iy, fz = z - iz;
  const u = fade(fx), v = fade(fy), w = fade(fz);
  const n000 = hash3i(ix, iy, iz, seed), n100 = hash3i(ix + 1, iy, iz, seed);
  const n010 = hash3i(ix, iy + 1, iz, seed), n110 = hash3i(ix + 1, iy + 1, iz, seed);
  const n001 = hash3i(ix, iy, iz + 1, seed), n101 = hash3i(ix + 1, iy, iz + 1, seed);
  const n011 = hash3i(ix, iy + 1, iz + 1, seed), n111 = hash3i(ix + 1, iy + 1, iz + 1, seed);
  const x00 = n000 + (n100 - n000) * u, x10 = n010 + (n110 - n010) * u;
  const x01 = n001 + (n101 - n001) * u, x11 = n011 + (n111 - n011) * u;
  const y0 = x00 + (x10 - x00) * v, y1 = x01 + (x11 - x01) * v;
  return y0 + (y1 - y0) * w;
}

export function fbm2(x, y, oct = 4, seed = 0, lac = 2.0, gain = 0.5) {
  let a = 1, s = 0, n = 0;
  for (let i = 0; i < oct; i++) {
    s += a * vnoise2(x, y, seed + i * 17);
    n += a;
    x *= lac;
    y *= lac;
    a *= gain;
  }
  return s / n;
}

export function fbm3(x, y, z, oct = 4, seed = 0) {
  let a = 1, s = 0, n = 0;
  for (let i = 0; i < oct; i++) {
    s += a * vnoise3(x, y, z, seed + i * 31);
    n += a;
    x *= 2.03;
    y *= 2.03;
    z *= 2.03;
    a *= 0.5;
  }
  return s / n;
}

// Simplex 3D (after Stefan Gustavson), returns roughly [-1, 1]
const grad3 = [
  [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
];
const perm = new Uint8Array(512);
(function () {
  const r = mulberry32(1234);
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    const t = p[i];
    p[i] = p[j];
    p[j] = t;
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
})();

export function snoise3(xin, yin, zin) {
  const F3 = 1 / 3, G3 = 1 / 6;
  let n0, n1, n2, n3;
  const s = (xin + yin + zin) * F3;
  const i = Math.floor(xin + s), j = Math.floor(yin + s), k = Math.floor(zin + s);
  const t = (i + j + k) * G3;
  const x0 = xin - (i - t), y0 = yin - (j - t), z0 = zin - (k - t);
  let i1, j1, k1, i2, j2, k2;
  if (x0 >= y0) {
    if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
    else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
  } else {
    if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
    else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
    else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
  }
  const x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
  const x2 = x0 - i2 + 2 * G3, y2 = y0 - j2 + 2 * G3, z2 = z0 - k2 + 2 * G3;
  const x3 = x0 - 1 + 3 * G3, y3 = y0 - 1 + 3 * G3, z3 = z0 - 1 + 3 * G3;
  const ii = i & 255, jj = j & 255, kk = k & 255;
  const g0 = grad3[perm[ii + perm[jj + perm[kk]]] % 12];
  const g1 = grad3[perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12];
  const g2 = grad3[perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12];
  const g3 = grad3[perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12];
  let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
  n0 = t0 < 0 ? 0 : (t0 *= t0, t0 * t0 * (g0[0] * x0 + g0[1] * y0 + g0[2] * z0));
  let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
  n1 = t1 < 0 ? 0 : (t1 *= t1, t1 * t1 * (g1[0] * x1 + g1[1] * y1 + g1[2] * z1));
  let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
  n2 = t2 < 0 ? 0 : (t2 *= t2, t2 * t2 * (g2[0] * x2 + g2[1] * y2 + g2[2] * z2));
  let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
  n3 = t3 < 0 ? 0 : (t3 *= t3, t3 * t3 * (g3[0] * x3 + g3[1] * y3 + g3[2] * z3));
  return 32 * (n0 + n1 + n2 + n3);
}

// Divergence-free wind turbulence: curl of a vector potential made of simplex noise.
export function curl3(x, y, z, out) {
  const e = 0.07;
  const P = (a, b, c) => snoise3(a, b, c);
  const Q = (a, b, c) => snoise3(a + 31.4, b - 17.2, c + 9.7);
  const R = (a, b, c) => snoise3(a - 21.9, b + 43.1, c - 5.3);
  const dRdy = (R(x, y + e, z) - R(x, y - e, z)) / (2 * e);
  const dQdz = (Q(x, y, z + e) - Q(x, y, z - e)) / (2 * e);
  const dPdz = (P(x, y, z + e) - P(x, y, z - e)) / (2 * e);
  const dRdx = (R(x + e, y, z) - R(x - e, y, z)) / (2 * e);
  const dQdx = (Q(x + e, y, z) - Q(x - e, y, z)) / (2 * e);
  const dPdy = (P(x, y + e, z) - P(x, y - e, z)) / (2 * e);
  out[0] = dRdy - dQdz;
  out[1] = dPdz - dRdx;
  out[2] = dQdx - dPdy;
  return out;
}

export const clamp = (x, a = 0, b = 1) => (x < a ? a : x > b ? b : x);
export const smooth = (x) => {
  x = clamp(x);
  return x * x * (3 - 2 * x);
};
export const smoother = (x) => {
  x = clamp(x);
  return x * x * x * (x * (x * 6 - 15) + 10);
};
export const ramp = (t, a, b) => clamp((t - a) / (b - a));
export const lerp = (a, b, x) => a + (b - a) * x;
