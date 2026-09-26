// Procedural PBR textures drawn in the page (no external assets).
import * as THREE from 'three';
import { Rng, fbm2, vnoise2, clamp } from './rng.js';

export function canvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/** Build maps from a per-pixel function returning {c:[r,g,b] (sRGB 0..1), h, rough, metal}. */
export function buildMaps(w, h, fn, { normalStrength = 2.0, wrap = true } = {}) {
  const albedo = canvas(w, h), rough = canvas(w, h), normal = canvas(w, h);
  const ai = albedo.getContext('2d').createImageData(w, h);
  const ri = rough.getContext('2d').createImageData(w, h);
  const ni = normal.getContext('2d').createImageData(w, h);
  const height = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = fn(x, y);
      const i = y * w + x;
      height[i] = o.h ?? 0;
      ai.data[i * 4] = clamp(o.c[0]) * 255;
      ai.data[i * 4 + 1] = clamp(o.c[1]) * 255;
      ai.data[i * 4 + 2] = clamp(o.c[2]) * 255;
      ai.data[i * 4 + 3] = 255;
      // three.js roughnessMap uses G, metalnessMap uses B
      ri.data[i * 4] = 255;
      ri.data[i * 4 + 1] = clamp(o.rough ?? 0.6) * 255;
      ri.data[i * 4 + 2] = clamp(o.metal ?? 0) * 255;
      ri.data[i * 4 + 3] = 255;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const xl = wrap ? (x - 1 + w) % w : Math.max(x - 1, 0), xr = wrap ? (x + 1) % w : Math.min(x + 1, w - 1);
      const yu = wrap ? (y - 1 + h) % h : Math.max(y - 1, 0), yd = wrap ? (y + 1) % h : Math.min(y + 1, h - 1);
      const dx = (height[y * w + xr] - height[y * w + xl]) * normalStrength;
      const dy = (height[yd * w + x] - height[yu * w + x]) * normalStrength;
      const n = [-dx, dy, 1];
      const l = Math.hypot(n[0], n[1], n[2]);
      const i = (y * w + x) * 4;
      ni.data[i] = (n[0] / l * 0.5 + 0.5) * 255;
      ni.data[i + 1] = (n[1] / l * 0.5 + 0.5) * 255;
      ni.data[i + 2] = (n[2] / l * 0.5 + 0.5) * 255;
      ni.data[i + 3] = 255;
    }
  }
  albedo.getContext('2d').putImageData(ai, 0, 0);
  rough.getContext('2d').putImageData(ri, 0, 0);
  normal.getContext('2d').putImageData(ni, 0, 0);
  const tex = (c, srgb) => {
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 4;
    return t;
  };
  return { map: tex(albedo, true), roughnessMap: tex(rough, false), metalnessMap: tex(rough, false), normalMap: tex(normal, false), canvas: albedo };
}

const mix3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

// ------------------------------------------------------------ painted steel --
export function safePaint(size = 1024, { door = false, seed = 3 } = {}) {
  const rng = new Rng(seed);
  const scratches = [];
  for (let i = 0; i < 70; i++) {
    const x = rng.range(0, size), y = rng.range(0, size), a = rng.range(-0.6, 0.6) + (rng.next() < 0.5 ? 0 : Math.PI / 2);
    scratches.push([x, y, Math.cos(a), Math.sin(a), rng.range(10, 90), rng.range(0.6, 1.6)]);
  }
  const paint = [0.2, 0.29, 0.23];
  const steel = [0.52, 0.52, 0.5];
  const rust = [0.36, 0.17, 0.07];
  const gold = [0.84, 0.64, 0.26];
  return buildMaps(size, size, (x, y) => {
    const u = x / size, v = y / size;
    const edge = Math.min(u, 1 - u, v, 1 - v);
    const n1 = fbm2(u * 9, v * 9, 4, seed);
    const n2 = fbm2(u * 40, v * 40, 3, seed + 5);
    let c = mix3(paint, [0.26, 0.33, 0.27], n1 * 0.8);
    c = c.map((q) => q * (0.85 + 0.3 * n2));
    let h = 0.5 + 0.02 * n2, rough = 0.58 + 0.22 * n1, metal = 0.0;
    // worn edges expose bare steel, with rust at the rim of the chips
    const wear = clamp((0.055 + 0.05 * (n1 - 0.5) * 2 - edge) / 0.03);
    const chipN = fbm2(u * 60, v * 60, 3, seed + 9);
    const chip = wear * (chipN > 0.42 ? 1 : 0) + (chipN > 0.8 ? 0.8 : 0) * clamp(1 - edge * 6);
    if (chip > 0.5) {
      const rim = clamp((chipN - 0.42) / 0.06);
      c = mix3(rust, steel, rim);
      metal = rim;
      rough = 0.35 + 0.5 * (1 - rim);
      h = 0.42;
    }
    // scratches through the paint
    for (const [sx, sy, dx, dy, len, w] of scratches) {
      const px = x - sx, py = y - sy;
      const along = px * dx + py * dy, across = Math.abs(-px * dy + py * dx);
      if (along > 0 && along < len && across < w) {
        c = mix3(c, steel, 0.8);
        metal = 0.8;
        rough = 0.3;
        h = 0.45;
      }
    }
    // rust streaks running down from the lower half
    const streak = fbm2(u * 30, v * 3, 3, seed + 21);
    if (v > 0.55 && streak > 0.62) {
      const k = clamp((streak - 0.62) / 0.2) * clamp((v - 0.55) / 0.3);
      c = mix3(c, rust, k * 0.7);
      rough = rough + (0.95 - rough) * k;
      metal *= 1 - k;
    }
    // gold pinstripes on the door panel
    if (door) {
      const inset = Math.min(Math.abs(edge - 0.085), Math.abs(edge - 0.1));
      if (inset < 0.0025 && chip < 0.5) {
        c = mix3(c, gold, 0.9);
        metal = 0.9;
        rough = 0.3;
      }
    }
    return { c, h, rough, metal };
  }, { normalStrength: 6 });
}

// ------------------------------------------------------------------- leather --
export function leather(size = 1024, base = [0.36, 0.2, 0.11], seed = 11, { scuff = 1.0 } = {}) {
  return buildMaps(size, size, (x, y) => {
    const u = x / size, v = y / size;
    // pebbled grain: sharp-ish value noise at high frequency
    const g = vnoise2(u * 220, v * 220, seed) * 0.6 + vnoise2(u * 470, v * 470, seed + 1) * 0.4;
    const grain = Math.pow(g, 1.6);
    const n1 = fbm2(u * 6, v * 6, 4, seed + 2);
    // a few soft creases where the leather has flexed
    const crease = Math.pow(clamp(1 - Math.abs(fbm2(u * 5, v * 5, 3, seed + 3) - 0.5) * 30), 2) * clamp((fbm2(u * 2, v * 2, 2, seed + 8) - 0.55) * 5);
    const edge = Math.min(u, 1 - u, v, 1 - v);
    const scuffK = scuff * clamp((0.06 - edge) / 0.06) * clamp((fbm2(u * 30, v * 30, 3, seed + 4) - 0.35) * 3);
    let c = base.map((q) => q * (0.75 + 0.5 * n1) * (0.92 + 0.12 * grain));
    c = mix3(c, c.map((q) => q * 0.7), crease * 0.35);
    c = mix3(c, c.map((q) => Math.min(1, q * 1.7 + 0.08)), scuffK * 0.8);
    // saddle stitching inset from the panel edges
    const inset = Math.abs(edge - 0.03);
    const stitch = inset < 0.0022 && ((u + v) * 90) % 1 < 0.62 ? 1 : 0;
    const groove = clamp(1 - Math.abs(edge - 0.03) / 0.006);
    c = mix3(c, c.map((q) => q * 0.72), groove * 0.5);
    if (stitch) c = [0.62, 0.5, 0.34];
    return { c, h: grain * 0.35 - crease * 0.3 - groove * 0.3 + stitch * 0.4, rough: 0.48 + 0.25 * (1 - grain) + scuffK * 0.2, metal: 0 };
  }, { normalStrength: 3.5 });
}

// ---------------------------------------------------------- paper page edges --
export function pageEdges(w = 256, h = 512, seed = 17) {
  return buildMaps(w, h, (x, y) => {
    const u = x / w, v = y / h;
    const line = 0.5 + 0.5 * Math.sin(v * h * 1.3 + fbm2(u * 4, v * 40, 2, seed) * 3.0);
    const n = fbm2(u * 8, v * 8, 3, seed + 1);
    const grime = clamp((Math.min(u, 1 - u) - 0.02) / 0.1);
    let c = [0.86, 0.8, 0.66].map((q) => q * (0.9 + 0.08 * line) * (0.88 + 0.16 * n));
    c = mix3([0.55, 0.45, 0.32], c, 0.5 + 0.5 * grime);
    return { c, h: line * 0.15, rough: 0.92, metal: 0 };
  }, { normalStrength: 2 });
}

// ----------------------------------------------------------- book spine/cover --
export function bookCover(base, seed, { spine = false } = {}) {
  const size = 512;
  return buildMaps(size, size, (x, y) => {
    const u = x / size, v = y / size;
    const g = vnoise2(u * 160, v * 160, seed) * 0.6 + vnoise2(u * 330, v * 330, seed + 1) * 0.4;
    const n = fbm2(u * 5, v * 5, 4, seed + 2);
    const edge = Math.min(u, 1 - u, v, 1 - v);
    let c = base.map((q) => q * (0.78 + 0.42 * n) * (0.93 + 0.1 * g));
    let rough = 0.55 + 0.2 * (1 - g), metal = 0, h = g * 0.25;
    c = mix3(c, c.map((q) => q * 1.5 + 0.06), clamp((0.04 - edge) / 0.04) * clamp((n - 0.3) * 2) * 0.7);
    // gold-foil bands (on the spine they run across)
    const bandCoord = spine ? v : u;
    for (const b of [0.12, 0.16, 0.84, 0.88]) {
      if (Math.abs(bandCoord - b) < 0.006) {
        c = mix3(c, [0.8, 0.62, 0.28], 0.85);
        metal = 0.85;
        rough = 0.35;
        h = 0.1;
      }
    }
    if (spine && Math.abs(v - 0.5) < 0.1 && Math.abs(u - 0.5) < 0.3) {
      // a worn gilt title block
      const k = fbm2(u * 60, v * 60, 2, seed + 7) > 0.5 ? 0.7 : 0.25;
      c = mix3(c, [0.78, 0.6, 0.27], k);
      metal = k;
      rough = 0.4;
    }
    return { c, h, rough, metal };
  }, { normalStrength: 2.5 });
}

// ----------------------------------------------------------- brass and steel --
export function metal(base, seed, { rough = 0.32, patina = 0.25, size = 512 } = {}) {
  const rng = new Rng(seed);
  const scratches = [];
  for (let i = 0; i < 90; i++) scratches.push([rng.range(0, size), rng.range(0, size), rng.range(0, Math.PI), rng.range(8, 70)]);
  return buildMaps(size, size, (x, y) => {
    const u = x / size, v = y / size;
    const n = fbm2(u * 7, v * 7, 4, seed);
    const pat = clamp((fbm2(u * 14, v * 14, 3, seed + 3) - 0.55) * 4) * patina;
    let c = base.map((q) => q * (0.85 + 0.25 * n));
    c = mix3(c, [0.2, 0.26, 0.2], pat * 0.6);
    let r = rough + 0.15 * n + pat * 0.4, h = 0;
    for (const [sx, sy, a, len] of scratches) {
      const px = x - sx, py = y - sy, dx = Math.cos(a), dy = Math.sin(a);
      const along = px * dx + py * dy;
      if (along > 0 && along < len && Math.abs(-px * dy + py * dx) < 0.7) {
        r = 0.18;
        h = -0.3;
        c = c.map((q) => Math.min(1, q * 1.15));
      }
    }
    return { c, h, rough: r, metal: 1 - pat * 0.6 };
  }, { normalStrength: 2 });
}

// --------------------------------------------------------------------- bark --
export function bark(kind = 'birch', seed = 31, w = 512, h = 1024) {
  const rng = new Rng(seed);
  const marks = [];
  if (kind === 'birch') {
    for (let i = 0; i < 260; i++) marks.push([rng.range(0, w), rng.range(0, h), rng.range(6, 38), rng.range(1.0, 3.2)]);
  }
  return buildMaps(w, h, (x, y) => {
    const u = x / w, v = y / h;
    if (kind === 'birch') {
      const n = fbm2(u * 4, v * 16, 4, seed);
      let c = [0.86, 0.84, 0.78].map((q) => q * (0.86 + 0.18 * n));
      let hh = 0.1 * n, rough = 0.7;
      // horizontal lenticels
      for (const [mx, my, len, th] of marks) {
        let dx = x - mx;
        if (dx > w / 2) dx -= w;
        if (dx < -w / 2) dx += w;
        if (Math.abs(dx) < len && Math.abs(y - my) < th * (1 - Math.abs(dx) / len)) {
          c = [0.12, 0.1, 0.09];
          hh = -0.4;
          rough = 0.9;
        }
      }
      // thin dark streaks (old branch scars) and peeling pinkish-cream areas
      const patch = fbm2(u * 2.5, v * 16, 3, seed + 4);
      if (patch > 0.7) {
        const k = clamp((patch - 0.7) / 0.06);
        c = mix3(c, [0.13, 0.11, 0.1], k * 0.9);
        hh -= 0.4 * k;
        rough = 0.95;
      } else if (patch < 0.32) {
        c = mix3(c, [0.82, 0.66, 0.55], 0.3);
      }
      const grey = fbm2(u * 6, v * 3, 3, seed + 12);
      c = mix3(c, [0.6, 0.58, 0.54], clamp((grey - 0.5) * 2) * 0.5);
      return { c, h: hh, rough, metal: 0 };
    }
    // maple: grey-brown with vertical plates and fissures
    const fiss = Math.abs(fbm2(u * 14, v * 2.5, 4, seed) - 0.5) * 2;
    const plates = Math.pow(clamp(fiss * 1.6), 0.5);
    const n = fbm2(u * 20, v * 20, 3, seed + 2);
    let c = mix3([0.12, 0.1, 0.08], [0.42, 0.38, 0.33], plates).map((q) => q * (0.8 + 0.35 * n));
    // lichen
    const lich = fbm2(u * 8, v * 8, 3, seed + 6);
    if (lich > 0.66) c = mix3(c, [0.55, 0.6, 0.45], clamp((lich - 0.66) * 6) * 0.6);
    return { c, h: plates * 0.8 + n * 0.1, rough: 0.85, metal: 0 };
  }, { normalStrength: 5 });
}

// --------------------------------------------------------------- forest soil --
export function soil(size = 512, seed = 41) {
  return buildMaps(size, size, (x, y) => {
    const u = x / size, v = y / size;
    const n = fbm2(u * 12, v * 12, 5, seed);
    const twig = Math.abs(fbm2(u * 40, v * 6, 2, seed + 3) - 0.5) < 0.012 ? 1 : 0;
    let c = mix3([0.1, 0.06, 0.035], [0.28, 0.18, 0.1], n);
    if (twig) c = [0.3, 0.2, 0.12];
    return { c, h: n + twig * 0.5, rough: 0.95, metal: 0 };
  }, { normalStrength: 4 });
}

// ---------------------------------------------------------------- wool knit --
export function knit(size = 512, base = [0.84, 0.77, 0.66], seed = 61, { cols = 16, rows = 22 } = {}) {
  return buildMaps(size, size, (x, y) => {
    const u = x / size, v = y / size;
    const cu = u * cols, cv = v * rows;
    const fu = cu - Math.floor(cu), fv = cv - Math.floor(cv);
    // every other column is a purl rib, recessed
    const rib = Math.floor(cu) % 2 === 0;
    // knit columns: two legs of a "V" per stitch
    const leg = Math.abs(fu - 0.5) * 2; // 0 centre .. 1 edge
    const vshape = Math.abs(leg - (1 - fv) * 0.9 - 0.05);
    const strand = Math.exp(-(vshape * vshape) / 0.02);
    const fib = fbm2(u * 90, v * 90, 3, seed);
    let h = rib ? 0.55 * strand + 0.25 : 0.2 + 0.25 * Math.exp(-((fv - 0.5) ** 2) / 0.05);
    h += 0.08 * fib;
    const shade = 0.72 + 0.34 * h;
    const c = base.map((q) => q * shade * (0.92 + 0.12 * fib));
    return { c, h: h * 0.9, rough: 0.95, metal: 0 };
  }, { normalStrength: 7 });
}

// -------------------------------------------------------------------- denim --
export function denim(w = 512, h = 1024, seed = 71) {
  return buildMaps(w, h, (x, y) => {
    const u = x / w, v = y / h;
    // 3/1 twill: diagonal wales
    const tw = 0.5 + 0.5 * Math.sin((x + y) * 0.9);
    const weft = 0.5 + 0.5 * Math.sin(y * 2.4);
    const n = fbm2(u * 6, v * 12, 4, seed);
    const slub = fbm2(u * 60, v * 150, 2, seed + 3);
    const fade = Math.pow(clamp((n - 0.35) * 1.8), 2);
    let c = mix3([0.2, 0.28, 0.44], [0.5, 0.58, 0.7], fade * 0.8 + 0.05 * slub);
    c = c.map((q) => q * (0.84 + 0.2 * tw * weft));
    return { c, h: tw * 0.35 + weft * 0.1, rough: 0.9, metal: 0 };
  }, { normalStrength: 3 });
}
