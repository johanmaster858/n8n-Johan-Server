// Procedural sugar-maple leaves: an atlas of 8 variants (colour + aux maps).
// Colour atlas: RGB albedo (sRGB) + alpha.  Aux atlas: R vein mask, G translucency,
// B roughness.  Every other part of the film samples these two canvases.
import { Rng, fbm2, vnoise2, clamp } from './rng.js';

export const LEAF_TILE = 256;
export const LEAF_COLS = 4;
export const LEAF_ROWS = 2;
export const LEAF_VARIANTS = LEAF_COLS * LEAF_ROWS;
export const LEAF_CY = 0.62;   // attachment point (fraction of tile height)
export const LEAF_R = 0.5;     // leaf radius (fraction of tile)

// right-half outline: [angle from the tip (deg, clockwise), radius (0..1)]
const OUTLINE = [
  [0, 1.0], [4, 0.83], [8, 0.86], [11, 0.73], [16, 0.77], [20, 0.58], [26, 0.43],
  [31, 0.47], [36, 0.7], [40, 0.63], [48, 0.97], [54, 0.76], [58, 0.81], [63, 0.63],
  [69, 0.67], [77, 0.46], [86, 0.49], [97, 0.62], [106, 0.49], [113, 0.52], [124, 0.33],
  [140, 0.18], [158, 0.08], [180, 0.055],
];

// colour schemes: [tip colour, body colour, centre colour, vein tint] (sRGB 0..1)
const SCHEMES = [
  { tip: [0.62, 0.05, 0.05], body: [0.80, 0.10, 0.06], centre: [0.93, 0.42, 0.08], vein: [0.95, 0.55, 0.25] }, // crimson
  { tip: [0.86, 0.14, 0.04], body: [0.92, 0.26, 0.05], centre: [0.98, 0.62, 0.10], vein: [1.0, 0.72, 0.3] }, // scarlet
  { tip: [0.85, 0.22, 0.04], body: [0.95, 0.45, 0.06], centre: [0.98, 0.70, 0.14], vein: [1.0, 0.8, 0.4] }, // orange
  { tip: [0.93, 0.50, 0.06], body: [0.96, 0.70, 0.12], centre: [0.90, 0.80, 0.22], vein: [1.0, 0.88, 0.5] }, // gold
  { tip: [0.90, 0.40, 0.06], body: [0.93, 0.62, 0.12], centre: [0.55, 0.62, 0.14], vein: [0.62, 0.7, 0.2] }, // turning
  { tip: [0.36, 0.03, 0.06], body: [0.55, 0.05, 0.07], centre: [0.78, 0.14, 0.06], vein: [0.85, 0.3, 0.2] }, // burgundy
  { tip: [0.72, 0.12, 0.04], body: [0.86, 0.30, 0.06], centre: [0.93, 0.55, 0.10], vein: [0.95, 0.6, 0.3] }, // spotted
  { tip: [0.80, 0.10, 0.05], body: [0.96, 0.62, 0.10], centre: [0.99, 0.80, 0.25], vein: [1.0, 0.85, 0.45] }, // yellow, red tips
];

function catmullPolar(points, samples) {
  // closed-ish open curve through keypoints in (angle, radius)
  const out = [];
  const n = points.length;
  for (let i = 0; i < n - 1; i++) {
    const p0 = points[Math.max(i - 1, 0)], p1 = points[i], p2 = points[i + 1], p3 = points[Math.min(i + 2, n - 1)];
    // keep lobe tips sharp: fewer, tighter samples near maxima
    for (let k = 0; k < samples; k++) {
      const t = k / samples, t2 = t * t, t3 = t2 * t;
      const f = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
    }
  }
  out.push(points[n - 1]);
  return out;
}

function leafPath(rng, cx, cy, R) {
  const wscale = rng.range(0.92, 1.08);
  const jitter = (arr, sgn) =>
    arr.map(([a, r], i) => {
      const tip = i === 0;
      return [a + (tip ? 0 : rng.range(-1.6, 1.6)), r * (tip ? rng.range(0.94, 1.03) : rng.range(0.93, 1.07))];
    });
  const right = catmullPolar(jitter(OUTLINE, 1), 6);
  const left = catmullPolar(jitter(OUTLINE, -1), 6);
  const pts = [];
  const toXY = ([a, r], s) => {
    const th = (a * Math.PI) / 180;
    return [cx + s * Math.sin(th) * r * R * wscale, cy - Math.cos(th) * r * R];
  };
  for (const p of right) pts.push(toXY(p, 1));
  for (let i = left.length - 1; i >= 0; i--) pts.push(toXY(left[i], -1));
  return pts;
}

function tipPoints(cx, cy, R) {
  // main vein targets: tips of the 5 lobes
  const tips = [];
  for (const [a, r] of [[0, 1.0], [48, 0.97], [-48, 0.97], [97, 0.62], [-97, 0.62]]) {
    const th = (a * Math.PI) / 180;
    tips.push([cx + Math.sin(th) * r * R * 0.96, cy - Math.cos(th) * r * R * 0.96]);
  }
  return tips;
}

export function makeLeafAtlas(seed = 7) {
  const T = LEAF_TILE;
  const W = T * LEAF_COLS, H = T * LEAF_ROWS;
  const color = document.createElement('canvas');
  color.width = W;
  color.height = H;
  const aux = document.createElement('canvas');
  aux.width = W;
  aux.height = H;
  const cctx = color.getContext('2d', { willReadFrequently: true });
  const actx = aux.getContext('2d', { willReadFrequently: true });
  const rng = new Rng(seed);
  for (let v = 0; v < LEAF_VARIANTS; v++) drawLeaf(cctx, actx, rng, v, (v % LEAF_COLS) * T, Math.floor(v / LEAF_COLS) * T, T);
  return { color, aux };
}

/** One big leaf (colour + aux canvases, T x T) for close-ups. */
export function makeHeroLeaf(T = 1024, variant = 0, seed = 11, { spots = false } = {}) {
  const mk = () => {
    const c = document.createElement('canvas');
    c.width = T;
    c.height = T;
    return c;
  };
  const color = mk(), aux = mk();
  drawLeaf(color.getContext('2d', { willReadFrequently: true }), aux.getContext('2d', { willReadFrequently: true }), new Rng(seed), variant, 0, 0, T, spots);
  return { color, aux };
}

function drawLeaf(cctx, actx, rng, v, ox, oy, T, allowSpots = true) {
  const px = T / 256; // pixel scale relative to the 256 px design size
  {
    const cx = ox + T * 0.5, cy = oy + T * LEAF_CY;
    const R = T * LEAF_R;
    const scheme = SCHEMES[v];
    const pts = leafPath(rng, cx, cy, R);

    // --- masks drawn with vector paths (shape, veins) on scratch canvases
    const shape = document.createElement('canvas');
    shape.width = T;
    shape.height = T;
    const sctx = shape.getContext('2d', { willReadFrequently: true });
    sctx.translate(-ox, -oy);
    sctx.fillStyle = '#fff';
    sctx.beginPath();
    pts.forEach(([x, y], i) => (i ? sctx.lineTo(x, y) : sctx.moveTo(x, y)));
    sctx.closePath();
    sctx.fill();
    // petiole (stem)
    const stemBend = rng.range(-0.25, 0.25);
    sctx.strokeStyle = '#fff';
    sctx.lineCap = 'round';
    sctx.lineWidth = T * 0.018;
    sctx.beginPath();
    sctx.moveTo(cx, cy - R * 0.02);
    sctx.quadraticCurveTo(cx + stemBend * R * 0.3, cy + R * 0.2, cx + stemBend * R * 0.45, cy + R * 0.34);
    sctx.stroke();

    const veins = document.createElement('canvas');
    veins.width = T;
    veins.height = T;
    const vctx = veins.getContext('2d', { willReadFrequently: true });
    vctx.translate(-ox, -oy);
    vctx.strokeStyle = '#fff';
    vctx.lineCap = 'round';
    const tips = tipPoints(cx, cy, R);
    for (let k = 0; k < tips.length; k++) {
      const [tx, ty] = tips[k];
      const main = k < 3 ? 1.0 : 0.7;
      // tapered main vein as a few segments
      const segs = 6;
      for (let s = 0; s < segs; s++) {
        const a = s / segs, b = (s + 1) / segs;
        vctx.lineWidth = T * 0.0105 * main * (1 - a * 0.8);
        vctx.beginPath();
        vctx.moveTo(cx + (tx - cx) * a, cy + (ty - cy) * a);
        vctx.lineTo(cx + (tx - cx) * b, cy + (ty - cy) * b);
        vctx.stroke();
      }
      // secondary veins towards the teeth
      const nsec = k < 3 ? 5 : 3;
      for (let s = 1; s <= nsec; s++) {
        const a = s / (nsec + 1);
        const bx = cx + (tx - cx) * a, by = cy + (ty - cy) * a;
        const dirx = tx - cx, diry = ty - cy;
        const len = Math.hypot(dirx, diry);
        for (const side of [-1, 1]) {
          const ang = side * rng.range(0.55, 0.85);
          const ca = Math.cos(ang), sa = Math.sin(ang);
          const ux = (dirx * ca - diry * sa) / len, uy = (dirx * sa + diry * ca) / len;
          const l = len * rng.range(0.18, 0.3) * (1 - a * 0.5);
          vctx.lineWidth = T * 0.0045 * main;
          vctx.beginPath();
          vctx.moveTo(bx, by);
          vctx.quadraticCurveTo(bx + ux * l * 0.5 + uy * l * 0.08, by + uy * l * 0.5 - ux * l * 0.08, bx + ux * l, by + uy * l);
          vctx.stroke();
        }
      }
    }
    const shapeData = sctx.getImageData(0, 0, T, T).data;
    const veinData = vctx.getImageData(0, 0, T, T).data;

    // --- per-pixel colour
    const img = cctx.createImageData(T, T);
    const aimg = actx.createImageData(T, T);
    const d = img.data, ad = aimg.data;
    const spots = [];
    let nspots = v === 6 ? 7 : rng.next() < 0.35 ? 2 : 0;
    if (!allowSpots) nspots = 0;
    for (let s = 0; s < nspots; s++) {
      const a = rng.range(-2.2, 2.2), r = rng.range(0.15, 0.7);
      spots.push([T * 0.5 + Math.sin(a) * r * R, T * LEAF_CY - Math.cos(a) * r * R, rng.range(2.5, 6.5) * px]);
    }
    const sseed = 100 + v * 7;
    for (let y = 0; y < T; y++) {
      for (let x = 0; x < T; x++) {
        const i = (y * T + x) * 4;
        const m = shapeData[i + 3] / 255;
        if (m <= 0) {
          d[i + 3] = 0;
          ad[i + 3] = 0;
          continue;
        }
        const dx = x - T * 0.5, dy = T * LEAF_CY - y;
        const rr = Math.hypot(dx, dy) / R;
        // gradient centre -> body -> tips
        const g1 = clamp(rr / 0.45);
        const g2 = clamp((rr - 0.45) / 0.55);
        const n1 = fbm2(x * 0.035 / px, y * 0.035 / px, 4, sseed);
        const n2 = vnoise2(x * 0.12 / px, y * 0.12 / px, sseed + 3);
        let c = [0, 0, 0];
        for (let k = 0; k < 3; k++) {
          const a = scheme.centre[k] + (scheme.body[k] - scheme.centre[k]) * g1;
          c[k] = a + (scheme.tip[k] - a) * clamp(g2 + (n1 - 0.5) * 0.9);
        }
        // mottling
        const mott = 0.86 + 0.26 * n1 + 0.06 * (n2 - 0.5);
        c = c.map((q) => q * mott);
        // veins: lighter on the upper surface
        const vm = veinData[i + 3] / 255;
        for (let k = 0; k < 3; k++) c[k] = c[k] * (1 - vm * 0.55) + scheme.vein[k] * vm * 0.55;
        // tar spots with a yellow halo
        for (const [sx, sy, sr] of spots) {
          const ds = Math.hypot(x - sx, y - sy);
          if (ds < sr * 2.2) {
            const halo = clamp(1 - (ds - sr) / (sr * 1.2));
            c = c.map((q, k) => q * (1 - 0.35 * halo) + [0.85, 0.62, 0.12][k] * 0.35 * halo);
            if (ds < sr) c = c.map((q) => q * (0.18 + 0.2 * (ds / sr)));
          }
        }
        // dry, darker margin
        let edge = 0;
        const eo = Math.max(2, Math.round(2 * px));
        if (m < 1 || x + eo >= T || x < eo || shapeData[i + 3 + 4 * eo] < 200 || shapeData[i + 3 - 4 * eo] < 200 ||
            (y + eo < T && shapeData[i + 3 + T * 4 * eo] < 200) || (y >= eo && shapeData[i + 3 - T * 4 * eo] < 200)) edge = 1;
        if (edge) c = c.map((q, k) => q * 0.72 + [0.35, 0.16, 0.05][k] * 0.1);
        d[i] = clamp(c[0]) * 255;
        d[i + 1] = clamp(c[1]) * 255;
        d[i + 2] = clamp(c[2]) * 255;
        d[i + 3] = m * 255;
        // aux: vein mask, translucency (thin lamina transmits more), roughness
        ad[i] = vm * 255;
        ad[i + 1] = clamp(0.95 - vm * 0.55 - edge * 0.3 - (n2 - 0.5) * 0.15) * 255;
        ad[i + 2] = clamp(0.55 + (n1 - 0.5) * 0.3 + vm * 0.15) * 255;
        ad[i + 3] = m * 255;
      }
    }
    cctx.putImageData(img, ox, oy);
    actx.putImageData(aimg, ox, oy);
  }
}

// ------------------------------------------------------- leaf-cluster cards --
export const CLUSTER_TILE = 512;
export const CLUSTER_COLS = 2;
export const CLUSTER_ROWS = 2;

/**
 * Canopy cards: each tile is a sprig of ~50 overlapping leaves with twigs, composited
 * from the leaf atlas (colour and aux), deeper leaves darker, so one quad reads as foliage.
 */
export function makeClusterAtlas(leaf, seed = 19) {
  const T = CLUSTER_TILE;
  const W = T * CLUSTER_COLS, H = T * CLUSTER_ROWS;
  const mk = () => {
    const c = document.createElement('canvas');
    c.width = W;
    c.height = H;
    return c;
  };
  const color = mk(), aux = mk();
  const cg = color.getContext('2d'), ag = aux.getContext('2d');
  const rng = new Rng(seed);
  const LT = LEAF_TILE;
  // colour mood per tile: crimson, orange-gold, yellow, mixed red
  const moods = [[0, 0, 1, 5, 1, 6], [2, 3, 2, 7, 3, 1], [3, 7, 3, 4, 7, 2], [0, 1, 2, 5, 3, 6, 7]];
  for (let v = 0; v < CLUSTER_COLS * CLUSTER_ROWS; v++) {
    const ox = (v % CLUSTER_COLS) * T, oy = Math.floor(v / CLUSTER_COLS) * T;
    const cx = ox + T / 2, cy = oy + T / 2;
    // twigs radiating from below the centre
    const twigs = [];
    for (let k = 0; k < 6; k++) {
      const a = -Math.PI / 2 + rng.range(-1.3, 1.3);
      const len = T * rng.range(0.25, 0.42);
      twigs.push([cx + rng.range(-20, 20), cy + T * 0.1, a, len]);
    }
    for (const g of [cg, ag]) {
      g.save();
      g.beginPath();
      g.rect(ox, oy, T, T);
      g.clip();
      g.strokeStyle = g === cg ? '#3b2a1d' : 'rgb(0,40,230)';
      g.lineCap = 'round';
      for (const [x, y, a, len] of twigs) {
        g.lineWidth = 5;
        g.beginPath();
        g.moveTo(x, y);
        g.quadraticCurveTo(x + Math.cos(a) * len * 0.5 + 12, y + Math.sin(a) * len * 0.5, x + Math.cos(a) * len, y + Math.sin(a) * len);
        g.stroke();
      }
      g.restore();
    }
    const n = 54;
    const leaves = [];
    for (let k = 0; k < n; k++) {
      const [tx, ty, ta, tlen] = twigs[rng.int(twigs.length)];
      const along = rng.range(0.3, 1.05);
      const px = tx + Math.cos(ta) * tlen * along + rng.normal() * 22;
      const py = ty + Math.sin(ta) * tlen * along + rng.normal() * 22;
      leaves.push({ px, py, rot: ta + Math.PI / 2 + rng.range(-1.2, 1.2), s: rng.range(0.34, 0.56) * T / LT, variant: rng.pick(moods[v]),
        depth: rng.next() });
    }
    leaves.sort((a, b) => a.depth - b.depth);
    for (const L of leaves) {
      const sx = (L.variant % LEAF_COLS) * LT, sy = Math.floor(L.variant / LEAF_COLS) * LT;
      const bright = 0.5 + 0.5 * L.depth;
      for (const [g, src] of [[cg, leaf.color], [ag, leaf.aux]]) {
        g.save();
        g.beginPath();
        g.rect(ox, oy, T, T);
        g.clip();
        g.translate(L.px, L.py);
        g.rotate(L.rot);
        g.scale(L.s, L.s);
        if (g === cg) g.filter = `brightness(${bright.toFixed(3)})`;
        // leaf attachment point (tile centre-bottom) sits on the twig
        g.drawImage(src, sx, sy, LT, LT, -LT / 2, -LT * LEAF_CY, LT, LT);
        g.restore();
      }
    }
  }
  return { color, aux };
}
