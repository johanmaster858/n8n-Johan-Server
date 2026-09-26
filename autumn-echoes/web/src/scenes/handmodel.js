// A right hand pinching a leaf stem, modelled as a smooth signed-distance field.
// Local frame: wrist at the origin, fingers along +Y, palm facing +Z, thumb towards +X.
import * as THREE from 'three';
import { V3, capsule, ellipsoid, roundBox, smin, polygonize } from '../core/sdf.js';
import { vnoise3 } from '../core/rng.js';

const DEG = Math.PI / 180;

// [knuckle position, spread, segment lengths, radii at joints, flexion angles (MCP, PIP, DIP)]
const FINGERS = {
  index: { k: V3(0.027, 0.097, 0.0), spread: 0.1, len: [0.042, 0.025, 0.02], rad: [0.0094, 0.0084, 0.0077, 0.0066], flex: [38, 52, 28] },
  middle: { k: V3(0.0085, 0.101, 0.0), spread: 0.02, len: [0.046, 0.028, 0.021], rad: [0.0097, 0.0087, 0.0079, 0.0068], flex: [62, 86, 48] },
  ring: { k: V3(-0.0105, 0.097, -0.001), spread: -0.08, len: [0.043, 0.026, 0.02], rad: [0.009, 0.0081, 0.0074, 0.0064], flex: [70, 88, 46] },
  little: { k: V3(-0.0275, 0.088, -0.002), spread: -0.2, len: [0.034, 0.02, 0.018], rad: [0.008, 0.0072, 0.0066, 0.0058], flex: [74, 86, 40] },
};

function fingerChain(f) {
  let d = V3(f.spread, 1, 0).normalize();
  const axis = new THREE.Vector3().crossVectors(d, V3(0, 0, 1)).normalize();
  const pts = [f.k.clone()];
  const dirs = [];
  let p = f.k.clone();
  for (let s = 0; s < 3; s++) {
    d = d.clone().applyAxisAngle(axis, f.flex[s] * DEG);
    dirs.push(d.clone());
    p = p.clone().addScaledVector(d, f.len[s]);
    pts.push(p);
  }
  return { pts, dirs, axis };
}

/**
 * Builds the hand.  Returns { geometry, pinch (local Vector3), stemDir (local), thumbTip, indexTip }.
 */
export function buildHand({ cell = 0.0011 } = {}) {
  const prims = []; // [sdf, region, thinness]
  const add = (f, region, thin) => prims.push([f, region, thin]);

  // palm, with a fleshy pad on the palm side and the thenar/hypothenar eminences
  add(roundBox(V3(0.0, 0.052, 0.0), V3(0.039, 0.05, 0.0115), 0.0105), 'palm', 0.25);
  add(ellipsoid(V3(0.001, 0.058, 0.004), V3(0.034, 0.04, 0.0105)), 'palm', 0.2);
  add(ellipsoid(V3(0.02, 0.028, 0.009), V3(0.017, 0.026, 0.012), new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -0.5))), 'palm', 0.3);
  add(ellipsoid(V3(-0.027, 0.036, 0.006), V3(0.011, 0.03, 0.01)), 'palm', 0.35);
  // knuckle ridge on the back of the hand
  for (const f of Object.values(FINGERS)) add(ellipsoid(f.k.clone().add(V3(0, -0.004, -0.004)), V3(0.009, 0.011, 0.009)), 'palm', 0.3);
  // wrist and forearm
  add(capsule(V3(0.0, 0.012, -0.001), V3(0.0, -0.06, -0.004), 0.027, 0.029), 'wrist', 0.08);
  add(roundBox(V3(0.0, -0.03, -0.002), V3(0.03, 0.045, 0.018), 0.016), 'wrist', 0.08);
  add(capsule(V3(0.0, -0.05, -0.004), V3(0.0, -0.3, -0.012), 0.029, 0.036), 'arm', 0.05);

  // fingers
  const chains = {};
  const nails = [];
  for (const [name, f] of Object.entries(FINGERS)) {
    const c = fingerChain(f);
    chains[name] = c;
    for (let s = 0; s < 3; s++) add(capsule(c.pts[s], c.pts[s + 1], f.rad[s], f.rad[s + 1]), 'finger', 1.0);
    // nail on the back of the distal phalanx
    const d = c.dirs[2];
    const back = new THREE.Vector3().crossVectors(c.axis, d).normalize().negate(); // dorsal direction
    const nc = c.pts[3].clone().addScaledVector(d, -0.0075).addScaledVector(back, f.rad[3] * 0.72);
    const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(c.axis.clone(), d.clone(), back.clone().negate()));
    nails.push(ellipsoid(nc, V3(f.rad[3] * 0.86, 0.0082, 0.0017), q));
  }

  // thumb: metacarpal from the base of the palm, then two phalanges reaching the index pad
  const iTip = chains.index.pts[3].clone();
  const iDir = chains.index.dirs[2];
  const iPalm = new THREE.Vector3().crossVectors(chains.index.axis, iDir).normalize(); // palmar side of the index tip
  const pinch = iTip.clone().addScaledVector(iPalm, 0.0105).addScaledVector(iDir, -0.004);
  const cmc = V3(0.024, 0.02, 0.011);
  const mcp = cmc.clone().addScaledVector(V3(0.62, 0.62, 0.48).normalize(), 0.046);
  const tTip = pinch.clone().addScaledVector(iPalm, 0.0095).add(V3(0.004, -0.003, 0.0));
  const mid = mcp.clone().lerp(tTip, 0.55).add(V3(0.006, 0.0, 0.007));
  add(capsule(cmc, mcp, 0.0135, 0.0118), 'thumb', 0.7);
  add(capsule(mcp, mid, 0.0112, 0.0101), 'thumb', 0.9);
  add(capsule(mid, tTip, 0.01, 0.0088), 'thumb', 1.0);
  {
    const d = tTip.clone().sub(mid).normalize();
    const back = pinch.clone().sub(tTip).normalize().negate();
    const lat = new THREE.Vector3().crossVectors(d, back).normalize();
    const nc = tTip.clone().addScaledVector(d, -0.0075).addScaledVector(back, 0.0062);
    const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(lat, d, back.clone().negate()));
    nails.push(ellipsoid(nc, V3(0.0078, 0.0088, 0.0017), q));
  }

  const K = 0.0055;
  const sdf = (x, y, z) => {
    let d = 1e9;
    for (const [f] of prims) d = smin(d, f(x, y, z), K);
    for (const n of nails) d = smin(d, n(x, y, z), 0.0012);
    return d;
  };
  // per-vertex skin colour + translucency (thin parts glow when backlit)
  const skin = new THREE.Color(0.86, 0.62, 0.5), flush = new THREE.Color(0.86, 0.5, 0.44), nailC = new THREE.Color(0.93, 0.8, 0.74);
  const attrs = (x, y, z) => {
    let best = 1e9, thin = 0.2, wsum = 0, tsum = 0;
    for (const [f, , t] of prims) {
      const d = f(x, y, z);
      const w = Math.exp(-Math.max(d, 0) / 0.004);
      wsum += w;
      tsum += w * t;
      if (d < best) best = d;
    }
    thin = tsum / Math.max(wsum, 1e-6);
    let nailK = 0;
    for (const n of nails) nailK = Math.max(nailK, 1 - Math.min(Math.max(n(x, y, z) / 0.0009, 0), 1));
    // mottled flush at knuckles and fingertips
    const m = vnoise3(x * 260, y * 260, z * 260, 5);
    const c = skin.clone().lerp(flush, Math.min(1, thin * 0.35 + 0.25 * m));
    c.lerp(nailC, nailK);
    return { color: [c.r, c.g, c.b], aThin: [thin * (1 - nailK * 0.6)], aNail: [nailK] };
  };
  const bounds = new THREE.Box3(V3(-0.05, -0.2, -0.05), V3(0.09, 0.19, 0.075));
  const geometry = polygonize(sdf, bounds, cell, { attrs, project: 2 });
  return { geometry, pinch, stemDir: iDir.clone(), thumbTip: tTip, indexTip: iTip, sdf };
}
