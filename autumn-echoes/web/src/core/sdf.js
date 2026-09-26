// Signed-distance modelling: primitives, smooth blends and a surface-nets polygonizer.
// Organic shapes (a hand, a pair of boots) are written as distance functions in JS,
// then turned into smooth indexed meshes whose normals come from the SDF gradient.
import * as THREE from 'three';

export const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);

export function smin(a, b, k) {
  if (k <= 0) return Math.min(a, b);
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}
export function smax(a, b, k) {
  return -smin(-a, -b, k);
}

/** Capsule with radius interpolated from ra (at a) to rb (at b). */
export function capsule(a, b, ra, rb = ra) {
  const ba = b.clone().sub(a);
  const l2 = ba.lengthSq();
  const ax = a.x, ay = a.y, az = a.z, bx = ba.x, by = ba.y, bz = ba.z;
  const f = (x, y, z) => {
    const px = x - ax, py = y - ay, pz = z - az;
    const h = Math.min(Math.max((px * bx + py * by + pz * bz) / l2, 0), 1);
    const dx = px - bx * h, dy = py - by * h, dz = pz - bz * h;
    return Math.sqrt(dx * dx + dy * dy + dz * dz) - (ra + (rb - ra) * h);
  };
  f.box = new THREE.Box3().setFromPoints([a, b]).expandByScalar(Math.max(ra, rb));
  return f;
}

/** Ellipsoid (approximate distance) with centre c, semi-axes r, orientation q. */
export function ellipsoid(c, r, q = new THREE.Quaternion()) {
  const inv = q.clone().invert();
  const v = new THREE.Vector3();
  const f = (x, y, z) => {
    v.set(x - c.x, y - c.y, z - c.z).applyQuaternion(inv);
    const k0 = Math.sqrt((v.x / r.x) ** 2 + (v.y / r.y) ** 2 + (v.z / r.z) ** 2);
    const k1 = Math.sqrt((v.x / (r.x * r.x)) ** 2 + (v.y / (r.y * r.y)) ** 2 + (v.z / (r.z * r.z)) ** 2);
    return k1 > 0 ? (k0 * (k0 - 1)) / k1 : -Math.min(r.x, r.y, r.z);
  };
  const m = Math.max(r.x, r.y, r.z);
  f.box = new THREE.Box3(c.clone().subScalar(m), c.clone().addScalar(m));
  return f;
}

/** Rounded box with half extents h and radius rad, placed by position c and quaternion q. */
export function roundBox(c, h, rad, q = new THREE.Quaternion()) {
  const inv = q.clone().invert();
  const v = new THREE.Vector3();
  const f = (x, y, z) => {
    v.set(x - c.x, y - c.y, z - c.z).applyQuaternion(inv);
    const qx = Math.abs(v.x) - h.x + rad, qy = Math.abs(v.y) - h.y + rad, qz = Math.abs(v.z) - h.z + rad;
    const o = Math.sqrt(Math.max(qx, 0) ** 2 + Math.max(qy, 0) ** 2 + Math.max(qz, 0) ** 2);
    return o + Math.min(Math.max(qx, qy, qz), 0) - rad;
  };
  const m = Math.hypot(h.x, h.y, h.z);
  f.box = new THREE.Box3(c.clone().subScalar(m), c.clone().addScalar(m));
  return f;
}

/**
 * Surface nets.  sdf(x,y,z) -> distance; bounds: Box3; cell: voxel size.
 * attrs(x,y,z) -> optional {color:[r,g,b], a:[...]} evaluated per vertex.
 */
export function polygonize(sdf, bounds, cell, { attrs = null, project = 2 } = {}) {
  const min = bounds.min;
  const nx = Math.ceil((bounds.max.x - min.x) / cell) + 1;
  const ny = Math.ceil((bounds.max.y - min.y) / cell) + 1;
  const nz = Math.ceil((bounds.max.z - min.z) / cell) + 1;
  const field = new Float32Array(nx * ny * nz);
  const idx = (i, j, k) => (k * ny + j) * nx + i;
  // narrow band: blocks whose centre is far from the surface share its sign
  const B = 6;
  const reach = B * cell * 0.9 + 2 * cell;
  for (let bk = 0; bk < nz; bk += B) for (let bj = 0; bj < ny; bj += B) for (let bi = 0; bi < nx; bi += B) {
    const ei = Math.min(bi + B, nx), ej = Math.min(bj + B, ny), ek = Math.min(bk + B, nz);
    const dc = sdf(min.x + (bi + ei - 1) * 0.5 * cell, min.y + (bj + ej - 1) * 0.5 * cell, min.z + (bk + ek - 1) * 0.5 * cell);
    const far = Math.abs(dc) > reach;
    for (let k = bk; k < ek; k++) {
      const z = min.z + k * cell;
      for (let j = bj; j < ej; j++) {
        const y = min.y + j * cell;
        for (let i = bi; i < ei; i++) field[idx(i, j, k)] = far ? dc : sdf(min.x + i * cell, y, z);
      }
    }
  }
  const vertOf = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const cidx = (i, j, k) => (k * (ny - 1) + j) * (nx - 1) + i;
  const pos = [];
  const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const cv = new Float32Array(8);
  for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    let mask = 0;
    for (let c = 0; c < 8; c++) {
      const [a, b, d] = corners[c];
      cv[c] = field[idx(i + a, j + b, k + d)];
      if (cv[c] < 0) mask |= 1 << c;
    }
    if (mask === 0 || mask === 255) continue;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (const [e0, e1] of edges) {
      const v0 = cv[e0], v1 = cv[e1];
      if ((v0 < 0) === (v1 < 0)) continue;
      const t = v0 / (v0 - v1);
      const c0 = corners[e0], c1 = corners[e1];
      sx += c0[0] + (c1[0] - c0[0]) * t;
      sy += c0[1] + (c1[1] - c0[1]) * t;
      sz += c0[2] + (c1[2] - c0[2]) * t;
      n++;
    }
    vertOf[cidx(i, j, k)] = pos.length / 3;
    pos.push(min.x + (i + sx / n) * cell, min.y + (j + sy / n) * cell, min.z + (k + sz / n) * cell);
  }
  // quads across every sign-changing grid edge
  const index = [];
  const quad = (a, b, c, d, flip) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) index.push(a, c, b, a, d, c);
    else index.push(a, b, c, a, c, d);
  };
  for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const v0 = field[idx(i, j, k)] < 0;
    if (i < nx - 1 && v0 !== (field[idx(i + 1, j, k)] < 0))
      quad(vertOf[cidx(i, j - 1, k - 1)], vertOf[cidx(i, j, k - 1)], vertOf[cidx(i, j, k)], vertOf[cidx(i, j - 1, k)], !v0);
    if (j < ny - 1 && v0 !== (field[idx(i, j + 1, k)] < 0))
      quad(vertOf[cidx(i - 1, j, k - 1)], vertOf[cidx(i - 1, j, k)], vertOf[cidx(i, j, k)], vertOf[cidx(i, j, k - 1)], !v0);
    if (k < nz - 1 && v0 !== (field[idx(i, j, k + 1)] < 0))
      quad(vertOf[cidx(i - 1, j - 1, k)], vertOf[cidx(i, j - 1, k)], vertOf[cidx(i, j, k)], vertOf[cidx(i - 1, j, k)], !v0);
  }
  // project vertices onto the surface and take normals from the gradient
  const e = cell * 0.25;
  const P = new Float32Array(pos), N = new Float32Array(pos.length);
  const grad = (x, y, z, out) => {
    out[0] = sdf(x + e, y, z) - sdf(x - e, y, z);
    out[1] = sdf(x, y + e, z) - sdf(x, y - e, z);
    out[2] = sdf(x, y, z + e) - sdf(x, y, z - e);
    const l = Math.hypot(out[0], out[1], out[2]) || 1;
    out[0] /= l; out[1] /= l; out[2] /= l;
  };
  const g = [0, 0, 0];
  for (let v = 0; v < P.length; v += 3) {
    let x = P[v], y = P[v + 1], z = P[v + 2];
    for (let it = 0; it < project; it++) {
      const d = sdf(x, y, z);
      grad(x, y, z, g);
      x -= g[0] * d; y -= g[1] * d; z -= g[2] * d;
    }
    grad(x, y, z, g);
    P[v] = x; P[v + 1] = y; P[v + 2] = z;
    N[v] = g[0]; N[v + 1] = g[1]; N[v + 2] = g[2];
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(P, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(N, 3));
  if (attrs) {
    const count = P.length / 3;
    let spec = null, arrays = null;
    for (let v = 0; v < count; v++) {
      const a = attrs(P[v * 3], P[v * 3 + 1], P[v * 3 + 2], N[v * 3], N[v * 3 + 1], N[v * 3 + 2]);
      if (!spec) {
        spec = Object.keys(a);
        arrays = spec.map((key) => new Float32Array(count * a[key].length));
      }
      spec.forEach((key, s) => arrays[s].set(a[key], v * a[key].length));
    }
    if (spec) spec.forEach((key, s) => geo.setAttribute(key, new THREE.BufferAttribute(arrays[s], arrays[s].length / count)));
  }
  geo.setIndex(index);
  return geo;
}
