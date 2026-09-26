// Birch and maple trees: irregular trunks with root flare, a few main branches,
// and crowns built from leaf-cluster cards (cheap, dense, backlit foliage).
import * as THREE from 'three';
import { fbm2 } from '../core/rng.js';
import { leafInstances } from '../core/materials.js';
import * as TX from '../core/textures.js';

function trunkGeometry(rng, h, r0, taper, lean, seed) {
  const g = new THREE.CylinderGeometry(r0 * taper, r0, h, 18, 14, true);
  g.translate(0, h / 2, 0);
  const p = g.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const y = v.y / h;
    // root flare and gentle sinuous bends
    const flare = 1 + 0.55 * Math.exp(-v.y / 0.35);
    const bx = (fbm2(y * 2.2, seed, 3, 7) - 0.5) * 0.9 + lean[0] * y * y;
    const bz = (fbm2(y * 2.2, seed + 9, 3, 7) - 0.5) * 0.9 + lean[1] * y * y;
    const a = Math.atan2(v.z, v.x);
    const bump = 1 + 0.06 * (fbm2(a * 1.6, y * 9 + seed, 2, 3) - 0.5);
    v.x = v.x * flare * bump + bx;
    v.z = v.z * flare * bump + bz;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  // bark texel aspect: the texture is twice as tall as wide and wraps once around
  const uv = g.attributes.uv;
  const rep = h / (4 * Math.PI * r0);
  for (let i = 0; i < uv.count; i++) uv.setY(i, uv.getY(i) * rep);
  return g;
}

/**
 * Builds a stand of trees into `scene`.
 * opts: rng, clusters {colorTex, auxTex}, positions: [[x, z, kind?, h?]], cardSize,
 *       cardsScale (density), castShadow
 * returns { trunks: Group, crowns: InstancedMesh, crownInfo: [...] }
 */
export function buildTrees(scene, opts) {
  const { rng, clusters } = opts;
  const birch = TX.bark('birch', 31);
  const maple = TX.bark('maple', 32);
  const mats = {
    birch: [0.95, 0.85, 0.75].map((k) => new THREE.MeshStandardMaterial({ map: birch.map, normalMap: birch.normalMap, roughness: 0.8, color: new THREE.Color(k, k, k * 0.97) })),
    maple: [1.0, 0.8, 0.62].map((k) => new THREE.MeshStandardMaterial({ map: maple.map, normalMap: maple.normalMap, roughness: 0.9, color: new THREE.Color(k, k * 0.95, k * 0.9) })),
  };
  const trunks = new THREE.Group();
  const crowns = [];
  const branchGeo = new THREE.CylinderGeometry(0.3, 1, 1, 10, 1, true);
  branchGeo.translate(0, 0.5, 0);
  const up = new THREE.Vector3(0, 1, 0);
  let seed = 1;
  for (const [x, z, kindIn, hIn] of opts.positions) {
    const kind = kindIn || (rng.next() < 0.45 ? 'birch' : 'maple');
    const isB = kind === 'birch';
    const h = hIn || rng.range(13, 22);
    const r0 = isB ? rng.range(0.11, 0.2) : rng.range(0.17, 0.36);
    const lean = [rng.range(-0.6, 0.6), rng.range(-0.6, 0.6)];
    const geo = trunkGeometry(rng, h, r0, isB ? 0.35 : 0.45, lean, seed++ * 3.7);
    const mat = mats[kind][rng.int(3)];
    const t = new THREE.Mesh(geo, mat);
    t.position.set(x, -0.05, z);
    t.rotation.y = rng.range(0, 6.28);
    t.castShadow = true;
    t.receiveShadow = true;
    trunks.add(t);
    t.updateMatrixWorld(true);
    // crown centre follows the lean of the trunk top
    const top = new THREE.Vector3(lean[0], h, lean[1]).applyMatrix4(t.matrixWorld);
    const crown = {
      c: new THREE.Vector3(top.x, h * (isB ? 0.74 : 0.7), top.z),
      r: new THREE.Vector3(h * (isB ? 0.14 : 0.21), h * (isB ? 0.26 : 0.24), h * (isB ? 0.14 : 0.21)),
      kind,
    };
    crowns.push(crown);
    // main branches reaching into the crown
    const nb = isB ? rng.int(3) + 3 : rng.int(3) + 4;
    for (let b = 0; b < nb; b++) {
      const y0 = h * rng.range(isB ? 0.45 : 0.38, 0.75);
      const az = rng.range(0, 6.28), el = rng.range(isB ? 0.9 : 0.55, isB ? 1.25 : 1.05);
      const dir = new THREE.Vector3(Math.cos(az) * Math.cos(el), Math.sin(el), Math.sin(az) * Math.cos(el));
      const len = h * rng.range(0.16, 0.28);
      const br = r0 * rng.range(0.28, 0.42) * (1 - y0 / h * 0.5);
      const m = new THREE.Mesh(branchGeo, mat);
      const base = new THREE.Vector3(lean[0] * (y0 / h) ** 2, y0, lean[1] * (y0 / h) ** 2).applyMatrix4(t.matrixWorld);
      m.position.copy(base);
      m.quaternion.setFromUnitVectors(up, dir);
      m.scale.set(br, len, br);
      m.castShadow = true;
      trunks.add(m);
    }
  }
  scene.add(trunks);

  // crowns: cards on an ellipsoid shell, facing roughly outwards
  const scale = opts.cardsScale ?? 1;
  const counts = crowns.map((c) => Math.round(scale * 4 * Math.PI * ((c.r.x * c.r.y + c.r.y * c.r.z + c.r.x * c.r.z) / 3) / 1.1));
  const total = counts.reduce((a, b) => a + b, 0);
  const size = opts.cardSize ?? 1.5;
  const mesh = leafInstances(total, clusters.colorTex, clusters.auxTex, { size, seg: 1, cols: 2, rows: 2, castShadow: opts.castShadow ?? true, trans: 1.25 });
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), qq = new THREE.Quaternion(), sv = new THREE.Vector3(), pv = new THREE.Vector3();
  const col = new THREE.Color(), n = new THREE.Vector3();
  let k = 0;
  crowns.forEach((c, ci) => {
    const moods = c.kind === 'birch' ? [2, 2, 1] : rng.pick([[0, 0, 3], [1, 1, 3], [0, 3, 3], [1, 2, 0]]);
    for (let j = 0; j < counts[ci]; j++) {
      // random direction, radius biased to the shell
      n.set(rng.normal(), rng.normal() * 0.8, rng.normal()).normalize();
      const rr = 0.55 + 0.45 * Math.sqrt(rng.next());
      pv.set(c.c.x + n.x * c.r.x * rr, c.c.y + n.y * c.r.y * rr, c.c.z + n.z * c.r.z * rr);
      // card normal (local +Y) towards the outside, randomly tilted
      q.setFromUnitVectors(up, n.clone().add(new THREE.Vector3(rng.normal() * 0.6, rng.normal() * 0.6 + 0.3, rng.normal() * 0.6)).normalize());
      qq.setFromAxisAngle(up, rng.range(0, 6.28));
      q.multiply(qq);
      sv.setScalar(rng.range(0.75, 1.25));
      m4.compose(pv, q, sv);
      mesh.setMatrixAt(k, m4);
      col.setScalar(rng.range(0.72, 1.08));
      mesh.setColorAt(k, col);
      mesh.geometry.attributes.aVariant.array[k] = rng.pick(moods);
      k++;
    }
  });
  mesh.count = k;
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  mesh.geometry.attributes.aVariant.needsUpdate = true;
  mesh.name = 'crowns';
  scene.add(mesh);
  return { trunks, crowns: mesh, crownInfo: crowns };
}
