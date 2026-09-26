// 17-20 s memory: first-person, looking down at worn hiking boots walking through crisp leaves.
import * as THREE from 'three';
import { Rng, clamp, smooth, smoother, ramp, lerp } from '../core/rng.js';
import { leafInstances, sdfMaterial } from '../core/materials.js';
import { makeEnvironment } from '../core/env.js';
import { makeSky } from '../core/sky.js';
import { Wind, LeafFlock, Dust } from '../core/particles.js';
import * as TX from '../core/textures.js';
import { buildBoot } from './bootmodel.js';
import { buildTrees } from './trees.js';

const T0 = 16.8, T1 = 20.2;
const SPEED = 0.55;          // m/s, an unhurried stroll
const CYCLE = 1.15;          // s per full gait cycle
const STRIDE = SPEED * CYCLE;

function footState(t, k) {
  // phase 0 = heel strike; stance until 0.6, then swing
  const tt = t - T0 + 0.35;
  const ph = tt / CYCLE + k * 0.5;
  const n = Math.floor(ph);
  const f = ph - n;
  const x = k === 0 ? -0.105 : 0.105;
  // each foot lands ~0.32 m ahead of the body
  const plantZ = (m) => -(m - k * 0.5) * STRIDE - 0.0375;
  let z, y = 0, pitch = 0, pivot = 'ball';
  if (f < 0.6) {
    z = plantZ(n);
    if (f < 0.1) { pitch = lerp(-0.26, 0, smooth(f / 0.1)); pivot = 'heel'; }
    else if (f > 0.42) { pitch = 0.62 * smooth((f - 0.42) / 0.18); pivot = 'ball'; }
  } else {
    const s = (f - 0.6) / 0.4;
    const e = smoother(s);
    z = lerp(plantZ(n), plantZ(n + 1), e);
    y = 0.085 * Math.sin(Math.PI * s) ** 1.2;
    pitch = lerp(0.62, -0.26, smooth(s));
    pivot = s < 0.5 ? 'ball' : 'heel';
  }
  return { x, y, z, pitch, pivot, phase: f, cycle: n };
}

export class BootsScene {
  constructor(ctx) {
    this.ctx = ctx;
  }

  init() {
    const { renderer, atlas } = this.ctx;
    const scene = (this.scene = new THREE.Scene());
    const rng = new Rng(77);
    this.camera = new THREE.PerspectiveCamera(42, this.ctx.W / this.ctx.H, 0.02, 3000);
    this.sunDir = new THREE.Vector3(-0.62, 0.2, 0.76).normalize(); // low sun ahead of the camera, behind the walker

    scene.add(makeSky({ sunDir: this.sunDir, horizon: [0.95, 0.6, 0.36], zenith: [0.3, 0.42, 0.7], glow1: 0.6, glow2: 1.2, disk: 30 }));
    scene.fog = new THREE.FogExp2(new THREE.Color(0.5, 0.33, 0.2), 0.02);
    scene.environment = makeEnvironment(renderer, { sunDir: this.sunDir, canopy: 0.75, trunks: 0.9, glowGain: 1.0, glowPow: 8 });
    scene.environmentIntensity = 0.7;
    const sun = (this.sun = new THREE.DirectionalLight(new THREE.Color(1.0, 0.7, 0.42), 6.0));
    sun.castShadow = true;
    sun.shadow.mapSize.set(4096, 4096);
    const sc = sun.shadow.camera;
    sc.left = sc.bottom = -3.2;
    sc.right = sc.top = 3.2;
    sc.near = 0.5;
    sc.far = 30;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.01;
    sun.shadow.radius = 2.5;
    scene.add(sun, sun.target);
    // warm bounce from the sunlit leaves towards the camera side, and from below
    const fill = (this.fill = new THREE.DirectionalLight(new THREE.Color(1.0, 0.7, 0.45), 1.3));
    scene.add(fill, fill.target);
    scene.add(new THREE.HemisphereLight(new THREE.Color(0.55, 0.6, 0.75), new THREE.Color(0.75, 0.3, 0.08), 0.8));

    // ground: soil and a thick carpet of leaves along the path
    const soil = TX.soil(512, 41);
    soil.map.repeat.set(40, 40);
    soil.normalMap.repeat.set(40, 40);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(120, 120), new THREE.MeshStandardMaterial({ map: soil.map, normalMap: soil.normalMap, roughness: 0.95, color: new THREE.Color(0.9, 0.6, 0.45) }));
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);
    const N = 46000;
    const litter = leafInstances(N, atlas.colorTex, atlas.auxTex, { size: 0.13, seg: 2, castShadow: false });
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), col = new THREE.Color();
    const varW = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7];
    for (let i = 0; i < N; i++) {
      const far = i > 36000;
      const x = far ? rng.range(-7, 7) : rng.range(-1.6, 1.6);
      const z = far ? rng.range(-9, 1.5) : rng.range(-3.6, 1.6);
      e.set(rng.range(-0.3, 0.3), rng.range(0, 6.28), rng.range(-0.3, 0.3));
      q.setFromEuler(e);
      m4.compose(new THREE.Vector3(x, 0.004 + rng.next() * 0.045, z), q, new THREE.Vector3().setScalar(rng.range(0.75, 1.25) * (far ? 1.5 : 1)));
      litter.setMatrixAt(i, m4);
      const dry = rng.next() < 0.22;
      const b = dry ? rng.range(0.5, 0.7) : rng.range(0.75, 1.1);
      col.setRGB(b * (dry ? 0.95 : 1), b * (dry ? 0.75 : 1), b * (dry ? 0.6 : 1));
      litter.setColorAt(i, col);
      litter.geometry.attributes.aVariant.array[i] = varW[rng.int(varW.length)];
      litter.geometry.attributes.aCurl.array[i] = rng.range(-0.004, 0.026);
    }
    litter.instanceMatrix.needsUpdate = true;
    litter.instanceColor.needsUpdate = true;
    litter.geometry.attributes.aVariant.needsUpdate = true;
    litter.geometry.attributes.aCurl.needsUpdate = true;
    litter.receiveShadow = true;
    scene.add(litter);

    // boots
    const t0 = performance.now();
    const bootR = buildBoot({ cell: 0.0017, mirror: false });
    const bootL = buildBoot({ cell: 0.0017, mirror: true });
    console.log('boot mesh', bootR.geometry.attributes.position.count, 'verts', (performance.now() - t0).toFixed(0), 'ms');
    const bm = sdfMaterial({ grain: 110, bump: 0.7, scuffColor: [0.5, 0.4, 0.3] });
    this.boots = [new THREE.Mesh(bootL.geometry, bm), new THREE.Mesh(bootR.geometry, bm)];
    for (const b of this.boots) {
      b.castShadow = b.receiveShadow = true;
      b.matrixAutoUpdate = false;
      scene.add(b);
    }

    // jeans legs (rebuilt every frame from a simple two-bone leg)
    const dn = TX.denim(512, 1024, 71);
    this.denimMat = new THREE.MeshStandardMaterial({ map: dn.map, normalMap: dn.normalMap, roughness: 1.0, normalScale: new THREE.Vector2(0.4, 0.4) });
    dn.map.repeat.set(2, 3);
    dn.normalMap.repeat.set(2, 3);
    this.legs = [null, null];

    // the walker's body: invisible to the camera but it casts the long shadow ahead
    const ghost = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false });
    this.body = new THREE.Group();
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.17, 0.5, 8, 16), ghost);
    torso.position.y = 1.22;
    torso.scale.set(1, 1, 0.65);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.105, 24, 16), ghost);
    head.position.y = 1.64;
    const pack = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.45, 0.18), ghost);
    pack.position.set(0, 1.25, 0.2);
    this.arms = [0, 1].map(() => new THREE.Mesh(new THREE.CapsuleGeometry(0.045, 0.55, 6, 10), ghost));
    for (const m of [torso, head, pack, ...this.arms]) { m.castShadow = true; this.body.add(m); }
    scene.add(this.body);

    // leaves kicked up by each step
    this.wind = new Wind({ dir: new THREE.Vector3(0.3, 0, -1).normalize(), speed: () => 0.25, turb: () => 0.35, scale: 0.8, lift: () => 0 });
    const kickN = 220;
    this.kick = new LeafFlock(kickN, atlas, { size: 0.12, seed: 31, castShadow: true });
    const spawnStep = (k, cycle, when, strength) => {
      const st = footState(when, k);
      return { x: st.x, z: st.z, when, strength };
    };
    const events = [];
    for (let c = -2; c < 6; c++) for (const k of [0, 1]) {
      // heel strike: find the time this foot's phase wraps
      const tStrike = T0 - 0.35 + (c - k * 0.5) * CYCLE;
      if (tStrike < T0 - 0.3 || tStrike > T1) continue;
      events.push({ k, t: tStrike, toe: false });
      events.push({ k, t: tStrike + 0.6 * CYCLE, toe: true });
    }
    let li = 0;
    for (const ev of events) {
      if (li >= kickN) break;
      const st = footState(ev.t + 0.01, ev.k);
      const n = ev.toe ? 16 : 10;
      for (let j = 0; j < n && li < kickN; j++) {
        e.set(rng.range(-0.4, 0.4), rng.range(0, 6.28), rng.range(-0.4, 0.4));
        const fwd = ev.toe ? -1 : 0.4;
        this.kick.setLeaf(li++, {
          pos: new THREE.Vector3(st.x + rng.range(-0.08, 0.08), 0.03, st.z + (ev.toe ? rng.range(0.08, 0.16) : rng.range(-0.14, -0.04))),
          quat: new THREE.Quaternion().setFromEuler(e), birth: ev.t + rng.range(0, 0.06),
          vel: new THREE.Vector3(rng.normal() * 0.35, rng.range(0.5, 1.3) * (ev.toe ? 1 : 0.6), fwd * rng.range(0.3, 0.9)),
          scale: rng.range(0.75, 1.15), variant: varW[rng.int(varW.length)], curl: rng.range(0, 0.025), drag: rng.range(2.2, 3.2), fall: rng.range(0.8, 1.2),
        });
      }
    }
    this.kick.finalize();
    scene.add(this.kick.mesh);

    // the forest behind the walker, and understory colour for bokeh
    const treePos = [];
    for (let i = 0; i < 200 && treePos.length < 40; i++) {
      const x = rng.range(-28, -2.5), z = rng.range(-26, 22);
      if (treePos.some(([px, pz]) => Math.hypot(px - x, pz - z) < 2)) continue;
      // keep the sun visible between trunks
      const dx = x - 0.7, dz = z + 1.4;
      const along = dx * this.sunDir.x + dz * this.sunDir.z;
      if (along > 0 && Math.abs(dx * this.sunDir.z - dz * this.sunDir.x) < 1.2 + along * 0.05) continue;
      treePos.push([x, z]);
    }
    buildTrees(scene, { rng, clusters: this.ctx.clusters, positions: treePos, cardSize: 1.5, castShadow: true });
    this.dust = new Dust(500, new THREE.Box3(new THREE.Vector3(-1.5, 0.02, -4), new THREE.Vector3(1.5, 1.0, 1)), { seed: 13, gain: 1.0 });
    this.overlay = new THREE.Scene();
    this.overlay.add(this.dust.e.points);
  }

  bootMatrix(st, k) {
    const piv = st.pivot === 'heel' ? new THREE.Vector3(0, 0, -0.118) : new THREE.Vector3(0, 0, 0.128);
    const yaw = k === 0 ? 0.07 : -0.07;
    // soles rest on top of the springy leaf litter
    const m = new THREE.Matrix4().makeTranslation(st.x, st.y + 0.022, st.z);
    m.multiply(new THREE.Matrix4().makeRotationY(Math.PI + yaw)); // toes towards -Z (walking direction)
    m.multiply(new THREE.Matrix4().makeTranslation(piv.x, piv.y, piv.z));
    // local +X rotation raises the heel for positive pitch (toe-off)
    m.multiply(new THREE.Matrix4().makeRotationX(st.pitch));
    m.multiply(new THREE.Matrix4().makeTranslation(-piv.x, -piv.y, -piv.z));
    return m;
  }

  legGeometry(ankle, knee, hip) {
    const down = ankle.clone().sub(knee).normalize();
    const curve = new THREE.CatmullRomCurve3([ankle.clone().addScaledVector(down, 0.05), ankle, ankle.clone().lerp(knee, 0.5), knee,
      knee.clone().lerp(hip, 0.5), hip], false, 'centripetal');
    const segs = 40, rs = 22;
    const geo = new THREE.TubeGeometry(curve, segs, 1, rs, false);
    // radius profile: hem stacked over the boot, knee, thigh
    const p = geo.attributes.position, nrm = geo.attributes.normal;
    const pts = curve.getSpacedPoints(segs);
    for (let i = 0; i <= segs; i++) {
      const u = i / segs;
      let r = u < 0.08 ? lerp(0.07, 0.063, u / 0.08) : u < 0.5 ? lerp(0.063, 0.06, (u - 0.08) / 0.42) : lerp(0.06, 0.085, (u - 0.5) / 0.5);
      for (let j = 0; j <= rs; j++) {
        const idx = i * (rs + 1) + j;
        const wr = r * (1 + 0.04 * Math.sin(j * 0.9 + i * 0.7) * (u < 0.2 ? 1.5 : 0.6));
        p.setXYZ(idx, pts[i].x + nrm.getX(idx) * wr, pts[i].y + nrm.getY(idx) * wr, pts[i].z + nrm.getZ(idx) * wr);
      }
    }
    geo.computeVertexNormals();
    return geo;
  }

  render(t, post) {
    const zb = -(t - T0) * SPEED;
    const bob = 0.018 * Math.cos(((t - T0 + 0.35) / CYCLE) * Math.PI * 4);
    const sway = 0.012 * Math.sin(((t - T0 + 0.35) / CYCLE) * Math.PI * 2);
    // boots and legs
    for (let k = 0; k < 2; k++) {
      const st = footState(t, k);
      const m = this.bootMatrix(st, k);
      this.boots[k].matrix.copy(m);
      this.boots[k].matrixWorldNeedsUpdate = true;
      const ankle = new THREE.Vector3(0, 0.188, -0.05).applyMatrix4(m);
      const hip = new THREE.Vector3(k === 0 ? -0.1 : 0.1, 0.93 + bob * 0.5, zb + 0.02);
      const l1 = 0.45, l2 = 0.46;
      const dv = hip.clone().sub(ankle);
      const d = Math.min(dv.length(), l1 + l2 - 1e-3);
      dv.normalize();
      const a = (l2 * l2 - l1 * l1 + d * d) / (2 * d);
      const h = Math.sqrt(Math.max(l2 * l2 - a * a, 0));
      const fwd = new THREE.Vector3(0, 0, -1);
      const nrm = fwd.clone().addScaledVector(dv, -fwd.dot(dv)).normalize();
      const knee = ankle.clone().addScaledVector(dv, a).addScaledVector(nrm, h);
      if (this.legs[k]) { this.legs[k].geometry.dispose(); this.scene.remove(this.legs[k]); }
      const leg = new THREE.Mesh(this.legGeometry(ankle, knee, hip), this.denimMat);
      leg.castShadow = leg.receiveShadow = true;
      this.scene.add(leg);
      this.legs[k] = leg;
      this.arms[k].position.set(k === 0 ? -0.24 : 0.24, 1.15, zb + 0.18 * Math.sin(((t - T0 + 0.35) / CYCLE) * Math.PI * 2 + k * Math.PI));
    }
    this.body.position.set(sway, bob, zb);
    const cam = this.camera;
    // low tracking shot beside the path, moving with the walker
    const hb = 0.006 * Math.sin(t * 3.1) + 0.004 * Math.sin(t * 5.3);
    cam.position.set(0.92, 0.3 + hb, zb - 1.0 + 0.004 * Math.sin(t * 2.3));
    cam.lookAt(new THREE.Vector3(0.0, 0.15, zb - 0.16));
    this.fill.position.set(2.5, 1.2, zb - 1.8);
    this.fill.target.position.set(0, 0.1, zb - 0.1);
    this.fill.target.updateMatrixWorld();
    if (window.DBGCAM) {
      const [p, q] = window.DBGCAM;
      cam.position.set(p[0], p[1], p[2] + zb);
      cam.lookAt(q[0], q[1], q[2] + zb);
    }
    cam.updateMatrixWorld(true);
    // sun follows the walker so shadows stay in the map
    this.sun.position.copy(this.sunDir).multiplyScalar(12).add(new THREE.Vector3(0, 0, zb - 1.0));
    this.sun.target.position.set(0, 0, zb - 1.0);
    this.sun.target.updateMatrixWorld();
    this.kick.advanceTo(t, this.wind);
    this.dust.update(t, new THREE.Vector3(0.03, 0.01, -0.02), this.sunDir, cam.position);
    const P = {
      sunDir: this.sunDir,
      exposure: 0.85,
      rays: 0.16,
      rayBase: 0.05,
      raySunSize: 0.12,
      rayThresh: 3.0,
      bloom: 0.06,
      bloomKnee: 1.2,
      dof: { focus: 1.22, amount: 14, max: 36, nearMul: 1.0 },
      saturation: 1.14,
      contrast: 1.07,
      temperature: 0.1,
      vignette: 0.36,
      volume: { light: this.sun, gain: 1.0, density: 0.0028, maxDist: 25, g: 0.55, outside: 0.5, noiseAmt: 0.4, noiseScale: 0.3, height: 20 },
    };
    this.dust.e.bind(post, cam, P.dof);
    post.render(this.scene, cam, P, null, this.overlay);
  }
}
