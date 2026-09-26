// 0-14 s: heavy possessions on the forest floor; a gust turns them into light and leaves.
import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { Rng, clamp, smooth, smoother, ramp, lerp } from '../core/rng.js';
import { leafInstances, addDissolve, dissolveFieldJS } from '../core/materials.js';
import { Wind, Embers, LeafFlock, Dust } from '../core/particles.js';
import * as TX from '../core/textures.js';
import { makeEnvironment } from '../core/env.js';
import { buildObjects } from './objects.js';
import { buildTrees } from './trees.js';

export const SUN_DIR = new THREE.Vector3(0.37, 0.156, -0.915).normalize();
const WIND_DIR = new THREE.Vector3(0.52, 0.1, -0.85).normalize();

// Objects dissolve in this order (seconds).
const DISSOLVE = {
  padlock: [6.5, 8.3],
  briefcase: [7.0, 9.7],
  books: [7.5, 10.3],
  safe: [8.1, 11.6],
};
const EMBERS_PER = { padlock: 1600, briefcase: 4200, books: 4200, safe: 6500 };
const LEAVES_PER = { padlock: 70, briefcase: 240, books: 260, safe: 380 };

export function lakeMaterial(sunDir, { horizon = new THREE.Color(1.0, 0.72, 0.45), zenith = new THREE.Color(0.35, 0.5, 0.75) } = {}) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSun: { value: sunDir.clone() }, uHorizon: { value: horizon }, uZenith: { value: zenith },
      uSunColor: { value: new THREE.Color(1.0, 0.75, 0.45) }, uTime: { value: 0 }, uShore: { value: new THREE.Color(0.08, 0.06, 0.05) },
      uGain: { value: 1.0 },
    },
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uSun, uHorizon, uZenith, uSunColor, uShore; uniform float uTime, uGain;
      varying vec3 vWorld;
      void main(){
        vec3 V = normalize(vWorld - cameraPosition);
        // glass-like water: tiny ripples only
        vec2 q = vWorld.xz * 0.35;
        vec3 n = normalize(vec3(sin(q.x * 3.1 + uTime * 0.6) * 0.012 + sin(q.y * 5.3 - uTime * 0.4) * 0.008, 1.0,
                                cos(q.y * 2.7 + uTime * 0.5) * 0.012));
        vec3 R = reflect(V, n);
        float up = clamp(R.y, 0.0, 1.0);
        vec3 sky = mix(uHorizon, uZenith, pow(up, 0.55));
        // reflected far shore: dark band just above the waterline in reflection
        float shore = smoothstep(0.075, 0.02, up) * 0.85;
        sky = mix(sky, uShore, shore);
        float fres = 0.02 + 0.98 * pow(1.0 - clamp(-V.y, 0.0, 1.0), 5.0);
        vec3 col = mix(vec3(0.02, 0.035, 0.04), sky, fres);
        float glint = pow(max(dot(R, uSun), 0.0), 900.0) * 60.0 + pow(max(dot(R, uSun), 0.0), 60.0) * 1.5;
        col += uSunColor * glint;
        gl_FragColor = vec4(col * uGain, 1.0);
        #include <tonemapping_fragment>
      }`,
  });
}

export function mountainRidge(rng, { dist = 1600, width = 7000, height = 380, color = new THREE.Color(0.42, 0.42, 0.55), seg = 256, yBase = -2 } = {}) {
  const geo = new THREE.PlaneGeometry(width, 1, seg, 1);
  const pos = geo.attributes.position;
  const peaks = [];
  for (let i = 0; i < 9; i++) peaks.push([rng.range(-width / 2, width / 2), rng.range(0.4, 1.0), rng.range(250, 900)]);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const top = pos.getY(i) > 0;
    let h = 0.12;
    for (const [px, ph, pw] of peaks) h = Math.max(h, ph * Math.exp(-((x - px) * (x - px)) / (2 * pw * pw)));
    h += 0.06 * Math.sin(x * 0.011) + 0.04 * Math.sin(x * 0.031 + 1.3);
    pos.setY(i, top ? yBase + h * height : yBase - 5);
    pos.setZ(i, 0);
  }
  geo.computeVertexNormals();
  const mat = new THREE.MeshBasicMaterial({ color, fog: false });
  const m = new THREE.Mesh(geo, mat);
  m.position.z = -dist;
  return m;
}

export class ForestScene {
  constructor(ctx) {
    this.ctx = ctx;
  }

  init() {
    const { atlas } = this.ctx;
    const scene = (this.scene = new THREE.Scene());
    const rng = new Rng(2025);
    this.camera = new THREE.PerspectiveCamera(46, this.ctx.W / this.ctx.H, 0.03, 6000);

    // --- sky, sun, fog
    const sky = new Sky();
    sky.scale.setScalar(5000);
    const su = sky.material.uniforms;
    su.turbidity.value = 7.5;
    su.rayleigh.value = 1.6;
    su.mieCoefficient.value = 0.012;
    su.mieDirectionalG.value = 0.86;
    su.sunPosition.value.copy(SUN_DIR);
    scene.add(sky);
    scene.fog = new THREE.FogExp2(new THREE.Color(0.5, 0.33, 0.2), 0.012);

    scene.environment = makeEnvironment(this.ctx.renderer, { sunDir: SUN_DIR, canopy: 0.75, trunks: 0.9, glowGain: 1.2, glowPow: 8 });
    scene.environmentIntensity = 0.7;
    const sun = (this.sun = new THREE.DirectionalLight(new THREE.Color(1.0, 0.7, 0.42), 7.5));
    sun.position.copy(SUN_DIR).multiplyScalar(30);
    sun.target.position.set(0, 0, -4);
    sun.castShadow = true;
    sun.shadow.mapSize.set(4096, 4096);
    const sc = sun.shadow.camera;
    sc.left = -13;
    sc.right = 13;
    sc.top = 13;
    sc.bottom = -13;
    sc.near = 1;
    sc.far = 80;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    sun.shadow.radius = 2;
    scene.add(sun, sun.target);
    // warm bounce from the sunlit litter behind the camera
    const fill = (this.fill = new THREE.DirectionalLight(new THREE.Color(1.0, 0.72, 0.5), 0.9));
    fill.position.set(2.0, 0.8, 4);
    scene.add(fill);

    // --- ground
    const soil = TX.soil(512, 41);
    soil.map.repeat.set(60, 60);
    soil.normalMap.repeat.set(60, 60);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(240, 240), new THREE.MeshStandardMaterial({
      map: soil.map, normalMap: soil.normalMap, roughness: 0.95, color: new THREE.Color(0.9, 0.6, 0.45),
    }));
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    ground.name = 'ground';
    scene.add(ground);

    // --- objects (with the wind dissolve)
    this.objects = buildObjects();
    this.dissolveU = {};
    for (const [name, o] of Object.entries(this.objects)) {
      scene.add(o.group);
      o.group.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(o.group);
      const origin = box.getCenter(new THREE.Vector3()).addScaledVector(WIND_DIR, -o.size * 0.5);
      const P = { origin, windDir: WIND_DIR, span: o.size, freq: 7.5, edge: 0.02, edgeGain: 3.5, edgeColor: 0xff5a0c, char: 0.08 };
      o.dissolveP = P;
      const us = [];
      o.group.traverse((m) => {
        if (!m.isMesh) return;
        const mats = Array.isArray(m.material) ? m.material : [m.material];
        const cloned = mats.map((mm) => {
          const c = mm.clone();
          c.side = THREE.DoubleSide; // the inside of a dissolving shell stays solid
          us.push(addDissolve(c, P));
          return c;
        });
        m.material = Array.isArray(m.material) ? cloned : cloned[0];
        m.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
        us.push(addDissolve(m.customDepthMaterial, P));
      });
      this.dissolveU[name] = us;
    }

    // --- leaf litter (static), avoiding the objects' footprint
    const boxes = Object.values(this.objects).map((o) => new THREE.Box3().setFromObject(o.group).expandByScalar(0.02));
    const blocked = (x, z) => boxes.some((b) => x > b.min.x && x < b.max.x && z > b.min.z && z < b.max.z);
    const litterN = 62000;
    const litter = leafInstances(litterN, atlas.colorTex, atlas.auxTex, { size: 0.13, seg: 2, castShadow: false });
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), sv = new THREE.Vector3(), pv = new THREE.Vector3();
    const col = new THREE.Color();
    const varW = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7];
    let k = 0;
    const place = (x, z, yJ, sMin, sMax) => {
      const flat = blocked(x, z); // pressed flat under the heavy objects
      e.set(flat ? 0 : rng.range(-0.25, 0.25), rng.range(0, Math.PI * 2), flat ? 0 : rng.range(-0.25, 0.25));
      q.setFromEuler(e);
      const s = rng.range(sMin, sMax);
      pv.set(x, flat ? 0.003 + rng.next() * 0.004 : 0.004 + rng.next() * yJ, z);
      sv.setScalar(s);
      m4.compose(pv, q, sv);
      litter.setMatrixAt(k, m4);
      const dry = rng.next() < 0.18;
      const b = dry ? rng.range(0.45, 0.65) : rng.range(0.75, 1.1);
      col.setRGB(b * (dry ? 0.95 : 1), b * (dry ? 0.75 : 1), b * (dry ? 0.6 : 1));
      litter.setColorAt(k, col);
      litter.geometry.attributes.aVariant.array[k] = varW[rng.int(varW.length)];
      litter.geometry.attributes.aCurl.array[k] = flat ? 0 : rng.range(-0.004, 0.022);
      k++;
    };
    // dense near the camera and the objects
    for (let i = 0; i < 26000; i++) place(rng.range(-2.6, 2.8), rng.range(-2.5, 3.2), 0.05, 0.75, 1.2);
    // mid ground
    for (let i = 0; i < 22000; i++) place(rng.range(-9, 9), rng.range(-14, -2.5), 0.06, 0.9, 1.5);
    // far ground (larger, sparser; fog + DOF hide the rest)
    while (k < litterN) place(rng.range(-26, 26), rng.range(-48, -14), 0.08, 1.2, 2.2);
    litter.count = k;
    litter.instanceMatrix.needsUpdate = true;
    litter.instanceColor.needsUpdate = true;
    litter.geometry.attributes.aVariant.needsUpdate = true;
    litter.geometry.attributes.aCurl.needsUpdate = true;
    litter.receiveShadow = true;
    litter.name = 'litter';
    scene.add(litter);

    // --- trees.  A corridor towards the sun is kept clear so the light reaches the objects
    // (and the camera can later tilt up into that gap of light).
    const corridor = (x, z, w) => Math.abs(x - (SUN_DIR.x / SUN_DIR.z) * z) < w + 0.03 * Math.abs(z);
    const treePos = [[-1.25, -3.2, 'birch', 19]];
    for (let i = 0; i < 160 && treePos.length < 96; i++) {
      const z = rng.range(-72, -2.8), x = rng.range(-18, 18) * (1 + Math.max(-z - 40, 0) / 40);
      if (Math.abs(x) < 1.3 && z > -6) continue;
      if (corridor(x, z, 1.3)) continue;
      // keep the final line of sight from the lifted camera to the sun open too
      if (Math.abs(x - 0.32 - (SUN_DIR.x / SUN_DIR.z) * (z - 1.9)) < 1.4 + 0.02 * Math.abs(z)) continue;
      if (treePos.some(([px, pz]) => Math.hypot(px - x, pz - z) < 1.6)) continue;
      treePos.push([x, z]);
    }
    const trees = buildTrees(scene, { rng, clusters: this.ctx.clusters, positions: treePos, cardSize: 1.5, cardsScale: 1.0, castShadow: true });
    trees.crowns.name = 'canopy';

    // understory maples: glowing clusters that become bokeh behind the objects
    const shrubPos = [];
    for (let i = 0; i < 46; i++) {
      const x = rng.range(-11, 11), z = rng.range(-30, -4.5);
      if (Math.abs(x) < 0.8 && z > -7) continue;
      if (corridor(x, z, 1.1)) continue;
      shrubPos.push([x, rng.range(0.6, 3.6), z, rng.range(0.5, 1.2)]);
    }
    const shrubN = shrubPos.length * 320;
    const shrubs = leafInstances(shrubN, atlas.colorTex, atlas.auxTex, { size: 0.13, seg: 1, castShadow: false });
    let c3 = 0;
    for (const [cx, cy, cz, cr] of shrubPos) {
      for (let j = 0; j < 320; j++) {
        pv.set(cx + rng.normal() * 0.5 * cr, cy + rng.normal() * 0.35 * cr, cz + rng.normal() * 0.5 * cr);
        e.set(rng.range(-1.3, 1.3), rng.range(0, 6.28), rng.range(-1.3, 1.3));
        q.setFromEuler(e);
        sv.setScalar(rng.range(0.8, 1.25));
        m4.compose(pv, q, sv);
        shrubs.setMatrixAt(c3, m4);
        col.setScalar(rng.range(0.85, 1.1));
        shrubs.setColorAt(c3, col);
        shrubs.geometry.attributes.aVariant.array[c3] = varW[rng.int(varW.length)];
        shrubs.geometry.attributes.aCurl.array[c3] = rng.range(0.0, 0.03);
        c3++;
      }
    }
    shrubs.instanceMatrix.needsUpdate = true;
    shrubs.instanceColor.needsUpdate = true;
    shrubs.geometry.attributes.aVariant.needsUpdate = true;
    shrubs.geometry.attributes.aCurl.needsUpdate = true;
    shrubs.name = 'shrubs';
    scene.add(shrubs);

    // --- lake and mountains in the distance
    const lake = new THREE.Mesh(new THREE.PlaneGeometry(1400, 1400), lakeMaterial(SUN_DIR));
    lake.rotation.x = -Math.PI / 2;
    lake.position.set(0, -0.4, -760);
    this.lake = lake;
    scene.add(lake);
    const bank = new THREE.Mesh(new THREE.PlaneGeometry(400, 20), new THREE.MeshStandardMaterial({ color: 0x2a1a10, roughness: 1 }));
    bank.rotation.x = -Math.PI / 2 + 0.2;
    bank.position.set(0, -0.2, -58);
    scene.add(bank);
    scene.add(mountainRidge(new Rng(9), { dist: 1700, height: 420, color: new THREE.Color(0.5, 0.45, 0.55) }));
    scene.add(mountainRidge(new Rng(10), { dist: 2600, height: 620, color: new THREE.Color(0.62, 0.55, 0.62) }));

    // --- wind
    this.wind = new Wind({
      dir: WIND_DIR,
      scale: 0.55,
      speed: (t) => 0.22 + 3.6 * smooth(ramp(t, 5.9, 7.4)) + 0.9 * Math.exp(-((t - 9.2) ** 2) / 2.0) - 0.8 * smooth(ramp(t, 12.2, 13.6)),
      turb: (t) => 0.25 + 1.3 * smooth(ramp(t, 6.0, 7.6)),
      lift: (t, p) => (0.25 + 1.2 * smooth(ramp(t, 6.2, 7.8))) * clamp(1 - p.y / 3.0) * smooth(ramp(t, 5.9, 6.6)),
      vortex: (t) => {
        const s = smooth(ramp(t, 12.2, 13.5));
        if (s <= 0) return null;
        return { c: this._vortexC || new THREE.Vector3(0, 0.9, 0.9), axis: this._vortexAxis || new THREE.Vector3(0, 0, -1),
          strength: 7.5 * s, radius: 0.9, inward: 0.35, axial: -0.15 };
      },
    });

    // --- particles
    const totalEmbers = Object.values(EMBERS_PER).reduce((a, b) => a + b, 0);
    this.embers = new Embers(totalEmbers, { gain: 5.0 });
    const totalLeaves = Object.values(LEAVES_PER).reduce((a, b) => a + b, 0);
    this.flockDissolve = new LeafFlock(totalLeaves, atlas, { size: 0.085, seed: 21 });
    let ie = 0, il = 0;
    const tri = new THREE.Triangle(), pa = new THREE.Vector3(), pb = new THREE.Vector3(), pc = new THREE.Vector3();
    for (const [name, o] of Object.entries(this.objects)) {
      const [t0, t1] = DISSOLVE[name];
      // area-weighted triangle list over all meshes of the object
      const tris = [];
      let area = 0;
      o.group.updateMatrixWorld(true);
      o.group.traverse((m) => {
        if (!m.isMesh) return;
        const g = m.geometry.index ? m.geometry.toNonIndexed() : m.geometry;
        const pos = g.attributes.position;
        for (let i = 0; i < pos.count; i += 3) {
          pa.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
          pb.fromBufferAttribute(pos, i + 1).applyMatrix4(m.matrixWorld);
          pc.fromBufferAttribute(pos, i + 2).applyMatrix4(m.matrixWorld);
          tri.set(pa, pb, pc);
          const a = tri.getArea();
          if (a <= 0) continue;
          area += a;
          tris.push([pa.clone(), pb.clone(), pc.clone(), area]);
        }
      });
      const sample = () => {
        const r = rng.next() * area;
        let lo = 0, hi = tris.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (tris[mid][3] < r) lo = mid + 1;
          else hi = mid;
        }
        const [A, B, C] = tris[lo];
        let u = rng.next(), v = rng.next();
        if (u + v > 1) { u = 1 - u; v = 1 - v; }
        return A.clone().multiplyScalar(1 - u - v).addScaledVector(B, u).addScaledVector(C, v);
      };
      const birthOf = (p) => {
        const d = dissolveFieldJS(p, o.dissolveP);
        const x = clamp((d + 0.42) / 1.5);
        return t0 + (t1 - t0) * x;
      };
      o.birthOf = birthOf;
      for (let i = 0; i < EMBERS_PER[name]; i++) {
        const p = sample();
        const v = WIND_DIR.clone().multiplyScalar(rng.range(0.2, 0.8)).add(new THREE.Vector3(rng.normal() * 0.2, rng.range(0.1, 0.5), rng.normal() * 0.2));
        this.embers.setSpawn(ie++, p, birthOf(p) + rng.range(-0.02, 0.05), rng.range(1.4, 3.6), rng.range(0.004, 0.009), rng.range(0.25, 0.72), v);
      }
      for (let i = 0; i < LEAVES_PER[name]; i++) {
        const p = sample();
        e.set(rng.range(0, 6.28), rng.range(0, 6.28), rng.range(0, 6.28));
        this.flockDissolve.setLeaf(il++, {
          pos: p, quat: new THREE.Quaternion().setFromEuler(e), birth: birthOf(p) + rng.range(0, 0.1), grow: 1,
          scale: rng.range(0.7, 1.15), variant: rng.pick([0, 0, 1, 1, 5, 2, 6]), curl: rng.range(0.005, 0.03),
          vel: new THREE.Vector3(rng.normal() * 0.3, rng.range(0.2, 0.6), rng.normal() * 0.3), drag: rng.range(1.6, 2.6), fall: rng.range(0.5, 1.0),
        });
      }
    }
    this.flockDissolve.finalize();
    scene.add(this.flockDissolve.mesh);

    // litter near the objects that the gust lifts, plus leaves resting on the objects
    const liftN = 1700;
    this.flockLift = new LeafFlock(liftN, atlas, { size: 0.12, seed: 22, castShadow: false });
    for (let i = 0; i < liftN; i++) {
      let x = rng.range(-2.2, 2.0), z = rng.range(-1.6, 1.9);
      while (blocked(x, z)) { x = rng.range(-2.2, 2.0); z = rng.range(-1.6, 1.9); }
      e.set(rng.range(-0.2, 0.2), rng.range(0, 6.28), rng.range(-0.2, 0.2));
      const gustArrive = 6.0 + (x + 2.2) * 0.22 + rng.range(0, 1.8) + (z > 1.0 ? 0.6 : 0);
      this.flockLift.setLeaf(i, {
        pos: new THREE.Vector3(x, 0.03 + rng.next() * 0.02, z), quat: new THREE.Quaternion().setFromEuler(e), birth: gustArrive,
        scale: rng.range(0.75, 1.15), variant: varW[rng.int(varW.length)], curl: rng.range(0, 0.02), drag: rng.range(1.8, 2.8), fall: rng.range(0.6, 1.1),
      });
    }
    this.flockLift.finalize();
    scene.add(this.flockLift.mesh);

    // gently falling leaves through the light, and a flurry that swirls round the lens at the end
    const fallN = 170 + 420;
    this.flockFall = new LeafFlock(fallN, atlas, { size: 0.1, seed: 23 });
    for (let i = 0; i < fallN; i++) {
      const flurry = i >= 170;
      e.set(rng.range(0, 6.28), rng.range(0, 6.28), rng.range(0, 6.28));
      const birth = flurry ? rng.range(11.9, 13.2) : rng.range(-6, 12.5);
      const pos = flurry
        ? new THREE.Vector3(rng.range(-2.5, -0.6), rng.range(0.2, 2.6), rng.range(0.5, 3.2))
        : new THREE.Vector3(rng.range(-3.5, 2.5), rng.range(3.5, 7.5), rng.range(-6, 1.2));
      this.flockFall.setLeaf(i, {
        pos, quat: new THREE.Quaternion().setFromEuler(e), birth, scale: rng.range(0.8, 1.2), variant: varW[rng.int(varW.length)],
        curl: rng.range(0.005, 0.03), drag: rng.range(1.8, 2.6), fall: rng.range(0.7, 1.2),
        vel: flurry ? new THREE.Vector3(2.5, 0.5, -0.5) : new THREE.Vector3(0, -0.3, 0),
      });
    }
    this.flockFall.finalize();
    scene.add(this.flockFall.mesh);

    this.dust = new Dust(1100, new THREE.Box3(new THREE.Vector3(-3, 0.05, -6), new THREE.Vector3(3, 3.5, 2.6)), { seed: 5, gain: 1.1 });
    this.overlay = new THREE.Scene();
    this.overlay.add(this.embers.points, this.dust.e.points);
  }

  cameraAt(t) {
    // slow push in; then the camera follows the wind up towards the light
    const a = smoother(t / 7.0);
    const pos = new THREE.Vector3(lerp(0.3, 0.22, a), lerp(0.98, 0.86, a), lerp(2.95, 2.5, a));
    const tgt = new THREE.Vector3(lerp(0.0, -0.01, a), lerp(0.2, 0.2, a), -0.03);
    const b = smoother(ramp(t, 10.2, 14.2));
    pos.add(new THREE.Vector3(0.1 * b, 0.69 * b, -0.6 * b));
    tgt.add(new THREE.Vector3(1.33 * b, 1.97 * b, -0.83 * b));
    // breath of a hand-held camera, stronger in the gust
    const g = 0.004 + 0.01 * smooth(ramp(t, 6.2, 8.0));
    pos.x += g * Math.sin(t * 1.7) + g * 0.5 * Math.sin(t * 4.1);
    pos.y += g * 0.8 * Math.sin(t * 2.3 + 1.0);
    return { pos, tgt };
  }

  params(t) {
    const vib = smooth(ramp(t, 8.0, 13.0));
    const muted = 1 - vib;
    const focus = lerp(2.95, 2.5, smoother(t / 7)) + 1.2 * smoother(ramp(t, 10.5, 13.8));
    const tilt = smoother(ramp(t, 10.2, 14.2)); // looking up into the light: expose down
    return {
      sun: lerp(4.6, 7.2, vib),
      env: lerp(0.62, 1.0, vib),
      sunDir: SUN_DIR,
      exposure: (1.0 + 0.12 * vib) * lerp(1, 0.5, tilt),
      rays: (0.55 + 0.35 * vib) * lerp(1, 0.7, tilt),
      raySunSize: 0.4,
      rayThresh: 0.4,
      rayColor: 0xffc080,
      rayLength: 1.0,
      bloom: (0.07 + 0.05 * vib) * lerp(1, 0.7, tilt),
      bloomKnee: 1.1,
      dof: { focus, amount: 14 - 3 * smooth(ramp(t, 10.5, 13.5)), max: 34, nearMul: 1.0 },
      saturation: lerp(1.18, 0.76, muted),
      contrast: lerp(1.07, 1.03, muted) + 0.05 * tilt,
      temperature: lerp(0.12, -0.12, muted),
      tint: 0.0,
      desatShadows: 0.35 * muted,
      lift: [0.012 * muted, 0.014 * muted, 0.022 * muted],
      gamma: [1, 1, 1],
      gain: [1.0, 1.0 - 0.01 * vib, 1.0 - 0.03 * vib],
      vignette: 0.32,
      volume: { light: this.sun, gain: 1.0, density: (0.011 + 0.003 * vib) * lerp(1, 0.6, tilt), maxDist: 38, g: 0.5, outside: lerp(0.3, 0.12, tilt), noiseScale: 0.09, noiseAmt: 0.55,
        height: 12, scroll: new THREE.Vector3(-0.04 * t, 0.01 * t, 0.05 * t).addScaledVector(WIND_DIR, -0.12 * Math.max(t - 6, 0) ** 1.3) },
    };
  }

  render(t, post) {
    const { pos, tgt } = this.cameraAt(t);
    const cam = this.camera;
    cam.position.copy(pos);
    cam.lookAt(tgt);
    cam.updateMatrixWorld(true);
    const fwd = cam.getWorldDirection(new THREE.Vector3());
    this._vortexC = pos.clone().addScaledVector(fwd, 1.5);
    this._vortexAxis = fwd.clone().negate();
    // dissolve progress
    for (const [name, us] of Object.entries(this.dissolveU)) {
      const [t0, t1] = DISSOLVE[name];
      const p = t < t0 ? -1.0 : lerp(-0.42, 1.08, clamp((t - t0) / (t1 - t0)));
      for (const u of us) u.uProgress.value = p;
      this.objects[name].group.visible = t < t1 + 0.1;
    }
    this.flockDissolve.advanceTo(t, this.wind);
    this.flockLift.advanceTo(t, this.wind);
    this.flockFall.advanceTo(t, this.wind);
    this.embers.advanceTo(t, this.wind, { follow: 3.2, buoy: 0.45 });
    this.dust.update(t, new THREE.Vector3(0.05 + 0.9 * smooth(ramp(t, 6, 8)), 0.02, -0.03 - 0.7 * smooth(ramp(t, 6, 8))), SUN_DIR, pos);
    this.lake.material.uniforms.uTime.value = t;
    const P = this.params(t);
    this.sun.intensity = P.sun;
    this.scene.environmentIntensity = P.env;
    this.embers.bind(post, cam, P.dof);
    this.dust.e.bind(post, cam, P.dof);
    post.render(this.scene, cam, P, null, this.overlay);
  }
}
