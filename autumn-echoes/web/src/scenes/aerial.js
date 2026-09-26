// 22-36 s: the camera lifts above the treeline into rolling hills ablaze with autumn,
// split by a winding blue river, the sun low on the horizon.
import * as THREE from 'three';
import { Rng, fbm2, clamp, smooth, smoother, ramp, lerp } from '../core/rng.js';
import { leafInstances } from '../core/materials.js';
import { makeEnvironment } from '../core/env.js';
import { makeSky } from '../core/sky.js';
import { mountainRidge } from './forest.js';

const T0 = 21.9;
const CANOPY = 17; // canopy top above the ground (m)

// ------------------------------------------------------------ landscape --
export const riverX = (z) => 230 * Math.sin(z / 1150 + 0.9) + 120 * Math.sin(z / 480 + 2.1) + 45 * Math.sin(z / 210 + 0.4) - 160;
export const riverW = (z) => 34 + 10 * Math.sin(z / 870) + 6 * Math.sin(z / 300 + 1);
export const waterY = (z) => -40 + z * 0.0015;

function roundRidge(rng, { dist, width, height, color, seg = 512, yBase = 0 }) {
  const geo = new THREE.PlaneGeometry(width, 1, seg, 1);
  const pos = geo.attributes.position;
  const hills = [];
  for (let i = 0; i < 26; i++) hills.push([rng.range(-width / 2, width / 2), rng.range(0.35, 1.0), rng.range(2500, 7000)]);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    let h = 0.15;
    for (const [px, ph, pw] of hills) h = Math.max(h, ph * Math.exp(-((x - px) ** 2) / (2 * pw * pw)));
    h += 0.05 * Math.sin(x * 0.0011) + 0.03 * Math.sin(x * 0.0037 + 1.3);
    pos.setY(i, pos.getY(i) > 0 ? yBase + h * height : yBase - 50);
  }
  const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, fog: false }));
  m.position.z = -dist;
  return m;
}

function groundH(x, z) {
  const a = fbm2(x / 2700 + 3.1, z / 2700 + 7.7, 5, 11) - 0.5;
  const b = fbm2(x / 760 + 5.2, z / 760 + 3.3, 4, 12) - 0.5;
  return 330 * a + 70 * b + 10;
}

/** Height of the visible surface: the canopy top, or the water/banks along the river. */
export function surfaceH(x, z) {
  const d = Math.abs(x - riverX(z));
  const w = riverW(z);
  const wy = waterY(z);
  const g = groundH(x, z);
  // valley: the hills ease down towards the river
  const valley = smoothstep(w + 40, w + 900, d);
  const ground = lerp(wy + 6, Math.max(g, wy + 6), valley);
  // forest canopy stands on the ground, with a bank strip and the channel
  const forest = smoothstep(w + 4, w + 34, d);
  const bank = wy + 1.2 + 2.5 * smoothstep(w, w + 10, d);
  const top = lerp(bank, ground + CANOPY, forest);
  return d < w ? wy - 3 : top;
}
function smoothstep(a, b, x) {
  const t = clamp((x - a) / (b - a));
  return t * t * (3 - 2 * t);
}

const CANOPY_PARS = /* glsl */ `
varying vec3 vWp;
uniform vec3 uSunDir;
float h21(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
vec2 h22(vec2 p){ float a = h21(p); return vec2(a, h21(p + a + 17.1)); }
float vn2(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
float fbm2g(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vn2(p); p = p * 2.03 + 7.1; a *= 0.5; } return s; }
vec3 species(float u, float stand, float conifer, float cold){
  // warm palette ordered crimson -> scarlet -> orange -> gold -> yellow
  vec3 c0 = vec3(0.30, 0.022, 0.018), c1 = vec3(0.46, 0.06, 0.02), c2 = vec3(0.56, 0.16, 0.02), c3 = vec3(0.56, 0.31, 0.03), c4 = vec3(0.5, 0.4, 0.06);
  float k = clamp(u * 0.55 + stand * 1.45 - 0.5, 0.0, 0.999) * 4.0;
  vec3 c = k < 1.0 ? mix(c0, c1, k) : k < 2.0 ? mix(c1, c2, k - 1.0) : k < 3.0 ? mix(c2, c3, k - 2.0) : mix(c3, c4, k - 3.0);
  c = mix(c, vec3(0.13, 0.17, 0.04), cold);                 // a few still green
  return mix(c, vec3(0.018, 0.045, 0.024), conifer);          // spruce and pine
}
`;

function canopyMaterial(sunDir, clusterTex) {
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.92, metalness: 0 });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uSunDir = { value: sunDir };
    sh.uniforms.tCluster = { value: clusterTex };
    sh.uniforms.uRiver = { value: new THREE.Vector4() };
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWp;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWp = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + CANOPY_PARS + '\nuniform sampler2D tCluster; vec3 gCrownN; float gCrownK;')
      .replace('#include <color_fragment>', `#include <color_fragment>
        {
          vec2 wp = vWp.xz;
          float stand = fbm2g(wp / 520.0 + 11.0);
          float stand2 = fbm2g(wp / 260.0 + 3.0);
          float coniferP = 0.08 + 0.62 * smoothstep(0.5, 0.7, fbm2g(wp / 700.0 + 29.0)) + 0.2 * smoothstep(0.6, 0.8, stand2);
          float coldP = 0.1 * smoothstep(0.55, 0.7, fbm2g(wp / 330.0 + 51.0));
          // crowns on a jittered grid (about 8 m apart)
          vec2 p = wp / 8.5;
          vec2 ip = floor(p), fp = fract(p);
          float d1 = 9.0; vec2 cid = vec2(0.0), toC = vec2(0.0);
          for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = h22(ip + g);
            vec2 r = g + 0.15 + 0.7 * o - fp;
            float d = dot(r, r);
            if (d < d1) { d1 = d; cid = ip + g; toC = r; }
          }
          float h1 = h21(cid + 3.7), h2 = h21(cid + 9.1), h3 = h21(cid + 1.3);
          float con = step(h2, coniferP), cold = step(h3, coldP) * (1.0 - con);
          vec3 cc = species(h1, stand, con, cold) * (0.72 + 0.5 * h21(cid + 5.5));
          float R = mix(0.62, 0.48, con);
          float r = sqrt(d1) / R;
          // average colour of the stand for distant pixels
          vec3 avg = mix(species(0.5, stand, 0.0, 0.0), vec3(0.018, 0.045, 0.024), coniferP * 0.9) * 0.9;
          // clumpy canopy texture that survives at a distance
          float clump = fbm2g(wp / 45.0 + 2.0);
          avg *= 0.74 + 0.52 * clump;
          float lod = smoothstep(0.25, 0.9, length(fwidth(p)));
          float gap = smoothstep(0.85, 1.05, r);
          // leafy detail: each crown wears a sprig from the cluster atlas, rotated per tree
          float ang = h1 * 6.2831;
          mat2 rot = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
          vec2 luv = clamp(0.5 + rot * toC * 0.62, 0.0, 1.0);
          vec2 luv2 = clamp(0.5 + rot * toC.yx * 0.9 + 0.08, 0.0, 1.0);
          float tile = floor(h1 * 3.99);
          vec2 toff = vec2(mod(tile, 2.0), 1.0 - floor(tile / 2.0)) * 0.5;
          vec4 lf = texture2D(tCluster, toff + luv * 0.5);
          vec4 lf2 = texture2D(tCluster, toff + luv2 * 0.5);
          float leafA = max(lf.a, lf2.a * 0.9);
          vec3 leafC = mix(lf2.rgb, lf.rgb, lf.a);
          float lum = dot(leafC, vec3(0.3, 0.59, 0.11));
          vec3 detail = mix(cc * (0.55 + 1.1 * lum), leafC * dot(cc, vec3(1.3)) * 1.1, 0.35 * (1.0 - con));
          vec3 col = mix(cc * 0.22, detail, leafA * (1.0 - gap));
          col = mix(col, avg, lod);
          // dome normal of the crown (world space, xz tilt)
          float slope = r < 1.0 ? r / sqrt(max(1.0 - r * r, 0.04)) : 0.0;
          vec2 tilt = -normalize(toC + 1e-5) * slope * mix(0.35, 0.9, con) * (1.0 - gap);
          vec2 cg = vec2(fbm2g(wp / 45.0 + vec2(2.3, 2.0)) - clump, fbm2g(wp / 45.0 + vec2(2.0, 2.3)) - clump);
          gCrownN = vec3(tilt.x, 0.0, tilt.y) * (1.0 - lod) - vec3(cg.x, 0.0, cg.y) * 9.0 * lod;
          gCrownK = 1.0 - lod;
          diffuseColor.rgb = col;
        }`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        {
          vec3 nw = inverseTransformDirection(normal, viewMatrix);
          nw = normalize(nw + gCrownN * 0.9);
          normal = normalize((viewMatrix * vec4(nw, 0.0)).xyz);
        }`);
  };
  mat.customProgramCacheKey = () => 'canopy';
  return mat;
}

function waterMaterial(sky) {
  const su = sky.material.uniforms;
  return new THREE.ShaderMaterial({
    uniforms: {
      uSun: su.uSun, uZenith: su.uZenith, uHorizon: su.uHorizon, uGlowColor: su.uGlowColor, uGlow1: su.uGlow1, uGlow2: su.uGlow2,
      uSunColor: su.uSunColor, uGain: su.uGain, uTime: { value: 0 }, uDeep: { value: new THREE.Color(0.01, 0.035, 0.075) },
      uReflZenith: { value: new THREE.Color(0.1, 0.24, 0.62) },
    },
    vertexShader: /* glsl */ `varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uSun, uZenith, uHorizon, uGlowColor, uSunColor, uDeep, uReflZenith; uniform float uGlow1, uGlow2, uGain, uTime;
      varying vec3 vW;
      float h21(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
      float vn2(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
      vec3 skyCol(vec3 d){
        float h = max(d.y, 0.0);
        vec3 col = mix(uHorizon * vec3(0.55, 0.6, 0.85), uReflZenith, pow(h, 0.35));
        float c = max(dot(d, uSun), 0.0);
        col += uGlowColor * (uGlow1 * pow(c, 6.0) + uGlow2 * pow(c, 90.0));
        return col * uGain;
      }
      void main(){
        vec3 V = normalize(vW - cameraPosition);
        vec2 q = vW.xz * 0.08 + vec2(0.0, uTime * 0.6);
        vec2 g = vec2(vn2(q) - vn2(q + vec2(0.7, 0.0)), vn2(q) - vn2(q + vec2(0.0, 0.7)));
        g += 0.5 * vec2(vn2(q * 3.1) - vn2(q * 3.1 + vec2(0.5, 0.0)), vn2(q * 3.1) - vn2(q * 3.1 + vec2(0.0, 0.5)));
        vec3 n = normalize(vec3(g.x * 0.12, 1.0, g.y * 0.12));
        vec3 R = reflect(V, n);
        R.y = abs(R.y);
        float fres = 0.02 + 0.98 * pow(1.0 - clamp(-V.y, 0.0, 1.0), 5.0);
        vec3 col = mix(uDeep, skyCol(R), clamp(fres * 0.9 + 0.2, 0.0, 1.0));
        col += uSunColor * pow(max(dot(R, uSun), 0.0), 700.0) * 30.0;
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
}

export class AerialScene {
  constructor(ctx) {
    this.ctx = ctx;
  }

  init() {
    const { renderer, clusters } = this.ctx;
    const scene = (this.scene = new THREE.Scene());
    const rng = new Rng(9090);
    this.camera = new THREE.PerspectiveCamera(50, this.ctx.W / this.ctx.H, 1.0, 90000);
    this.sunDir = new THREE.Vector3(0.6, 0.075, -0.8).normalize();

    this.sky = makeSky({ sunDir: this.sunDir, horizon: [1.3, 0.6, 0.3], zenith: [0.14, 0.24, 0.58], glowColor: [2.0, 0.8, 0.28],
      glow1: 0.7, glow2: 2.6, disk: 60, sunSize: 0.012, horizonPow: 0.5, clouds: 1.0, cloudColor: [1.6, 0.55, 0.4], radius: 60000 });
    scene.add(this.sky);
    scene.environment = makeEnvironment(renderer, { sunDir: this.sunDir, horizon: [1.1, 0.6, 0.32], zenith: [0.22, 0.32, 0.62],
      ground: [0.18, 0.06, 0.02], glow: [2.0, 0.85, 0.3], canopy: 0.0, trunks: 0.0, band: 0.0, glowGain: 1.0, glowPow: 8 });
    scene.environmentIntensity = 0.8;
    const sun = (this.sun = new THREE.DirectionalLight(new THREE.Color(1.0, 0.66, 0.38), 5.5));
    sun.castShadow = true;
    sun.shadow.mapSize.set(4096, 4096);
    const sc = sun.shadow.camera;
    sc.left = sc.bottom = -2600;
    sc.right = sc.top = 2600;
    sc.near = 100;
    sc.far = 12000;
    sun.shadow.bias = -0.0025;
    sun.shadow.normalBias = 14;
    scene.add(sun, sun.target);

    // terrain: grid dense near the start, coarse far away
    const t0 = performance.now();
    const N = 560, S = 17000, Z0 = -700;
    const geo = new THREE.PlaneGeometry(2, 2, N, N);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position;
    const warp = (u) => Math.sign(u) * Math.pow(Math.abs(u), 1.9) * S;
    for (let i = 0; i < pos.count; i++) {
      const x = warp(pos.getX(i)), z = warp(pos.getZ(i)) + Z0;
      pos.setXYZ(i, x, surfaceH(x, z), z);
    }
    geo.computeVertexNormals();
    this.terrain = new THREE.Mesh(geo, canopyMaterial(this.sunDir, clusters.colorTex));
    this.terrain.receiveShadow = true;
    this.terrain.castShadow = true;
    scene.add(this.terrain);
    console.log('terrain', pos.count, 'verts', (performance.now() - t0).toFixed(0), 'ms');

    // river ribbon
    const zs = [];
    for (let z = 900; z > -22000; z -= z > -3000 ? 12 : 60) zs.push(z);
    const rv = new Float32Array(zs.length * 2 * 3);
    const ri = [];
    zs.forEach((z, k) => {
      const c = riverX(z), w = riverW(z) + 4, y = waterY(z);
      rv.set([c - w, y, z, c + w, y, z], k * 6);
      if (k > 0) ri.push(2 * k - 2, 2 * k - 1, 2 * k, 2 * k - 1, 2 * k + 1, 2 * k);
    });
    const rgeo = new THREE.BufferGeometry();
    rgeo.setAttribute('position', new THREE.BufferAttribute(rv, 3));
    rgeo.setIndex(ri);
    this.water = new THREE.Mesh(rgeo, waterMaterial(this.sky));
    this.water.frustumCulled = false;
    scene.add(this.water);

    // distant ranges
    scene.add(roundRidge(new Rng(31), { dist: 22000, height: 700, width: 110000, color: new THREE.Color(0.2, 0.15, 0.22), yBase: -150 }));
    scene.add(roundRidge(new Rng(37), { dist: 34000, height: 1200, width: 150000, color: new THREE.Color(0.3, 0.22, 0.32), yBase: -300 }));

    // 3D crowns around the start so the lift clears real treetops
    const trees = [];
    for (let i = 0; i < 9000 && trees.length < 2100; i++) {
      const x = rng.range(-230, 230), z = rng.range(-520, 110);
      if (Math.abs(x - riverX(z)) < riverW(z) + 30) continue;
      if (trees.some(([px, pz]) => Math.abs(px - x) < 5 && Math.abs(pz - z) < 5)) continue;
      trees.push([x, z]);
    }
    const perTree = 26;
    const crowns = leafInstances(trees.length * perTree, clusters.colorTex, clusters.auxTex, { size: 3.4, seg: 1, cols: 2, rows: 2, castShadow: true, trans: 1.2 });
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), col = new THREE.Color(), v = new THREE.Vector3();
    let k = 0;
    for (const [x, z] of trees) {
      const top = surfaceH(x, z);

      const R = rng.range(3.2, 4.8);
      const mood = rng.pick([[0, 0, 3], [1, 1, 3], [0, 1], [2, 1], [3, 0]]);
      for (let j = 0; j < perTree; j++) {
        v.set(rng.normal(), rng.normal() * 0.6 + 0.35, rng.normal()).normalize();
        const rr = 0.5 + 0.5 * Math.sqrt(rng.next());
        const p = new THREE.Vector3(x + v.x * R * rr, top - 0.6 + v.y * R * 0.75 * rr, z + v.z * R * rr);
        q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), v.clone().add(new THREE.Vector3(rng.normal() * 0.5, 0.6, rng.normal() * 0.5)).normalize());
        q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), rng.range(0, 6.28)));
        m4.compose(p, q, new THREE.Vector3().setScalar(rng.range(0.75, 1.2)));
        crowns.setMatrixAt(k, m4);
        col.setScalar(rng.range(0.6, 1.0));
        crowns.setColorAt(k, col);
        crowns.geometry.attributes.aVariant.array[k] = rng.pick(mood);
        k++;
      }
    }
    crowns.count = k;
    crowns.instanceMatrix.needsUpdate = true;
    crowns.instanceColor.needsUpdate = true;
    crowns.geometry.attributes.aVariant.needsUpdate = true;
    scene.add(crowns);
  }

  cameraAt(t) {
    const u = t - T0;
    const lift = smoother(ramp(t, 22.0, 26.2));
    const glide = smooth(ramp(t, 25.0, 36.5));
    const z = 60 - u * 26 - 6.5 * u * u;
    const x = lerp(-60, -250, smooth(ramp(t, 22, 36))) + 25 * Math.sin(u * 0.35);
    const base = surfaceH(x, z);
    const y = Math.max(base + 34 + 195 * lift + 190 * glide, 80 + 195 * lift + 190 * glide);
    const pitch = lerp(-0.72, -0.24, smoother(ramp(t, 22.0, 26.2))) + 0.05 * glide;
    const yaw = lerp(0.06, -0.1, smooth(ramp(t, 23, 36)));
    const pos = new THREE.Vector3(x, y, z);
    const dir = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
    return { pos, tgt: pos.clone().add(dir), roll: 0.03 * Math.sin(u * 0.5) * (1 - glide * 0.5) };
  }

  placeCamera(t) {
    const { pos, tgt, roll } = this.cameraAt(t);
    const cam = this.camera;
    cam.position.copy(pos);
    cam.lookAt(tgt);
    cam.rotateZ(roll);
    cam.updateMatrixWorld(true);
    return pos;
  }

  render(t, post) {
    const pos = this.placeCamera(t);
    const cam = this.camera;
    // shadow frustum follows the view
    const fwd = cam.getWorldDirection(new THREE.Vector3());
    const focus = pos.clone().addScaledVector(new THREE.Vector3(fwd.x, 0, fwd.z).normalize(), 2000);
    focus.y = 0;
    this.sun.target.position.copy(focus);
    this.sun.position.copy(focus).addScaledVector(this.sunDir, 6000);
    this.sun.target.updateMatrixWorld();
    this.water.material.uniforms.uTime.value = t;
    const hi = smooth(ramp(t, 22.5, 26));
    const P = {
      sunDir: this.sunDir,
      exposure: 0.92,
      rays: 0.25,
      rayBase: 0.0,
      raySunSize: 0.2,
      rayThresh: 3.0,
      bloom: 0.07,
      bloomKnee: 1.3,
      dof: { focus: lerp(55, 900, hi), amount: lerp(8, 0.0, hi), max: 24 },
      saturation: 1.22,
      contrast: 1.08,
      temperature: 0.1,
      vignette: 0.3,
      atmo: { density: 0.00008, height: 900, warm: 0xffa860, cool: 0x7f88c0, warmGain: 1.2, coolGain: 0.72, phase: 2.5, sky: 0.0 },
    };
    // the rapid lift gets a 180-degree shutter (three sub-frames) so the near canopy doesn't strobe
    let sub = null;
    // (sub-frame motion blur ghosts at this speed; the lift is paced to read cleanly instead)
    post.render(this.scene, cam, P, sub);
    this.placeCamera(t);
  }
}
