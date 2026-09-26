// 19.5-22.8 s memory: a crackling campfire at the edge of a glassy lake as the sun sets.
import * as THREE from 'three';
import { Reflector } from 'three/addons/objects/Reflector.js';
import { Rng, clamp, smooth, smoother, ramp, lerp, fbm3 } from '../core/rng.js';
import { leafInstances, noise3Texture, GLSL_NOISE } from '../core/materials.js';
import { makeEnvironment } from '../core/env.js';
import { makeSky } from '../core/sky.js';
import { Wind, Embers } from '../core/particles.js';
import * as TX from '../core/textures.js';
import { buildTrees } from './trees.js';
import { mountainRidge } from './forest.js';

const T0 = 19.3, T1 = 23.0;
const FIRE = new THREE.Vector3(0, 0, 0);

// ------------------------------------------------------------------ flame --
const FLAME_VERT = /* glsl */ `
varying vec3 vWorld;
void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const FLAME_FRAG = /* glsl */ `
uniform vec3 uBoxMin, uBoxMax, uBase; uniform float uTime, uGain, uNear, uFar, uHeight; uniform sampler2D tDepth; uniform vec2 uRes;
uniform vec3 uCamFwd;
varying vec3 vWorld;
${GLSL_NOISE}
float linDepth(float d) { float z = d * 2.0 - 1.0; return 2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear)); }
vec3 fireColor(float T){
  vec3 c = mix(vec3(0.6, 0.05, 0.005), vec3(1.0, 0.32, 0.04), smoothstep(0.05, 0.4, T));
  c = mix(c, vec3(1.0, 0.62, 0.18), smoothstep(0.35, 0.7, T));
  return mix(c, vec3(1.0, 0.9, 0.65), smoothstep(0.7, 1.0, T));
}
void main(){
  vec3 ro = cameraPosition, rd = normalize(vWorld - cameraPosition);
  vec3 t0 = (uBoxMin - ro) / rd, t1 = (uBoxMax - ro) / rd;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  float tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0)), tf = min(min(tmax.x, tmax.y), tmax.z);
  float sceneT = linDepth(texture2D(tDepth, gl_FragCoord.xy / uRes).x) / max(dot(rd, uCamFwd), 1e-3);
  tf = min(tf, sceneT);
  if (tf <= tn) discard;
  const int N = 56;
  float st = (tf - tn) / float(N);
  float jit = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  vec3 col = vec3(0.0); float trans = 1.0;
  for (int i = 0; i < N; i++) {
    vec3 p = ro + rd * (tn + (float(i) + jit) * st) - uBase;
    float h = p.y / uHeight;
    if (h < 0.0 || h > 1.0) continue;
    // turbulence carried upwards
    vec3 q = p * vec3(2.6, 1.3, 2.6) - vec3(0.0, uTime * 1.9, 0.0);
    vec2 warp = vec2(n3(q * 0.35 + 0.1), n3(q * 0.35 + 3.7)) - 0.5;
    vec2 xz = p.xz + warp * (0.05 + 0.34 * h);
    // three licking tongues that wander and flicker at different heights
    vec2 c1 = vec2(0.03 * sin(uTime * 2.1), 0.03 * cos(uTime * 1.7));
    vec2 c2 = vec2(-0.07, 0.035) + 0.03 * vec2(sin(uTime * 3.1 + 1.0), cos(uTime * 2.3));
    vec2 c3 = vec2(0.05, -0.065) + 0.03 * vec2(sin(uTime * 2.7 + 2.0), cos(uTime * 3.3 + 1.0));
    float r = min(length(xz - c1), min(length(xz - c2) * (1.2 + 0.25 * sin(uTime * 4.1)), length(xz - c3) * (1.3 + 0.25 * sin(uTime * 3.7 + 1.0))));
    float radius = 0.2 * pow(1.0 - h, 0.75) + 0.015;
    float body = smoothstep(radius, radius * 0.2, r);
    float n = fbmN3(p * vec3(3.2, 1.5, 3.2) * 0.6 - vec3(0.0, uTime * 1.35, 0.0));
    float d = body * smoothstep(0.26 + 0.46 * h, 0.72, n + 0.28 * (1.0 - h));
    if (d <= 0.001) continue;
    float T = clamp(d * (1.25 - h * 0.95), 0.0, 1.0);
    vec3 e = fireColor(T) * pow(T, 1.3) * uGain;
    col += e * trans * st * 12.0;
    trans *= exp(-d * st * 6.0);
    if (trans < 0.02) break;
  }
  gl_FragColor = vec4(col, 1.0);
}`;

function treeline(rng, { dist, width, color, scale = 1 }) {
  const trees = [];
  for (let x = -width / 2; x < width / 2; x += rng.range(3, 7) * scale) {
    const spruce = rng.next() < 0.4;
    trees.push([x, spruce, (spruce ? rng.range(16, 27) : rng.range(12, 19)) * scale, (spruce ? rng.range(2.2, 3.4) : rng.range(4, 7)) * scale]);
  }
  const seg = Math.round(width / 1.2);
  const geo = new THREE.PlaneGeometry(width, 1, seg, 1);
  const pos = geo.attributes.position;
  let j = 0;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    if (pos.getY(i) < 0) { pos.setY(i, -2); continue; }
    while (j > 0 && trees[j][0] > x) j--;
    while (j < trees.length - 1 && trees[j + 1][0] < x) j++;
    let h = 6 * scale;
    for (let k = Math.max(0, j - 4); k < Math.min(trees.length, j + 5); k++) {
      const [tx, sp, th, tw] = trees[k];
      const d = Math.abs(x - tx);
      h = Math.max(h, sp ? th * Math.max(0, 1 - d / tw) : th * Math.sqrt(Math.max(0, 1 - (d / tw) ** 2)));
    }
    pos.setY(i, h - 1.5);
  }
  const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color }));
  m.position.z = -dist;
  return m;
}

function rockGeometry(rng, r, flat = 0.6) {
  const g = new THREE.SphereGeometry(1, 40, 28);
  const p = g.attributes.position;
  const v = new THREE.Vector3();
  const s = rng.range(0, 100);
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const n = fbm3(v.x * 1.6 + s, v.y * 1.6, v.z * 1.6, 4);
    const k = 0.78 + 0.45 * n;
    v.multiplyScalar(k);
    v.y *= flat;
    if (v.y < -0.25 * flat) v.y = -0.25 * flat + (v.y + 0.25 * flat) * 0.2;
    p.setXYZ(i, v.x * r, v.y * r, v.z * r);
  }
  g.computeVertexNormals();
  return g;
}

export class CampfireScene {
  constructor(ctx) {
    this.ctx = ctx;
  }

  init() {
    const { renderer, atlas, clusters } = this.ctx;
    const scene = (this.scene = new THREE.Scene());
    const rng = new Rng(4242);
    this.camera = new THREE.PerspectiveCamera(42, this.ctx.W / this.ctx.H, 0.03, 6000);
    this.sunDir = new THREE.Vector3(0.1, 0.105, -1).normalize(); // sun just above the far hills

    scene.add(makeSky({ sunDir: this.sunDir, horizon: [1.5, 0.48, 0.16], zenith: [0.13, 0.12, 0.3], ground: [0.05, 0.04, 0.05],
      glowColor: [2.0, 0.7, 0.2], glow1: 0.45, glow2: 2.2, disk: 40, sunSize: 0.016, horizonPow: 0.4, clouds: 0.9, cloudColor: [1.2, 0.4, 0.25], gain: 0.8 }));
    scene.fog = new THREE.FogExp2(new THREE.Color(0.32, 0.2, 0.18), 0.0022);
    scene.environment = makeEnvironment(renderer, { sunDir: this.sunDir, horizon: [0.9, 0.42, 0.25], zenith: [0.12, 0.14, 0.3],
      ground: [0.06, 0.035, 0.02], glow: [1.8, 0.7, 0.25], foliageLit: [0.5, 0.2, 0.06], foliageDark: [0.03, 0.015, 0.01],
      haze: [0.8, 0.35, 0.15], canopy: 0.25, trunks: 0.5, glowGain: 1.0, glowPow: 10 });
    scene.environmentIntensity = 0.35;
    const sun = (this.sun = new THREE.DirectionalLight(new THREE.Color(1.0, 0.5, 0.22), 2.2));
    sun.position.copy(this.sunDir).multiplyScalar(20);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = sc.bottom = -6;
    sc.right = sc.top = 6;
    sc.near = 1;
    sc.far = 60;
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    scene.add(sun, sun.target);

    // fire light (flickers)
    this.fireLight = new THREE.PointLight(new THREE.Color(1.0, 0.48, 0.16), 1.6, 0, 2);
    this.fireLight.position.set(0, 0.32, 0);
    this.fireLight.castShadow = true;
    this.fireLight.shadow.mapSize.set(512, 512);
    this.fireLight.shadow.bias = -0.002;
    this.fireLight.shadow.camera.near = 0.05;
    this.fireLight.shadow.camera.far = 12;
    scene.add(this.fireLight);

    // ground: pebbly shore with scattered leaves; lake beyond z < -2.2
    const soil = TX.soil(512, 44);
    soil.map.repeat.set(30, 30);
    soil.normalMap.repeat.set(30, 30);
    const groundGeo = new THREE.PlaneGeometry(60, 34, 150, 110);
    groundGeo.rotateX(-Math.PI / 2);
    const gp = groundGeo.attributes.position;
    for (let i = 0; i < gp.count; i++) {
      const x = gp.getX(i), wz = gp.getZ(i) + 10;
      // the beach slopes gently down into the water at z ~ -2.4
      const shore = -2.4 + 0.5 * Math.sin(x * 0.35) + 0.3 * Math.sin(x * 1.1 + 1);
      const y = wz < shore ? (wz - shore) * 0.18 : 0.012 * Math.sin(x * 3.1) * Math.sin(wz * 2.7);
      gp.setY(i, y);
    }
    groundGeo.computeVertexNormals();
    const ground = new THREE.Mesh(groundGeo, new THREE.MeshStandardMaterial({ map: soil.map, normalMap: soil.normalMap, color: new THREE.Color(0.95, 0.8, 0.7), roughness: 0.95 }));
    ground.position.z = 10;
    ground.receiveShadow = true;
    scene.add(ground);
    // pebbles along the waterline and leaves on the shore
    const pebG = new THREE.IcosahedronGeometry(1, 2);
    const pebM = new THREE.MeshStandardMaterial({ color: 0x8a8076, roughness: 0.7 });
    const peb = new THREE.InstancedMesh(pebG, pebM, 900);
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), col = new THREE.Color();
    for (let i = 0; i < 900; i++) {
      const x = rng.range(-6, 6), z = rng.range(-2.9, 1.5);
      const s = rng.range(0.012, 0.045);
      e.set(rng.range(0, 6), rng.range(0, 6), rng.range(0, 6));
      m4.compose(new THREE.Vector3(x, s * 0.3, z), q.setFromEuler(e), new THREE.Vector3(s, s * 0.55, s * rng.range(0.8, 1.3)));
      peb.setMatrixAt(i, m4);
      col.setScalar(rng.range(0.45, 1.1));
      peb.setColorAt(i, col);
    }
    peb.receiveShadow = peb.castShadow = true;
    scene.add(peb);
    const leafN = 9000;
    const leaves = leafInstances(leafN, atlas.colorTex, atlas.auxTex, { size: 0.12, seg: 2 });
    let k = 0;
    for (let i = 0; i < leafN; i++) {
      const x = rng.range(-6, 6), z = rng.range(-1.9, 4.5);
      if (Math.hypot(x, z) < 0.62) continue;
      e.set(rng.range(-0.3, 0.3), rng.range(0, 6.28), rng.range(-0.3, 0.3));
      m4.compose(new THREE.Vector3(x, 0.006 + rng.next() * 0.02, z), q.setFromEuler(e), new THREE.Vector3().setScalar(rng.range(0.7, 1.2)));
      leaves.setMatrixAt(k, m4);
      col.setScalar(rng.range(0.5, 0.95));
      leaves.setColorAt(k, col);
      leaves.geometry.attributes.aVariant.array[k] = rng.int(8);
      leaves.geometry.attributes.aCurl.array[k] = rng.range(0, 0.025);
      k++;
    }
    leaves.count = k;
    leaves.instanceMatrix.needsUpdate = true;
    leaves.instanceColor.needsUpdate = true;
    leaves.geometry.attributes.aVariant.needsUpdate = true;
    leaves.geometry.attributes.aCurl.needsUpdate = true;
    leaves.receiveShadow = true;
    scene.add(leaves);

    // fire ring stones
    const rockMat = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.24, 0.22, 0.21), roughness: 0.85 });
    rockMat.onBeforeCompile = (sh) => {
      sh.uniforms.tNoise3 = { value: noise3Texture() };
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vW;')
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vW;\n' + GLSL_NOISE)
        .replace('#include <color_fragment>', `#include <color_fragment>
          float rn = fbmN3(vW * 3.0);
          diffuseColor.rgb *= 0.65 + 0.7 * rn;
          // soot on the inner faces, lichen speckles outside
          float soot = smoothstep(0.55, 0.25, length(vW.xz));
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.05, 0.045, 0.04), soot * 0.8 * smoothstep(0.3, 0.7, n3(vW * 9.0)));
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.55, 0.58, 0.42), smoothstep(0.72, 0.8, n3(vW * 14.0)) * 0.5);`);
    };
    for (let i = 0; i < 13; i++) {
      const a = (i / 13) * Math.PI * 2 + rng.range(-0.1, 0.1);
      const r = rng.range(0.085, 0.12);
      const g = rockGeometry(rng, r, rng.range(0.55, 0.8));
      const m = new THREE.Mesh(g, rockMat);
      m.position.set(Math.cos(a) * 0.46, r * 0.25, Math.sin(a) * 0.46);
      m.rotation.y = rng.range(0, 6.28);
      m.castShadow = m.receiveShadow = true;
      scene.add(m);
    }
    // logs: a small teepee over a bed of embers
    const maple = TX.bark('maple', 36, 256, 512);
    const logMat = new THREE.MeshStandardMaterial({ map: maple.map, normalMap: maple.normalMap, roughness: 0.9 });
    logMat.onBeforeCompile = (sh) => {
      sh.uniforms.tNoise3 = { value: noise3Texture() };
      sh.uniforms.uTime = { value: 0 };
      logMat.userData.shader = sh;
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vW;')
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vW; uniform float uTime;\n' + GLSL_NOISE)
        .replace('#include <color_fragment>', `#include <color_fragment>
          // charring towards the heart of the fire
          float heat = smoothstep(0.24, 0.02, length(vW - vec3(0.0, 0.1, 0.0)));
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.02, 0.015, 0.012), smoothstep(0.1, 0.5, heat));`)
        .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          float cracks = smoothstep(0.62, 0.76, fbmN3(vW * vec3(11.0, 5.0, 11.0) + vec3(0.0, uTime * 0.05, 0.0)));
          float flick = 0.7 + 0.3 * n3(vec3(vW.xz * 3.0, uTime * 0.7));
          totalEmissiveRadiance += vec3(1.0, 0.28, 0.04) * 9.0 * cracks * smoothstep(0.25, 0.75, heat) * flick;`);
    };
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2 + 0.3;
      const len = rng.range(0.42, 0.5), r = rng.range(0.028, 0.04);
      const g = new THREE.CylinderGeometry(r * 0.9, r, len, 14, 6);
      const m = new THREE.Mesh(g, logMat);
      const foot = new THREE.Vector3(Math.cos(a) * 0.24, 0.025, Math.sin(a) * 0.24);
      const top = new THREE.Vector3(Math.cos(a) * 0.035, 0.3, Math.sin(a) * 0.035);
      m.position.copy(foot).lerp(top, 0.5);
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), top.clone().sub(foot).normalize());
      m.scale.y = foot.distanceTo(top) / len;
      m.castShadow = m.receiveShadow = true;
      scene.add(m);
    }
    this.logMat = logMat;
    // glowing ember bed
    const bedMat = new THREE.MeshStandardMaterial({ color: 0x100806, roughness: 1, emissive: 0xffffff, emissiveIntensity: 1 });
    bedMat.onBeforeCompile = (sh) => {
      sh.uniforms.tNoise3 = { value: noise3Texture() };
      sh.uniforms.uTime = { value: 0 };
      bedMat.userData.shader = sh;
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vW;')
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vW; uniform float uTime;\n' + GLSL_NOISE)
        .replace('#include <emissivemap_fragment>', `
          float g = smoothstep(0.5, 0.75, fbmN3(vec3(vW.xz * 7.0, uTime * 0.12)));
          float r0 = length(vW.xz);
          totalEmissiveRadiance = vec3(1.0, 0.3, 0.05) * 6.0 * g * smoothstep(0.36, 0.05, r0) * (0.7 + 0.3 * n3(vec3(vW.xz * 4.0, uTime)));`);
    };
    const bedGeo = new THREE.CircleGeometry(0.36, 48);
    bedGeo.rotateX(-Math.PI / 2);
    const bp = bedGeo.attributes.position;
    for (let i = 0; i < bp.count; i++) bp.setY(i, 0.012 + 0.03 * Math.max(0, 1 - Math.hypot(bp.getX(i), bp.getZ(i)) / 0.3) * (0.5 + 0.5 * Math.sin(i * 12.9)));
    bedGeo.computeVertexNormals();
    this.bedMat = bedMat;
    const bed = new THREE.Mesh(bedGeo, bedMat);
    bed.receiveShadow = true;
    scene.add(bed);

    // lake: planar reflection with gentle ripples
    const lakeShader = {
      name: 'Lake',
      uniforms: { color: { value: null }, tDiffuse: { value: null }, textureMatrix: { value: null }, uTime: { value: 0 },
        uDeep: { value: new THREE.Color(0.02, 0.025, 0.04) }, tNoise3: { value: noise3Texture() } },
      vertexShader: /* glsl */ `
        uniform mat4 textureMatrix; varying vec4 vUv; varying vec3 vW;
        void main(){ vUv = textureMatrix * vec4(position, 1.0); vW = (modelMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: /* glsl */ `
        uniform vec3 color, uDeep; uniform sampler2D tDiffuse; uniform float uTime; varying vec4 vUv; varying vec3 vW;
        ${GLSL_NOISE}
        void main(){
          vec3 V = normalize(cameraPosition - vW);
          vec2 w = vec2(n3(vec3(vW.xz * 0.9, uTime * 0.25)) - 0.5, n3(vec3(vW.xz * 0.9 + 7.0, uTime * 0.25)) - 0.5);
          w += 0.5 * vec2(n3(vec3(vW.xz * 3.1, uTime * 0.5)) - 0.5, n3(vec3(vW.xz * 3.1 + 3.0, uTime * 0.5)) - 0.5);
          float dist = length(vW - cameraPosition);
          vec4 uv = vUv; uv.xy += w * 0.018 * uv.w / (1.0 + dist * 0.05);
          vec3 refl = texture2DProj(tDiffuse, uv).rgb;
          float fres = 0.02 + 0.98 * pow(1.0 - clamp(V.y, 0.0, 1.0), 5.0);
          gl_FragColor = vec4(mix(uDeep, refl * 0.92, clamp(fres + 0.25, 0.0, 1.0)), 1.0);
        }`,
    };
    const lake = new Reflector(new THREE.PlaneGeometry(3000, 3000), { shader: lakeShader, textureWidth: this.ctx.W >> 1, textureHeight: this.ctx.H >> 1, multisample: 0, clipBias: 0.002 });
    lake.rotation.x = -Math.PI / 2;
    lake.position.y = -0.12;
    this.lake = lake;
    scene.add(lake);

    // far shore: a treeline silhouette (spruce spires among rounded maples) and hills in the dusk haze
    scene.add(treeline(new Rng(77), { dist: 420, width: 2600, color: new THREE.Color(0.06, 0.035, 0.03) }));
    scene.add(treeline(new Rng(78), { dist: 700, width: 3600, color: new THREE.Color(0.1, 0.055, 0.05), scale: 1.3 }));
    scene.add(mountainRidge(new Rng(19), { dist: 1500, height: 120, width: 7000, color: new THREE.Color(0.16, 0.09, 0.12) }));
    scene.add(mountainRidge(new Rng(23), { dist: 2600, height: 240, width: 9000, color: new THREE.Color(0.3, 0.15, 0.18) }));
    // trees framing the campsite
    const treePos = [[-2.6, 0.2, 'birch', 16], [2.9, -0.4, 'maple', 18], [-3.8, 2.6], [3.6, 2.9], [-5.2, -0.8], [5.5, 0.6]];
    buildTrees(scene, { rng, clusters, positions: treePos, cardSize: 1.4, castShadow: true });

    // sparks
    this.wind = new Wind({ dir: new THREE.Vector3(0.15, 1, -0.1).normalize(), speed: () => 0.9, turb: () => 0.8, scale: 2.2,
      lift: () => 0.4 });
    const nSparks = 1400;
    this.sparks = new Embers(nSparks, { gain: 9 });
    for (let i = 0; i < nSparks; i++) {
      const a = rng.range(0, 6.28), r = Math.sqrt(rng.next()) * 0.16;
      this.sparks.setSpawn(i, new THREE.Vector3(Math.cos(a) * r, rng.range(0.1, 0.35), Math.sin(a) * r), rng.range(T0 - 3, T1),
        rng.range(0.6, 2.0), rng.range(0.002, 0.0048), rng.range(0.55, 1.0), new THREE.Vector3(rng.normal() * 0.3, rng.range(0.6, 1.6), rng.normal() * 0.3));
    }
    // flame volume (overlay pass, depth-tested against the scene)
    const box = new THREE.Box3(new THREE.Vector3(-0.42, 0.0, -0.42), new THREE.Vector3(0.42, 1.05, 0.42));
    this.flameMat = new THREE.ShaderMaterial({
      vertexShader: FLAME_VERT, fragmentShader: FLAME_FRAG, side: THREE.BackSide, transparent: true, depthTest: false, depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: { uBoxMin: { value: box.min }, uBoxMax: { value: box.max }, uBase: { value: new THREE.Vector3(0, 0.02, 0) }, uTime: { value: 0 },
        uGain: { value: 3.2 }, uNear: { value: 0.03 }, uFar: { value: 6000 }, tDepth: { value: null }, uRes: { value: new THREE.Vector2() },
        uCamFwd: { value: new THREE.Vector3() }, uHeight: { value: 0.95 }, tNoise3: { value: noise3Texture() } },
    });
    const flame = new THREE.Mesh(new THREE.BoxGeometry(0.84, 1.05, 0.84), this.flameMat);
    flame.position.set(0, 0.525, 0);
    this.overlay = new THREE.Scene();
    this.overlay.add(flame, this.sparks.points);
  }

  cameraAt(t) {
    const u = clamp((t - T0) / (T1 - T0));
    const a = smoother(clamp(u / 0.7));
    const pos = new THREE.Vector3(lerp(0.55, 0.32, a), lerp(0.62, 0.52, a), lerp(2.45, 1.75, a));
    const tgt = new THREE.Vector3(lerp(0.05, 0.0, a), 0.3, 0.0);
    // at the end the camera lifts off over the lake, towards the aerial
    const b = smoother(ramp(t, 21.7, 23.0));
    pos.add(new THREE.Vector3(0, 2.6 * b * b, -1.4 * b));
    tgt.add(new THREE.Vector3(0, 1.2 * b, -6 * b));
    pos.x += 0.004 * Math.sin(t * 1.3);
    pos.y += 0.003 * Math.sin(t * 1.9 + 1);
    return { pos, tgt };
  }

  render(t, post) {
    const { pos, tgt } = this.cameraAt(t);
    const cam = this.camera;
    cam.position.copy(pos);
    cam.lookAt(tgt);
    cam.updateMatrixWorld(true);
    // firelight flicker: layered noise
    const fl = 0.75 + 0.18 * Math.sin(t * 13.1) * Math.sin(t * 7.3 + 1) + 0.12 * Math.sin(t * 23.7 + 2) + 0.08 * Math.sin(t * 41.3);
    this.fireLight.intensity = 2.6 * fl;
    this.fireLight.position.set(0.02 * Math.sin(t * 5.1), 0.3 + 0.03 * Math.sin(t * 7.7), 0.02 * Math.cos(t * 4.3));
    if (this.logMat.userData.shader) this.logMat.userData.shader.uniforms.uTime.value = t;
    if (this.bedMat.userData.shader) this.bedMat.userData.shader.uniforms.uTime.value = t;
    this.lake.material.uniforms.uTime.value = t;
    this.sparks.advanceTo(t, this.wind, { follow: 2.2, buoy: 0.9 });
    const u = this.flameMat.uniforms;
    u.uTime.value = t;
    u.tDepth.value = post.rtScene.depthTexture;
    u.uRes.value.set(post.W, post.H);
    u.uNear.value = cam.near;
    u.uFar.value = cam.far;
    cam.getWorldDirection(u.uCamFwd.value);
    const lift = smoother(ramp(t, 21.7, 23.0));
    const P = {
      sunDir: this.sunDir,
      exposure: 0.72,
      rays: 0.3,
      rayBase: 0.1,
      raySunSize: 0.15,
      rayThresh: 2.0,
      rayColor: 0xff9050,
      skyDist: 1200,
      bloom: 0.055,
      bloomKnee: 1.5,
      dof: { focus: pos.distanceTo(new THREE.Vector3(0, 0.3, 0)) * (1 + 6 * lift), amount: 9 * (1 - lift), max: 30, nearMul: 1.0 },
      saturation: 1.12,
      contrast: 1.06,
      temperature: 0.08,
      vignette: 0.38,
    };
    this.sparks.bind(post, cam, P.dof);
    post.render(this.scene, cam, P, null, this.overlay);
  }
}
