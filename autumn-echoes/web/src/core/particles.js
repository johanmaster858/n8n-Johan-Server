// Wind-driven particles: glowing embers (points) and tumbling maple leaves (instances).
// Simulated on the CPU with a fixed time step from t = 0, so any frame is reproducible.
import * as THREE from 'three';
import { Rng, curl3, clamp, smooth } from './rng.js';
import { leafInstances } from './materials.js';

export const SIM_DT = 1 / 120;

/** Mean wind with gusts + divergence-free curl-noise turbulence + optional vortex. */
export class Wind {
  constructor(opts) {
    this.o = opts; // { dir: Vector3, speed(t), turb(t), scale, lift(t), vortex(t) -> {c, axis, strength, radius} }
    this._c = [0, 0, 0];
  }
  sample(p, t, out) {
    const o = this.o;
    const s = o.speed(t);
    const tu = o.turb(t);
    const sc = o.scale ?? 0.6;
    // turbulence is advected with the mean flow so gusts travel through the scene
    const adv = s * t * 0.8;
    curl3(p.x * sc - o.dir.x * adv * sc, p.y * sc + t * 0.15, p.z * sc - o.dir.z * adv * sc, this._c);
    out.set(o.dir.x * s + this._c[0] * tu, o.dir.y * s + this._c[1] * tu + (o.lift ? o.lift(t, p) : 0), o.dir.z * s + this._c[2] * tu);
    if (o.vortex) {
      const v = o.vortex(t);
      if (v && v.strength > 0) {
        const d = p.clone().sub(v.c);
        const ax = v.axis;
        const along = d.dot(ax);
        const radial = d.clone().sub(ax.clone().multiplyScalar(along));
        const r = radial.length() + 1e-4;
        const tang = ax.clone().cross(radial).multiplyScalar(1 / r);
        const k = v.strength * Math.exp(-r / v.radius) * (1 + r / v.radius);
        out.addScaledVector(tang, k);
        out.addScaledVector(radial, -k * (v.inward ?? 0.25) / r * Math.min(r, 1));
        out.addScaledVector(ax, k * (v.axial ?? 0.2));
      }
    }
    return out;
  }
}

// ------------------------------------------------------------------ embers --
const EMBER_VERT = /* glsl */ `
attribute float aSize; attribute float aHeat; attribute float aAlpha;
uniform float uScale, uFocus, uAmount, uMaxCoc;
varying float vHeat; varying float vAlpha; varying float vBlur; varying float vDepth;
void main(){
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float z = -mv.z;
  float size = aSize * uScale / max(z, 0.05);
  // bokeh: out-of-focus embers grow and dim, matching the post DOF
  float coc = abs(uAmount * (1.0 - uFocus / max(z, 0.01)));
  coc = min(coc, uMaxCoc);
  float s = max(size, 1.5) + coc;
  vBlur = clamp(coc / max(s, 1.0), 0.0, 1.0);
  gl_PointSize = s;
  vHeat = aHeat;
  vAlpha = aAlpha * (size * size + 2.0) / (s * s + 2.0) * 1.0;
  vDepth = z;
}`;

const EMBER_FRAG = /* glsl */ `
uniform sampler2D tDepth; uniform vec2 uRes; uniform float uNear, uFar, uGain;
varying float vHeat; varying float vAlpha; varying float vBlur; varying float vDepth;
float linDepth(float d) { float z = d * 2.0 - 1.0; return 2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear)); }
vec3 heat(float h){ // 0 = deep red, 1 = white-gold
  return mix(mix(vec3(1.0, 0.18, 0.03), vec3(1.0, 0.55, 0.12), smoothstep(0.0, 0.5, h)), vec3(1.0, 0.92, 0.7), smoothstep(0.55, 1.0, h));
}
void main(){
  float sceneZ = linDepth(texture2D(tDepth, gl_FragCoord.xy / uRes).x);
  if (vDepth > sceneZ + 0.02) discard;
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float r = length(q);
  if (r > 1.0) discard;
  float core = exp(-r * r * 6.0);
  float disc = smoothstep(1.0, 0.8, r) * (0.55 + 0.45 * smoothstep(0.4, 0.95, r));
  float a = mix(core, disc, vBlur);
  gl_FragColor = vec4(heat(vHeat) * a * vAlpha * uGain, 1.0);
}`;

export class Embers {
  constructor(count, { gain = 6.0 } = {}) {
    this.n = count;
    this.pos = new Float32Array(count * 3);
    this.vel = new Float32Array(count * 3);
    this.birth = new Float32Array(count).fill(1e9);
    this.life = new Float32Array(count).fill(2);
    this.size = new Float32Array(count);
    this.heat0 = new Float32Array(count);
    const g = new THREE.BufferGeometry();
    this.aPos = new THREE.BufferAttribute(new Float32Array(count * 3), 3);
    this.aSize = new THREE.BufferAttribute(new Float32Array(count), 1);
    this.aHeat = new THREE.BufferAttribute(new Float32Array(count), 1);
    this.aAlpha = new THREE.BufferAttribute(new Float32Array(count), 1);
    g.setAttribute('position', this.aPos);
    g.setAttribute('aSize', this.aSize);
    g.setAttribute('aHeat', this.aHeat);
    g.setAttribute('aAlpha', this.aAlpha);
    this.material = new THREE.ShaderMaterial({
      vertexShader: EMBER_VERT, fragmentShader: EMBER_FRAG,
      uniforms: {
        uScale: { value: 1000 }, uFocus: { value: 2 }, uAmount: { value: 0 }, uMaxCoc: { value: 40 },
        tDepth: { value: null }, uRes: { value: new THREE.Vector2(1080, 1920) }, uNear: { value: 0.1 }, uFar: { value: 1000 },
        uGain: { value: gain },
      },
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(g, this.material);
    this.points.frustumCulled = false;
    this.spawn0 = new Float32Array(count * 3);
    this.v0 = new Float32Array(count * 3);
  }

  setSpawn(i, p, birth, life, size, heat, v = null) {
    this.spawn0[i * 3] = p.x;
    this.spawn0[i * 3 + 1] = p.y;
    this.spawn0[i * 3 + 2] = p.z;
    if (v) {
      this.v0[i * 3] = v.x;
      this.v0[i * 3 + 1] = v.y;
      this.v0[i * 3 + 2] = v.z;
    }
    this.birth[i] = birth;
    this.life[i] = life;
    this.size[i] = size;
    this.heat0[i] = heat;
  }

  reset() {
    this.pos.set(this.spawn0);
    this.vel.set(this.v0);
    this.simT = 0;
  }

  step(wind, t, dt, { follow = 4.0, buoy = 0.35 } = {}) {
    const w = new THREE.Vector3(), p = new THREE.Vector3();
    for (let i = 0; i < this.n; i++) {
      if (t < this.birth[i] || t > this.birth[i] + this.life[i]) continue;
      const k = i * 3;
      p.set(this.pos[k], this.pos[k + 1], this.pos[k + 2]);
      wind.sample(p, t, w);
      this.vel[k] += (w.x - this.vel[k]) * follow * dt;
      this.vel[k + 1] += ((w.y - this.vel[k + 1]) * follow + buoy) * dt;
      this.vel[k + 2] += (w.z - this.vel[k + 2]) * follow * dt;
      this.pos[k] += this.vel[k] * dt;
      this.pos[k + 1] += this.vel[k + 1] * dt;
      this.pos[k + 2] += this.vel[k + 2] * dt;
    }
  }

  advanceTo(t, wind, opts) {
    if (this.simT === undefined || t < this.simT - 1e-6) this.reset();
    while (this.simT + SIM_DT <= t + 1e-9) {
      this.step(wind, this.simT, SIM_DT, opts);
      this.simT += SIM_DT;
    }
    this.upload(t);
  }

  upload(t) {
    const P = this.aPos.array, S = this.aSize.array, H = this.aHeat.array, A = this.aAlpha.array;
    for (let i = 0; i < this.n; i++) {
      const age = t - this.birth[i];
      const k = i * 3;
      P[k] = this.pos[k];
      P[k + 1] = this.pos[k + 1];
      P[k + 2] = this.pos[k + 2];
      if (age < 0 || age > this.life[i]) {
        A[i] = 0;
        continue;
      }
      const x = age / this.life[i];
      A[i] = smooth(age / 0.12) * Math.pow(1 - x, 1.4);
      H[i] = clamp(this.heat0[i] * (1 - 0.8 * x) + 0.15 * Math.sin(age * 17 + i));
      S[i] = this.size[i];
    }
    this.aPos.needsUpdate = this.aSize.needsUpdate = this.aHeat.needsUpdate = this.aAlpha.needsUpdate = true;
  }

  bind(post, camera, dof) {
    const u = this.material.uniforms;
    u.tDepth.value = post.rtScene.depthTexture;
    u.uRes.value.set(post.W, post.H);
    u.uNear.value = camera.near;
    u.uFar.value = camera.far;
    u.uScale.value = post.H / (2 * Math.tan((camera.fov * Math.PI) / 360));
    u.uFocus.value = dof ? dof.focus : 1;
    u.uAmount.value = dof ? dof.amount : 0;
    u.uMaxCoc.value = dof ? dof.max ?? 40 : 0;
  }
}

// ------------------------------------------------------------------ leaves --
/**
 * Tumbling leaves.  Each has a spawn pose and birth time; once born it is carried
 * by the wind with drag, falls slowly, and spins with a flutter.
 */
export class LeafFlock {
  constructor(count, atlas, opts = {}) {
    this.n = count;
    this.mesh = leafInstances(count, atlas.colorTex, atlas.auxTex, { size: opts.size ?? 0.1, castShadow: opts.castShadow ?? false, trans: 1.0 });
    this.pos = new Float32Array(count * 3);
    this.vel = new Float32Array(count * 3);
    this.q = new Float32Array(count * 4);
    this.w = new Float32Array(count * 3);
    this.spawnPos = new Float32Array(count * 3);
    this.spawnQ = new Float32Array(count * 4);
    this.spawnV = new Float32Array(count * 3);
    this.birth = new Float32Array(count).fill(1e9);
    this.scale = new Float32Array(count).fill(1);
    this.grow = new Float32Array(count); // 1 = grows from zero at birth (formed from dissolving matter)
    this.drag = new Float32Array(count).fill(2.2);
    this.fall = new Float32Array(count).fill(1.1);
    this.seed = new Float32Array(count);
    this.death = new Float32Array(count).fill(1e9);
    const rng = new Rng(opts.seed ?? 5);
    for (let i = 0; i < count; i++) this.seed[i] = rng.next() * 100;
    this.simT = undefined;
    this.m = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._p = new THREE.Vector3();
    this._s = new THREE.Vector3();
  }

  setLeaf(i, { pos, quat, vel = null, birth = 0, scale = 1, variant = 0, curl = 0.02, grow = 0, drag = 2.2, fall = 1.1, death = 1e9 }) {
    this.spawnPos.set([pos.x, pos.y, pos.z], i * 3);
    this.spawnQ.set([quat.x, quat.y, quat.z, quat.w], i * 4);
    if (vel) this.spawnV.set([vel.x, vel.y, vel.z], i * 3);
    this.birth[i] = birth;
    this.scale[i] = scale;
    this.grow[i] = grow;
    this.drag[i] = drag;
    this.fall[i] = fall;
    this.death[i] = death;
    this.mesh.geometry.attributes.aVariant.array[i] = variant;
    this.mesh.geometry.attributes.aCurl.array[i] = curl;
  }

  finalize() {
    this.mesh.geometry.attributes.aVariant.needsUpdate = true;
    this.mesh.geometry.attributes.aCurl.needsUpdate = true;
  }

  reset() {
    this.pos.set(this.spawnPos);
    this.vel.set(this.spawnV);
    this.q.set(this.spawnQ);
    this.w.fill(0);
    this.simT = 0;
  }

  step(wind, t, dt) {
    const wv = new THREE.Vector3(), p = new THREE.Vector3(), q = new THREE.Quaternion(), dq = new THREE.Quaternion();
    const up = new THREE.Vector3();
    for (let i = 0; i < this.n; i++) {
      if (t < this.birth[i] || t > this.death[i]) continue;
      const k = i * 3, k4 = i * 4;
      p.set(this.pos[k], this.pos[k + 1], this.pos[k + 2]);
      wind.sample(p, t, wv);
      const d = this.drag[i];
      // leaves resist moving along their normal much more than edge-on: flutter
      q.set(this.q[k4], this.q[k4 + 1], this.q[k4 + 2], this.q[k4 + 3]);
      up.set(0, 1, 0).applyQuaternion(q);
      const rx = wv.x - this.vel[k], ry = wv.y - this.vel[k + 1], rz = wv.z - this.vel[k + 2];
      const rn = rx * up.x + ry * up.y + rz * up.z;
      const kk = d * (0.55 + 0.45 * Math.abs(up.y));
      this.vel[k] += (rx * kk + rn * up.x * d * 0.8) * dt;
      this.vel[k + 1] += (ry * kk + rn * up.y * d * 0.8 - this.fall[i] * 1.6) * dt;
      this.vel[k + 2] += (rz * kk + rn * up.z * d * 0.8) * dt;
      // ground contact: leaves come to rest on the litter
      if (p.y < 0.01 && this.vel[k + 1] < 0) {
        this.vel[k + 1] *= -0.1;
        this.vel[k] *= 0.9;
        this.vel[k + 2] *= 0.9;
      }
      this.pos[k] += this.vel[k] * dt;
      this.pos[k + 1] = Math.max(0.005, this.pos[k + 1] + this.vel[k + 1] * dt);
      this.pos[k + 2] += this.vel[k + 2] * dt;
      // spin: torque from the relative wind speed, with a per-leaf flutter rhythm
      const rel = Math.min(Math.hypot(rx, ry, rz), 8);
      const s = this.seed[i];
      const tq = rel * 3.2;
      this.w[k] += (Math.sin(t * 3.1 + s) * tq - this.w[k] * 1.4) * dt;
      this.w[k + 1] += (Math.sin(t * 2.3 + s * 1.7) * tq * 0.7 - this.w[k + 1] * 1.4) * dt;
      this.w[k + 2] += (Math.cos(t * 2.7 + s * 0.6) * tq - this.w[k + 2] * 1.4) * dt;
      const wl = Math.hypot(this.w[k], this.w[k + 1], this.w[k + 2]);
      if (wl > 1e-6) {
        dq.setFromAxisAngle(new THREE.Vector3(this.w[k] / wl, this.w[k + 1] / wl, this.w[k + 2] / wl), wl * dt);
        q.premultiply(dq).normalize();
        this.q[k4] = q.x;
        this.q[k4 + 1] = q.y;
        this.q[k4 + 2] = q.z;
        this.q[k4 + 3] = q.w;
      }
    }
  }

  advanceTo(t, wind) {
    if (this.simT === undefined || t < this.simT - 1e-6) this.reset();
    while (this.simT + SIM_DT <= t + 1e-9) {
      this.step(wind, this.simT, SIM_DT);
      this.simT += SIM_DT;
    }
    this.upload(t);
  }

  upload(t) {
    for (let i = 0; i < this.n; i++) {
      const k = i * 3, k4 = i * 4;
      let s = this.scale[i];
      if (this.grow[i] > 0) s *= smooth((t - this.birth[i]) / 0.35);
      if (t > this.death[i]) s = 0;
      if (this.grow[i] > 0 && t < this.birth[i]) s = 0;
      this._p.set(this.pos[k], this.pos[k + 1], this.pos[k + 2]);
      this._q.set(this.q[k4], this.q[k4 + 1], this.q[k4 + 2], this.q[k4 + 3]);
      this._s.setScalar(Math.max(s, 1e-4));
      this.m.compose(this._p, this._q, this._s);
      this.mesh.setMatrixAt(i, this.m);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

// ------------------------------------------------------------------- dust --
export class Dust {
  constructor(count, box, { seed = 3, gain = 1.0 } = {}) {
    const rng = new Rng(seed);
    this.n = count;
    this.base = new Float32Array(count * 3);
    this.ph = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      this.base[i * 3] = rng.range(box.min.x, box.max.x);
      this.base[i * 3 + 1] = rng.range(box.min.y, box.max.y);
      this.base[i * 3 + 2] = rng.range(box.min.z, box.max.z);
      this.ph[i * 3] = rng.range(0, 6.28);
      this.ph[i * 3 + 1] = rng.range(0.3, 1.2);
      this.ph[i * 3 + 2] = rng.range(0.5, 1.6);
    }
    this.box = box;
    this.e = new Embers(count, { gain });
    for (let i = 0; i < count; i++) this.e.setSpawn(i, new THREE.Vector3(), -1, 1e9, rng.range(0.0025, 0.007), 0.95);
    this.e.material.uniforms.uGain.value = gain;
  }

  update(t, drift = new THREE.Vector3(0.05, 0.01, -0.03), sunDir = null, camPos = null) {
    const P = this.e.aPos.array, A = this.e.aAlpha.array, S = this.e.aSize.array, H = this.e.aHeat.array;
    const b = this.box;
    const sx = b.max.x - b.min.x, sy = b.max.y - b.min.y, sz = b.max.z - b.min.z;
    const v = new THREE.Vector3();
    for (let i = 0; i < this.n; i++) {
      const k = i * 3;
      const ph = this.ph[k];
      let x = this.base[k] + drift.x * t + 0.06 * Math.sin(t * 0.4 * this.ph[k + 1] + ph);
      let y = this.base[k + 1] + drift.y * t + 0.05 * Math.sin(t * 0.33 * this.ph[k + 2] + ph * 1.3);
      let z = this.base[k + 2] + drift.z * t + 0.06 * Math.cos(t * 0.37 + ph);
      x = b.min.x + ((((x - b.min.x) % sx) + sx) % sx);
      y = b.min.y + ((((y - b.min.y) % sy) + sy) % sy);
      z = b.min.z + ((((z - b.min.z) % sz) + sz) % sz);
      P[k] = x;
      P[k + 1] = y;
      P[k + 2] = z;
      // forward scattering: motes glitter when seen against the sun
      let phase = 1.0;
      if (sunDir && camPos) {
        v.set(x - camPos.x, y - camPos.y, z - camPos.z).normalize();
        const c = v.dot(sunDir);
        phase = 0.25 + 2.5 * Math.pow(Math.max(c, 0), 6);
      }
      const tw = 0.35 + 0.65 * Math.pow(0.5 + 0.5 * Math.sin(t * 2.0 * this.ph[k + 2] + ph * 5), 4);
      A[i] = phase * tw;
      S[i] = this.e.size[i];
      H[i] = 0.72;
    }
    this.e.aPos.needsUpdate = this.e.aAlpha.needsUpdate = this.e.aSize.needsUpdate = this.e.aHeat.needsUpdate = true;
  }
}
