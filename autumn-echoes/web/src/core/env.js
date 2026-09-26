// Image-based lighting: a procedural surround (sky, sun glow, canopy, trunks, ground)
// rendered once into a PMREM so metals, leather and paper pick up the forest around them.
import * as THREE from 'three';

const ENV_FRAG = /* glsl */ `
uniform vec3 uSun, uHorizon, uZenith, uGround, uGlow, uFoliageLit, uFoliageDark, uTrunk, uHaze;
uniform float uCanopy, uTrunks, uGlowPow, uGlowGain, uGroundLit, uSeed, uBand;
varying vec3 vDir;
float h31(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float vn(vec3 x){ vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(h31(i), h31(i + vec3(1,0,0)), f.x), mix(h31(i + vec3(0,1,0)), h31(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(h31(i + vec3(0,0,1)), h31(i + vec3(1,0,1)), f.x), mix(h31(i + vec3(0,1,1)), h31(i + vec3(1,1,1)), f.x), f.y), f.z); }
float fbm(vec3 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 5; i++) { s += a * vn(p); p = p * 2.03 + 11.7; a *= 0.5; } return s; }
void main(){
  vec3 d = normalize(vDir);
  float h = d.y;
  float sunA = max(dot(d, uSun), 0.0);
  float sunAz = max(dot(normalize(d.xz + 1e-5), normalize(uSun.xz)), 0.0);
  vec3 sky = mix(uHorizon, uZenith, smoothstep(0.0, 0.65, h));
  sky += uGlow * (pow(sunA, uGlowPow) * uGlowGain + pow(sunA, 3.0) * 0.25);
  // canopy: patchy foliage over the upper hemisphere, glowing where the sun shines through
  float leaves = fbm(d * 7.0 + uSeed);
  float cover = uCanopy * smoothstep(0.05, 0.3, h);
  float mask = smoothstep(0.5 - 0.5 * cover, 0.62 - 0.45 * cover, leaves) * min(cover * 1.4, 1.0);
  vec3 fol = mix(uFoliageDark, uFoliageLit, pow(sunA, 2.0) * 0.85 + 0.15 * fbm(d * 23.0 + 3.0));
  vec3 col = mix(sky, fol, mask);
  // the forest itself around the horizon: dark depths, a golden haze towards the sun
  vec3 forest = mix(uFoliageDark * 0.8, uHaze, pow(sunAz, 5.0) * smoothstep(0.35, 0.0, abs(h - 0.05)));
  col = mix(col, forest, smoothstep(0.32, 0.04, h) * uBand);
  // trunks: dark vertical bands around the horizon
  float az = atan(d.z, d.x);
  float band = fbm(vec3(az * 9.0, 0.0, uSeed));
  float trunk = smoothstep(0.62, 0.66, band) * smoothstep(0.7, 0.05, h) * smoothstep(-0.15, 0.0, h) * uTrunks;
  col = mix(col, uTrunk * (0.6 + 0.8 * pow(sunA, 2.0)), trunk);
  // ground: leaf litter, brighter towards the sun (sunlit patches seen at grazing angles)
  float gmix = smoothstep(0.02, -0.04, h);
  vec3 gcol = uGround * (0.7 + 0.6 * fbm(d * 13.0)) * (1.0 + uGroundLit * pow(sunAz, 3.0));
  col = mix(col, gcol, gmix);
  gl_FragColor = vec4(col, 1.0);
}`;

/**
 * opts: sunDir, horizon, zenith, ground, glow, foliageLit, foliageDark, trunk (THREE.Color),
 *       canopy (0..1), trunks (0..1), glowPow, glowGain, groundLit, seed.
 */
export function makeEnvironment(renderer, opts = {}) {
  const c = (v, d) => (v instanceof THREE.Color ? v : new THREE.Color(...(v || d)));
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uSun: { value: (opts.sunDir || new THREE.Vector3(0, 0.2, -1)).clone().normalize() },
      uHorizon: { value: c(opts.horizon, [1.6, 0.95, 0.55]) },
      uZenith: { value: c(opts.zenith, [0.35, 0.5, 0.85]) },
      uGround: { value: c(opts.ground, [0.2, 0.08, 0.035]) },
      uGlow: { value: c(opts.glow, [3.0, 1.7, 0.8]) },
      uFoliageLit: { value: c(opts.foliageLit, [1.4, 0.6, 0.15]) },
      uFoliageDark: { value: c(opts.foliageDark, [0.12, 0.045, 0.02]) },
      uTrunk: { value: c(opts.trunk, [0.035, 0.028, 0.024]) },
      uHaze: { value: c(opts.haze, [1.1, 0.62, 0.3]) },
      uBand: { value: opts.band ?? 1.0 },
      uCanopy: { value: opts.canopy ?? 0.7 },
      uTrunks: { value: opts.trunks ?? 0.8 },
      uGlowPow: { value: opts.glowPow ?? 24 },
      uGlowGain: { value: opts.glowGain ?? 4 },
      uGroundLit: { value: opts.groundLit ?? 1.5 },
      uSeed: { value: opts.seed ?? 3.1 },
    },
    vertexShader: /* glsl */ `varying vec3 vDir; void main(){ vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: ENV_FRAG,
  });
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(100, 96, 48), mat));
  const pmrem = new THREE.PMREMGenerator(renderer);
  const rt = pmrem.fromScene(scene, 0, 0.1, 1000);
  pmrem.dispose();
  mat.dispose();
  return rt.texture;
}
