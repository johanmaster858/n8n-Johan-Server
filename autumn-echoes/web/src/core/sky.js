// Art-directable golden-hour sky: horizon/zenith gradient, sun disc, two glow lobes,
// optional soft cloud streaks.  All values are linear HDR so exposure stays predictable.
import * as THREE from 'three';

export function makeSky(opts = {}) {
  const c = (v, d) => (v instanceof THREE.Color ? v : new THREE.Color(...(v || d)));
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    uniforms: {
      uSun: { value: (opts.sunDir || new THREE.Vector3(0, 0.1, -1)).clone().normalize() },
      uZenith: { value: c(opts.zenith, [0.28, 0.42, 0.75]) },
      uHorizon: { value: c(opts.horizon, [1.5, 0.95, 0.6]) },
      uGround: { value: c(opts.ground, [0.25, 0.16, 0.1]) },
      uGlowColor: { value: c(opts.glowColor, [1.6, 0.85, 0.4]) },
      uSunColor: { value: c(opts.sunColor, [1.0, 0.85, 0.6]) },
      uGlow1: { value: opts.glow1 ?? 1.2 },
      uGlow2: { value: opts.glow2 ?? 3.0 },
      uDisk: { value: opts.disk ?? 60 },
      uSunSize: { value: opts.sunSize ?? 0.012 },
      uHorizonPow: { value: opts.horizonPow ?? 0.5 },
      uClouds: { value: opts.clouds ?? 0.0 },
      uCloudColor: { value: c(opts.cloudColor, [1.3, 0.7, 0.45]) },
      uGain: { value: opts.gain ?? 1.0 },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vDir = w.xyz - cameraPosition;
        gl_Position = projectionMatrix * viewMatrix * w; gl_Position.z = gl_Position.w; }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uSun, uZenith, uHorizon, uGround, uGlowColor, uSunColor, uCloudColor;
      uniform float uGlow1, uGlow2, uDisk, uSunSize, uHorizonPow, uClouds, uGain;
      varying vec3 vDir;
      float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y); }
      float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * vn(p); p *= 2.07; a *= 0.5; } return s; }
      void main(){
        vec3 d = normalize(vDir);
        float h = d.y;
        vec3 col = mix(uHorizon, uZenith, pow(clamp(h, 0.0, 1.0), uHorizonPow));
        float c = max(dot(d, uSun), 0.0);
        col += uGlowColor * (uGlow1 * pow(c, 6.0) + uGlow2 * pow(c, 90.0));
        // wispy clouds lit from behind near the sun
        if (uClouds > 0.0 && h > 0.0) {
          vec2 p = d.xz / (h + 0.12) * 1.6;
          float cl = smoothstep(0.52, 0.8, fbm(p * vec2(1.0, 3.2) + 3.0)) * smoothstep(0.0, 0.08, h) * uClouds;
          vec3 cc = uCloudColor * (0.4 + 2.2 * pow(c, 8.0));
          col = mix(col, cc, cl);
        }
        col += uSunColor * uDisk * smoothstep(cos(uSunSize), cos(uSunSize * 0.7), dot(d, uSun));
        col = mix(col, uGround, smoothstep(0.0, -0.08, h));
        gl_FragColor = vec4(col * uGain, 1.0);
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 32), mat);
  mesh.scale.setScalar(opts.radius ?? 3000);
  mesh.frustumCulled = false;
  mesh.renderOrder = -1;
  return mesh;
}
