// Shared materials: backlit maple leaves, a 3D noise texture shared by GLSL and JS,
// and the dissolve effect that turns the heavy objects into light.
import * as THREE from 'three';
import { Rng, clamp } from './rng.js';
import { LEAF_COLS, LEAF_ROWS } from './leafatlas.js';

// ---------------------------------------------------------------- noise 3D --
export const N3 = 32;
let noiseData = null;
let noiseTex = null;

export function noise3Texture() {
  if (noiseTex) return noiseTex;
  const rng = new Rng(77);
  const n = N3;
  let a = new Float32Array(n * n * n);
  for (let i = 0; i < a.length; i++) a[i] = rng.next();
  // periodic separable blur -> smooth tileable noise
  const blur = (src, axis) => {
    const out = new Float32Array(src.length);
    const k = [0.06, 0.24, 0.4, 0.24, 0.06];
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      let s = 0;
      for (let j = -2; j <= 2; j++) {
        const xx = axis === 0 ? (x + j + n) % n : x, yy = axis === 1 ? (y + j + n) % n : y, zz = axis === 2 ? (z + j + n) % n : z;
        s += src[(zz * n + yy) * n + xx] * k[j + 2];
      }
      out[(z * n + y) * n + x] = s;
    }
    return out;
  };
  for (let pass = 0; pass < 2; pass++) a = blur(blur(blur(a, 0), 1), 2);
  let mn = 1e9, mx = -1e9;
  for (const v of a) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  noiseData = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) noiseData[i] = Math.round(((a[i] - mn) / (mx - mn)) * 255);
  noiseTex = new THREE.Data3DTexture(noiseData, n, n, n);
  noiseTex.format = THREE.RedFormat;
  noiseTex.type = THREE.UnsignedByteType;
  noiseTex.minFilter = noiseTex.magFilter = THREE.LinearFilter;
  noiseTex.wrapS = noiseTex.wrapT = noiseTex.wrapR = THREE.RepeatWrapping;
  noiseTex.unpackAlignment = 1;
  noiseTex.needsUpdate = true;
  return noiseTex;
}

/** Same trilinear lookup as the GPU (texture coordinates, repeat wrap). */
export function sampleNoise3(x, y, z) {
  const n = N3;
  const fx = x * n - 0.5, fy = y * n - 0.5, fz = z * n - 0.5;
  const ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz);
  const tx = fx - ix, ty = fy - iy, tz = fz - iz;
  const g = (i, j, k) => noiseData[((((iz + k) % n) + n) % n * n + ((((iy + j) % n) + n) % n)) * n + ((((ix + i) % n) + n) % n)] / 255;
  const l = (a, b, t) => a + (b - a) * t;
  return l(l(l(g(0, 0, 0), g(1, 0, 0), tx), l(g(0, 1, 0), g(1, 1, 0), tx), ty),
    l(l(g(0, 0, 1), g(1, 0, 1), tx), l(g(0, 1, 1), g(1, 1, 1), tx), ty), tz);
}

export function fbmN3(x, y, z) {
  return 0.55 * sampleNoise3(x, y, z) + 0.3 * sampleNoise3(x * 2.03 + 0.31, y * 2.03 + 0.17, z * 2.03 + 0.73) +
    0.15 * sampleNoise3(x * 4.07 + 0.67, y * 4.07 + 0.29, z * 4.07 + 0.11);
}

export const GLSL_NOISE = /* glsl */ `
uniform highp sampler3D tNoise3;
float n3(vec3 p) { return texture(tNoise3, p).r; }
float fbmN3(vec3 p) { return 0.55 * n3(p) + 0.3 * n3(p * 2.03 + vec3(0.31, 0.17, 0.73)) + 0.15 * n3(p * 4.07 + vec3(0.67, 0.29, 0.11)); }
`;

// ------------------------------------------------------------------ leaves --
export function leafGeometry(size = 0.12, seg = 4) {
  const g = new THREE.PlaneGeometry(size, size, seg, seg);
  g.rotateX(-Math.PI / 2); // lies in XZ, normal +Y, uv v runs towards -Z (leaf tip)
  return g;
}

function leafVertexHooks(shader, size, cols = LEAF_COLS, rows = LEAF_ROWS) {
  const s2 = ((size / 2) * (size / 2)).toFixed(6);
  shader.vertexShader = shader.vertexShader
    .replace('#include <common>', `#include <common>
      attribute float aVariant;
      attribute float aCurl;`)
    .replace('#include <uv_vertex>', `#include <uv_vertex>
      #ifdef USE_MAP
      vMapUv = (uv + vec2(mod(aVariant, ${cols}.0), ${rows - 1}.0 - floor(aVariant / ${cols}.0))) / vec2(${cols}.0, ${rows}.0);
      #endif`)
    .replace('#include <begin_vertex>', `#include <begin_vertex>
      transformed.y += aCurl * (position.x * position.x + 0.45 * position.z * position.z) / ${s2};
      transformed.y -= aCurl * 0.35 * position.z * abs(position.z) / ${s2};`);
  if (shader.vertexShader.includes('#include <beginnormal_vertex>')) {
    shader.vertexShader = shader.vertexShader.replace('#include <beginnormal_vertex>', `
      vec3 objectNormal = normalize(vec3(-2.0 * aCurl * position.x / ${s2}, 1.0, -(0.9 * aCurl * position.z - 0.7 * aCurl * abs(position.z)) / ${s2}));
      #ifdef USE_TANGENT
      vec3 objectTangent = vec3( tangent.xyz );
      #endif`);
  }
}

export function makeLeafMaterial(colorTex, auxTex, { size = 0.12, trans = 1.0, rough = 0.6, glow = 0.0, cols = LEAF_COLS, rows = LEAF_ROWS } = {}) {
  const mat = new THREE.MeshStandardMaterial({ map: colorTex, alphaTest: 0.5, side: THREE.DoubleSide, roughness: rough, metalness: 0 });
  mat.userData.uniforms = { uTrans: { value: trans }, uGlow: { value: glow } };
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.tAux = { value: auxTex };
    Object.assign(sh.uniforms, mat.userData.uniforms);
    leafVertexHooks(sh, size, cols, rows);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform sampler2D tAux; uniform float uTrans, uGlow;
        vec3 gTrans;`)
      .replace('#include <lights_physical_pars_fragment>', `#include <lights_physical_pars_fragment>
        void RE_Direct_Leaf( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal,
            const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material,
            inout ReflectedLight reflectedLight ) {
          RE_Direct_Physical( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
          float back = saturate( -dot( geometryNormal, directLight.direction ) );
          float fwd = pow( saturate( dot( -geometryViewDir, directLight.direction ) ), 4.0 );
          reflectedLight.directDiffuse += gTrans * directLight.color * back * ( 0.25 + 2.2 * fwd );
        }
        #undef RE_Direct
        #define RE_Direct RE_Direct_Leaf`)
      .replace('#include <map_fragment>', `#include <map_fragment>
        vec4 auxS = texture2D( tAux, vMapUv );
        // light through the lamina is deeper and more saturated than the reflected colour
        gTrans = diffuseColor.rgb * ( 0.35 + 1.1 * diffuseColor.rgb ) * auxS.g * uTrans;`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor = mix( 0.42, 0.85, auxS.b );`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        totalEmissiveRadiance += diffuseColor.rgb * uGlow;`);
  };
  mat.customProgramCacheKey = () => 'leaf' + size + '_' + cols;
  const depth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: colorTex, alphaTest: 0.5, side: THREE.DoubleSide });
  depth.onBeforeCompile = (sh) => leafVertexHooks(sh, size, cols, rows);
  depth.customProgramCacheKey = () => 'leafdepth' + size + '_' + cols;
  return { mat, depth };
}

/** InstancedMesh of leaves with per-instance variant + curl attributes. */
export function leafInstances(count, colorTex, auxTex, opts = {}) {
  const size = opts.size ?? 0.12;
  const geo = leafGeometry(size, opts.seg ?? 3);
  geo.setAttribute('aVariant', new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
  geo.setAttribute('aCurl', new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
  const { mat, depth } = makeLeafMaterial(colorTex, auxTex, { size, ...opts });
  const mesh = new THREE.InstancedMesh(geo, mat, count);
  mesh.customDepthMaterial = depth;
  mesh.castShadow = !!opts.castShadow;
  mesh.receiveShadow = opts.receiveShadow !== false;
  mesh.frustumCulled = false;
  return mesh;
}

// ---------------------------------------------------------------- dissolve --
/**
 * Adds a wind-driven dissolve to a standard material.  field d(p) in world space:
 *   d = 0.62 * dot(p - origin, windDir) / span + 0.38 * fbm(noise at p*freq)
 * The surface disappears where d < uProgress; a band just above it glows amber.
 */
export function addDissolve(mat, P) {
  const u = {
    uProgress: { value: -1 },
    uOrigin: { value: P.origin.clone() },
    uWindDir: { value: P.windDir.clone().normalize() },
    uSpan: { value: P.span },
    uFreq: { value: P.freq ?? 2.2 },
    uEdge: { value: P.edge ?? 0.05 },
    uEdgeColor: { value: new THREE.Color(P.edgeColor ?? 0xffa040) },
    uEdgeGain: { value: P.edgeGain ?? 6.0 },
    uChar: { value: P.char ?? 0.12 },
    tNoise3: { value: noise3Texture() },
  };
  mat.userData.dissolve = u;
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (sh, r) => {
    if (prev) prev(sh, r);
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>
        varying vec3 vWorldPosD;`)
      .replace('#include <worldpos_vertex>', `#include <worldpos_vertex>
        vWorldPosD = (modelMatrix * vec4(transformed, 1.0)).xyz;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec3 vWorldPosD;
        uniform float uProgress, uSpan, uFreq, uEdge, uEdgeGain, uChar; uniform vec3 uOrigin, uWindDir, uEdgeColor;
        ${GLSL_NOISE}
        float dissolveField(vec3 p) { return 0.62 * dot(p - uOrigin, uWindDir) / uSpan + 0.38 * fbmN3(p * uFreq * 0.1); }`)
      .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
        float dField = dissolveField(vWorldPosD);
        if (dField < uProgress) discard;
        float dAhead = (dField - uProgress) / max(uChar, 1e-4);`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        float band = 1.0 - clamp((dField - uProgress) / uEdge, 0.0, 1.0);
        float live = step(-0.5, uProgress);
        diffuseColor.rgb *= 1.0 - 0.75 * live * (1.0 - smoothstep(0.0, 1.0, dAhead));
        totalEmissiveRadiance += uEdgeColor * uEdgeGain * band * band * band * live;`);
  };
  const key = mat.customProgramCacheKey ? mat.customProgramCacheKey() : '';
  mat.customProgramCacheKey = () => key + 'dissolve';
  return u;
}

/** JS twin of dissolveField, used to time the particles that leave each surface point. */
export function dissolveFieldJS(p, P) {
  const d = P.windDir.clone().normalize();
  const f = (P.freq ?? 2.2) * 0.1;
  const q = p.clone().sub(P.origin);
  return 0.62 * q.dot(d) / P.span + 0.38 * fbmN3(p.x * f, p.y * f, p.z * f);
}

export { clamp };

// ------------------------------------------------------- SDF-mesh surfaces --
/**
 * Material for polygonized SDF meshes: per-vertex colour plus aRM = (roughness, metalness,
 * bump strength, scuff).  Detail (grain, creases, scuffs) comes from object-space noise.
 */
export function sdfMaterial({ grain = 90, bump = 1.0, scuffColor = [0.75, 0.6, 0.45] } = {}) {
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 1 });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.tNoise3 = { value: noise3Texture() };
    sh.uniforms.uScuff = { value: new THREE.Color(...scuffColor) };
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>
        attribute vec4 aRM; varying vec4 vRM; varying vec3 vObj;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vRM = aRM; vObj = position;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec4 vRM; varying vec3 vObj; uniform vec3 uScuff;
        ${GLSL_NOISE}`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        float nC = fbmN3(vObj * 9.0);
        float nS = fbmN3(vObj * 31.0 + 0.4);
        diffuseColor.rgb *= 0.78 + 0.44 * nC;
        float sc = vRM.w * smoothstep(0.42, 0.62, nS);
        diffuseColor.rgb = mix(diffuseColor.rgb, uScuff * (0.6 + 0.4 * nC), sc * 0.6);`)
      .replace('#include <roughnessmap_fragment>', `float roughnessFactor = clamp(vRM.x + 0.25 * (nS - 0.5) + sc * 0.25, 0.05, 1.0);`)
      .replace('#include <metalnessmap_fragment>', `float metalnessFactor = vRM.y;`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        {
          float hb = n3(vObj * ${grain.toFixed(1)}) * 0.6 + n3(vObj * ${(grain * 3.3).toFixed(1)} + 0.37) * 0.4;
          vec2 dh = vec2(dFdx(hb), dFdy(hb)) * vRM.z * ${bump.toFixed(3)};
          vec3 sx = dFdx(-vViewPosition), sy = dFdy(-vViewPosition);
          vec3 r1 = cross(sy, normal), r2 = cross(normal, sx);
          float det = dot(sx, r1) * faceDirection;
          normal = normalize(abs(det) * normal - sign(det) * (dh.x * r1 + dh.y * r2) * 0.0011);
        }`);
  };
  mat.customProgramCacheKey = () => 'sdfmat' + grain + bump;
  return mat;
}
