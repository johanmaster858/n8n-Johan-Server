// HDR post-processing: god rays, depth of field, bloom, ACES tone mapping + grade.
// Every pass is a full-screen triangle with a ShaderMaterial, so the chain is explicit.
import * as THREE from 'three';
import { noise3Texture, GLSL_NOISE } from './materials.js';

const VERT = /* glsl */ `
varying vec2 vUv;
void main() { vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

function rt(w, h, opts = {}) {
  return new THREE.WebGLRenderTarget(w, h, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    ...opts,
  });
}

const LINEARIZE = /* glsl */ `
uniform float uNear, uFar;
float linDepth(float d) { float z = d * 2.0 - 1.0; return 2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear)); }`;

export class Post {
  constructor(renderer, W, H) {
    this.r = renderer;
    this.W = W;
    this.H = H;
    const depthTex = new THREE.DepthTexture(W, H);
    depthTex.type = THREE.FloatType;
    this.rtScene = new THREE.WebGLRenderTarget(W, H, {
      type: THREE.HalfFloatType, depthBuffer: true, depthTexture: depthTex,
      minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
    });
    this.rtAccum = rt(W, H, { type: THREE.FloatType });
    this.rtFull = rt(W, H);
    this.rtFull2 = rt(W, H);
    const w2 = W >> 1, h2 = H >> 1;
    this.rtRayA = rt(w2, h2);
    this.rtRayB = rt(w2, h2);
    this.rtDofPrep = rt(w2, h2);
    this.rtDofBlur = rt(w2, h2);
    this.rtVolA = rt(W >> 2, H >> 2);
    this.rtVolB = rt(W >> 2, H >> 2);
    this.rtTile = rt(Math.ceil(W / 16), Math.ceil(H / 16));
    this.rtTile2 = rt(Math.ceil(W / 16), Math.ceil(H / 16));
    this.bloomDown = [];
    this.bloomUp = [];
    let bw = w2, bh = h2;
    for (let i = 0; i < 6; i++) {
      this.bloomDown.push(rt(Math.max(bw, 1), Math.max(bh, 1)));
      this.bloomUp.push(rt(Math.max(bw, 1), Math.max(bh, 1)));
      bw >>= 1;
      bh >>= 1;
    }
    this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const tri = new THREE.BufferGeometry();
    tri.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    this.quad = new THREE.Mesh(tri, null);
    this.quad.frustumCulled = false;
    this.qscene = new THREE.Scene();
    this.qscene.add(this.quad);
    this._build();
  }

  _mat(frag, uniforms, extra = {}) {
    return new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: frag, uniforms, depthTest: false, depthWrite: false, ...extra });
  }

  _build() {
    this.mCopy = this._mat(`
      uniform sampler2D tSrc; varying vec2 vUv;
      void main(){ gl_FragColor = vec4(texture2D(tSrc, vUv).rgb, 1.0); }`, { tSrc: { value: null } });

    // accumulate sub-frames (AA / motion blur)
    this.mAccum = this._mat(`
      uniform sampler2D tSrc; uniform float uW; varying vec2 vUv;
      void main(){ gl_FragColor = vec4(texture2D(tSrc, vUv).rgb * uW, uW); }`,
      { tSrc: { value: null }, uW: { value: 1 } },
      { blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor });

    // god rays: source = sky pixels near the sun
    this.mRaySrc = this._mat(`
      uniform sampler2D tColor, tDepth; uniform vec2 uSun; uniform float uAspect, uSunSize, uThresh, uSkyDist, uBase;
      ${LINEARIZE}
      varying vec2 vUv;
      void main(){
        float d = texture2D(tDepth, vUv).x;
        vec3 c = texture2D(tColor, vUv).rgb;
        float sky = d >= 0.999999 ? 1.0 : step(uSkyDist, linDepth(d));
        vec2 dd = (vUv - uSun) * vec2(uAspect, 1.0);
        float fall = exp(-dot(dd, dd) / (uSunSize * uSunSize));
        float l = max(dot(c, vec3(0.2126, 0.7152, 0.0722)) - uThresh, 0.0);
        gl_FragColor = vec4(c * sky * (uBase + (1.0 - uBase) * fall) * min(l, 8.0) / max(dot(c, vec3(0.2126,0.7152,0.0722)), 1e-4), 1.0);
      }`,
      { tColor: { value: null }, tDepth: { value: null }, uSun: { value: new THREE.Vector2() }, uAspect: { value: 1 },
        uSunSize: { value: 0.35 }, uThresh: { value: 0.2 }, uSkyDist: { value: 3000 }, uNear: { value: 0.1 }, uFar: { value: 1000 }, uBase: { value: 0.35 } });

    this.mRayBlur = this._mat(`
      uniform sampler2D tSrc; uniform vec2 uSun; uniform float uStep, uDecay; varying vec2 vUv;
      void main(){
        vec2 dir = (uSun - vUv) * uStep;
        vec2 uv = vUv; vec3 acc = vec3(0.0); float w = 1.0, wsum = 0.0;
        for (int i = 0; i < 20; i++) {
          acc += texture2D(tSrc, uv).rgb * w; wsum += w; w *= uDecay; uv += dir;
        }
        gl_FragColor = vec4(acc / wsum, 1.0);
      }`,
      { tSrc: { value: null }, uSun: { value: new THREE.Vector2() }, uStep: { value: 0.02 }, uDecay: { value: 0.96 } });

    // aerial perspective: analytic height fog, warm towards the sun, cool elsewhere
    this.mAtmo = this._mat(`
      uniform sampler2D tColor, tDepth; uniform mat4 uProjInv, uCamWorld; uniform vec3 uCamPos, uSun, uWarm, uCool;
      uniform float uDensity, uHeight, uSkyAmt, uPhase;
      varying vec2 vUv;
      void main(){
        vec3 c = texture2D(tColor, vUv).rgb;
        float d = texture2D(tDepth, vUv).x;
        vec4 v = uProjInv * vec4(vUv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
        v /= v.w;
        vec3 wp = (uCamWorld * vec4(v.xyz, 1.0)).xyz;
        vec3 dir = wp - uCamPos;
        float L = length(dir);
        dir /= L;
        bool sky = d >= 0.999999;
        if (sky) L = 60000.0;
        // optical depth of exponential height fog along the ray
        float h0 = uCamPos.y, k = dir.y * L / uHeight;
        float od = uDensity * L * exp(-h0 / uHeight) * (abs(k) > 1e-4 ? (1.0 - exp(-k)) / k : 1.0);
        float T = exp(-od);
        float cs = max(dot(dir, uSun), 0.0);
        vec3 fogC = mix(uCool, uWarm, pow(cs, uPhase)) * (1.0 + 2.5 * pow(cs, 12.0));
        float amt = sky ? uSkyAmt : 1.0;
        gl_FragColor = vec4(mix(c, fogC, (1.0 - T) * amt), 1.0);
      }`,
      { tColor: { value: null }, tDepth: { value: null }, uProjInv: { value: new THREE.Matrix4() }, uCamWorld: { value: new THREE.Matrix4() },
        uCamPos: { value: new THREE.Vector3() }, uSun: { value: new THREE.Vector3() }, uWarm: { value: new THREE.Color() },
        uCool: { value: new THREE.Color() }, uDensity: { value: 0.0001 }, uHeight: { value: 800 }, uSkyAmt: { value: 0.0 }, uPhase: { value: 3.0 } });

    // volumetric sun light: ray-march the sun's shadow map through a noisy haze
    this.mVol = this._mat(`
      #include <packing>
      uniform sampler2D tDepth, tShadow; uniform mat4 uShadowMatrix, uProjInv, uCamWorld;
      uniform vec3 uCamPos, uSunDir, uSunColor, uScroll; uniform vec2 uRes;
      uniform float uDensity, uMaxDist, uG, uOutside, uNoiseScale, uNoiseAmt, uHeight, uBias, uExtinct;
      ${GLSL_NOISE}
      varying vec2 vUv;
      float hg(float c, float g){ float g2 = g * g; return (1.0 - g2) / (12.566 * pow(1.0 + g2 - 2.0 * g * c, 1.5)); }
      void main(){
        float d = texture2D(tDepth, vUv).x;
        vec4 v = uProjInv * vec4(vUv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
        v /= v.w;
        vec3 wp = (uCamWorld * vec4(v.xyz, 1.0)).xyz;
        vec3 dir = wp - uCamPos;
        float L = length(dir);
        dir /= L;
        L = min(L, uMaxDist);
        const int N = 40;
        float st = L / float(N);
        float jit = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        float acc = 0.0, T = 1.0;
        for (int i = 0; i < N; i++) {
          vec3 p = uCamPos + dir * ((float(i) + jit) * st);
          vec4 sc = uShadowMatrix * vec4(p, 1.0);
          sc.xyz /= sc.w;
          float lit = uOutside;
          if (sc.x > 0.0 && sc.x < 1.0 && sc.y > 0.0 && sc.y < 1.0 && sc.z < 1.0)
            lit = step(sc.z - uBias, unpackRGBAToDepth(texture2D(tShadow, sc.xy)));
          float dens = uDensity * exp(-max(p.y, 0.0) / uHeight) * (1.0 - uNoiseAmt + 2.0 * uNoiseAmt * fbmN3(p * uNoiseScale + uScroll));
          acc += T * lit * dens * st;
          T *= exp(-dens * st * uExtinct);
        }
        float c = dot(dir, uSunDir);
        float ph = mix(hg(c, 0.2), hg(c, uG), 0.7);
        gl_FragColor = vec4(uSunColor * acc * ph, 1.0);
      }`,
      { tDepth: { value: null }, tShadow: { value: null }, uShadowMatrix: { value: new THREE.Matrix4() }, uProjInv: { value: new THREE.Matrix4() },
        uCamWorld: { value: new THREE.Matrix4() }, uCamPos: { value: new THREE.Vector3() }, uSunDir: { value: new THREE.Vector3() },
        uSunColor: { value: new THREE.Color() }, uScroll: { value: new THREE.Vector3() }, uRes: { value: new THREE.Vector2() },
        uDensity: { value: 0.05 }, uMaxDist: { value: 40 }, uG: { value: 0.75 }, uOutside: { value: 0.6 }, uNoiseScale: { value: 0.08 },
        uNoiseAmt: { value: 0.5 }, uHeight: { value: 14 }, uBias: { value: 0.0015 }, uExtinct: { value: 0.3 }, tNoise3: { value: noise3Texture() } });
    this.mBlur = this._mat(`
      uniform sampler2D tSrc; uniform vec2 uDir; varying vec2 vUv;
      void main(){
        vec3 s = texture2D(tSrc, vUv).rgb * 0.2270270270;
        s += (texture2D(tSrc, vUv + uDir * 1.3846153846).rgb + texture2D(tSrc, vUv - uDir * 1.3846153846).rgb) * 0.3162162162;
        s += (texture2D(tSrc, vUv + uDir * 3.2307692308).rgb + texture2D(tSrc, vUv - uDir * 3.2307692308).rgb) * 0.0702702703;
        gl_FragColor = vec4(s, 1.0);
      }`, { tSrc: { value: null }, uDir: { value: new THREE.Vector2() } });

    // depth of field
    this.mDofPrep = this._mat(`
      uniform sampler2D tColor, tDepth; uniform float uFocus, uAmount, uMaxCoc, uNearMul; uniform vec2 uTexel;
      ${LINEARIZE}
      varying vec2 vUv;
      // signed circle of confusion (full-res px): negative in front of the focus plane
      float coc(vec2 uv){ float z = linDepth(texture2D(tDepth, uv).x); float c = uAmount * (1.0 - uFocus / z);
        c = c < 0.0 ? c * uNearMul : c; return clamp(c, -uMaxCoc, uMaxCoc); }
      void main(){
        vec3 c = texture2D(tColor, vUv).rgb;
        float a = coc(vUv + uTexel * vec2(-0.5, -0.5)), b = coc(vUv + uTexel * vec2(0.5, -0.5));
        float e = coc(vUv + uTexel * vec2(-0.5, 0.5)), f = coc(vUv + uTexel * vec2(0.5, 0.5));
        // keep the most-in-front value so near blur is not lost
        float m = min(min(a, b), min(e, f));
        float M = max(max(a, b), max(e, f));
        gl_FragColor = vec4(c, m < -0.5 ? m : M);
      }`,
      { tColor: { value: null }, tDepth: { value: null }, uFocus: { value: 2 }, uAmount: { value: 0 }, uMaxCoc: { value: 36 },
        uNearMul: { value: 1 }, uTexel: { value: new THREE.Vector2() }, uNear: { value: 0.1 }, uFar: { value: 1000 } });

    this.mTileMax = this._mat(`
      uniform sampler2D tSrc; uniform vec2 uSrcTexel; uniform int uN; varying vec2 vUv;
      void main(){
        float m = 0.0;
        for (int j = 0; j < 8; j++) for (int i = 0; i < 8; i++) {
          vec2 o = (vec2(float(i), float(j)) - 3.5) * uSrcTexel;
          m = max(m, abs(texture2D(tSrc, vUv + o).a));
        }
        gl_FragColor = vec4(m);
      }`, { tSrc: { value: null }, uSrcTexel: { value: new THREE.Vector2() }, uN: { value: 8 } });

    this.mTileDilate = this._mat(`
      uniform sampler2D tSrc; uniform vec2 uTexel; varying vec2 vUv;
      void main(){ float m = 0.0;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) m = max(m, texture2D(tSrc, vUv + vec2(float(i), float(j)) * uTexel).r);
        gl_FragColor = vec4(m); }`, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() } });

    this.mDofBlur = this._mat(`
      uniform sampler2D tSrc, tTile; uniform vec2 uTexel; varying vec2 vUv;
      void main(){
        vec4 c0 = texture2D(tSrc, vUv);
        float R = texture2D(tTile, vUv).r * 0.5;           // gather radius (half-res px)
        float cc = abs(c0.a) * 0.5;
        if (R < 0.6) { gl_FragColor = vec4(c0.rgb, 0.0); return; }
        vec3 acc = vec3(0.0); float wsum = 0.0, fg = 0.0, fgw = 0.0;
        const int N = 56;
        for (int i = 0; i < N; i++) {
          float fi = float(i) + 0.5;
          float r = sqrt(fi / float(N)) * R;
          float a = fi * 2.39996323;
          vec2 o = vec2(cos(a), sin(a)) * r;
          vec4 s = texture2D(tSrc, vUv + o * uTexel);
          float sc = abs(s.a) * 0.5;
          // samples behind the centre may not spread wider than the centre itself
          if (s.a > c0.a) sc = min(sc, max(cc, 0.0) * 2.0);
          float w = clamp(sc - r + 1.0, 0.0, 1.0) / max(sc * sc, 1.0);
          acc += s.rgb * w; wsum += w;
          if (s.a < -1.0 && -s.a > abs(c0.a) + 2.0) fg += w;
          fgw += w;
        }
        float ow = 1.0 / max(cc * cc, 1.0);
        acc += c0.rgb * ow; wsum += ow;
        gl_FragColor = vec4(acc / max(wsum, 1e-5), fg / max(fgw, 1e-5));
      }`,
      { tSrc: { value: null }, tTile: { value: null }, uTexel: { value: new THREE.Vector2() } });

    this.mDofComp = this._mat(`
      uniform sampler2D tColor, tBlur, tPrep; varying vec2 vUv;
      void main(){
        vec3 sharp = texture2D(tColor, vUv).rgb;
        vec4 b = texture2D(tBlur, vUv);
        float coc = abs(texture2D(tPrep, vUv).a);
        float k = max(smoothstep(0.8, 3.0, coc), smoothstep(0.05, 0.4, b.a));
        gl_FragColor = vec4(mix(sharp, b.rgb, k), 1.0);
      }`, { tColor: { value: null }, tBlur: { value: null }, tPrep: { value: null } });

    // bloom (dual filter)
    this.mDown = this._mat(`
      uniform sampler2D tSrc; uniform vec2 uTexel; uniform float uKnee, uFirst; varying vec2 vUv;
      vec3 pre(vec3 c){ if (uFirst < 0.5) return c; float l = max(max(c.r, c.g), c.b);
        float s = clamp(l - uKnee, 0.0, 4.0 * uKnee); s = s * s / (16.0 * uKnee + 1e-4);
        float k = max(s, l - uKnee) / max(l, 1e-4); return c * max(k, 0.0); }
      void main(){
        vec3 s = texture2D(tSrc, vUv).rgb * 4.0;
        s += texture2D(tSrc, vUv + uTexel * vec2(-1, -1)).rgb;
        s += texture2D(tSrc, vUv + uTexel * vec2(1, -1)).rgb;
        s += texture2D(tSrc, vUv + uTexel * vec2(-1, 1)).rgb;
        s += texture2D(tSrc, vUv + uTexel * vec2(1, 1)).rgb;
        gl_FragColor = vec4(pre(min(s / 8.0, vec3(64.0))), 1.0);
      }`, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uKnee: { value: 1.0 }, uFirst: { value: 0 } });
    this.mUp = this._mat(`
      uniform sampler2D tSrc, tPrev; uniform vec2 uTexel; uniform float uHasPrev; varying vec2 vUv;
      void main(){
        vec3 s = texture2D(tSrc, vUv + uTexel * vec2(-2, 0)).rgb + texture2D(tSrc, vUv + uTexel * vec2(2, 0)).rgb
               + texture2D(tSrc, vUv + uTexel * vec2(0, -2)).rgb + texture2D(tSrc, vUv + uTexel * vec2(0, 2)).rgb;
        s += (texture2D(tSrc, vUv + uTexel * vec2(-1, -1)).rgb + texture2D(tSrc, vUv + uTexel * vec2(1, -1)).rgb
            + texture2D(tSrc, vUv + uTexel * vec2(-1, 1)).rgb + texture2D(tSrc, vUv + uTexel * vec2(1, 1)).rgb) * 2.0;
        s /= 12.0;
        vec3 p = uHasPrev > 0.5 ? texture2D(tPrev, vUv).rgb : vec3(0.0);
        gl_FragColor = vec4(s + p, 1.0);
      }`, { tSrc: { value: null }, tPrev: { value: null }, uTexel: { value: new THREE.Vector2() }, uHasPrev: { value: 0 } });

    // final: rays + bloom + exposure + ACES + grade + vignette
    this.mFinal = this._mat(`
      uniform sampler2D tColor, tRays, tBloom, tVol; uniform float uVol;
      uniform float uExposure, uRays, uBloom, uSat, uContrast, uVignette, uTemp, uTint, uFade, uDesatShadows;
      uniform vec3 uRayColor, uLift, uGamma, uGain;
      uniform float uAspect;
      varying vec2 vUv;
      vec3 aces(vec3 x){ const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14; return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0,1.0); }
      vec3 toSRGB(vec3 c){ return mix(c*12.92, 1.055*pow(c, vec3(1.0/2.4)) - 0.055, step(0.0031308, c)); }
      void main(){
        vec3 c = texture2D(tColor, vUv).rgb;
        c += texture2D(tRays, vUv).rgb * uRays * uRayColor;
        c += texture2D(tBloom, vUv).rgb * uBloom;
        c += texture2D(tVol, vUv).rgb * uVol;
        c *= uExposure;
        // white balance in linear light
        c *= vec3(1.0 + uTemp * 0.10 + uTint * 0.02, 1.0 - uTint * 0.04, 1.0 - uTemp * 0.12 + uTint * 0.02);
        c = aces(c);
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        float satk = uSat * mix(1.0, smoothstep(0.0, 0.35, l), uDesatShadows);
        c = max(vec3(l) + (c - vec3(l)) * satk, 0.0);
        c = (c - 0.5) * uContrast + 0.5;
        c = clamp(c, 0.0, 1.0);
        c = pow(c * uGain + uLift * (1.0 - c), 1.0 / uGamma);
        vec2 d = (vUv - 0.5) * vec2(uAspect, 1.0);
        c *= 1.0 - uVignette * smoothstep(0.25, 0.95, length(d) * 1.25);
        c = c * (1.0 - uFade) + uFade * 0.06;
        gl_FragColor = vec4(toSRGB(clamp(c, 0.0, 1.0)), 1.0);
      }`,
      { tColor: { value: null }, tRays: { value: null }, tBloom: { value: null }, tVol: { value: null }, uVol: { value: 0 }, uExposure: { value: 1 }, uRays: { value: 0 },
        uBloom: { value: 0.1 }, uSat: { value: 1 }, uContrast: { value: 1 }, uVignette: { value: 0.3 }, uTemp: { value: 0 },
        uTint: { value: 0 }, uFade: { value: 0 }, uDesatShadows: { value: 0 }, uRayColor: { value: new THREE.Color(1, 0.8, 0.55) },
        uLift: { value: new THREE.Vector3(0, 0, 0) }, uGamma: { value: new THREE.Vector3(1, 1, 1) },
        uGain: { value: new THREE.Vector3(1, 1, 1) }, uAspect: { value: this.W / this.H } });
  }

  pass(mat, target) {
    this.quad.material = mat;
    this.r.setRenderTarget(target);
    this.r.render(this.qscene, this.cam);
  }

  /**
   * Render scene -> HDR -> post -> screen.
   * P: { exposure, rays, raySun(Vector3 dir or null), dof:{focus, amount}, bloom, grade... }
   * subframes: array of callbacks that set up the scene/camera for each sub-sample.
   */
  render(scene, camera, P, subframes = null, overlay = null) {
    if (typeof window !== 'undefined' && window.POVERRIDE) P = { ...P, ...window.POVERRIDE };
    const r = this.r;
    r.autoClear = true;
    const n = subframes ? subframes.length : 1;
    if (n > 1) {
      r.setRenderTarget(this.rtAccum);
      r.setClearColor(0x000000, 0);
      r.clear(true, false, false);
      for (let i = 0; i < n; i++) {
        subframes[i]();
        r.setRenderTarget(this.rtScene);
        r.render(scene, camera);
        this.mAccum.uniforms.tSrc.value = this.rtScene.texture;
        this.mAccum.uniforms.uW.value = 1 / n;
        r.autoClear = false;
        this.pass(this.mAccum, this.rtAccum);
        r.autoClear = true;
      }
    } else {
      if (subframes) subframes[0]();
      r.setRenderTarget(this.rtScene);
      r.render(scene, camera);
    }
    let colorTex = n > 1 ? this.rtAccum.texture : this.rtScene.texture;
    const depthTex = this.rtScene.depthTexture;
    const W = this.W, H = this.H;

    // ---- aerial perspective
    if (P.atmo) {
      const A = P.atmo, u = this.mAtmo.uniforms;
      camera.updateMatrixWorld();
      u.tColor.value = colorTex;
      u.tDepth.value = depthTex;
      u.uProjInv.value.copy(camera.projectionMatrixInverse);
      u.uCamWorld.value.copy(camera.matrixWorld);
      u.uCamPos.value.setFromMatrixPosition(camera.matrixWorld);
      u.uSun.value.copy(P.sunDir).normalize();
      u.uWarm.value.set(A.warm ?? 0xffb070);
      if (A.warmGain) u.uWarm.value.multiplyScalar(A.warmGain);
      u.uCool.value.set(A.cool ?? 0x8090b0);
      if (A.coolGain) u.uCool.value.multiplyScalar(A.coolGain);
      u.uDensity.value = A.density ?? 0.0001;
      u.uHeight.value = A.height ?? 800;
      u.uSkyAmt.value = A.sky ?? 0.0;
      u.uPhase.value = A.phase ?? 3.0;
      this.pass(this.mAtmo, this.rtFull2);
      colorTex = this.rtFull2.texture;
    }

    // ---- god rays
    let raysTex = null;
    if (P.rays > 0 && P.sunDir) {
      const v = P.sunDir.clone().multiplyScalar(1000).add(camera.position).project(camera);
      const behind = P.sunDir.dot(camera.getWorldDirection(new THREE.Vector3())) < 0.05;
      if (!behind) {
        const sun = new THREE.Vector2(v.x * 0.5 + 0.5, v.y * 0.5 + 0.5);
        const u = this.mRaySrc.uniforms;
        u.tColor.value = colorTex;
        u.tDepth.value = depthTex;
        u.uSun.value.copy(sun);
        u.uAspect.value = W / H;
        u.uSunSize.value = P.raySunSize ?? 0.35;
        u.uBase.value = P.rayBase ?? 0.35;
        u.uThresh.value = P.rayThresh ?? 0.2;
        u.uSkyDist.value = P.skyDist ?? camera.far * 0.8;
        u.uNear.value = camera.near;
        u.uFar.value = camera.far;
        this.pass(this.mRaySrc, this.rtRayA);
        const b = this.mRayBlur.uniforms;
        b.uSun.value.copy(sun);
        b.uDecay.value = P.rayDecay ?? 0.965;
        let src = this.rtRayA, dst = this.rtRayB;
        for (const step of [0.012, 0.03, 0.075]) {
          b.tSrc.value = src.texture;
          b.uStep.value = step * (P.rayLength ?? 1.0);
          this.pass(this.mRayBlur, dst);
          [src, dst] = [dst, src];
        }
        raysTex = src.texture;
      }
    }

    // ---- volumetric light (needs the shadow map rendered with the scene above)
    let volTex = null;
    const V = P.volume;
    if (V && V.light && V.light.shadow.map && V.gain > 0) {
      const u = this.mVol.uniforms;
      camera.updateMatrixWorld();
      u.tDepth.value = depthTex;
      u.tShadow.value = V.light.shadow.map.texture;
      u.uShadowMatrix.value.copy(V.light.shadow.matrix);
      u.uProjInv.value.copy(camera.projectionMatrixInverse);
      u.uCamWorld.value.copy(camera.matrixWorld);
      u.uCamPos.value.setFromMatrixPosition(camera.matrixWorld);
      u.uSunDir.value.copy(P.sunDir).normalize();
      u.uSunColor.value.copy(V.light.color).multiplyScalar(V.light.intensity);
      u.uDensity.value = V.density ?? 0.05;
      u.uMaxDist.value = V.maxDist ?? 40;
      u.uG.value = V.g ?? 0.75;
      u.uOutside.value = V.outside ?? 0.6;
      u.uNoiseScale.value = V.noiseScale ?? 0.08;
      u.uNoiseAmt.value = V.noiseAmt ?? 0.5;
      u.uHeight.value = V.height ?? 14;
      u.uBias.value = V.bias ?? 0.0015;
      u.uExtinct.value = V.extinct ?? 0.3;
      if (V.scroll) u.uScroll.value.copy(V.scroll);
      this.pass(this.mVol, this.rtVolA);
      const b = this.mBlur.uniforms;
      b.tSrc.value = this.rtVolA.texture;
      b.uDir.value.set(1 / this.rtVolA.width, 0);
      this.pass(this.mBlur, this.rtVolB);
      b.tSrc.value = this.rtVolB.texture;
      b.uDir.value.set(0, 1 / this.rtVolA.height);
      this.pass(this.mBlur, this.rtVolA);
      volTex = this.rtVolA.texture;
    }

    // ---- depth of field
    let afterDof = colorTex;
    if (P.dof && P.dof.amount > 0.3) {
      const u = this.mDofPrep.uniforms;
      u.tColor.value = colorTex;
      u.tDepth.value = depthTex;
      u.uFocus.value = P.dof.focus;
      u.uAmount.value = P.dof.amount;
      u.uMaxCoc.value = P.dof.max ?? 40;
      u.uNearMul.value = P.dof.nearMul ?? 1.0;
      u.uNear.value = camera.near;
      u.uFar.value = camera.far;
      u.uTexel.value.set(1 / W, 1 / H);
      this.pass(this.mDofPrep, this.rtDofPrep);
      this.mTileMax.uniforms.tSrc.value = this.rtDofPrep.texture;
      this.mTileMax.uniforms.uSrcTexel.value.set(2 / W, 2 / H);
      this.pass(this.mTileMax, this.rtTile);
      this.mTileDilate.uniforms.tSrc.value = this.rtTile.texture;
      this.mTileDilate.uniforms.uTexel.value.set(1 / this.rtTile.width, 1 / this.rtTile.height);
      this.pass(this.mTileDilate, this.rtTile2);
      const bl = this.mDofBlur.uniforms;
      bl.tSrc.value = this.rtDofPrep.texture;
      bl.tTile.value = this.rtTile2.texture;
      bl.uTexel.value.set(2 / W, 2 / H);
      this.pass(this.mDofBlur, this.rtDofBlur);
      const c = this.mDofComp.uniforms;
      c.tColor.value = colorTex;
      c.tBlur.value = this.rtDofBlur.texture;
      c.tPrep.value = this.rtDofPrep.texture;
      this.pass(this.mDofComp, this.rtFull);
      afterDof = this.rtFull.texture;
    }

    // ---- overlay particles (depth-tested against the scene, bokeh-sized) after DOF
    if (overlay) {
      if (afterDof !== this.rtFull.texture) {
        this.mCopy.uniforms.tSrc.value = afterDof;
        this.pass(this.mCopy, this.rtFull);
        afterDof = this.rtFull.texture;
      }
      r.autoClear = false;
      r.setRenderTarget(this.rtFull);
      r.render(overlay, camera);
      r.autoClear = true;
    }

    // ---- bloom
    let src = afterDof;
    for (let i = 0; i < this.bloomDown.length; i++) {
      const d = this.mDown.uniforms;
      d.tSrc.value = src;
      const sw = i === 0 ? W : this.bloomDown[i - 1].width, sh = i === 0 ? H : this.bloomDown[i - 1].height;
      d.uTexel.value.set(1 / sw, 1 / sh);
      d.uKnee.value = P.bloomKnee ?? 0.9;
      d.uFirst.value = i === 0 ? 1 : 0;
      this.pass(this.mDown, this.bloomDown[i]);
      src = this.bloomDown[i].texture;
    }
    let prev = null;
    for (let i = this.bloomDown.length - 1; i >= 0; i--) {
      const u = this.mUp.uniforms;
      const s = i === this.bloomDown.length - 1 ? this.bloomDown[i] : this.bloomUp[i + 1];
      u.tSrc.value = s.texture;
      u.uTexel.value.set(1 / s.width, 1 / s.height);
      u.tPrev.value = this.bloomDown[i].texture;
      u.uHasPrev.value = i === this.bloomDown.length - 1 ? 0 : 1;
      this.pass(this.mUp, this.bloomUp[i]);
      prev = this.bloomUp[i];
    }

    // ---- final grade
    const f = this.mFinal.uniforms;
    f.tColor.value = afterDof;
    f.tRays.value = raysTex || this.rtRayB.texture;
    f.uRays.value = raysTex ? P.rays : 0;
    f.uRayColor.value.set(P.rayColor ?? 0xffc98a);
    f.tBloom.value = prev.texture;
    f.tVol.value = volTex || this.rtVolB.texture;
    f.uVol.value = volTex ? V.gain : 0;
    f.uBloom.value = P.bloom ?? 0.08;
    f.uExposure.value = P.exposure ?? 1;
    f.uSat.value = P.saturation ?? 1;
    f.uContrast.value = P.contrast ?? 1;
    f.uVignette.value = P.vignette ?? 0.3;
    f.uTemp.value = P.temperature ?? 0;
    f.uTint.value = P.tint ?? 0;
    f.uFade.value = P.fade ?? 0;
    f.uDesatShadows.value = P.desatShadows ?? 0;
    f.uLift.value.set(...(P.lift ?? [0, 0, 0]));
    f.uGamma.value.set(...(P.gamma ?? [1, 1, 1]));
    f.uGain.value.set(...(P.gain ?? [1, 1, 1]));
    this.pass(this.mFinal, null);
  }
}
