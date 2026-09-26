// 14-17 s memory: a hand holds a red maple leaf up to the low sun.
import * as THREE from 'three';
import { Rng, clamp, smooth, smoother, ramp, lerp } from '../core/rng.js';
import { leafInstances, noise3Texture, GLSL_NOISE } from '../core/materials.js';
import { makeHeroLeaf } from '../core/leafatlas.js';
import { makeEnvironment } from '../core/env.js';
import { makeSky } from '../core/sky.js';
import { Dust } from '../core/particles.js';
import * as TX from '../core/textures.js';
import { buildTrees } from './trees.js';
import { buildHand } from './handmodel.js';

const T0 = 13.8, T1 = 17.5; // time span this scene covers
const LEAF_SIZE = 0.125;

export function skinMaterial() {
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0 });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uSSS = { value: new THREE.Color(1.0, 0.3, 0.14) };
    sh.uniforms.tNoise3 = { value: noise3Texture() };
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>
        attribute float aThin; attribute float aNail; varying float vThin; varying float vNail; varying vec3 vObjPos;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vThin = aThin; vNail = aNail; vObjPos = position;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
        varying float vThin; varying float vNail; varying vec3 vObjPos; uniform vec3 uSSS; float gThin;
        ${GLSL_NOISE}`)
      .replace('#include <lights_physical_pars_fragment>', `#include <lights_physical_pars_fragment>
        void RE_Direct_Skin( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal,
            const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material,
            inout ReflectedLight reflectedLight ) {
          RE_Direct_Physical( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
          float nl = dot( geometryNormal, directLight.direction );
          // soft wrap lighting: light bleeds past the terminator in skin
          float wrapD = saturate( ( nl + 0.4 ) / 1.4 ) - saturate( nl );
          reflectedLight.directDiffuse += directLight.color * material.diffuseColor * RECIPROCAL_PI * wrapD * vec3( 1.0, 0.55, 0.45 ) * 1.6;
          // transmitted light through thin parts (fingers, edges) when backlit
          float back = saturate( -nl * 0.6 + 0.4 );
          float fwd = pow( saturate( dot( -geometryViewDir, directLight.direction ) ), 3.0 );
          float rim = pow( 1.0 - saturate( dot( geometryNormal, geometryViewDir ) ), 2.0 );
          reflectedLight.directDiffuse += uSSS * directLight.color * gThin * ( back * back * ( 0.05 + 1.4 * fwd ) + rim * fwd * 0.9 );
        }
        #undef RE_Direct
        #define RE_Direct RE_Direct_Skin`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor = mix( roughnessFactor, 0.25, vNail );
        gThin = vThin;`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        {
          // pores and fine creases (bump from object-space noise)
          float hb = n3( vObjPos * 42.0 ) * 0.55 + n3( vObjPos * 150.0 + 0.3 ) * 0.45;
          vec2 dh = vec2( dFdx( hb ), dFdy( hb ) ) * 0.35 * ( 1.0 - vNail );
          vec3 sx = dFdx( -vViewPosition ), sy = dFdy( -vViewPosition );
          vec3 r1 = cross( sy, normal ), r2 = cross( normal, sx );
          float det = dot( sx, r1 ) * faceDirection;
          vec3 gr = sign( det ) * ( dh.x * r1 + dh.y * r2 );
          normal = normalize( abs( det ) * normal - gr * 0.0025 );
        }`);
  };
  mat.customProgramCacheKey = () => 'skin';
  return mat;
}

export class HandScene {
  constructor(ctx) {
    this.ctx = ctx;
  }

  init() {
    const { renderer, clusters } = this.ctx;
    const scene = (this.scene = new THREE.Scene());
    const rng = new Rng(314);
    this.camera = new THREE.PerspectiveCamera(40, this.ctx.W / this.ctx.H, 0.02, 4000);
    // the sun sits just behind the leaf's right edge
    this.sunDir = new THREE.Vector3(0.02, 0.045, -0.4).normalize();

    scene.add(makeSky({ sunDir: this.sunDir, horizon: [1.1, 0.7, 0.45], zenith: [0.28, 0.4, 0.68], glow1: 0.7, glow2: 1.3, disk: 45,
      clouds: 0.6, gain: 1.0 }));
    scene.fog = new THREE.FogExp2(new THREE.Color(0.55, 0.36, 0.22), 0.01);
    scene.environment = makeEnvironment(renderer, { sunDir: this.sunDir, canopy: 0.7, trunks: 0.8, glowGain: 1.2, glowPow: 8 });
    scene.environmentIntensity = 0.75;

    const sun = (this.sun = new THREE.DirectionalLight(new THREE.Color(1.0, 0.72, 0.45), 6.5));
    sun.position.copy(this.sunDir).multiplyScalar(3).add(new THREE.Vector3(0, 0, -0.4));
    sun.target.position.set(0, -0.05, -0.4);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = sc.bottom = -0.3;
    sc.right = sc.top = 0.3;
    sc.near = 0.5;
    sc.far = 6;
    sun.shadow.bias = -0.0002;
    sun.shadow.normalBias = 0.002;
    sun.shadow.radius = 3;
    scene.add(sun, sun.target);
    const fill = new THREE.DirectionalLight(new THREE.Color(1.0, 0.78, 0.6), 0.45);
    fill.position.set(0.4, 0.3, 1);
    scene.add(fill);

    // ground far below, and trees all around (heavily out of focus)
    const soil = TX.soil(256, 41);
    soil.map.repeat.set(80, 80);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(300, 300), new THREE.MeshStandardMaterial({ map: soil.map, color: new THREE.Color(1.2, 0.55, 0.3), roughness: 1 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1.55;
    scene.add(ground);
    const treePos = [];
    for (let i = 0; i < 140 && treePos.length < 44; i++) {
      const z = rng.range(-45, -5), x = rng.range(-22, 22);
      // leave the line of sight to the sun open
      if (Math.abs(x - (this.sunDir.x / this.sunDir.z) * z) < 2.2 + 0.05 * Math.abs(z)) continue;
      if (treePos.some(([px, pz]) => Math.hypot(px - x, pz - z) < 2)) continue;
      treePos.push([x, z]);
    }
    const trees = buildTrees(scene, { rng, clusters, positions: treePos, cardSize: 1.6, castShadow: false });
    trees.trunks.position.y = -1.6;
    trees.crowns.position.y = -1.6;
    // understory maples: warm bokeh behind the leaf
    const shrubPos = [];
    for (let i = 0; i < 400 && shrubPos.length < 60; i++) {
      const z = rng.range(-30, -3), x = rng.range(-14, 14);
      if (Math.abs(x - (this.sunDir.x / this.sunDir.z) * z) < 0.8 + 0.04 * Math.abs(z)) continue;
      shrubPos.push([x, rng.range(-1.2, 1.6), z, rng.range(0.8, 1.8)]);
    }
    const nPer = 26;
    const shrubs = leafInstances(shrubPos.length * nPer, clusters.colorTex, clusters.auxTex, { size: 1.0, seg: 1, cols: 2, rows: 2, trans: 1.3 });
    const m4 = new THREE.Matrix4(), qq = new THREE.Quaternion(), e = new THREE.Euler(), col = new THREE.Color();
    let k = 0;
    for (const [x, y, z, r] of shrubPos) {
      const mood = rng.pick([[0, 0, 1], [0, 3], [1, 3], [0, 1]]);
      for (let j = 0; j < nPer; j++) {
        e.set(rng.range(-1.2, 1.2), rng.range(0, 6.28), rng.range(-1.2, 1.2));
        qq.setFromEuler(e);
        m4.compose(new THREE.Vector3(x + rng.normal() * 0.6 * r, y + rng.normal() * 0.4 * r, z + rng.normal() * 0.6 * r), qq,
          new THREE.Vector3().setScalar(rng.range(0.7, 1.2)));
        shrubs.setMatrixAt(k, m4);
        col.setScalar(rng.range(0.55, 0.9));
        shrubs.setColorAt(k, col);
        shrubs.geometry.attributes.aVariant.array[k] = rng.pick(mood);
        k++;
      }
    }
    shrubs.instanceMatrix.needsUpdate = true;
    shrubs.instanceColor.needsUpdate = true;
    shrubs.geometry.attributes.aVariant.needsUpdate = true;
    scene.add(shrubs);

    // the hand
    const t0 = performance.now();
    const hand = buildHand({ cell: 0.00105 });
    console.log('hand mesh', hand.geometry.attributes.position.count, 'verts', (performance.now() - t0).toFixed(0), 'ms');
    this.handData = hand;
    this.hand = new THREE.Mesh(hand.geometry, skinMaterial());
    this.hand.castShadow = true;
    this.hand.receiveShadow = true;
    // sweater cuff over the wrist
    const kn = TX.knit(512, [0.8, 0.72, 0.6], 61, { cols: 18, rows: 26 });
    const cuffGeo = new THREE.CylinderGeometry(0.041, 0.047, 0.2, 72, 24, true);
    cuffGeo.translate(0, -0.135, -0.004);
    const cuffMat = new THREE.MeshStandardMaterial({ map: kn.map, normalMap: kn.normalMap, roughness: 0.95, side: THREE.DoubleSide,
      color: new THREE.Color(1, 1, 1) });
    cuffMat.map.repeat.set(3, 2);
    cuffMat.normalMap.repeat.set(3, 2);
    const cuff = new THREE.Mesh(cuffGeo, cuffMat);
    cuff.castShadow = true;
    cuff.receiveShadow = true;
    const lip = new THREE.Mesh(new THREE.TorusGeometry(0.0425, 0.0058, 16, 72), cuffMat);
    lip.rotation.x = Math.PI / 2;
    lip.position.set(0, -0.036, -0.004);
    this.handGroup = new THREE.Group();
    this.handGroup.add(this.hand, cuff, lip);
    scene.add(this.handGroup);

    // the leaf
    const hero = makeHeroLeaf(1024, 0, 11);
    const lc = new THREE.CanvasTexture(hero.color);
    lc.colorSpace = THREE.SRGBColorSpace;
    lc.anisotropy = 8;
    const la = new THREE.CanvasTexture(hero.aux);
    this.leaf = leafInstances(1, lc, la, { size: LEAF_SIZE, seg: 40, cols: 1, rows: 1, castShadow: true, trans: 1.35 });
    this.leaf.geometry.attributes.aCurl.array[0] = 0.012;
    this.leaf.geometry.attributes.aCurl.needsUpdate = true;
    scene.add(this.leaf);

    // dust motes drifting in the backlight
    this.dust = new Dust(500, new THREE.Box3(new THREE.Vector3(-0.8, -0.6, -3), new THREE.Vector3(0.8, 0.8, -0.15)), { seed: 9, gain: 1.4 });
    this.overlay = new THREE.Scene();
    this.overlay.add(this.dust.e.points);
  }

  // leaf pose: centre, twirl about its stem, tilt
  leafPose(t) {
    const u = clamp((t - T0) / (T1 - T0));
    const c = new THREE.Vector3(lerp(0.02, 0.011, smoother(u)), lerp(0.02, 0.027, smoother(u)), -0.4);
    c.x += 0.0015 * Math.sin(t * 1.3);
    c.y += 0.001 * Math.sin(t * 1.9 + 1);
    const twirl = lerp(-0.32, 0.22, smoother(u)) + 0.04 * Math.sin(t * 2.1);
    const tilt = -0.12 + 0.03 * Math.sin(t * 1.1);
    return { c, twirl, tilt };
  }

  render(t, post) {
    const { c, twirl, tilt } = this.leafPose(t);
    // leaf frame: blade upright facing the camera, turned about the vertical stem axis
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2 + tilt, twirl, 0.08, 'YXZ'));
    const m = new THREE.Matrix4().compose(c, q, new THREE.Vector3(1, 1, 1));
    this.leaf.setMatrixAt(0, m);
    this.leaf.instanceMatrix.needsUpdate = true;
    // stem end in world space: 0.29 of the leaf size below the centre along the blade
    const stemEnd = new THREE.Vector3(0, 0, 0.29 * LEAF_SIZE + 0.004).applyQuaternion(q).add(c);
    // hand: thumb side towards the camera, fingers leaning left and away
    const hq = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.42, -0.95 + 0.35 * (twirl + 0.05), 0.36, 'ZXY'));
    this.handGroup.quaternion.copy(hq);
    const pinchW = this.handData.pinch.clone().applyQuaternion(hq);
    this.handGroup.position.copy(stemEnd).sub(pinchW);
    this.handGroup.updateMatrixWorld(true);

    const cam = this.camera;
    const u = clamp((t - T0) / (T1 - T0));
    cam.position.set(0.002 * Math.sin(t * 0.9), 0.0015 * Math.sin(t * 1.3 + 2), 0.012 * smoother(u));
    cam.lookAt(new THREE.Vector3(0.0, 0.052, -1));
    cam.updateMatrixWorld(true);
    this.dust.update(t, new THREE.Vector3(0.02, 0.012, 0.0), this.sunDir, cam.position);
    const focus = c.distanceTo(cam.position);
    const P = {
      sunDir: this.sunDir,
      exposure: 0.72,
      rays: 0.16,
      rayBase: 0.0,
      raySunSize: 0.1,
      rayThresh: 5.0,
      rayColor: 0xffb070,
      rayLength: 1.1,
      skyDist: 50,
      bloom: 0.04,
      bloomKnee: 1.8,
      dof: { focus, amount: 36, max: 44, nearMul: 1.0 },
      saturation: 1.15,
      contrast: 1.08,
      temperature: 0.12,
      vignette: 0.34,
      volume: { light: this.sun, gain: 1.0, density: 0.0025, maxDist: 30, g: 0.8, outside: 1.0, noiseAmt: 0.3, noiseScale: 0.2, height: 50 },
    };
    this.dust.e.bind(post, cam, P.dof);
    post.render(this.scene, cam, P, null, this.overlay);
  }
}
