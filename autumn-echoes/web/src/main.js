// Page host: one scene per page load.  The driver calls window.renderFrame(i) for
// global frame i (t = i / 30 s) and receives the finished frame as a PNG data URL.
import * as THREE from 'three';
import { Post } from './core/post.js';
import { makeLeafAtlas, makeClusterAtlas } from './core/leafatlas.js';

const SCENES = {
  forest: ['./scenes/forest.js', 'ForestScene'],
  hand: ['./scenes/hand.js', 'HandScene'],
  boots: ['./scenes/boots.js', 'BootsScene'],
  campfire: ['./scenes/campfire.js', 'CampfireScene'],
  aerial: ['./scenes/aerial.js', 'AerialScene'],
};

export const FPS = 30;

async function main() {
  const q = new URLSearchParams(location.search);
  const W = +(q.get('w') || 1080), H = +(q.get('h') || 1920);
  const name = q.get('scene') || 'forest';

  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  document.body.appendChild(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);
  // the post chain tone-maps and converts to sRGB itself
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const post = new Post(renderer, W, H);
  const { color, aux } = makeLeafAtlas(7);
  const colorTex = new THREE.CanvasTexture(color);
  colorTex.colorSpace = THREE.SRGBColorSpace;
  colorTex.anisotropy = 4;
  const auxTex = new THREE.CanvasTexture(aux);
  auxTex.colorSpace = THREE.NoColorSpace;
  const cl = makeClusterAtlas({ color, aux }, 19);
  const clColorTex = new THREE.CanvasTexture(cl.color);
  clColorTex.colorSpace = THREE.SRGBColorSpace;
  clColorTex.anisotropy = 4;
  const clAuxTex = new THREE.CanvasTexture(cl.aux);
  clAuxTex.colorSpace = THREE.NoColorSpace;

  const [path, cls] = SCENES[name];
  const mod = await import(path);
  const scene = new mod[cls]({ renderer, post, atlas: { colorTex, auxTex }, clusters: { colorTex: clColorTex, auxTex: clAuxTex }, W, H, query: q });
  const t0 = performance.now();
  await scene.init();
  console.log(`scene ${name} ready in ${(performance.now() - t0).toFixed(0)} ms`);

  const renderAt = (t) => {
    const a = performance.now();
    scene.render(t, post);
    const b = performance.now();
    const url = canvas.toDataURL('image/png');
    return { url, render: b - a, encode: performance.now() - b };
  };
  window.renderAt = renderAt;
  window.renderFrame = (i) => renderAt(i / FPS);
  window.SCENE = scene;
  window.READY = true;
}

main().catch((e) => {
  window.FAILED = String((e && e.stack) || e);
  console.error(window.FAILED);
});
