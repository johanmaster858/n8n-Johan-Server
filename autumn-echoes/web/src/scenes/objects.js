// The heavy, cold possessions resting on the leaves: safe, books, briefcase, padlock.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import * as TX from '../core/textures.js';

function std(maps, extra = {}) {
  return new THREE.MeshStandardMaterial({
    map: maps.map, roughnessMap: maps.roughnessMap, metalnessMap: maps.metalnessMap, normalMap: maps.normalMap,
    roughness: 1, metalness: 1, ...extra,
  });
}

function mesh(geo, mat) {
  const m = new THREE.Mesh(geo, mat);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

function dialTexture() {
  const c = TX.canvas(512, 512);
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(256, 256, 20, 256, 256, 256);
  grd.addColorStop(0, '#2a2a28');
  grd.addColorStop(1, '#121211');
  g.fillStyle = grd;
  g.fillRect(0, 0, 512, 512);
  g.translate(256, 256);
  g.fillStyle = '#e8e0cc';
  g.strokeStyle = '#e8e0cc';
  g.font = 'bold 34px Georgia, serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  for (let i = 0; i < 100; i++) {
    const a = (i / 100) * Math.PI * 2 - Math.PI / 2;
    const long = i % 10 === 0, mid = i % 5 === 0;
    g.lineWidth = long ? 5 : 3;
    g.beginPath();
    g.moveTo(Math.cos(a) * 240, Math.sin(a) * 240);
    g.lineTo(Math.cos(a) * (long ? 200 : mid ? 214 : 224), Math.sin(a) * (long ? 200 : mid ? 214 : 224));
    g.stroke();
    if (long) g.fillText(String(i), Math.cos(a) * 172, Math.sin(a) * 172);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function plateTexture(text, sub) {
  const c = TX.canvas(512, 128);
  const g = c.getContext('2d');
  const grd = g.createLinearGradient(0, 0, 0, 128);
  grd.addColorStop(0, '#caa45a');
  grd.addColorStop(0.5, '#a07a36');
  grd.addColorStop(1, '#7a5a26');
  g.fillStyle = grd;
  g.fillRect(0, 0, 512, 128);
  g.strokeStyle = '#4a3212';
  g.lineWidth = 6;
  g.strokeRect(10, 10, 492, 108);
  g.fillStyle = '#3e2a10';
  g.font = 'bold 44px Georgia, serif';
  g.textAlign = 'center';
  g.fillText(text, 256, 66);
  g.font = '24px Georgia, serif';
  g.fillText(sub, 256, 100);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function buildObjects() {
  const objects = {};

  // ------------------------------------------------------------------ safe
  {
    const g = new THREE.Group();
    const paint = TX.safePaint(1024, { seed: 3 });
    const doorPaint = TX.safePaint(1024, { door: true, seed: 4 });
    const steel = TX.metal([0.55, 0.56, 0.58], 51, { rough: 0.3, patina: 0.05 });
    const brass = TX.metal([0.78, 0.58, 0.26], 52, { rough: 0.35, patina: 0.35 });
    const mBody = std(paint);
    const mDoor = std(doorPaint);
    const mSteel = std(steel);
    const mBrass = std(brass);
    const W = 0.46, H = 0.54, D = 0.44;
    g.add(mesh(new RoundedBoxGeometry(W, H, D, 4, 0.03), mBody));
    const door = mesh(new RoundedBoxGeometry(0.36, 0.43, 0.024, 3, 0.008), mDoor);
    door.position.set(0.01, -0.01, D / 2 + 0.006);
    g.add(door);
    const dialMat = new THREE.MeshStandardMaterial({ map: dialTexture(), roughness: 0.35, metalness: 0.2 });
    const dial = mesh(new THREE.CylinderGeometry(0.052, 0.056, 0.022, 64), [mSteel, dialMat, mSteel]);
    dial.rotation.x = Math.PI / 2;
    dial.position.set(0.01, 0.07, D / 2 + 0.028);
    g.add(dial);
    const knob = mesh(new THREE.CylinderGeometry(0.016, 0.02, 0.03, 32), mSteel);
    knob.rotation.x = Math.PI / 2;
    knob.position.set(0.01, 0.07, D / 2 + 0.045);
    g.add(knob);
    const hub = mesh(new THREE.CylinderGeometry(0.014, 0.016, 0.03, 24), mSteel);
    hub.rotation.x = Math.PI / 2;
    hub.position.set(0.01, -0.075, D / 2 + 0.03);
    g.add(hub);
    const bar = mesh(new THREE.CapsuleGeometry(0.008, 0.1, 6, 12), mSteel);
    bar.rotation.z = Math.PI / 2;
    bar.position.set(0.01, -0.075, D / 2 + 0.046);
    g.add(bar);
    for (const y of [-0.14, 0.14]) {
      const h = mesh(new THREE.CylinderGeometry(0.013, 0.013, 0.07, 20), mSteel);
      h.position.set(-0.172, y, D / 2 + 0.012);
      g.add(h);
    }
    const plateMat = new THREE.MeshStandardMaterial({ map: plateTexture('HAVEN & SONS', 'FIRE PROOF'), roughness: 0.35, metalness: 0.9 });
    const plate = mesh(new THREE.BoxGeometry(0.15, 0.036, 0.004), [mBrass, mBrass, mBrass, mBrass, plateMat, mBrass]);
    plate.position.set(0.01, 0.165, D / 2 + 0.02);
    g.add(plate);
    const esc = mesh(new THREE.CylinderGeometry(0.013, 0.013, 0.006, 24), mBrass);
    esc.rotation.x = Math.PI / 2;
    esc.position.set(0.01, -0.15, D / 2 + 0.02);
    g.add(esc);
    for (const [x, z] of [[-0.19, -0.18], [0.19, -0.18], [-0.19, 0.18], [0.19, 0.18]]) {
      const f = mesh(new THREE.BoxGeometry(0.06, 0.03, 0.06), mSteel);
      f.position.set(x, -H / 2 - 0.012, z);
      g.add(f);
    }
    g.position.set(0.0, H / 2 + 0.015, -0.15);
    g.rotation.y = 0.28;
    objects.safe = { group: g, size: 0.62 };
  }

  // ----------------------------------------------------------------- books
  {
    const g = new THREE.Group();
    const specs = [
      { w: 0.27, h: 0.058, d: 0.2, col: [0.34, 0.07, 0.06], rot: 0.05 },
      { w: 0.25, h: 0.05, d: 0.185, col: [0.08, 0.14, 0.24], rot: -0.12 },
      { w: 0.28, h: 0.064, d: 0.205, col: [0.12, 0.2, 0.12], rot: 0.18 },
      { w: 0.23, h: 0.046, d: 0.17, col: [0.3, 0.18, 0.09], rot: -0.06 },
      { w: 0.215, h: 0.04, d: 0.16, col: [0.36, 0.09, 0.07], rot: 0.26 },
    ];
    const pages = TX.pageEdges(256, 512, 17);
    const mPages = std(pages, { metalness: 0 });
    let y = 0;
    specs.forEach((b, i) => {
      const cover = TX.bookCover(b.col, 60 + i);
      const spineT = TX.bookCover(b.col, 70 + i, { spine: true });
      const mCover = std(cover);
      const mSpine = std(spineT);
      const book = new THREE.Group();
      const ct = 0.0065;
      const top = mesh(new RoundedBoxGeometry(b.w, ct, b.d, 2, 0.002), mCover);
      top.position.y = b.h / 2 - ct / 2;
      const bot = mesh(new RoundedBoxGeometry(b.w, ct, b.d, 2, 0.002), mCover);
      bot.position.y = -b.h / 2 + ct / 2;
      const spine = mesh(new RoundedBoxGeometry(0.012, b.h, b.d, 3, 0.005), [mSpine, mSpine, mSpine, mSpine, mSpine, mSpine]);
      spine.position.x = -b.w / 2 + 0.004;
      const block = mesh(new THREE.BoxGeometry(b.w - 0.012, b.h - 2 * ct, b.d - 0.008), mPages);
      block.position.x = 0.002;
      book.add(top, bot, spine, block);
      book.position.y = y + b.h / 2;
      book.position.x = (i % 2 ? 0.012 : -0.01) * i;
      book.rotation.y = b.rot;
      y += b.h;
      g.add(book);
    });
    g.position.set(0.34, 0.012, 0.12);
    g.rotation.y = -0.4;
    objects.books = { group: g, size: 0.42 };
  }

  // ------------------------------------------------------------- briefcase
  {
    const outer = new THREE.Group();
    const g = new THREE.Group();
    const lea = TX.leather(1024, [0.42, 0.22, 0.11], 11);
    const brass = TX.metal([0.8, 0.6, 0.28], 53, { rough: 0.3, patina: 0.2 });
    const mLeather = std(lea, { metalness: 0 });
    const mBrass = std(brass);
    const W = 0.46, H = 0.1, D = 0.34;
    g.add(mesh(new RoundedBoxGeometry(W, H, D, 4, 0.022), mLeather));
    // lid seam: a slightly darker band around the case
    const seam = mesh(new RoundedBoxGeometry(W + 0.002, 0.006, D + 0.002, 2, 0.003),
      new THREE.MeshStandardMaterial({ color: 0x1a0c06, roughness: 0.7 }));
    seam.position.y = 0.015;
    g.add(seam);
    const handle = mesh(new THREE.TorusGeometry(0.052, 0.011, 14, 32, Math.PI), mLeather);
    handle.rotation.x = -Math.PI / 2 + 0.25;
    handle.rotation.z = Math.PI;
    handle.position.set(0, 0.018, D / 2 + 0.004);
    g.add(handle);
    for (const x of [-0.052, 0.052]) {
      const mnt = mesh(new RoundedBoxGeometry(0.022, 0.02, 0.018, 2, 0.004), mBrass);
      mnt.position.set(x, 0.018, D / 2 + 0.004);
      g.add(mnt);
    }
    for (const x of [-0.15, 0.15]) {
      const latch = mesh(new RoundedBoxGeometry(0.045, 0.032, 0.012, 2, 0.004), mBrass);
      latch.position.set(x, 0.02, D / 2 + 0.004);
      g.add(latch);
    }
    for (const [x, z] of [[-W / 2, -D / 2], [W / 2, -D / 2], [-W / 2, D / 2], [W / 2, D / 2]]) {
      const cap = mesh(new THREE.SphereGeometry(0.018, 16, 12), mBrass);
      cap.scale.set(1, 2.2, 1);
      cap.position.set(x * 0.985, 0, z * 0.985);
      g.add(cap);
    }
    // stand it upright on its long edge, handle on top
    g.rotation.x = -Math.PI / 2;
    outer.add(g);
    outer.position.set(-0.42, D / 2 + 0.008, 0.12);
    outer.rotation.y = 0.62;
    objects.briefcase = { group: outer, size: 0.5 };
  }

  // --------------------------------------------------------------- padlock
  {
    const g = new THREE.Group();
    const brass = TX.metal([0.82, 0.62, 0.28], 54, { rough: 0.28, patina: 0.3, size: 256 });
    const steel = TX.metal([0.62, 0.62, 0.64], 55, { rough: 0.22, patina: 0.02, size: 256 });
    const mBrass = std(brass);
    const mSteel = std(steel);
    const body = mesh(new RoundedBoxGeometry(0.064, 0.072, 0.028, 3, 0.009), mBrass);
    g.add(body);
    const shackle = mesh(new THREE.TorusGeometry(0.021, 0.0058, 12, 28, Math.PI), mSteel);
    shackle.position.y = 0.036 + 0.018;
    g.add(shackle);
    for (const x of [-0.021, 0.021]) {
      const leg = mesh(new THREE.CylinderGeometry(0.0058, 0.0058, 0.02, 12), mSteel);
      leg.position.set(x, 0.036 + 0.008, 0);
      g.add(leg);
    }
    const hole = mesh(new THREE.CylinderGeometry(0.0045, 0.0045, 0.002, 16), new THREE.MeshStandardMaterial({ color: 0x050403, roughness: 0.9 }));
    hole.rotation.x = Math.PI / 2;
    hole.position.set(0, -0.012, 0.0142);
    g.add(hole);
    // a hefty old padlock standing on top of the book stack
    g.scale.setScalar(1.3);
    g.position.set(0.315, 0.27 + 0.036 * 1.3 + 0.001, 0.125);
    g.rotation.set(0, -0.25, 0.04);
    objects.padlock = { group: g, size: 0.16 };
  }
  return objects;
}
