// Page: draws what the simulation worker posts with three.js (MuJoCo's z-up world, the studio look of ms-render)
// and wires the controls.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const dataUrl = new URL(params.get('data') || 'data/myofullbody/', location.href).href;

// ---------------- renderer, studio, camera ----------------
const renderer = new THREE.WebGLRenderer({ antialias: true, canvas: $('view') });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf5f5f7);
scene.fog = new THREE.Fog(0xf5f5f7, 6, 16);
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 100);
camera.position.set(3.2, -2.2, 1.5);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.9);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 1;
controls.maxDistance = 10;

scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d8dc, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: 0.5, far: 12 });
sun.shadow.bias = -0.0005;
sun.shadow.radius = 4;
scene.add(sun, sun.target);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0xfbfbfb, roughness: 1 }));
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(200, 200, 0xe2e2e6, 0xe9e9ec);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.0005;
scene.add(grid);

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.fov = w < h ? 58 : 40; // keep the whole body in a portrait view
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

// ---------------- body, ghost, muscles ----------------
let body = [], ghost = [], muscles, joints, ghostGroup;
const m4 = new THREE.Matrix4();

function buildScene(msg) {
  const geometries = {};
  for (const [k, mesh] of Object.entries(msg.meshes)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(mesh.vert, 3));
    g.setIndex(new THREE.BufferAttribute(mesh.face, 1));
    g.computeVertexNormals();
    geometries[k] = g;
  }
  const boneMat = new THREE.MeshStandardMaterial({ color: 0xf4f1ea, roughness: 0.55, metalness: 0 });
  const ghostMat = new THREE.MeshStandardMaterial({ color: 0x5a99ff, transparent: true, opacity: 0.28, depthWrite: false, roughness: 0.8 });
  ghostGroup = new THREE.Group();
  scene.add(ghostGroup);
  for (const g of msg.geoms) {
    if (g.mesh === undefined) continue;
    const b = new THREE.Mesh(geometries[g.mesh], boneMat);
    b.matrixAutoUpdate = false;
    b.userData.geom = g.id;
    b.castShadow = true;
    scene.add(b);
    body.push(b);
    const h = new THREE.Mesh(geometries[g.mesh], ghostMat);
    h.matrixAutoUpdate = false;
    ghostGroup.add(h);
    ghost.push(h);
  }
  const muscleMat = new THREE.MeshStandardMaterial({ roughness: 0.5 });
  muscles = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 10, 1, true), muscleMat, msg.maxSegments);
  joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 8), muscleMat, msg.maxSegments);
  for (const im of [muscles, joints]) {
    im.castShadow = true;
    im.frustumCulled = false;
    im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    im.setColorAt(0, new THREE.Color());
    scene.add(im);
  }
}

function placeGeoms(meshes, pose) {
  for (let i = 0; i < meshes.length; i++) {
    const p = pose.xpos, r = pose.xmat, a = 9 * i, b = 3 * i;
    meshes[i].matrix.set(r[a], r[a + 1], r[a + 2], p[b], r[a + 3], r[a + 4], r[a + 5], p[b + 1], r[a + 6], r[a + 7], r[a + 8], p[b + 2], 0, 0, 0, 1);
  }
}

const up = new THREE.Vector3(0, 1, 0), p0 = new THREE.Vector3(), p1 = new THREE.Vector3(), dir = new THREE.Vector3();
const quat = new THREE.Quaternion(), scale = new THREE.Vector3(), color = new THREE.Color(), one = new THREE.Quaternion();
function placeMuscles(seg) {
  const n = seg.length / 10;
  for (let i = 0; i < n; i++) {
    const s = 10 * i, r = seg[s + 9];
    p0.set(seg[s], seg[s + 1], seg[s + 2]);
    p1.set(seg[s + 3], seg[s + 4], seg[s + 5]);
    dir.subVectors(p1, p0);
    const len = dir.length();
    quat.setFromUnitVectors(up, len > 1e-9 ? dir.divideScalar(len) : up);
    m4.compose(p0.clone().add(p1).multiplyScalar(0.5), quat, scale.set(r, len, r));
    muscles.setMatrixAt(i, m4);
    m4.compose(p1, one, scale.set(r, r, r));
    joints.setMatrixAt(i, m4);
    color.setRGB(seg[s + 6], seg[s + 7], seg[s + 8], THREE.SRGBColorSpace);
    muscles.setColorAt(i, color);
    joints.setColorAt(i, color);
  }
  for (const im of [muscles, joints]) {
    im.count = n;
    im.instanceMatrix.needsUpdate = true;
    im.instanceColor.needsUpdate = true;
  }
}

// camera follows the root (smoothed), keeping the user's orbit offset
const look = new THREE.Vector3(0, 0, 0.9);
let follow = true, lookInit = false;
function followRoot(root) {
  const target = new THREE.Vector3(root[0], root[1], 0.9);
  if (!lookInit) { look.copy(target); lookInit = true; }
  look.lerp(target, 0.08);
  if (follow) {
    const delta = look.clone().sub(controls.target);
    controls.target.add(delta);
    camera.position.add(delta);
  }
  sun.position.set(look.x + 1.2, look.y + 1.8, 6);
  sun.target.position.set(look.x, look.y, 0);
}

// ---------------- worker and controls ----------------
const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
const ui = { playing: true, policy: true, ghost: true };
let clips = [];
const fmt = (frames) => `${(frames / 100).toFixed(1)} s`;

worker.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === 'progress') {
    const pct = Math.floor(100 * msg.frac);
    $('bootMsg').textContent = `${pct} % · ${msg.text}`;
    $('bootBar').firstChild.style.width = `${pct}%`;
  }
  else if (msg.type === 'error') {
    $('bootMsg').textContent = msg.text;
    $('boot').classList.add('error');
  } else if (msg.type === 'scene') {
    buildScene(msg);
    clips = msg.clips;
    const sel = $('clip');
    clips.forEach((c, i) => sel.add(new Option(c.name.replace(/_poses$/, ''), i)));
    $('boot').hidden = true;
    $('dock').hidden = false;
  } else if (msg.type === 'frame') {
    placeGeoms(body, msg.body);
    ghostGroup.visible = !!msg.ghost;
    if (msg.ghost) placeGeoms(ghost, msg.ghost);
    placeMuscles(msg.seg);
    followRoot(msg.root);
    drawGrab(msg.grab);
    $('progress').style.width = `${(100 * msg.frame) / Math.max(1, msg.length - 1)}%`;
    $('time').textContent = `${fmt(msg.frame)} / ${fmt(msg.length - 1)}`;
    $('cost').textContent = `${msg.stepMs.toFixed(1)} ms per 10 ms step`;
    $('fail').hidden = !msg.failed;
  }
};
worker.postMessage({ type: 'load', data: dataUrl });

const send = (state) => worker.postMessage({ type: 'set', state });
$('play').onclick = () => {
  ui.playing = !ui.playing;
  $('play').textContent = ui.playing ? 'Pause' : 'Play';
  send({ playing: ui.playing && visible });
};
$('restart').onclick = () => worker.postMessage({ type: 'restart' });
$('clip').onchange = (e) => {
  worker.postMessage({ type: 'clip', clip: Number(e.target.value) });
  lookInit = false;
};
for (const key of ['policy', 'ghost']) {
  $(key).onchange = (e) => { ui[key] = e.target.checked; send({ [key]: ui[key] }); };
}
$('follow').onchange = (e) => { follow = e.target.checked; };

// ---------------- the info card: open on a large screen until the first touch of the view ----------------
function collapseHead(collapsed) {
  $('head').classList.toggle('collapsed', collapsed);
  $('about').textContent = collapsed ? 'About' : 'Hide';
  $('about').setAttribute('aria-expanded', String(!collapsed));
}
$('about').onclick = () => collapseHead(!$('head').classList.contains('collapsed'));
collapseHead(matchMedia('(max-width: 900px), (max-height: 700px)').matches);
renderer.domElement.addEventListener('pointerdown', () => collapseHead(true));

// ---------------- embedded in the project page, full screen ----------------
// Windowed in the page, the wheel and vertical swipes scroll the page instead of zooming and orbiting; the page also
// pauses the simulation while the viewer is scrolled out of sight. Full screen gives the controls back.
const embed = params.has('embed'), root = document.documentElement;
const fsElement = () => document.fullscreenElement || document.webkitFullscreenElement;
const canFullscreen = document.fullscreenEnabled || document.webkitFullscreenEnabled;
let visible = true;
function setMode() {
  const fs = !!fsElement(), windowed = embed && !fs;
  root.classList.toggle('embed', embed);
  root.classList.toggle('fs', fs);
  controls.enableZoom = !windowed;
  renderer.domElement.style.touchAction = windowed ? 'pan-y' : 'none';
  $('full').textContent = fs ? 'Exit full screen' : 'Full screen';
}
$('full').hidden = !canFullscreen && !embed;
$('full').onclick = () => {
  if (fsElement()) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  else if (canFullscreen) (root.requestFullscreen || root.webkitRequestFullscreen).call(root);
  else { // no element full screen (iPhone): the demo's own page in a new tab
    const url = new URL(location.href);
    url.searchParams.delete('embed');
    open(url.href, '_blank', 'noopener');
  }
};
for (const ev of ['fullscreenchange', 'webkitfullscreenchange']) document.addEventListener(ev, setMode);
setMode();
addEventListener('message', (ev) => {
  if (ev.origin !== location.origin || ev.data?.type !== 'visible') return;
  visible = ev.data.visible;
  send({ playing: ui.playing && visible });
});

// ---------------- dragging the body ----------------
const ray = new THREE.Raycaster(), pointer = new THREE.Vector2(), dragPlane = new THREE.Plane(), hit = new THREE.Vector3();
const grabLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineBasicMaterial({ color: 0x1d1d1f, depthTest: false }));
grabLine.renderOrder = 10;
grabLine.frustumCulled = false; // its bounding sphere stays at the origin
grabLine.visible = false;
const grabDot = new THREE.Mesh(new THREE.SphereGeometry(0.02, 16, 12), new THREE.MeshBasicMaterial({ color: 0x1d1d1f, depthTest: false }));
grabDot.renderOrder = 10;
scene.add(grabLine, grabDot);
grabDot.visible = false;
let dragging = false;

function castFrom(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  pointer.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(pointer, camera);
}

/** The bone under the pointer, or else the one whose center passes closest to the ray (a skeleton is mostly gaps). */
function pickBone() {
  const hits = ray.intersectObjects(body, false);
  if (hits.length) return hits[0];
  let best = null, bestDist = 0.08;
  const c = new THREE.Vector3();
  for (const b of body) {
    if (!b.geometry.boundingSphere) b.geometry.computeBoundingSphere();
    c.copy(b.geometry.boundingSphere.center).applyMatrix4(b.matrixWorld);
    const dist = ray.ray.distanceToPoint(c);
    if (dist < bestDist) {
      bestDist = dist;
      best = { object: b, point: ray.ray.closestPointToPoint(c, new THREE.Vector3()) };
    }
  }
  return best;
}

renderer.domElement.addEventListener('pointerdown', (ev) => {
  if (!body.length || ev.button !== 0) return;
  castFrom(ev);
  for (const b of body) b.updateMatrixWorld(true);
  const pick = pickBone();
  if (!pick) return;
  dragging = true;
  controls.enabled = false;
  renderer.domElement.setPointerCapture(ev.pointerId);
  dragPlane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()).negate(), pick.point);
  worker.postMessage({ type: 'grab', geom: pick.object.userData.geom, point: pick.point.toArray() });
  $('hint').hidden = true;
});
renderer.domElement.addEventListener('pointermove', (ev) => {
  if (!dragging) return;
  castFrom(ev);
  if (ray.ray.intersectPlane(dragPlane, hit)) worker.postMessage({ type: 'drag', target: hit.toArray() });
});
const endDrag = () => {
  if (!dragging) return;
  dragging = false;
  controls.enabled = true;
  worker.postMessage({ type: 'release' });
};
renderer.domElement.addEventListener('pointerup', endDrag);
renderer.domElement.addEventListener('pointercancel', endDrag);

function drawGrab(g) {
  grabLine.visible = grabDot.visible = !!g;
  if (!g) return;
  grabDot.position.set(g[3], g[4], g[5]);
  grabLine.geometry.attributes.position.array.set(g);
  grabLine.geometry.attributes.position.needsUpdate = true;
}

renderer.setAnimationLoop(() => {
  if (!visible) return;
  controls.update();
  renderer.render(scene, camera);
});
