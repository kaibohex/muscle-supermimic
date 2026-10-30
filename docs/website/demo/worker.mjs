// Simulation worker: loads an exported checkpoint, runs the Tracker's closed loop at real time, and posts the
// poses the page draws (bone meshes, muscle paths colored by activation, the reference as a ghost). It switches
// checkpoints (models) and clips on request, between control steps; a clip's frames download when it is chosen.

import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs';
import { Tracker, quatToMat } from './tracker.mjs';

const MUJOCO = 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/', ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads = 1; // a static host cannot send the cross-origin isolation headers threads need
ort.env.wasm.wasmPaths = ORT; // its JavaScript glue; the binary is passed in

const DRAW_GROUP = 0; // the bone meshes; collision geoms sit in groups 3 and 4
const ACT_REST = [0.82, 0.79, 0.78], ACT_RED = [0.72, 0.03, 0.05]; // ms-render's activation colors
const FAIL_DIST = 1.0; // the task's root-position termination [m]

let tr, mj, dg, geoms, tendons, muscleOf, dataUrl;
const state = { playing: true, policy: true, ghost: true, clip: 0, failed: false, failedAt: 0 };
let simT = 0, wallT0 = 0, stepMs = 0, episode = 0; // episode: counts resets, so the page can cut its camera
let facing = [1, 0]; // the body's forward direction (horizontal) at the reset
let pendingModel = null, pendingClip = null; // requests the loop applies between steps
const clipFrames = new Map(); // clip file URL -> its frames, inflated

const post = (msg, transfer = []) => self.postMessage(msg, transfer);

// The two WebAssembly binaries, fetched here to count them in the progress: [url, size, size sent (brotli)]
const WASM = {
  mujoco: [`${MUJOCO}mujoco.wasm`, 10117577, 2099961],
  ort: [`${ORT}ort-wasm-simd-threaded.wasm`, 11246032, 2493249],
};
const DOWNLOAD = 0.9; // the share of the progress bar for the downloads; the rest is starting MuJoCo and the policy

/** Fetches files {key: [url, size, size sent]}, reporting the progress; each counts by its size on the wire, the
 * binaries' decoded bytes scaled down to it. Gzipped files come back inflated. */
async function download(files) {
  const sent = Object.values(files).reduce((a, f) => a + f[2], 0), got = {};
  const report = () => {
    const mb = Object.entries(got).reduce((a, [k, n]) => a + Math.min(1, n / files[k][1]) * files[k][2], 0);
    post({ type: 'progress', frac: DOWNLOAD * (mb / sent), text: `downloading ${(mb / 1e6).toFixed(0)} / ${(sent / 1e6).toFixed(0)} MB` });
  };
  const keys = Object.keys(files);
  const bufs = await Promise.all(keys.map((k) => fetchFile(files[k][0], (n) => { got[k] = (got[k] || 0) + n; report(); })));
  return Object.fromEntries(bufs.map((b, i) => [keys[i], b]));
}

async function fetchFile(url, onBytes = () => {}) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.split('/').pop()}: HTTP ${r.status}`);
  const reader = r.body.getReader(), chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    onBytes(value.byteLength);
  }
  const blob = new Blob(chunks), head = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
  // still gzip unless the host sent it with Content-Encoding: gzip and the browser already inflated it
  if (head[0] !== 0x1f || head[1] !== 0x8b) return blob.arrayBuffer();
  return new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

/** Loads the checkpoint exported at url (the libraries too, the first time), at the clip of that name if it has one. */
async function loadModel(url, clipName) {
  if (tr) {
    tr.delete();
    dg.delete();
    tr = null;
  }
  dataUrl = url;
  post({ type: 'progress', frac: 0, text: 'downloading' });
  const meta = await (await fetch(new URL('meta.json', url))).json();
  const clip = Math.max(0, meta.clips.findIndex((c) => c.name === clipName));
  const files = Object.fromEntries(['model', 'policy', 'init'].map((k) => [k, [new URL(meta.files[k], url).href, meta.sizes[k], meta.sizes[k]]]));
  const clipUrl = new URL(meta.clips[clip].file, url).href;
  files.clip = [clipUrl, meta.clips[clip].size, meta.clips[clip].size];
  if (!mj) Object.assign(files, WASM);
  const bufs = await download(files);
  if (!mj) {
    post({ type: 'progress', frac: DOWNLOAD, text: 'starting MuJoCo' });
    mj = await loadMujoco({ wasmBinary: bufs.mujoco });
    ort.env.wasm.wasmBinary = bufs.ort;
  }
  post({ type: 'progress', frac: DOWNLOAD + 0.04, text: 'loading the model and the policy' });
  tr = await Tracker.create(mj, ort, { meta, model: new Uint8Array(bufs.model), policy: new Uint8Array(bufs.policy), init: bufs.init });
  dg = new mj.MjData(tr.m);
  clipFrames.set(clipUrl, bufs.clip);
  const { transfer, ...scene } = sceneDescription();
  post({ type: 'scene', ...scene, nmuscles: meta.nmuscles, clip, clips: meta.clips.map((c) => ({ name: c.name, length: c.length })) }, transfer);
  reset(clip, bufs.clip);
}

/** Switches to the clip of that name, downloading its frames the first time. */
async function loadClip(name) {
  const clip = tr.meta.clips.findIndex((c) => c.name === name);
  if (clip < 0) return;
  const url = new URL(tr.meta.clips[clip].file, dataUrl).href;
  if (!clipFrames.has(url)) clipFrames.set(url, await fetchFile(url));
  reset(clip, clipFrames.get(url));
}

/** Static geometry: the drawn geoms with their meshes and colors, and the tendons. */
function sceneDescription() {
  const m = tr.m, meshes = {}, transfer = [];
  geoms = [];
  for (let g = 0; g < m.ngeom; g++) {
    if (m.geom_group[g] !== DRAW_GROUP || m.geom_type[g] === 0) continue; // the plane is the page's own floor
    const mat = m.geom_matid[g], rgba = mat >= 0 ? m.mat_rgba.slice(4 * mat, 4 * mat + 4) : m.geom_rgba.slice(4 * g, 4 * g + 4);
    const desc = { id: g, body: m.geom_bodyid[g], type: m.geom_type[g], size: Array.from(m.geom_size.slice(3 * g, 3 * g + 3)), rgba: Array.from(rgba) };
    if (desc.type === 7) {
      const k = m.geom_dataid[g];
      desc.mesh = k;
      if (!meshes[k]) {
        const va = m.mesh_vertadr[k], vn = m.mesh_vertnum[k], fa = m.mesh_faceadr[k], fn = m.mesh_facenum[k];
        meshes[k] = { vert: Float32Array.from(m.mesh_vert.subarray(3 * va, 3 * (va + vn))), face: Uint32Array.from(m.mesh_face.subarray(3 * fa, 3 * (fa + fn))) };
        transfer.push(meshes[k].vert.buffer, meshes[k].face.buffer);
      }
    }
    geoms.push(desc);
  }
  // muscles: the tendon each actuator pulls (actuator order = the policy's excitation order)
  muscleOf = new Int32Array(m.ntendon).fill(-1);
  tr.meta.ctrl_ids.forEach((a, i) => { if (m.actuator_trntype[a] === 3) muscleOf[m.actuator_trnid[2 * a]] = i; });
  tendons = [];
  for (let t = 0; t < m.ntendon; t++) {
    if (m.tendon_group[t] > 2) continue;
    tendons.push({ id: t, width: m.tendon_width[t], rgba: Array.from(m.tendon_rgba.slice(4 * t, 4 * t + 4)), muscle: muscleOf[t] });
  }
  return { geoms, meshes, maxSegments: 2 * m.nwrap, transfer };
}

function reset(clip, frames) {
  tr.reset(clip, frames);
  episode++;
  // the task's root frame (the free joint's rotation times R_off) has x forward, for both bodies
  const q = tr.d.qpos, fq = tr.meta.free_q, R = new Float64Array(9), o = tr.meta.R_off;
  quatToMat(q[fq[3]], q[fq[4]], q[fq[5]], q[fq[6]], R);
  facing = [R[0] * o[0] + R[1] * o[3] + R[2] * o[6], R[3] * o[0] + R[4] * o[3] + R[5] * o[6]];
  state.clip = clip;
  state.failed = false;
  simT = 0;
  wallT0 = performance.now();
}

function poses(d) {
  const n = geoms.length, xpos = new Float32Array(3 * n), xmat = new Float32Array(9 * n);
  geoms.forEach((g, i) => {
    xpos.set(d.geom_xpos.subarray(3 * g.id, 3 * g.id + 3), 3 * i);
    xmat.set(d.geom_xmat.subarray(9 * g.id, 9 * g.id + 9), 9 * i);
  });
  return { xpos, xmat };
}

/** Muscle path segments: [x0 y0 z0 x1 y1 z1 r g b radius] each (mjv's tendon segments, pulleys excluded). */
function segments(d) {
  const out = [];
  for (const t of tendons) {
    const adr = d.ten_wrapadr[t.id], num = d.ten_wrapnum[t.id];
    let rgb = t.rgba;
    if (t.muscle >= 0) {
      const w = Math.min(1, Math.max(0, d.act[tr.meta.act_adr[t.muscle]]));
      rgb = [0, 1, 2].map((k) => ACT_REST[k] + w * (ACT_RED[k] - ACT_REST[k]));
    }
    for (let j = adr; j < adr + num - 1; j++) {
      if (d.wrap_obj[j] === -2 || d.wrap_obj[j + 1] === -2) continue;
      out.push(d.wrap_xpos[3 * j], d.wrap_xpos[3 * j + 1], d.wrap_xpos[3 * j + 2],
        d.wrap_xpos[3 * j + 3], d.wrap_xpos[3 * j + 4], d.wrap_xpos[3 * j + 5], rgb[0], rgb[1], rgb[2], t.width);
    }
  }
  return Float32Array.from(out);
}

function ghostPoses() {
  const meta = tr.meta, ref = tr.referenceQpos();
  dg.qpos.set(tr.d.qpos);
  meta.free_q.forEach((a, i) => { dg.qpos[a] = ref[i]; });
  meta.joint_q_adr.forEach((a, j) => { dg.qpos[a] = ref[7 + j]; });
  mj.mj_kinematics(tr.m, dg);
  return poses(dg);
}

async function tick() {
  try {
    if (pendingModel) {
      const url = pendingModel, name = pendingClip || (tr && tr.meta.clips[state.clip].name);
      pendingModel = pendingClip = null;
      await loadModel(url, name);
    } else if (pendingClip && tr) {
      const name = pendingClip;
      pendingClip = null;
      await loadClip(name);
    }
  } catch (e) {
    post({ type: 'error', text: String((e && e.message) || e) });
  }
  if (tr) await step();
  setTimeout(tick, 4);
}

async function step() {
  const dt = tr.meta.timestep * tr.meta.decimation;
  if (state.playing) {
    const target = (performance.now() - wallT0) / 1000;
    if (target - simT > 0.25) wallT0 += (target - simT - 0.25) * 1000; // fall behind gracefully (slow motion)
    let n = 0;
    while (simT < (performance.now() - wallT0) / 1000 && n < 8) {
      if (tr.f >= tr.clipEnd) { reset(state.clip); break; } // loop the clip
      if (state.failed) {
        if (performance.now() - state.failedAt > 1500) reset(state.clip);
        break;
      }
      const t0 = performance.now();
      if (state.policy) await tr.step();
      else tr.apply(new Float32Array(tr.meta.nu));
      stepMs = 0.9 * stepMs + 0.1 * (performance.now() - t0);
      simT += dt;
      n++;
      const ref = tr.referenceQpos(), q = tr.d.qpos, fq = tr.meta.free_q;
      if (Math.hypot(q[fq[0]] - ref[0], q[fq[1]] - ref[1], q[fq[2]] - ref[2]) > FAIL_DIST) {
        state.failed = true;
        state.failedAt = performance.now();
      }
    }
  }
  const body = poses(tr.d), seg = segments(tr.d), ghost = state.ghost ? ghostPoses() : null;
  const c = tr.meta.clips[state.clip];
  const root = tr.meta.free_q.slice(0, 3).map((a) => tr.d.qpos[a]);
  post({ type: 'frame', body, ghost, seg, root, grab: tr.grabbed ? [...tr.grabPoint(), ...tr.grabbed.target] : null, frame: tr.f, episode, clip: state.clip, facing, length: c.length, stepMs, failed: state.failed, playing: state.playing },
    [body.xpos.buffer, body.xmat.buffer, seg.buffer, ...(ghost ? [ghost.xpos.buffer, ghost.xmat.buffer] : [])]);
}

self.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === 'model') { // a checkpoint's data URL, and optionally the clip to start at
    pendingModel = msg.data;
    if (msg.clip) pendingClip = msg.clip;
  } else if (msg.type === 'clip') pendingClip = msg.name;
  else if (msg.type === 'set') {
    Object.assign(state, msg.state);
    if ('playing' in msg.state && msg.state.playing) wallT0 = performance.now() - simT * 1000;
  } else if (!tr) return;
  else if (msg.type === 'restart') reset(state.clip);
  else if (msg.type === 'grab') tr.grab(tr.m.geom_bodyid[msg.geom], msg.point);
  else if (msg.type === 'drag') tr.drag(msg.target);
  else if (msg.type === 'release') tr.release();
};

tick();
