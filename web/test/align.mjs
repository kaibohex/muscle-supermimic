// Checks the JavaScript port against the native reference of `ms-export-web --reference` (same MuJoCo, C build):
//  1. open loop: on every recorded state, the observation per term and the policy's excitations;
//  2. closed loop: every clip from its first frame, the trajectory against the reference's.
//
//   node test/align.mjs <export dir>

import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';
import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web';
import { Tracker } from '../../docs/website/demo/tracker.mjs';

ort.env.wasm.numThreads = 1;
const dir = process.argv[2];
if (!dir) throw new Error('usage: node test/align.mjs <export dir>');
const read = (f) => (f.endsWith('.gz') ? gunzipSync(readFileSync(join(dir, f))) : readFileSync(join(dir, f)));
const buf = (f) => { const b = read(f); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const meta = JSON.parse(read('meta.json'));
if (!meta.reference) throw new Error('no reference: run ms-export-web with --reference');
const mj = await loadMujoco();
const tr = await Tracker.create(mj, ort, {
  meta, model: new Uint8Array(read(meta.files.model)), policy: new Uint8Array(read(meta.files.policy)), frames: buf(meta.files.frames),
  init: buf(meta.files.init),
});
const ref = new Float32Array(buf('reference.bin')), refState = new Float64Array(buf('reference_state.bin'));
const { nobs, nu, nq, nv, na } = meta;
const rw = nobs + nu, sw = nq + nv + na + nv;
const rows = [];
meta.reference.clips.forEach((c, ci) => { for (let t = 0; t < c.steps; t++) rows.push([ci, t]); });

// ---------------- 1. open loop ----------------
const terms = meta.obs_terms, worst = terms.map(() => [0, 0]); // [max abs error, max |reference|]
let eAbs = 0;
const stride = Math.max(1, Math.floor(rows.length / 400));
let t0 = performance.now(), n = 0;
for (let r = 0; r < rows.length; r += stride) {
  const [ci, t] = rows[r], s = refState.subarray(r * sw, (r + 1) * sw), o = ref.subarray(r * rw, r * rw + nobs);
  tr.reset(ci);
  tr.d.qpos.set(s.subarray(0, nq));
  tr.d.qvel.set(s.subarray(nq, nq + nv));
  tr.d.act.set(s.subarray(nq + nv, nq + nv + na));
  tr.d.qacc_warmstart.set(s.subarray(nq + nv + na));
  mj.mj_forward(tr.m, tr.d);
  tr.f = meta.clips[ci].start + t;
  if (t > 0) tr.action.set(ref.subarray((r - 1) * rw + nobs, r * rw));
  const e = await tr.policy(), obs = tr.obs;
  let k = 0;
  terms.forEach(([, dim], i) => {
    for (let j = 0; j < dim; j++, k++) {
      worst[i][0] = Math.max(worst[i][0], Math.abs(obs[k] - o[k]));
      worst[i][1] = Math.max(worst[i][1], Math.abs(o[k]));
    }
  });
  const re = ref.subarray(r * rw + nobs, (r + 1) * rw);
  for (let i = 0; i < nu; i++) eAbs = Math.max(eAbs, Math.abs(e[i] - re[i]));
  n++;
}
console.log(`open loop, ${n} of ${rows.length} recorded states (${((performance.now() - t0) / n).toFixed(2)} ms each):`);
console.log('  max |js - ref| (max |ref|) per term:');
terms.forEach(([name], i) => console.log(`    ${name.padEnd(18)} ${worst[i][0].toExponential(1)}  (${worst[i][1].toExponential(1)})`));
console.log(`  excitations: max |js - ref| ${eAbs.toExponential(1)}`);

// ---------------- 2. closed loop ----------------
console.log('closed loop, every clip from its first frame:');
let r0 = 0;
for (const [ci, c] of meta.reference.clips.entries()) {
  tr.reset(ci);
  let diverge = -1, maxRoot = 0, maxJoint = 0, failAt = -1, tStep = 0;
  const steps = meta.clips[ci].length - 1;
  for (let t = 0; t < steps; t++) {
    if (t < c.steps) {
      const s = refState.subarray((r0 + t) * sw, (r0 + t + 1) * sw);
      let dr = 0, dj = 0;
      for (let i = 0; i < 3; i++) dr = Math.max(dr, Math.abs(tr.d.qpos[meta.free_q[i]] - s[meta.free_q[i]]));
      for (const a of meta.joint_q_adr) dj = Math.max(dj, Math.abs(tr.d.qpos[a] - s[a]));
      maxRoot = Math.max(maxRoot, dr);
      maxJoint = Math.max(maxJoint, dj);
      if (diverge < 0 && dr > 0.01) diverge = t;
    }
    const ts = performance.now();
    await tr.step();
    tStep += performance.now() - ts;
    // the env's root-position termination against frame f (= the step's f + 1)
    const g = tr.referenceQpos();
    const dist = Math.hypot(...[0, 1, 2].map((i) => tr.d.qpos[meta.free_q[i]] - g[i]));
    if (dist > 1.0) { failAt = t + 1; break; }
  }
  r0 += c.steps;
  console.log(`  ${c.name}: ${failAt < 0 ? steps : failAt} steps vs reference ${c.steps}` +
    `${failAt < 0 ? '' : ', failed (root > 1 m off its reference frame)'}${c.failed ? ', reference failed' : ''}; ` +
    `max |dq| root ${maxRoot.toFixed(4)} m, joints ${maxJoint.toFixed(4)} rad; ` +
    `root > 1 cm off the reference from step ${diverge < 0 ? '-' : diverge}; ${(tStep / steps).toFixed(2)} ms/step`);
}
tr.delete();
