// The demo's closed loop, for the browser and for Node: MuJoCo's WebAssembly build steps the task's scene model,
// the policy's decoder (onnxruntime-web) maps the observation and the encoder latent of the current reference
// frame to muscle excitations.
//
// The observation is the task's (src/muscle_supermimic/tasks/tracking/mdp/heads.py), recomputed from MuJoCo
// quantities. The task evaluates kinematics and muscle geometry on an idealized state, whose coupled joints sit
// exactly on their <equality> polynomials of the independent joints; here a second MjData holds that state and
// MuJoCo itself evaluates it. Everything else (rotations, root-relative site quantities) is ported line by line.
//
// Files are those of `ms-export-web` (meta.json and the files it names), passed in inflated.

const mjOBJ_BODY = 1;

export class Tracker {
  /**
   * @param mj the loaded @mujoco/mujoco module
   * @param ort onnxruntime-web
   * @param files {meta, model: Uint8Array, policy: Uint8Array, frames: ArrayBuffer, init: ArrayBuffer}
   */
  static async create(mj, ort, files) {
    // MjVFS.addBuffer copies element by element (tens of seconds for a model); Emscripten's FS takes the bytes at once
    mj.FS.writeFile('/model.mjb', files.model);
    const vfs = new mj.MjVFS();
    const m = mj.MjModel.from_binary_path('/model.mjb', vfs);
    vfs.delete();
    mj.FS.unlink('/model.mjb');
    const session = await ort.InferenceSession.create(files.policy, { executionProviders: ['wasm'] });
    return new Tracker(mj, ort, m, session, files);
  }

  constructor(mj, ort, m, session, files) {
    const meta = files.meta;
    Object.assign(this, { mj, ort, m, session, meta });
    this.d = new mj.MjData(m); // the simulated state
    this.di = new mj.MjData(m); // the idealized state the observation is evaluated on
    this.frames = new Float32Array(files.frames);
    this.init = new Float64Array(files.init);
    this.nIndep = meta.q_adr.length;
    this.ncoupled = meta.coupled.src.length;
    this.full = new Float64Array(this.nIndep + this.ncoupled);
    this.fulld = new Float64Array(this.nIndep + this.ncoupled);
    this.obs = new Float32Array(meta.nobs);
    this.action = new Float32Array(meta.nu);
    this.offsets = {};
    let o = 0;
    for (const [name, dim] of meta.obs_terms) {
      this.offsets[name] = o;
      o += dim;
    }
    if (o !== meta.nobs) throw new Error(`observation terms add up to ${o}, not ${meta.nobs}`);
    this.clip = 0;
    this.f = 0;
    this.grabbed = null; // {body, local: the point in the body frame, target: world} while the user pulls the body
  }

  /** The reference frame's root + joint qpos (for a ghost or the camera). */
  referenceQpos(f = this.f) {
    const w = this.meta.frame_width, nz = this.meta.nz;
    return this.frames.subarray(f * w + nz, (f + 1) * w);
  }

  /** Last frame of the current clip: the episode ends when the step would pass it. */
  get clipEnd() {
    const c = this.meta.clips[this.clip];
    return c.start + c.length - 1;
  }

  /** The env's reset at a clip's first frame: the reference state, zero activation, zero previous action. */
  reset(clip) {
    const { mj, m, d, meta } = this;
    mj.mj_resetData(m, d); // clears xfrc_applied; a grab in progress keeps pulling from the next step
    const w = meta.nq + meta.nv, row = this.init.subarray(clip * w, (clip + 1) * w);
    d.qpos.set(row.subarray(0, meta.nq));
    d.qvel.set(row.subarray(meta.nq));
    for (const a of meta.act_adr) d.act[a] = meta.init_activation;
    mj.mj_forward(m, d);
    this.action.fill(0);
    this.clip = clip;
    this.f = meta.clips[clip].start;
  }

  /** The decoder observation of the current state (heads.py's terms, in the env's order). */
  observe() {
    const { mj, m, d, di, meta, obs } = this;
    const qpos = d.qpos, qvel = d.qvel;
    // independent coordinates, then the coupled joints on their polynomials (SkeletonKinematics.coupled)
    const nq = this.nIndep, cp = meta.coupled;
    for (let i = 0; i < nq; i++) {
      this.full[i] = qpos[meta.q_adr[i]];
      this.fulld[i] = qvel[meta.v_adr[i]];
    }
    for (let k = 0; k < this.ncoupled; k++) {
      const s = cp.src[k], c = cp.coef[k], x = this.full[s] - cp.src_q0[k];
      this.full[nq + k] = cp.q0[k] + c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])));
      this.fulld[nq + k] = (c[1] + x * (2 * c[2] + x * (3 * c[3] + x * 4 * c[4]))) * this.fulld[s];
    }
    di.qpos.set(qpos);
    di.qvel.set(qvel);
    di.act.set(d.act);
    const nj = meta.full_index.length;
    for (let j = 0; j < nj; j++) {
      di.qpos[meta.joint_q_adr[j]] = this.full[meta.full_index[j]];
      di.qvel[meta.joint_v_adr[j]] = this.fulld[meta.full_index[j]];
    }
    // kinematics, tendons and actuator lengths, velocities (cvel, ten_velocity, actuator_velocity), muscle forces
    mj.mj_kinematics(m, di);
    mj.mj_comPos(m, di);
    mj.mj_tendon(m, di);
    mj.mj_transmission(m, di);
    mj.mj_fwdVelocity(m, di);
    mj.mj_fwdActuation(m, di);

    const off = this.offsets;
    // root_pos: height and the free joint's quaternion (w >= 0); root_vel: [v world, w body]
    const fq = meta.free_q, fv = meta.free_v, Rb = new Float64Array(9);
    quatToMat(qpos[fq[3]], qpos[fq[4]], qpos[fq[5]], qpos[fq[6]], Rb);
    obs[off.root_pos] = qpos[fq[2]];
    matToQuat(Rb, 0, obs, off.root_pos + 1);
    for (let i = 0; i < 6; i++) obs[off.root_vel + i] = qvel[fv[i]];
    for (let j = 0; j < nj; j++) {
      obs[off.joint_pos + j] = this.full[meta.full_index[j]];
      obs[off.joint_vel + j] = this.fulld[meta.full_index[j]];
    }
    // muscles, in actuator order
    const nu = meta.nu;
    for (let i = 0; i < nu; i++) {
      const a = meta.ctrl_ids[i];
      obs[off.muscle_length + i] = di.actuator_length[a];
      obs[off.muscle_velocity + i] = di.actuator_velocity[a];
      obs[off.muscle_force + i] = di.actuator_force[a];
      obs[off.muscle_excitation + i] = this.action[i];
      obs[off.muscle_activation + i] = d.act[meta.act_adr[i]];
    }
    for (let i = 0; i < meta.touch_adr.length; i++) obs[off.touch + i] = Math.abs(d.sensordata[meta.touch_adr[i]]);
    this.sites(off);
    return obs;
  }

  /** heads.rel_sites on the tracked bodies of the idealized state (the first one is the pelvis). */
  sites(off) {
    const { m, di, meta, obs } = this;
    const ids = meta.tracked_body_ids, xpos = di.xpos, xmat = di.xmat;
    const vel = (b, out) => {
      // mj_objectVelocity(BODY, world frame): cvel is [ang, lin] at the subtree com of the body's root
      const r = m.body_rootid[b], c = di.cvel;
      const dx = xpos[3 * b] - di.subtree_com[3 * r], dy = xpos[3 * b + 1] - di.subtree_com[3 * r + 1], dz = xpos[3 * b + 2] - di.subtree_com[3 * r + 2];
      const wx = c[6 * b], wy = c[6 * b + 1], wz = c[6 * b + 2];
      out[0] = wx; out[1] = wy; out[2] = wz;
      out[3] = c[6 * b + 3] + wy * dz - wz * dy;
      out[4] = c[6 * b + 4] + wz * dx - wx * dz;
      out[5] = c[6 * b + 5] + wx * dy - wy * dx;
    };
    const b0 = ids[0], R0 = xmat.subarray(9 * b0, 9 * b0 + 9), v0 = new Float64Array(6), vi = new Float64Array(6);
    const rrot = new Float64Array(9), q = new Float64Array(4);
    vel(b0, v0);
    for (let k = 1; k < ids.length; k++) {
      const b = ids[k], Ri = xmat.subarray(9 * b, 9 * b + 9), j = k - 1;
      vel(b, vi);
      for (let a = 0; a < 3; a++) obs[off.site_rpos + 3 * j + a] = xpos[3 * b + a] - xpos[3 * b0 + a];
      // rrot = R0^T Ri
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++) rrot[3 * r + c] = R0[r] * Ri[c] + R0[3 + r] * Ri[3 + c] + R0[6 + r] * Ri[6 + c];
      matToQuat(rrot, 0, q, 0);
      quatToRotvec(q, obs, off.site_rangles + 3 * j);
      // [ang, lin]: rrot^T w_i - w_0 and R0 (v_0 - v_i)
      for (let r = 0; r < 3; r++) {
        obs[off.site_rvel + 6 * j + r] = rrot[r] * vi[0] + rrot[3 + r] * vi[1] + rrot[6 + r] * vi[2] - v0[r];
        obs[off.site_rvel + 6 * j + 3 + r] = R0[3 * r] * (v0[3] - vi[3]) + R0[3 * r + 1] * (v0[4] - vi[4]) + R0[3 * r + 2] * (v0[5] - vi[5]);
      }
    }
  }

  /** The policy's excitations for the current state and frame. */
  async policy() {
    const { ort, meta } = this;
    const z = this.frames.subarray(this.f * meta.frame_width, this.f * meta.frame_width + meta.nz);
    const out = await this.session.run({
      z: new ort.Tensor('float32', Float32Array.from(z), [1, meta.nz]),
      obs: new ort.Tensor('float32', Float32Array.from(this.observe()), [1, meta.nobs]),
    });
    return out.excitation.data;
  }

  /** One control step under the excitations e (the env's: clip, hold for `decimation` substeps, forward). */
  apply(e) {
    const { mj, m, d, meta } = this;
    for (let i = 0; i < meta.nu; i++) d.ctrl[meta.ctrl_ids[i]] = Math.min(1, Math.max(0, e[i]));
    for (let k = 0; k < meta.decimation; k++) {
      this.pull();
      mj.mj_step(m, d);
    }
    mj.mj_forward(m, d); // the env refreshes derived quantities (the touch sensors) before observing
    this.action.set(e);
    this.f += 1;
  }

  /** Policy then step; returns the excitations applied. */
  async step() {
    const e = await this.policy();
    this.apply(e);
    return e;
  }

  // ---------------- pulling the body (the demo's mouse): a damped spring from a body point to a target ----------------

  /** Start pulling body b at the world point p (the target starts there). */
  grab(b, p) {
    const d = this.d, R = d.xmat.subarray(9 * b, 9 * b + 9), r = [0, 1, 2].map((k) => p[k] - d.xpos[3 * b + k]);
    this.release();
    this.grabbed = { body: b, local: [0, 1, 2].map((k) => R[k] * r[0] + R[3 + k] * r[1] + R[6 + k] * r[2]), target: [...p] };
  }

  drag(target) {
    if (this.grabbed) this.grabbed.target = [...target];
  }

  release() {
    if (this.grabbed) this.d.xfrc_applied.fill(0, 6 * this.grabbed.body, 6 * this.grabbed.body + 6);
    this.grabbed = null;
  }

  /** The grabbed point in world coordinates. */
  grabPoint() {
    const d = this.d, { body: b, local: l } = this.grabbed, R = d.xmat.subarray(9 * b, 9 * b + 9);
    return [0, 1, 2].map((k) => d.xpos[3 * b + k] + R[3 * k] * l[0] + R[3 * k + 1] * l[1] + R[3 * k + 2] * l[2]);
  }

  /** Writes the spring force (stiffness 25 g / m per kg of the whole body, half-critical damping, at most 1.5 times
   * its weight) on the grabbed body, as xfrc_applied at its center of mass. */
  pull() {
    if (!this.grabbed) return;
    const { m, d } = this, b = this.grabbed.body, p = this.grabPoint(), r = m.body_rootid[b], c = d.cvel;
    const M = m.body_subtreemass[r], k = 25 * M, damp = Math.sqrt(k * M), fmax = 1.5 * M * 9.81;
    const w = [c[6 * b], c[6 * b + 1], c[6 * b + 2]], dp = [0, 1, 2].map((i) => p[i] - d.subtree_com[3 * r + i]);
    const v = [c[6 * b + 3] + w[1] * dp[2] - w[2] * dp[1], c[6 * b + 4] + w[2] * dp[0] - w[0] * dp[2], c[6 * b + 5] + w[0] * dp[1] - w[1] * dp[0]];
    let F = [0, 1, 2].map((i) => k * (this.grabbed.target[i] - p[i]) - damp * v[i]);
    const n = Math.hypot(...F);
    if (n > fmax) F = F.map((x) => (x * fmax) / n);
    const a = [0, 1, 2].map((i) => p[i] - d.xipos[3 * b + i]);
    d.xfrc_applied.set([F[0], F[1], F[2], a[1] * F[2] - a[2] * F[1], a[2] * F[0] - a[0] * F[2], a[0] * F[1] - a[1] * F[0]], 6 * b);
  }

  delete() {
    this.d.delete();
    this.di.delete();
    this.m.delete();
  }
}

// rotations.py's conventions

export function quatToMat(w, x, y, z, R) {
  const n = Math.hypot(w, x, y, z);
  w /= n; x /= n; y /= n; z /= n;
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z); R[2] = 2 * (x * z + w * y);
  R[3] = 2 * (x * y + w * z); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
  R[6] = 2 * (x * z - w * y); R[7] = 2 * (y * z + w * x); R[8] = 1 - 2 * (x * x + y * y);
}

/** mat_to_quat: the largest-diagonal candidate, normalized, w >= 0. */
export function matToQuat(M, i, out, j) {
  const m00 = M[i], m01 = M[i + 1], m02 = M[i + 2], m10 = M[i + 3], m11 = M[i + 4], m12 = M[i + 5];
  const m20 = M[i + 6], m21 = M[i + 7], m22 = M[i + 8];
  const qw = Math.sqrt(Math.max(1 + m00 + m11 + m22, 1e-10)) / 2, qx = Math.sqrt(Math.max(1 + m00 - m11 - m22, 1e-10)) / 2;
  const qy = Math.sqrt(Math.max(1 - m00 + m11 - m22, 1e-10)) / 2, qz = Math.sqrt(Math.max(1 - m00 - m11 + m22, 1e-10)) / 2;
  let q;
  if (qw >= qx && qw >= qy && qw >= qz) q = [qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)];
  else if (qx >= qy && qx >= qz) q = [(m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)];
  else if (qy >= qz) q = [(m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)];
  else q = [(m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz];
  const n = Math.max(Math.hypot(q[0], q[1], q[2], q[3]), 1e-12), s = q[0] < 0 ? -1 : 1;
  for (let k = 0; k < 4; k++) out[j + k] = (s * q[k]) / n;
}

/** quat_to_rotvec (w >= 0). */
export function quatToRotvec(q, out, j) {
  const s = q[0] < 0 ? -1 : 1, w = s * q[0], x = s * q[1], y = s * q[2], z = s * q[3];
  const sinHalf = Math.hypot(x, y, z);
  const scale = sinHalf < 1e-6 ? 2 / Math.max(w, 1e-6) : (2 * Math.atan2(sinHalf, w)) / (sinHalf + 1e-12);
  out[j] = x * scale; out[j + 1] = y * scale; out[j + 2] = z * scale;
}
