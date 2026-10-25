// muscle-supermimic project page: lazy video playback, BibTeX copy, and the charts
// (SVG drawn from window.CHART_DATA, re-drawn on resize; crosshair / per-bar tooltips; data tables).

const SVG = 'http://www.w3.org/2000/svg';

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function html(tag, cls, parent, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  if (parent) parent.appendChild(n);
  return n;
}

// ── Number formatting ─────────────────────────────────────────
function compact(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1).replace(/\.0$/, '') + 'M';
  if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1).replace(/\.0$/, '') + 'K';
  if (a >= 100 || Number.isInteger(v)) return String(Math.round(v));
  if (a >= 1) return v.toFixed(1).replace(/\.0$/, '');
  return v.toPrecision(2);
}
const superscript = (n) => String(n).replace(/-/g, '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
const pow10 = (e) => (e === 0 ? '1' : e === 1 ? '10' : '10' + superscript(e));

// ── Scales ────────────────────────────────────────────────────
function linear(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  f.ticks = (n) => {
    const step0 = (d1 - d0) / n, mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
    const out = [];
    for (let v = Math.ceil(d0 / step) * step; v <= d1 + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
  return f;
}
function log10(d0, d1, r0, r1) {
  const l0 = Math.log10(d0), l1 = Math.log10(d1);
  const f = (v) => r0 + ((Math.log10(Math.max(v, d0)) - l0) / (l1 - l0)) * (r1 - r0);
  f.ticks = () => {
    const out = [];
    for (let e = Math.ceil(l0); e <= Math.floor(l1); e++) out.push(10 ** e);
    return out;
  };
  return f;
}

// ── Tooltip ───────────────────────────────────────────────────
function tooltip(host) {
  const t = html('div', 'tooltip', host);
  t.hidden = true;
  return {
    show(x, y, head, rows) {
      t.replaceChildren();
      html('div', 'tt-head', t, head);
      for (const r of rows) {
        const row = html('div', 'tt-row', t);
        if (r.color) html('i', null, row).style.background = r.color;
        html('b', null, row, r.value);
        html('span', null, row, r.name);
      }
      t.hidden = false;
      const hw = host.clientWidth, tw = t.offsetWidth;
      t.style.left = Math.min(Math.max(x + 12, 0), hw - tw) + 'px';
      t.style.top = Math.max(y - 10, 0) + 'px';
    },
    hide() { t.hidden = true; },
  };
}

// ── Data table (the table view of every chart) ────────────────
function dataTable(host, head, rows) {
  const d = html('details', 'data', host);
  html('summary', null, d, 'Data table');
  const wrap = html('div', 'table-wrap', d);
  const table = html('table', null, wrap);
  const tr = html('tr', null, html('thead', null, table));
  for (const h of head) html('th', null, tr, h);
  const tb = html('tbody', null, table);
  for (const r of rows) {
    const row = html('tr', null, tb);
    for (const c of r) html('td', null, row, c);
  }
}

// ── Line chart: one y scale, crosshair + tooltip listing every series ─────
function lineChart(host, o) {
  const plot = html('div', 'plot', host);
  plot.style.position = 'relative';
  if (o.series.length > 1) {
    const lg = html('div', 'legend', host.querySelector('.chart-legend') || host);
    for (const s of o.series) {
      const it = html('span', null, lg);
      html('i', null, it).style.background = s.color;
      it.appendChild(document.createTextNode(s.name));
    }
  }
  const tip = tooltip(plot);
  const draw = () => {
    plot.querySelector('svg')?.remove();
    const W = Math.max(plot.clientWidth, 260), H = o.height || 220;
    const m = { l: 46, r: o.rightPad || 14, t: 10, b: 34 };
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.aria, tabindex: 0 });
    plot.prepend(svg);
    const x = linear(o.xDomain[0], o.xDomain[1], m.l, W - m.r);
    const y = o.yLog ? log10(o.yDomain[0], o.yDomain[1], H - m.b, m.t) : linear(o.yDomain[0], o.yDomain[1], H - m.b, m.t);
    const g = el('g', { class: 'grid' }, svg), ax = el('g', { class: 'axis' }, svg);
    for (const v of y.ticks(4)) {
      el('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, g);
      const t = el('text', { x: m.l - 6, y: y(v) + 3.5, 'text-anchor': 'end', class: 'tick' }, ax);
      t.textContent = o.yLog ? pow10(Math.round(Math.log10(v))) : compact(v);
      t.setAttribute('class', 'tick');
    }
    el('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, ax);
    for (const v of x.ticks(W < 420 ? 4 : 6)) {
      const t = el('text', { x: x(v), y: H - m.b + 15, 'text-anchor': 'middle' }, ax);
      t.textContent = (o.xFmt || compact)(v);
      t.setAttribute('class', 'tick');
    }
    const xl = el('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'axis-label' }, svg);
    xl.textContent = o.xLabel;
    for (const r of o.refLines || []) {
      el('line', { x1: m.l, x2: W - m.r, y1: y(r.y), y2: y(r.y), class: 'ref-line', 'stroke-dasharray': '0' }, svg);
      const t = el('text', { x: W - m.r, y: y(r.y) - 5, 'text-anchor': 'end', class: 'ref-label' }, svg);
      t.textContent = r.label;
    }
    for (const b of o.bands || []) {
      const r = el('rect', { x: x(b.from), y: m.t, width: x(b.to) - x(b.from), height: H - m.b - m.t }, svg);
      r.style.fill = 'var(--bg-alt)';
      svg.insertBefore(r, g);
      const t = el('text', { x: x(b.to) + 4, y: H - m.b - 6, class: 'ref-label' }, svg);
      t.textContent = b.label;
    }
    const sx = (s) => (s.x || o.x).slice(0, s.values.length);  // each series on its own x, or on the shared grid
    for (const s of o.series) {
      const xs = sx(s);
      const d = xs.map((xv, i) => `${i ? 'L' : 'M'}${x(xv).toFixed(1)},${y(s.values[i]).toFixed(1)}`).join('');
      const p = el('path', { d, fill: 'none', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
      p.style.stroke = s.color;
      if (s.dots) {
        xs.forEach((xv, i) => {
          const c = el('circle', { cx: x(xv), cy: y(s.values[i]), r: 4, 'stroke-width': 2 }, svg);
          c.style.fill = s.color; c.style.stroke = 'var(--surface)';
        });
      }
      if (s.endLabel) {
        const i = xs.length - 1;
        const t = el('text', { x: x(xs[i]) + 6, y: y(s.values[i]) + 4, class: 'direct-label' }, svg);
        t.textContent = s.name;
      }
    }
    for (const mk of o.markers || []) {
      const c = el('circle', { cx: x(mk.x), cy: y(mk.y), r: 4.5, 'stroke-width': 2 }, svg);
      c.style.fill = mk.color; c.style.stroke = 'var(--surface)';
      const t = el('text', { x: x(mk.x) + (mk.anchor === 'end' ? -8 : 8), y: y(mk.y) + (mk.dy ?? -8), 'text-anchor': mk.anchor || 'start', class: 'direct-label' }, svg);
      t.textContent = mk.label;
    }
    // crosshair: snaps to the nearest data x of any series; each series shows its point nearest to it
    const allX = [...new Set(o.series.flatMap(sx))].sort((a, b) => a - b);
    const nearest = (xs, v) => {
      let b = 0;
      for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - v) < Math.abs(xs[b] - v)) b = i;
      return b;
    };
    const cross = el('line', { y1: m.t, y2: H - m.b, class: 'crosshair', visibility: 'hidden' }, svg);
    const dots = o.series.map((s) => {
      const c = el('circle', { r: 4, 'stroke-width': 2, visibility: 'hidden' }, svg);
      c.style.fill = s.color; c.style.stroke = 'var(--surface)';
      return c;
    });
    let idx = -1;
    const show = (i) => {
      idx = Math.max(0, Math.min(allX.length - 1, i));
      const xv = allX[idx], px = x(xv), rows = [], ys = [];
      cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
      o.series.forEach((s, k) => {
        const xs = sx(s), gap = xs.length > 1 ? (xs[xs.length - 1] - xs[0]) / (xs.length - 1) : 0;
        const on = xs.length > 0 && xv >= xs[0] - gap / 2 && xv <= xs[xs.length - 1] + gap / 2;  // within the series' span
        dots[k].setAttribute('visibility', on ? 'visible' : 'hidden');
        if (!on) return;
        const jx = nearest(xs, xv);
        dots[k].setAttribute('cx', x(xs[jx])); dots[k].setAttribute('cy', y(s.values[jx]));
        rows.push({ name: s.name, value: (o.yFmt || compact)(s.values[jx]), color: s.color }), ys.push(y(s.values[jx]));
      });
      tip.show(px * plot.clientWidth / W, Math.min(...ys) * plot.clientWidth / W, `${o.xName} ${(o.xFmt || compact)(xv)}`.trim(), rows);
    };
    const hide = () => { cross.setAttribute('visibility', 'hidden'); dots.forEach((d) => d.setAttribute('visibility', 'hidden')); tip.hide(); };
    const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
    hit.addEventListener('pointermove', (e) => {
      const box = svg.getBoundingClientRect(), px = ((e.clientX - box.left) * W) / box.width;
      show(nearest(allX, x.invert(px)));
    });
    hit.addEventListener('pointerleave', hide);
    svg.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); show(idx + (e.key === 'ArrowRight' ? 1 : -1)); }
      if (e.key === 'Escape') hide();
    });
    svg.addEventListener('blur', hide);
  };
  new ResizeObserver(draw).observe(plot);
}

// ── Horizontal bars on a log scale (bars grow from the baseline at x = 1) ───────────
function barChart(host, o) {
  const plot = html('div', 'plot', host);
  plot.style.position = 'relative';
  const tip = tooltip(plot);
  const draw = () => {
    plot.querySelector('svg')?.remove();
    const W = Math.max(plot.clientWidth, 280), row = 22, m = { l: 128, r: 40, t: 24, b: 30 };
    const H = m.t + m.b + row * o.rows.length;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.aria });
    plot.prepend(svg);
    const x = log10(o.xDomain[0], o.xDomain[1], m.l, W - m.r);
    const g = el('g', { class: 'grid' }, svg), ax = el('g', { class: 'axis' }, svg);
    for (const v of x.ticks()) {
      el('line', { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }, g);
      const t = el('text', { x: x(v), y: H - m.b + 15, 'text-anchor': 'middle' }, ax);
      t.textContent = pow10(Math.round(Math.log10(v)));
      t.setAttribute('class', 'tick');
    }
    el('line', { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b }, ax);
    const xl = el('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'axis-label' }, svg);
    xl.textContent = o.xLabel;
    for (const r of o.refLines) {
      el('line', { x1: x(r.x), x2: x(r.x), y1: m.t - 6, y2: H - m.b, class: 'ref-line' }, svg);
      const t = el('text', { x: x(r.x), y: m.t - 10, 'text-anchor': r.anchor || 'middle', class: 'ref-label' }, svg);
      t.textContent = r.label;
    }
    o.rows.forEach((r, i) => {
      const cy = m.t + row * i + row / 2, h = 14, x0 = m.l, x1 = Math.max(x(r.value), x0 + 4), rad = 4;
      const lab = el('text', { x: m.l - 8, y: cy + 4, 'text-anchor': 'end', class: 'tick' }, ax);
      lab.textContent = r.name;
      const d = `M${x0},${cy - h / 2}H${x1 - rad}Q${x1},${cy - h / 2} ${x1},${cy - h / 2 + rad}V${cy + h / 2 - rad}Q${x1},${cy + h / 2} ${x1 - rad},${cy + h / 2}H${x0}Z`;
      const b = el('path', { d, class: 'bar', tabindex: 0, role: 'img', 'aria-label': `${r.name}: ${compact(r.value)}` }, svg);
      b.style.fill = r.color;
      if (r.label) {
        const t = el('text', { x: x1 + 5, y: cy + 4, class: 'direct-label' }, svg);
        t.textContent = compact(r.value);
      }
      const show = () => tip.show(x1 * plot.clientWidth / W, (cy - 12) * plot.clientWidth / W, o.tipHead, [{ name: r.name, value: compact(r.value), color: r.color }]);
      b.addEventListener('pointermove', show); b.addEventListener('focus', show);
      b.addEventListener('pointerleave', tip.hide); b.addEventListener('blur', tip.hide);
    });
  };
  new ResizeObserver(draw).observe(plot);
}

// ── The page's charts ─────────────────────────────────────────
function charts() {
  const D = window.CHART_DATA;
  if (!D) return;
  const S1 = 'var(--series-1)', S2 = 'var(--series-2)', MUTED = 'var(--bar-muted)';
  const run = D.run;
  const hFmt = (v) => `${Number.isInteger(v) ? v : v.toFixed(1)} h`;
  const H_LABEL = 'wall-clock training time (hours)';
  const at = (r, it) => r.wall_h[r.iteration.indexOf(it)];
  const k12 = run.iteration.findIndex((v) => v > 12000);
  const hours = run.wall_h.slice(0, k12);
  const wmBand = (h) => [{ from: 0, to: h, label: 'WM only' }];

  const h1 = document.getElementById('chart-eplen');
  if (h1) {
    lineChart(h1, { x: hours, xDomain: [0, 11], yDomain: [0, 300], xLabel: H_LABEL, xName: '', xFmt: hFmt,
      series: [{ name: 'steps', values: run.ep_len.slice(0, k12), color: S1 }], bands: wmBand(at(run, 500)),
      markers: [{ x: at(run, 12000), y: run.ep_len[k12 - 1], label: 'model_012000', color: S1, anchor: 'end', dy: 22 }],
      yFmt: (v) => v.toFixed(0), aria: 'Mean training episode length rises from about 60 to 268 steps over 10.7 hours of training' });
    dataTable(h1, ['iteration', 'hours', 'episode length (steps)', 'ending in failure (%)'],
      run.iteration.slice(0, k12).map((v, i) => [v, hours[i].toFixed(2), run.ep_len[i], run.fail_pct[i]]).filter((r) => r[0] % 1000 === 0));
  }
  const h2 = document.getElementById('chart-fail');
  if (h2) lineChart(h2, { x: hours, xDomain: [0, 11], yDomain: [0, 100], xLabel: H_LABEL, xName: '', xFmt: hFmt,
    series: [{ name: '% of episodes', values: run.fail_pct.slice(0, k12), color: S1 }], bands: wmBand(at(run, 500)),
    markers: [{ x: at(run, 12000), y: run.fail_pct[k12 - 1], label: `${run.fail_pct[k12 - 1].toFixed(0)} %`, color: S1, anchor: 'end' }],
    yFmt: (v) => v.toFixed(0) + ' %', aria: 'Share of training episodes ending in a failure falls from about 85 % to 7 % over 10.7 hours of training' });

  // MS-Human-700: two seeds with the state-gradient clip against the run without it
  const ms = D.mshuman;
  const GRAY = 'var(--text-light)';
  const runs = [['s42', 'clip, seed 42', S1], ['s1', 'clip, seed 1', S2], ['noclip', 'no clip, seed 42', GRAY]];
  const msEnd = ms ? Math.ceil(Math.max(...runs.map(([k]) => ms.train[k].wall_h.at(-1))) * 2) / 2 : 1;
  const h7 = document.getElementById('chart-ms-eplen');
  if (h7 && ms) {
    lineChart(h7, { xDomain: [0, msEnd], yDomain: [0, 300], xLabel: H_LABEL, xName: '', xFmt: hFmt, bands: wmBand(at(ms.train.s42, 500)),
      series: runs.map(([k, name, color]) => ({ name, color, x: ms.train[k].wall_h, values: ms.train[k].ep_len })), yFmt: (v) => v.toFixed(0),
      aria: 'With the clip, episode length climbs past 250 steps within 2 hours in both seeds; without it, it stays near 90 through 4 hours' });
    dataTable(h7, ['iteration'].concat(runs.map((r) => `${r[1]}: steps (hours)`)),
      ms.train.noclip.iteration.filter((v) => v % 500 === 0).map((v) => [v].concat(runs.map(([k]) => {
        const i = ms.train[k].iteration.indexOf(v);
        return i < 0 ? '–' : `${ms.train[k].ep_len[i]} (${ms.train[k].wall_h[i].toFixed(2)})`;
      }))));
  }
  const h8 = document.getElementById('chart-ms-eval');
  if (h8 && ms) {
    lineChart(h8, { xDomain: [0, msEnd], yDomain: [0, 1], xLabel: H_LABEL, xName: '', xFmt: hFmt,
      series: runs.map(([k, name, color]) => ({ name, color, x: ms.eval[k].wall_h, values: ms.eval[k].norm_return, dots: true })), yFmt: (v) => v.toFixed(2),
      aria: 'Held-out normalized return: with the clip 0.61 after 1.7 hours (seed 42); without it 0.07 to 0.18 through 3.7 hours' });
    dataTable(h8, ['checkpoint'].concat(runs.map((r) => `${r[1]}: return (clips tracked) at hours`)),
      ms.eval.noclip.iteration.map((v) => [v].concat(runs.map(([k]) => {
        const i = ms.eval[k].iteration.indexOf(v);
        return i < 0 ? '–' : `${ms.eval[k].norm_return[i].toFixed(2)} (${ms.eval[k].tracked[i]}) at ${ms.eval[k].wall_h[i].toFixed(2)}`;
      }))));
  }

  // the collapse: iterations 11000-14000
  const a = run.iteration.findIndex((v) => v >= 11000), b = run.iteration.findIndex((v) => v > 14000);
  const xs = run.wall_h.slice(a, b), dom = [Math.floor(xs[0] * 2) / 2, Math.ceil(xs.at(-1) * 2) / 2];
  const h3 = document.getElementById('chart-collapse-len');
  if (h3) lineChart(h3, { x: xs, xDomain: dom, yDomain: [0, 300], xLabel: H_LABEL, xName: '', xFmt: hFmt, height: 170,
    series: [{ name: 'steps', values: run.ep_len.slice(a, b), color: S1 }], yFmt: (v) => v.toFixed(0),
    aria: 'Episode length stays near 270 steps until 11.3 hours, then drops to about 110' });
  const h4 = document.getElementById('chart-collapse-grad');
  if (h4) {
    lineChart(h4, { x: xs, xDomain: dom, yDomain: [0.1, 1e5], yLog: true, xLabel: H_LABEL, xName: '', xFmt: hFmt, height: 170,
      series: [{ name: 'gradient norm', values: run.pi_gradnorm.slice(a, b), color: S2 }], yFmt: compact,
      aria: 'Policy-gradient norm stays near 1 until 11.3 hours (iteration 12700), then jumps to 10^3-10^4' });
    dataTable(h4, ['iteration', 'hours', 'episode length (steps)', 'policy-gradient norm, before clipping'],
      run.iteration.slice(a, b).map((v, i) => [v, xs[i].toFixed(2), run.ep_len[a + i], compact(run.pi_gradnorm[a + i])]).filter((r) => r[0] % 250 === 0));
  }

  // per-step growth of the state gradient, top 1 % vs the rest
  const gr = D.growth;
  const h5 = document.getElementById('chart-growth');
  if (h5) {
    lineChart(h5, { x: gr.step, xDomain: [0, 31], yDomain: [1, 1e5], yLog: true, xLabel: 'imagined step t (the gradient flows from t = 31 back to t = 0)', xName: 'step', rightPad: 70,
      xFmt: (v) => String(v), yFmt: compact, height: 230,
      series: [{ name: 'top 1 %', values: gr.top1, color: S2, endLabel: false }, { name: 'other 99 %', values: gr.rest, color: S1 }],
      markers: [{ x: 0, y: gr.top1[0], label: `top 1 %: ${compact(gr.top1[0])}`, color: S2 }, { x: 0, y: gr.rest[0], label: `other 99 %: ${compact(gr.rest[0])}`, color: S1 }],
      aria: 'Going back from step 31 to step 0, the state gradient of the top 1 % of trajectories grows from about 3 to 22600; the rest stay between 3 and 12' });
    dataTable(h5, ['step t', 'top 1 %', 'other 99 %'], gr.step.map((t, i) => [t, compact(gr.top1[i]), compact(gr.rest[i])]));
  }

  // leave-one-out ablation
  const ab = D.ablation;
  const h6 = document.getElementById('chart-ablation');
  if (h6) {
    const loop = new Set(['muscle_velocity', 'muscle_force', 'muscle_excitation', 'muscle_activation']);
    barChart(h6, { xDomain: [1, 1e5], xLabel: 'gradient norm before clipping', tipHead: 'gradient cut through',
      rows: ab.terms.map(([n, v]) => ({ name: n, value: v, color: loop.has(n) ? S1 : MUTED, label: loop.has(n) })),
      refLines: [{ x: ab.full, label: `nothing cut: ${compact(ab.full)}`, anchor: 'end' }, { x: ab.healthy, label: 'a run that learns: ~2', anchor: 'start' }],
      aria: 'Cutting the gradient through muscle_velocity brings the norm from 49000 to 16; muscle_force 113, excitation 269, activation 331; every other term leaves it above 3500' });
    dataTable(h6, ['observation term cut', 'gradient norm'], ab.terms.map(([n, v]) => [n, compact(v)]).concat([['(nothing cut)', compact(ab.full)]]));
  }
}

// ── Videos play while visible ─────────────────────────────────
function lazyVideos() {
  const vids = document.querySelectorAll('video[data-autoplay]');
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) e.target.play().catch(() => {});
      else e.target.pause();
    }
  }, { threshold: 0.25 });
  vids.forEach((v) => io.observe(v));
}

function copyButtons() {
  document.querySelectorAll('.copy-btn').forEach((b) => b.addEventListener('click', async () => {
    const text = document.getElementById(b.dataset.target).textContent;
    try { await navigator.clipboard.writeText(text); b.textContent = 'Copied'; } catch { b.textContent = 'Select and copy'; }
    setTimeout(() => (b.textContent = 'Copy'), 1500);
  }));
}

document.addEventListener('DOMContentLoaded', () => { charts(); lazyVideos(); copyButtons(); });
