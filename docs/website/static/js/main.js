// muscle-supermimic project page: lazy video playback, BibTeX copy, the MS-Human-700 clip gallery, and the charts
// (SVG drawn from window.CHART_DATA, re-drawn on resize; crosshair tooltips; the time axis shows training
// iterations while hovered, or tapped on a touch screen).

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
// log scale, d0 > 0: runs whose lengths differ many times over on one time axis (ticks are given by the chart)
function logScale(d0, d1, r0, r1) {
  const l0 = Math.log(d0), l1 = Math.log(d1);
  const f = (v) => r0 + ((Math.log(v) - l0) / (l1 - l0)) * (r1 - r0);
  f.invert = (p) => Math.exp(l0 + ((p - r0) / (r1 - r0)) * (l1 - l0));
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

// piecewise-linear y(x) through (xs, ys), xs ascending, clamped at both ends
function interp(xs, ys, v) {
  if (v <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) if (v <= xs[i]) return ys[i - 1] + ((v - xs[i - 1]) / (xs[i] - xs[i - 1])) * (ys[i] - ys[i - 1]);
  return ys[ys.length - 1];
}

// ── Line chart: one y scale, crosshair + tooltip listing every series ─────
// o.iter = {h, it}: a run's hours and iterations; with it, the time axis switches to iterations while hovered.
// o.xLog with o.xTicks = [[value, label], ...]: a log time axis; series of different runs (each with its own s.it)
// list their own iterations in the tooltip
function lineChart(host, o) {
  let iterAxis = false;
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
    const x = (o.xLog ? logScale : linear)(o.xDomain[0], o.xDomain[1], m.l, W - m.r);
    const y = linear(o.yDomain[0], o.yDomain[1], H - m.b, m.t);
    const g = el('g', { class: 'grid' }, svg), ax = el('g', { class: 'axis' }, svg);
    for (const v of y.ticks(4)) {
      el('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, g);
      const t = el('text', { x: m.l - 6, y: y(v) + 3.5, 'text-anchor': 'end', class: 'tick' }, ax);
      t.textContent = compact(v);
      t.setAttribute('class', 'tick');
    }
    el('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, ax);
    const xg = el('g', { class: 'axis' }, svg);
    const xl = el('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'axis-label' }, svg);
    const xTicks = () => {  // hours, or round iterations placed at the hours they were reached
      xg.replaceChildren();
      const n = W < 420 ? 4 : 6;
      const ticks = iterAxis
        ? linear(0, interp(o.iter.h, o.iter.it, o.xDomain[1]), 0, 1).ticks(n).map((v) => [interp(o.iter.it, o.iter.h, v), compact(v)])
        : o.xTicks || x.ticks(n).map((v) => [v, (o.xFmt || compact)(v)]);
      for (const [v, label] of ticks) {
        const t = el('text', { x: x(v), y: H - m.b + 15, 'text-anchor': 'middle' }, xg);
        t.textContent = label;
        t.setAttribute('class', 'tick');
      }
      xl.textContent = iterAxis ? 'training iteration' : o.xLabel;
    };
    xTicks();
    for (const r of o.refLines || []) {
      el('line', { x1: m.l, x2: W - m.r, y1: y(r.y), y2: y(r.y), class: 'ref-line', 'stroke-dasharray': '0' }, svg);
      const t = el('text', r.anchor === 'start' ? { x: m.l + 4, y: y(r.y) - 5, class: 'ref-label' }
        : { x: W - m.r, y: y(r.y) - 5, 'text-anchor': 'end', class: 'ref-label' }, svg);
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
    }
    for (const mk of o.markers || []) {
      const c = el('circle', { cx: x(mk.x), cy: y(mk.y), r: 4.5, 'stroke-width': 2 }, svg);
      c.style.fill = mk.color; c.style.stroke = 'var(--surface)';
      const t = el('text', { x: x(mk.x) + (mk.anchor === 'end' ? -8 : 8), y: y(mk.y) + (mk.dy ?? -8), 'text-anchor': mk.anchor || 'start', class: 'direct-label' }, svg);
      t.textContent = mk.label;
    }
    // crosshair: snaps to the nearest data x of any series; each series shows its point nearest to it (on screen)
    const allX = [...new Set(o.series.flatMap(sx))].sort((a, b) => a - b);
    const nearest = (xs, p) => {
      let b = 0;
      for (let i = 1; i < xs.length; i++) if (Math.abs(x(xs[i]) - p) < Math.abs(x(xs[b]) - p)) b = i;
      return b;
    };
    const cross = el('line', { y1: m.t, y2: H - m.b, class: 'crosshair', visibility: 'hidden' }, svg);
    const dots = o.series.map((s) => {
      const c = el('circle', { r: 4, 'stroke-width': 2, visibility: 'hidden' }, svg);
      c.style.fill = s.color; c.style.stroke = 'var(--surface)';
      return c;
    });
    let idx = -1;
    const runs = o.series.filter((s) => s.it).length > 1;  // series of different runs: an iteration per row
    const show = (i) => {
      idx = Math.max(0, Math.min(allX.length - 1, i));
      const xv = allX[idx], px = x(xv), rows = [], ys = [];
      let it = null;  // the training iteration of the point shown, from the first series that carries them (s.it)
      cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
      o.series.forEach((s, k) => {
        const xs = sx(s), p0 = x(xs[0]), p1 = x(xs[xs.length - 1]), gap = xs.length > 1 ? (p1 - p0) / (xs.length - 1) : 0;
        const on = xs.length > 0 && px >= p0 - gap / 2 && px <= p1 + gap / 2;  // within the series' span
        dots[k].setAttribute('visibility', on ? 'visible' : 'hidden');
        if (!on) return;
        const jx = nearest(xs, px);
        dots[k].setAttribute('cx', x(xs[jx])); dots[k].setAttribute('cy', y(s.values[jx]));
        const name = runs && s.it ? `${s.name} · iteration ${s.it[jx].toLocaleString('en-US')}` : s.name;
        rows.push({ name, value: (o.yFmt || compact)(s.values[jx]), color: s.color }), ys.push(y(s.values[jx]));
        if (it === null && s.it && !runs) it = s.it[jx];
      });
      const head = `${o.xName} ${(o.xFmt || compact)(xv)}`.trim() + (it === null ? '' : ` · iteration ${it.toLocaleString('en-US')}`);
      tip.show(px * plot.clientWidth / W, Math.min(...ys) * plot.clientWidth / W, head, rows);
    };
    const hide = () => { cross.setAttribute('visibility', 'hidden'); dots.forEach((d) => d.setAttribute('visibility', 'hidden')); tip.hide(); };
    const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
    hit.addEventListener('pointermove', (e) => {
      const box = svg.getBoundingClientRect(), px = ((e.clientX - box.left) * W) / box.width;
      show(nearest(allX, px));
    });
    hit.addEventListener('pointerleave', hide);
    if (o.iter) {  // the axis band: hover with a mouse, tap on a touch screen
      const band = el('rect', { x: m.l, y: H - m.b, width: W - m.l - m.r, height: m.b, fill: 'transparent', class: 'axis-band' }, svg);
      const set = (v) => { if (iterAxis !== v) { iterAxis = v; xTicks(); } };
      let touch = false;
      band.addEventListener('pointerdown', (e) => { touch = e.pointerType !== 'mouse'; });
      band.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') set(true); });
      band.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') set(false); });
      band.addEventListener('click', () => { if (touch) set(!iterAxis); });
    }
    svg.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); show(idx + (e.key === 'ArrowRight' ? 1 : -1)); }
      if (e.key === 'Escape') hide();
    });
    svg.addEventListener('blur', hide);
  };
  new ResizeObserver(draw).observe(plot);
}

// ── Bar chart: horizontal bars from one baseline, the value at each tip, a tooltip per bar ─────
// o.bars = [{label, sub, value, valueLabel, stepLabel, color, stripes: [color, color], head, rows}], o.xMax,
// o.xTicks = [[value, label], ...], o.legend = [{name, color, cls}]; on a narrow screen the labels move above their bars
function barChart(host, o) {
  const plot = html('div', 'plot', host);
  plot.style.position = 'relative';
  if (o.legend) {
    const lg = html('div', 'legend swatches', host);
    for (const s of o.legend) {
      const it = html('span', null, lg), sw = html('i', s.cls || null, it);
      if (s.color) sw.style.background = s.color;
      it.appendChild(document.createTextNode(s.name));
    }
  }
  const tip = tooltip(plot);
  const draw = () => {
    plot.querySelector('svg')?.remove();
    const W = Math.max(plot.clientWidth, 260), narrow = W < 600, BAR = 20, ROW = narrow ? 64 : 46;
    const m = { l: narrow ? 2 : 290, r: 70, t: 4, b: 26 };
    const H = m.t + ROW * o.bars.length + m.b;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.aria });
    plot.prepend(svg);
    const defs = el('defs', {}, svg);
    o.bars.forEach((b, i) => {  // two-colour stripes for a step that comes from both sources
      if (!b.stripes) return;
      const pat = el('pattern', { id: `bar-stripes-${i}`, patternUnits: 'userSpaceOnUse', width: 8, height: 8, patternTransform: 'rotate(45)' }, defs);
      el('rect', { width: 4, height: 8 }, pat).style.fill = b.stripes[0];
      el('rect', { x: 4, width: 4, height: 8 }, pat).style.fill = b.stripes[1];
    });
    const x = linear(0, o.xMax, m.l, W - m.r);
    const g = el('g', { class: 'grid' }, svg), ax = el('g', { class: 'axis' }, svg);
    for (const [v, label] of o.xTicks) {
      if (v > 0) el('line', { x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }, g);
      const t = el('text', { x: x(v), y: H - m.b + 15, 'text-anchor': 'middle', class: 'tick' }, ax);
      t.textContent = label;
    }
    el('line', { x1: x(0), x2: x(0), y1: m.t, y2: H - m.b }, ax);
    o.bars.forEach((b, i) => {
      const top = m.t + ROW * i, y = top + (narrow ? 34 : (ROW - BAR) / 2), x0 = x(0), x1 = Math.max(x(b.value), x0 + 3);
      const r = Math.min(4, (x1 - x0) / 2, BAR / 2);  // rounded data end, square at the baseline
      const p = el('path', { d: `M${x0},${y}H${x1 - r}A${r},${r} 0 0 1 ${x1},${y + r}V${y + BAR - r}A${r},${r} 0 0 1 ${x1 - r},${y + BAR}H${x0}Z` }, svg);
      p.style.fill = b.stripes ? `url(#bar-stripes-${i})` : b.color;
      const lx = narrow ? m.l : m.l - 12, anchor = narrow ? 'start' : 'end';
      const l1 = el('text', { x: lx, y: narrow ? top + 13 : y + 8, 'text-anchor': anchor, class: 'bar-label' }, svg);
      l1.textContent = b.label;
      const l2 = el('text', { x: lx, y: narrow ? top + 27 : y + 22, 'text-anchor': anchor, class: 'bar-sub' }, svg);
      l2.textContent = b.sub;
      const val = el('text', { x: x1 + 8, y: y + BAR / 2 + 4, class: 'bar-value' }, svg);
      val.textContent = b.valueLabel;
      if (b.stepLabel) {
        const st = el('tspan', { class: 'bar-step' }, val);
        st.textContent = `  ${b.stepLabel}`;
      }
      const hit = el('rect', { x: 0, y: top, width: W, height: ROW, fill: 'transparent' }, svg);
      hit.addEventListener('pointermove', () => {
        const s = plot.clientWidth / W;
        tip.show(Math.min(x1, W - m.r) * s, y * s, b.head, b.rows);
      });
      hit.addEventListener('pointerleave', () => tip.hide());
    });
  };
  new ResizeObserver(draw).observe(plot);
}

// ── The page's charts ─────────────────────────────────────────
function charts() {
  const D = window.CHART_DATA;
  if (!D) return;
  const S1 = 'var(--series-1)';
  const hFmt = (v) => `${Number.isInteger(v) ? v : v.toFixed(1)} h`;
  const H_LABEL = 'wall-clock training time (hours)';
  const at = (r, it) => r.wall_h[r.iteration.indexOf(it)];
  const wmBand = (h) => [{ from: 0, to: h, label: 'WM only' }];
  const hEnd = (hs) => Math.ceil(Math.max(...hs) * 2) / 2;
  const iters = (r) => ({ h: [0, ...r.wall_h], it: [0, ...r.iteration] });  // a run's training log: hours -> iteration
  const lenTop = (v) => Math.max(300, Math.ceil(Math.max(...v) / 100) * 100);  // episodes have no time limit

  // MyoFullBody: one run, its evaluated checkpoint marked once there is one
  const mf = D.mfb, ck = mf.checkpoint, k = ck ? mf.iteration.indexOf(ck) : mf.iteration.length - 1, mfEnd = hEnd(mf.wall_h);
  const ckName = `model_${String(ck).padStart(6, '0')}`;
  const h1 = document.getElementById('chart-eplen');
  if (h1) lineChart(h1, { x: mf.wall_h, xDomain: [0, mfEnd], yDomain: [0, lenTop(mf.ep_len)], xLabel: H_LABEL, xName: '', xFmt: hFmt, iter: iters(mf),
    series: [{ name: 'steps', values: mf.ep_len, it: mf.iteration, color: S1 }], bands: wmBand(at(mf, 500)),
    markers: ck ? [{ x: mf.wall_h[k], y: mf.ep_len[k], label: ckName, color: S1, anchor: 'end', dy: 40 }] : [],
    yFmt: (v) => v.toFixed(0), aria: `Mean training episode length rises from about 60 to ${mf.ep_len[k].toFixed(0)} steps in ${mf.wall_h[k].toFixed(1)} hours of training` });
  const h2 = document.getElementById('chart-mfb-eval');
  if (h2) lineChart(h2, { x: mf.eval.wall_h, xDomain: [0, mfEnd], yDomain: [0, 1], xLabel: H_LABEL, xName: '', xFmt: hFmt, iter: iters(mf),
    series: [{ name: 'normalized return', values: mf.eval.norm_return, it: mf.eval.iteration, color: S1, dots: true }], yFmt: (v) => v.toFixed(2),
    aria: `Held-out normalized return ${mf.eval.norm_return.at(-1).toFixed(2)} after ${mf.eval.wall_h.at(-1).toFixed(1)} hours of training` });

  // MS-Human-700: one run
  const ms = D.mshuman;
  const msEnd = ms ? hEnd(ms.train.wall_h) : 1;
  const h7 = document.getElementById('chart-ms-eplen');
  if (h7 && ms) lineChart(h7, { x: ms.train.wall_h, xDomain: [0, msEnd], yDomain: [0, lenTop(ms.train.ep_len)], xLabel: H_LABEL, xName: '', xFmt: hFmt, iter: iters(ms.train),
    series: [{ name: 'steps', values: ms.train.ep_len, it: ms.train.iteration, color: S1 }], bands: wmBand(at(ms.train, 500)), yFmt: (v) => v.toFixed(0),
    aria: `Episode length climbs past 250 steps within 2 hours of training and reaches ${ms.train.ep_len.at(-1).toFixed(0)}` });
  // held-out return: the first release's run and, in orange on the same log time axis, the current fastest recipe
  const h8 = document.getElementById('chart-ms-eval'), fast = D.mshuman_fast?.eval;
  const tFmt = (v) => (v < 1 ? `${(v * 60).toFixed(1).replace(/\.0$/, '')} min` : `${v.toFixed(1).replace(/\.0$/, '')} h`);
  const first = (e, r) => e.wall_h[e.norm_return.findIndex((v) => v >= r)];
  if (h8 && ms && fast) lineChart(h8, { xDomain: [0.05, msEnd], xLog: true, yDomain: [0, 1], xName: '', xFmt: tFmt,
    xLabel: 'wall-clock training time (log scale)', xTicks: [[0.05, '3 min'], [1 / 6, '10 min'], [0.5, '30 min'], [1, '1 h'], [3, '3 h'], [10, '10 h']],
    series: [{ name: 'first release', x: ms.eval.wall_h, values: ms.eval.norm_return, it: ms.eval.iteration, color: S1, dots: true },
      { name: 'fastest recipe', x: fast.wall_h, values: fast.norm_return, it: fast.iteration, color: 'var(--series-2)', dots: true }],
    refLines: [{ y: 0.83, label: '0.83', anchor: 'start' }], yFmt: (v) => v.toFixed(2),
    aria: `Held-out normalized return: the first release passes 0.83 after ${first(ms.eval, 0.83).toFixed(1)} hours of training and reaches ` +
      `${Math.max(...ms.eval.norm_return).toFixed(2)}; the fastest recipe passes 0.83 after ${(first(fast, 0.83) * 60).toFixed(1)} minutes ` +
      `and reaches ${Math.max(...fast.norm_return).toFixed(2)} within ${Math.round(fast.wall_h.at(-1) * 60)} minutes` });
  else if (h8 && ms) lineChart(h8, { x: ms.eval.wall_h, xDomain: [0, msEnd], yDomain: [0, 1], xLabel: H_LABEL, xName: '', xFmt: hFmt, iter: iters(ms.train),
    series: [{ name: 'normalized return', values: ms.eval.norm_return, it: ms.eval.iteration, color: S1, dots: true }], yFmt: (v) => v.toFixed(2),
    aria: `Held-out normalized return rises to ${ms.eval.norm_return.at(-1).toFixed(2)} after ${ms.eval.wall_h.at(-1).toFixed(1)} hours of training` });
  // held-out tracking errors on the same log time axis: the first release's two published checkpoints, the fastest recipe's every one
  const err = ms?.eval_err;
  const errChart = (id, key, unit, digits, what) => {
    const h = document.getElementById(id);
    if (!h || !err || !fast?.[key]) return;
    const top = Math.ceil(Math.max(...err[key], ...fast[key]) * 1.15 / 5) * 5;
    const v = (a, i) => `${a[i].toFixed(digits)}${unit}`;
    lineChart(h, { xDomain: [0.05, msEnd], xLog: true, yDomain: [0, top], xName: '', xFmt: tFmt,
      xLabel: 'wall-clock training time (log scale)', xTicks: [[0.05, '3 min'], [1 / 6, '10 min'], [0.5, '30 min'], [1, '1 h'], [3, '3 h'], [10, '10 h']],
      series: [{ name: 'first release', x: err.wall_h, values: err[key], it: err.iteration, color: S1, dots: true },
        { name: 'fastest recipe', x: fast.wall_h, values: fast[key], it: fast.iteration, color: 'var(--series-2)', dots: true }],
      yFmt: (y) => `${y.toFixed(digits)}${unit}`,
      aria: `Held-out ${what}: the first release ${v(err[key], 0)} at ${err.wall_h[0].toFixed(1)} hours and ${v(err[key], err[key].length - 1)} at ` +
        `${err.wall_h.at(-1).toFixed(1)} hours of training; the fastest recipe ${v(fast[key], fast[key].length - 1)} after ` +
        `${Math.round(fast.wall_h.at(-1) * 60)} minutes, lowest ${Math.min(...fast[key]).toFixed(digits)}${unit}` });
  };
  errChart('chart-ms-mpjpe', 'mpjpe_mm', ' mm', 1, 'MPJPE');
  errChart('chart-ms-joint', 'joint_mae_deg', '°', 2, 'joint angle error');

  // training time to 0.83, step by step, each bar coloured by where its step comes from
  const sp = D.mshuman_speed, h9 = document.getElementById('chart-ms-time');
  if (h9 && sp) {
    const tLabel = (m) => (m >= 60 ? `${(m / 60).toFixed(1)} h` : `${m >= 10 ? Math.round(m) : m.toFixed(1)} min`);
    const S2 = 'var(--series-2)', S3 = 'var(--series-3)';
    const color = { base: S1, theirs: S3, ours: S2, both: S3 };
    const n = sp.minutes.length, fold = sp.minutes[0] / sp.minutes[n - 1];
    barChart(h9, {
      xMax: 270, xTicks: [[0, '0'], [60, '1 h'], [120, '2 h'], [180, '3 h'], [240, '4 h']],
      legend: [{ name: 'first release', color: S1 }, { name: 'from InstantTrack / BlinkMimic', color: S3 }, { name: 'ours', color: S2 },
        { name: 'both', cls: 'both' }],
      bars: sp.minutes.map((v, i) => ({
        label: sp.label[i], sub: sp.sub[i], value: v, color: color[sp.source[i]],
        stripes: sp.source[i] === 'both' ? [S3, S2] : null,
        valueLabel: tLabel(v),
        stepLabel: i === 0 ? '' : `${(sp.minutes[i - 1] / v).toFixed(1)}× faster` + (i === n - 1 ? `, ${Math.round(fold)}× in all` : ''),
        head: sp.step[i],
        rows: [{ name: `to 0.83, at iteration ${sp.iteration[i].toLocaleString('en-US')} (${sp.runs[i]})`, value: tLabel(v), color: color[sp.source[i]] },
          { name: 'per iteration', value: `${sp.s_per_it[i].toFixed(2)} s` }],
      })),
      aria: 'Training time to a held-out normalized return of 0.83 on MS-Human-700, step by step: ' +
        sp.minutes.map((v, i) => `${sp.label[i]} (${sp.sub[i]}) ${tLabel(v)}`).join('; ') + `; ${Math.round(fold)} times faster in all`,
    });
  }
}

// ── MS-Human-700 clip gallery: pick a kind of motion, then one of its clips ─────
function gallery() {
  const G = window.MS_GALLERY, root = document.getElementById('ms-gallery');
  if (!G || !root) return;
  const video = root.querySelector('video'), kinds = root.querySelector('.gallery-kinds');
  const list = root.querySelector('.gallery-clips'), info = root.querySelector('.gallery-info');
  const press = (box, key, v) => box.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset[key] === v)));
  const chip = (box, label, onClick) => {
    const b = html('button', 'chip', box, label);
    b.type = 'button';
    b.addEventListener('click', onClick);
    return b;
  };
  const pick = (c, play) => {
    video.src = c.src;
    video.setAttribute('aria-label', `MS-Human-700 tracking ${c.id} (${c.kind})`);
    if (play) video.play().catch(() => {});  // on load, the visibility observer starts it
    press(list, 'id', c.id);
    info.textContent = `${c.id} · ${c.seconds} s · normalized return ${c.ret.toFixed(2)} · `
      + (c.fell_at === undefined ? 'tracked to the end' : `falls at ${c.fell_at} s`);
  };
  const showKind = (k, c, play) => {
    press(kinds, 'kind', k);
    list.replaceChildren();
    for (const x of G.clips.filter((x) => x.kind === k)) chip(list, x.id, () => pick(x, true)).dataset.id = x.id;
    pick(c || G.clips.find((x) => x.kind === k), play);
  };
  for (const k of new Set(G.clips.map((x) => x.kind))) chip(kinds, k, () => showKind(k, null, true)).dataset.kind = k;
  const start = G.clips.find((x) => x.id === G.start) || G.clips[0];
  showKind(start.kind, start, false);
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

// ── The live viewer: loaded on request (a 66 MB download for MS-Human-700), paused while out of sight ──
function liveViewer() {
  const start = document.querySelector('.live-start');
  if (!start) return;
  start.addEventListener('click', () => {
    const frame = document.createElement('iframe');
    frame.src = start.dataset.src;
    frame.title = 'A musculoskeletal body tracking a held-out clip, simulated live';
    frame.allow = 'fullscreen';
    frame.allowFullscreen = true;
    start.replaceWith(frame);
    let visible = true;
    const tell = () => frame.contentWindow?.postMessage({ type: 'visible', visible }, location.origin);
    frame.addEventListener('load', tell);
    new IntersectionObserver((entries) => {
      for (const e of entries) { visible = e.isIntersecting; tell(); }
    }, { threshold: 0.1 }).observe(frame);
  });
}

function copyButtons() {
  document.querySelectorAll('.copy-btn').forEach((b) => b.addEventListener('click', async () => {
    const text = document.getElementById(b.dataset.target).textContent;
    try { await navigator.clipboard.writeText(text); b.textContent = 'Copied'; } catch { b.textContent = 'Select and copy'; }
    setTimeout(() => (b.textContent = 'Copy'), 1500);
  }));
}

document.addEventListener('DOMContentLoaded', () => { charts(); gallery(); lazyVideos(); liveViewer(); copyButtons(); });
