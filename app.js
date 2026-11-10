import * as P from './core/pipeline.mjs';
import * as F from './core/forecast.mjs';
import { CATS, rng } from './core/sim.mjs';

const $ = (id) => document.getElementById(id);
const data = window.SEQ_DATA;
const pct = (x, d = 2) => (x >= 0 ? '+' : '−') + Math.abs(100 * x).toFixed(d) + '%';
const pp = (x, d = 2) => (x >= 0 ? '+' : '−') + Math.abs(100 * x).toFixed(d);
const num = (x) => x.toLocaleString('en-US');
const M = (x) => (x / 1e6).toFixed(1) + 'M';
const tick = () => new Promise((r) => setTimeout(r, 20));
let res, logs, truth = null, selected = null, showAll = false, sortBy = 'fc', ch1mode = 'raw';
const promote = new Set();

async function main() {
  $('loadmsg').textContent = 'Decoding 6.0M logged page views';
  await tick();
  logs = P.decode(data);
  $('loadmsg').textContent = `Scoring ${data.candidates.length} experiments counterfactually`;
  await tick();
  res = P.run(data, logs);
  P.plan(res, {});
  res.recommended.forEach((r) => promote.add(r.id));
  selected = res.recommended[0].id;
  $('arm').value = Math.round(res.opts.armShare * 100);
  render();
  $('loading').classList.add('hide');
  wire();
  if (location.hash === '#truth') reveal();
}

function render() {
  const o = res.opts, H = res.H, cal = res.cal, rows = res.rows;
  const nPast = H.rows.length, sessions = data.history.map((h) => h.session);
  $('lede').textContent = `RRI ran ${rows.length} offline experiments on ${M(res.totalViews)} logged page views. Live Lift Forecast corrects each counterfactual estimate for the winner's curse and for the errors experiments share, then calibrates it on ${nPast} past live A/B tests to forecast the readout you will actually see.`;
  $('chips').innerHTML = [
    ['', data.meta.surface],
    ['', `${M(res.totalViews)} page views · ${data.meta.days} days`],
    ['g', `Exploration · ${Math.round(100 * data.meta.eps)}% shuffled slates`],
    ['g', `${rows.length} experiments`],
    ['y', `${nPast} past live tests · sessions ${sessions[0]}–${sessions[sessions.length - 1]}`],
  ].map(([c, t]) => `<span class="chip ${c}">${t}</span>`).join('');
  $('nexp').textContent = rows.length;

  // KPIs
  const top = res.byRaw[0];
  const hitsRaw = Math.round(H.rawCoverage * nPast), hitsF = Math.round(H.looCoverage * nPast);
  const ord10 = res.byRaw.slice(0, 10);
  const gap10 = ord10.reduce((a, r) => a + (r.post.mean - r.postIndep.mean), 0) / 10;
  $('kpis').innerHTML = [
    ['var(--pink)', 'Top offline lift', pct(top.lift), `<b>${top.id}</b> · DR estimate ± ${(100 * top.se).toFixed(2)} pp`],
    ['var(--teal)', 'Its forecast readout', pct(top.fc.center), `80% band <b>${pp(top.fc.lo)} to ${pp(top.fc.hi)}</b> pp · ${o.days}-day test`],
    ['var(--sage)', 'Past promotions', `${pp(H.meanRaw, 2)}<span class="arrow">→</span>${pp(H.meanLive, 2)}<small>%</small>`, `Offline vs. live, mean of ${nPast} promoted experiments`],
    ['var(--yellow)', 'Band calibration', `${hitsF}<small>/${nPast}</small>`, `Past readouts inside the forecast band (leave-one-out) vs. <b>${hitsRaw}/${nPast}</b> for raw ± SE`],
    ['var(--blue)', 'Shared error', `${Math.round(100 * res.factor.meanCorr)}<small>%</small>`, `Mean correlation of experiment errors. Pooling as independent shifts the top 10 by <b>${pp(-gap10)} pp</b>`],
  ].map(([c, k, v, s]) => `<div class="card kpi"><div class="k"><i style="background:${c}"></i>${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('');

  $('ch1sub').textContent = `Counterfactual (doubly robust) revenue per page view vs. production, and the forecast readout of a ${o.days}-day test at ${Math.round(o.armShare * 100)}% traffic per arm.`;
  $('ch2sub').textContent = `${nPast} promoted experiments. Fitted offline-to-online slope ${cal.b.toFixed(2)}, extra online SD ${(100 * Math.sqrt(cal.tau2)).toFixed(2)} pp, conformal multiplier ${cal.q.toFixed(2)}.`;
  drawCh1(); drawCh2(); drawCh3(); renderTable(); renderPlan(); renderDrawer(); renderPool();
  if (truth) renderTruth();
  if (window.lucide) lucide.createIcons();
}

// ---------- chart 1: dumbbell ----------
function drawCh1() {
  const svg = $('ch1'); const W = svg.clientWidth || 720, Hh = 300;
  const list = (ch1mode === 'raw' ? res.byRaw : res.byWin).slice(0, 30);
  const vals = list.flatMap((r) => [r.lift, r.fc.lo, r.fc.hi, ...(truth ? [truth[r.id].trueLiveLift] : [])]);
  let lo = Math.min(0, ...vals), hi = Math.max(...vals);
  const step = 0.005; lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const L = 44, R = 10, T = 12, B = 30;
  const x = (i) => L + (i + 0.5) * (W - L - R) / list.length;
  const y = (v) => T + (hi - v) / (hi - lo) * (Hh - T - B);
  let s = '';
  for (let v = lo; v <= hi + 1e-9; v += step) s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="${Math.abs(v) < 1e-9 ? '#bdbdbd' : 'rgba(34,34,34,0.07)'}" ${Math.abs(v) < 1e-9 ? '' : 'stroke-dasharray="3 4"'}/><text x="${L - 8}" y="${y(v) + 3}" text-anchor="end">${(100 * v).toFixed(1)}%</text>`;
  list.forEach((r, i) => {
    const rec = res.recommended.includes(r);
    s += `<g data-id="${r.id}" class="pt" style="cursor:pointer">`;
    s += `<rect x="${x(i) - 9}" y="${T}" width="18" height="${Hh - T - B}" fill="${r.id === selected ? 'rgba(167,212,228,0.22)' : 'transparent'}"/>`;
    s += `<line x1="${x(i)}" x2="${x(i)}" y1="${y(r.lift)}" y2="${y(r.fc.center)}" stroke="rgba(84,112,118,0.35)" stroke-width="1.2"/>`;
    s += `<rect x="${x(i) - 3.5}" y="${y(r.fc.hi)}" width="7" height="${Math.max(2, y(r.fc.lo) - y(r.fc.hi))}" rx="3.5" fill="rgba(167,212,228,0.85)"/>`;
    if (rec) s += `<circle cx="${x(i)}" cy="${y(r.fc.center)}" r="8" fill="none" stroke="#fcb315" stroke-width="2"/>`;
    s += `<circle cx="${x(i)}" cy="${y(r.fc.center)}" r="4.6" fill="#064f6e"/>`;
    s += `<circle cx="${x(i)}" cy="${y(r.lift)}" r="4.2" fill="#fff" stroke="#f37f94" stroke-width="2"/>`;
    if (truth) { const t = truth[r.id].trueLiveLift; s += `<rect x="${x(i) - 3.2}" y="${y(t) - 3.2}" width="6.4" height="6.4" fill="#3d7a5a" transform="rotate(45 ${x(i)} ${y(t)})"/>`; }
    s += `</g>`;
    if (i % 3 === 0) s += `<text x="${x(i)}" y="${Hh - 10}" text-anchor="middle">${r.id.slice(-4)}</text>`;
  });
  svg.innerHTML = s;
  svg.querySelectorAll('.pt').forEach((g) => {
    const r = res.rows.find((q) => q.id === g.dataset.id);
    g.addEventListener('mousemove', (e) => tip(e, r.id, [['Family', r.family], ['Offline (DR)', pct(r.lift) + ' ± ' + (100 * r.se).toFixed(2)], ['Forecast readout', pct(r.fc.center)], ['80% band', `${pp(r.fc.lo)} to ${pp(r.fc.hi)}`], ['P(significant win)', Math.round(100 * r.pWin) + '%'], ...(truth ? [['Simulated true live', pct(truth[r.id].trueLiveLift)]] : [])]));
    g.addEventListener('mouseleave', hideTip);
    g.addEventListener('click', () => { selected = r.id; render(); });
  });
}

// ---------- chart 2: calibration scatter ----------
function drawCh2() {
  const svg = $('ch2'); const W = svg.clientWidth || 420, Hh = 300;
  const H = res.H, cal = res.cal;
  const pts = H.rows.map((r) => ({ f: r.loo.center, raw: r.raw, live: r.live, r }));
  const vals = pts.flatMap((p) => [p.f, p.raw, p.live]);
  const step = 0.005;
  const lo = Math.min(0, Math.floor(Math.min(...vals) / step) * step), hi = Math.ceil(Math.max(...vals) / step) * step;
  const L = 40, R = 12, T = 12, B = 34;
  const x = (v) => L + (v - lo) / (hi - lo) * (W - L - R);
  const y = (v) => T + (hi - v) / (hi - lo) * (Hh - T - B);
  let s = '';
  for (let v = lo; v <= hi + 1e-9; v += step) {
    s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="rgba(34,34,34,0.07)" stroke-dasharray="3 4"/><text x="${L - 7}" y="${y(v) + 3}" text-anchor="end">${(100 * v).toFixed(1)}</text>`;
    s += `<text x="${x(v)}" y="${Hh - 18}" text-anchor="middle">${(100 * v).toFixed(1)}</text>`;
  }
  s += `<text x="${(L + W - R) / 2}" y="${Hh - 3}" text-anchor="middle">predicted lift at promotion (%)</text>`;
  s += `<text x="12" y="${(T + Hh - B) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(T + Hh - B) / 2})">live readout (%)</text>`;
  s += `<line x1="${x(lo)}" y1="${y(lo)}" x2="${x(hi)}" y2="${y(hi)}" stroke="#cfcfcf" stroke-width="1.5"/>`;
  pts.forEach((p) => {
    s += `<line x1="${x(p.raw)}" y1="${y(p.live)}" x2="${x(p.f)}" y2="${y(p.live)}" stroke="rgba(243,127,148,0.25)"/>`;
  });
  pts.forEach((p) => {
    s += `<g class="cp"><circle cx="${x(p.raw)}" cy="${y(p.live)}" r="4" fill="#fff" stroke="#f37f94" stroke-width="1.8"/>`;
    s += `<circle cx="${x(p.f)}" cy="${y(p.live)}" r="4.6" fill="#709390"/></g>`;
  });
  svg.innerHTML = s;
  svg.querySelectorAll('.cp').forEach((g, i) => {
    const p = pts[i];
    g.addEventListener('mousemove', (e) => tip(e, `Session ${p.r.session} · promoted`, [['Raw offline', pct(p.raw)], ['Forecast (held out)', pct(p.f)], ['80% band', `${pp(p.r.loo.lo)} to ${pp(p.r.loo.hi)}`], ['Live readout', pct(p.live) + ' ± ' + (100 * p.r.liveSe).toFixed(2)], ['Test length', p.r.days + ' days']]));
    g.addEventListener('mouseleave', hideTip);
  });
}

// ---------- chart 3: prior density ----------
function drawCh3() {
  const svg = $('ch3'); const W = svg.clientWidth || 260, Hh = 120;
  const prior = res.opts.pooling === 'indep' ? res.priorIndep : res.prior;
  const lo = -0.03, hi = 0.025, n = 160;
  const xs = Array.from({ length: n }, (_, i) => lo + (hi - lo) * i / (n - 1));
  const d = F.priorDensity(prior, xs);
  const mx = Math.max(...d);
  const x = (v) => 6 + (v - lo) / (hi - lo) * (W - 12), y = (v) => 10 + (1 - v / mx) * (Hh - 34);
  let path = `M${x(lo)},${y(0)}`; xs.forEach((v, i) => { path += `L${x(v)},${y(d[i])}`; }); path += `L${x(hi)},${y(0)}Z`;
  let s = `<path d="${path}" fill="rgba(112,147,144,0.22)" stroke="#709390" stroke-width="1.5"/>`;
  s += `<line x1="${x(0)}" x2="${x(0)}" y1="6" y2="${Hh - 24}" stroke="#bdbdbd"/>`;
  [-0.03, -0.02, -0.01, 0, 0.01, 0.02].forEach((v) => { s += `<text x="${x(v)}" y="${Hh - 8}" text-anchor="middle">${(100 * v).toFixed(0)}%</text>`; });
  svg.innerHTML = s;
}

// ---------- table ----------
function renderTable() {
  const list0 = sortBy === 'fc' ? res.byWin : res.byRaw;
  const list = showAll ? list0 : list0.slice(0, 14);
  const mn = -0.01, mx = 0.03;
  const X = (v) => Math.max(0, Math.min(100, (v - mn) / (mx - mn) * 100));
  $('tbody').innerHTML = list.map((r, i) => {
    const rec = res.recommended.includes(r);
    const mv = r.rawRank - r.fcRank;
    const dd = !isFinite(r.days80) ? '—' : r.days80 > 90 ? '> 90' : r.days80.toFixed(1);
    const was = sortBy === 'fc' ? (mv === 0 ? 'same as offline' : `offline #${r.rawRank}`) : `forecast #${r.fcRank}`;
    return `<tr class="row ${r.id === selected ? 'sel' : ''}" data-id="${r.id}">
      <td class="rk">${sortBy === 'fc' ? r.fcRank : r.rawRank}<small class="${sortBy === 'fc' ? (mv > 0 ? 'up' : mv < 0 ? 'dn' : '') : ''}">${was}</small></td>
      <td class="xp"><div class="id">${r.id}${rec ? '<span class="tag rec">Rec</span>' : ''}${r.aggressive ? '<span class="tag agg">Deep</span>' : ''}</div><div class="fm">${r.family}</div></td>
      <td class="num ${r.ndcgDelta >= 0 ? 'pos' : 'neg'}">${pct(r.ndcgDelta, 1)}</td>
      <td class="num">${pct(r.lift)}<small style="display:block;color:#999;font-size:10.5px">± ${(100 * r.se).toFixed(2)}</small></td>
      <td><div style="display:flex;align-items:center;gap:12px"><div class="ib"><div class="axis"></div><div class="zero" style="left:${X(0)}%"></div><div class="bar" style="left:${X(r.fc.lo)}%;width:${X(r.fc.hi) - X(r.fc.lo)}%"></div><div class="raw" style="left:${X(r.lift)}%"></div><div class="c" style="left:${X(r.fc.center)}%"></div></div><span class="fcn">${pct(r.fc.center)}<small>${pp(r.fc.lo)} to ${pp(r.fc.hi)}</small></span></div></td>
      <td class="num"><div class="pw"><div class="tr"><i style="width:${Math.round(100 * r.pWin)}%"></i></div>${Math.round(100 * r.pWin)}%</div></td>
      <td class="num">${dd}</td>
      <td><span class="cb ${promote.has(r.id) ? 'on' : ''}" data-cb="${r.id}"></span></td></tr>`;
  }).join('');
  const changed = res.byWin.slice(0, 14).filter((r) => r.rawRank > 14).length;
  $('tblsub').textContent = `Ranked by the probability that a ${res.opts.days}-day test at ${Math.round(res.opts.armShare * 100)}% per arm reads out as a significant win. ${changed} of the top 14 by P(win) are outside the offline top 14.`;
  $('showall').textContent = showAll ? "Show top 14" : `Show all ${res.rows.length}`;
  $('morenote').textContent = `Harness ΔNDCG@6 replays logged purchases; it moves in the opposite direction to revenue for ${res.rows.filter((r) => Math.sign(r.ndcgDelta) !== Math.sign(r.cur.mean) && Math.abs(r.cur.mean) > 0.001).length} experiments.`;
  document.querySelectorAll('tr.row').forEach((tr) => tr.addEventListener('click', (e) => {
    const id = tr.dataset.id;
    if (e.target.dataset.cb) { promote.has(id) ? promote.delete(id) : promote.add(id); renderTable(); renderPlan(); return; }
    selected = id; render();
  }));
}

function renderPlan() {
  const o = res.opts;
  $('armv').textContent = `${Math.round(o.armShare * 100)}% · ${num(Math.round(res.daily * o.armShare))}/day`;
  document.querySelectorAll('#dur button').forEach((b) => b.classList.toggle('on', +b.dataset.v === o.days));
  document.querySelectorAll('#slots button').forEach((b) => b.classList.toggle('on', +b.dataset.v === o.slots));
  const sel = res.rows.filter((r) => promote.has(r.id));
  const ew = sel.reduce((a, r) => a + r.pWin, 0);
  const best = sel.length ? sel.reduce((a, r) => (r.fc.center > a.fc.center ? r : a)) : null;
  $('plansum').innerHTML = `
    <div class="r"><span>Readout SE (one test)</span><b>± ${(100 * res.abSe).toFixed(2)} pp</b></div>
    <div class="r"><span>Lift detectable (80% power)</span><b>${pp(2.8 * res.abSe)} pp</b></div>
    <div class="r"><span>Selected for live tests</span><b>${sel.length}</b></div>
    <div class="r"><span>Expected significant wins</span><b>${ew.toFixed(2)} of ${sel.length}</b></div>
    <div class="r"><span>Best selected forecast</span><b>${best ? pct(best.fc.center) : '—'}</b></div>`;
  $('promote').textContent = `Promote ${sel.length} to live A/B`;
  $('promote2').textContent = `Promote ${sel.length} selected`;
  $('promote').disabled = $('promote2').disabled = sel.length === 0;
}

function renderPool() {
  const o = res.opts;
  document.querySelectorAll('#pool button').forEach((b) => b.classList.toggle('on', b.dataset.v === o.pooling));
  const f = res.factor;
  $('poolsub').textContent = `All ${res.rows.length} experiments are scored on the same logged page views, so their errors move together: one shared factor explains ${Math.round(100 * f.share)}% of the error variance.`;
  const prior = o.pooling === 'indep' ? res.priorIndep : res.prior;
  const above = prior.grid.reduce((a, g, i) => a + prior.w[i] * F.Phi(g / prior.h), 0);
  $('poolnote').innerHTML = `Estimated distribution of true offline lifts in this session (smoothed NPMLE): <b>${Math.round(100 * above)}%</b> of experiments beat production. ${o.pooling === 'indep' ? 'Treating errors as independent makes this distribution too narrow, so top forecasts are pulled down too far.' : 'Pooling uses each experiment\'s idiosyncratic error; the shared part stays in its forecast band.'}`;
}

function renderDrawer() {
  const r = res.rows.find((q) => q.id === selected);
  const F_ = data.meta.features;
  const maxw = Math.max(...r.w.map(Math.abs), 1);
  const wb = r.w.map((w, f) => `<div class="wbar"><span>${F_[f]}</span><div class="t"><i style="left:${w >= 0 ? 50 : 50 - 50 * Math.abs(w) / maxw}%;width:${50 * Math.abs(w) / maxw}%;background:${f === 0 ? '#709390' : w >= 0 ? '#064f6e' : '#f37f94'}"></i><i style="left:50%;width:1px;background:#ccc"></i></div><span class="n">${w.toFixed(2)}</span></div>`).join('');
  // the highest-traffic page context this experiment changes
  const { feats, views, ids } = logs; const prod = res.pre.prod;
  let best = -1, bv = -1, bt = null;
  for (let c = 0; c < logs.C; c++) {
    const t = F.topK(feats, r.w, c * 72);
    let diff = false; for (let k = 0; k < 6; k++) if (t[k] !== prod[c * 6 + k]) diff = true;
    if (diff && views[c] > bv) { bv = views[c]; best = c; bt = t; }
  }
  const item = (c, j) => data.catalog[ids[c * 12 + j]];
  const ps = Array.from({ length: 6 }, (_, k) => prod[best * 6 + k]);
  const li = (c, js, other) => js.map((j, k) => { const it = item(c, j); return `<li class="${other && !other.includes(j) ? 'new' : ''}"><span class="p">${k + 1}</span><span>${it.n}</span><span class="pr">$${Math.round(it.p).toLocaleString()}</span></li>`; }).join('');
  const intent = CATS[data.contexts.intent[best]][0];
  const dev = data.contexts.device[best] ? 'mobile' : 'desktop';
  $('drawer').innerHTML = `<div class="dg">
    <div><h4>${r.id} · estimate</h4>
      <div class="kv"><span>Offline lift, doubly robust</span><b>${pct(r.lift)} ± ${(100 * r.se).toFixed(2)}</b></div>
      <div class="kv"><span>Offline lift, IPS</span><b>${pct(r.ipsLift)} ± ${(100 * r.ipsSe).toFixed(2)}</b></div>
      <div class="kv"><span>Shared / own error SD</span><b>${(100 * Math.sqrt(Math.max(r.se ** 2 - r.seIdio ** 2, 0))).toFixed(2)} / ${(100 * r.seIdio).toFixed(2)} pp</b></div>
      <div class="kv"><span>Pooled offline lift</span><b>${pct(r.cur.mean)} ± ${(100 * r.cur.sd).toFixed(2)}</b></div>
      <div class="kv"><span>P(true lift > 0)</span><b>${r.cur.pPos > 0.995 ? '> 99%' : r.cur.pPos < 0.005 ? '< 1%' : Math.round(100 * r.cur.pPos) + '%'}</b></div>
      <div class="kv"><span>Forecast readout (× ${res.cal.b.toFixed(2)})</span><b>${pct(r.fc.center)}</b></div>
      <div class="kv"><span>Harness ΔNDCG@6</span><b>${pct(r.ndcgDelta, 1)}</b></div></div>
    <div><h4>Agent hypothesis · scorer weights</h4><div style="font-family:var(--serif);font-size:14px;margin:-2px 0 12px">${r.family}${r.aggressive ? ' <span class="tag agg">Deep re-rank</span>' : ''}</div>${wb}
      <div class="note">Scores = weights × features; top 6 of 12 retrieved items are shown. ${r.slotsChanged.toFixed(2)} of 6 slots differ from production on an average page view.</div></div>
    <div><h4>Largest page it changes</h4><div class="ctx">${intent} intent · ${dev} · ${num(views[best])} logged views</div>
      <div class="slates"><div class="slate"><div class="hd">Production</div><ol>${li(best, ps)}</ol></div><div class="slate"><div class="hd">${r.id}</div><ol>${li(best, bt, ps)}</ol></div></div></div>
  </div>`;
}

// ---------- simulation check ----------
function reveal() {
  if (truth) return;
  const s = document.createElement('script');
  s.src = 'data/truth.js';
  s.onload = () => { truth = Object.fromEntries(window.SEQ_TRUTH.candidates.map((c) => [c.id, c])); $('lgTruth').classList.remove('hide'); render(); };
  document.body.appendChild(s);
}
function renderTruth() {
  const rows = res.rows;
  const r0 = rng(20261110); // seeded simulated readouts of the configured test
  let coverRead = 0, coverTrue = 0;
  for (const r of rows) {
    const t = truth[r.id].trueLiveLift;
    const readout = t + res.abSe * r0.normal();
    if (readout >= r.fc.lo && readout <= r.fc.hi) coverRead++;
    if (t >= r.fc.lo && t <= r.fc.hi) coverTrue++;
  }
  const top = res.byRaw.slice(0, 10);
  const rawBias = top.reduce((a, r) => a + r.lift - truth[r.id].trueOfflineLift, 0) / 10;
  const fcBias = top.reduce((a, r) => a + r.cur.mean - truth[r.id].trueOfflineLift, 0) / 10;
  const avg = (l) => l.reduce((a, r) => a + truth[r.id].trueLiveLift, 0) / l.length;
  const same = res.recommended.every((r) => res.naive.includes(r));
  const k = res.opts.slots;
  const pick = same ? `The ${k} recommended experiments are also the offline top ${k} here (true live lift ${pct(avg(res.recommended))}).` : `True live lift of the ${k} recommended: <b>${pct(avg(res.recommended))}</b> vs. offline top ${k}: <b>${pct(avg(res.naive))}</b>.`;
  $('simtxt').innerHTML = `Simulation check (synthetic ground truth): forecast bands contain a simulated ${res.opts.days}-day readout for <b>${coverRead}/${rows.length}</b> experiments (target 80%) and the noise-free true live lift for ${coverTrue}/${rows.length}. For the offline top 10, raw estimates overstate the true offline lift by <b>${pp(rawBias)} pp</b>; pooled estimates are off by <b>${pp(fcBias)} pp</b>. ${pick}`;
  $('reveal').textContent = 'Truth shown';
  $('reveal').disabled = true;
}

// ---------- tooltip ----------
function tip(e, title, rows) {
  const t = $('tip');
  t.innerHTML = `<div class="t">${title}</div>` + rows.map(([k, v]) => `<div class="r"><span>${k}</span><b>${v}</b></div>`).join('');
  t.style.display = 'block';
  const x = Math.min(e.clientX + 14, window.innerWidth - t.offsetWidth - 10);
  t.style.left = x + 'px'; t.style.top = (e.clientY + 14) + 'px';
}
function hideTip() { $('tip').style.display = 'none'; }

function replan(o) {
  P.plan(res, o);
  promote.clear(); res.recommended.forEach((r) => promote.add(r.id));
  render();
}

function wire() {
  $('arm').addEventListener('input', (e) => replan({ armShare: +e.target.value / 100 }));
  document.querySelectorAll('#dur button').forEach((b) => b.addEventListener('click', () => replan({ days: +b.dataset.v })));
  document.querySelectorAll('#slots button').forEach((b) => b.addEventListener('click', () => replan({ slots: +b.dataset.v })));
  document.querySelectorAll('#pool button').forEach((b) => b.addEventListener('click', () => replan({ pooling: b.dataset.v })));
  document.querySelectorAll('#sortseg button').forEach((b) => b.addEventListener('click', () => { sortBy = b.dataset.v; document.querySelectorAll('#sortseg button').forEach((q) => q.classList.toggle('on', q === b)); renderTable(); }));
  document.querySelectorAll('#ch1seg button').forEach((b) => b.addEventListener('click', () => { ch1mode = b.dataset.v; document.querySelectorAll('#ch1seg button').forEach((q) => q.classList.toggle('on', q === b)); drawCh1(); }));
  $('showall').addEventListener('click', () => { showAll = !showAll; renderTable(); });
  $('reveal').addEventListener('click', reveal);
  const toast = () => { const ids = [...promote]; alert(`Concept demo: ${ids.length} experiment(s) would be queued for live A/B on ${data.meta.surface}:\n${ids.join('\n')}`); };
  $('promote').addEventListener('click', toast); $('promote2').addEventListener('click', toast);
  $('export').addEventListener('click', () => {
    const head = 'experiment,family,offline_dr_lift,se,pooled_lift,forecast,band_lo,band_hi,p_win,days_to_80pct,offline_rank,forecast_rank\n';
    const body = res.byWin.map((r) => [r.id, r.family, r.lift, r.se, r.cur.mean, r.fc.center, r.fc.lo, r.fc.hi, r.pWin, isFinite(r.days80) ? r.days80 : '', r.rawRank, r.fcRank].join(',')).join('\n');
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([head + body], { type: 'text/csv' })); a.download = 'session-418-live-lift-forecast.csv'; a.click();
  });
  window.addEventListener('resize', () => { drawCh1(); drawCh2(); drawCh3(); });
}

main().catch((e) => { $('loadmsg').textContent = 'Error: ' + e.message; console.error(e); });
