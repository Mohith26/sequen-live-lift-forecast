// End-to-end Live Lift Forecast for one RRI session. Used by the page and by
// scripts/report.mjs, so every number on the page comes from this code path.
import { M, K, D } from './sim.mjs';
import * as F from './forecast.mjs';

function fromB64(s, Ctor) {
  let bytes;
  if (typeof Buffer !== 'undefined') bytes = Uint8Array.from(Buffer.from(s, 'base64'));
  else { const bin = atob(s); bytes = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i); }
  return new Ctor(bytes.buffer, 0, bytes.byteLength / Ctor.BYTES_PER_ELEMENT);
}

export function decode(data) {
  const { meta, b64, catalog } = data;
  const ids = fromB64(b64.ids, Uint16Array);
  const price = new Float64Array(ids.length);
  for (let i = 0; i < ids.length; i++) price[i] = catalog[ids[i]].p;
  return {
    C: meta.C, eps: meta.eps,
    feats: fromB64(b64.feats, Int8Array), ids, price,
    views: fromB64(b64.views, Uint32Array),
    cellN: Uint32Array.from(fromB64(b64.cellN, Uint16Array)),
    cellY: Uint32Array.from(fromB64(b64.cellY, Uint16Array)),
  };
}

// Past sessions: recompute each session's prior, the promoted experiments'
// posteriors, then fit the offline->online calibration.
export function analyzeHistory(history, { alpha = 0.2, pooling = 'factor' } = {}) {
  const rows = [];
  for (const h of history) {
    const { idio, shared } = pooling === 'factor' ? F.splitSE(h.se, h.lambda) : { idio: h.se, shared: h.se.map(() => 0) };
    const prior = F.npmle(h.est, idio);
    for (const t of h.tests) {
      const p = F.posterior(prior, h.est[t.index], idio[t.index], shared[t.index]);
      rows.push({ session: h.session, raw: h.est[t.index], rawSe: h.se[t.index], m: p.mean, v: p.sd * p.sd, live: t.live, liveSe: t.liveSe, days: t.days });
    }
  }
  const cal = F.calibrate(rows, { alpha });
  // leave-one-out coverage of the conformal intervals and of naive raw intervals
  const z = 1.2816; // two-sided 80% normal quantile for the naive interval
  let looHit = 0, rawHit = 0;
  rows.forEach((r, i) => {
    const c = F.calibrate(rows.filter((_, j) => j !== i), { alpha });
    const f = F.forecast(c, r.m, r.v, r.liveSe);
    r.loo = f;
    if (r.live >= f.lo && r.live <= f.hi) looHit++;
    const half = z * Math.sqrt(r.rawSe ** 2 + r.liveSe ** 2);
    r.rawLo = r.raw - half; r.rawHi = r.raw + half;
    if (r.live >= r.rawLo && r.live <= r.rawHi) rawHit++;
  });
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  return {
    rows, cal,
    looCoverage: looHit / rows.length, rawCoverage: rawHit / rows.length,
    meanRaw: mean(rows.map((r) => r.raw)), meanLive: mean(rows.map((r) => r.live)),
    meanForecast: mean(rows.map((r) => cal.b * r.m)),
    winsLive: rows.filter((r) => r.live > 1.96 * r.liveSe).length,
  };
}

export function run(data, logs, opts = {}) {
  const o = { alpha: 0.2, armShare: data.meta.abArmShare, days: data.meta.abDays, slots: 3, ...opts };
  const model = F.fitRewardModel(logs);
  const pre = F.prepare(logs, model);
  const rev = F.revenueStats(pre);
  const totalViews = pre.Ntot;
  const daily = totalViews / data.meta.days / data.meta.logShare;
  const rows = data.candidates.map((c) => ({ ...c, ...F.estimateLift(pre, c.w) }));
  const cov = F.liftCovariance(pre, data.candidates.map((c) => c.w));
  const factor = F.oneFactor(cov, rows.length);
  const { idio, shared } = F.splitSE(rows.map((r) => r.se), factor.lambda);
  const prior = F.npmle(rows.map((r) => r.lift), idio);
  rows.forEach((r, i) => { r.seIdio = idio[i]; r.post = F.posterior(prior, r.lift, idio[i], shared[i]); });
  // the same pooling assuming independent errors, for comparison
  const priorIndep = F.npmle(rows.map((r) => r.lift), rows.map((r) => r.se));
  rows.forEach((r) => { r.postIndep = F.posterior(priorIndep, r.lift, r.se); });
  const hist = analyzeHistory(data.history, { alpha: o.alpha });
  const histIndep = analyzeHistory(data.history, { alpha: o.alpha, pooling: 'indep' });
  const res = { model, pre, rev, totalViews, daily, rows, prior, priorIndep, factor, hist, histIndep, opts: { pooling: 'factor', ...o } };
  plan(res, o);
  return res;
}

// Planner (cheap; re-run when traffic share, duration or slots change).
export function plan(res, o) {
  res.opts = { ...res.opts, ...o };
  const { armShare, days, slots, pooling } = res.opts;
  const H = pooling === 'indep' ? res.histIndep : res.hist;
  const cal = H.cal;
  const perArm = res.daily * armShare * days;
  const se = F.abSe(res.rev.sdRel, perArm);
  for (const r of res.rows) {
    const post = pooling === 'indep' ? r.postIndep : r.post;
    r.cur = post;
    r.fc = F.forecast(cal, post.mean, post.sd ** 2, se);
    r.pWin = F.pWin(cal, post, se);
    r.days80 = F.daysTo80(cal, post.mean, res.rev.sdRel, res.daily * armShare);
  }
  res.H = H; res.cal = cal;
  const byRaw = [...res.rows].sort((a, b) => b.lift - a.lift);
  const byWin = [...res.rows].sort((a, b) => b.pWin - a.pWin || b.fc.center - a.fc.center);
  byRaw.forEach((r, i) => { r.rawRank = i + 1; });
  byWin.forEach((r, i) => { r.fcRank = i + 1; });
  res.abSe = se; res.perArm = perArm;
  res.recommended = byWin.slice(0, slots);
  res.naive = byRaw.slice(0, slots);
  res.byWin = byWin; res.byRaw = byRaw;
  return res;
}

export { M, K, D };
