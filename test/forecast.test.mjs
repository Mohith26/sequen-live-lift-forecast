import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as S from '../core/sim.mjs';
import * as F from '../core/forecast.mjs';
import * as P from '../core/pipeline.mjs';

const world = S.makeWorld({ seed: 418, C: 600 });
const cands = S.makeCandidates({ N: 40 }).slice(0, 10);
const v0 = S.trueValue(world, S.PROD_W);
const truth = cands.map((c) => S.trueValue(world, c.w) / v0 - 1);

test('topK returns the K highest-scoring distinct items', () => {
  const feats = world.feats;
  const t = S.topK(feats, [0, 1, 0, 0, 0, 0], 0);
  assert.equal(new Set(t).size, S.K);
  const score = (j) => feats[j * S.D + 1];
  const rest = [...Array(S.M).keys()].filter((j) => !t.includes(j));
  for (const j of rest) assert.ok(score(j) <= score(t[S.K - 1]));
});

test('logging propensities sum to 1 over items at every slot', () => {
  const logs = S.simulateLogs(world, { N: 50000, seed: 1 });
  const prod = F.prodSlates(logs);
  for (let c = 0; c < 50; c++) for (let k = 0; k < S.K; k++) {
    let s = 0; for (let j = 0; j < S.M; j++) s += F.propensity(logs.eps, prod[c * S.K + k] === j);
    assert.ok(Math.abs(s - 1) < 1e-12);
  }
});

test('DR and IPS lifts are unbiased with calibrated SEs across independent logs', () => {
  const zDR = [], zIPS = [];
  for (let s = 0; s < 14; s++) {
    const logs = S.simulateLogs(world, { seed: 7000 + s, N: 1500000 });
    const pre = F.prepare(logs, F.fitRewardModel(logs, { iters: 120 }));
    cands.forEach((c, i) => { const e = F.estimateLift(pre, c.w); zDR.push((e.lift - truth[i]) / e.se); zIPS.push((e.ipsLift - truth[i]) / e.ipsSe); });
  }
  for (const z of [zDR, zIPS]) {
    const m = z.reduce((a, b) => a + b, 0) / z.length;
    const rms = Math.sqrt(z.reduce((a, b) => a + b * b, 0) / z.length);
    assert.ok(Math.abs(m) < 0.25, `mean z ${m}`);
    assert.ok(rms > 0.8 && rms < 1.25, `rms z ${rms}`);
    const cover = z.filter((x) => Math.abs(x) < 1.645).length / z.length;
    assert.ok(cover > 0.82 && cover < 0.97, `90% coverage ${cover}`);
  }
});

test('covariance diagonal equals the per-experiment variance; off-diagonals match replication', () => {
  const logs = S.simulateLogs(world, { seed: 9, N: 1500000 });
  const pre = F.prepare(logs, F.fitRewardModel(logs, { iters: 100 }));
  const ws = cands.map((c) => c.w);
  const cov = F.liftCovariance(pre, ws);
  ws.forEach((w, i) => { const e = F.estimateLift(pre, w); assert.ok(Math.abs(cov[i * ws.length + i] - e.se ** 2) / e.se ** 2 < 1e-9); });
  // replication: empirical correlation of two related experiments vs predicted
  const n = ws.length;
  let best = [0, 1], bc = -1;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const r = cov[i * n + j] / Math.sqrt(cov[i * n + i] * cov[j * n + j]); if (r > bc) { bc = r; best = [i, j]; } }
  const [a, b] = best, xa = [], xb = [];
  for (let s = 0; s < 30; s++) {
    const L = S.simulateLogs(world, { seed: 8000 + s, N: 1500000 });
    const pr = F.prepare(L, F.fitRewardModel(L, { iters: 60 }));
    xa.push(F.estimateLift(pr, ws[a]).lift); xb.push(F.estimateLift(pr, ws[b]).lift);
  }
  const mean = (v) => v.reduce((p, q) => p + q, 0) / v.length;
  const ma = mean(xa), mb = mean(xb);
  const emp = xa.reduce((p, q, i) => p + (q - ma) * (xb[i] - mb), 0) / Math.sqrt(xa.reduce((p, q) => p + (q - ma) ** 2, 0) * xb.reduce((p, q) => p + (q - mb) ** 2, 0));
  assert.ok(Math.abs(emp - bc) < 0.3, `predicted corr ${bc.toFixed(2)} vs replicated ${emp.toFixed(2)}`);
});

function synthMeans(seed, n, factorShare) {
  const r = S.rng(seed);
  const theta = [], est = [], se = [], lambda = [];
  const f = r.normal();
  for (let i = 0; i < n; i++) {
    const t = r() < 0.8 ? 0.002 * r.normal() - 0.002 : 0.004 + 0.003 * r.normal();
    const s = 0.003 + 0.003 * r();
    const l = Math.sqrt(factorShare) * s;
    theta.push(t); se.push(s); lambda.push(l);
    est.push(t + l * f + Math.sqrt(s * s - l * l) * r.normal());
  }
  return { theta, est, se, lambda };
}

test('empirical Bayes beats raw estimates and removes most of the winner\'s curse (independent errors)', () => {
  let rawMse = 0, ebMse = 0, rawTop = 0, ebTop = 0;
  for (let s = 0; s < 8; s++) {
    const { theta, est, se } = synthMeans(100 + s, 240, 0);
    const prior = F.npmle(est, se);
    const post = est.map((x, i) => F.posterior(prior, x, se[i]).mean);
    est.forEach((x, i) => { rawMse += (x - theta[i]) ** 2; ebMse += (post[i] - theta[i]) ** 2; });
    const top = est.map((_, i) => i).sort((a, b) => est[b] - est[a]).slice(0, 10);
    top.forEach((i) => { rawTop += est[i] - theta[i]; ebTop += post[i] - theta[i]; });
  }
  assert.ok(ebMse < 0.75 * rawMse, `eb ${ebMse} raw ${rawMse}`);
  assert.ok(rawTop / 80 > 0.002, `raw top-10 bias ${rawTop / 80}`);
  assert.ok(Math.abs(ebTop / 80) < 0.4 * rawTop / 80, `eb top-10 bias ${ebTop / 80}`);
});

test('with shared errors, factor-aware pooling is better calibrated than assuming independence', () => {
  let indepHit = 0, factHit = 0, n = 0;
  for (let s = 0; s < 30; s++) {
    const { theta, est, se, lambda } = synthMeans(300 + s, 240, 0.35);
    const pI = F.npmle(est, se, { iters: 300 });
    const { idio, shared } = F.splitSE(se, lambda);
    const pF = F.npmle(est, idio, { iters: 300 });
    est.forEach((x, i) => {
      const a = F.posterior(pI, x, se[i]); const b = F.posterior(pF, x, idio[i], shared[i]);
      if (Math.abs(a.mean - theta[i]) < 1.2816 * a.sd) indepHit++;
      if (Math.abs(b.mean - theta[i]) < 1.2816 * b.sd) factHit++;
      n++;
    });
  }
  assert.ok(Math.abs(factHit / n - 0.8) < 0.06, `factor coverage ${factHit / n}`);
  assert.ok(Math.abs(factHit / n - 0.8) < Math.abs(indepHit / n - 0.8), `indep ${indepHit / n} vs factor ${factHit / n}`);
});

test('calibration recovers the online slope and conformal intervals reach nominal coverage', () => {
  const r = S.rng(55);
  const mk = (n) => Array.from({ length: n }, () => {
    const m = 0.004 + 0.006 * r(), v = (0.002 + 0.002 * r()) ** 2, liveSe = 0.001 + 0.002 * r();
    const L = m + Math.sqrt(v) * r.normal();
    return { m, v, liveSe, live: 0.8 * L + 0.002 * r.normal() + liveSe * r.normal() };
  });
  const train = mk(400), testRows = mk(4000);
  const cal = F.calibrate(train, { alpha: 0.2 });
  assert.ok(Math.abs(cal.b - 0.8) < 0.12, `slope ${cal.b}`);
  const hit = testRows.filter((t) => { const f = F.forecast(cal, t.m, t.v, t.liveSe); return t.live >= f.lo && t.live <= f.hi; }).length / testRows.length;
  assert.ok(hit > 0.76 && hit < 0.86, `coverage ${hit}`);
});

test('shipped dataset: page numbers are internally consistent and pass the simulation check', () => {
  const src = readFileSync(new URL('../data/logs.js', import.meta.url), 'utf8');
  const data = JSON.parse(src.slice(src.indexOf('{'), src.lastIndexOf('}') + 1));
  const T = JSON.parse(readFileSync(new URL('../data/truth.json', import.meta.url), 'utf8'));
  const res = P.run(data, P.decode(data));
  assert.equal(res.rows.length, 240);
  for (const r of res.rows) {
    assert.ok(Number.isFinite(r.lift) && Number.isFinite(r.fc.center) && r.pWin >= 0 && r.pWin <= 1);
    assert.ok(Math.abs(r.fc.center - res.hist.cal.b * r.post.mean) < 1e-12);
    assert.ok(r.fc.lo <= r.fc.center && r.fc.center <= r.fc.hi);
  }
  assert.equal(res.recommended.length, 3);
  const truthById = Object.fromEntries(T.candidates.map((c) => [c.id, c]));
  const cover = res.rows.filter((r) => { const t = truthById[r.id].trueLiveLift; return t >= r.fc.lo && t <= r.fc.hi; }).length / res.rows.length;
  assert.ok(cover > 0.75, `forecast coverage of true live lift ${cover}`);
  const top = [...res.rows].sort((a, b) => b.lift - a.lift).slice(0, 10);
  const raw = top.reduce((a, r) => a + r.lift - truthById[r.id].trueOfflineLift, 0) / 10;
  const eb = top.reduce((a, r) => a + r.post.mean - truthById[r.id].trueOfflineLift, 0) / 10;
  assert.ok(Math.abs(eb) < Math.abs(raw), `top-10 bias raw ${raw} eb ${eb}`);
});

test('position-bias curve of the reward model recovers the true examination probabilities', () => {
  const logs = S.simulateLogs(world, { seed: 31, N: 3000000 });
  const m = F.fitRewardModel(logs, { iters: 50 });
  m.theta.forEach((t, k) => assert.ok(Math.abs(t - S.THETA[k]) < 0.05, `slot ${k + 1}: ${t} vs ${S.THETA[k]}`));
});
