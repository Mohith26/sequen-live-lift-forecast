// Live Lift Forecast: core statistics (no DOM). Pure functions.
//
// 1. Counterfactual revenue for each offline experiment from the exploration
//    bucket: item-position IPS and doubly robust (DR) estimators, paired
//    against production on the same page views, with delta-method SEs.
// 2. Empirical Bayes deconvolution (NPMLE on a grid, Koenker-Mizera / Jiang-
//    Zhang EM) across all experiments in the session: posterior lift, P(>0),
//    which removes the winner's curse of picking the max of hundreds.
// 3. Offline-to-online calibration from past live A/B tests: slope b (offline
//    lift that survives online), extra online variance tau^2, and a split-
//    conformal multiplier so forecast intervals for an A/B readout have
//    finite-sample coverage on exchangeable past tests.
// 4. Planner: P(significant win) and days to 80% power for each experiment.

import { M, K, D, SCALE, topK, PROD_W } from './sim.mjs';

// ---------- normal helpers ----------
export function phi(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
export function Phi(x) {
  // Abramowitz-Stegun 7.1.26 via erf, |err| < 1.5e-7
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2);
  return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

// ---------- logging policy ----------
// Exploration bucket: with prob eps a uniformly shuffled top-12, else production top-K.
export function propensity(eps, isProd) { return (1 - eps) * (isProd ? 1 : 0) + eps / M; }

// ---------- reward model (PBM): E[rev | item j at slot k] = theta_k * sigma(beta.x) * price ----------
export function fitRewardModel(logs, { iters = 300, lr = 0.05 } = {}) {
  const { C, feats, cellN, cellY } = logs;
  const prod = prodSlates(logs);
  // theta_k by IPS over all items at slot k: E[sum_j 1[j at k] y / rho(j,k)] = theta_k * sum_j p_j,
  // so ratios across slots are unbiased whatever the items' purchase rates.
  const cnt = new Float64Array(K), buy = new Float64Array(K);
  for (let c = 0; c < C; c++) for (let j = 0; j < M; j++) for (let k = 0; k < K; k++) {
    const cell = (c * M + j) * K + k;
    const rho = propensity(logs.eps, prod[c * K + k] === j);
    buy[k] += cellY[cell] / rho; cnt[k] += cellN[cell] / rho;
  }
  const rate = Array.from(cnt, (x, k) => buy[k] / Math.max(x, 1));
  const theta = rate.map((x) => x / rate[0]);
  // beta: binomial MLE of Y ~ Bin(N, theta_k * sigma(beta.x)) over all logged cells, Adam
  const cells = [];
  for (let c = 0; c < C; c++) for (let j = 0; j < M; j++) for (let k = 0; k < K; k++) {
    const cell = (c * M + j) * K + k; if (cellN[cell]) cells.push(cell);
  }
  const nc = cells.length, P = D + 1;
  const X = new Float32Array(nc * D), Nn = new Float32Array(nc), Yy = new Float32Array(nc), T = new Float32Array(nc);
  let totN = 0;
  cells.forEach((cell, i) => {
    const k = cell % K, cj = (cell - k) / K;
    for (let f = 0; f < D; f++) X[i * D + f] = feats[cj * D + f] * SCALE[f];
    Nn[i] = cellN[cell]; Yy[i] = cellY[cell]; T[i] = theta[k]; totN += cellN[cell];
  });
  const beta = new Float64Array(P);
  beta[0] = Math.log(rate[0] / (1 - rate[0]));
  const m1 = new Float64Array(P), m2 = new Float64Array(P), g = new Float64Array(P);
  for (let it = 1; it <= iters; it++) {
    g.fill(0);
    for (let i = 0; i < nc; i++) {
      let z = beta[0];
      for (let f = 0; f < D; f++) z += beta[f + 1] * X[i * D + f];
      const sg = 1 / (1 + Math.exp(-z)), q = T[i] * sg;
      const dz = Yy[i] * (1 - sg) - (Nn[i] - Yy[i]) * T[i] * sg * (1 - sg) / Math.max(1 - q, 1e-9);
      g[0] += dz;
      for (let f = 0; f < D; f++) g[f + 1] += dz * X[i * D + f];
    }
    for (let p = 0; p < P; p++) {
      const gp = g[p] / totN * 50;
      m1[p] = 0.9 * m1[p] + 0.1 * gp;
      m2[p] = 0.999 * m2[p] + 0.001 * gp * gp;
      const mh = m1[p] / (1 - 0.9 ** it), vh = m2[p] / (1 - 0.999 ** it);
      beta[p] += lr * mh / (Math.sqrt(vh) + 1e-8);
    }
  }
  return { theta, beta: Array.from(beta) };
}

export function prodSlates(logs) {
  const out = new Uint8Array(logs.C * K);
  for (let c = 0; c < logs.C; c++) topK(logs.feats, PROD_W, c * M * D).forEach((j, k) => { out[c * K + k] = j; });
  return out;
}

// Precompute per cell the DR / IPS sufficient statistics.
//   A = (R - N rhat)/rho   (DR residual sum)     B = sum over views of (rev - rhat)^2 / rho^2
//   Ai = R/rho             (IPS)                  Bi = sum rev^2 / rho^2
// Revenue of every purchase in a cell is the item's price, so sums of squares are exact.
export function prepare(logs, model) {
  const { C, eps, views, cellN, cellY, feats, price } = logs;
  const prod = prodSlates(logs);
  const nCell = C * M * K;
  const rhat = new Float64Array(nCell), A = new Float64Array(nCell), B = new Float64Array(nCell), Ai = new Float64Array(nCell), Bi = new Float64Array(nCell);
  for (let c = 0; c < C; c++) for (let j = 0; j < M; j++) {
    const b = c * M + j;
    let z = model.beta[0];
    for (let f = 0; f < D; f++) z += model.beta[f + 1] * feats[b * D + f] * SCALE[f];
    const pr = 1 / (1 + Math.exp(-z)), pz = price[b];
    for (let k = 0; k < K; k++) {
      const cell = b * K + k;
      const rh = model.theta[k] * pr * pz;
      rhat[cell] = rh;
      const rho = propensity(eps, prod[c * K + k] === j);
      const N = cellN[cell], Y = cellY[cell], R = Y * pz;
      A[cell] = (R - N * rh) / rho;
      B[cell] = (Y * pz * pz - 2 * rh * R + N * rh * rh) / (rho * rho);
      Ai[cell] = R / rho;
      Bi[cell] = Y * pz * pz / (rho * rho);
    }
  }
  let Ntot = 0, sumProd = 0, sumProdI = 0;
  const dmProd = new Float64Array(C);
  for (let c = 0; c < C; c++) {
    Ntot += views[c];
    let dm = 0, e = 0, ei = 0;
    for (let k = 0; k < K; k++) { const cell = (c * M + prod[c * K + k]) * K + k; dm += rhat[cell]; e += A[cell]; ei += Ai[cell]; }
    dmProd[c] = dm;
    sumProd += views[c] * dm + e; sumProdI += ei;
  }
  return { logs, prod, rhat, A, B, Ai, Bi, dmProd, Ntot, baseDR: sumProd / Ntot, baseIPS: sumProdI / Ntot };
}

// Offline lift of a scorer vs production on the same logged page views.
// DR and IPS relative lifts with standard errors (iid page views; cross-slot
// products inside one view are dropped, i.e. Poissonized cell counts), plus
// the harness replay metric (purchase NDCG@K on logged purchases).
export function estimateLift(pre, w) {
  const { logs, prod, rhat, A, B, Ai, Bi, dmProd, Ntot } = pre;
  const { C, views, feats, cellY } = logs;
  let sd = 0, sd2 = 0, si = 0, si2 = 0, changed = 0, gT = 0, gP = 0;
  for (let c = 0; c < C; c++) {
    const t = topK(feats, w, c * M * D);
    let dm = 0, E = 0, Bsum = 0, Ei = 0, Bis = 0, diff = 0;
    for (let k = 0; k < K; k++) {
      const j = t[k], pj = prod[c * K + k];
      const ct = (c * M + j) * K + k;
      dm += rhat[ct];
      if (j !== pj) {
        const cp = (c * M + pj) * K + k;
        E += A[ct] - A[cp]; Bsum += B[ct] + B[cp];
        Ei += Ai[ct] - Ai[cp]; Bis += Bi[ct] + Bi[cp];
        diff++;
      }
    }
    const dc = dm - dmProd[c];
    sd += views[c] * dc + E;
    sd2 += views[c] * dc * dc + 2 * dc * E + Bsum;
    si += Ei; si2 += Bis;
    changed += diff * views[c];
    // harness: logged purchases of each item, re-ranked by the target vs production
    for (let j = 0; j < M; j++) {
      let y = 0; for (let k = 0; k < K; k++) y += cellY[(c * M + j) * K + k];
      if (!y) continue;
      const rt = t.indexOf(j);
      let rp = -1; for (let k = 0; k < K; k++) if (prod[c * K + k] === j) rp = k;
      if (rt >= 0) gT += y / Math.log2(rt + 2);
      if (rp >= 0) gP += y / Math.log2(rp + 2);
    }
  }
  const N = Ntot;
  const md = sd / N, mi = si / N;
  const se = Math.sqrt(Math.max(sd2 / N - md * md, 0) / N);
  const sei = Math.sqrt(Math.max(si2 / N - mi * mi, 0) / N);
  return {
    lift: md / pre.baseDR, se: se / pre.baseDR,
    ipsLift: mi / pre.baseIPS, ipsSe: sei / pre.baseIPS,
    ndcgDelta: (gT - gP) / gP, slotsChanged: changed / N,
  };
}

// Production revenue per page view: mean and SD (for A/B power), from production cells.
export function revenueStats(pre) {
  const { logs, prod } = pre;
  let n = 0, s1 = 0, s2 = 0;
  for (let c = 0; c < logs.C; c++) {
    // views that got the production slate: the slot-1 production cell count minus
    // the (rare) explored views that reproduced production at slot 1 is close enough;
    // use slot-1 production cell impressions as the view count.
    const c1 = (c * M + prod[c * K]) * K;
    n += logs.cellN[c1];
    for (let k = 0; k < K; k++) {
      const cell = (c * M + prod[c * K + k]) * K + k;
      const pz = logs.price[c * M + prod[c * K + k]];
      // scale to per-production-view totals
      const scale = logs.cellN[c1] / Math.max(logs.cellN[cell], 1);
      s1 += logs.cellY[cell] * pz * scale; s2 += logs.cellY[cell] * pz * pz * scale;
    }
  }
  const mean = s1 / n;
  return { mean, sd: Math.sqrt(s2 / n - mean * mean), sdRel: Math.sqrt(s2 / n - mean * mean) / mean, views: n };
}

// ---------- empirical Bayes: smoothed NPMLE of the lift distribution ----------
// Prior G = sum_g w_g N(grid_g, h^2) (a grid NPMLE with Gaussian atoms of width h,
// i.e. a log-concave-free, smoothed Kiefer-Wolfowitz estimator). w by EM.
// h defaults to 0.4 x median SE; h = 0 gives the plain NPMLE.
export function npmle(est, se, { grid = 241, iters = 600, h } = {}) {
  const n = est.length;
  if (h === undefined) { const s = [...se].sort((x, y) => x - y); h = 0.4 * s[Math.floor(n / 2)]; }
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < n; i++) { mn = Math.min(mn, est[i]); mx = Math.max(mx, est[i]); }
  const G = Array.from({ length: grid }, (_, g) => mn + (mx - mn) * g / (grid - 1));
  let w = new Float64Array(grid).fill(1 / grid);
  const L = new Float64Array(n * grid);
  for (let i = 0; i < n; i++) { const s = Math.sqrt(se[i] ** 2 + h * h); for (let g = 0; g < grid; g++) L[i * grid + g] = phi((est[i] - G[g]) / s) / s; }
  let ll = 0;
  for (let it = 0; it < iters; it++) {
    const nw = new Float64Array(grid);
    ll = 0;
    for (let i = 0; i < n; i++) {
      let s = 0; for (let g = 0; g < grid; g++) s += w[g] * L[i * grid + g];
      ll += Math.log(s);
      for (let g = 0; g < grid; g++) nw[g] += w[g] * L[i * grid + g] / s;
    }
    for (let g = 0; g < grid; g++) nw[g] /= n;
    w = nw;
  }
  return { grid: G, w: Array.from(w), h, loglik: ll };
}

// Prior density on a grid of x values (for plotting).
export function priorDensity(prior, xs) {
  const { grid, w, h } = prior;
  return xs.map((x) => { let d = 0; for (let g = 0; g < grid.length; g++) d += w[g] * phi((x - grid[g]) / h) / h; return d; });
}

// Posterior of the true lift given estimate x with (idiosyncratic) SE s: a
// mixture of normals. extraVar adds the shared-error variance lambda^2 that
// pooling across experiments cannot remove (see oneFactor).
export function posterior(prior, x, s, extraVar = 0) {
  const { grid, w, h } = prior;
  const tot = Math.sqrt(s * s + h * h);
  const v = (h > 0 ? 1 / (1 / (h * h) + 1 / (s * s)) : 0) + extraVar;
  const pi = new Float64Array(grid.length), mu = new Float64Array(grid.length);
  let z = 0;
  for (let g = 0; g < grid.length; g++) {
    pi[g] = w[g] * phi((x - grid[g]) / tot);
    mu[g] = h > 0 ? (grid[g] / (h * h) + x / (s * s)) / (1 / (h * h) + 1 / (s * s)) : grid[g];
    z += pi[g];
  }
  let m = 0, m2 = 0, pos = 0;
  for (let g = 0; g < grid.length; g++) {
    pi[g] /= z; m += pi[g] * mu[g]; m2 += pi[g] * (v + mu[g] * mu[g]);
    pos += pi[g] * (v > 0 ? Phi(mu[g] / Math.sqrt(v)) : (mu[g] > 0 ? 1 : 0));
  }
  return { mean: m, sd: Math.sqrt(Math.max(m2 - m * m, 0)), pPos: pos, pi, mu, v };
}

// ---------- offline -> online calibration from past live tests ----------
// rows: { m, v, live, liveSe } where m, v are the EB posterior mean/var of the
// promoted experiment at promotion time and live/liveSe the A/B readout.
export function calibrate(rows, { alpha = 0.2 } = {}) {
  // slope b by weighted least squares through the origin, iterated with tau^2
  let b = 1, tau2 = 0;
  for (let it = 0; it < 50; it++) {
    let num = 0, den = 0;
    for (const r of rows) { const wt = 1 / (b * b * r.v + r.liveSe ** 2 + tau2); num += wt * r.m * r.live; den += wt * r.m * r.m; }
    b = num / den;
    // method-of-moments extra online variance (novelty, interference, drift)
    let acc = 0;
    for (const r of rows) acc += (r.live - b * r.m) ** 2 - b * b * r.v - r.liveSe ** 2;
    tau2 = Math.max(0, acc / rows.length);
  }
  const score = (r) => Math.abs(r.live - b * r.m) / Math.sqrt(b * b * r.v + r.liveSe ** 2 + tau2);
  const scores = rows.map(score).sort((x, y) => x - y);
  const n = scores.length;
  const kq = Math.ceil((n + 1) * (1 - alpha));
  const q = kq > n ? Infinity : scores[kq - 1];
  return { b, tau2, q, alpha, n, scores };
}

// Forecast for an A/B readout of an experiment with posterior (m, v).
export function forecast(cal, m, v, abSe) {
  const center = cal.b * m;
  const half = cal.q * Math.sqrt(cal.b * cal.b * v + abSe * abSe + cal.tau2);
  return { center, lo: center - half, hi: center + half };
}

// SE of relative lift for a two-arm test with `perArm` page views per arm.
export function abSe(sdRel, perArm) { return sdRel * Math.sqrt(2 / perArm); }

// P(the A/B readout is a significant win at two-sided 5%), integrating over the
// mixture posterior: readout X ~ N(b L, tau^2 + se^2), L ~ sum pi_g N(mu_g, v).
export function pWin(cal, post, se) {
  const sdx = Math.sqrt(cal.tau2 + se * se + cal.b * cal.b * post.v);
  let p = 0;
  for (let g = 0; g < post.pi.length; g++) p += post.pi[g] * Phi((cal.b * post.mu[g] - 1.96 * se) / sdx);
  return p;
}

export function daysTo80(cal, m, sdRel, perArmPerDay) {
  const eff = cal.b * m;
  if (eff <= 0) return Infinity;
  // need eff >= 2.8 * sdRel * sqrt(2 / N)  =>  N >= 2 * (2.8 sdRel / eff)^2
  const N = 2 * (2.8 * sdRel / eff) ** 2;
  return N / perArmPerDay;
}

export { topK, PROD_W };

// ---------- error covariance across experiments ----------
// All experiments are scored on the same logged page views, so their errors
// are correlated (they re-rank the same items in the same contexts). Under the
// same iid-view / Poissonized-cell approximation as estimateLift:
//   sum_s d_is d_js = sum_c [n_c dc_i dc_j + dc_i E_j + dc_j E_i] + sum_shared cells B
// Returns the covariance matrix of the relative lifts (row-major, n x n).
export function liftCovariance(pre, ws) {
  const { logs, prod, rhat, A, B, dmProd, Ntot } = pre;
  const { C, views, feats } = logs;
  const n = ws.length;
  const slates = new Uint8Array(n * C * K), dcs = new Float64Array(n * C), Es = new Float64Array(n * C);
  const sumD = new Float64Array(n);
  for (let i = 0; i < n; i++) for (let c = 0; c < C; c++) {
    const t = topK(feats, ws[i], c * M * D);
    let dm = 0, E = 0;
    for (let k = 0; k < K; k++) {
      const j = t[k], pj = prod[c * K + k], ct = (c * M + j) * K + k;
      slates[(i * C + c) * K + k] = j;
      dm += rhat[ct];
      if (j !== pj) E += A[ct] - A[(c * M + pj) * K + k];
    }
    const dc = dm - dmProd[c];
    dcs[i * C + c] = dc; Es[i * C + c] = E;
    sumD[i] += views[c] * dc + E;
  }
  const cov = new Float64Array(n * n);
  const N = Ntot;
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
    let s = 0;
    for (let c = 0; c < C; c++) {
      const a = dcs[i * C + c], b = dcs[j * C + c];
      s += views[c] * a * b + a * Es[j * C + c] + b * Es[i * C + c];
      const oi = (i * C + c) * K, oj = (j * C + c) * K;
      for (let k = 0; k < K; k++) {
        const ti = slates[oi + k], tj = slates[oj + k], pj = prod[c * K + k];
        if (ti === pj || tj === pj) continue;
        s += B[(c * M + pj) * K + k]; // both moved production's item out of slot k
        if (ti === tj) s += B[(c * M + ti) * K + k]; // both put the same item there
      }
    }
    const v = (s / N - (sumD[i] / N) * (sumD[j] / N)) / N / (pre.baseDR * pre.baseDR);
    cov[i * n + j] = v; cov[j * n + i] = v;
  }
  return cov;
}

// Idiosyncratic SEs and shared variances for correlation-aware pooling.
export function splitSE(se, lambda) {
  const idio = se.map((s, i) => Math.sqrt(Math.max(s * s - lambda[i] ** 2, 0.05 * s * s)));
  const shared = se.map((s, i) => s * s - idio[i] ** 2);
  return { idio, shared };
}

// One-factor approximation of a covariance matrix: cov ~ lambda lambda^T + diag(psi).
// lambda from the leading eigenvector (power iteration), clipped so psi >= 5% of the variance.
export function oneFactor(cov, n) {
  let v = new Float64Array(n).fill(1 / Math.sqrt(n)), mu = 0;
  for (let it = 0; it < 200; it++) {
    const u = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += cov[i * n + j] * v[j]; u[i] = s; }
    mu = Math.hypot(...u); for (let i = 0; i < n; i++) v[i] = u[i] / mu;
  }
  const lambda = Array.from(v, (x, i) => {
    const l = Math.sqrt(mu) * x, cap = Math.sqrt(0.95 * cov[i * n + i]);
    return Math.sign(l) * Math.min(Math.abs(l), cap);
  });
  let corr = 0, cnt = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) { corr += cov[i * n + j] / Math.sqrt(cov[i * n + i] * cov[j * n + j]); cnt++; }
  const share = mu / Array.from({ length: n }, (_, i) => cov[i * n + i]).reduce((a, b) => a + b, 0);
  return { lambda, eigen: mu, meanCorr: corr / cnt, share };
}
