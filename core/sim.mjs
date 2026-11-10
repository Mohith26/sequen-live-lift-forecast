// Seeded simulator for a furniture retailer's category pages (synthetic).
//
// Traffic is grouped into page contexts (category intent x price band x
// in-session style cluster x device). Each context retrieves M items and the
// ranker shows K of them. The forecaster sees only what a ranking platform
// logs: per-context features, the logging policy, and per (context, item,
// slot) impression and purchase counts. Ground truth (true purchase
// probabilities, exact policy values) is used only by tests and the
// simulation check.
//
// User model: position-based model (PBM). A shopper examines slot k with
// probability THETA[k] and buys the item there with probability p(c, j),
// independently across slots. Revenue = price of each purchased item.

export const M = 12; // retrieved candidates per page view
export const K = 6; // slots above the fold
export const D = 6; // ranking features
export const THETA = [1.0, 0.74, 0.58, 0.47, 0.39, 0.33];
export const FEATURES = ['Production score', 'In-session intent', 'Price-band affinity', 'Hover-dwell similarity', 'Popularity', 'Price tilt'];
export const SCALE = [0.03, 0.008, 0.03, 0.01, 0.03, 0.03]; // int8 code * SCALE = value
const BETA = [0.0, 1.55, 0.62, 0.62, 0.18, -0.78];
const Q_WEIGHT = 0.85;
const INTERCEPT = -4.35;
export const CATS = [
  ['Sofas', 1150], ['Sectionals', 1890], ['Accent chairs', 420], ['Coffee tables', 340],
  ['Area rugs', 260], ['Floor lamps', 150], ['Side tables', 180], ['Bookcases', 290],
];
export const PROD_W = [1, 0, 0, 0, 0, 0];

export function rng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  next.normal = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0;
    while (u === 0) u = next();
    const v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  next.int = (n) => Math.floor(next() * n);
  next.poisson = (lam) => {
    if (lam > 40) return Math.max(0, Math.round(lam + Math.sqrt(lam) * next.normal()));
    const L = Math.exp(-lam); let k = 0, p = 1;
    do { k++; p *= next(); } while (p > L);
    return k - 1;
  };
  next.binom = (n, p) => {
    if (n * p > 30 && n * (1 - p) > 30) return Math.max(0, Math.min(n, Math.round(n * p + Math.sqrt(n * p * (1 - p)) * next.normal())));
    if (n > 200) return Math.min(n, next.poisson(n * p));
    let x = 0; for (let i = 0; i < n; i++) if (next() < p) x++;
    return x;
  };
  return next;
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const q8 = (v, s) => Math.max(-127, Math.min(127, Math.round(v / s)));

export function makeCatalog(seed = 7, P = 896) {
  const r = rng(seed);
  const items = [];
  for (let i = 0; i < P; i++) {
    const c = i % CATS.length;
    const price = Math.round(CATS[c][1] * Math.exp(0.38 * r.normal()) / 5) * 5 - 0.01;
    const quality = r.normal();
    const pop = 0.55 * quality + 0.85 * r.normal();
    const st = [r.normal(), r.normal(), r.normal()];
    const n = Math.hypot(...st);
    items.push({ id: i, cat: c, price: Math.max(price, 39.99), quality, pop, style: st.map((x) => x / n) });
  }
  const lp = items.map((it) => Math.log(it.price));
  const mu = lp.reduce((a, b) => a + b, 0) / P;
  const sd = Math.sqrt(lp.reduce((a, b) => a + (b - mu) ** 2, 0) / P);
  items.forEach((it, i) => { it.lpz = (lp[i] - mu) / sd; });
  return items;
}

// Build C page contexts with features and true purchase probabilities.
export function makeWorld({ seed = 418, C = 2400, catalogSeed = 7 } = {}) {
  const items = makeCatalog(catalogSeed);
  const r = rng(seed);
  const NC = CATS.length;
  const feats = new Int8Array(C * M * D);
  const ids = new Uint16Array(C * M);
  const price = new Float64Array(C * M);
  const p = new Float64Array(C * M);
  const weight = new Float64Array(C);
  const intentOf = new Uint8Array(C);
  const device = new Uint8Array(C);
  let wsum = 0;
  for (let c = 0; c < C; c++) {
    const intent = r.int(NC);
    intentOf[c] = intent;
    device[c] = r() < 0.64 ? 1 : 0; // 1 = mobile
    weight[c] = Math.exp(1.1 * r.normal());
    wsum += weight[c];
    const raw = Array.from({ length: NC }, (_, k) => -Math.log(r() + 1e-12) * (k === intent ? 5.5 : 0.55));
    const tot = raw.reduce((a, b) => a + b, 0);
    const share = raw.map((x) => x / tot);
    const priceCenter = r.normal() * 0.9;
    const hs = [r.normal(), r.normal(), r.normal()];
    const hn = Math.hypot(...hs);
    const per = Math.floor(items.length / NC);
    const set = new Set();
    while (set.size < M) set.add(r() < 0.5 ? intent + NC * r.int(per) : r.int(items.length));
    [...set].forEach((id, j) => {
      const it = items[id];
      const fHover = (it.style[0] * hs[0] + it.style[1] * hs[1] + it.style[2] * hs[2]) / hn;
      const fPrice = -Math.abs(it.lpz - priceCenter) + 0.8;
      const fProd = 0.62 * it.quality + 1.0 * share[it.cat] + 0.36 * fPrice + 0.34 * fHover + 0.15 * it.pop + 0.42 * r.normal();
      const v = [fProd, share[it.cat], fPrice, fHover, it.pop, it.lpz].map((x, f) => q8(x, SCALE[f]));
      const b = (c * M + j);
      for (let f = 0; f < D; f++) feats[b * D + f] = v[f];
      let z = INTERCEPT + Q_WEIGHT * it.quality + 0.45 * r.normal() + (device[c] ? -0.15 : 0.1);
      for (let f = 1; f < D; f++) z += BETA[f] * v[f] * SCALE[f];
      p[b] = sigmoid(z);
      ids[b] = id;
      price[b] = it.price;
    });
  }
  for (let c = 0; c < C; c++) weight[c] /= wsum;
  return { C, feats, ids, price, p, weight, intentOf, device, items };
}

export function topK(feats, w, base = 0) {
  const s = new Float64Array(M);
  for (let j = 0; j < M; j++) {
    let v = 0;
    for (let f = 0; f < D; f++) v += w[f] * feats[base + j * D + f] * SCALE[f];
    s[j] = v;
  }
  const idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
  idx.sort((a, b) => s[b] - s[a] || a - b);
  return idx.slice(0, K);
}

// Exploration bucket: with prob eps a page view gets a uniformly shuffled
// top-12 (a random K-permutation of the M items), otherwise production's top-K.
// Logs are aggregated per (context, item, slot): impressions and purchases.
export function simulateLogs(world, { seed = 4181, N = 3200000, eps = 0.2 } = {}) {
  const r = rng(seed);
  const { C, feats, p } = world;
  const views = new Uint32Array(C);
  const cellN = new Uint32Array(C * M * K);
  const cellY = new Uint32Array(C * M * K);
  const pool = new Uint8Array(M);
  for (let c = 0; c < C; c++) {
    const n = r.poisson(N * world.weight[c]);
    views[c] = n;
    const prod = topK(feats, PROD_W, c * M * D);
    const ne = r.binom(n, eps);
    for (let k = 0; k < K; k++) cellN[(c * M + prod[k]) * K + k] += n - ne;
    for (let v = 0; v < ne; v++) {
      for (let j = 0; j < M; j++) pool[j] = j;
      for (let k = 0; k < K; k++) {
        const t = k + r.int(M - k);
        const tmp = pool[k]; pool[k] = pool[t]; pool[t] = tmp;
        cellN[(c * M + pool[k]) * K + k]++;
      }
    }
    for (let j = 0; j < M; j++) for (let k = 0; k < K; k++) {
      const cell = (c * M + j) * K + k;
      if (cellN[cell]) cellY[cell] = r.binom(cellN[cell], THETA[k] * p[c * M + j]);
    }
  }
  return { C, eps, views, cellN, cellY, feats: world.feats, price: world.price, ids: world.ids };
}

// Exact expected revenue per page view of a scorer (true probabilities).
export function trueValue(world, w) {
  let tot = 0;
  for (let c = 0; c < world.C; c++) {
    const t = topK(world.feats, w, c * M * D);
    let v = 0;
    for (let k = 0; k < K; k++) { const b = c * M + t[k]; v += THETA[k] * world.p[b] * world.price[b]; }
    tot += world.weight[c] * v;
  }
  return tot;
}

// The RRI session's offline experiments: agent hypotheses that re-weight
// in-session features on top of the production score.
export const FAMILIES = [
  { name: 'In-session intent boost', f: [1], scale: [0.5] },
  { name: 'Price-band affinity', f: [2], scale: [0.45] },
  { name: 'Hover-dwell similarity', f: [3], scale: [0.45] },
  { name: 'Popularity prior', f: [4], scale: [0.3] },
  { name: 'Margin tilt (price)', f: [5], scale: [0.2] },
  { name: 'Intent x hover blend', f: [1, 3], scale: [0.4, 0.4] },
  { name: 'Intent x price-band blend', f: [1, 2], scale: [0.4, 0.35] },
  { name: 'Hover x price-band blend', f: [3, 2], scale: [0.4, 0.35] },
  { name: 'Full in-session re-weight', f: [1, 2, 3, 4, 5], scale: [0.3, 0.25, 0.3, 0.2, 0.15] },
];

export function makeCandidates({ seed = 41800, N = 240 } = {}) {
  const r = rng(seed);
  const out = [];
  for (let c = 0; c < N; c++) {
    const fam = FAMILIES[c % FAMILIES.length];
    const aggressive = r() < 0.2;
    const w = [1, 0, 0, 0, 0, 0];
    fam.f.forEach((f, t) => {
      const mag = fam.scale[t] * (aggressive ? 1.8 + 1.6 * r() : 0.1 + 0.9 * r());
      w[f] = (r() < 0.15 ? -1 : 1) * mag;
    });
    if (r() < 0.3) w[0] = 0.7 + 0.5 * r();
    out.push({ id: `EXP-418-${String(c + 1).padStart(4, '0')}`, family: fam.name, aggressive, w: w.map((x) => Math.round(x * 1000) / 1000) });
  }
  return out;
}

export { sigmoid };
