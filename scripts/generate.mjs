// Generates the synthetic dataset the demo reads (seeded, deterministic).
//   node scripts/generate.mjs
// Writes data/logs.js (what a ranking platform logs + the RRI session's
// experiments + past live A/B history) and data/truth.json (ground truth,
// used only by tests and the page's optional simulation check).
import { writeFileSync } from 'node:fs';
import * as S from '../core/sim.mjs';

export const CONFIG = {
  worldSeed: 418, logSeed: 4181, candSeed: 41800, histSeed: 41700,
  C: 2400, N: 6000000, eps: 0.2, days: 14, logShare: 0.15, nCandidates: 240,
  // hidden truths of the online environment (the forecaster must recover them)
  liveSlope: 0.8, liveExtraSd: 0.002,
  pastSessions: [412, 413, 414, 415, 416, 417], promotedPerSession: 4,
  abArmShare: 0.05, abDays: 14,
};

const NAMES = {
  Sofas: ['Harlow', 'Mercer', 'Calloway', 'Ellison', 'Brinley', 'Ashby', 'Delancey', 'Rowan'],
  Sectionals: ['Monterey', 'Halden', 'Bexley', 'Corbin', 'Larkin', 'Sutton', 'Wren', 'Avalon'],
  'Accent chairs': ['Odette', 'Linus', 'Marlow', 'Tessa', 'Hugo', 'Ines', 'Bram', 'Celine'],
  'Coffee tables': ['Arlo', 'Quinn', 'Sloane', 'Emery', 'Kit', 'Nash', 'Remy', 'Juno'],
  'Area rugs': ['Kilim', 'Sahara', 'Tidewater', 'Marrakesh', 'Fjord', 'Ember', 'Sable', 'Meadow'],
  'Floor lamps': ['Arc', 'Halo', 'Beacon', 'Spindle', 'Lumen', 'Crane', 'Pillar', 'Orbit'],
  'Side tables': ['Pebble', 'Drum', 'Tripod', 'Nesting', 'Ledge', 'Cube', 'Tulip', 'Plinth'],
  Bookcases: ['Ladder', 'Gallery', 'Etagere', 'Cubby', 'Library', 'Atlas', 'Barrister', 'Loft'],
};
const SUFFIX = {
  Sofas: ['84" Sofa', '78" Sofa', 'Sleeper Sofa', 'Loveseat'], Sectionals: ['L-Sectional', 'Modular Sectional', 'U-Sectional', 'Chaise Sectional'],
  'Accent chairs': ['Accent Chair', 'Swivel Chair', 'Barrel Chair', 'Lounge Chair'], 'Coffee tables': ['Coffee Table', 'Lift-Top Table', 'Round Coffee Table', 'Storage Table'],
  'Area rugs': ['8x10 Rug', '5x8 Rug', '9x12 Rug', 'Runner'], 'Floor lamps': ['Floor Lamp', 'Arc Lamp', 'Tripod Lamp', 'Torchiere'],
  'Side tables': ['Side Table', 'End Table', 'C-Table', 'Drink Table'], Bookcases: ['Bookcase', 'Shelf', 'Cabinet', 'Etagere'],
};

function b64(typed) { return Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString('base64'); }

export function build(cfg = CONFIG) {
  const world = S.makeWorld({ seed: cfg.worldSeed, C: cfg.C });
  const logs = S.simulateLogs(world, { seed: cfg.logSeed, N: cfg.N, eps: cfg.eps });
  let mx = 0; for (const x of logs.cellN) mx = Math.max(mx, x);
  if (mx > 65535) throw new Error('cell count overflow ' + mx);
  const cands = S.makeCandidates({ seed: cfg.candSeed, N: cfg.nCandidates });
  const v0 = S.trueValue(world, S.PROD_W);
  const trueLift = cands.map((c) => S.trueValue(world, c.w) / v0 - 1);
  const r = S.rng(cfg.histSeed);
  const liveTruth = trueLift.map((t) => cfg.liveSlope * t + cfg.liveExtraSd * r.normal());
  return { world, logs, cands, v0, trueLift, liveTruth, rng: r };
}

// Past RRI sessions on the same surface. Offline estimates and SEs are what the
// harness recorded then; the promoted experiments' live A/B readouts are what the
// experimentation platform reported. Historic sessions promoted the top raw estimates.
export function makeHistory(cfg, trueLift, seList, lamRatio, r, sdRel, perArmPerDay) {
  const sessions = [];
  for (const sid of cfg.pastSessions) {
    const n = 180 + r.int(81);
    const est = [], se = [], truth = [], lambda = [];
    const f = r.normal(); // the session's shared error factor
    for (let i = 0; i < n; i++) {
      const pick = r.int(trueLift.length);
      const t = trueLift[pick] + 0.0015 * r.normal();
      const s = seList[pick] * (0.85 + 0.3 * r());
      const l = lamRatio[pick] * s;
      truth.push(t); se.push(s); lambda.push(l);
      est.push(t + l * f + Math.sqrt(s * s - l * l) * r.normal());
    }
    const order = est.map((_, i) => i).sort((a, b) => est[b] - est[a]).slice(0, cfg.promotedPerSession);
    const tests = order.map((i) => {
      const days = [10, 14, 14, 21][r.int(4)];
      const liveSe = sdRel * Math.sqrt(2 / (perArmPerDay * days));
      const liveTrue = cfg.liveSlope * truth[i] + cfg.liveExtraSd * r.normal();
      return { index: i, days, liveSe, live: liveTrue + liveSe * r.normal(), _liveTrue: liveTrue };
    });
    sessions.push({ session: sid, est, se, lambda, tests, _truth: truth });
  }
  return sessions;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const cfg = CONFIG;
  const { world, logs, cands, v0, trueLift, liveTruth, rng } = build(cfg);
  // SE scale for history comes from re-estimating with the shipped estimator
  const F = await import('../core/forecast.mjs');
  const model = F.fitRewardModel(logs);
  const pre = F.prepare(logs, model);
  const seList = cands.map((c) => F.estimateLift(pre, c.w).se);
  const fac = F.oneFactor(F.liftCovariance(pre, cands.map((c) => c.w)), cands.length);
  const lamRatio = fac.lambda.map((l, i) => l / seList[i]);
  const rs = F.revenueStats(pre);
  const perArmPerDay = cfg.N / cfg.days / cfg.logShare * cfg.abArmShare;
  const history = makeHistory(cfg, trueLift, seList, lamRatio, rng, rs.sdRel, perArmPerDay);
  const catalog = world.items.map((it, i) => {
    const cat = S.CATS[it.cat][0];
    return { n: `${NAMES[cat][i % 8]} ${SUFFIX[cat][Math.floor(i / 8) % 4]}`, c: it.cat, p: Math.round(it.price * 100) / 100 };
  });
  const data = {
    meta: {
      surface: 'Category pages · Living room', client: 'Demo retailer (synthetic)', session: 418,
      C: cfg.C, M: S.M, K: S.K, D: S.D, eps: cfg.eps, days: cfg.days, logShare: cfg.logShare, scale: S.SCALE, features: S.FEATURES,
      cats: S.CATS.map((c) => c[0]), abArmShare: cfg.abArmShare, abDays: cfg.abDays,
      generator: 'scripts/generate.mjs', seeds: { world: cfg.worldSeed, logs: cfg.logSeed, candidates: cfg.candSeed, history: cfg.histSeed },
    },
    contexts: { intent: Array.from(world.intentOf), device: Array.from(world.device) },
    b64: { feats: b64(logs.feats), ids: b64(logs.ids), views: b64(logs.views), cellN: b64(Uint16Array.from(logs.cellN)), cellY: b64(Uint16Array.from(logs.cellY)) },
    catalog,
    candidates: cands,
    history: history.map((h) => ({ session: h.session, est: h.est.map((x) => +x.toFixed(6)), se: h.se.map((x) => +x.toFixed(6)), lambda: h.lambda.map((x) => +x.toFixed(6)), tests: h.tests.map((t) => ({ index: t.index, days: t.days, liveSe: +t.liveSe.toFixed(6), live: +t.live.toFixed(6) })) })),
  };
  writeFileSync(new URL('../data/logs.js', import.meta.url), 'window.SEQ_DATA = ' + JSON.stringify(data) + ';\n');
  const truth = {
    note: 'Ground truth from the simulator. The forecaster never reads this; the page uses it only for the optional simulation check.',
    productionValue: v0, liveSlope: cfg.liveSlope, liveExtraSd: cfg.liveExtraSd,
    candidates: cands.map((c, i) => ({ id: c.id, trueOfflineLift: +trueLift[i].toFixed(6), trueLiveLift: +liveTruth[i].toFixed(6) })),
  };
  writeFileSync(new URL('../data/truth.json', import.meta.url), JSON.stringify(truth));
  writeFileSync(new URL('../data/truth.js', import.meta.url), 'window.SEQ_TRUTH = ' + JSON.stringify(truth) + ';\n');
  console.log('views', logs.views.reduce((a, b) => a + b, 0), 'v0', v0.toFixed(3), 'sdRel', rs.sdRel.toFixed(3), 'perArmPerDay', perArmPerDay);
}
