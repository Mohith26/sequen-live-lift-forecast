// Prints the headline numbers the README quotes (same code path as the page).
//   node scripts/report.mjs
import { readFileSync } from 'node:fs';
import * as P from '../core/pipeline.mjs';
const src = readFileSync(new URL('../data/logs.js', import.meta.url), 'utf8');
const data = JSON.parse(src.slice(src.indexOf('{'), src.lastIndexOf('}') + 1));
const T = Object.fromEntries(JSON.parse(readFileSync(new URL('../data/truth.json', import.meta.url), 'utf8')).candidates.map((c) => [c.id, c]));
const res = P.run(data, P.decode(data));
const pct = (x) => (100 * x).toFixed(2) + '%';
const H = res.hist, top = res.byRaw[0];
console.log(`page views ${res.totalViews}, experiments ${res.rows.length}, past live tests ${H.rows.length}`);
console.log(`reward model theta ${res.model.theta.map((x) => x.toFixed(2)).join(' ')}`);
console.log(`top offline: ${top.id} DR ${pct(top.lift)} ± ${pct(top.se)} (IPS ${pct(top.ipsLift)} ± ${pct(top.ipsSe)}); pooled ${pct(top.post.mean)}; forecast readout ${pct(top.fc.center)} [${pct(top.fc.lo)}, ${pct(top.fc.hi)}]`);
console.log(`calibration: slope ${H.cal.b.toFixed(3)}, extra SD ${pct(Math.sqrt(H.cal.tau2))}, conformal q ${H.cal.q.toFixed(3)}`);
console.log(`past promotions: offline mean ${pct(H.meanRaw)} -> live mean ${pct(H.meanLive)}; LOO band coverage ${(H.looCoverage * H.rows.length).toFixed(0)}/${H.rows.length}, raw ± SE coverage ${(H.rawCoverage * H.rows.length).toFixed(0)}/${H.rows.length}`);
console.log(`shared error: mean corr ${res.factor.meanCorr.toFixed(3)}, factor share ${res.factor.share.toFixed(3)}`);
const t10 = res.byRaw.slice(0, 10), b = (f) => t10.reduce((a, r) => a + f(r) - T[r.id].trueOfflineLift, 0) / 10;
console.log(`simulation check, offline top-10 bias: raw ${pct(b((r) => r.lift))}, correlation-aware ${pct(b((r) => r.post.mean))}, independent pooling ${pct(b((r) => r.postIndep.mean))}`);
const cov = res.rows.filter((r) => T[r.id].trueLiveLift >= r.fc.lo && T[r.id].trueLiveLift <= r.fc.hi).length;
console.log(`simulation check, forecast bands containing the noise-free true live lift: ${cov}/${res.rows.length}`);
console.log(`planner (${res.opts.days} d, ${res.opts.armShare * 100}% per arm): readout SE ${pct(res.abSe)}, recommended ${res.recommended.map((r) => r.id + ' P(win) ' + r.pWin.toFixed(2)).join(', ')}`);
