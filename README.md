# Live Lift Forecast for Sequen RRI (concept demo)

**Live demo:** https://mohith26.github.io/sequen-live-lift-forecast/

An unofficial concept for Sequen's Recursive Ranking Intelligence: a panel on an RRI session that answers *which of this session's offline wins will hold up in a live A/B test, and by how much*. Synthetic data; not affiliated with Sequen.

## Why

RRI runs thousands of offline experiments per week and promotes winners to live A/B tests on the customer's own instrumentation. Picking the best of hundreds of noisy offline estimates has a predictable problem: the winner's estimate is inflated (the winner's curse), and every experiment in a session is scored on the same logged traffic, so their errors are correlated. On the demo data, the 24 experiments promoted from six past sessions promised +1.77% revenue per page view offline and delivered +1.13% live, and raw "estimate ± SE" bands contained only 14 of the 24 live readouts.

Live Lift Forecast turns each offline estimate into a forecast of the readout the live test will produce, with a calibrated band, P(significant win) and days to 80% power, and ranks the promotion queue by that.

## How it works (all in `core/`, every number on the page is computed in the browser by the same code)

1. **Counterfactual revenue from the exploration bucket** (`forecast.mjs: estimateLift`). Logs are per (page context, item, slot) impressions and purchases; 20% of page views get a uniformly shuffled top 12, so the marginal propensity of item j at slot k is `(1-eps)[production put j at k] + eps/12`. Each experiment's revenue per page view is estimated with item-position IPS and a doubly robust estimator whose reward model is a position-based model (`theta_k * sigmoid(beta.x) * price`, fit by binomial maximum likelihood; `theta` comes from an IPS ratio that is unbiased across slots). Lift is paired against production on the same views; the SE uses the exact per-cell sums of squares (each purchase's revenue is the item price) under a Poissonized-cell approximation.
2. **Shared error** (`liftCovariance`, `oneFactor`). The full 240 x 240 covariance of the estimates follows from the same cell statistics (two experiments share error where they both move production's item out of a slot or put the same item in it). A one-factor fit splits each SE into a shared part and an idiosyncratic part. On the demo data the mean correlation is 0.23.
3. **Empirical Bayes pooling** (`npmle`, `posterior`). A smoothed nonparametric maximum-likelihood prior (Kiefer-Wolfowitz NPMLE on a grid with Gaussian atoms, fit by EM) is estimated from the session's own estimates using the idiosyncratic SEs; posteriors are normal mixtures, and the shared variance is added back to each posterior (pooling cannot remove it). Treating the errors as independent over-shrinks: on the demo, it biases the offline top 10 by -0.44 pp versus -0.04 pp for the correlation-aware version (raw: +0.36 pp). The page has a toggle to compare.
4. **Offline-to-online calibration** (`calibrate`, `forecast`). On past sessions' promoted experiments: slope `b` of live readout on pooled offline lift (weighted least squares, demo: 0.84), extra online variance `tau^2` by method of moments (novelty, interference, drift), and a split-conformal multiplier on standardized residuals so the band for a new readout has finite-sample coverage if past and future tests are exchangeable. Leave-one-out on the 24 past tests: 20/24 inside the 80% band.
5. **Planner** (`pWin`, `daysTo80`). Given traffic per arm and duration, readout SE comes from the production revenue SD; P(significant win at 5%) integrates the normal-mixture posterior; ranks and the "Recommended" set update live.

The harness metric (purchase NDCG@6 replayed on logged purchases) is shown next to each experiment: it moves opposite to counterfactual revenue for 140 of 240 experiments, which is the gap Zoë Weil describes between easy-to-evaluate metrics and business ground truth.

## Data

`scripts/generate.mjs` (seeded) builds a furniture retailer's category pages: 896 products, 2,400 page contexts (category intent, price band, hover-style cluster, device), 6.0M logged page views over 14 days with a position-based shopper model, 240 RRI experiments that re-weight in-session features on top of the production score, and six past sessions (412-417) with recorded offline estimates and live A/B readouts. `data/truth.json` holds the simulator's ground truth; the forecaster never reads it, the page's "Reveal simulated truth" check does. Hidden online truths: live lift = 0.8 x offline lift + N(0, 0.2 pp).

## Tests

`node --test test/` (9 tests): top-K, propensities, DR/IPS unbiasedness and SE calibration over 14 independent log replications (90% coverage), covariance diagonal identity plus replication of a predicted off-diagonal correlation, empirical Bayes vs raw MSE and top-10 bias, factor-aware vs independent pooling coverage under a shared error, slope recovery and conformal coverage on held-out tests, position-bias recovery, and end-to-end consistency plus simulation check on the shipped dataset. `node scripts/report.mjs` prints the numbers quoted here.

## Honest limitations

- Synthetic data. The shopper model is a position-based model with independent slots, so IPS/DR are unbiased by construction; real pages have cascade effects, so a production version would need a cascade-aware or slate estimator (e.g. pseudoinverse) and the variance formula would change.
- Uniform shuffles of 20% of traffic are a stand-in; Sequen's real exploration (if any) would set the propensities.
- The covariance uses a Poissonized-cell approximation and one factor; a few experiments with distinctive overlap structure may need more factors.
- Conformal coverage assumes past and future promoted tests are exchangeable. Only 24 past tests are available, so the band's multiplier is noisy.
- The planner ignores sequential testing, multiple live arms sharing a control, and interference between concurrent tests.
- The RRI app shell is invented: Sequen has no public screenshots of its product, so the UI follows sequen.ai's design system (Aleo, DM Sans, Departure Mono, their palette and dot motifs) rather than a real screen.

## Novelty note

Checked sequen.ai (home, platform, use cases, about, blog, careers job posts for RRI Harness and Recursive Self-Improvement), the Series A/B announcements, The Stack and Prysm profiles, and the TWIML episode summary (Oct 2026): Sequen describes thousands of offline experiments, evaluation harnesses and live A/B proof, but nothing public forecasts live readouts from offline estimates, corrects for selection across an RRI session, or models correlated errors. The building blocks are published research (off-policy ranking evaluation, NPMLE empirical Bayes, conformal prediction); the combination for agent-generated experiment sessions is the contribution here.

Unofficial concept demo by Mohith Gajjela. Not affiliated with Sequen. Synthetic data. Departure Mono font (MIT) by Helena Zhang and Tobias Fried, vendored in `assets/`.
