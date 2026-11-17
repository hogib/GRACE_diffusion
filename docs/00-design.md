# 00 — Design

grace-diffusion sharpens GRACE/GRACE-FO mass-change maps with a conditional
diffusion model. Phase 1 asks one question before anything else is built:
**does a model that learned to add one octave of spatial detail to GRACE also
add the next octave correctly?** If it does, we try to go below what GRACE
resolves (Phase 2). If it does not, we have learned that cheaply.

This document is the plan agreed before implementation. Anything that changes
it is recorded here first. The literature behind it is in
`survey/grace_diffusion_survey.pdf`. Data findings that shaped it are in
Appendix A.

Decided so far:
- 2026-11-13 Spatial only. Each month is an independent map; no time
  modelling in Phase 1.
- 2026-11-13 GRACE only. The network sees only (blurred) GRACE, no hydrology
  models or covariates.
- 2026-11-13 Regional test on Türkiye and its surroundings; box 30–45° N,
  20–50° E (Türkiye, Tigris–Euphrates, Lake Urmia).
- 2026-11-13 Octave transfer is the gate for going further; gate criteria
  in §7.2.
- 2026-11-13 Model: EDM-preconditioned 2D U-Net diffusion, as in imonet but 2D.
- 2026-11-13 Data sources provisional: whatever GRACE series and hydrological
  model we can obtain now, to be confirmed with the supervisor; swapping
  sources must stay cheap. Current: ITSG-Grace2018 + ITSG-Grace_operational.
- 2026-11-13 (after Appendix A) The ladder moves down one octave on GRACE:
  train 4→8 and 8→16, gate on 16→32. GRACE has almost no clean signal above
  degree ~40, so 32→64 and 64→128 are tested on the hydrological twin only.
- 2026-11-13 (after Appendix A) Global images at rung-scaled resolution with
  random rotations, instead of local patches: at low degrees a patch would
  span half the planet.
- 2026-11-13 The gate runs on two GRACE targets, unfiltered and DDK5-filtered,
  and both are reported.

## 1. Why this design

**Why generative.** Many fine-scale maps are consistent with one blurry map. A
regression network returns their average, which is too smooth (survey §III).
A diffusion model samples from them. The spread of the samples is the
uncertainty of the added detail, and it can be checked for calibration. No
published GRACE study reports calibrated, spatially coherent ensembles.

**Why octaves.** GRACE has no truth below its resolution, so detail there can
never be scored against GRACE directly. Every learned product in the
literature scores near-perfectly at GRACE scale because its loss forces it,
and falls sharply against wells. Instead we test the assumption that
generating finer detail rests on: that "adding the next octave" is a skill
that transfers between scales. That can be tested with real GRACE between
scales GRACE does resolve.

**Why GRACE only first.** Any detail the model adds then comes from what
GRACE maps typically look like, not from a hydrological model's patterns. A
later phase adds model inputs and measures how much they contribute. No
published study separates the two.

## 2. Scales and the octave ladder

A field on the sphere is a sum of spherical-harmonic degrees *l*. Degree *l*
carries detail of half-wavelength ≈ 20,000 km / *l*.

| rung | input → target degrees | finest detail | data | role |
|---|---|---|---|---|
| R1 | ≤ 4 → ≤ 8 | ~2,500 km | GRACE, twin | training |
| R2 | ≤ 8 → ≤ 16 | ~1,250 km | GRACE, twin | training |
| R3 | ≤ 16 → ≤ 32 | ~625 km | GRACE, twin | **gate** (held out from training) |
| R4 | ≤ 32 → ≤ 64 | ~310 km | twin only | transfer check with truth |
| R5 | ≤ 64 → ≤ 128 | ~160 km | twin only | the Phase 2 step, with truth |

**The gate:** train on R1 and R2 only, then evaluate on R3 without retraining.
Compare against (a) a model trained on R3 itself, the "oracle", which shows
how much transfer costs, and (b) the baselines in §5. On the twin, the same
R1+R2 model is also evaluated on R4 and R5, i.e. two and three octaves
beyond its training.

**Low-pass operator.** Hard truncation at degree *L* rings (Gibbs). We use a
smooth taper in degree, w_L(l): 1 up to 0.8 L, a cosine roll-off to 0 at L
(final form open, §10). The input taper w_{L/2} lies inside the flat part of
w_L, so blurring a target to the input degree reproduces the input exactly.

**Scale invariance by construction.** A CNN sees pixels, not kilometres. Each
rung is synthesised on a global equiangular grid with p = 4 pixels per
half-wavelength of its target degree L: 8L × 4L pixels (lon × lat). That is
64×32 at R1, 128×64 at R2, 256×128 at R3, 512×256 at R4 and 1024×512 at R5.
Features occupy the same number of pixels at every rung, and the globe gets
larger in pixels as the rung rises. Amplitude falls with degree, so each map
is normalised by the RMS of its input. The model predicts in those units and
the result is scaled back (alternatives in §10).

**Rotations.** Each training sample is a month's coefficients rotated by a
random rotation of the sphere (exact in spherical harmonics), then
synthesised. This turns about 180 training months into an effectively
unlimited set of distinct images. It also spreads the equirectangular
distortion and the poles over all features, and stops the model from
memorising geography. At evaluation, the field is rotated so the test box sits
on the equator, where distortion is smallest. Stripes are north–south in the
original frame, so rotation also reorients the noise of the unfiltered target
(open question, §10).

## 3. Problem

**Input:** the input-rung field w_{L/2} x of one (rotated) month on the
rung's grid, normalised. Nothing else.

**Output:** samples of the target-rung field w_L x, as an ensemble.

**Network:** a fully convolutional 2D U-Net with circular padding in
longitude and reflection across the poles, EDM preconditioning and loss,
Heun sampler. Fully convolutional, so the same weights run on every rung's
grid size.

**Consistency (hard constraint).** Blurring a sample back to the input degree
must return the input. The taper is not itself a projection (w² ≠ w in its
roll-off), so the constraint is set per degree. Wherever w_{L/2}(l) > 0, the
output's degree-l coefficients are fixed to input_l / w_{L/2}(l). That equals
the true target's, because w_L = 1 there. All degrees with w_{L/2}(l) = 0 are
generated freely. This is a projection (idempotent, true target is a fixed
point), computed by spherical-harmonic analysis and synthesis on the grid,
and exact on the sphere (cf. DDNM's range/null-space split;
`sh.project_consistent`, tested in `tests/test_sh.py`). It is applied to the final sample, and
optionally at every sampling step (§10). The remaining violation from
discretisation is reported as a metric.

## 4. Data

**GRACE/GRACE-FO.** ITSG-Grace2018 (Apr 2002 – Jun 2017, 162 months) and
ITSG-Grace_operational (Jun 2018 – Sep 2026, 98 months): unconstrained
monthly solutions to degree 120 with formal errors. Anomalies are taken
relative to the 2004–2009 mean and converted to equivalent water height with
PREM load Love numbers. Degrees 0 and 1 are dropped (not observed).
Corrections (2026-11-13, provisional sources like the rest):
- C20, and C30 where given, replaced by GSFC TN-14 SLR values (version of
  3 Nov 2026, zero tide like ITSG). TN-14 rows follow JPL solution spans,
  so each calendar month takes the row overlapping it most; all 260 months
  are matched.
- GIA removed with the Caron et al. (2018) expectation model (JPL VESL,
  Stokes rates to degree 89; covariances also available). ICE-6G_D, the
  more common choice, was unreachable.
Check (Gaussian 300 km EWH trends, mm/yr, raw → corrected): Hudson Bay
+31.2 → −0.6, Scandinavia +21.8 → +0.1, West Antarctica −112.7 → −124.3,
NW India −23.6 → −25.4, Amazon +3.7 → +4.2, Türkiye −2.1 → −1.5. Degree 1
(geocentre, TN-13) is not yet added.

**Two GRACE targets.** The gate runs on:
- *unfiltered*: R3 band (16–32) signal/noise ~11.5;
- *DDK5-filtered*: keeps ~94% of the R3 band at signal/noise ~13.5 and
  removes the stripes.
If the two disagree, the filter is shaping the result.

**Hydrological twin.** The same experiments on a global hydrological model's
TWS (WaterGAP v2.2e, monthly means of its daily storage output, or similar;
source to confirm), expanded to degree ≥ 128. Its fine degrees are clean
signal, so on the twin:
- the octave transfer is tested without GRACE noise;
- R4 and R5 can be scored against truth.

The twin is a closed-loop check. It is not training data for the GRACE model.

*Twin as built (2026-11-13, `prep.build_twin`):* WaterGAP v2.2e,
gswp3-w5e5, histsoc, monthly TWS 1901–2019 (1,428 months, 0.5°, land only;
ocean set to zero anomaly), regridded to a Driscoll–Healy grid and expanded to
degree 128. About 2,600 cells (4% of land) whose storage drifts by more than
1 m between 1901–1910 and 2010–2019 are excluded (set to zero anomaly). These
are glacier cells where snow accumulates without bound (mostly high latitude,
Greenland included) and city cells with point-like groundwater abstraction
(New York −100 m, Los Angeles −30 m, Tokyo −22 m). Left in, they dominated
the fine scales. At R5 a few isolated sharp features (lakes) still ring
slightly. Twin splits: train 1901–1989, validation 1990–2001, test 2002–2019.

**Splits.** Months are correlated, so splits are blocks, never random months.
- Train: Apr 2002 – Dec 2019.
- Validation: Jan 2020 – Dec 2022 (all model selection and tuning).
- Test: Jan 2023 – latest. Untouched until the gate decision.
- Spatial hold-out: the test box (plus a margin of one target half-wavelength)
  is masked out of the training loss, rotating with the field. The network
  still sees the box in its inputs, but is never trained to produce its
  detail.
- Greenland and Antarctica are masked out of the loss (proposed): ice-sheet
  trends dominate the variance there.

## 5. Baselines

Generators and point predictors on the same fields:
1. **input**: the blurred field itself (no added detail). The floor.
2. **regress**: a deterministic U-Net with the same backbone and inputs,
   trained with MSE. The conditional mean.
3. **spectral-gauss**: the strong Gaussian generator, the analogue of the GP
   in imonet. Fit a power law to the degree variance of the input month,
   extrapolate it into the missing degrees, draw random coefficients with
   that spectrum, and add them to the input. On the sphere this is exact. Its
   samples have the right power and no learned structure.
4. **regress + spectral-gauss**: the regression mean plus Gaussian residuals
   scaled to the validation error. This separates "better mean" from "better
   spread".

## 6. Metrics

On the added detail (target minus input), in the test box rotated to the
equator, and globally on land outside the box for validation:
- Point: RMSE of the ensemble mean; correlation.
- Probabilistic: fair CRPS (per pixel, area-weighted); energy score (per
  field); spread/error ratio; 50/90% coverage; rank histograms.
- Spectral: degree variance of samples against the truth, per degree, in the
  target band.
- Structure: a classifier two-sample test, real vs generated added detail (as
  in imonet's C2ST), with a positive control.
- Stripes: north–south vs east–west power ratio of the added detail, in the
  original (unrotated) frame, against the truth.
- Consistency: RMS of `w_{L/2}(output) − input`.

Paired comparisons (method vs method on the same fields) use a block
bootstrap over months and a sign test over months, as in imonet.

## 7. Experiments

### 7.1 A — Within-rung recovery (R2, validation)
Train on R1 + R2 and score on R2 validation months. This sets the bar: does
diffusion beat `spectral-gauss` and `regress` on CRPS and calibration at all?

### 7.2 B — Octave transfer, the gate (R3, validation)
Same model, no retraining, scored on R3, on both GRACE targets. Also train the
R3 oracle. Report the transfer cost (B vs oracle) and the margin over the
baselines.

**Gate criteria (accepted 2026-11-13):**
- the transferred model beats `spectral-gauss` in CRPS on R3 (paired, block
  bootstrap interval excludes zero);
- its CRPS is within 10% of the R3 oracle;
- spread/error between 0.85 and 1.15, coverage within 5 points of nominal;
- no excess stripe power relative to the truth;
- the same holds on the twin, **and** on the twin at R4 and R5 against truth.

### 7.3 C — Twin (closed loop)
A and B repeated on the twin, plus R4 and R5 scored against twin truth with
the R1+R2 model (and with an R1–R3 model, for comparison).

### 7.4 D — Türkiye (test)
Run once, after the gate decision: R3 on the test box over the test months,
all methods. Nothing tuned on it.

## 8. Order of work

1. **Data (done in part):** ITSG series fetched; anomalies and EWH conversion;
   DDK filters available. To do: SLR C20 and GIA (§10); fetch the twin.
2. **sh:** degree tapers, rotations, synthesis on rung grids, analysis on the
   grid for the consistency projection. Unit tests: blurring a rung-L field
   to L/2 equals the rung-L/2 input; projection idempotence; rotation
   preserves degree variance.
3. **data:** sampler over (month, rung, rotation) with the temporal splits
   and the rotated loss masks; normalisation.
4. **baselines, metrics:** experiment A for the baselines, so the bar is set
   before any model is trained.
5. **model, train:** EDM U-Net (2D) and `regress`; experiment A, then B.
6. **twin:** experiment C.
7. Gate decision (with the user). Then D.

## 9. Code layout

```
src/grace_diffusion/
  fetch.py       download GRACE SH series, DDK matrices, the twin
  sh.py          coefficients, anomalies, EWH, Love numbers, DDK, tapers,
                 rotations, synthesis/analysis on rung grids
  grids.py       rung grids, masks (test box, ice sheets), rotation of masks
  data.py        torch Dataset over (month, rung, rotation)
  model/         U-Net denoiser, EDM loss and sampler, consistency projection
  baselines.py   input, regress, spectral-gauss, regress+gauss
  metrics.py     CRPS, energy score, spread/error, coverage, degree variance,
                 stripe ratio, C2ST, consistency
  train.py
  evaluate.py
  cli.py
scripts/         exploratory analyses (explore_filters.py, explore_ddk.py)
tests/           unit tests on synthetic fields
docs/            this document
survey/          literature survey (LaTeX) and notes
```

Dependencies: numpy, scipy, torch, pyshtools, gravity-toolkit (Love numbers),
shxarray (DDK reader), matplotlib.

## 10. Open questions

- **Taper shape** of the low-pass operator.
- **Normalisation:** per-map RMS of the input (scale-free, part of the
  hypothesis) or per-rung degree-variance scaling.
- **Rotations and stripes:** full random rotations reorient the stripes of the
  unfiltered target. Alternatives: rotations about the polar axis only (keeps
  stripe orientation and latitude structure, less augmentation), or full
  rotations on the DDK5 target only.
- **Projection at every sampling step** or only on the final sample.
- **Degree 1 (geocentre):** add a TN-13-style estimate, or leave degree 1 out
  of all rungs (it only affects the largest scales of R1).
- **Ensemble size** for scoring and the number of sampler steps.
- **Data volume:** about 180 training months with rotation augmentation.
  Whether the rotated copies of the same month are diverse enough is unknown
  until A.

## 11. Later phases (not designed yet)

- **Phase 2 — Finer than GRACE over Türkiye** (only if the gate passes).
  Generate the next octaves on real GRACE. Validate against independent data:
  GNSS vertical loading (imonet's station data) and wells where available.
  Report how much fine-scale variance comes from the model's learned
  structure.
- **Phase 3 — Adding a prior.** Condition on hydrological-model fields and
  covariates; measure their contribution against GRACE-only.
- **Phase 4 — Time.** Gap filling, nowcasting over the release latency, and
  reconstruction, with GRACE context months as optional inputs (dropped at
  random in training, like imonet's null token).
- **Phase 5 — Operator-aware sampling.** Guide a prior through each product's
  real observation operator and error covariance (posterior sampling), which
  also turns the noisy 31–60 band into usable information.

## Appendix A — Data findings (2026-11-13)

**Where GRACE's signal ends** (`scripts/explore_filters.py`, ITSG, 260 months,
`runs/explore/degree_amplitudes.png`). The median monthly anomaly rises above
the noise only up to about degree 40. From ~45 upward it equals the noise
(month-to-month difference / √2), which matches the ITSG formal errors. Over
the test box, the degree-60 field of April 2010 is north–south stripes of
±400 mm against a real signal of ~100–150 mm; at degree 96 the stripes reach
±3,000 mm.

**Filters per band** (`scripts/explore_ddk.py`, EWH with PREM Love numbers,
`runs/explore/filters_by_band.csv`, `maps_ddk_turkiye_box.png`). Signal kept
and residual signal/noise power:

| filter | 16–30 kept | 16–30 S/N | 31–60 kept | 31–60 S/N |
|---|---|---|---|---|
| none | 1.00 | 11.5 | 1.00 | 0.29 |
| Gaussian 300 km | 0.42 | 12.5 | 0.06 | 1.3 |
| DDK3 | 0.87 | 21 | 0.16 | 31 |
| DDK4 | 0.90 | 18 | 0.22 | 22 |
| DDK5 | 0.94 | 13.5 | 0.41 | 9 |

No filter keeps any signal above degree 60. DDK removes the stripes in the
test box; Gaussian 300 km leaves elongated north–south features. The noise
proxy includes real month-to-month change, so signal estimates are lower
bounds, and the 31–60 "kept" values rest on a noisy signal estimate.

## Appendix B — Phase 1 results (2026-11-16/17)

Validation months only (GRACE 2020–2022, twin 1990–2001); scored over validation land;
CRPS in mm EWH; 32 members (16 at R5). Raw scores: `runs/eval/*.csv` on the server;
scripts `scripts/phase1.sh`, `scripts/phase1b_noise.sh`, `scripts/diagnose_sampler.py`.

**Phase 1 (commit 3be72f3/9fe39fd, 40k steps, EDM default noise schedule).**

| CRPS | input | spectral-gauss | diffusion | regress | oracle |
|---|---|---|---|---|---|
| A: GRACE DDK5 R2 | 62.9 | 45.9 | 37.2 | **26.3** | — |
| B: GRACE DDK5 R3 | 64.0 | 48.7 | 45.5 | 71.1 | **24.8** |
| B: GRACE raw R3 | 65.1 | 49.4 | 46.4 | 72.0 | 25.5 |
| C: twin R2 / R3 / R4 / R5 | 21.2 / 19.2 / 19.6 / 20.1 | 16.1 / 14.4 / 15.0 / 15.5 | 9.4 / 13.0 / 14.1 / 14.7 | — | — |

Gate (§7.2), Phase 1 model: beats spectral-gauss at R3 (−3.3 mm, block-bootstrap 95%
[−3.7, −2.9]) ✅; within 10% of the oracle ❌ (+84%); spread/error 0.82, coverage 40/82% ❌;
stripes not measured; twin R3–R5 beats baselines by a shrinking margin (−1.4, −0.8, −0.8 mm).
**Gate not passed.**

**Diagnosis.** The Phase 1 diffusion model was weaker than `regress` in-distribution. The
sampler was not the cause (32 vs 128 steps identical). Its denoiser barely used the input
at high noise (corr with true detail at σ=10: 0.26 at R1, 0.40 at R2, vs 0.89/0.92 for
regress): EDM's default ln σ ~ N(−1.2, 1.2²) puts only ~0.2% of batches at σ > 10. The R3
oracle learned it anyway (0.79), plausibly because its grids carry 4–16× more pixels per batch.

**Phase 1b (commit c78433c, 60k steps, sampler from σ_max = 20).** `wide`: ln σ ~ N(−0.4, 1.6²);
`logu`: ln σ uniform on [0.002, 20].

| | A: R2 CRPS | R2 corr (mean) | B: R3 CRPS (DDK5 / raw) | R3 corr | R3 spread/error | R3 vs spectral-gauss (95%) |
|---|---|---|---|---|---|---|
| Phase 1 diffusion | 37.2 | 0.53 | 45.5 / 46.4 | 0.28 | 0.82 | −3.3 [−3.7, −2.9] |
| wide | **20.5** | 0.90 | 52.5 / 53.5 | 0.21 | 0.37 | +3.8 [+2.7, +4.6] |
| logu | 21.6 | 0.89 | 51.0 / 52.0 | 0.23 | 0.49 | +2.3 [+1.7, +2.8] |

**Reading.** With the noise schedule fixed, the model is now the best method in-distribution
(R2: 20.5 vs 26.3 regress, 45.9 spectral-gauss; corr 0.90), but it transfers *worse*:
at R3 it loses to spectral-gauss and is badly over-confident (spread/error 0.37–0.49).
The Phase 1 model's apparent transfer win came from it being weak and close to generic
noise. Like `regress`, a model that has learned R1/R2 well does not carry the skill one
octave up on GRACE. The R3 oracle shows the R3 detail itself is learnable from R3 inputs.
Not yet separated: whether the failure is the physics (GRACE fields not self-similar
across these degrees) or GRACE/DDK-specific (filter attenuation and noise growing in
16–32). The twin, retrained with the fixed schedule, separates these.
