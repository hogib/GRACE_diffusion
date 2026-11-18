# 00 — Design

*Agreed 18 November 2026 (evening). This document replaces the earlier designs (octave
transfer; paired GRACE–WaterGAP training), which remain in the git history. Their results are
summarised in §1 and Appendix B. The written proposal is `reports/proposal/proposal.pdf`; the
first experiment is written up in `reports/phase1/phase1_report.pdf`.*

**In one sentence:** the network learns from hydrological models what fine detail goes with a
given large-scale field, and applies that to a coarse GRACE/GRACE-FO map, leaving GRACE's own
large-scale picture untouched.

The plan is recorded here before implementation; anything that changes it is recorded here
first. The literature behind the project is in `survey/grace_diffusion_survey.pdf`. Data
findings that shaped it are in Appendix A.

## Decisions

| # | point | decision |
|---|---|---|
| 1 | input | GRACE only: one coarse map of a single month |
| 2 | training data | hydrological models only: blurred model in, the same model's detail out |
| 3 | models | ten ISIMIP3a models, used jointly; GLWS2.0 is a benchmark, not a validation |
| 4 | coarse part | fixed to GRACE by the consistency projection |
| 5 | model | conditional diffusion with the stochastic sampler; direct network as baseline |
| 6 | spread | factor fitted on held-out models, plus the margin GRACE needs at coarse scales |
| 7 | checks | five levels, from held-out model years to independent data in Türkiye |
| 8 | resolution | degree 128; one step or two decided by experiment on model data |

Standing decisions:
- Spatial only: each month is an independent map, no time modelling.
- Regional test on Türkiye and surroundings: box 30–45° N, 20–50° E, held out of all training.
- Data sources are provisional until confirmed with the supervisor; swapping them must stay
  cheap.
- Whole-globe images on scale-matched grids with random rotations (§2).
- GRACE months from 2023 onward and the Türkiye box are not looked at until the design is
  frozen.

## 1. Background: three sources of training examples

No observations exist at the scales to be generated, so the question is where the training
examples come from. Three sources were tried. Numbers are CRPS in mm (lower is better);
details in Appendix B.

**1. Coarser scales of GRACE itself (octave transfer): not viable.** A network trained to add
one octave of detail where GRACE resolves it was applied one octave finer.
- At its training scale it was the best method (20.5 against 45.9 for the spectral baseline).
- One octave finer it was worse than the baseline (52.5 against 48.7) and far too confident.
- A network trained at the finer scale itself reached 24.8, so the detail is learnable there.
- The same happened on WaterGAP, which has no GRACE noise or filtering.

Fine detail must be learned from examples at that scale.

**2. Hydrological models alone, applied to GRACE: viable. This design.** A network trained on
ten ISIMIP3a models (blurred model in, that model's detail out) was given real GRACE at scales
GRACE can check.
- It beat the spectral baseline by 7–11 mm (24.0 against 33.8 at 8 → 16; 30.3 against 38.1 at
  16 → 32, land-only GRACE), with correlations of 0.64–0.79 with GRACE's real detail.
- Ten models did better than WaterGAP alone (24.0 against 28.0 at 8 → 16).
- Its spread was too narrow (spread/error about 0.65). The stochastic sampler and a spread
  factor fitted on GRACE raised it to 0.87–0.89; a factor fitted on the models alone gave
  0.72–0.81.
- It stayed 7–10 mm behind a network trained on GRACE itself.

**3. One model's detail for the same month as the GRACE input (paired training): not viable.**
To remove the mismatch between model inputs in training and GRACE inputs in use, the
ten-model network was fine-tuned on pairs: GRACE in, WaterGAP's detail for that month out.
- Against WaterGAP's detail it improved (12.0 and 12.8 against 18.0 and 20.9).
- Against GRACE's own detail it got worse (23.3 and 38.0 against 18.6 and 33.0), with half the
  detail power GRACE has and a spread half its error.

The reason is the training pair. With models alone, input and target belong to the same field,
and the network learns a general relation that it applies to whatever coarse field it is
given. In paired training the target is another source's state for that month, so the network
learns to return WaterGAP's detail whatever GRACE shows. Over land GRACE's detail at these
scales is about twice as strong as WaterGAP's.

**Method lessons carried over.**
- EDM's default noise schedule was unsuitable here; the widened schedule is used (§3).
- The denoiser diagnostic (`scripts/diagnose_sampler.py`) is run on every trained network.
- The stochastic sampler improves both accuracy and spread and is the default.
- A method is tested at scales GRACE can check before it is trusted at finer ones.

## 2. Scales and grids

**Blurring.** Degree l has a half-wavelength of about 20,000 km / l. A field is limited to
degree L with a cosine taper w_L (1 up to 0.8 L, cosine roll-off to 0 at L), which avoids the
ringing of a sharp cut.

**Steps.** One step raises the resolution from an input degree to a target degree. The
*detail* is the target-limited field minus the input-limited field.

| step | degrees | finest half-wavelength | grid (lat × lon) | GRACE can check? |
|---|---|---|---|---|
| R1 | 4 → 8 | ~2500 km | 32 × 64 | yes |
| R2 | 8 → 16 | ~1250 km | 64 × 128 | yes |
| R3 | 16 → 32 | ~625 km | 128 × 256 | yes |
| R4 | 32 → 64 | ~310 km | 256 × 512 | no |
| R5 | 64 → 128 | ~160 km | 512 × 1024 | no |

GRACE has clean signal to about degree 40 (Appendix A), so it supplies degrees up to 32 and
everything finer is generated. Steps may also span two octaves (8 → 32, 32 → 128).

**Grids.** A field with target degree L is drawn on a global Driscoll–Healy grid of 4L × 8L
points: four pixels per half-wavelength at every step. The network is fully convolutional, so
one architecture serves every grid size.

**Normalisation.** Each map is divided by the area-weighted RMS of its input; predictions are
multiplied back.

**Rotations.** In training, each month is rotated by a random rotation of the sphere (exact in
spherical harmonics) before it is drawn on the grid; masks rotate with it. This multiplies the
training months, spreads grid distortion over all features, and prevents memorising geography.
Scoring uses unrotated fields.

## 3. Problem and model

**Input (decision 1).** The GRACE field of one month, limited to the input degree, on the
step's grid, normalised. Nothing else: no precipitation, temperature or topography. Static
topography and a land/water mask are a possible later addition.

**Output.** An ensemble of fields at the target degree. The network generates only the detail;
the input is added back.

**Training (decision 2).** Only hydrological-model fields are used. A model field is limited to
the input degree to form the input, and the network learns to generate the detail of that same
field. Input and target always belong together, so what the network learns is a statistical
relation: the fine detail that goes with a given large-scale field. GRACE is not used in
training.

**Application to GRACE.** The network receives the GRACE field limited to the input degree and
generates detail for it. Because inputs are normalised by their own RMS, the detail scales
with the field it is given: a strong observed signal gets correspondingly strong detail.

**Consistency projection (decision 4).** Blurring an output back to the input degree must
return the input. For every degree where the input taper is non-zero, the output's
coefficients are replaced by input / taper; all finer degrees are free. This is an exact linear
projection (`sh.project_consistent`), applied to every final sample of every method. GRACE
alone determines the coarse part of the product; the models contribute a relation, not their
own state for that month.

**Network.** 2D U-Net, two input channels (noisy detail, input field), one output; base width
64, multipliers (1, 2, 2, 4), two residual blocks per level, GroupNorm, SiLU, Fourier noise
embedding, no attention; 16.3 M parameters. Padding wraps in longitude and reflects across the
poles with a 180° shift.

**Diffusion (decision 5).** EDM preconditioning and loss, pixel weights = cell area × loss
mask.
- Training noise: ln σ ~ N(−0.4, 1.6²) (the widened schedule), σ_max = 20.
- Sampler: 32 Heun steps from σ_max to 0.002 (ρ = 7), stochastic (EDM Algorithm 2,
  churn 20).
- Ensemble members differ in their noise. The ensemble mean is the single best-estimate map.

**Direct network (baseline).** The same U-Net trained with a weighted squared error to predict
the detail in one pass. It gives the conditional mean: one smooth map, no uncertainty.

**Spread calibration (decision 6).** Each member's deviation from the ensemble mean is scaled
by a factor k per step, which leaves the coarse degrees exact.
- At the target scales k cannot be fitted on GRACE. It is fitted on hydrological models
  withheld from training (§7, level 3).
- It is then enlarged by the extra margin GRACE needs relative to the models at scales where
  both can be assessed. So far: model-fitted 1.1 and 1.0, GRACE-fitted 1.2 and 1.2 at 8 → 16
  and 16 → 32, a margin of 10–20%.
- That this margin holds at finer scales is an assumption and is reported as one.

**What the spread means.** The ambiguity of the detail given the large-scale field, plus the
differences between the models. It does not include errors common to all models.

## 4. Data

**GRACE/GRACE-FO (input).** ITSG-Grace2018 (Apr 2002 – Jun 2017, 162 months) and
ITSG-Grace_operational (Jun 2018 – Sep 2026, 98 months), unconstrained monthly solutions to
degree 120.
- C20, and C30 where given, replaced by GSFC TN-14 SLR values; each calendar month takes the
  TN-14 row overlapping it most.
- GIA removed with the Caron et al. (2018) expectation model.
- Anomalies relative to 2004–2009; equivalent water height with PREM load Love numbers;
  degrees 0 and 1 dropped.
- Two versions kept, unfiltered and DDK5-filtered, to degree 64. Which one is the input is an
  open question (§10); experiments so far report both.
- Check of the corrections (Gaussian 300 km trends, mm/yr, raw → corrected): Hudson Bay
  +31.2 → −0.6, Scandinavia +21.8 → +0.1, Türkiye −2.1 → −1.5.

**Land-only GRACE.** The models have no ocean and no ice sheets. GRACE inputs are therefore set
to zero over ocean, Greenland and Antarctica before use (`prep.land_only`: synthesise at degree
128, mask, expand again, zero degrees 0 and 1), so they share the models' domain.

**Hydrological models (training data, decision 3).** Ten ISIMIP3a water-sector models,
GSWP3-W5E5 forcing, `obsclim`, `histsoc`, monthly TWS 1901–2019 at 0.5°: CWatM, H08, HydroPy,
JULES-ES-VN6P3, JULES-W2, LPJmL5-7-10-fire, MIROC-INTEG-LAND, ORCHIDEE-MICT, WaterGAP2-2e and
WEB-DHM-SG (`fetch isimip`, `prep isimip`, files `isimip_<model>.npz`; `twin.npz` is WaterGAP).
- Expanded to degree 128; anomalies relative to 2004–2009; degrees 0 and 1 dropped.
- Cells whose storage drifts by more than 1 m between 1901–1910 and 2010–2019 are set to zero
  anomaly (unbounded snow accumulation on glaciers, point-like urban groundwater abstraction):
  68 cells (WEB-DHM-SG) to 5,603 (JULES-ES).
- Noted and kept: JULES-ES varies 3–4× more than the others (real variability); CWatM and
  MIROC show early-century spin-up offsets.

**GLWS2.0 (benchmark).** GRACE/GRACE-FO assimilated into WaterGAP (Gerdener et al. 2023). It is
built from GRACE and one of our models, so agreement is partly by construction: a benchmark
against the established method, not a validation. Compared at the same resolution. *To do:
obtain and read the paper, confirm grid and period, download the data.*

**Masks.**
- Test box (Türkiye, 30–45° N, 20–50° E) plus a margin of one target half-wavelength:
  excluded from every training loss and from validation scores.
- Greenland and Antarctica (south of 60° S): excluded from losses and scores.
- Scores are computed over land only.

**Splits.** Blocks of months, never random months.

| data | use | period | months |
|---|---|---|---|
| models | training | 1901–1989 | 1068 each |
| models | validation | 1990–2001 | 144 each |
| models | reserved | 2002–2019 | 216 each |
| GRACE | spread factor at coarse scales | 2002-04 to 2019-12 | 179 |
| GRACE | validation at coarse scales | 2020-01 to 2022-12 | 36 |
| GRACE | test, not yet looked at | 2023-01 onward | 45 |

GRACE is never used to train the network. The months 2015-01 to 2017-06 were also used as
validation in the paired experiment (Appendix B.5).

## 5. Baselines

1. **input**: the input field with no added detail. The floor.
2. **spectral baseline** (`spectral-gauss`): fit a power law to the degree variance of the
   input, extrapolate it to the missing degrees, add random coefficients with that spectrum.
   Right power, no learned structure.
3. **climatological spectral baseline** (`spectral-gauss-clim`): the same with the mean
   training spectrum.
4. **direct network** (`regress`): §3.
5. **trained on GRACE** at coarse steps: an upper bound that does not exist at the target
   scales.

## 6. Metrics

On validation months, unrotated, weighted by cell area over land outside the test box and ice
sheets, in mm; ensembles of 32 members.
- **CRPS** (fair ensemble estimator), the main score.
- **RMSE** of the ensemble mean, and its **correlation** with the true detail.
- **Spread/error**: ensemble standard deviation over ensemble-mean RMSE, times √((M+1)/M).
  1 is honest; below 1 is over-confident.
- **Coverage** of the central 50% and 90% ensemble intervals (randomised PIT).
- **Detail power ratio**: generated over true detail power.
- **Consistency**: RMS difference between the re-blurred output and the input.
- **Paired comparisons**: per-month CRPS difference with a 95% interval from a moving-block
  bootstrap (6-month blocks for GRACE, 12 for the models).

Not yet implemented: a stripe metric (north–south against east–west power of the added detail)
and a classifier two-sample test of generated against real detail.

## 7. Checks and experiments

### 7.1 Five levels of evidence (decision 7)

Below degree 32 the product equals GRACE by construction. Above it no global observation
exists to score against; this holds for any downscaled GRACE product. The evidence is
therefore assembled at five levels, none sufficient alone.

1. **Held-out years of the training models, at the target scales.** Did the network learn the
   task? Gives an upper bound.
2. **GRACE's own detail at coarse steps (8 → 16, 16 → 32).** The same method applied where
   GRACE can check it. The only global test against observations. *Done* (Appendix B.3, B.4).
3. **Leave one model out, at the target scales.** Train on the other models, score on the
   withheld one, for several models in turn. Stands in for the test against observations that
   cannot be made: if the models did not agree on the relation at fine scales, the product
   could not be trusted there. Also yields the spread factor (§3).
4. **GLWS2.0 at the target scales.** Benchmark; reported as a comparison, no pass or fail.
5. **Independent data in Türkiye.** Well levels, GNSS loading or basin water budgets: data that
   involve neither GRACE nor the models. The only test of the generated detail against
   independent observations. Data to be found (§10).

**Success** for levels 1–3: better than the baselines and the direct network in CRPS, with a
bootstrap interval excluding zero, and spread/error between 0.85 and 1.15. For level 5: closer
to the independent data than coarse GRACE alone, and ideally at least as close as the models
and GLWS.

**The claim this supports.** The product contains the fine detail that hydrological models
associate with the large-scale field GRACE observes, with a stated uncertainty; the method is
verified against GRACE between about 600 and 1250 km; below 600 km it agrees with independent
observations over Türkiye to a stated degree. It does not support the claim that the product
is the true water storage at 160 km.

### 7.2 Order of work (decision 8)

1. **Patch training.** Whole-globe training at degree 128 does not fit the 32 GB card at a
   useful batch size (16 → 32 already used 29.6 GB at batch 32). Train on cut-out patches, with
   care at the patch edges.
2. **Target-scale networks** on the ten models: one step (32 → 128) and two steps
   (32 → 64, 64 → 128), compared at level 1; the direct network alongside.
3. **Leave-one-model-out** runs at the target scales (level 3). The number of withheld models
   is set by the compute available; at least three structurally different ones.
4. **Spread factor** from step 3 plus the coarse-scale margin.
5. **Product** for the GRACE/GRACE-FO record; comparison with GLWS2.0 (level 4).
6. **Türkiye** (level 5), then the reserved GRACE months.

## 8. What exists and what needs building

Exists: data pipeline for GRACE and the ten models; U-Net, EDM, both samplers; baselines;
metrics; evaluation with spread inflation; the diagnostic and the calibration script; training
on several datasets; steps spanning two octaves and chaining of steps.

Also in the repository but not part of this design: paired samples and fine-tuning
(`data.PairedSampler`, `train --init --target`, `evaluate --target`, `scripts/stage1*.sh`), kept
as the record of the paired experiment.

To build:
- Patch training for fine scales.
- Leave-one-model-out run script.
- `fetch` / `prep` for GLWS2.0.
- Stripe metric and classifier two-sample test (§6).

## 9. Code layout

```
src/grace_diffusion/
  fetch.py       download GRACE series, WaterGAP, ISIMIP3a models
  prep.py        build datasets: GRACE (raw, DDK5, land-only), hydrological models
  sh.py          coefficients, tapers, rotations, synthesis/analysis, consistency projection
  grids.py       grids, masks (test box, ice sheets, land), rotation of masks
  data.py        splits, samplers over (month, step, rotation), several datasets
  model/         U-Net; EDM loss, samplers; direct network
  baselines.py   input, spectral baselines
  metrics.py     CRPS, spread/error, coverage, power ratio, consistency
  train.py
  evaluate.py
scripts/         run scripts, diagnostics, calibration, exploration
tests/           unit tests on synthetic fields
docs/            this document
reports/         written reports and the proposal (LaTeX)
survey/          literature survey (LaTeX) and notes
```

## 10. Open questions

For the supervisor:
1. **Independent data for Türkiye** (level 5): wells, GNSS loading, basin budgets, regional
   products.
2. **The set of hydrological models** and the forcing protocol (now ten ISIMIP3a models,
   GSWP3-W5E5).
3. **GRACE product and filter** for the input: keep ITSG; DDK5 or unfiltered?
4. **Target resolution:** degree 128, or finer later?
5. **Is the limited claim of §7.1 an acceptable aim?**

Technical:
- How to handle patch edges in patch training.
- Whether the coarse-scale spread margin can be supported by anything beyond the two steps
  measured so far.
- Whether the paired months can help calibrate amplitude or spread without changing what
  detail the network draws.
- Static inputs (topography, land/water mask) as a later addition.
- Degree 1 (geocentre) is still left out.

## Appendix A — Data findings (2026-11-13)

**Where GRACE's signal ends** (`scripts/explore_filters.py`, ITSG, 260 months,
`runs/explore/degree_amplitudes.png`). The median monthly anomaly rises above the noise only up
to about degree 40. From ~45 upward it equals the noise (month-to-month difference / √2), which
matches the ITSG formal errors. Over the test box, the degree-60 field of April 2010 is
north–south stripes of ±400 mm against a real signal of ~100–150 mm; at degree 96 the stripes
reach ±3,000 mm.

**Filters per band** (`scripts/explore_ddk.py`, `runs/explore/filters_by_band.csv`). Signal
kept and residual signal/noise power:

| filter | 16–30 kept | 16–30 S/N | 31–60 kept | 31–60 S/N |
|---|---|---|---|---|
| none | 1.00 | 11.5 | 1.00 | 0.29 |
| Gaussian 300 km | 0.42 | 12.5 | 0.06 | 1.3 |
| DDK3 | 0.87 | 21 | 0.16 | 31 |
| DDK4 | 0.90 | 18 | 0.22 | 22 |
| DDK5 | 0.94 | 13.5 | 0.41 | 9 |

No filter keeps any signal above degree 60. DDK removes the stripes in the test box; Gaussian
300 km leaves elongated north–south features. The noise proxy includes real month-to-month
change, so signal estimates are lower bounds.

## Appendix B — Results so far

CRPS in mm, validation months (GRACE 2020–2022, WaterGAP 1990–2001), land outside the test box
and ice sheets, 32 members. Scores: `runs/eval/*.csv`. Full account:
`reports/phase1/phase1_report.pdf`.

**B.1 Octave transfer on GRACE DDK5** (`scripts/phase1.sh`, `phase1b_noise.sh`). Networks
trained on R1 + R2.

| method | R2 (trained) | R3 (one octave finer) | R3 spread/error | R3 − spectral baseline (95%) |
|---|---|---|---|---|
| input | 62.9 | 64.0 | — | |
| spectral baseline | 45.9 | 48.7 | 1.23 | |
| direct network | 26.3 | 71.1 | — | |
| diffusion, default schedule | 37.2 | 45.5 | 0.82 | −3.3 [−3.7, −2.9] |
| diffusion, widened schedule | 20.5 | 52.5 | 0.37 | +3.8 [+2.7, +4.6] |
| diffusion, log-uniform schedule | 21.6 | 51.0 | 0.49 | +2.3 [+1.7, +2.8] |
| diffusion trained on R3 itself | — | 24.8 | 0.88 | |

The default-schedule network barely used its input at high noise (denoiser correlation with
the true detail at σ = 10: 0.26 at R1, 0.40 at R2, against 0.89 and 0.92 for the direct
network); only ~0.2% of its training batches had σ > 10. Its small lead at R3 came from being
close to generic noise.

**B.2 Octave transfer on WaterGAP** (`scripts/phase1c_twin.sh`). Trained on R1 + R2.

| step | spectral baseline | default schedule | widened schedule | widened − baseline (95%) | spread/error |
|---|---|---|---|---|---|
| R2 (trained) | 16.1 | 9.4 | 7.2 | −8.9 [−9.1, −8.6] | 0.85 |
| R3 | 14.4 | 13.0 | 14.5 | +0.1 [−0.2, +0.3] | 0.48 |
| R4 | 15.0 | 14.1 | 15.5 | +0.5 [+0.4, +0.6] | 0.43 |
| R5 | 15.5 | 14.7 | — | | |

**B.3 Trained on ten models, applied to land-only GRACE** (`scripts/g1_models.sh`). Trained on
R2 + R3.

| input | step | input only | spectral baseline | model-trained | spread/error | GRACE-trained | model-trained − baseline (95%) |
|---|---|---|---|---|---|---|---|
| DDK5 | R2 | 45.8 | 33.8 | 24.0 | 0.66 | 13.7 | −9.9 [−11.5, −8.5] |
| DDK5 | R3 | 50.7 | 38.1 | 30.3 | 0.64 | 21.8 | −7.9 [−8.6, −7.1] |
| unfiltered | R2 | 46.5 | 34.3 | 24.9 | 0.67 | 15.4 | −9.4 [−11.0, −8.0] |
| unfiltered | R3 | 54.3 | 40.6 | 33.5 | 0.64 | 26.1 | −7.1 [−7.8, −6.3] |

WaterGAP alone as training data scored 28.0 at R2 (DDK5). On full GRACE fields (ocean and ice
included) the model-trained network scored 41.0 against 48.7 at R3 and 44.3 against 45.9 at R2.

**B.4 Spread calibration of the ten-model network** (`scripts/g1_calibrate.sh`,
`scripts/calibrate_spread.py`). Land-only GRACE DDK5, validation 2020–2022.

| version | R2 CRPS | R2 spread/error | R3 CRPS | R3 spread/error |
|---|---|---|---|---|
| deterministic sampler (B.3) | 24.0 | 0.66 | 30.3 | 0.64 |
| stochastic sampler (churn 20) | 22.6 | 0.74 | 29.8 | 0.72 |
| + inflation fitted on models (1.1, 1.0) | 22.3 | 0.81 | 29.8 | 0.72 |
| + inflation fitted on GRACE training years (1.2, 1.2) | 22.1 | 0.89 | 29.3 | 0.87 |

On the models' own held-out years the network was also too narrow with the deterministic
sampler (0.71 and 0.85) and close to calibrated with the stochastic one (0.93 and 0.98).

**B.5 Paired design, short first pass** (`scripts/stage1_short.sh`, 2026-11-18). Land-only
GRACE DDK5 in, WaterGAP detail as target; 22 validation months (2015-01 to 2017-06);
stochastic sampler; 32 members. CRPS in mm.

| method | vs WaterGAP R2 | vs WaterGAP R3 | vs GRACE R2 | vs GRACE R3 | spread/error vs GRACE | corr. vs GRACE |
|---|---|---|---|---|---|---|
| input | 21.5 | 19.5 | 40.2 | 53.0 | | |
| spectral baseline (WaterGAP mean spectrum / input spectrum) | 16.4 | 14.9 | 30.0 | 40.1 | 0.83 / 0.85 | 0.15 / 0.29 |
| ten-model network, not fine-tuned | 18.0 | 20.9 | 18.6 | 33.0 | 0.77 / 0.72 | 0.79 / 0.65 |
| fine-tuned on paired months (10k steps, lr 5e-5) | 12.0 | 12.8 | 23.3 | 38.0 | 0.48 / 0.48 | 0.72 / 0.57 |
| paired months only (30k steps) | 12.8 | 13.5 | 25.5 | 39.4 | 0.46 / 0.49 | 0.68 / 0.57 |

Fine-tuned minus not fine-tuned: −6.0 [−6.5, −5.5] and −8.1 [−8.9, −6.7] against WaterGAP;
+4.6 [+3.5, +5.5] and +5.0 [+3.7, +5.6] against GRACE. The fine-tuned network generates 52% and
39% of GRACE's detail power. Detail RMS over land in these months: GRACE 56 and 77 mm, WaterGAP
32 and 33 mm. Paired training teaches the network to return WaterGAP's detail whatever GRACE
shows, so the product moves towards the model and away from the observations. Not run:
WaterGAP-only pre-training, the one-step variant, the direct network.

**B.6 Tests without training** (2026-11-18). Ten-model network (`runs/isimip10-r23`),
land-only GRACE DDK5, against GRACE's own detail.

*The models' own detail for the month* (`scripts/test_model_month_detail.py`, 22 months
2015-01 to 2017-06): GRACE's blurred field completed with each model's detail for that month,
as a ten-member ensemble.

| method | R2 CRPS | R3 CRPS | corr. |
|---|---|---|---|
| spectral baseline | 30.0 | 40.1 | 0.15 / 0.29 |
| ten models' detail for the month | 21.5 | 38.7 | 0.73 / 0.53 |
| the same, scaled to the GRACE input | 20.6 | 37.9 | 0.75 / 0.53 |
| network (GRACE-only input) | 18.6 | 33.0 | 0.79 / 0.65 |

*Mixing* (`scripts/test_mixing.py`, same months): network ensemble mean replaced by
a·(network mean) + b·(ten-model mean detail), weights cross-fitted over two halves. CRPS 16.5
and 32.0, i.e. −2.2 [−2.4, −1.9] and −1.0 [−1.2, −0.9] against the network alone; weights about
(0.7, 0.55) at R2 and (0.95, 0.35) at R3. The models' monthly fields are worse than the network
alone but complementary to it, less so at the finer step.

*Coherence in time* (`scripts/test_time_coherence.py`, 36 months 2020–2022, 8 members): RMS
month-to-month change of the detail (mm) and correlation between consecutive months.

| | R2 change | R2 corr. | R3 change | R3 corr. |
|---|---|---|---|---|
| GRACE's own detail | 21.6 | 0.94 | 24.5 | 0.94 |
| members, fresh noise each month | 47.4 | 0.65 | 60.8 | 0.53 |
| members, the same noise every month | 23.6 | 0.91 | 19.9 | 0.95 |

Sharing the starting noise of each member across months restores realistic persistence at no
cost in CRPS (22.1 against 22.7 and 28.9 against 29.9). It is now the default in `evaluate.run`.

*Model pattern, GRACE amplitude* (`scripts/test_gain_baseline.py`, same 22 months): each
model's detail for the month multiplied by a per-pixel gain fitted so that the model's blurred
field matches GRACE's over 2002–2014. Ten-member ensemble: 23.0 and 41.0 mm (correlation 0.76
and 0.55); with an additional global rescaling per month 21.0 and 38.7; one global scale alone
20.6 and 37.9. The fitted gains have medians of 0.7–1.1 over land, so the models are not GRACE
at half strength: GRACE's larger amplitude is mostly signal absent from the models. JULES-ES
receives a gain of zero over most of the land.

**Extra input channels** (`data.Extras`, `scripts/wall.sh`): previous months of the blurred
field, the other models' detail for the month, the models' seasonal climatology of the detail,
land mask and latitude can each be switched on per run (`train --extras`). GRACE-only input is
no longer a fixed decision (2026-11-18); these variants are tested at the coarse steps first.
