# 00 — Design

*Agreed 18 November 2026. This document replaces the two earlier designs (octave transfer;
detail trained on models only), which remain in the git history. Their results are summarised
in §1 and Appendix B, and written up in `reports/phase1/phase1_report.pdf`. Nothing in §3–§8
that is marked "to build" exists yet.*

**In one sentence:** the network takes a coarse GRACE/GRACE-FO map, adds fine detail learned
from the WaterGAP hydrological model, and leaves GRACE's own large-scale picture untouched.

The plan is recorded here before implementation; anything that changes it is recorded here
first. The literature behind the project is in `survey/grace_diffusion_survey.pdf`. Data
findings that shaped it are in Appendix A.

## Decisions

| # | point | decision |
|---|---|---|
| 1 | input | GRACE only: one coarse map of a single month |
| 2 | loss | on the fine detail only; the coarse part is fixed to GRACE |
| 3 | target | WaterGAP v2.2e; GLWS2.0 is a benchmark; nine other models are a side check |
| 4 | training | pre-train on WaterGAP alone, then fine-tune on paired GRACE–WaterGAP months |
| 5 | model | conditional diffusion, with a direct (regression) network as baseline |
| 6 | checks | split inside 2002–2019; four checks, from WaterGAP hold-out to independent data |
| 7 | resolution | degree 128; one step or two decided by experiment |

Standing decisions from earlier work, still in force:
- Spatial only: each month is an independent map, no time modelling.
- Regional test on Türkiye and surroundings: box 30–45° N, 20–50° E, held out of all training.
- Data sources are provisional until confirmed with the supervisor; swapping them must stay
  cheap.
- Whole-globe images on scale-matched grids with random rotations (§2).

## 1. Background: what was tried, and why this design

Two approaches were tested before this one. Numbers are CRPS in mm (lower is better), on
validation months; details in Appendix B.

**Octave transfer (failed).** A network was trained to add one octave of detail at coarse
scales, where GRACE resolves both input and target, and applied one octave finer.
- At its training scale it was the best method (20.5 against 45.9 for the spectral baseline).
- One octave finer it was worse than the baseline (52.5 against 48.7) and far too confident.
- A network trained at the finer scale itself reached 24.8, so the detail is learnable there.
- The same happened on WaterGAP, which has no GRACE noise or filtering.

Conclusion: fine detail must be learned from examples at that scale. GRACE has none below its
resolution; hydrological models do.

**Detail trained on models only (partly worked).** A network was trained on ten ISIMIP3a
models (blurred model in, model detail out) and then given real GRACE.
- At scales GRACE can check, it beat the spectral baseline by 8–10 mm (24.0 against 33.8 at
  8→16; 30.3 against 38.1 at 16→32, land-only GRACE).
- Its spread was too narrow (spread/error about 0.65). A stochastic sampler and spread
  inflation raised this to 0.87–0.89, but only with a factor fitted on GRACE. The factor fitted
  on the models, which is the only one available at fine scales, gave 0.72–0.81.
- It stayed 7–10 mm behind a network trained on GRACE itself.

The remaining gap comes from a mismatch: the network trains on blurred *model* fields and is
then handed *GRACE*, which it has never seen. The present design removes that mismatch by
putting real GRACE on the input side during training (§7).

**Method lessons carried over.**
- EDM's default noise schedule was unsuitable here; the widened schedule is used (§3).
- The denoiser diagnostic (`scripts/diagnose_sampler.py`) is run on every trained network.
- The stochastic sampler improves both accuracy and spread and is now the default.

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
everything finer is generated. Steps may also span two octaves (8 → 32, 32 → 128); see §7.

**Grids.** A field with target degree L is drawn on a global Driscoll–Healy grid of 4L × 8L
points: four pixels per half-wavelength at every step. The network is fully convolutional, so
one architecture serves every grid size.

**Normalisation.** Each map is divided by the area-weighted RMS of its input; predictions are
multiplied back.

**Rotations.** In training, each month is rotated by a random rotation of the sphere (exact in
spherical harmonics) before it is drawn on the grid; masks rotate with it. This multiplies the
training months, spreads grid distortion over all features, and prevents memorising geography.
For paired samples (§7) the GRACE input and the WaterGAP target get the same rotation. Scoring
uses unrotated fields.

## 3. Problem and model

**Input (decision 1).** The GRACE field of one month, limited to the input degree, on the
step's grid, normalised. Nothing else: no precipitation, temperature or topography. Static
topography and a land/water mask are a possible later addition.

**Output.** An ensemble of fields at the target degree. The network generates only the detail;
the input is added back.

**Loss on the detail only (decision 2).** The loss compares the generated detail with the
target's detail, i.e. only the part finer than the input. The coarse part never enters the
loss.
- A whole-map loss would pull GRACE's large-scale picture towards the model, for example
  weakening groundwater depletion that GRACE sees and the model misses.
- Known weakness: where the model misses a signal, the network adds no fine detail for it.
  The large-scale signal is kept but not sharpened.

**Paired samples.** For a paired month the coefficients are spliced: degrees below the input
degree from GRACE, degrees from the input degree upward from WaterGAP (`data.splice_coeffs`).
The input is then exactly the blurred GRACE field, the detail above the input degree is
WaterGAP's alone, and the target is consistent with the input.

**Consistency projection.** Blurring an output back to the input degree must return the input.
For every degree where the input taper is non-zero, the output's coefficients are replaced by
input / taper; all finer degrees are free. This is an exact linear projection
(`sh.project_consistent`), applied to every final sample of every method. With GRACE as input,
it guarantees that GRACE alone determines the coarse part of the product.

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
- Spread inflation about the ensemble mean (a factor per step) is available if needed; it keeps
  the coarse degrees exact. Any factor is fitted on validation months, never on test months.

**Direct network (baseline, decision 5).** The same U-Net trained with a weighted squared
error to predict the detail in one pass. It gives the conditional mean: one smooth map, no
uncertainty. It answers whether diffusion buys anything over a plain network.

**What the spread means.** With one target model, the spread expresses how ambiguous the detail
is given GRACE, assuming WaterGAP's fine-scale behaviour is right. It does not know about
WaterGAP's own errors.

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

**Land-only GRACE.** WaterGAP has no ocean and no ice sheets. GRACE inputs are therefore set to
zero over ocean, Greenland and Antarctica before use (`prep.land_only`: synthesise at degree
128, mask, expand again, zero degrees 0 and 1), so input and target share a domain.

**WaterGAP v2.2e (target, decision 3).** GSWP3-W5E5 forcing, historical human water use,
monthly TWS 1901–2019 at 0.5°, land only; expanded to degree 128 (`prep.build_model`).
- Cells whose storage drifts by more than 1 m between 1901–1910 and 2010–2019 are set to zero
  anomaly (about 2600 cells: unbounded snow accumulation on glaciers, point-like urban
  groundwater abstraction).
- Anomalies relative to 2004–2009, degrees 0 and 1 dropped, as for GRACE.

**Nine other ISIMIP3a models (side check).** CWatM, H08, HydroPy, JULES-ES-VN6P3, JULES-W2,
LPJmL5-7-10-fire, MIROC-INTEG-LAND, ORCHIDEE-MICT, WEB-DHM-SG; same forcing and protocol, same
processing (`isimip_<model>.npz`). Not training targets. Noted: JULES-ES varies 3–4× more
than the others (real variability); CWatM and MIROC show early-century spin-up offsets.

**GLWS2.0 (benchmark, decision 3).** GRACE/GRACE-FO assimilated into WaterGAP (Gerdener et al.
2023). It is built from the same two ingredients as our product, so agreement is partly by
construction: it is a benchmark against the established method, not a validation. Compared at
the same resolution (GLWS smoothed to our target degree). *To do: obtain and read the paper,
confirm grid and period, download the data.*

**Masks.**
- Test box (Türkiye, 30–45° N, 20–50° E) plus a margin of one target half-wavelength:
  excluded from every training loss and from validation scores.
- Greenland and Antarctica (south of 60° S): excluded from losses and scores.
- Scores are computed over land only.

**Splits.** Blocks of months, never random months.

| data | use | period | months |
|---|---|---|---|
| WaterGAP alone | pre-training | 1901–1989 | 1068 |
| WaterGAP alone | pre-training validation | 1990–2001 | 144 |
| paired GRACE + WaterGAP | fine-tuning | 2002-04 to 2014-12 | 140 |
| paired GRACE + WaterGAP | validation | 2015-01 to 2017-06 (end of GRACE) | 22 |
| paired GRACE + WaterGAP | test | 2018-06 to 2019-12 (GRACE-FO) | 17 |

WaterGAP ends in 2019, so paired months exist only for 2002–2019 and the earlier GRACE
validation years (2020–2022) cannot serve this design. Pre-training never sees the paired
period. GRACE months after 2019 have no target; they are where the finished product is
applied.

## 5. Baselines

1. **input**: the input field with no added detail. The floor.
2. **spectral baseline** (`spectral-gauss`): fit a power law to the degree variance of the
   input, extrapolate it to the missing degrees, add random coefficients with that spectrum.
   Right power, no learned structure.
3. **climatological spectral baseline** (`spectral-gauss-clim`): the same with the mean
   training spectrum.
4. **direct network** (`regress`): §3.
5. **pre-trained only**: the diffusion network before fine-tuning (the previous approach).

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
  bootstrap (6-month blocks).

Not yet implemented: a stripe metric (north–south against east–west power of the added detail)
and a classifier two-sample test of generated against real detail.

## 7. Training and experiments

### 7.1 Training in two phases (decision 4)

1. **Pre-train** on WaterGAP alone: blurred WaterGAP in, WaterGAP detail out. The network
   learns what WaterGAP's fine detail looks like.
2. **Fine-tune** on paired months: real GRACE in (land-only, limited to the input degree),
   WaterGAP detail of the same month out. The network adapts to real GRACE inputs.

Settings as before unless a test says otherwise: AdamW, learning rate 2×10⁻⁴ with warm-up,
batch 32, gradient clipping, EMA 0.999, best checkpoint by validation loss. The fine-tuning
learning rate and length are to be chosen on the paired validation months; overfitting to 140
months is the main risk and is watched through the validation loss and the denoiser
diagnostic.

Fallbacks: paired months only (no pre-training), or both kinds of sample mixed in one run.

### 7.2 Checks (decision 6)

From weakest to strongest:

1. **Held-out years against WaterGAP.** Did the network learn its task? Diffusion against the
   direct network and the spectral baseline; spread checked.
2. **GRACE's own detail at coarse steps.** The same two-phase training at 8 → 16 and 16 → 32,
   scored against the detail GRACE actually measured. This is the only global check against
   observations. To beat: 22.1 and 29.3 mm (the previous approach with the stochastic sampler
   and inflation; those numbers are for 2020–2022 and are recomputed on the new validation
   months for a like-for-like comparison).
3. **GLWS at fine scales.** Benchmark; reported as a comparison, with no pass or fail.
4. **Independent data in Türkiye.** Well levels, GNSS loading or basin water budgets: data that
   involve neither GRACE nor WaterGAP. The only real proof at fine scales. Data to be found
   (§10).

**Success** for checks 1 and 2: better than the spectral baseline and the direct network in
CRPS, with a bootstrap interval excluding zero, and spread/error between 0.85 and 1.15. For
check 4: closer to the independent data than coarse GRACE alone, and ideally at least as close
as WaterGAP and GLWS.

The validation and test periods are short (22 and 17 months), so intervals will be wide. Test
months and the Türkiye box are not touched until the design is frozen.

### 7.3 Order of experiments (decision 7)

Target: degree 128 (~160 km). Whether to reach it in one step (32 → 128) or two
(32 → 64 → 128) is decided by experiment. In the two-step version only the first step sees
real GRACE; the second is fine-tuned on the first step's output.

**Stage 1 — coarse scales.** Cheap, on grids we already handle, and GRACE can check the result
(check 2).

| variant | what it tests |
|---|---|
| A. pre-trained only | the reference |
| B. pre-train + fine-tune, two steps (8 → 16, 16 → 32) | does this design beat the previous one? |
| C. pre-train + fine-tune, one step (8 → 32) | one step against two |
| D. paired months only, no pre-training | is pre-training needed? |
| E. direct network, trained as B or C | does diffusion buy anything? |

Each run is roughly 1–4 hours. A WaterGAP-only pre-trained network exists for 4 → 8 and
8 → 16; pre-training at 16 → 32 and at 8 → 32 are new runs. A ten-model pre-trained network for
8 → 16 and 16 → 32 exists and can serve as an extra variant of B (pre-train on ten models,
fine-tune towards WaterGAP).

**Stage 2 — fine scales.** Take the stage 1 winner to 32 → 128. If one step and two steps are
close in stage 1, run both. Scored by checks 1 and 3.

**Stage 3 — Türkiye.** Check 4, on the test box, once independent data are in hand.

**Later, only if wanted:** degree 256 (~80 km); needs finer handling of memory and a judgement
on how much of WaterGAP's finest detail is believable.

## 8. What needs building

Built for stage 1 (2026-11-18): paired samples and splits (`data.PairedSampler`), steps
spanning two octaves (`8-32`), fine-tuning from a checkpoint (`train --init`), paired scoring
and chaining of two steps (`evaluate --target`, `chain` checkpoints), and the run script
`scripts/stage1.sh`. Still to build:
- **Patch training for fine scales.** Whole-globe training at degree 128 does not fit the 32 GB
  card at a useful batch size (16 → 32 already used 29.6 GB at batch 32). Stage 2 needs
  training on cut-out patches, with care at the patch edges.
- `fetch` / `prep`: GLWS2.0.
- Stripe metric and classifier two-sample test (§6).

## 9. Code layout

```
src/grace_diffusion/
  fetch.py       download GRACE series, WaterGAP, ISIMIP3a models
  prep.py        build datasets: GRACE (raw, DDK5, land-only), WaterGAP, other models
  sh.py          coefficients, tapers, rotations, synthesis/analysis, consistency projection
  grids.py       grids, masks (test box, ice sheets, land), rotation of masks
  data.py        splits, samplers over (month, step, rotation), several datasets
  model/         U-Net; EDM loss, samplers; direct network
  baselines.py   input, spectral baselines
  metrics.py     CRPS, spread/error, coverage, power ratio, consistency
  train.py
  evaluate.py
scripts/         run scripts (phase1*.sh, g1_*.sh), diagnostics, calibration, exploration
tests/           unit tests on synthetic fields
docs/            this document
reports/         written reports (LaTeX)
survey/          literature survey (LaTeX) and notes
```

## 10. Open questions

For the supervisor:
1. **Independent data for Türkiye** (check 4): wells, GNSS loading, basin budgets, regional
   products.
2. **GRACE product and filter** for the input: keep ITSG; DDK5 or unfiltered?
3. **Target resolution:** degree 128, or finer later?
4. **WaterGAP as the only target:** in the earlier test ten models did better than WaterGAP
   alone on GRACE (24.0 against 28.0 mm at 8 → 16). Should the other models return as targets
   once the paired design is working?

Technical:
- Fine-tuning length and learning rate; how quickly 140 paired months overfit.
- Static inputs (topography, land/water mask) as a later addition.
- Degree 1 (geocentre) is still left out.
- Whether the spread needs inflation after fine-tuning, and on what it is fitted.

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

## Appendix B — Results of the earlier approaches

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
