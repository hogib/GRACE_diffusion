# imonet: prior in-house work (notes for the GRACE + diffusion survey)

Source repo: `/home/hogib/Projects/imonet` (read-only survey, 2026-11-13; HEAD `bdeabd7`, 34 commits).
Working title (paper/report.tex:33-34): *Generating virtual GNSS stations from their neighbours with a
conditional diffusion model*, H. Oğuz Bolat (YTU Geomatics). `paper/paper.tex` is still a skeleton
(title literally `TITLE`, paper/paper.tex:27); the full draft is `paper/report.tex`. All results are on
the **validation period**; the test period has not been used (paper/report.tex:35-36, 1257-1258).

BibTeX for this work and its relevant references: `imonet.bib` (same folder), key `imonet`.

---

## What imonet does

### Problem formulation
- Generate **samples** of the daily E/N/U residual position series (mm) of a station that does not
  exist ("virtual station") at an arbitrary (lat, lon), from the K nearest real stations
  (docs/00-design.md:29-47).
- Motivation: a station's residual = shared part (common-mode error, loading, tectonic transients) +
  own part (monument/site noise, multipath). The best deterministic neighbour predictor only reduces the
  fast-part nRMSE from 0.73 to ~0.52, so regression gives series "too smooth and too quiet"
  (paper/report.tex:111-117; docs/00-design.md:12-27). The generative model takes the shared part from
  the neighbours and samples the station's own part.
- Only residuals are generated (trajectory: velocity, seasonals, steps are out of scope). **No absolute
  coordinates enter the model, only target-relative offsets**, to allow California → Türkiye transfer
  (docs/00-design.md:43-47).

### Data pipeline (src/imonet/build.py, cube.py, data.py)
- Source: Nevada Geodetic Laboratory IGS20 daily `.tenv3` series + station table + step catalogue
  (`imonet fetch`), SCEC CFM 7.0 for planted fault-slip events, Natural Earth borders.
- **Cube** (`Cube`, src/imonet/cube.py:27-48): `r` S×T×3 residuals (mm, NaN missing), formal σ, `avail`
  mask, station ids/coords, `split_day` (train/val/test), `split_sta` (normal/held-out), `exclude`
  (never used in training), kept earthquake offsets, `fallback_fit` flag, country code.
- Preprocessing: per station & component fit constant + rate + annual + semi-annual + steps on
  2008–2019; equipment steps removed, earthquake offsets kept; 5-robust-σ outlier screen vs a 61-day
  running median; noisy stations and co-located twins (<1 km) removed (paper/report.tex:247-262).
- California: 1251 stations × 6844 days (2008-01-01..2026-09-26). Splits: train 2008–2019, val
  2020–2021, test 2022+; 173 held-out stations **removed from training completely** (never a target,
  never a neighbour, never in the input network) (docs/00-design.md:63-79).
- Türkiye (test only, no tuning): 233 stations in a 35.8–42.2°N, 25.6–44.9°E box; fit to ≤2012, main
  scores 2013–2015 (dense national CORS period), sparse case 2016+ (~17 stations/day), in-fit control
  experiment; Kahramanmaraş window held out as a case study (docs/00-design.md:81-125).
- **Scaling**: each component divided by its robust scale (1.4826·MAD) of trainable cells (2.12, 2.24,
  5.60 mm), because SDs (9.1/15.3/10.3 mm) are dominated by El Mayor–Cucapah offsets (~200 mm)
  (docs/00-design.md:227-232).
- **Per-window detrending**: every series in a 256-day window (target and neighbours) has its own LSQ
  mean and trend removed in training and evaluation, because post-2019 residuals carry the trajectory
  fit's extrapolation drift (slow-part RMS doubles in validation years; mean+trend removal makes the
  periods indistinguishable; mean-only fails for Up) (paper/report.tex:325-359, Table `tab:drift`).
- **Sample** = (target station, 256-day window, thinned network) (docs/00-design.md:131-147):
  target drawn from normal stations with ≥50% usable days; network thinned; K = 16 nearest within
  300 km with ≥25% of window usable, padded+masked otherwise (src/imonet/network.py:26-27, 125-147);
  target's missing days excluded from the loss.
- **Network thinning as augmentation** (docs/00-design.md:149-188): 80% of samples thin the network
  to a median NN spacing log-uniform 10–100 km and drop stations closer than a min distance log-uniform
  1–100 km; 20% use the full network. Greedy minimum-separation ("disk") thinning reproduces Türkiye's
  spacing distribution (median 60 km, 90th pct 72–73 vs Türkiye 77 km); random subsampling keeps
  California's clustering and fails (table at docs/00-design.md:172-177). Thinnings precomputed
  (64 spacings × 4 seeds).
- **Planted synthetic transients** (paper/report.tex:487-505): with p = 0.8 a sample gets one transient
  (Gaussian footprint 10–300 km, uniform-direction or radial; 70% raised-cosine 5–120 d, 1–30 mm; 30%
  Pareto-tailed steps ≤50 mm) added to target and all neighbours *before* detrending; a **visibility
  guard** keeps it only if ≥2 neighbours move ≥1 mm, so the target never carries an excursion the input
  cannot explain → 49% of samples planted. Deliberately simpler than experiment B's fault-slip events
  (tests generalisation, not recall).

### What is conditioned on vs generated
- Conditioning: for each of K ≤ 16 neighbours, its 256-day E/N/U series, availability mask, formal σ,
  and geometry (east/north offset from target, distance); day-of-year sin/cos; optionally the target's
  **station noise level** (log fast-part robust scale relative to a typical station, per component, +
  "given" flag) (docs/00-design.md:250-267).
- Generated: the target's 256×3 window. Windows are generated independently; stitching is future work
  (docs/00-design.md:290-291).

### Model architecture (src/imonet/model/net.py, edm.py; paper/report.tex:433-452)
- **Neighbour encoder** (noise-level independent, computed once per sample at sampling time): per
  neighbour conv over series/availability/σ, then one residual block per U-Net level (stride-2 conv
  between levels), shared across neighbours; FiLM-style modulation (scale & shift) by a **geometry
  embedding**: offsets/100 km, log distance, sin/cos of offsets at wavelengths 10, 30, 100, 300 km
  (Fourier features of the local azimuthal-equidistant offset; src/imonet/cube.py:99-104).
- **Denoiser**: 1D U-Net over time, channels 64/128/192/256 at L, L/2, L/4, L/8, 2 res-blocks per level,
  modulated by the noise-level embedding. After every residual block (and the bottleneck) each time
  step **cross-attends to the K neighbours at the same time step** (4 heads) with a **learned null token**
  always in the keys, so padding and dropped conditioning are handled identically. Temporal
  self-attention at the bottleneck and two coarsest decoder levels. 11.1 M parameters.
- **Noise-level input**: small zero-initialised MLP (~70k params) added to the noise embedding; hidden
  for 30% of training samples so one model works with or without it.
- **Diffusion formulation**: EDM (Karras et al. 2022) preconditioning, σ_data = 1, c_noise = ln σ / 4,
  i.e. the network predicts the denoised signal through c_skip/c_out (neither ε- nor v-prediction).
  Training ln σ ~ N(−1.2, 1.2²); loss weight (σ²+σ_d²)/(σσ_d)², averaged over target days with data
  (src/imonet/model/edm.py:1-91).
- **CFG**: 10% of samples drop all neighbours (classifier-free guidance possible); tested, **not used**
  (w = 1.2 narrows the ensemble only by exaggerating the shared signal; docs/00-design.md:573-575).
- `regress` baseline = the same network called with zero input and zero noise level, trained with MSE
  (src/imonet/model/edm.py:136-152).

### Training setup (src/imonet/train.py:46-67; paper/report.tex:454-465)
- AdamW, lr 2e-4, 2k warm-up + cosine decay, batch 64, 100k steps, grad clip 1, bf16, EMA 0.999 (EMA
  weights are the model), `torch.compile`.
- Validation loss every 2k steps on 1024 fixed samples (held-out stations, validation days) at fixed
  σ ∈ {0.02, 0.1, 0.3, 1, 3, 10} with fixed noise → comparable across steps (src/imonet/train.py:40,
  91-108). Val loss 0.66 → 0.58.
- Compute: single RTX 3060 Ti; 0.12–0.14 s/step compiled, 5.7 GiB peak; 32 samples for ~500 target
  windows in 13.7 min (paper/report.tex:464-465; docs/00-design.md:279-282).
- Model versions: v1 (base), v2 (+noise-level input), v3 (+planted transients; the paper model),
  v3-s1 (second seed).

### Sampling
- Heun's 2nd-order solver on the probability-flow ODE, 32 steps, σ 80 → 0.002, ρ = 7; deterministic
  given the initial noise from a **seeded generator**, so paired runs (planted vs unplanted network)
  are exactly repeatable (src/imonet/model/edm.py:93-133). 32 samples per target window.

---

## Evaluation protocol

### Baselines (docs/00-design.md:293-375; src/imonet/baselines.py, kriging.py, gp.py)
| Method | Notes |
| --- | --- |
| `zero` | predict zero |
| `idw` | Gaussian-distance × 1/σ̄² weighted mean; ℓ chosen per density on validation days using *normal* stations as targets (held-out play no part) |
| `idw+noise` | idw + white + flicker noise, amplitudes (P(f)=A+B/f) fitted to neighbours' departures from their own idw predictions — "simplest generative model to beat" |
| `regress` | same network, MSE (conditional mean) |
| `krige` | per-day kriging, C(d) = s_cm + s_r·Matérn-3/2(d/ℓ) + nugget, fitted once on training years; common mode ≈ unknown mean |
| `krige+noise` | as idw+noise |
| `gp` | space–time GP: common-mode + regional (Matérn 3/2) + station part, each separable, temporal kernel white+flicker+Gauss–Markov; fitted to **median** lag curves; conditions jointly on whole window with flat prior on neighbour mean/trend (R&W eq. 2.42) — the strongest Gaussian baseline |

Finding on kriging fit: shared covariance is almost entirely network-wide (flat 25–300 km), small
regional part within ~20 km (docs/00-design.md:332-344) → best linear weights are near-uniform.

### Metrics (paper/report.tex:602-657; src/imonet/metrics.py, c2st.py, significance.py)
- Errors normalised by each station's training-years robust scale; per-station scores summarised by the
  **median over stations**; fast part (< ~2 months; slow part = 61-day running median) / slow / total.
- **Point**: nRMSE of the sample mean.
- **Probabilistic**: CRPS and energy score with **fair ensemble estimators** (Ferro 2014); coverage of
  50/90% intervals with **Weibull plotting positions** (numpy default gives ~85% for a 90% interval with
  32 members); spread/error ratio with (m+1)/m factor; calibration also **pooled** and **by quartile
  of the station's own noise level** ("calibration a median hides", docs/00-design.md:439-444).
- **Realism**: periodogram ratio by period band (2–4 … 64–128 d), robust-scale ratio, 99th-percentile
  tail ratio, correlation with neighbours vs distance bins.
- **Indistinguishability**: classifier two-sample test (Lopez-Paz & Oquab 2017): logistic regression on
  window summary features (log scales, tail ratio, lag-1/7 autocorrelation, band powers; conditional
  variant adds correlation/scale ratio with 4 nearest neighbours), station-split 2-fold, exact binomial
  test; a small CNN as second check; **IDW without noise as positive control**.
- **Significance**: paired differences on the same stations/networks; 95% bootstrap (10,000) over
  stations keeping a station's thinned networks together + exact sign test on station counts
  (src/imonet/significance.py:1-16). Caveat stated: stations treated as independent → intervals
  optimistic (paper/report.tex:657-660).
- Samples also scored after the cube's own outlier screen (like-for-like with the screened truth).

### Experiments
- **A — held-out stations** (California, 173 held-out, 107 scored): four densities (full, 25, 60 km =
  Türkiye-matched main result, 100 km), 5 thinning seeds each, 256-day tiling of the validation period
  (paper/report.tex:592-600).
- **B — planted fault-slip events**: triangular dislocations on SCEC CFM faults (cutde), slow-slip and
  afterslip templates, 60 events per template; each held-out target generated twice with the same seeds
  (planted vs original network); score ρ = ⟨d,s⟩/⟨s,s⟩ (fraction of planted signal transferred), shape,
  timing, spurious signal outside the footprint; main plot ρ vs footprint / nearest-station gap.
  Plus the real 2019 Ridgecrest earthquakes (excluded from training).
- **C — Türkiye**: frozen model, every station generated from all others; main 2013–2015, sparse
  2016+, in-fit control.

### Headline results (validation period)
- Point accuracy barely depends on density: IDW fast nRMSE 0.776 (zero) → 0.560 full, 0.565 at 60 km,
  0.586 at 100 km; regress beats IDW by only 0.5–1.4% (paper/report.tex:765-781). Shared fast signal is
  common mode coherent over hundreds of km.
- Adding noise to the average lowers fast CRPS by 27% (0.421 → 0.308) (paper/report.tex:783-786).
- **Diffusion is the best probabilistic generator at every density**: fast CRPS 0.297/0.299/0.302/0.309
  (full/25/60/100 km) vs GP 0.312/0.310/0.312/0.319; paired 3.3–4.9% better than GP, better at 99–105 of
  107 stations; energy score 0.624 vs 0.643 (IDW+noise)/0.645 (GP) at 60 km; second seed within 0.2%
  (paper/report.tex:792-806).
- Spatial structure: kriging+noise, GP and diffusion reproduce the decay of neighbour correlation with
  distance (|Δr| 0.007–0.017) vs IDW+noise 0.033–0.052 — **no learned generator needed for that**
  (paper/report.tex:808-818).
- Calibration: without the noise level every generator hedges — diffusion spread/error 1.38–1.42 for
  the quietest quartile to 0.92–0.93 for the noisiest (paper/report.tex:876-891). Given the level:
  1.04–1.06 → 1.15–1.18, 90% coverage 0.91–0.93 per quartile; the GP gains less because the level
  scales its width but not its tails/spectrum (paper/report.tex:913-927).
- Realism: diffusion spectra within ±0.11 log10 in every band; tail ratio +13% (falls to 0.06 log2 given
  the level); occasional spurious multi-month drifts of ~15 mm (paper/report.tex:944-962).
- **C2ST**: Gaussian generators detected 78–92%; diffusion 0.50–0.52 at full/25 km, 0.53–0.57 at 60/100 km
  (paper/report.tex:989-1008). CRPS differs by a few % but only Gaussian samples are recognisably
  synthetic.
- Planted transients cost nothing on ordinary days (paired −0.0…+0.2%, n.s.) (paper/report.tex:1018-1025).
- Experiment B: nothing transfers if footprint < ~2× gap to nearest station; slow slip (~11 km)
  invisible at every density. Full network median ρ: diffusion 0.41, GP 0.37, averages 0.12–0.23;
  over all networks diffusion > GP for footprint > 4× gap (0.62 vs 0.57) and 2–4× (0.22 vs 0.10). GP best
  for shape of slow regional signals (corr 0.97 vs 0.92) (paper/report.tex:1053-1087).
- **"The transfer comes from the training data"**: without planted transients diffusion ρ 0.23; regress
  gains as much from transients (0.44) → generative formulation adds calibration/probabilistic
  accuracy, not transfer (paper/report.tex:1089-1099).
- Ridgecrest (12 held-out stations > 5 mm): full network diffusion recovers 0.85, error 2.2 mm, 79% of
  offset components inside 90% range vs 17–21% for Gaussian generators; at 60 km diffusion 0.61–0.66,
  3.7 mm vs 11.1–12.7 mm for every method trained without transients; GP fails because its regional
  covariance (fitted on ordinary days) decays within 5–21 km (paper/report.tex:1111-1134).
- Türkiye (no retraining): main period fast CRPS 0.369 vs GP 0.376, IDW/kriging+noise 0.380; −1.8%
  vs GP, better at 135–138/150 stations; GP too narrow (S/E 0.84, cov90 0.88), diffusion slightly wide
  (1.12, 0.90). Sparse 2016+: diffusion ≈ GP (−1.1%, CI spans zero), both ~7–8% ahead of averages;
  diffusion tails heavier (+31%). In-fit scoring lowers everyone, ranking unchanged
  (paper/report.tex:1172-1216).

Bottom line (paper/report.tex:1222-1230): **the gain is in the shape of the distribution, not in point
accuracy**; a few % CRPS over a well-fitted GP, consistent across stations, densities, seeds, regions.

---

## Lessons and pitfalls (docs + commit history)

1. **Point scores are bounded by the shared signal.** Point accuracy is saturated by common mode; the
   real contribution of generative modelling shows only in proper scores, calibration and realism.
   Always include a strong *Gaussian generative* baseline (GP), not just deterministic ones.
2. **Calibration at the median station lies.** v1's "21–26% too wide" was diagnosed (not by screening,
   not by CFG, not earthquakes) as the model hedging over the unknowable station noise level; fixed by
   an optional noise-level input with dropout (docs/00-design.md:566-594; commit `f5bb099`). Report
   pooled and per-quartile calibration.
3. **Covariance fits to means describe a disturbed station.** GP fitted to mean lag curves gave samples
   25% too large and 2.6× too strong at 64–128 d; switching to medians fixed it (commit `9ff8e29`;
   docs/00-design.md:363-370). The switch was prompted by validation, so both are reported.
4. **Train/score drift from the trajectory fit.** Residuals outside the fit years carry extrapolation
   offsets/drifts; per-window mean+trend removal in both training and evaluation (commit `0f20391`).
5. **Fallback-fit leakage.** Stations with <1 year of training data were fitted (and scale-normalised)
   on days overlapping scored years → excluded from every score; applied first to A only, then found
   in an audit to be missing in B, Ridgecrest and C2ST (commits `b660296`, `0789684`). Removed 47 of 154
   stations; all medians moved ~6%, rankings unchanged.
6. **Blind evaluation must be enforced in code, not convention.** `regress-v2` had been evaluated with
   the held-out station's noise level passed through (no switch); fixed to blind-by-default with a
   separate "given" run (commit `0789684`).
7. **Held-out targets are fully isolated** (never neighbours), hyper-parameters (IDW ℓ) chosen on
   normal stations, Türkiye never used for tuning, test period untouched; Ridgecrest days excluded from
   training; Ridgecrest targets chosen by distance not by catalogue offsets, because the catalogue was
   biased exactly near the rupture (commit `c56dbe8`).
8. **Synthetic augmentation drives transfer of localised signals**; a visibility guard prevents
   teaching the model to hallucinate unexplained excursions; augmentation family deliberately different
   from the evaluation family.
9. **Numerical artefacts in simulators**: near-vertical CFM fault edges made cutde's half-space solution
   ill-conditioned (one event moved 978 stations > 1 mm instead of 35); fixed by snapping vertices
   (commit `e8ecdec`). Lesson: sanity-check the footprint of planted signals.
10. **Estimator details matter**: fair CRPS (finite ensemble), Weibull plotting positions for coverage.
11. **Reproducibility**: seeded sampler for paired experiments; `rescore` recomputes summaries from
    stored per-station rows; README warns that numbers quoted in prose don't auto-update after a
    rescore. Indeed design §4.3 still says p = 0.5 / 31% planted (docs/00-design.md:212-213), while the
    trained v3 uses p = 0.8 / 49% (docs/00-design.md:607; paper/report.tex:490, 503) — keep a single
    source of truth for configs.
12. Thinning method matters: random subsampling preserves source-network clustering and cannot mimic
    the target region; minimum-separation thinning can.
13. Paired bootstrap over stations ignores spatial correlation → optimistic intervals; sign-test counts
    more robust.

---

## Transfer to GRACE

### Carries over directly
- **Problem framing as conditional generation scored as samples**: ensembles of TWSA fields rather than
  a single reconstruction; directly addresses UQ, which most GRACE DL gap-filling papers treat
  superficially (single-model MC dropout or ensembles of deterministic nets).
- **Scoring rules**: fair CRPS per pixel/basin, **energy score** over multivariate targets (basin time
  series, or a patch), Weibull-position coverage, spread/error, pooled and stratified calibration
  (stratify by climate zone, signal amplitude, latitude, or month-of-year instead of station noise).
- **Realism checks**: temporal PSD by period band (sub-seasonal / seasonal / interannual) per pixel;
  **SH degree-variance spectra** and spatial variograms as the analogue of "correlation vs distance";
  tail ratios (droughts/floods, ice-sheet extremes); **stripe diagnostics** (N–S anisotropy of the
  spatial spectrum / order-dependent power).
- **C2ST** with interpretable summary features + CNN second check + positive control (e.g., the
  climatology or a linear interpolation).
- **Baselines-as-kriging**: the GRACE equivalents are climatology+trend, linear/spline interpolation,
  SSA/M-SSA gap filling, GP/kriging in time and space, and the statistical/ML reconstructions
  (GRACE-REC-style, CNN/LSTM gap fillers). The imonet lesson is to include the strongest *Gaussian
  generative* baseline (a space–time GP or a stochastic SSA) with noise, not just deterministic ones.
- **Paired significance** (bootstrap + sign test) — but use block bootstrap over basins/months, since
  GRACE pixels are far more correlated than stations.
- **Blind-evaluation discipline**: held-out regions/basins never seen in training; hyper-parameters
  chosen without held-out data; a "transfer" test region; an untouched test period (e.g. GRACE-FO
  years); audit that every evaluation path enforces the same exclusions.
- **Optional "noise level" conditioning with dropout** → per-month GRACE error level (formal/calibrated
  errors, GRACE accelerometer degradation 2016–17, GRACE-FO ACC-transplant months) or per-basin
  uncertainty; lets the model be calibrated across months of different quality.
- **Augmentation to mimic the target regime** (thinning California → Türkiye) → degrade high-quality
  months or model fields to GRACE-like quality: SH truncation, sampled stripe noise from the
  full L2 covariance, synthetic gaps of 1–11 months.
- **Planted signals + paired seeds** (experiment B) → plant synthetic hydrological events (drought,
  flood pulse, reservoir impoundment, glacier mass loss) of controlled footprint and measure ρ vs
  footprint / effective resolution (~300 km). imonet's "nothing transfers below ~2× the gap" maps onto
  "nothing below GRACE resolution is *observed*; anything finer comes from the prior/covariates".
- EDM preconditioning, Heun sampler, EMA, fixed-σ validation loss, and the **null-token
  cross-attention** (missing months handled like padded neighbours) are directly reusable.

### What must change (where the analogy breaks)
- **Geometry**: imonet conditions a 1D series on *scattered* point series via target-relative offsets.
  GRACE is a gridded field on the **sphere** (mascons ~3°/0.5° grids, or SH coefficients to d/o 60–96).
  Backbone should be 2D/3D (space×time) on an equal-area/HEALPix grid or local tangent-plane patches
  (imonet's azimuthal-equidistant offsets ≈ local-patch coordinates); poles and dateline matter;
  absolute location probably *should* enter (climate/hydrology is location-specific), unlike imonet's
  deliberate translation invariance.
- **Small data**: imonet trains on ~1078 stations × 12 years of daily data, giving millions of
  (station, window) samples. GRACE has **~250 monthly global fields** (2002-04 → present, ~20
  missing months + the 11-month 2017-07→2018-05 gap). Patches multiply samples but are strongly
  correlated within a month. A diffusion model with 11 M params will overfit unless pretrained on
  **hydrological/land-surface model output** (GLDAS, ERA5-Land, WGHM, LISFLOOD) or reanalysis, then
  fine-tuned — the analogue of planted transients, with the same lesson: transfer comes from the
  training data, and the sim-to-real gap must be tested.
- **The target is never directly observed at high resolution.** imonet's held-out stations give a
  real ground truth; GRACE has none beyond ~300 km. Downscaling "truth" is a model; validation must use
  independent sensors (wells, GNSS vertical loading, altimetry, basin water budgets P−ET−R).
- **Observation operator**: GRACE data = SH truncation + filtering (Gaussian/DDK) + leakage + GIA and
  low-degree (C20, degree-1) replacement of the true mass field. imonet had an identity operator with
  per-observation σ. GRACE needs explicit forward models; a linear-Gaussian operator makes
  **posterior sampling** (DPS/DDRM/ΠGDM-type guidance) natural.
- **Correlated, structured noise**: stripes are N–S correlated errors, not white; the likelihood needs
  the full (or approximated) error covariance (e.g. ITSG-Grace2018/ITSG-Grace_operational covariances).
- **Mass conservation / consistency**: samples should integrate to the observed regional/global mass;
  imposing that re-coarsened samples match GRACE (hard projection or guidance) has no imonet
  counterpart.
- **Trends are the science**: imonet discards per-window mean and trend; for GRACE, trends and
  accelerations (ice loss, groundwater depletion) and the seasonal cycle are the signal. Keep them, or
  model them explicitly with a separate deterministic component and generate the residual — but then
  the gap filling of the trend is a forecasting problem, not interpolation.
- **Heavy tails, spatially**: imonet's robust per-component scaling → per-pixel robust scaling;
  Greenland/Antarctica/large-river extremes dominate SDs exactly as El Mayor–Cucapah did.
- **"Neighbours" become temporal and multi-modal**: in the 11-month gap the nearest observed months are
  up to ~6 months away; conditioning is mainly on covariates (precipitation, temperature, soil moisture,
  SLR low-degree fields, Swarm). That is closer to **reconstruction/forecasting** than imonet's
  kriging; the risk of confident hallucination is larger and must be measured.
- **Significance**: effective sample size is small (few independent basins, few years); imonet's
  per-station bootstrap would be badly optimistic.

### Candidate GRACE task formulations in imonet's style
1. **Virtual months (gap filling with UQ).** Generate x_t = TWSA field (mascon grid, global or patch) at
   a missing month t, conditioned on GRACE months within ±k months (variable, masked, null-token
   cross-attention as with missing neighbours), monthly covariates (ERA5/ERA5-Land P, T2m, soil moisture,
   SLR/Swarm low-degree gravity), month-of-year, and an optional data-quality level. Train by hiding
   real months, including **synthetic 1–11-month blocks** inside the GRACE era (the analogue of
   held-out stations); validate on hidden blocks in 2002–2016; untouched test period (e.g. GRACE-FO);
   the real 2017–18 gap has no truth, so score it only against independent data (GNSS loading, SLR,
   Swarm, basin budgets). Critical: performance will decay with distance-in-time to the nearest
   observed month — report scores by that distance (analogue of the "distance to nearest station"
   table) and compare to a space–time GP and M-SSA.
2. **Virtual high-resolution pixels (downscaling).** Generate a 0.25–0.5° TWSA patch conditioned on the
   coarse GRACE field (mascon or SH-truncated) plus high-resolution static/dynamic covariates
   (topography, land cover, irrigation, precipitation). Train on hydrological model fields passed
   through the GRACE operator (degradation augmentation ≈ thinning); add a **consistency constraint**
   that the re-coarsened sample reproduces the GRACE field (mass conservation). Blind test on held-out
   basins/regions and against wells/GNSS. Critical: the fine-scale structure is whatever the training
   model contains — the C2ST would compare against *model* fields, not reality; label it as
   "model-informed downscaling" and quantify what is constrained by GRACE vs by the prior (imonet's
   ρ-vs-footprint analysis, with resolution in place of station gap).
3. **Posterior sampling under the GRACE filter operator (destriping + leakage + UQ).** Learn a
   diffusion prior p(x) over (high-res or mascon-resolution) TWSA fields; at inference sample
   p(x | y) with y = A x + n, A = SH analysis to d/o N (+ optional filter), n with the L2 error
   covariance — DPS/ΠGDM/DDRM-style guidance or a plug-and-play scheme. One prior serves destriping,
   leakage correction and gap-adjacent reconstruction; samples are consistent with the data by
   construction. Evaluate in a closed-loop simulation (model truth → operator → noise) plus real-data
   checks (residual stripe power, basin totals vs unfiltered, coastal leakage). Critical: posterior
   sampling approximations (DPS) are known to be miscalibrated — imonet's calibration toolkit is
   exactly what is needed to expose that; and an SH-domain operator on a sphere requires careful
   implementation.

Priority suggestion: (3) is the most novel and physically principled; (1) is the closest analogue to
imonet and has the clearest evaluation; (2) is the most popular in the literature but has the weakest
ground truth.

---

## Key references from imonet

Copied into `imonet.bib` (keys as in imonet's `paper/references.bib`):

| Key | Why relevant for GRACE + diffusion |
| --- | --- |
| `imonet` | this in-house work (@unpublished) |
| `ho2020ddpm` | DDPM, diffusion foundation |
| `karras2022edm` | EDM preconditioning/sampler used by imonet |
| `ho2022cfg` | classifier-free guidance (conditioning dropout) |
| `tashiro2021csdi` | conditional diffusion for probabilistic time-series imputation, CRPS evaluation |
| `liu2023pristi` | conditional diffusion for spatiotemporal imputation |
| `hu2024ustd` | diffusion for probabilistic spatiotemporal kriging/forecasting |
| `wu2021ignnk`, `xu2025kits` | inductive spatiotemporal kriging (deterministic) |
| `cini2022grin`, `marisca2022spin` | GNN spatiotemporal imputation (deterministic baselines) |
| `gneiting2007` | proper scoring rules, CRPS, energy score |
| `ferro2014` | fair ensemble scores |
| `hyndman1996` | sample quantile definitions (coverage) |
| `lopezpaz2017` | classifier two-sample tests |
| `rasmussen2006` | GP regression / kriging baseline |
| `gneiting2002` | space–time covariance functions |
| `li2025loading` | environmental (hydrological) loading in GNSS — bridge to GNSS-based validation of TWS |
| `blewitt2018` | NGL GNSS data (GNSS loading as independent TWS validation) |
| `williams2004`, `langbein2008` | coloured (flicker/Gauss–Markov) noise models, relevant to temporal noise baselines |

Not copied (GNSS-tectonics specific): wdowinski1997, dong2006, kreemer2021, bachelot2025, ngl2026,
nikkhoo2015, plesch*, tian2016, shen2015, sandwell2016, qiu2024, materna2024, vavra2024,
gonzalezortega2014, naturalearth, teutschmann2026gnssfm, costantino2023/2024, li2023interp, benoist2020.
