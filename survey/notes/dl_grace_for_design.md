# Deep-learning GRACE papers: what is useful for our design

Compiled 2026-11-18 from the 35 PDFs in `literature/dl_grace/`.

**How this was read.** Text was extracted with `pdftotext -layout`. Arzoumanidis et al. was read in full by the compiler. The other 34 papers were read by six parallel reader agents, each given the project context and the same recording template; their notes were merged here. About 25 load-bearing facts (numbers, dataset names, quoted sentences) were then re-checked by text search in the extractions. Figures were not inspected (captions only) and no supplementary material was available. Read status per paper is in Section 6.

**Conventions.** "Paper says" = stated in the text, with a location. "Our inference" = a suggestion of ours, not in the paper. "Not stated" = not in the text read.

**Main message in four lines.**
1. No paper in the folder validates TWS detail below GRACE resolution against an independent TWS observation. Several say so in plain words (Gou & Soja 2024; Goracci 2026; Xiong 2026; Vishwakarma 2021; Li 2026; He 2021).
2. Every downscaling paper takes its fine pattern from a hydrological model or from fine-grid covariates, and ties the coarse part to GRACE with a soft penalty. None uses an exact projection, none is generative, none reports CRPS.
3. The proxies they use at fine scale are always the same three: small-basin water-budget closure, groundwater wells (correlation), and re-filtered agreement with GRACE.
4. Nobody tests a downscaling method at a scale where GRACE itself can check it. Our 8→16 and 16→32 tests are new, and they can be used to test other people's methods and validation proxies too.

---

## 1. Arzoumanidis et al. (2026) in detail

**Citation.** Arzoumanidis, L., Johannsen, L., Middendorf, K., Eicker, A., Dehbi, Y. (2026). Reconstructing GRACE Terrestrial Water Storage with Spatio-Temporal Graph Neural Networks: An Application to South America. arXiv:2606.23833v1, 22 June 2026 (conference manuscript, ACM template placeholders still present; not peer reviewed as far as the text shows). Code: github.com/hcu-cml/MTGNN-TWS-Reconstruction-GRACE. **Read: full main text (Sects. 1-8); references skimmed.**

**Correction to our expectation.** This paper does not use a hydrological model. It reconstructs GRACE-like TWS from ERA5 atmospheric-reanalysis fluxes. It is a reconstruction (hindcast to 1940), not a downscaling: the output has GRACE's own resolution.

### Task
Learn a mapping from meteorological forcing to monthly GRACE TWS anomaly over South America, then apply it to 1940-2002 where GRACE does not exist (Sect. 2.3). 1° grid, 1,120 land cells.

### Inputs (Sects. 2.3, 4.2)
- Three ERA5 single-level variables: total precipitation P, evapotranspiration E (from surface latent heat flux), total runoff R.
- Hourly fields aggregated to daily; a window of 30 daily values per month.
- Re-gridded to 1°, and "expressed in EWH via a spherical harmonic expansion to degree 96 ... ensuring full spatial consistency with the GRACE prediction target" (Sect. 4.2). Whether the DDK3 filter is also applied to the inputs is not stated.
- Feature-wise z-score normalisation, fitted on the training months only (Sect. 5.5).
- No GRACE history, no static maps, no hydrological model.

### Target (Sect. 4.1)
- ITSG-Grace2018 and ITSG-Grace_operational (the same series we use), to degree 96, degree-1 restored, C20 from SLR, GIA removed, **DDK3** filtered, anomalies relative to 2004-2009, coastal leakage mask.
- Effective resolution "roughly 300 km" (Sect. 2.1).
- April 2002 to December 2023. Gaps including 2017/18 are interpolated, but the interpolated inter-mission gap is excluded from training (Sect. 5.5).
- So the "truth" is filtered GRACE at GRACE resolution. Nothing below GRACE resolution is used or claimed.

### Graph construction (Sect. 5.4)
- Nodes: the 1,120 one-degree land cells.
- Edges: a **static, prescribed** weighted undirected graph. The adjacency weight is a convex mix of two similarities (Eq. 5): A_ij = α·s_climate + (1−α)·s_distance, with α = 0.6.
  - s_distance = exp(−d_ij/σ), geodesic distance, σ = 3500 km (Eq. 4).
  - s_climate = weighted sum over variables of (|ρ_ij| + 1)/2, where ρ_ij is the Pearson correlation of the two nodes' time series at the lag in [0, τ_max] that gives the largest absolute correlation (Eq. 3). Anti-correlation counts as much as correlation.
- Not stated: τ_max, the per-variable weights w_f, which time series and period the correlations are computed on, and whether the adjacency is thresholded or dense.
- Why not learned: the original MTGNN learns its graph; the authors replaced it because learned graphs are hard to interpret and because "jointly optimizing a latent graph and a deep predictive model proved prone to convergence instability" with random subgraph sampling cutting physically meaningful links (Sect. 5.4).

### Temporal component (Sects. 5.2, 5.3)
- MTGNN (Wu et al., a traffic-forecasting network), kept in three parts: graph layer, graph convolution, temporal convolution.
- Graph convolution: two mix-hop propagation layers with a retention factor against over-smoothing.
- Temporal convolution: dilated inception layers with kernel sizes 1×2, 1×3, 1×6, 1×7 and tanh/sigmoid gating; receptive field grows exponentially. No recurrent layer, no attention.
- Time is the 30-day axis inside one month. There is **no memory across months**: each month's TWS is predicted from that month's 30 daily fluxes only (Sect. 2.3, 5.5). Our inference: storage is an integral of fluxes, so a one-month window can only work because the network learns the seasonal state from the fluxes' own seasonality; this probably explains the damped extremes and the weak arid regions.
- Size: graph-convolution depth 4, 32 convolution channels, 64 residual/skip channels, 128 end channels (Sect. 5.5).

### Loss and training (Sect. 5.5)
- MSE. Adam, learning rate 1e-3, batch 64, dropout 0.4, weight decay 0.005, 40 epochs, hyperparameters tuned by hand on validation error.
- Sequential split: training April 2002 to June 2017 (183 months), then 34 validation and 33 test months.
- A water-balance loss is **proposed, not implemented** (Sect. 7.2, Eq. 7): penalise (TWSA(t_i) − TWSA(t_{i−1})) − (P − E − R) between consecutive months. "Ongoing work ... integrates the water balance constraint into training" (Sect. 8).

### Uncertainty
None. Deterministic output.

### Validation (Sect. 6)
- Only against GRACE and against three other reconstructions. No in-situ data, no independent data.
- Test months, normalised units: MSE 0.0128, RMSE 0.1131, MAE 0.0862; validation 0.0117 / 0.1083 / 0.0830. Spatial correlation of time-averaged test fields 0.61 (Sect. 5.5).
- Table 3, captioned 2002-2023, all cells and months pooled: RMSE 0.132 m, MAE 0.096 m, bias −0.004 m, correlation 0.693. This period includes the training months. Note: the same 0.693 appears as the *validation* correlation in Table 5, and Sect. 7.1 quotes a basin-level 0.945 against 0.939 in Table 4. The paper does not explain which months each number covers.
- Error structure: correlation highest in the central Amazon, below 0.5 in the arid south (Bolivia, Paraguay, Argentina, Chile) (Fig. 5 caption, Sect. 6.2). Dry-season errors are spatially fragmented, wet-season errors large but coherent (Sect. 6.2).
- De-trended, de-seasonalised residuals at one Amazon cell agree in timing; extremes are smoothed (Sect. 6.3).
- 2015/16 El Niño and 2020/21 La Niña maps: pattern right, "underestimating their amplitude" (Sect. 6.3).
- Comparison on a common 1° grid, 2002-2020, basin-mean series (Table 4):

| Model | Inputs | Corr. | Std. dev. | RMSE | Corr. (one Amazon cell) |
|---|---|---|---|---|---|
| GRACE | – | 1.000 | 0.083 | – | 1.000 |
| MTGNN | 3 | 0.939 | 0.088 | 0.030 | 0.866 |
| GRAiCE | 6 | 0.964 | 0.083 | 0.026 | 0.907 |
| GTWS-MLrec | 16 | 0.940 | 0.074 | 0.031 | 0.932 |
| RM-REC | 25 | 0.931 | 0.078 | 0.031 | 0.902 |

- Pilot with 2 m temperature added: validation correlation 0.693 → 0.713 (Table 5).

### Limitations
Stated by the authors: weak in arid, human-influenced regions; damped extremes and trend magnitude; ERA5 quality before 1979; the assumption that the forcing-to-storage mapping is stationary back to 1940 (Sect. 2.3); only 183 training months, which forced strong regularisation (Sect. 5.2).

Our reading: one region; the comparators' test periods overlap their own training data; headline metrics pool training and test months; the graph's contribution is never ablated (no run without the climate-similarity edges, no CNN baseline on the same inputs), so the paper does not show that the graph is what makes it work.

### What we could take from it
1. **Band-limit every auxiliary channel exactly like the conditioning field** (paper: inputs expanded to degree 96 to match the target). For us: precipitation, temperature or model fields given as channels should be limited to the same degree and taper as the GRACE input, or to the target degree, on purpose and consistently between training and use. Testable at 8→16 and 16→32.
2. **A water-balance check between consecutive months** (paper: proposed loss, Eq. 7). For us it is more useful as a diagnostic than as a loss: with shared starting noise our members are coherent in time, so the month-to-month change of the generated detail can be compared with band-passed P − E − R. At coarse scales we can first check whether GRACE's own detail passes this test; if it does not, the test is not usable at fine scales either.
3. **Scoring habits**: score de-trended, de-seasonalised residuals separately; look at named extreme events; map skill by climate zone and expect arid regions to be worst.
4. **Prescribed rather than learned structure** when data are few. Our U-Net already has this property.
5. **Low priority: the graph itself.** Long-range edges serve reconstruction from forcing (teleconnections). Our task is adding local detail to a field whose large scales are given. The only ablation of graph message passing in this folder (Steidl 2026, Table B1) found it worth 0.05-0.12 cm. We see no case for a graph network in our design; if long-range context matters, a few attention layers at the coarsest U-Net level are the cheaper test.

---

## 2. Table of all papers

Abbreviations: P precipitation, T temperature, ET evapotranspiration, R runoff, SM soil moisture, WGHM WaterGAP, LSM land-surface model, n.s. not stated, wells = groundwater wells. "Target" means what the method treats as truth. Locations in brackets.

### 2a. Spatial downscaling

| Paper | Task, region, resolution | Inputs | Target / fine-scale "truth" | Architecture; time | Uncertainty | Validation | Amplitude, consistency with GRACE |
|---|---|---|---|---|---|---|---|
| **Gou & Soja 2024**, Nature Water | Global land, 0.5°, 2002-2019 | JPL mascon (no gain factors), WGHM 2.2d TWS, GLDAS Noah P, ET, 3 runoff terms, lat, lon; 32×32 patches (16°) | No fine truth. Loss = abs. error of 16° patch mean vs GRACE + (1 − corr with WGHM) × MAE vs WGHM (Eqs. 2-5). Detail "inherited from the WGHM simulations" | Conv encoder-decoder; months independent ("did not explicitly feed any temporal information") | 5 networks × 20 perturbed GRACE inputs; median 7.3 mm; "may be underestimated"; no calibration | "We cannot directly evaluate". Budget closure dTWS/dt vs ERA5-Land P−ET−R in 1,342 HydroBASINS L4 basins: NSE > 0 on 83% of area (GRACE 77%, WGHM 75%); gain largest below 63,000 km². Altimetry in supplement (not read). No wells/GNSS | WGHM underestimates annual amplitude by >10% in more than half of 147 basins; phase shifts 2-3 months. GRACE constraint is soft, on patch mean only |
| **Gou et al. 2025**, J Geodesy | Ocean bottom pressure, global ocean, 0.25° | CSR mascon, two ocean reanalyses (GLORYS, ORAS5), steric height, bathymetry, wind stress | Same loss family; the two reanalyses weighted per patch by their coarse agreement with GRACE (Eqs. 2-4) | Same encoder-decoder; no time | None | 119 bottom-pressure recorders, 465 PSMSL tide gauges. Recorders "inconclusive" (corr 0.49 vs 0.48 for CSR); tide gauges corr 0.44 → 0.67 | Output amplitude smaller than reanalyses, "inevitable" trade-off (Sect. 4.1) |
| **Goracci et al. 2026**, EGUsphere preprint | 3° (or 2°) → 1°; five 48° regions incl. Europe (24-72°N, 3°W-45°E); simulated NGGM/MAGIC and real COST-G GRACE | 13 ERA5-Land variables with 1-3 month lags, elevation, water fraction, sin/cos day of year. Coarse TWSA is **not** an input channel | No fine truth. MSE between area-averaged output and the satellite product (Eq. 1). Nothing constrains the 1° field | Small U-Net; lags as channels | 10 seeds; authors: "should not be interpreted as the physical uncertainty" | Simulated: vs ESA ESM truth, composite of pixel and structure metrics incl. Fractions Skill Score. Real GRACE: coarse consistency only; budget check on Congo, Danube uses ERA5-Land that is also an input | "Fine-scale redistribution within each coarse satellite cell is not uniquely determined" (Sect. 6). Soft constraint |
| **Uz et al. 2026**, J Hydrology | 3° → 0.5°, global land, 2002-2022 | WGHM 2.2e TWS, ERA5 P, T, cumulative P−E−R, ERA5 TWS. GRACE not an input | JPL mascon 6×6 block means (soft) + pixel MSE and Sobel-gradient MSE vs **WGHM**; weight between them = current correlation with WGHM (Eqs. 3-11) | Variational U-Net; no temporal module | Latent sampling + MC dropout (30); compared only with mascon formal error | Spectra (follows mascon >600 km, WGHM <300 km); SMAP L4 SM; G3P glacier trends; US wells (Jasechko 2023/2024): median corr 0.33 vs 0.39 for the mascon, 0.00 for WGHM; NSE vs GLWS2.0 and GRACE-SeDA | Annual amplitude mascon/WGHM/output: Zambezi 140/86/124 mm, Nile 53/32/47 (Table 1). Authors: "potential underestimation of the trend"; hard constraint is future work (Sect. 5) |
| **Li et al. 2026**, ESSD preprint | 3° → 0.1°, China | ERA5-Land P, ET, R and their cumulative anomalies, T, NDVI; static irrigation, groundwater-irrigation share, population, elevation. GRACE not an input | 3° patch mean vs JPL (soft) + pixel MAE vs PCR-GLOBWB 2 + altimetry glacier mass change (Eqs. 6-8) | U-Net; no temporal module (ConvLSTM named as future work) | Std of 3 seeds | Budget closure in 163 L5 basins (ERA5-Land also an input): corr 0.80 vs 0.79 for the target itself. 559 North China Plain wells: 51% improve, 49% worsen vs GRACE. Independent glacier altimetry: 0.90/0.97 vs 0.12/0.05 for the model | Model overestimates amplitude in three humid basins, misses a trend in another (Sect. 4.1) |
| **Xiong et al. 2026**, ESSD (not ML) | Global, 0.5°, 2002-2022 | CSR SH to degree 90, DDK3; WGHM 2.2e compartments; glacier outlines; lake data | No training. Fixed spatial patterns (ICA of each WGHM compartment, glacier and lake masks) are filtered like GRACE; their monthly amplitudes are fitted to GRACE by least squares; unfiltered patterns are then recombined (Eqs. 1-2) | Least squares, month by month | Formal, patterns held fixed; "not a complete budget" | Re-filtered agreement with GRACE in 89 basins; budget closure in 1,200 L4 basins (NSE > 0: 71.7% vs 59.2% for GRACE; gain only below 63,000 km²); **28,248 GGMN wells**: corr increases at 67.7%; compared with GLWS2.0 and Gou & Soja | WGHM trends "substantially smaller" than GRACE; but WGHM has more high-degree power than DDK3 GRACE (Sect. 4.4.1). Consistency in least-squares sense only |
| **Vishwakarma et al. 2021**, Sci. Data (not DL) | ~3° → 0.5°, 160 catchments | GRACE catchment means; P, ET, R products with 6 lags | WGHM 0.5° residual after removing trend and GRACE's mean annual cycle | Partial least squares; lags | None | "Validating ... is not possible". Only catchment mass conservation | WGHM "informed the spatial redistribution and not the signal amplitude". Tigris named as a catchment where WGHM does not match GRACE |
| **Suryawanshi et al. 2026**, EGUsphere preprint | Evaluation of three 0.5° products over India | Their own PLSR variant per 3° mascon | WGHM residuals | PLSR | None | CGWB wells; storage change between two months of the same year; Pascal's gain metric. Positive gain in corr over GRACE: mascon-wise PLSR 13 mascons, catchment-wise PLSR 3, Gou & Soja 10 | States that fine-coarse relations are not the same at all resolutions (Sect. 1) |
| **Pascal et al. 2022**, HESS | Evaluation method; one 113,000 km² region → 0.5°, Telangana | JPL mascon; TRMM, NDVI, CCI SM; models avoided | GRACE at coarse scale; regression applied to fine predictors | Linear regression, random forest | None | 1,006 state wells. **Gain against the null of no downscaling**, temporal and spatial (Eqs. 2-7). Spatial gain in corr +27-29%, in RMSE +2% | Mascon × scaling-factor product "should not be used at the 0.5° resolution" (Sect. 5.1) |
| **He et al. 2021**, Can. J. Remote Sensing | 110 km → 10 km, Canada | GRACE only at inference | **Model-only pairs**: EALCO LSM degraded to 110 km in, EALCO 10 km out; then applied to GRACE | SRCNN, VDSR, RCAN; no time | None | Only averaging back to 110 km: RMSE 14-22 mm | "Domain transferring ... may introduce uncertainties". No projection |
| **Zhang et al. 2026**, ESSD | Groundwater anomaly, global, 0.25° → 1 km | CSR mascon minus GLDAS; 19 fine predictors (climate, NDVI, soil, elevation, month) | GRACE-derived value at 0.25°; model applied at 1 km assuming relations are "invariant across diverse spatial scales" (Sect. 2.4) | XGBoost, MLP per aquifer class; no spatial context | None per pixel | 152,069 wells (Jasechko 2024) for trends; 140 monthly stations (GROW, China yearbook). **North Caucasus Basin: 3,971 wells, CC = −0.01; Arabian Aquifer: 1,377 wells, CC = 0.07** | Trend slope vs wells 0.51; std ratio 0.59-0.91; High Plains trend has the wrong sign vs published |
| **Mousavimehr & Kavianpour 2025**, Appl. Water Sci. | 1° → 0.25°, Tehran province | CSR mascon; GLDAS, ERA5 | Detrended GRACE at coarse scale | 8 regressors, 1D CNN | None | ~350 Tehran Regional Water Company wells, time-series plots only | Detrended test corr only 0.5-0.6; headline 0.93 is mostly the linear trend |
| **Pakrashi et al. 2026**, arXiv | 11-year trends, High Plains, 10 km | CSR mascon, 9,869 kriged wells, irrigation, rainfall | No training; Bayesian latent field | Hierarchical model, MCMC | Posterior std; not checked | Qualitative map comparison | GRACE block = mean of latent cells + noise (likelihood, Eq. 14) |
| **Dombos et al. 2026**, arXiv | Sharpen daily ITSG fields (degree 40, 300 km Gaussian) to 0.25° mascon-like | Low-res GRACE, GLDAS CLSM, ERA5 | CSR monthly mascon (another GRACE product) | ESRGAN-type GAN; no time | None | vs CSR only; held-out 2008: explained variance 0.6 | Daily output has unverified day-to-day noise |

### 2b. Reconstruction from climate drivers or models (output at GRACE resolution)

| Paper | Task, region | Inputs | Target | Architecture; time | Uncertainty | Validation | Amplitude |
|---|---|---|---|---|---|---|---|
| **Arzoumanidis et al. 2026** | see Section 1 | ERA5 P, E, R daily | ITSG DDK3 | MTGNN | None | GRACE, other reconstructions | Damps extremes |
| **Humphrey & Gudmundsson 2019**, ESSD (not NN) | Global, 1901-2019 | Daily P and T | De-trended, de-seasonalised JPL / GSFC mascons | One linear store per mascon | 100 members: MCMC parameters + spatially and temporally autoregressive residual noise (Eqs. 8-13). **Rank histogram on 90 large basins** (Fig. 4f), in-sample | Sea-level budget; basin water balance (BSWB, 341 basins); GSIM streamflow, 12,496 stations in basins < 10,000 km² | Fits local amplitude better than LSMs |
| **Yin et al. 2023**, ESSD | Global, 0.25°, 1940-2022 | 16 ERA5 and land variables, 0-3 month lags | Three mascon products | Per-pixel random forest | None | Random month split; sea-level budget; BSWB; 10,168 gauges | – |
| **Mandal et al. 2025**, ESSD | Global, 0.5°, 1960-2022 | GLDAS CLSM and Noah TWS, P, T (lags 0-2), three climate indices | JPL mascon | Bayesian-network feature selection, then per-cell CNN / SVR / trees | Residual std per cell | GRACE; GRDC outlets via water balance | CLSM TWS selected in 71% of cells, ahead of P and T (Sect. 4.1) |
| **Palazzoli et al. 2025**, Sci. Data | Global, 0.5°, 1984-2021 | 5 ERA5(-Land) variables, 3-24 month history | JPL mascon | Per-cell LSTM / BiLSTM | None | Shuffled chunks; BSWB; lowest skill incl. "the Middle East", southwestern Russia; bias −13.6 cm at Absheron | – |
| **Wang et al. 2024**, ISPRS Annals | Global, 1°, WGHM → GRACE-like | WGHM TWS; 12 months of CRU P, T | JPL mascon | Conditional GAN | None | GRACE only | GAN term gives modest gain over the same generator without it |
| **Sun et al. 2019**, arXiv | India, GRACE resolution | Noah TWS at t, t−1, t−2; optional P, T | **Noah − GRACE mismatch** | VGG16, U-Net, SegNet; lags as channels | 1.96 × RMSE | GRACE; 3,989 Indian wells (Bhanja 2016), median corr ~0.4 | Noah "underestimates most of the wet and dry events"; P and T give "little improvement" |
| **Uz et al. 2024**, Sci. Data | Global, 1°, 1994-2021 | SLR degree-10 field, ERA5 P, T, SST, storage terms, day of year; t and t−1 | CSR mascon | Residual conv autoencoder | None | Basin to global scale only | – |
| **Sarkar et al. 2026**, manuscript | Groundwater anomaly, global, 0.5° | 5 dynamic, 7 static predictors; lat/lon removed on purpose | JPL mascon minus GLDAS | Per-cell LSTM, output = trend + season + residual; basin-trend penalty | Monte Carlo budget; not checked | **GROW wells** (Bäthge 2026, 204,292 sites) | MSE compressed variance to 0.77 (median); per-cell variance matching ×1.30; vs model-assimilating SeDA a factor 1.58 at corr 0.996 |

### 2c. Gap filling, forecasting, processing

| Paper | Task | Inputs | Target | Architecture; time | Uncertainty | Validation | Amplitude |
|---|---|---|---|---|---|---|---|
| **Mo et al. 2022**, J Hydrology (+ 2021 preprint, same method) | Fill 2017-18 gap, global, 1° | ERA5-Land P, T, cumulative storage change, TWS at t−2..t | Detrended JPL mascon | CNN with attention blocks; lags as channels ("longer lags did not clearly improve") | 20 networks trained jointly (SVGD); 95% band said to "almost completely" contain GRACE; no coverage number | Held-out GRACE; Swarm; Tigris-Euphrates basin NSE 0.90 | Basin series standardised "to exclude ... amplitude and magnitude differences" |
| **Mo et al. 2025**, GRL | Predict 1-3 months ahead, global, 1° | 12 past GRACE months + ERA5-Land P, T, TWS incl. target months | JPL mascon | CNN; all months stacked; **each window normalised by its own mean and std, output rescaled with the GRACE window's statistics** | SVGD; no coverage number | GRACE-FO only | Normalisation helps: NSE 0.89 vs 0.87 at 1 month. Reanalysis TWS "substantially enhances" accuracy |
| **Steidl et al. 2026**, EGUsphere preprint | Forecast 1-6 months, global, 1° | 12 months of TWSA and 8 ERA5 variables | A smooth GRACE-like reconstruction (Li et al. 2021), not GRACE | Basin-level graph encoder + LSTM; residual on last month | None | Reconstruction test; real GRACE; Euphrates among 16 basins (appendix table) | RMSE 1.83 cm on the reconstruction → 5.59 cm on real GRACE: "domain gap" (Sect. 4.3). ConvLSTM baseline worse than climatology beyond lead 1. Removing message passing costs 0.05-0.12 cm |
| **Lu et al. 2024**, Sci. Rep. | Forecast region-mean series, China | Past 5 months of GRACE | Next month | MLP, LSTM, BiLSTM + attention | None | Same GRACE product | MLP as good as LSTM; forecasts lag peaks by a month |
| **Gao et al. 2025** (CIDR), arXiv | Temporal gap filling of the Gou & Soja 0.5° product | The cell's own series | Withheld months | Improved SSA (not DL) | None | Artificial gaps | 11-month gap: relative RMS ~0.48 |
| **Chen et al. 2025**, JSTARS | Emulate GRACE post-processing (Level-2 → Level-3 grids) | Unfiltered CSR degree-60 grids | CSR Level-3 grids | Shallow CNN; "Bayesian" = hyperparameter search; "attention" = land mask | None | vs CSR L3; Euphrates-Tigris annual amplitude 5.96 vs 6.93 cm target | MSE network damps amplitude 10-30% in mid-sized basins |
| **Heller-Kaikov et al. 2023**, GeoAI workshop | Separate atmosphere+ocean from hydrology, synthetic | Sum field, land mask, latitude | Simulated components | pix2pix | None | Synthetic only | – |

### 2d. Reviews

| Paper | Content relevant to us |
|---|---|
| **Springer et al. 2026**, HESS (review of GRACE assimilation) | Assimilation is described as downscaling in which "the finer-scale patterns are informed by the model" (Sect. 5.2.1). Recommends computing the GRACE-model difference at observation resolution (Sect. 7). Rescaling GRACE to the model's variance "sacrifice[s] information contained in the measurements" (Sect. 2.5). Models have "underestimated amplitudes" (Sect. 4.2). GRACE error structure "is not well understood" (Sect. 5.4). ML downscaling "fail[s] to provide comprehensive uncertainty estimates" (Sect. 2.4). Validation data listed in Table 5. Benchmarks: GLWS2.0, GLWS release 3, CLSM-DA |
| **Gou et al. 2026**, Surv. Geophys. (review of uncertainty in DL) | Taxonomy (Bayesian NN, MC dropout, deep ensembles, input-perturbed ensembles "DEMC"). Worked TWS example at basin scale with a coverage table at 1/2/3 sigma (Table 2). Says the Gou & Soja 2024 uncertainties "might be overly optimistic" (Sect. 3.4.4). Diffusion models get one sentence. Lack of ground truth blocks calibration (Sect. 1) |
| **Sun & Scanlon 2019**, ERL (general ML review) | Marginal. One GRACE item (the Sun 2019 mismatch CNN) |

---

## 3. Ranked ideas to test

Ranking is by expected value for the current coarse-scale experiments, weighed against cost. "Coarse test" = can be scored at 8→16 and 16→32 against GRACE's own detail.

### 1. Use the models' pattern but not their amplitude (addresses e, a, f)
**Sources.** Xiong 2026 (patterns filtered like GRACE, amplitudes fitted to GRACE); Vishwakarma 2021 (model gives redistribution, not amplitude); Gou & Soja 2024 (scale-free correlation term); Mo 2025 (per-window normalisation with GRACE statistics); Mo 2022 (standardise before comparing).

**What we change.**
- (a) *New baseline, no training.* For each month: take model fields (each of the ten models, or their leading patterns), limit them to the input degree, fit their amplitudes to the GRACE input by least squares, then use the same amplitudes on the models' detail band. This is the Xiong method in our spectral setting.
- (b) *Input channels.* Give the network the models' detail for the month as channels, but normalised: divided by the RMS of the model's own input-band field (the same normalisation the GRACE input already gets), optionally with the model's climatology removed. The network then sees pattern and relative strength, not millimetres.
- (c) Our existing "mix" result (2.2 and 1.0 mm gain) is the reference to beat.

**Coarse test:** yes, directly, for both. Idea (a) also tests a published method at a scale where it can be checked, which its authors could not do.

### 2. Add structured, input-driven spread (addresses c)
**Sources.** Gou 2026 review (input-perturbed ensembles, Sect. 3.5); Gou & Soja 2024 (20 perturbed GRACE inputs per network); Humphrey 2019 (autoregressive, spatially correlated residual noise fitted to the residuals, Eqs. 8-13); Pakrashi 2026 and Springer 2026 (GRACE as a noisy constraint).

**What we change.**
- (a) At sampling, perturb the GRACE input for each member with a draw from GRACE's coefficient errors, and project each member onto its own perturbed input. The spread then contains GRACE's observation error, which a fixed input cannot give. Requires the ITSG covariance or at least formal errors per coefficient.
- (b) Replace the single spread factor by a residual model fitted at the coarse steps: variance, spatial correlation length and lag-1 autocorrelation of (GRACE detail − ensemble mean). Compare the lag-1 value with the month-to-month member correlation our shared-noise sampling produces.
- (c) Report a coverage table at 1/2/3 sigma (Gou 2026, Table 2) and a rank histogram of basin or regional averages (Humphrey 2019, Fig. 4f). The basin-average rank histogram tests the ensemble's spatial covariance, which pixel-wise CRPS does not.

**Coarse test:** yes for all three.

### 3. Time as stacked channels first; temporal architectures later (addresses d)
**Sources.** Mo 2025 (12 past GRACE months + model TWS for the month, all stacked; both help); Mo 2022 (3 months, "longer lags did not clearly improve"); Sun 2019 and Uz 2024 (two lags); Li 2026, Uz 2026, Mo 2022 (cumulative P − ET − R as a memory channel); Steidl 2026 (ConvLSTM worse than climatology beyond lead 1 with < 400 sequences; graph worth 0.05-0.12 cm); Lu 2024 (MLP equals LSTM).

**What we change.** Input channels: previous 1, 2, 3 and then 12 months of the conditioning field (model months in training, GRACE months in use), plus month-of-year sin/cos. Keep the U-Net. Only if a clear gain appears, try attention over time or ConvLSTM. Score gains with and without the climatology channel, because most temporal predictability at cell level is seasonal (Gao 2025: an 11-month gap is filled with relative RMS ~0.5).

**Coarse test:** yes. Caution from Lu 2024: networks fed past values tend to lag peaks by a month; check phase.

### 4. Fine-scale validation kit for the Türkiye box (addresses b)
No dataset below is stated in any paper to cover Türkiye; coverage must be checked. Tests are all relative to the degree-32 GRACE input (Pascal's null).

| Test | Source papers | Data named | Notes |
|---|---|---|---|
| Small-basin budget closure: NSE of dTWS/dt vs P − ET − R, binned by basin area | Gou & Soja 2024; Xiong 2026; Li 2026 | HydroBASINS level 4 or 5; ERA5-Land P, ET, R | Gain should appear only below ~63,000 km² (both global papers find this). Must not use budget terms that are also network inputs (Goracci and Li did; they call it internal consistency) |
| Share of wells whose correlation improves vs coarse GRACE | Xiong 2026; Li 2026 | GGMN (IGRAC), 28,248 wells; GROW (Bäthge et al. 2026), 204,292 sites; Jasechko et al. 2024, 152,069 wells | Correlation only, no specific yield. Expect modest numbers (51-68% improved in the two papers) |
| Storage change between two months of the same year | Suryawanshi 2026 | (their data: CGWB India) | Suits sparse or seasonal well records |
| Spatial gain per month vs the null | Pascal 2022 | – | Separates within-footprint skill from the seasonal signal |
| Streamflow in small basins, annual correlation | Humphrey 2019; Yin 2023 | GSIM; GRDC | Units differ, correlation only |
| Basin water balance from atmospheric fluxes | Humphrey 2019; Yin 2023; Palazzoli 2025 | BSWB (Hirschi & Seneviratne 2017), 341 basins | Half of the weak basins are weak for GRACE too (Palazzoli) |
| GNSS vertical loading | Springer 2026 (cites Springer 2019, Klos 2021 for Europe) | not specified | Weak beyond the seasonal cycle, depends on station density (Sect. 6.6) |
| Observed components | Li 2026; Xiong 2026; Mo 2021 preprint | Glacier elevation change (Hugonnet et al. 2021); RGI region "Caucasus and Middle East"; DAHITI reservoir levels | The only clear independent fine-scale success in the folder is glaciers in Li 2026 |
| Other products as comparisons (not validation) | several | GLWS2.0; Gou & Soja 2024; SYSU TWSA v1.0; Uz 2026; GRACE-SeDA; Vishwakarma 2021 catchment product (includes the Tigris) | All take their detail from WaterGAP or another model |

Regional numbers found: Tigris-Euphrates basin (Mo 2022, Table 2; Chen 2025, Fig. 8); Euphrates basin (Steidl 2026 appendix); North Caucasus Basin and Arabian Aquifer wells (Zhang 2026, Sect. 3.2.1); Tehran wells (Mousavimehr 2025); Danube (Mandal 2025; Goracci 2026). Poor ML skill is reported for the Middle East, southwestern Russia and Absheron (Palazzoli 2025) and for arid regions in almost every paper.

**Our inference, and the part we recommend doing first:** validate the validators at coarse scale. At 16→32 we know the truth (GRACE's detail). Run the budget-closure test and the well test there on our methods and baselines and see whether they rank methods the same way GRACE's own detail does. If a proxy cannot tell a good method from a bad one where the answer is known, it should not be trusted at degree 128.

**Coarse test:** the metrics yes; the proxies can be calibrated there.

### 5. Weight the ten models by how well each matches GRACE at coarse scale (addresses a, e)
**Source.** Gou et al. 2025 (two reanalyses weighted per patch by their patch-mean error against GRACE, Eqs. 2-4; two sources "clearly improve" on one).

**What we change.** In the baseline of idea 1 and in the model-field channels, weight each model per region and month by its agreement with GRACE in the input band. Or give the network one channel per model plus each model's coarse misfit to GRACE.

**Coarse test:** yes.

### 6. Covariates: static human-use maps and observed components before meteorology (addresses f)
**Sources.** Li 2026 and Sarkar 2026 (irrigation intensity, groundwater-irrigation share, population, elevation, permeability); Goracci 2026 (elevation, water fraction); Mandal 2025 and Sun 2019 (once a model's TWS is an input, P and T add little); Palazzoli 2025 (meteorology alone fails in the Middle East); Li 2026 and Xiong 2026 (glacier and lake observations); Mandal 2025 (climate indices).

**What we change.** Order of channel experiments: (1) land mask, elevation, water fraction; (2) irrigation and groundwater-irrigation maps; (3) model fields (idea 1); (4) P and T last, and as accumulated anomalies rather than single months.

**Warning from Zhang 2026 (Sect. 3.1.3):** static variables took 62-64% of feature importance when the target was GRACE. A network can paint a fixed map. Check that the added detail changes from month to month, and score de-seasonalised detail.

**Coarse test:** yes, but human-use maps act mainly below 600 km, so a null result at coarse scales would not settle it.

### 7. Structure metrics for the model-world tests
**Source.** Goracci 2026 (Sect. 4.1, Table 4): Fractions Skill Score for the driest and wettest 10%, SSIM, gradient similarity, low-pass and high-pass scores. Dombos 2026 and Xiong 2026: skill as a function of basin area.

**What we change.** Add to level-1 and level-3 evaluation. **Coarse test:** yes.

### 8. A learned model-to-GRACE correction at checkable scales (addresses e) — lower priority
**Sources.** Sun 2019 (target = Noah − GRACE); Wang 2024 (WGHM → GRACE-like).

**What we change.** Train a network on the residual between GRACE detail and model detail at 8→16 and 16→32. Risk: it is a relation learned at one scale and applied at another, which is exactly what failed in our octave-transfer experiment. Worth a test only as an amplitude calibration, with the transfer 8→16 → 16→32 as the check.

---

## 4. Approaches our results suggest would not work, or that carry a risk we have seen

| Approach | Papers | Why it is at risk for us |
|---|---|---|
| Pixel loss toward one model's state for the same month | Gou & Soja 2024; Uz 2026; Li 2026; Vishwakarma 2021; Suryawanshi 2026 | This is our paired experiment. It made agreement with GRACE's own detail worse and halved the detail power. The papers' own numbers point the same way: Uz 2026 loses amplitude and trend and scores below the mascon at wells (0.33 vs 0.39); Li 2026 gets 51% / 49% at wells and no budget gain over its target (0.80 vs 0.79); the catchment PLSR product has positive gain in only 3 mascons (Suryawanshi 2026). None of them tested the detail against GRACE at a checkable scale |
| Fit at GRACE resolution, apply to finer predictors (relations assumed scale-invariant) | Zhang 2026; Mousavimehr 2025; Pascal 2022 (as a stated hypothesis) | Our octave-transfer result. Zhang's own validation shows attenuation (slope 0.51) and a sign error in the High Plains. Suryawanshi 2026 and Gou & Soja 2024 criticise this assumption explicitly |
| Soft consistency with GRACE (penalty on block or patch means) | Gou & Soja 2024; Uz 2026; Li 2026; Goracci 2026; Xiong 2026; Zhang 2026 | Leaves 1.3-3.8 cm disagreement with GRACE and forces a weighting trade-off the authors themselves describe (Li 2026, Sect. 3.2; Uz 2026, Sect. 5). Our projection has neither problem. Keep it |
| Deterministic networks with MSE or MAE loss as the source of detail | Chen 2025; Sarkar 2026; Arzoumanidis 2026; Gou et al. 2025 | Amplitude is damped: 10-30% (Chen), variance ratio 0.77 (Sarkar), smoothed extremes (Arzoumanidis). Sarkar's per-cell variance rescaling restores variance but cannot make an ensemble |
| Uncertainty from seeds, dropout or weight particles alone | Li 2026; Goracci 2026; Uz 2026; Mo 2022; Mo 2025 | Captures training randomness only. No paper checks coverage. Goracci says its spread is not a physical uncertainty. Our own spread is already too narrow; these methods would not fix it |
| Training on a smooth surrogate and applying to GRACE without checking the gap | Steidl 2026; He 2021 | RMSE triples on real GRACE (Steidl). This is the same model-to-GRACE gap we carry; our coarse-scale GRACE test is the control they lack |
| ConvLSTM or other recurrent models on short records | Steidl 2026; Lu 2024 | Worse than climatology beyond one month with < 400 sequences. Our ten-model training set is much larger, so the risk is smaller but not gone |
| Graph networks | Arzoumanidis 2026; Steidl 2026 | No ablation in the first; 0.05-0.12 cm in the second |
| GAN losses | Wang 2024; Heller-Kaikov 2023; Dombos 2026 | Modest or doubtful gain over the same generator without the adversarial term; unstable; no ensembles |
| Random splits of months, or spatial-only splits | Yin 2023; Palazzoli 2025; Mousavimehr 2025; Zhang 2026; Uz 2026; Li 2026 | Scores are optimistic because neighbouring months are correlated. Keep our block splits |
| Validation that reuses the model or the inputs | Uz 2026 (spectra vs WGHM, SMAP L4); Li 2026 and Goracci 2026 (budget with input fluxes); Suryawanshi 2026 (positive correlation with WaterGAP as a reliability criterion) | Circular. Do not adopt |
| Wells with a specific yield fitted to GRACE | Pascal 2022; Zhang 2026 | Amplitude is then not independent of GRACE; only within-region contrasts are. 69.7% of cells with three or more wells have a well-to-well spread as large as the cell trend (Sarkar 2026, Sect. 3.9) |

---

## 5. Claims that contradict or strain our findings

1. **"Relations are the same at all scales."** Zhang 2026 (Sect. 2.4) states it as the premise of the method; Mousavimehr 2025 and Pascal 2022 rely on it. This contradicts our finding 1. None of the three tests it; Zhang's results are compatible with it failing.
2. **"The model's fine detail is realistic and should be inherited."** Gou & Soja 2024, Uz 2026 (Sect. 4.1.3), Li 2026 and the assimilation literature (Springer 2026, Sect. 5.2.1). We found that pairing to WaterGAP's detail worsens agreement with GRACE's own detail at 600-1250 km. Not a direct contradiction, because their claim concerns scales below 300 km and ours scales above 600 km, and because they never test against GRACE's detail. It is the central open disagreement.
3. **Evidence that model patterns do help below the footprint.** Suryawanshi 2026: the Gou & Soja product has positive gain over GRACE at wells in 10 mascons and is "reliable" in 16. Sarkar 2026 (Sect. 4.7): inside depletion hotspots, WaterGAP and SeDA patterns correlate better with wells (0.58, 0.56) than a GRACE-only reconstruction (0.37). Xiong 2026: wells improve at 67.7%. This fits our "mixing helps a little" result better than our "paired is worse" result, and suggests the model's same-month state has more value where human water use dominates. Türkiye and its surroundings are such a region.
4. **Direction of the amplitude mismatch.** We found GRACE's detail about twice as strong as WaterGAP's. Most papers agree that models underestimate (Gou & Soja 2024; Uz 2026 Table 1; Sun 2019; Springer 2026; Sarkar 2026: factor 1.58). But:
   - Xiong 2026 (Sect. 4.4.1, Figs. 8-9) describes WGHM as having *more* high-degree power than DDK3-filtered GRACE.
   - Li 2026 (Sect. 4.1): PCR-GLOBWB *overestimates* amplitude in three humid basins.
   - Mandal 2025 (Introduction, citing Bibi et al. 2024): global hydrological models overestimate peaks in snow-dominated basins.
   
   So the sign depends on region, model, filter and degree band. Our factor of two should be reported per region and per band, with the GRACE filter stated, and not as one global number.
5. **"Precipitation and temperature are the key predictors."** The reconstruction papers are built on this. But where a model's TWS is offered as well, it dominates (Mandal 2025), and P and T add little (Sun 2019). No contradiction with our results; it argues against expecting much from P and T channels once model fields are in.
6. **Latitude and longitude as inputs.** Gou & Soja 2024 and Heller-Kaikov 2023 include them. Sarkar 2026 removes them "so that the network could not learn position as a proxy"; Palazzoli 2025 found a shared model with coordinates worse than per-cell models. Our random rotations already prevent memorising geography; a latitude channel partly undoes that. Treat it as an experiment, not a default.
7. **Nothing in the folder contradicts** the value of the exact projection, the stochastic sampler, or the shared-noise result. Nothing addresses them either: no paper uses a diffusion model for GRACE.

---

## 6. Read status

Read in full = main text; supplements were not available for any paper. Figures were not inspected. "Reader" = read by a reader agent from the extracted text; the compiler merged the notes and spot-checked key facts by text search.

| Paper | Status | Not read |
|---|---|---|
| Arzoumanidis 2026 | Full (compiler) | reference list skimmed |
| Chen 2025 | Full (reader) | – |
| Gao 2025 (CIDR; file `CIDR2025_SSA_interp.pdf`) | Full (reader) | tables partly garbled |
| Dombos 2026 | Full (reader) | – |
| Goracci 2026 | Full main text, App. A, start of App. B (reader) | rest of App. B |
| Gou 2025 (ocean bottom pressure) | Full (reader) | supplement |
| Gou 2026 (uncertainty review) | **In part**: abstract, end of Sect. 1, end of 2.2, Sects. 2.3, 3 (all), 4 intro, start of 4.1.2, 4.2, 5, App. A-B | start of Sect. 1, 2.1, 4.1.1, Table 1 |
| Gou & Soja 2024 | Full main text and Methods (reader) | supplement (altimetry comparison) |
| He 2021 | Full (reader) | – |
| Heller-Kaikov 2023 | Full (reader) | – |
| Humphrey & Gudmundsson 2019 | Full (reader) | supplement |
| Li 2026 | Sects. 1-6 full, start of 7 (reader) | – |
| Lu 2024 | Full (reader) | tables 2-4 garbled; numbers from text |
| Mandal 2025 | Full (reader) | – |
| Mo 2021 (preprint) | **In part**: abstract, Sects. 3.2-5, captions, read for differences from the journal version | Sects. 1-3.1 |
| Mo 2022 | Full (reader) | supporting material |
| Mo 2025 | Full main text (reader) | Supporting Information, which holds architecture, loss and SVGD details |
| Mousavimehr 2025 | Full (reader) | – |
| Pakrashi 2026 | Full incl. appendices (reader) | – |
| Palazzoli 2025 | Full (reader) | supplementary tables |
| Pascal 2022 | Full (reader) | – |
| Sarkar 2026 | **In part**: abstract, Sects. 1, 2 (all), 3.6-3.11, 4.2-4.7 | Sects. 3.1-3.5, 3.12 on, 4.1, 5 |
| Springer 2026 | Full body, Tables 1-5, App. A (reader) | – |
| Steidl 2026 | Full incl. appendices (reader) | – |
| Sun 2019 | Full main text and appendix (reader) | Supporting Information |
| Sun & Scanlon 2019 | **In part**: abstract, Table 1 GRACE row, Sects. 4.2.2, 4.2.3, 5; keyword search of the rest | Sects. 1-3, 4.1, 4.2.1 |
| Suryawanshi 2026 | Full (reader) | – |
| Uz 2024 | Full main text (reader) | supplement |
| Uz 2026 | Full main text (reader) | supplement |
| Vishwakarma 2021 | Full (reader) | – |
| Wang 2024 | Full (reader) | – |
| Xiong 2026 | Sects. 1-5 and 7 full (reader) | appendix figures, supplement |
| Yin 2023 | Full (reader) | supplement |
| Zhang 2026 | **In part**: abstract, Sects. 1, 2, 3.1-3.3, 3.5-3.6, 4.1-4.3, 6 | most of Sect. 3.4 |

**Inconsistencies inside papers, noted while reading** (check the PDF before quoting): Arzoumanidis 2026 (0.693 given both for 2002-2023 and for validation); Pascal 2022 (mean R 0.79 in abstract, 0.74-0.76 in conclusion); Suryawanshi 2026 (abstract and body disagree on mascon counts and maximum gains); Gao 2025 (two sets of medians for the long gap); Uz 2024 (number of ERA5 inputs); Yin 2023 (basin-size thresholds); Dombos 2026 (2.3 cm in abstract vs Table II); Sun 2019 (sign of the well correlation looks odd for a Noah − GRACE target).

**Cited inside these papers, not in the folder, not read** (candidates for the fetch list; do not cite as read): Gerdener et al. 2023 (GLWS2.0); Tourian et al. 2023 (copula-based Bayesian downscaling with posterior spread); Mehrnegar et al. 2021, 2023 (Bayesian merging of GRACE and W3RA); Bäthge et al. 2026 (GROW well compilation); Jasechko et al. 2024 (global wells); Hugonnet et al. 2021 (glacier elevation change); Hirschi & Seneviratne 2017 (BSWB); Klos et al. 2021 and Springer et al. 2019 (GNSS loading vs assimilation, Europe); Li et al. 2021 (RM-REC reconstruction); Gentner et al. (DeepRec, CNN + LSTM reconstruction); Li & Kusche 2026 (GRACE-FCast).
