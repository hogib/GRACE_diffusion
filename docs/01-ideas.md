# 01 — Ideas list

*Started 18 November 2026. A running list of things to try, so that none is forgotten. The
agreed design is `00-design.md`; results go into its Appendix B. Literature notes behind the
"from the papers" ideas: `survey/notes/dl_grace_for_design.md` (an agent's reading of
`literature/dl_grace/`; not yet checked against the papers by us).*

Status words: **running**, **done**, **next**, **open** (not started), **dropped**.

## A. Being tested now: extra inputs to the network ("wall", `scripts/wall.sh`)

Ten-model network at the coarse steps (8 → 16, 16 → 32), scored against GRACE's own detail.
Reference without extras: 18.5 / 33.2 mm (2015–2017) and 22.6 / 29.8 mm (2020–2022). Results: design doc, Appendix B.7.

| # | idea | status |
|---|---|---|
| A1 | **History**: the blurred map 1, 2, 3, 6 and 12 months earlier | done: no measurable gain |
| A2 | **Models' fields for the month**: the other models' detail for the same month. Exists only up to 2019 | done: −3.0 / −1.4 mm where the fields exist |
| A3 | **Models' seasonal climatology** of the detail. Exists for any month | done: −0.7 to −1.6 mm |
| A4 | **Land mask and latitude** | done: −1.7 to −3.8 mm, in both periods. Conflicts with the Türkiye hold-out (see design doc B.7) |
| A5 | **All of the above together** | done: best or near best (−2.1 to −3.1 mm); narrowest spread (0.59–0.66) |

## B. Ideas of ours

| # | idea | what it attacks | cost | status |
|---|---|---|---|---|
| B1 | **Use what GRACE knows between 300 and 600 km.** DDK5-filtered GRACE keeps about 40% of the signal at degrees 31–60 (design doc, Appendix A). (a) As a check: generate 32 → 64, apply the same filter, compare with filtered GRACE. (b) As an input: treat those degrees as a noisy measurement with known error and sample detail consistent with it (posterior sampling) | no validation below 600 km | (a) moderate: needs a 32 → 64 network and filter-aware scoring; (b) large | (a) in progress. Signal confirmed without training (design doc B.12): in 2004–2010 GRACE's unmasked band at strict degrees 33–60 correlates 0.37 with the filtered models' fields for the month, against 0.05–0.11 for candidates with no information. 2015–2017 too noisy. Network for 32 → 64 training. Formal-error draws are not a usable noise model in this band |
| B2 | **Per-place time behaviour as input maps**: long-term trend, size and timing of the yearly cycle of the coarse field. Tells a steady decline (pumping, ice) from a seasonal swing | accuracy | small: one more wall run | open |
| B3 | **Static maps**: topography, glacier cover, irrigated area, lakes and reservoirs, basin outlines | accuracy; model blind spots | moderate: downloads | open. Caution from the papers: static maps can dominate; check that detail still changes month to month |
| B4 | **Fix the amplitude gap in the training data**: rescale model fields to GRACE's large-scale strength by region; add GRACE-like noise and trends to training inputs | narrow spread; amplitude | small to moderate | rescaling: dropped (see F). Input noise in training: dropped. Spread/error +0.06 to +0.08, no CRPS gain, and no advantage with perturbed inputs at sampling time either (design doc B.9, B.11) |
| B5 | **Harder unseen-world test**: hold out a model of a different family and forcing (e.g. a land-surface reanalysis), not just one of the ten | trust at fine scales | one or more runs; data | open; raised in priority by B.10 (one of the ten held out is nearly as easy as a seen model) |
| B6 | **Two-stage model**: direct network predicts the mean detail, diffusion generates the remainder | training ease; calibration | one run | open |
| B7 | **Several networks** trained on different model subsets, ensembles pooled | spread (which models to trust) | several runs | open |
| B8 | **GNSS loading in Türkiye** as independent validation: turn our maps into predicted station heights and compare | validation at fine scales | depends on station data | open; ask about data from the earlier project |
| B9 | **Leave one model out at the target scales** (design doc §7.1, level 3) | trust at fine scales; spread factor | several runs; needs patch training | done at the coarse steps (design doc B.10): unseen model 5.3–5.8 / 7.1–8.2 mm against 4.9–5.5 / 6.7–7.9 seen and 17 / 32 on GRACE. Target scales: open |
| B10 | **Patch training** so degree-128 grids fit the 32 GB card | needed for anything at target scales | code | open |
| B11 | **Temporal architecture** (attention over time, ConvLSTM) | accuracy | large | A1 showed no gain: low priority. Papers: stacked lags first; ConvLSTM and graphs added little |
| B12 | **Paired months for calibration only**: use GRACE–model pairs to set amplitude or spread, not to change what detail is drawn | spread; amplitude | small | open |

## C. Ideas from the papers (agent's reading)

| # | idea | source (as reported) | cost | status |
|---|---|---|---|---|
| C1 | **Model pattern, GRACE amplitude**: fit the models' patterns to GRACE in the band GRACE sees, reuse those amplitudes on the models' detail. Also a new baseline | Xiong 2026; Vishwakarma 2021; Mo 2025 | small, no training | done as a baseline: no gain over simple scaling (see F) |
| C2 | **Structured spread**: perturb each member's GRACE input with GRACE's own coefficient errors; a spatially and temporally correlated residual model in place of one spread factor; rank histogram of basin averages; 1/2/3-sigma coverage table | Gou 2026 review; Gou & Soja 2024; Humphrey 2019 | moderate | input perturbation done at sampling time: with a small spread factor it meets both criteria on 2015–2017 (design doc B.8), but has almost no effect on 2020–2022, where GRACE-FO's formal errors are about five times smaller (B.11). Not a general fix. Rest open |
| C3 | **Stacked lags before any temporal architecture** | Mo 2025; Mo 2022; Steidl 2026; Lu 2024 | — | this is A1 |
| C4 | **Calibrate the validation tools at coarse scale**: run well and water-budget comparisons at 16 → 32, where GRACE's detail is known, to see whether they rank methods correctly | the agent's own suggestion | moderate: data | open |
| C5 | **Weight the ten models** by their agreement with GRACE at coarse scales | Gou et al. 2025 | small | open. Hint from C1: JULES-ES does not co-vary with GRACE over most land |
| C6 | **Precipitation and temperature as inputs**: reported to add little once a model TWS field is an input | Mandal 2025; Sun 2019 | moderate | low priority |

## D. Validation data named in the papers

- Wells: GGMN/IGRAC; GROW; Jasechko 2024 compilation.
- Water-budget closure: HydroBASINS (levels 4/5) with ERA5-Land P − ET − R; gains reported only
  for basins below about 63,000 km².
- Streamflow and basin balance: GSIM, GRDC; BSWB.
- Observed components: Hugonnet glacier elevation change; DAHITI reservoir levels.
- Comparison products: GLWS2.0; Gou & Soja 2024; SYSU TWSA; GRACE-SeDA.
- Nothing stated for Türkiye itself. Nearest: Tigris–Euphrates basin means; wells in Tehran,
  the North Caucasus and the Arabian aquifer (well correlations near zero in the last two).

## E. Things to keep in mind

- Report the GRACE-to-model amplitude ratio per region and band, not as one number; the papers
  show it depends on model, filter and region.
- Latitude or coordinates as input are debatable (one paper removed them on purpose); A4 will
  show their effect.
- Published designs that train towards a model's state for the month never test against
  GRACE's own detail; their well comparisons are mixed. Consistent with our paired result
  (design doc, Appendix B.5).
- Some evidence favours model patterns where human water use dominates; relevant for Türkiye.
- As reported, no paper in the folder uses an exact consistency projection, a generative model
  or CRPS. If confirmed, that is what is new in our approach.
- Arzoumanidis et al. 2026, as reported: no hydrological model; ERA5 fluxes to filtered GRACE at
  1° over South America; reconstruction at GRACE resolution, no uncertainty.

## F. Done or dropped

| idea | outcome | where |
|---|---|---|
| Learn detail from GRACE's coarser scales (octave transfer) | dropped: does not transfer | Appendix B.1, B.2 |
| Train on models, apply to GRACE | works; the current design | Appendix B.3 |
| Stochastic sampler; spread inflation | done: helps; GRACE-fitted factor passes, model-fitted does not | Appendix B.4 |
| Paired training towards WaterGAP's detail | dropped: worse against GRACE's own detail | Appendix B.5 |
| Models' own detail for the month, added to GRACE | done: worse than the network alone | Appendix B.6 |
| Mixing the network with the models' monthly detail | done: improves on the network (−2.2 / −1.0 mm) | Appendix B.6 |
| Model pattern with a per-pixel amplitude fitted to GRACE (C1) | done: 23.0 / 41.0 mm, worse than one global scale per month (20.6 / 37.9) and than the network (18.6 / 33.0). Fitted gains are 0.7–1.1, not 2: GRACE's extra strength is mostly signal the models do not have, not the same pattern at twice the size. One model (JULES-ES) gets a gain of zero over most land | Appendix B.6 |
| Same starting noise every month | done: members coherent in time at no cost; now the default | Appendix B.6 |
