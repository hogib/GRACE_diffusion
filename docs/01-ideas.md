# 01 — Ideas list

*Started 18 November 2026. A running list of things to try, so that none is forgotten. The
agreed design is `00-design.md`; results go into its Appendix B. Literature notes behind the
"from the papers" ideas: `survey/notes/dl_grace_for_design.md` (an agent's reading of
`literature/dl_grace/`; not yet checked against the papers by us).*

Status words: **running**, **done**, **next**, **open** (not started), **dropped**.

## A. Being tested now: extra inputs to the network ("wall", `scripts/wall.sh`)

Ten-model network at the coarse steps (8 → 16, 16 → 32), scored against GRACE's own detail.
Reference without extras: 18.6 / 33.0 mm (2015–2017) and 22.6 / 29.8 mm (2020–2022).

| # | idea | status |
|---|---|---|
| A1 | **History**: the blurred map 1, 2, 3, 6 and 12 months earlier | running |
| A2 | **Models' fields for the month**: the other models' detail for the same month. Exists only up to 2019 | running |
| A3 | **Models' seasonal climatology** of the detail. Exists for any month | running |
| A4 | **Land mask and latitude** | running |
| A5 | **All of the above together** | running |

## B. Ideas of ours

| # | idea | what it attacks | cost | status |
|---|---|---|---|---|
| B1 | **Use what GRACE knows between 300 and 600 km.** DDK5-filtered GRACE keeps about 40% of the signal at degrees 31–60 (design doc, Appendix A). (a) As a check: generate 32 → 64, apply the same filter, compare with filtered GRACE. (b) As an input: treat those degrees as a noisy measurement with known error and sample detail consistent with it (posterior sampling) | no validation below 600 km | (a) moderate: needs a 32 → 64 network and filter-aware scoring; (b) large | open; our top pick |
| B2 | **Per-place time behaviour as input maps**: long-term trend, size and timing of the yearly cycle of the coarse field. Tells a steady decline (pumping, ice) from a seasonal swing | accuracy | small: one more wall run | open |
| B3 | **Static maps**: topography, glacier cover, irrigated area, lakes and reservoirs, basin outlines | accuracy; model blind spots | moderate: downloads | open. Caution from the papers: static maps can dominate; check that detail still changes month to month |
| B4 | **Fix the amplitude gap in the training data**: rescale model fields to GRACE's large-scale strength by region; add GRACE-like noise and trends to training inputs | narrow spread; amplitude | small to moderate | open |
| B5 | **Harder unseen-world test**: hold out a model of a different family and forcing (e.g. a land-surface reanalysis), not just one of the ten | trust at fine scales | one or more runs; data | open |
| B6 | **Two-stage model**: direct network predicts the mean detail, diffusion generates the remainder | training ease; calibration | one run | open |
| B7 | **Several networks** trained on different model subsets, ensembles pooled | spread (which models to trust) | several runs | open |
| B8 | **GNSS loading in Türkiye** as independent validation: turn our maps into predicted station heights and compare | validation at fine scales | depends on station data | open; ask about data from the earlier project |
| B9 | **Leave one model out at the target scales** (design doc §7.1, level 3) | trust at fine scales; spread factor | several runs; needs patch training | open |
| B10 | **Patch training** so degree-128 grids fit the 32 GB card | needed for anything at target scales | code | open |
| B11 | **Temporal architecture** (attention over time, ConvLSTM) | accuracy | large | only if A1 shows a gain. Papers: stacked lags first; ConvLSTM and graphs added little |
| B12 | **Paired months for calibration only**: use GRACE–model pairs to set amplitude or spread, not to change what detail is drawn | spread; amplitude | small | open |

## C. Ideas from the papers (agent's reading)

| # | idea | source (as reported) | cost | status |
|---|---|---|---|---|
| C1 | **Model pattern, GRACE amplitude**: fit the models' patterns to GRACE in the band GRACE sees, reuse those amplitudes on the models' detail. Also a new baseline | Xiong 2026; Vishwakarma 2021; Mo 2025 | small, no training | next (baseline being run) |
| C2 | **Structured spread**: perturb each member's GRACE input with GRACE's own coefficient errors; a spatially and temporally correlated residual model in place of one spread factor; rank histogram of basin averages; 1/2/3-sigma coverage table | Gou 2026 review; Gou & Soja 2024; Humphrey 2019 | moderate | open |
| C3 | **Stacked lags before any temporal architecture** | Mo 2025; Mo 2022; Steidl 2026; Lu 2024 | — | this is A1 |
| C4 | **Calibrate the validation tools at coarse scale**: run well and water-budget comparisons at 16 → 32, where GRACE's detail is known, to see whether they rank methods correctly | the agent's own suggestion | moderate: data | open |
| C5 | **Weight the ten models** by their agreement with GRACE at coarse scales | Gou et al. 2025 | small | open |
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
| Same starting noise every month | done: members coherent in time at no cost; now the default | Appendix B.6 |
