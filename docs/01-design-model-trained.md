# 01 — Design: model-trained detail, GRACE-constrained (draft)

*Draft, 2026-11-17, for discussion with the supervisor. Updated 2026-11-18: G1 has been run
(§8). Supersedes the octave-transfer route of `00-design.md` §2 after Phase 1 (results:
`00-design.md` Appendix B, `reports/phase1/phase1_report.pdf`).*

## 0. Where we are

Phase 1 asked whether a model that learns to add one octave of detail at coarse scales
can do the same at finer scales it never saw. **It cannot.** On GRACE and on a clean
hydrological-model twin alike, a model is excellent at the scale it trained on and no
better than a simple random-detail baseline one octave up, with over-confident
uncertainty. A model trained *at* a given scale, however, learns it well (the R3
oracle: CRPS 24.8 mm against 48.7 mm for the baseline).

So fine detail must be learned from examples at that scale. GRACE has none below
~500 km. Global hydrological models do.

## 1. The idea in one paragraph

Train the diffusion model on **hydrological-model** TWS fields: blur a model's field to
GRACE's resolution, and learn to restore the model's own fine detail. Then give it
**real GRACE** fields as input. The consistency projection we already have fixes every
coarse degree to GRACE's value exactly, so GRACE decides the large-scale picture and
the network adds only the detail below GRACE's resolution, as an ensemble. Use several
models so the network learns hydrology, not one model's quirks.

GRACE is the *input*, not the target. (Predicting GRACE from models is a different,
useful tool: correcting a model towards GRACE. Its output is no sharper than GRACE, so
it is not this project's goal. It could come later.)

## 2. Scales

| | degrees | half-wavelength | source |
|---|---|---|---|
| GRACE keeps (input) | ≤ 32 (DDK5: ≤ ~40 usable) | ≥ ~600 km | GRACE |
| Generated detail | 32 → 128 | 600 → ~160 km | learned from models |

Two ways to generate 32 → 128:
- **Cascade (recommended):** two stages, 32 → 64 and 64 → 128, each trained *at its own
  scale* on model data (no transfer: these are the twin R4 and R5 rungs trained
  directly). Reuses the rung code unchanged; each stage is a single octave, which
  we know works.
- **One step:** 32 → 128 on the 512 × 1024 grid. No error passed between stages, but a
  harder problem and heavier training.

Start with the cascade; try the one-step model if the cascade's errors compound.

## 3. Data

**Hydrological models (to confirm with the supervisor).** Requirements: global,
monthly (or daily, aggregated), ≤ 0.5°, total water storage or all its components
(soil, groundwater, snow, surface water), a long record. Candidates:
- WaterGAP v2.2e (already prepared, 1901–2019);
- other global models, e.g. the ISIMIP3a hydrology ensemble (several models, same
  forcing and protocol) — PCR-GLOBWB, H08, CWatM and others that report TWS;
- land-surface reanalyses (e.g. GLDAS), noting that some lack groundwater.

Each model is cleaned like the WaterGAP twin (runaway glacier and urban cells removed),
expanded to degree 128, and stored like `twin.npz`.

**Chosen for now (2026-11-18, pending confirmation):** ten ISIMIP3a water-sector models,
GSWP3-W5E5 `obsclim` `histsoc`, monthly TWS at 0.5°, 1901–2019: CWatM, H08, HydroPy,
JULES-ES-VN6P3, JULES-W2, LPJmL5-7-10-fire, MIROC-INTEG-LAND, ORCHIDEE-MICT, WaterGAP2-2e
(the twin) and WEB-DHM-SG. Fetched with `python -m grace_diffusion.fetch data/raw isimip`,
built with `python -m grace_diffusion.prep data/raw data/proc isimip` into
`isimip_<model>.npz`. Runaway cells removed: 68 (WEB-DHM-SG) to 5,603 (JULES-ES). Noted and
kept: JULES-ES varies 3–4× more than the others (real variability, not drift); CWatM
and MIROC show early-century spin-up offsets.

**Land-only GRACE** (`grace_ddk5_land`, `grace_raw_land`): ocean, Greenland and Antarctica set to
zero, and degrees 0/1 set to zero, so GRACE inputs cover the same domain as the models.

**GRACE.** Unchanged from Phase 1 (ITSG, TN-14, Caron GIA, DDK5 / unfiltered),
provisional pending confirmation.

## 4. The main risk: models and GRACE disagree

The network learns "fine detail given a *model's* coarse field". Real GRACE coarse fields
differ from models': models miss or misrepresent groundwater depletion, long-term trends,
glaciers and some human water use. Those differences reach the network as inputs it has
not seen. Ways to limit and to measure this:
- **Several models**, so the network sees a spread of coarse fields, not one model's.
- **Ice sheets and glaciers masked**, as in Phase 1.
- **Input augmentation:** add GRACE-like noise and coarse perturbations (e.g. random
  trends) to the training inputs.
- **Measure it directly** (gate G1 below) where GRACE can still check us.

## 5. Experiments and gates

All on validation years; test years and the Türkiye box stay untouched until the gates
are decided, as in Phase 1. Main score CRPS, plus spread/error and coverage; baseline
`spectral-gauss` (and `input`), paired block bootstrap.

**G0 — Within model (upper bound).** Train and test on the same model(s), held-out
years, at 32 → 64 and 64 → 128. Shows what is achievable without any mismatch. We
already have this for WaterGAP at one octave (Phase 1c R2: CRPS 7.2 vs 16.1 mm).

**G1 — Against real GRACE, at an observable scale.** Train on models at 8 → 16 and
16 → 32 (degrees GRACE resolves), apply to blurred GRACE, and score against GRACE's own
detail. Compare with the GRACE-trained models from Phase 1 (R2: 20.5 mm; R3 oracle:
24.8 mm) and `spectral-gauss` (45.9 / 48.7 mm). This is the only gate scored against real
data, and it measures the model-to-GRACE gap.
*Pass:* beats `spectral-gauss` with calibrated spread (spread/error 0.85–1.15); close to
the GRACE-trained models.

**G2 — Leave one model out, at the target scales.** Train on all models but one, test on
the held-out model at 32 → 64 and 64 → 128. Stands in for "trained on models, applied
to something else" where GRACE cannot check us.
*Pass:* beats `spectral-gauss` with calibrated spread for every held-out model.

**G3 — Türkiye (the old experiment D).** GRACE in, 32 → 128 out, in the test box.
Checked against independent data where available (e.g. well levels, basin water budgets,
high-resolution regional products), to be chosen with the supervisor.

## 6. What carries over and what is new

Carries over unchanged: spherical-harmonic tools, taper and rung grids, rotations, masks,
consistency projection, U-Net and EDM with the fixed noise schedule, baselines,
metrics, evaluation and the diagnostic.

New:
- `prep`: one builder per hydrological model (same output format as `twin.npz`);
- `data.Sampler`: draw training months from several datasets;
- `evaluate`: input dataset different from the training dataset (already almost
  supported: checkpoints and datasets are separate arguments);
- leave-one-model-out runs; input augmentation; the cascade in `evaluate`.

Rough effort: about a week of work once the model data are downloaded, plus training
time (each single-octave run is a few hours on the workstation).

## 7. Questions for the supervisor

1. **Which hydrological models**, and which experiment/forcing (e.g. ISIMIP3a)?
2. **Target resolution:** is degree 128 (~160 km half-wavelength) the right goal, or
   should we aim finer (degree 256, ~80 km; heavier and needs ≤ 0.25° models)?
3. **GRACE product and filter** for the input: keep ITSG with DDK5?
4. **Independent validation for G3:** what data exist for Türkiye (wells, basin
   budgets, regional products)?
5. **Scope of Phase 1 in a write-up:** the negative octave-transfer result plus the
   noise-schedule finding may be worth reporting on their own.

## 8. Results so far

### G1 (2026-11-18)

Setup:
- Two networks, both trained on R2 + R3 (8 → 16, 16 → 32) with the `wide` schedule,
  σ_max 20 and 60k steps:
  - `models`: the ten ISIMIP3a models, each sample drawn from a model chosen at random;
  - `grace-land`: land-only GRACE DDK5, training years only. This is the upper bound.
- Script: `scripts/g1_models.sh`. Scores: `runs/eval/G1_*.csv`.
- Validation is 2020–2022 (36 months), with a 6-month block bootstrap.

| input | rung | input | spectral-gauss | **models** | grace-land | spread/err | models − s.-gauss |
|---|---|---|---|---|---|---|---|
| DDK5 land | R2 | 45.8 | 33.8 | **24.0** | 13.7 | 0.66 | −9.9 [−11.5, −8.5] |
| DDK5 land | R3 | 50.7 | 38.1 | **30.3** | 21.8 | 0.64 | −7.9 [−8.6, −7.1] |
| raw land | R2 | 46.5 | 34.3 | **24.9** | 15.4 | 0.67 | −9.4 [−11.0, −8.0] |
| raw land | R3 | 54.3 | 40.6 | **33.5** | 26.1 | 0.64 | −7.1 [−7.8, −6.3] |

- **Skill:** the models network clearly beats the baseline. The correlation of its ensemble mean
  with GRACE's detail is 0.73 (R2) and 0.64 (R3), against 0.17 and 0.29 for the baseline.
- **Several models beat one:** WaterGAP alone scored 28.0 at R2 on DDK5 land; the ten models
  score 24.0.
- **Full GRACE DDK5** (ocean and ice included):
  - R3: 41.0 against 48.7, −7.7 [−8.9, −6.6]. In Phase 1 only the oracle, which was trained at
    R3, beat the baseline here.
  - R2: 44.3 against 45.9, −1.6 [−2.7, −0.8]. Here the ocean and ice detail dominates.
- **Gap to the GRACE-trained network:** 8–10 mm.
- **Spread too narrow:** spread/error is 0.64–0.67, and 90% coverage 72–76%.
  `grace-land` is narrow too (0.67–0.81), so this is partly the method itself.

**Verdict:** passed on skill, failed on calibration (gate 0.85–1.15).

**Next:**
1. Fix the spread. First the cheap options on the G1 checkpoints: sampler settings, and spread
   inflation fitted on validation years. Retrain (e.g. input augmentation) only if those fail.
2. Then G2 (leave one model out) at 32 → 64 and 64 → 128.
