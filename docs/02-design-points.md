# 02 — Agreed design points (notes, 2026-11-18)

*Notes from the discussion of 18 November 2026. These seven points are agreed; the design
documents (`00-design.md`, `01-design-model-trained.md`) have not been rewritten yet and will be
replaced by a full write-up of this design later. Nothing here is built.*

**In one sentence:** the network takes a coarse GRACE map, adds fine detail learned from
WaterGAP, and leaves GRACE's own large-scale picture untouched.

## 1. Input: GRACE only

- One coarse GRACE map for a single month, limited to degree 32 (about 600 km).
- No auxiliary inputs (precipitation, temperature, topography) for now.
- Possible later addition: static topography and a land/water mask.

## 2. Loss: on the fine detail only

- Both maps are split at GRACE's resolution. The loss compares only the part finer than GRACE.
- The coarse part is never in the loss; the consistency projection fixes it to GRACE's values,
  so blurring the output gives GRACE back exactly.
- Reason: a whole-map loss would pull GRACE's large-scale picture towards the model (for
  example, weakening groundwater depletion that GRACE sees and the model misses).
- Known weakness: where the model misses a signal, the network adds no fine detail for it.
  The large-scale signal is kept but not sharpened.

## 3. Target: WaterGAP; GLWS as a benchmark

- Training target: the fine detail of WaterGAP v2.2e.
- GLWS2.0 (GRACE assimilated into WaterGAP, Gerdener et al. 2023) is a benchmark, not a
  validation: it is built from the same two ingredients, so agreement is partly by construction.
  Compare at the same resolution (GLWS smoothed to our target degree).
- The other nine ISIMIP3a models are kept aside as a side check, not as targets.
- Consequence: the ensemble spread expresses the ambiguity of the detail given GRACE, assuming
  WaterGAP's fine-scale behaviour is right. It does not know about WaterGAP's own errors.
- To do: obtain and read the GLWS2.0 paper (not on disk); confirm its grid and period.

## 4. Training: pre-train, then fine-tune

1. Pre-train on WaterGAP alone, 1901–2001: blurred WaterGAP in, WaterGAP detail out.
2. Fine-tune on paired months: real GRACE in, WaterGAP detail out.

- Paired months exist only for April 2002 – December 2019 (179 months).
- Fallbacks if needed: paired only (simplest), or mixing both kinds of sample in one run.

## 5. Model: diffusion, with a direct network as baseline

- Diffusion model as built (U-Net, EDM, widened noise schedule), now with the stochastic
  sampler.
- A direct (regression) network with the same architecture is trained alongside as the
  baseline: it answers whether diffusion buys anything over a plain network.
- The diffusion ensemble mean serves as the single best-estimate map.

## 6. Checks

Split of the paired months (WaterGAP ends in 2019, so the old 2020–2022 validation years
cannot be used):

| | period | months |
|---|---|---|
| fine-tune | 2002-04 to 2014-12 | 140 |
| validation | 2015-01 to 2017-06 (end of GRACE) | 22 |
| test | 2018-06 to 2019-12 (GRACE-FO) | 17 |

The Türkiye box stays held out spatially. Pre-training never sees 2002–2019.

Four checks, weakest to strongest:

1. **Held-out years against WaterGAP.** Did it learn the task? Compared with the direct network
   and the spectral baseline; spread checked.
2. **GRACE's own detail at coarse steps (8→16, 16→32).** The same design trained at scales GRACE
   can see, scored against what GRACE measured. The only global check against observations.
   To beat: 22.1 and 29.3 mm (current approach, stochastic sampler with inflation).
3. **GLWS at fine scales.** Benchmark; reported as a comparison, no pass or fail.
4. **Independent data in Türkiye** (wells, GNSS loading, basin budgets; to be found). The only
   real proof at fine scales.

Success for checks 1 and 2: beats the spectral baseline and the direct network, with
spread/error between 0.85 and 1.15. The validation and test periods are short, so confidence
intervals will be wide.

## 7. Resolution: degree 128; one step or two decided by experiment

- Target: degree 128 (about 160 km). Degree 256 only later, if this works and it is wanted.
- One step (32→128) or two (32→64→128) is tested, not decided up front.
- Cost: whole-globe training at degree 128 does not fit the 32 GB card at a useful batch size
  (the 16→32 run already used 29.6 GB). Fine-scale training needs patch-based training, which
  is a code change.

Testing order:

**Stage 1 — coarse scales (cheap; GRACE can check).**

| variant | tests |
|---|---|
| A. pre-trained only (exists) | the reference |
| B. pre-train + fine-tune, two steps (8→16, 16→32) | does the design beat the current one? |
| C. the same in one step (8→32) | one step against two |
| D. paired only, no pre-training | is pre-training needed? |
| E. direct network for B or C | does diffusion buy anything? |

**Stage 2 — fine scales.** Build patch training; run the stage 1 winner at 32→128. If stage 1
is close between one and two steps, run both.

**Stage 3 — degree 256**, only if stage 2 works.

## Still open

- Independent validation data for Türkiye.
- Confirmation of the GRACE product and filter for the input.
- Whether the nine other models should later return as targets (ten models beat WaterGAP alone
  in the G1 test: 24.0 against 28.0 mm at 8→16).
