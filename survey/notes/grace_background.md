# GRACE/GRACE-FO mission background and classical data processing

Sub-topic notes for the survey "Enhancing GRACE/GRACE-FO satellite gravimetry using deep learning and diffusion models".
Compiled 2026-11-13. PDFs are in `literature/grace_background/`. BibTeX is in `survey/notes/grace_background.bib`. The keys match.

**Citation rule (from the user):** cite only papers with status FULLTEXT_READ. FULLTEXT_READ here means the PDF was converted with `pdftotext`, the abstract was read, and every claim in the summary below was checked against the text with targeted searches. It does not mean every page was read. Summaries of all other papers are based on metadata and abstracts and have **not** been checked against the full text.

Status counts: 50 verified papers. 18 FULLTEXT_READ, 0 FULLTEXT_AVAILABLE_UNREAD, 32 PAYWALLED, 0 ABSTRACT_ONLY. One more item (Jekeli 1981) is UNVERIFIED and is not in the .bib.

The 32 PAYWALLED papers include about 15 that are open access at the publisher, mostly AGU/Wiley and some Nature, but the publishers block automated downloads (HTTP 403). The user can fetch these in a browser. Each one is marked "publisher OA" in the table below.

---

## Verification of the user's preliminary review (sections 2.1 and 6.3)

Source checked: `survey/notes/preliminary_review_2026-11-12.md`.

| Claim in review | Verdict | Evidence |
|---|---|---|
| CSR releases a 0.25° grid with a true resolution of about 200–300 km (Bonin et al., 2026) | **Correct.** Two caveats. | Bonin et al. 2026, ESSD 18, 3481–3505, doi:10.5194/essd-18-3481-2026, Sect. 2: "Though the data is released on a uniform 1/4° grid, the true resolution of the data is ∼200–300 km, because the regularization uses information with ∼200 km resolution and the mascon signal is still limited by the fundamental ∼300 km spatial resolution defined by GRACE/FO's orbit altitude." The underlying mascons are ~120 km hexagons (40,962 of them). Caveat 1: the paper is about an *ocean-dynamics* variant of the CSR mascons, so it is a secondary source for land TWS resolution. Save et al. (2016) should be cited alongside it as the primary CSR reference. Caveat 2: the reference list gives only "Bonin et al. (2026)". The full author list is Bonin, Pie, Tamisiea, Chambers, Save. |
| GSFC 1° mascons have an effective resolution near 520 km outside Antarctica (Loomis et al., 2019) | **Essentially correct, but needs nuance and a better citation.** | Loomis, Luthcke & Sabaka 2019, J. Geod. 93, doi:10.1007/s00190-019-01252-y (the NTRS PDF is the accepted manuscript of this paper). From the resolution-operator analysis: "appropriate values are z=6 (resolution ∼270 km) for regional analysis of the Antarctic Ice Sheet, and z=22 (∼520 km) for regional analysis everywhere else." This is the minimum *regional-aggregation* scale recommended for analysis. It is not a per-grid-cell resolution. The ~270 km value for the AIS should be stated too. The reference list should cite the J. Geod. article and DOI rather than "NASA Technical Reports Server". |
| Smallest resolvable catchment ≈63,000 km² at 2 cm error (Vishwakarma et al., 2018) | **Correct.** Add the qualifier below. | Vishwakarma, Devaraju & Sneeuw 2018, Remote Sens. 10(6), 852, doi:10.3390/rs10060852, abstract: "a catchment of the size ≈63,000 km² can be resolved at an error level of 2 cm". This holds only for the *scaling* and *data-driven* correction schemes applied to spherical-harmonic (SH) products. For a 2 cm error the paper gives ≈152,000 km² for filtered/additive products and ≈810,000 km² for multiplicative ones. For a 1 cm error the best case is ≈250,000 km². The study does **not** cover mascons. |
| JPL: 4,551 equal-area 3° mascons delivered on a 0.5° grid | Consistent with Watkins et al. (2015) and the JPL documentation (not re-fetched; Watkins 2015 is PAYWALLED/publisher OA). | — |
| GRACE-C scheduled for 2028, laser ranging, polar orbit (AEI press release) | **Correct, but the source is dated.** | The AEI press release is from 19 March 2024. Current sources (fetched 2026-11-13) give the same picture. GFZ project page: launch "end of 2028" on a Falcon-9, polar orbit at about 500 km altitude and 89° inclination, satellites 100–300 km apart. DLR: "late 2028", aiming for overlap with GRACE-FO. Gunter's Space Page: December 2028 from Vandenberg, at least 5 years of operations. One important detail is missing from the review: **GRACE-C carries no microwave (KBR) instrument. Ranging is by LRI only.** NASA supplies a new Scale Factor Unit because the LRI scale can no longer be calibrated against the KBR (GFZ page). A news commentary from September 2026 (newspaceeconomy.ca, a secondary source) says a NASA audit uses July 2029 as the committed launch-readiness date and reports narrowing reserves. That is UNVERIFIED against the primary audit. Suggested citations: https://www.gfz.de/en/section/global-geomonitoring-and-gravity-field/projects/grace-c-gravity-recovery-and-climate-experiment-continuity and https://www.dlr.de/en/rb/research-operation/missions/earth-observation-science/future-1/grace-c . |
| NGGM pair and MAGIC reduce "flood-signal noise" over the Danube from 29.3 to 17.7 and 13.0 mm (Pfeffer et al., 2026) | **Numbers correct. Description is imprecise.** | Pfeffer et al. 2026, EGUsphere preprint egusphere-2026-4642 (open for discussion, *not peer-reviewed*), Appendix B "Flood monitoring and prediction", lines 780–784. These numbers are the **standard deviation of the noise (residuals vs. truth) in simulated basin-average TWS** used to set up data assimilation. For the Danube the signal std is 59.4 mm and the noise is 29.3 mm (GRACE-C), 17.7 mm (NGGM) and 13.0 mm (MAGIC). For the Brahmaputra the signal is 165.5 mm and the noise is 41.7/17.4/12.3 mm. The data are unfiltered, simulated 5-day fields for 2003–2006. The "GRACE-C" case is a *simulated* single polar pair, not real GRACE-FO data. Better wording: "basin-average TWS noise in closed-loop simulation". The NGGM orbit given there (~400 km, 70° inclination, citing Daras 2026) is also correct. |
| The review's own limitation: "programmatic status of NGGM ... not confirmed" | Still open. | Haagmans et al. (2020) and Pfeffer et al. (2026) describe NGGM and MAGIC as a concept. I did not verify an ESA launch date. Pfeffer et al. say NGGM would continue alone after GRACE-C ends, with reduced high-latitude coverage. |

No outright factual errors were found. The issues are caveats and citation hygiene. Bonin 2026 is a secondary source for land resolution. The 520 km figure is an aggregation scale. The 63,000 km² figure applies only to the best correction schemes on SH data. The Pfeffer numbers are basin-noise std in simulation, not "flood-signal noise", and come from a non-peer-reviewed preprint. The GRACE-C source is from 2024, and the review omits that GRACE-C is LRI-only.

---

## Summary of the sub-field

**Mission design and measurement.** GRACE (March 2002 to October 2017; decommissioned November 2017) and GRACE-FO (launched 22 May 2018) are twin satellites in a polar orbit at about 450–500 km altitude, separated by about 220 km. They measure inter-satellite range changes with a K/Ka-band microwave ranging system (KBR/MWI). Each satellite also carries a SuperSTAR electrostatic accelerometer for non-gravitational forces, GPS receivers for precise orbits and star cameras for attitude (Tapley et al. 2004a,b; Kornfeld et al. 2019; Chen et al. 2022). GRACE-FO added a laser ranging interferometer (LRI) as a technology demonstrator. In orbit the LRI reaches a ranging noise of about 1 nm/√Hz above 100 mHz, far below the KBR (Abich et al. 2019). Pre-launch simulations showed that the gain at the level of monthly gravity fields is limited, about 23% globally and only a few percent after DDK filtering. The reason is that accelerometer noise and errors in ocean-tide and non-tidal background models dominate the error budget (Flechtner et al. 2016; Kvas & Mayer-Gürr 2019; Dobslaw et al. 2017). This matters for ML because the residual errors in Level-2 fields are structured by the instruments and by aliasing. They are not white noise.

**Level-2 products: spherical harmonics vs. mascons.** The Science Data System centres CSR, GFZ and JPL, plus others such as ITSG-Graz and COST-G, invert Level-1B data into monthly Stokes coefficients up to degree and order 60–96 (Dahle et al. 2019; Kvas et al. 2019; Jäggi et al. 2020; Wahr et al. 1998 for the mass-from-Stokes theory). The SH solutions show north–south "stripes" from correlated errors. These come from the along-track sensitivity of the measurement and from undersampled sub-monthly signals. The fields therefore need post-processing. The options are Gaussian smoothing (Jekeli 1981; Wahr et al. 1998), polynomial destriping of even/odd coefficients (Swenson & Wahr 2006), or the decorrelation and anisotropic filters of Kusche (2007; DDK1–8, Kusche et al. 2009). Each of these attenuates the signal and leaks it across boundaries. That makes "restoration" corrections necessary: model-based scale factors (Landerer & Swenson 2012), additive or multiplicative corrections, and data-driven approaches (Vishwakarma et al. 2017). Mascon solutions take a different route. They estimate surface-mass parcels directly from Level-1B data (GSFC: Luthcke et al. 2013; Loomis et al. 2019) or from regularised SH normal equations (JPL: Watkins et al. 2015; CSR: Save et al. 2012, 2016). Regularisation built from prior/geophysical information suppresses the stripes. Wiese et al. (2016) handle land/ocean leakage with a Coastline Resolution Improvement filter and gain factors. Mascons are now the default for hydrology (Scanlon et al. 2016). They are biased, however. Loomis et al. (2019) show that regularised mascon errors must be split into a noise part and a leakage/bias part, the latter computed from the resolution operator. Ensemble means across centres reduce noise further (Sakumura et al. 2014; COST-G).

**Effective resolution.** The fields carry far less independent information than their delivery grids imply. Orbit altitude limits recoverable signal to about 300 km half-wavelength in practice. CSR mascons are delivered on a 0.25° grid but have a true resolution of about 200–300 km (Bonin et al. 2026). GSFC recommends aggregating to about 520 km for regional analysis outside Antarctica and about 270 km inside it (Loomis et al. 2019). For corrected SH products, catchments smaller than about 63,000 km² cannot be resolved better than 2 cm EWH (Vishwakarma et al. 2018). Rodell et al. (2018) give a monthly TWS resolution of roughly 200,000 km² at low latitudes and 90,000 km² near the poles. This is the core physical limit for super-resolution. Fine-scale structure in any downscaled product must come from a prior such as a hydrological model or covariates. A learned method should therefore be judged on whether it stays consistent with GRACE at the scales GRACE actually resolves.

**Low-degree terms, GIA and background models.** GRACE determines C20 poorly and C30 poorly after the accelerometer problems, and it cannot observe degree-1 (geocentre) at all. Standard practice replaces C20 and C30 with satellite laser ranging values (Cheng & Ries 2017; Loomis et al. 2019b, 2020; GRACE TN-14) and adds degree-1 from GRACE plus ocean models (Swenson et al. 2008; Sun et al. 2016). These choices change ice-sheet trends by tens of Gt/yr. Loomis et al. (2019b) report −15.4 Gt/yr for Antarctica from the C20 choice alone. Glacial isostatic adjustment (GIA) has to be removed with a model, either ICE-6G_D (Peltier et al. 2018) or A et al. (2013). It is the dominant error term for Antarctic mass balance (Velicogna & Wahr 2006; Chen et al. 2022). High-frequency atmosphere and ocean mass is removed with AOD1B (Dobslaw et al. 2017). Errors in this dealiasing product and in ocean tides cause temporal aliasing (Daras & Pail 2017). Kvas & Mayer-Gürr (2019) put background-model uncertainty into the stochastic model and obtain up to 25% noise reduction.

**Record gaps and GRACE-FO degradation.** The record has an 11-month gap from July 2017 to May 2018 between the two missions (Chen et al. 2022; Landerer et al. 2020). It also has scattered missing months late in the GRACE mission, caused by battery management. The count is UNVERIFIED here: roughly 20 is commonly quoted, and the exact number should be taken from the product documentation. On GRACE-FO, the accelerometer on GF2 began showing correlated bias jumps and noise on 21 June 2018. Since then its data have been replaced with a "transplant" from GF1 plus thruster modelling (ACT1B product). The same was done for the last seven GRACE months (Landerer et al. 2020; Harvey et al. 2022). Even with these problems, GRACE-FO continues the GRACE record at equivalent precision (Landerer et al. 2020) and keeps ice-sheet loss estimates continuous (Velicogna et al. 2020). For ML work, the gap months and the transplant months are where the quality and error structure of the data change.

**Applications and future missions.** GRACE has shown groundwater depletion in northern India (Rodell et al. 2009), 34 regional freshwater trends with attributed causes (Rodell et al. 2018), and ice-sheet and glacier loss (Velicogna & Wahr 2006; Luthcke et al. 2013; Velicogna et al. 2020). It has also contributed to closing the sea-level budget. These results are reviewed by Tapley et al. (2019), Wouters et al. (2014), Chen et al. (2022) and Rodell & Reager (2023). TWS is now an Essential Climate Variable. For continuity, GRACE-C is scheduled for late 2028 in a polar orbit at about 500 km with LRI-only ranging. NGGM, ESA's inclined pair at about 70° and 400 km, would combine with GRACE-C in the MAGIC Bender-type constellation. This would double space-time sampling and reduce aliasing (Haagmans et al. 2020; Pail et al. 2015). Closed-loop simulations show substantially lower TWS noise for NGGM and MAGIC (Pfeffer et al. 2026, preprint). Further out, cold-atom quantum accelerometers and gradiometers are being studied, with modest gains of 10–25% in simulation (Lévèque et al. 2021).

**Open problems relevant to ML enhancement.**
1. There is no high-resolution truth below about 300 km. This motivates generative and probabilistic methods that sample plausible fine-scale structure under an explicit GRACE observation operator (resolution operator, filter kernel or mascon basis).
2. Error covariances are structured, anisotropic and leakage-biased. Full covariances or resolution operators exist for some products (Loomis et al. 2019; ITSG). They are rarely used in ML training losses.
3. The 11-month gap, the scattered missing months and the change in error regime at the accelerometer transplant are natural targets for gap filling. Validation should be stratified across these regimes.
4. Low-degree replacements, GIA model choice and background models introduce systematic uncertainty. ML products should state the conventions they inherit.
5. MAGIC and NGGM will change the input information content. Methods trained on GRACE-like inputs need a path to future higher-resolution inputs.

---

## Papers

Format: key — citation — DOI — summary — status.

### Mission, instruments, overview reviews

**tapley2004grace** — Tapley, B. D., Bettadpur, S., Ries, J. C., Thompson, P. F., Watkins, M. M. (2004). GRACE measurements of mass variability in the Earth system. *Science*, 305(5683), 503–505. DOI: 10.1126/science.1099192.
(Summary based on metadata/abstract; not checked against full text.) The first science results from GRACE. Monthly gravity fields resolve hydrological mass variability at scales of a few hundred km. Canonical citation for the mission concept. — **PAYWALLED**

**tapley2004mission** — Tapley, B. D., Bettadpur, S., Watkins, M., Reigber, C. (2004). The gravity recovery and climate experiment: Mission overview and early results. *Geophys. Res. Lett.*, 31, L09607. DOI: 10.1029/2004GL019920.
(Summary based on metadata/abstract; not checked against full text.) The mission overview: twin satellites, KBR, accelerometers, GPS, orbit design, and early gravity-model performance. — **PAYWALLED** (publisher OA; automated download blocked)

**tapley2019contributions** — Tapley, B. D., Watkins, M. M., Flechtner, F., et al. (2019). Contributions of GRACE to understanding climate change. *Nature Climate Change*, 9, 358–369. DOI: 10.1038/s41558-019-0456-2.
(Summary based on metadata/abstract; not checked against full text.) A landmark review of 15 years of GRACE results covering ice sheets, glaciers, sea level, terrestrial water storage, groundwater and ocean circulation. It is the standard citation for the scientific importance of the record. — **PAYWALLED** (Unpaywall lists a nature.com PDF, but the download failed)

**kornfeld2019gracefo** — Kornfeld, R. P., Arnold, B. W., Gross, M. A., et al. (2019). GRACE-FO: The Gravity Recovery and Climate Experiment Follow-On mission. *J. Spacecraft Rockets*, 56(3), 931–951. DOI: 10.2514/1.A34326.
(Summary based on metadata/abstract; not checked against full text.) An engineering description of GRACE-FO: spacecraft, instruments including the LRI, launch in May 2018 and early operations. — **PAYWALLED**

**landerer2020extending** — Landerer, F. W., Flechtner, F. M., Save, H., et al. (2020). Extending the global mass change data record: GRACE Follow-On instrument and science data performance. *Geophys. Res. Lett.*, 47, e2020GL088306. DOI: 10.1029/2020GL088306.
Shows that GRACE-FO fields from June 2018 to December 2019 continue the GRACE record "at an equivalent precision and spatiotemporal sampling". Documents the GF2 accelerometer degradation: "Starting on 21 June 2018, bias jumps and noise that are highly correlated across all six ACC axes". The fix is an accelerometer "full transplant" from GF1 plus thruster calibration, released as the new ACT1B product, the same approach used in the last seven GRACE months. It also describes LRI and KBR data acquired together. The key reference for the GRACE→GRACE-FO transition and the change in data quality. — **FULLTEXT_READ** (Landerer2020_GRACEFOperformance.pdf, GFZ repository)

**abich2019inorbit** — Abich, K., Abramovici, A., Amparan, B., et al. (2019). In-orbit performance of the GRACE Follow-on laser ranging interferometer. *Phys. Rev. Lett.*, 123, 031101. DOI: 10.1103/PhysRevLett.123.031101.
The first in-orbit demonstration of laser interferometric ranging between satellites. The LRI operated for more than 50 days without breaks in phase tracking. Its range measurements are biased like the microwave instrument's but have "much less noise at a level of 1 nm/√Hz at Fourier frequencies above 100 mHz". This is the basis for LRI-only ranging on GRACE-C. — **FULLTEXT_READ** (Abich2019_LRIperformance.pdf)

**flechtner2016lri** — Flechtner, F., Neumayer, K.-H., Dahle, C., et al. (2016). What can be expected from the GRACE-FO laser ranging interferometer for Earth science applications? *Surv. Geophys.*, 37, 453–470 (online 2015). DOI: 10.1007/s10712-015-9338-y.
A full-scale five-year closed-loop simulation. The LRI improves unfiltered monthly fields by about 23% globally. The gain is at wavelengths below about 240 km with Gaussian filtering and drops to a few percent after DDK filtering. The study identifies accelerometer noise, followed by ocean-tide and non-tidal background model errors, as the main error contributors. This is evidence that better ranging alone does not raise resolution. — **FULLTEXT_READ** (Flechtner2016_LRIexpectations.pdf, GFZ repository)

**harvey2022accelerometer** — Harvey, N., McCullough, C. M., Save, H. (2022). Modeling GRACE-FO accelerometer data for the version 04 release. *Adv. Space Res.*, 69(3), 1393–1407. DOI: 10.1016/j.asr.2021.10.056.
(Summary based on metadata/abstract; not checked against full text.) Describes how GF2 accelerometer data are modelled and transplanted for the RL04 Level-1B release, including thruster and residual modelling. Relevant to the error regime of the transplant era. — **PAYWALLED** (an accepted manuscript exists on ScienceDirect, but the download failed)

**chen2022applications** — Chen, J., Cazenave, A., Dahle, C., et al. (2022). Applications and challenges of GRACE and GRACE Follow-On satellite gravimetry. *Surv. Geophys.*, 43, 305–345. DOI: 10.1007/s10712-021-09685-x.
A comprehensive review of instruments, processing (SH, mascons, filters, leakage) and applications in hydrology, ocean, cryosphere and solid Earth. It discusses longitudinal stripes, following Swenson & Wahr's even/odd correlation, and leakage from filtering. It covers SLR replacement of C20 and C30 (TN-14), the large effect of C30 on Antarctica, GIA as the dominant AIS error source, and the "about 1-year gap (July 2017–May 2018)" between missions. The best single modern review for this sub-topic. — **FULLTEXT_READ** (Chen2022_applicationsChallenges.pdf)

**wouters2014grace** — Wouters, B., Bonin, J. A., Chambers, D. P., Riva, R. E. M., Sasgen, I., Wahr, J. (2014). GRACE, time-varying gravity, Earth system dynamics and climate change. *Rep. Prog. Phys.*, 77, 116801. DOI: 10.1088/0034-4885/77/11/116801.
(Summary based on metadata/abstract; not checked against full text.) A physics-oriented review of GRACE theory, processing and Earth-system applications. — **PAYWALLED** (a TU Delft repository copy may exist; not downloaded)

**rodell2023water** — Rodell, M., Reager, J. T. (2023). Water cycle science enabled by the GRACE and GRACE-FO satellite missions. *Nature Water*, 1, 47–59. DOI: 10.1038/s44221-022-00005-0.
A hydrology-focused review. It traces satellite gravimetry from early scepticism to a staple of large-scale water-cycle science and the only source of the Essential Climate Variable TWS. It notes that GRACE-FO "jumped NASA's Earth Science mission queue" and that a third Mass Change mission is in development. — **FULLTEXT_READ** (Rodell2023_waterCycleScience.pdf, NTRS preprint/accepted version)

### Theory and spherical-harmonic processing

**wahr1998time** — Wahr, J., Molenaar, M., Bryan, F. (1998). Time variability of the Earth's gravity field: Hydrological and oceanic effects and their possible detection using GRACE. *J. Geophys. Res.*, 103(B12), 30205–30229. DOI: 10.1029/98JB02844.
(Summary based on metadata/abstract; not checked against full text.) The foundational paper on converting Stokes coefficients to surface mass (EWH), including load Love numbers and Gaussian averaging kernels. The basis of all SH-to-TWS processing. — **PAYWALLED** (publisher OA; automated download blocked)

**swenson2002methods** — Swenson, S., Wahr, J. (2002). Methods for inferring regional surface-mass anomalies from GRACE measurements of time-variable gravity. *J. Geophys. Res.*, 107(B9), 2193. DOI: 10.1029/2001JB000576.
(Summary based on metadata/abstract; not checked against full text.) Optimal averaging kernels for basin-scale mass estimates and the trade-off between leakage and noise. — **PAYWALLED** (publisher OA; automated download blocked)

**swenson2006postprocessing** — Swenson, S., Wahr, J. (2006). Post-processing removal of correlated errors in GRACE data. *Geophys. Res. Lett.*, 33, L08402. DOI: 10.1029/2005GL025285.
(Summary based on metadata/abstract; not checked against full text.) The classic "destriping" filter. It fits and removes smooth polynomials from even and odd degree coefficients of each order, which suppresses north–south stripes. Still a common baseline. — **PAYWALLED** (publisher OA; automated download blocked)

**wahr2006accuracy** — Wahr, J., Swenson, S., Velicogna, I. (2006). Accuracy of GRACE mass estimates. *Geophys. Res. Lett.*, 33, L06401. DOI: 10.1029/2005GL025305.
(Summary based on metadata/abstract; not checked against full text.) An empirical error assessment of GRACE monthly mass estimates as a function of smoothing radius and latitude. — **PAYWALLED** (publisher OA; automated download blocked)

**kusche2007approximate** — Kusche, J. (2007). Approximate decorrelation and non-isotropic smoothing of time-variable GRACE-type gravity field models. *J. Geod.*, 81, 733–749. DOI: 10.1007/s00190-007-0143-3.
Introduces the DDK-type filter. A regularisation built from a synthetic error covariance (from GRACE orbits) and an a-priori signal covariance "mimics" Tikhonov regularisation of the normal equations. It reduces stripes more efficiently than isotropic Gaussian filters while keeping signal amplitudes. The paper notes that Gaussian filtering biases amplitudes and ignores both the latitude-dependent data density and the anisotropic error correlations. — **FULLTEXT_READ** (Kusche2007_DDK.pdf)

**kusche2009decorrelated** — Kusche, J., Schmidt, R., Petrovic, S., Rietbroek, R. (2009). Decorrelated GRACE time-variable gravity solutions by GFZ, and their validation using a hydrological model. *J. Geod.*, 83, 903–913. DOI: 10.1007/s00190-009-0308-3.
(Summary based on metadata/abstract; not checked against full text.) Defines the DDK1–3 filter family and validates the filtered fields against WGHM. — **PAYWALLED**

**landerer2012accuracy** — Landerer, F. W., Swenson, S. C. (2012). Accuracy of scaled GRACE terrestrial water storage estimates. *Water Resour. Res.*, 48, W04531. DOI: 10.1029/2011WR011453.
(Summary based on metadata/abstract; not checked against full text.) Derives gridded scale factors from a land-surface model (CLM4) to restore signal lost to destriping and smoothing. Quantifies the leakage and measurement errors of the GRACE Tellus land grids. Its scaling approach is a standard baseline. — **PAYWALLED**

**vishwakarma2017datadriven** — Vishwakarma, B. D., Horwath, M., Devaraju, B., Groh, A., Sneeuw, N. (2017). A data-driven approach for repairing the hydrological catchment signal damage due to filtering of GRACE products. *Water Resour. Res.*, 53, 9824–9844. DOI: 10.1002/2017WR021150.
(Summary based on metadata/abstract; not checked against full text.) A model-independent leakage correction that uses only GRACE data and catchment geometry. An alternative to model-based scaling. — **PAYWALLED** (publisher OA; automated download blocked)

**vishwakarma2018resolution** — Vishwakarma, B. D., Devaraju, B., Sneeuw, N. (2018). What is the spatial resolution of GRACE satellite products for hydrology? *Remote Sens.*, 10(6), 852. DOI: 10.3390/rs10060852.
A closed-loop simulation over 255 catchments with four correction schemes (multiplicative, additive, scaling, data-driven) applied to SH products. Closure error falls as catchment area grows. At 2 cm EWH error, about 63,000 km² is resolvable with the scaling or data-driven correction, but about 152,000 km² for filtered/additive and about 810,000 km² for multiplicative. At 1 cm error the best case is about 250,000 km². This is the standard citation for effective resolution and for why "scale factor" claims must be qualified. — **FULLTEXT_READ** (Vishwakarma2018_spatialResolutionGRACE.pdf)

**dahle2019gfz** — Dahle, C., Murböck, M., Flechtner, F., et al. (2019). The GFZ GRACE RL06 monthly gravity field time series: Processing details and quality assessment. *Remote Sens.*, 11(18), 2116. DOI: 10.3390/rs11182116.
(Summary based on metadata/abstract; not checked against full text.) Processing standards and quality assessment of the GFZ RL06 SH series. — **PAYWALLED** (MDPI OA; the download returned non-PDF this session, so the user can fetch it manually)

**kvas2019itsg** — Kvas, A., Behzadpour, S., Ellmer, M., et al. (2019). ITSG-Grace2018: Overview and evaluation of a new GRACE-only gravity field time series. *J. Geophys. Res. Solid Earth*, 124, 9332–9344. DOI: 10.1029/2019JB017415.
(Summary based on metadata/abstract; not checked against full text.) The TU Graz ITSG-Grace2018 series: monthly and daily (Kalman) solutions with full covariances and low noise. It is a candidate input when error covariances are needed. — **PAYWALLED** (publisher OA; automated download blocked)

**jaggi2020costg** — Jäggi, A., Meyer, U., Lasser, M., et al. (2020). International Combination Service for Time-Variable Gravity Fields (COST-G): Start of operational phase and future perspectives. In *Beyond 100: The Next Century in Geodesy*, IAG Symposia 152, Springer, pp. 57–65. DOI: 10.1007/1345_2020_109.
COST-G is a new IGFS product centre. It combines monthly SH solutions or normal equations from several analysis centres (CSR, GFZ, JPL, AIUB, ITSG, CNES, and others) under consistent standards. The combined solution has lower noise than any single input. — **FULLTEXT_READ** (Jaggi2020_COSTG.pdf)

**sakumura2014ensemble** — Sakumura, C., Bettadpur, S., Bruinsma, S. (2014). Ensemble prediction and intercomparison analysis of GRACE time-variable gravity field models. *Geophys. Res. Lett.*, 41, 1389–1397. DOI: 10.1002/2013GL058632.
(Summary based on metadata/abstract; not checked against full text.) Compares the CSR, GFZ, JPL and GRGS solutions. The ensemble mean reduces noise, and the inter-centre spread is a proxy for uncertainty. Relevant to ensemble-based uncertainty in ML. — **PAYWALLED** (publisher OA; automated download blocked)

### Mascon solutions

**save2012regularization** — Save, H., Bettadpur, S., Tapley, B. D. (2012). Reducing errors in the GRACE gravity solutions using regularization. *J. Geod.*, 86, 695–711. DOI: 10.1007/s00190-012-0548-5.
(Summary based on metadata/abstract; not checked against full text.) Tikhonov regularisation of GRACE SH solutions with an L-curve choice of the parameter. The precursor to the CSR mascons. — **PAYWALLED**

**watkins2015mascons** — Watkins, M. M., Wiese, D. N., Yuan, D.-N., Boening, C., Landerer, F. W. (2015). Improved methods for observing Earth's time variable mass distribution with GRACE using spherical cap mascons. *J. Geophys. Res. Solid Earth*, 120, 2648–2671. DOI: 10.1002/2014JB011547.
(Summary based on metadata/abstract; not checked against full text.) The JPL RL05M mascons: 3° equal-area spherical-cap mascons (4,551), estimated with Bayesian constraints from geophysical models. They need no post-hoc destriping. — **PAYWALLED** (publisher OA; automated download blocked)

**wiese2016leakage** — Wiese, D. N., Landerer, F. W., Watkins, M. M. (2016). Quantifying and reducing leakage errors in the JPL RL05M GRACE mascon solution. *Water Resour. Res.*, 52, 7490–7502. DOI: 10.1002/2016WR019344.
Mascons are constrained "within a Bayesian framework to remove correlated errors rather than relying on empirical filters". Even so, they need different post-processing. The paper introduces the Coastline Resolution Improvement (CRI) filter, which reduces land/ocean leakage by 50% in simulation to about 1 cm EWH residual globally. It also introduces gain factors for continental hydrology, with the largest leakage reductions (1–2 cm) in small basins. — **FULLTEXT_READ** (Wiese2016_JPLmasconLeakage.pdf, Caltech repository)

**save2016csr** — Save, H., Bettadpur, S., Tapley, B. D. (2016). High-resolution CSR GRACE RL05 mascons. *J. Geophys. Res. Solid Earth*, 121, 7547–7569. DOI: 10.1002/2016JB013007.
(Summary based on metadata/abstract; not checked against full text.) The CSR mascons: about 120 km hexagonal mascons on a geodesic grid, released on a 0.25° grid, with time-variable Tikhonov regularisation. The primary reference for the CSR mascon product. — **PAYWALLED**

**luthcke2013iterated** — Luthcke, S. B., Sabaka, T. J., Loomis, B. D., et al. (2013). Antarctica, Greenland and Gulf of Alaska land-ice evolution from an iterated GRACE global mascon solution. *J. Glaciol.*, 59(216), 613–631. DOI: 10.3189/2013JoG12J147.
The GSFC global mascon solution, estimated directly from K-band range-rate observations with full noise covariance and formal iteration. It uses 10-day and 1° equal-area sampling with anisotropic constraints. For December 2003 to December 2010 it gives a combined GIS+AIS+GOA trend of −380 ± 31 Gt/a, equal to 1.05 mm/a of sea-level rise, and an acceleration of −41 ± 27 Gt/a². — **FULLTEXT_READ** (Luthcke2013_GSFCmascons.pdf)

**loomis2019regularization** — Loomis, B. D., Luthcke, S. B., Sabaka, T. J. (2019). Regularization and error characterization of GRACE mascons. *J. Geod.*, 93, 1381–1398. DOI: 10.1007/s00190-019-01252-y.
The new GSFC mascon product from Level-1B data, with iterative, data-driven, time-dependent regularisation. The paper shows mathematically that regularised mascons are biased, so total uncertainty must combine noise variance with leakage computed from monthly resolution operators. The recommended regional-analysis resolution is about 270 km for the AIS and about 520 km elsewhere. It closes the sea-level budget. The key reference for error characterisation and the resolution operator, both of which are directly usable as an observation operator for conditional generative models. — **FULLTEXT_READ** (Loomis2019_regularizationMascons.pdf, NTRS accepted manuscript)

**scanlon2016mascons** — Scanlon, B. R., Zhang, Z., Save, H., et al. (2016). Global evaluation of new GRACE mascon products for hydrologic applications. *Water Resour. Res.*, 52, 9412–9429. DOI: 10.1002/2016WR019494.
(Summary based on metadata/abstract; not checked against full text.) Compares CSR and JPL mascons with SH products across basins. Mascons show less leakage and higher signal amplitudes. A widely used justification for choosing mascons in hydrology. — **PAYWALLED** (publisher OA; a HAL copy is listed but was not retrieved)

**bonin2026ocean** — Bonin, J., Pie, N., Tamisiea, M. E., Chambers, D., Save, H. (2026). GRACE and GRACE-FO mascons for ocean dynamic applications. *Earth Syst. Sci. Data*, 18, 3481–3505. DOI: 10.5194/essd-18-3481-2026.
A new CSR-based ocean-dynamics mascon series. It removes barystatic GRD signals and four major oceanic earthquakes and adjusts the regularisation to reduce Arctic and orbital errors. Sect. 2 describes the standard CSR mascons (40,962 hexagons of about 120 km, 0.25° grid) and states the true resolution is about 200–300 km. It also recalls the common advice to ignore GRACE ocean grids within 300 km of land because of leakage. — **FULLTEXT_READ** (Bonin2026_oceanMascons.pdf, final ESSD version)

### Low-degree replacement, GIA, background models, aliasing

**swenson2008geocenter** — Swenson, S., Chambers, D., Wahr, J. (2008). Estimating geocenter variations from a combination of GRACE and ocean model output. *J. Geophys. Res.*, 113, B08410. DOI: 10.1029/2007JB005338.
(Summary based on metadata/abstract; not checked against full text.) The standard degree-1 (geocentre) estimate from GRACE plus an ocean model (GRACE TN-13 heritage). — **PAYWALLED** (publisher OA; a USF repository copy failed to download)

**sun2016geocenter** — Sun, Y., Riva, R., Ditmar, P. (2016). Optimizing estimates of annual variations and trends in geocenter motion and J2 from a combination of GRACE data and geophysical models. *J. Geophys. Res. Solid Earth*, 121, 8352–8370. DOI: 10.1002/2016JB013073.
(Summary based on metadata/abstract; not checked against full text.) An improved degree-1 and C20 estimate that accounts for GIA and leakage. Used in the current TN-13 products. — **PAYWALLED** (publisher OA; automated download blocked)

**cheng2017c20** — Cheng, M., Ries, J. (2017). The unexpected signal in GRACE estimates of C20. *J. Geod.*, 91, 897–914. DOI: 10.1007/s00190-016-0995-5.
(Summary based on metadata/abstract; not checked against full text.) Identifies a spurious signal in GRACE-derived C20, probably thermally or tidally related. This justifies replacing it with SLR values. — **PAYWALLED**

**loomis2019oblateness** — Loomis, B. D., Rachlin, K. E., Luthcke, S. B. (2019). Improved Earth oblateness rate reveals increased ice sheet losses and mass-driven sea level rise. *Geophys. Res. Lett.*, 46, 6910–6917. DOI: 10.1029/2019GL082929.
A new SLR C20 product that includes GRACE low-degree information in the forward model. Compared with CSR TN-11, it changes AIS trends by −15.4 Gt/yr and GIS trends by −3.5 Gt/yr and improves closure of the sea-level budget by +0.08 mm/yr. Recommended for RL06 (the basis of TN-14). — **FULLTEXT_READ** (Loomis2019_C20oblateness.pdf, NTRS accepted manuscript)

**loomis2020c30** — Loomis, B. D., Rachlin, K. E., Wiese, D. N., Landerer, F. W., Luthcke, S. B. (2020). Replacing GRACE/GRACE-FO C30 with satellite laser ranging: Impacts on Antarctic Ice Sheet mass change. *Geophys. Res. Lett.*, 47, e2019GL085488. DOI: 10.1029/2019GL085488.
(Summary based on metadata/abstract; not checked against full text.) Shows that C30 is poorly determined when accelerometer data are degraded (late GRACE, GF2) and recommends replacing it with SLR, which matters strongly for Antarctica. — **PAYWALLED**

**a2013gia** — A, G., Wahr, J., Zhong, S. (2013). Computations of the viscoelastic response of a 3-D compressible Earth to surface loading: an application to Glacial Isostatic Adjustment in Antarctica and Canada. *Geophys. J. Int.*, 192, 557–572. DOI: 10.1093/gji/ggs030. (Crossref "issued" year is 2012 for the online version; the print issue is 2013.)
(Summary based on metadata/abstract; not checked against full text.) The A13 GIA model, widely used for the GRACE GIA correction (e.g., in JPL and CSR products). — **PAYWALLED** (OUP lists OA, but the download failed)

**peltier2018ice6g** — Peltier, W. R., Argus, D. F., Drummond, R. (2018). Comment on "An assessment of the ICE-6G_C (VM5a) glacial isostatic adjustment model" by Purcell et al. *J. Geophys. Res. Solid Earth*, 123, 2019–2028. DOI: 10.1002/2016JB013844.
(Summary based on metadata/abstract; not checked against full text.) Introduces ICE-6G_D (VM5a), the GIA model most often used for GRACE corrections. — **PAYWALLED** (publisher OA; automated download blocked)

**dobslaw2017aod1b** — Dobslaw, H., Bergmann-Wolf, I., Dill, R., et al. (2017). A new high-resolution model of non-tidal atmosphere and ocean mass variability for de-aliasing of satellite gravity observations: AOD1B RL06. *Geophys. J. Int.*, 211, 263–269. DOI: 10.1093/gji/ggx302.
(Summary based on metadata/abstract; not checked against full text.) The AOD1B RL06 dealiasing product: 3-hourly atmosphere and ocean mass removed before inversion. Its errors are a leading aliasing source. — **PAYWALLED**

**kvas2019background** — Kvas, A., Mayer-Gürr, T. (2019). GRACE gravity field recovery with background model uncertainties. *J. Geod.*, 93, 2543–2552. DOI: 10.1007/s00190-019-01314-1.
Models the uncertainty of the AOD dealiasing product as an autoregressive process derived from the ESA Earth System Model and includes it in the GRACE stochastic model. Results: up to 25% RMS noise reduction for 250–300 km Gaussian radii, better agreement between formal and empirical errors, and monthly signal preserved. This shows background models remain a "limiting factor" of solution quality. — **FULLTEXT_READ** (Kvas2019_backgroundModelUncertainty.pdf)

**daras2017aliasing** — Daras, I., Pail, R. (2017). Treatment of temporal aliasing effects in the context of next generation satellite gravimetry missions. *J. Geophys. Res. Solid Earth*, 122, 7343–7362. DOI: 10.1002/2017JB014250.
(Summary based on metadata/abstract; not checked against full text.) Analyses temporal-aliasing errors and mitigation strategies (co-parameterisation, Bender constellations) for future missions. — **PAYWALLED**

### Key applications (hydrology, ice)

**velicogna2006antarctica** — Velicogna, I., Wahr, J. (2006). Measurements of time-variable gravity show mass loss in Antarctica. *Science*, 311, 1754–1756. DOI: 10.1126/science.1123785.
(Summary based on metadata/abstract; not checked against full text.) The first GRACE estimate of Antarctic ice mass loss. Highlights that the GIA correction is the dominant uncertainty. — **PAYWALLED**

**rodell2009india** — Rodell, M., Velicogna, I., Famiglietti, J. S. (2009). Satellite-based estimates of groundwater depletion in India. *Nature*, 460, 999–1002. DOI: 10.1038/nature08238.
(Summary based on metadata/abstract; not checked against full text.) Uses GRACE together with land-surface models to separate groundwater depletion in NW India. The landmark hydrology application. — **PAYWALLED**

**rodell2018emerging** — Rodell, M., Famiglietti, J. S., Wiese, D. N., et al. (2018). Emerging trends in global freshwater availability. *Nature*, 557, 651–659. DOI: 10.1038/s41586-018-0123-1.
Quantifies 34 regional TWS trends from GRACE for 2002–2016 and attributes them to natural variability, unsustainable groundwater use or climate change. States that GRACE monthly TWS is resolvable "over scales that approximately range from 200,000 km² at low latitudes to 90,000 km² near the poles". — **FULLTEXT_READ** (Rodell2018_freshwaterTrends.pdf, NTRS accepted manuscript)

**velicogna2020continuity** — Velicogna, I., Mohajerani, Y., A, G., et al. (2020). Continuity of ice sheet mass loss in Greenland and Antarctica from the GRACE and GRACE Follow-On missions. *Geophys. Res. Lett.*, 47, e2020GL087291. DOI: 10.1029/2020GL087291.
(Summary based on metadata/abstract; not checked against full text.) Shows that GRACE-FO continues the ice-sheet mass-loss record across the 2017–2018 gap and checks consistency between the missions. — **PAYWALLED** (publisher OA; automated download blocked)

### Future missions

**pail2015science** — Pail, R., Bingham, R., Braitenberg, C., et al. (2015). Science and user needs for observing global mass transport to understand global change and to benefit society. *Surv. Geophys.*, 36, 743–772. DOI: 10.1007/s10712-015-9348-9.
(Summary based on metadata/abstract; not checked against full text.) The IUGG-consolidated science requirements (target accuracy and resolution) for next-generation gravity missions. The basis for the NGGM and MAGIC requirements. — **PAYWALLED**

**haagmans2020nggm** — Haagmans, R., Siemes, C., Massotti, L., Carraz, O., Silvestrin, P. (2020). ESA's next-generation gravity mission concepts. *Rend. Fis. Acc. Lincei*, 31 (Suppl 1), S15–S25. DOI: 10.1007/s12210-020-00875-0.
Reviews ESA's NGGM studies since 2003: laser ranging, drag-free/electric propulsion and constellation optimisation. Describes the Bender double-pair constellation, one polar pair plus one at about 67° inclination, which doubles space-time sampling and adds east–west sensitivity to reduce aliasing. Also covers cold-atom interferometry sensor studies, including an airborne CAI gravimeter survey. — **FULLTEXT_READ** (Haagmans2020_NGGMconcepts.pdf)

**leveque2021quantum** — Lévèque, T., Fallet, C., Mandea, M., et al. (2021). Gravity field mapping using laser-coupled quantum accelerometers in space. *J. Geod.*, 95, 15. DOI: 10.1007/s00190-020-01462-9.
The CNES GRICE study. Two satellites at 373 km, each with a cold-atom accelerometer (6×10⁻¹⁰ m s⁻² Hz⁻¹/²), coupled by a laser link to form a long-baseline gradiometer. In simulation it improves on range-rate by 10–25% between 1000 and 222 km resolution. A representative quantum-gravimetry concept. — **FULLTEXT_READ** (Leveque2021_quantumAccelerometers.pdf, arXiv:2011.03382)

**pfeffer2026nggm** — Pfeffer, J., Meyssignac, B., Bingham, R., et al. (2026). Impact of the NGGM and MAGIC future satellite gravity missions for scientific applications and operational services. *EGUsphere* preprint (under review for Earth Observation). DOI: 10.5194/egusphere-2026-4642.
The ESA SING project. A closed-loop simulation with realistic instrument, tidal and aliasing errors evaluates GRACE-C (near-polar, about 500 km), NGGM (70°, about 400 km) and MAGIC across hydrology, ocean, ice and solid Earth. In flood data assimilation the basin-average TWS noise std is 29.3/17.7/13.0 mm (Danube) and 41.7/17.4/12.3 mm (Brahmaputra) for GRACE-C/NGGM/MAGIC. MAGIC and NGGM DA improve KGE by 21.8%/17.0% over GRACE-C. **Not peer-reviewed.** — **FULLTEXT_READ** (Pfeffer2026_NGGM_MAGIC_impact.pdf)

### Unverified / not in .bib

**jekeli1981gaussian** — Jekeli, C. (1981). Alternative methods to smooth the Earth's gravity field. Report 327, Dept. of Geodetic Science and Surveying, Ohio State University. No DOI. Canonical source of the Gaussian averaging kernel used by Wahr et al. (1998). — **UNVERIFIED** (metadata not checked against a primary record; not in .bib)

### Non-paper sources on GRACE-C status (for a footnote or URL citation, not in .bib)
- GFZ GRACE-C project page (fetched 2026-11-13): launch end of 2028, Falcon-9, polar orbit, about 500 km, 89°, separation 100–300 km, LRI-only ranging, new Scale Factor Unit, PODRIX GNSS receiver. https://www.gfz.de/en/section/global-geomonitoring-and-gravity-field/projects/grace-c-gravity-recovery-and-climate-experiment-continuity
- DLR GRACE-C page: "launch date of late 2028", overlap with GRACE-FO intended. https://www.dlr.de/en/rb/research-operation/missions/earth-observation-science/future-1/grace-c
- AEI press release, 19 March 2024: launch scheduled 2028. https://www.aei.mpg.de/1130150/erfolgreiche-deutsch-amerikanische-umweltmission-wird-mit-grace-c-fortgesetzt
- UNVERIFIED secondary source: newspaceeconomy.ca (23 Sep 2026) reports that a NASA audit uses July 2029 as the committed launch-readiness date and that reserves are narrowing.

---

## Paywalled

"Publisher OA" means free at the publisher but blocked for automated download. The user can fetch these in a browser.

| key | title | DOI |
|---|---|---|
| tapley2004grace | GRACE measurements of mass variability in the Earth system | 10.1126/science.1099192 |
| tapley2004mission | The gravity recovery and climate experiment: Mission overview and early results (publisher OA) | 10.1029/2004GL019920 |
| tapley2019contributions | Contributions of GRACE to understanding climate change | 10.1038/s41558-019-0456-2 |
| kornfeld2019gracefo | GRACE-FO: The Gravity Recovery and Climate Experiment Follow-On Mission | 10.2514/1.A34326 |
| harvey2022accelerometer | Modeling GRACE-FO accelerometer data for the version 04 release | 10.1016/j.asr.2021.10.056 |
| wouters2014grace | GRACE, time-varying gravity, Earth system dynamics and climate change | 10.1088/0034-4885/77/11/116801 |
| wahr1998time | Time variability of the Earth's gravity field: Hydrological and oceanic effects... (publisher OA) | 10.1029/98JB02844 |
| swenson2002methods | Methods for inferring regional surface-mass anomalies from GRACE... (publisher OA) | 10.1029/2001JB000576 |
| swenson2006postprocessing | Post-processing removal of correlated errors in GRACE data (publisher OA) | 10.1029/2005GL025285 |
| wahr2006accuracy | Accuracy of GRACE mass estimates (publisher OA) | 10.1029/2005GL025305 |
| kusche2009decorrelated | Decorrelated GRACE time-variable gravity solutions by GFZ... | 10.1007/s00190-009-0308-3 |
| landerer2012accuracy | Accuracy of scaled GRACE terrestrial water storage estimates | 10.1029/2011WR011453 |
| vishwakarma2017datadriven | A data-driven approach for repairing the hydrological catchment signal damage... (publisher OA) | 10.1002/2017WR021150 |
| dahle2019gfz | The GFZ GRACE RL06 monthly gravity field time series (MDPI OA) | 10.3390/rs11182116 |
| kvas2019itsg | ITSG-Grace2018: Overview and evaluation of a new GRACE-only gravity field time series (publisher OA) | 10.1029/2019JB017415 |
| sakumura2014ensemble | Ensemble prediction and intercomparison analysis of GRACE time-variable gravity field models (publisher OA) | 10.1002/2013GL058632 |
| save2012regularization | Reducing errors in the GRACE gravity solutions using regularization | 10.1007/s00190-012-0548-5 |
| watkins2015mascons | Improved methods for observing Earth's time variable mass distribution with GRACE using spherical cap mascons (publisher OA) | 10.1002/2014JB011547 |
| save2016csr | High-resolution CSR GRACE RL05 mascons | 10.1002/2016JB013007 |
| scanlon2016mascons | Global evaluation of new GRACE mascon products for hydrologic applications (publisher OA) | 10.1002/2016WR019494 |
| swenson2008geocenter | Estimating geocenter variations from a combination of GRACE and ocean model output (publisher OA) | 10.1029/2007JB005338 |
| sun2016geocenter | Optimizing estimates of annual variations and trends in geocenter motion and J2... (publisher OA) | 10.1002/2016JB013073 |
| cheng2017c20 | The unexpected signal in GRACE estimates of C20 | 10.1007/s00190-016-0995-5 |
| loomis2020c30 | Replacing GRACE/GRACE-FO C30 with satellite laser ranging: Impacts on Antarctic Ice Sheet mass change | 10.1029/2019GL085488 |
| a2013gia | Computations of the viscoelastic response of a 3-D compressible Earth to surface loading... (OUP OA) | 10.1093/gji/ggs030 |
| peltier2018ice6g | Comment on "An assessment of the ICE-6G_C (VM5a) GIA model" (ICE-6G_D) (publisher OA) | 10.1002/2016JB013844 |
| dobslaw2017aod1b | A new high-resolution model of non-tidal atmosphere and ocean mass variability for de-aliasing: AOD1B RL06 | 10.1093/gji/ggx302 |
| daras2017aliasing | Treatment of temporal aliasing effects in the context of next generation satellite gravimetry missions | 10.1002/2017JB014250 |
| velicogna2006antarctica | Measurements of time-variable gravity show mass loss in Antarctica | 10.1126/science.1123785 |
| rodell2009india | Satellite-based estimates of groundwater depletion in India | 10.1038/nature08238 |
| velicogna2020continuity | Continuity of ice sheet mass loss in Greenland and Antarctica from GRACE and GRACE-FO (publisher OA) | 10.1029/2020GL087291 |
| pail2015science | Science and user needs for observing global mass transport... | 10.1007/s10712-015-9348-9 |
