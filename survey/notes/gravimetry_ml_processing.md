# ML in the satellite-gravimetry processing chain, complementary geodetic sensors, data assimilation and error characterisation

Scope: the geodetic, upstream and complementary side of "Enhancing GRACE/GRACE-FO with deep learning and diffusion models". This covers Level-1/Level-2 processing, low-degree coefficients, GIA, neural gravity-field representations, Swarm/SLR/GNSS/altimetry/InSAR, GRACE data assimilation and hybrid hydrological models, ice and ocean mass, uncertainty quantification and benchmark data. A separate notes file covers gridded-TWSA ML (gap filling, downscaling, reconstruction). This file includes only the gridded-TWSA papers that matter for the geodetic or assimilation argument.

BibTeX: `survey/notes/gravimetry_ml_processing.bib` (keys below match the BibTeX keys).
PDFs: `literature/gravimetry_ml_processing/`.

**Status legend (per coordinator rule change):**
- FULLTEXT_READ: the PDF is in `literature/gravimetry_ml_processing/`. I extracted its text with `pdftotext` and checked the summary against it (at least the abstract, key points, method description and conclusions). Where the local PDF is a preprint or accepted-manuscript version, the entry says so.
- FULLTEXT_AVAILABLE_UNREAD: none (every downloaded PDF was read).
- PAYWALLED: no local full text; the DOI is given so the user can fetch it. Entries marked "(publisher OA, automated download blocked)" are open access at the publisher, PMC or a repository, but bot protection (Cloudflare or PMC proof-of-work) blocked curl. They can be downloaded manually in a browser.
- ABSTRACT_ONLY: no full text could be found anywhere (none in this list).
- **Only FULLTEXT_READ papers may be cited in the review** until the user supplies the others.

---

## Summary of the sub-field

**Where ML enters the gravimetry pipeline.** The GRACE/GRACE-FO chain runs from Level-1 instrument data to Level-2 spherical harmonics, then to post-processing (destriping and filtering, replacement of the low-degree terms C20, C30 and degree-1, GIA correction), and finally to Level-3 grids or mascons. ML so far touches this chain mostly at the edges.

At Level-1, the GRACE-D accelerometer failure made "accelerometer transplant" an operational necessity:
- The JPL/SDS model-based transplant (Harvey et al. 2022) and the physics-based recovery from non-gravitational force models (Behzadpour et al. 2021) both measurably improved C20/C30.
- ML versions of transplant are now being studied: MLP, LSTM and BiLSTM transplant of cold-atom-interferometer accelerometer data in closed-loop simulations (Romeshkani et al. 2025), and Noise2Noise-style self-supervised denoising of GRACE-FO and Taiji-1 accelerometer data (Yang et al. 2023).
- We found no peer-reviewed ML work on KBR/LRI ranging noise or attitude data. The LRI performance baseline is documented in Landerer et al. (2020). This is an open niche.

At Level-2, ML has so far been used to *detect* correlated (striped) orders in Stokes coefficients so that decorrelation can be applied selectively (Piretzidis et al. 2018). More recently it has been used to *separate* superimposed gravity signals and noise with image-to-image networks: a PCA baseline versus a conditional GAN (Heller-Kaikov et al. 2023), then a multi-channel U-Net framework (Heller-Kaikov et al. 2026). Heller-Kaikov et al. (2026) is the closest existing analogue to a "learned destriping filter" and the natural baseline for a diffusion-based Level-2 denoiser. The classical filters it must beat are the Swenson & Wahr (2006) destriping and the DDK decorrelation of Kusche (2007).

For mascons, regularisation is still hand-designed: CSR's time-variable Tikhonov regularisation (Save et al. 2016), JPL's CRI/a priori constraints (Watkins et al. 2015), and GSFC's regularisation from binned residuals (Loomis et al. 2019). Loomis et al. (2019) state explicitly that regularised mascons are biased (leakage) and that total uncertainty must combine leakage with noise variance. That is exactly what a learned prior would have to quantify. We found no published learned-prior or ML-regularised mascon inversion. This is an explicit gap and a strong motivation for score-based or diffusion priors.

**Neural gravity-field representations.** PINN gravity models (PINN-GM I–III; Martin & Schaub 2022a,b, 2023) and neural density fields such as geodesyNets (Izzo & Gómez 2022) learn a potential or density as a continuous, differentiable function constrained by Laplace's equation or Newtonian integration. PINN-GM-III adds fixes for feature divergence, low-altitude sample bias, numerical instability and extrapolation error, plus six evaluation metrics. These models have so far been applied to static fields of the Earth, Moon and small bodies, not to time-variable GRACE fields. A time-conditioned neural field, or a diffusion model whose output is constrained to be harmonic or band-limited, is an untested way to merge mascon and SH representations.

**Low degrees and GIA.** C20 and degree-1 are taken from SLR or computed from GRACE plus ocean models: Loomis et al. 2019 for C20, Cheng & Ries 2017 on the anomalous C20 signal, and Swenson et al. 2008 and Sun et al. 2016 for degree-1 and the J2 combination. No peer-reviewed ML method for C20 or degree-1 was found; searches surfaced only thesis-level attempts. For GIA, ML so far appears only as fast neural surrogates of forward 3D-Earth GIA models (Love et al. 2024). Bayesian hierarchical spatio-temporal models that jointly separate GIA, ice and firn using GRACE, altimetry and GPS (Zammit-Mangion et al. 2015) are the closest probabilistic precedent for generative source separation.

**Complementary sensors.**
- Swarm GPS-derived gravity fields are informative only to roughly SH degree 12 and need about 750 km smoothing to agree with GRACE at about 4 cm EWH over land (Teixeira da Encarnação et al. 2020; Lück et al. 2018).
- Swarm can be "lifted" to GRACE-type resolution by projecting onto GRACE EOFs (Richter et al. 2021). Uz et al. (2022) use deep learning with Swarm and ERA5 for the same purpose.
- SLR does the same for the pre-GRACE era: EOF-based hybrid SLR fields (Löcher & Kusche 2021) and a residual deep convolutional autoencoder trained on mascons, SLR and ERA5 from 1994 to 2021 (Uz et al. 2024).
- GNSS vertical loading displacements give an independent, higher-resolution TWS constraint. Argus et al. (2014) used GPS alone. Carlson et al. (2022) jointly inverted GNSS and GRACE with wavelet decomposition. Recent ML versions use LSTM weighting (Shen et al. 2022) and a 3D-CNN that beats a Green's-function inversion (Wei et al. 2025).
- Altimetry enters through sea-level-budget joint inversions (Rietbroek et al. 2016) and global water-cycle reanalysis (van Dijk et al. 2014).
- InSAR enters through downscaling of groundwater storage (Reshi et al. 2023).

These sensors give a diffusion model either conditioning information or independent validation data.

**Data assimilation and hybrid models.**
- Classical GRACE data assimilation runs from the EnKS/EnKF in Catchment and NLDAS (Zaitchik et al. 2008; Kumar et al. 2016; Li et al. 2019; Girotto et al. 2017) to calibration plus assimilation in WaterGAP (Eicker et al. 2014). Girotto et al. (2017) show the pitfall that DA cannot fix trends the model lacks, such as groundwater pumping. Khaki et al. (2017) handle correlated GRACE errors through localisation, and Mehrnegar et al. (2020) use Bayesian dynamic model averaging.
- ML-augmented DA started with CNNs learning the model–GRACE mismatch (Sun et al. 2019). It matured into self-supervised deep-learning "data assimilation" that downscales GRACE TWSA to 0.5° with deep-ensemble uncertainty (Gou & Soja 2024), and similarly downscales ocean bottom pressure to 0.25° (Gou et al. 2025).
- Differentiable hybrid hydrology trains embedded neural networks against GRACE TWS together with runoff, ET, SWE and fAPAR: H2M (Kraft et al. 2022) and H2MV (Baghirov et al. 2025). Shen et al. (2023) review the differentiable-modelling paradigm.

In both families GRACE acts as a weak, coarse-scale constraint. A diffusion model can serve as the learned prior or emulator inside such a scheme, for example an ensemble generator replacing the EnKF background, or a generative posterior sampler.

**Error characterisation and evaluation.**
- A probabilistic ML product must be validated against established GRACE error models. Wahr et al. (2006) give latitude- and smoothing-dependent, approximately Gaussian errors. Sakumura et al. (2014) build an ensemble across processing centres.
- Kvas & Mayer-Gürr (2019) add background-model (AOD) uncertainty to the stochastic model, giving up to 25% RMS noise reduction and better formal/empirical error agreement. ITSG-Grace2018 (Kvas et al. 2019) provides full covariances.
- Boergens et al. (2020) give an anisotropic, latitude-dependent spatial covariance model for TWS grids. Loomis et al. (2019) give mascon leakage plus noise.
- Triple collocation gives error estimates without a reference: about 11–12 mm for GRACE over land (van Dijk et al. 2014). See also Hasan & Tarhule (2021) and Gao et al. (2023).
- Ice-sheet continuity across the GRACE/GRACE-FO gap is checked against the mass-budget method (Velicogna et al. 2020), and ML gap-fillers exist for Greenland and Antarctica (Zhang et al. 2022; Shi et al. 2024, 2025). The ocean RL05 standard error is about 1 cm at low and mid latitudes and 1.5–2 cm at high latitudes (Chambers & Bonin 2012).

**Benchmarks and opportunities.** There is no community ML benchmark for gravimetry processing. The closest "ground-truth" resources are:
- the ESA Earth System Model, 6-hourly Stokes coefficients to d/o 180 for 1995–2006 (Dobslaw et al. 2015), and its realistically perturbed de-aliasing error model (Dobslaw et al. 2016), as used for closed-loop tests (Heller-Kaikov et al. 2026; Romeshkani et al. 2025; Boergens et al. 2020);
- reconstruction datasets that can serve as weak targets or baselines (GTWS-MLrec, Yin et al. 2023; the SLR-DL product, Uz et al. 2024).

A sound protocol for a diffusion model would be:
- (i) closed-loop simulation with the ESA ESM plus realistic noise (AOD errors, striping), with known truth;
- (ii) real-data checks against independent geodetic and hydrological observations (GNSS loading, OBP recorders and tide gauges, mass-budget ice data, altimetry);
- (iii) calibration of the predictive spread against formal covariances (ITSG), triple-collocation errors and the Wahr et al. (2006) error model.

Main opportunities for generative models:
- (a) learned priors replacing ad-hoc destriping, DDK filtering and mascon Tikhonov regularisation, through posterior sampling with the gravimetric forward operator;
- (b) stochastic signal separation (hydrology, ocean, GIA, noise) as conditional generation;
- (c) generative ML transplant and denoising at Level-1;
- (d) multi-sensor conditional generation, combining Swarm, SLR, GNSS and altimetry, for gaps and the pre-GRACE era;
- (e) ensemble generation for data assimilation, with calibrated uncertainty.

---

## Papers

### A. Level-1 instrument data (accelerometer, ranging, mission performance)

**behzadpour2021acc_recovery**
Behzadpour, S., Mayer-Gürr, T., Krauss, S. (2021). GRACE Follow-On accelerometer data recovery. *Journal of Geophysical Research: Solid Earth*, 126(5), e2020JB021297. DOI: 10.1029/2020JB021297
The GRACE-D accelerometer degraded one month into the mission, and the SDS transplant (from GRACE-C, with time and attitude corrections plus modelled thruster residuals) leaves C20 and C30 sensitive to how transplant is done. The authors recover GRACE-D ACT1B using non-gravitational force models with drag-model corrections. This reduces noise in monthly fields and improves the C20 and C30 estimates (the improved C30 is a "significant improvement" relative to the SLR values used in TN-14). It is the physics-based baseline any ML transplant must beat.
Status: FULLTEXT_READ (Behzadpour2021_GRACEFO_ACCrecovery.pdf, TU Graz repository copy of the published version)

**harvey2022acc_v04**
Harvey, N., McCullough, C. M., Save, H. (2022). Modeling GRACE-FO accelerometer data for the version 04 release. *Advances in Space Research*, 69(3), 1393–1407. DOI: 10.1016/j.asr.2021.10.056
Describes the operational JPL accelerometer-data modelling (transplant) used for the GRACE-FO Level-1 v04 release. It is the operational reference for the ACC problem.
(Summary based on abstract/metadata only.)
Status: PAYWALLED (an accepted-manuscript version is listed by Unpaywall at ScienceDirect, but automated download was blocked)

**romeshkani2025ml_transplant**
Romeshkani, M., Müller, J., Ebadi, S., Kupriyanov, A., Knabe, A., Fletling, N., Schilling, M. (2025). Machine learning algorithms for transplanting accelerometer observations in future satellite gravimetry missions. arXiv:2508.03522.
Closed-loop LL-SST simulations compare an electrostatic-accelerometer-only setup, cold-atom-interferometry (CAI) setups, and hybrid EA-CAI setups, including transplant of CAI data between satellites. MLP, LSTM and BiLSTM transplant reduces transplant noise and comes close to the ideal dual-hybrid configuration. This is the first systematic ML-for-ACC-transplant study and a direct entry point for sequence or generative models at Level-1.
Status: FULLTEXT_READ (Romeshkani2025_MLaccTransplant.pdf; preprint)

**yang2023noise2noise_acc**
Yang, Z., Zhang, H., Xu, P., Luo, Z. (2023). Unsupervised noise reductions for gravitational reference sensors or accelerometers based on the Noise2Noise method. *Sensors*, 23(13), 6030. DOI: 10.3390/s23136030
Self-supervised (Noise2Noise) denoising of inertial-sensor data. Because only one noisy realisation exists in orbit, odd/even and periodic sub-samplers create the training pairs, and fully connected reconstruction layers are added. On simulated data and on real GRACE-FO and Taiji-1 acceleration data it outperforms conventional smoothing. It is a precedent for self-supervised learning when no clean truth is available, a setting that also suits diffusion models.
Status: FULLTEXT_READ (Yang2023_N2N_accelerometerDenoising.pdf; arXiv:2305.06735 preprint version)

**landerer2020gracefo**
Landerer, F. W., Flechtner, F. M., Save, H., et al. (2020). Extending the global mass change data record: GRACE Follow-On instrument and science data performance. *Geophysical Research Letters*, 47(12), e2020GL088306. DOI: 10.1029/2020GL088306
The mission-performance reference for GRACE-FO. It extends the GRACE record at equivalent precision and sampling, reports instrument and flight-system performance largely improved over GRACE, and confirms the novel LRI works. Cite it for context on the instrument and noise budget (KBR and LRI) and on the ACC transplant. (I read the key points and front matter.)
Status: FULLTEXT_READ (Landerer2020_GRACEFO_performance.pdf; GFZpublic copy)

### B. Level-2 filtering, destriping and signal separation

**swenson2006destriping**
Swenson, S., Wahr, J. (2006). Post-processing removal of correlated errors in GRACE data. *Geophysical Research Letters*, 33(8), L08402. DOI: 10.1029/2005GL025285
The classic empirical destriping filter: a polynomial fit across even and odd coefficients of fixed order above a minimum order. It is the standard baseline for any learned destriping.
(Summary based on abstract/metadata and general knowledge.)
Status: PAYWALLED (publisher OA, automated download blocked)

**kusche2007ddk**
Kusche, J. (2007). Approximate decorrelation and non-isotropic smoothing of time-variable GRACE-type gravity field models. *Journal of Geodesy*, 81(11), 733–749. DOI: 10.1007/s00190-007-0143-3
Introduces the DDK-type decorrelation and anisotropic filter, built from an approximate error covariance and a signal covariance. It reduces stripes more effectively than isotropic Gaussian smoothing while keeping more signal amplitude, and accounts for anisotropic errors and data density increasing towards the poles. Because it is a regularised (Wiener-like) estimator, it is the conceptual precursor of a learned prior.
Status: FULLTEXT_READ (Kusche2007_DDK.pdf)

**piretzidis2018ml_correlated**
Piretzidis, D., Sra, G., Karantaidis, G., Sideris, M. G., Kabirzadeh, H. (2018). Identifying presence of correlated errors using machine learning algorithms for the selective de-correlation of GRACE harmonic coefficients. *Geophysical Journal International*, 215(1), 375–388. DOI: 10.1093/gji/ggy272
Trains classifiers, including a feed-forward NN, to detect which spherical-harmonic orders carry correlated (striping) errors, so that decorrelation is applied only where needed rather than above a fixed minimum order. It is the earliest ML-on-Stokes-coefficients destriping work.
(Summary based on abstract/metadata and a related AGU abstract.)
Status: PAYWALLED (publisher OA, automated download blocked)

**hellerkaikov2023signal_separation_geoai**
Heller-Kaikov, B., Pail, R., Werner, M. (2023). Signal separation in global, temporal gravity data. *Proc. 6th ACM SIGSPATIAL GeoAI Workshop*. DOI: 10.1145/3615886.3627743
Compares a PCA-based method with a conditional GAN (pix2pix-style image-to-image translation) for separating atmosphere/ocean from continental-hydrology signals in synthetic closed-loop gravity data. The two achieve comparable prediction errors, and the choice of gravity functional used in pre-processing strongly affects performance. It is the first generative-model (cGAN) result on gravity signal separation, and a direct precursor to diffusion-based separation.
Status: FULLTEXT_READ (HellerKaikov2023_signalSeparation_GeoAI.pdf, TUM author copy)

**hellerkaikov2026signal_separation**
Heller-Kaikov, B., Pail, R., Werner, M. (2026). Neural network-based framework for signal separation in spatio-temporal gravity data. *Computers & Geosciences*, 207, 106057. DOI: 10.1016/j.cageo.2025.106057
A multi-channel U-Net maps the sum of gravity signals (plus GRACE-type noise) to its components, trained on ESA ESM data. Reported closed-loop relative RMS errors are 19–67%, and separation works for SNR > 1. Follow-up abstracts apply it to real GRACE/-FO Level-2 data as an alternative to de-striping filters. It is the most important Level-2 DL baseline for a diffusion denoiser.
(Summary based on abstract/metadata and EGU abstracts.)
Status: PAYWALLED (publisher OA, automated download blocked)

### C. Mascon estimation and regularisation

**save2016csr_mascons**
Save, H., Bettadpur, S., Tapley, B. D. (2016). High-resolution CSR GRACE RL05 mascons. *JGR Solid Earth*, 121(10), 7547–7569. DOI: 10.1002/2016JB013007
The CSR mascon solution with time-variable Tikhonov regularisation derived from the data themselves. This is the "hand-designed prior" that a learned prior would replace.
(Summary based on abstract/metadata.)
Status: PAYWALLED

**watkins2015jpl_mascons**
Watkins, M. M., Wiese, D. N., Yuan, D.-N., Boening, C., Landerer, F. W. (2015). Improved methods for observing Earth's time variable mass distribution with GRACE using spherical cap mascons. *JGR Solid Earth*, 120(4), 2648–2671. DOI: 10.1002/2014JB011547
The JPL spherical-cap mascons, with a priori geophysical constraints in the inversion. It is the reference for the explicit Bayesian (a priori covariance) mascon regularisation.
(Summary based on abstract/metadata.)
Status: PAYWALLED (publisher OA, automated download blocked)

**loomis2019mascon_regularization**
Loomis, B. D., Luthcke, S. B., Sabaka, T. J. (2019). Regularization and error characterization of GRACE mascons. *Journal of Geodesy*, 93(9), 1381–1398. DOI: 10.1007/s00190-019-01252-y
The GSFC mascons, with time-dependent regularisation matrices built from geographically binned inter-satellite range-acceleration residuals in an iterative scheme. The design is conservative, to minimise the influence of the regularisation. They prove regularised mascons are biased (leakage) and that total uncertainty must combine leakage from the resolution operators with noise variance. This is key for UQ of any learned-prior inversion.
Status: FULLTEXT_READ (Loomis2019_masconRegularization.pdf)

### D. Neural gravity-field representations

**martin2022pinn_earthmoon**
Martin, J., Schaub, H. (2022). Physics-informed neural networks for gravity field modeling of the Earth and Moon. *Celestial Mechanics and Dynamical Astronomy*, 134(2), 13. DOI: 10.1007/s10569-022-10069-5
PINN-GM I/II: an NN potential trained on position and acceleration data with Laplace/conservative-field constraints. For Earth it gives an order-of-magnitude reduction in parameters at comparable accuracy to SH; the gain is smaller for the Moon.
(Summary based on abstract/metadata and secondary sources.)
Status: PAYWALLED

**martin2022pinn_smallbodies**
Martin, J., Schaub, H. (2022). Physics-informed neural networks for gravity field modeling of small bodies. *Celestial Mechanics and Dynamical Astronomy*, 134(5), 46. DOI: 10.1007/s10569-022-10101-8
The PINN gravity model applied to irregular small bodies.
(Summary based on metadata only.)
Status: PAYWALLED

**martin2024pinngm3**
Martin, J., Schaub, H. (2023/2025). The Physics-Informed Neural Network Gravity Model: Generation III. arXiv:2312.10257 (v3, Feb 2025).
PINN-GM-III fixes four failure modes of ML gravity models: feature divergence, low-altitude sample bias, numerical instability and extrapolation error. It also fuses analytic and NN models and proposes six evaluation metrics, tested on homogeneous and heterogeneous-density asteroids with robustness to sparse and noisy data. It is the reference design for a neural-field representation of the geopotential.
Status: FULLTEXT_READ (Martin2024_PINNGM3.pdf; preprint)

**izzo2022geodesynets**
Izzo, D., Gómez, P. (2022). Geodesy of irregular small bodies via neural density fields. *Communications Engineering*, 1, 48. DOI: 10.1038/s44172-022-00050-3
geodesyNets are a 3D differentiable neural density field, trained so that the induced accelerations match observations, giving relative acceleration error below 1% even near the surface. With a shape model they recover heterogeneous interior density. They are a complementary "neural density" alternative to SH, mascon and polyhedral models.
Status: FULLTEXT_READ (Izzo2022_neuralDensityFields.pdf)

### E. Low-degree coefficients (C20, C30, degree-1) and GIA

**loomis2019c20_slr**
Loomis, B. D., Rachlin, K. E., Luthcke, S. B. (2019). Improved Earth oblateness rate reveals increased ice sheet losses and mass-driven sea level rise. *GRL*, 46(12), 6910–6917. DOI: 10.1029/2019GL082929
The GSFC SLR C20 series, the source of the TN-14 replacement values. It shows that the C20 rate affects ice-sheet and sea-level estimates.
(Summary based on abstract/metadata.)
Status: PAYWALLED

**cheng2017c20**
Cheng, M., Ries, J. (2017). The unexpected signal in GRACE estimates of C20. *Journal of Geodesy*, 91(8), 897–914. DOI: 10.1007/s00190-016-0995-5
Diagnoses the anomalous C20 signal in GRACE solutions (linked to thermal and accelerometer effects) and motivates replacing C20 with SLR values.
(Summary based on abstract/metadata.)
Status: PAYWALLED

**swenson2008geocenter**
Swenson, S., Chambers, D., Wahr, J. (2008). Estimating geocenter variations from a combination of GRACE and ocean model output. *JGR*, 113, B08410. DOI: 10.1029/2007JB005338
Derives degree-1 coefficients from GRACE degree ≥2 coefficients plus the oceanic degree-1 component from ocean models, consistent with Level-2 processing, and compares them with SLR and GPS geocentre estimates. It is the standard degree-1 (TN-13) approach.
Status: FULLTEXT_READ (Swenson2008_geocenter.pdf; USF Digital Commons copy)

**sun2016geocenter**
Sun, Y., Riva, R., Ditmar, P. (2016). Optimizing estimates of annual variations and trends in geocenter motion and J2 from a combination of GRACE data and geophysical models. *JGR Solid Earth*, 121(11), 8352–8370. DOI: 10.1002/2016JB013073
Refines the GRACE-plus-ocean-model approach for joint degree-1 and J2 estimation (GIA, self-consistent sea level).
(Summary based on abstract/metadata.)
Status: PAYWALLED (publisher OA / TU Delft repository, automated download blocked)

**love2024gia_surrogate**
Love, R., Milne, G. A., Ajourlou, P., Parang, S., Tarasov, L., Latychev, K. (2024). A fast surrogate model for 3D Earth glacial isostatic adjustment using Tensorflow (v2.8.0) artificial neural networks. *Geoscientific Model Development*, 17(23), 8535–8551. DOI: 10.5194/gmd-17-8535-2024
An ANN surrogate predicts the 3D-minus-1D (spherically symmetric) difference in GIA relative sea level and uplift, so that 3D-Earth parameter space can be explored cheaply. It reproduces model–data misfit patterns for North American RSL data. It is the main ML-for-GIA reference; GIA is a key correction and uncertainty term in GRACE trends.
Status: FULLTEXT_READ (Love2024_GIA_NNsurrogate.pdf)

**zammitmangion2015antarctica**
Zammit-Mangion, A., Rougier, J., Schön, N., Lindgren, F., Bamber, J. (2015). Multivariate spatio-temporal modelling for assessing Antarctica's present-day contribution to sea-level rise. *Environmetrics*, 26(3), 159–177. DOI: 10.1002/env.2323
A Bayesian hierarchical (GMRF/SPDE) model jointly separating ice, firn and GIA from GRACE, altimetry and GPS. It is a principled probabilistic source-separation precedent, and the statistical counterpart to a generative separation model.
(Summary based on abstract/metadata.)
Status: PAYWALLED (publisher OA / Bath repository, automated download blocked)

### F. Complementary sensors: Swarm, SLR, GNSS, altimetry, InSAR

**teixeira2020swarm_gfm**
Teixeira da Encarnação, J., Visser, P., Arnold, D., et al. (2020). Description of the multi-approach gravity field models from Swarm GPS data. *Earth System Science Data*, 12(2), 1385–1417. DOI: 10.5194/essd-12-1385-2020
Combined monthly Swarm GPS gravity fields from four estimation approaches, unconstrained and independent month to month. The signal is restricted to SH degree below 12. With 750 km smoothing they agree with GRACE at about 4 cm EWH over land; the ocean needs about 3000 km smoothing. Kinematic baselines and accelerometer data give little or conditional benefit. It is the main low-resolution conditioning source for gap filling.
Status: FULLTEXT_READ (TeixeiraDaEncarnacao2020_SwarmGravity.pdf)

**luck2018swarm_ocean**
Lück, C., Kusche, J., Rietbroek, R., Löcher, A. (2018). Time-variable gravity fields and ocean mass change from 37 months of kinematic Swarm orbits. *Solid Earth*, 9(2), 323–339. DOI: 10.5194/se-9-323-2018
More than 500 Swarm gravity fields via the integral-equation short-arc approach, for ocean-mass change. Monthly solutions have 4.0 mm RMSE versus GRACE; directly estimating constant, trend, annual and semiannual (CTAS) terms gives 1.7 mm. It quantifies how much Swarm can substitute for GRACE over the oceans.
Status: FULLTEXT_READ (Luck2018_SwarmTVG.pdf)

**richter2021swarm_reconstruction**
Richter, H. M. P., Lück, C., Klos, A., Sideris, M. G., Rangelova, E., Kusche, J. (2021). Reconstructing GRACE-type time-variable gravity from the Swarm satellites. *Scientific Reports*, 11, 1117. DOI: 10.1038/s41598-020-80752-w
Projects Swarm fields onto the leading GRACE spatial modes (EOFs) to bridge the 11-month gap. Noise is greatly reduced, basin RMS errors reach the cm level, basins down to about 1000×1000 km are usable, and Swarm informs at least three leading modes. Map uncertainty is 2–3 cm EWH. It is a linear "learned-basis" baseline that a conditional diffusion model on Swarm input should beat.
Status: FULLTEXT_READ (Richter2021_SwarmReconstruction.pdf)

**uz2022dl_gap**
Uz, M., Atman, K. G., Akyilmaz, O., Shum, C. K., et al. (2022). Bridging the gap between GRACE and GRACE-FO missions with deep learning aided water storage simulations. *Science of the Total Environment*, 830, 154701. DOI: 10.1016/j.scitotenv.2022.154701
Deep learning with GRACE/-FO mascons, Swarm gravimetry, ERA5 and time tags fills the 11-month gap with global TWSA at about 100 km. It is validated with groundwater wells and InSAR subsidence.
(Summary based on abstract/metadata.)
Status: PAYWALLED

**locher2021slr_hybrid**
Löcher, A., Kusche, J. (2021). A hybrid approach for recovering high-resolution temporal gravity fields from satellite laser ranging. *Journal of Geodesy*, 95(1), 6. DOI: 10.1007/s00190-020-01460-x
Leading GRACE EOFs are used as base functions in SLR dynamic orbit determination, with only the lowest degrees estimated separately. This yields GRACE-resolution fields back to the pre-GRACE era. Secular and seasonal GRACE signals are reproduced; for example, Greenland was stable until the late 1990s. The method is limited in small basins with strong interannual variability.
Status: FULLTEXT_READ (Locher2021_SLRhybrid.pdf)

**uz2024slr_dl**
Uz, M., Akyılmaz, O., Shum, C. K., Atman, K. G., Olgun, S., Güneş, Ö. (2024). High-resolution temporal gravity field data products: Monthly mass grids and spherical harmonics from 1994 to 2021. *Scientific Data*, 11, 71. DOI: 10.1038/s41597-023-02887-5
A residual deep convolutional autoencoder (ResDCAE) trained on GRACE/-FO mascons, SLR-only fields (d/o 10) and ERA5 simulates monthly 1° mass anomalies from 1994 to 2021. It is validated with RMSE/NSE, ENSO indices, GMSL, EOP-derived low degrees and in-situ OBP. It is a multi-sensor DL product and a pre-GRACE baseline.
Status: FULLTEXT_READ (Uz2024_SLR_DL_1994_2021.pdf)

**argus2014gps_tws**
Argus, D. F., Fu, Y., Landerer, F. W. (2014). Seasonal variation in total water storage in California inferred from GPS observations of vertical land motion. *GRL*, 41(6), 1971–1980. DOI: 10.1002/2014GL059570
The seminal inversion of GPS vertical loading displacement for TWS, at finer resolution than GRACE.
(Summary based on abstract/metadata.)
Status: PAYWALLED (publisher OA, automated download blocked)

**carlson2022gnss_grace**
Carlson, G., Werth, S., Shirzaei, M. (2022). Joint inversion of GNSS and GRACE for terrestrial water storage change in California. *JGR Solid Earth*, 127(3), e2021JB023135. DOI: 10.1029/2021JB023135
A joint inversion of GNSS vertical displacements and GRACE mascon ΔTWS, accounting for observation errors, gaps and non-hydrological signals. It uses a continuous wavelet transform to invert long- and short-term variations separately. The joint solution is regionally consistent with GRACE but has GNSS-scale detail. It is a classical multi-sensor fusion baseline.
Status: FULLTEXT_READ (Carlson2022_GNSS_GRACE_jointInversion.pdf; VTechWorks copy)

**shen2022lstm_gnss**
Shen, Y., Zheng, W., Yin, W., Xu, A., Zhu, H., Wang, Q., et al. (2022). Improving the inversion accuracy of terrestrial water storage anomaly by combining GNSS and LSTM algorithm and its application in mainland China. *Remote Sensing*, 14(3), 535. DOI: 10.3390/rs14030535
An LSTM-weighted GNSS loading inversion (with IDW and a crustal load model) for unevenly distributed stations in China.
(Summary based on abstract/metadata.)
Status: PAYWALLED (publisher OA, MDPI download blocked)

**wei2025cnn_gnss_loading**
Wei, C., Zhou, M., Du, Z., Han, L., Gao, H. (2025). Assessment of hydrological loading displacement from GNSS and GRACE data using deep learning algorithms. *Scientific Reports*, 15, 6070. DOI: 10.1038/s41598-025-90363-y
A 3D-CNN combines 41 GNSS stations in Yunnan with GRACE TWS to invert hydrological loading displacement. It beats the load Green's-function inversion: mean absolute deviation drops by 79.6% and standard deviation by 31.4% relative to GNSS observations. Annual and semiannual cycles explain over 90% of the variance.
Status: FULLTEXT_READ (Wei2025_3DCNN_GNSS_loading.pdf; Research Square preprint version rs-4917007)

**rietbroek2016sea_level**
Rietbroek, R., Brunnabend, S.-E., Kusche, J., Schröter, J., Dahle, C. (2016). Revisiting the contemporary sea-level budget on global and regional scales. *PNAS*, 113(6), 1504–1509. DOI: 10.1073/pnas.1519132113
A joint inversion of GRACE and altimetry (fingerprints) for mass and steric sea-level contributions. It is the reference for GRACE plus altimetry fusion.
(Summary based on abstract/metadata.)
Status: PAYWALLED (PMC OA, automated download blocked)

**reshi2023insar_grace**
Reshi, A. R., Sandhu, H. A. S., Cherubini, C., Tripathi, A. (2023). Estimating land subsidence and gravimetric anomaly induced by aquifer overexploitation in the Chandigarh Tri-City region, India by coupling remote sensing with a deep learning neural network model. *Water*, 15(6), 1206. DOI: 10.3390/w15061206
An MLP combining PSInSAR (Sentinel-1), GRACE and well data (2017–2022) to downscale groundwater storage and subsidence. It is an example of InSAR plus GRACE plus ML.
(Summary based on abstract/metadata.)
Status: PAYWALLED (publisher OA, MDPI download blocked)

### G. Data assimilation of GRACE and hybrid physics-ML models

**zaitchik2008grace_da**
Zaitchik, B. F., Rodell, M., Reichle, R. H. (2008). Assimilation of GRACE terrestrial water storage data into a land surface model: Results for the Mississippi River Basin. *Journal of Hydrometeorology*, 9(3), 535–548. DOI: 10.1175/2007JHM951.1
The first GRACE DA: an ensemble Kalman smoother into the Catchment LSM, improving groundwater and the vertical and spatial disaggregation of TWS.
(Summary based on abstract/metadata.)
Status: PAYWALLED (AMS free access, automated download blocked)

**kumar2016grace_da_nldas**
Kumar, S. V., Zaitchik, B. F., Peters-Lidard, C. D., et al. (2016). Assimilation of gridded GRACE terrestrial water storage estimates in the North American Land Data Assimilation System. *Journal of Hydrometeorology*, 17(7), 1951–1972. DOI: 10.1175/JHM-D-15-0157.1
EnKF assimilation of gridded GRACE TWS into NLDAS (Noah, CLSM). Groundwater and drought skill improve; effects on soil moisture and streamflow are mixed.
(Summary based on abstract/metadata.)
Status: PAYWALLED (OA per Unpaywall, automated download blocked)

**girotto2017grace_da_india**
Girotto, M., De Lannoy, G. J. M., Reichle, R. H., et al. (2017). Benefits and pitfalls of GRACE data assimilation: A case study of terrestrial water storage depletion in India. *GRL*, 44(9), 4107–4115. DOI: 10.1002/2017GL072994
Shows that GRACE DA cannot correctly attribute trends from processes missing in the model (groundwater pumping in India), producing unrealistic adjustments to other storages. This motivates hybrid or ML model-error components.
(Summary based on abstract/metadata.)
Status: PAYWALLED (PMC OA, automated download blocked)

**eicker2014watergap_da**
Eicker, A., Schumacher, M., Kusche, J., Döll, P., Müller Schmied, H. (2014). Calibration/data assimilation approach for integrating GRACE data into the WaterGAP Global Hydrology Model (WGHM) using an ensemble Kalman filter. *Surveys in Geophysics*, 35(6), 1285–1309. DOI: 10.1007/s10712-014-9309-8
Joint parameter calibration and state assimilation of GRACE (with full error covariances) into WGHM with an EnKF.
(Summary based on abstract/metadata.)
Status: PAYWALLED

**li2019global_grace_da**
Li, B., Rodell, M., Kumar, S., et al. (2019). Global GRACE data assimilation for groundwater and drought monitoring: Advances and challenges. *Water Resources Research*, 55(9), 7564–7586. DOI: 10.1029/2018WR024618
Global GRACE DA into CLSM, evaluated against groundwater wells and drought indicators. It reviews the challenges: scale mismatch, error characterisation, missing human processes.
(Summary based on abstract/metadata.)
Status: PAYWALLED (publisher OA, automated download blocked)

**khaki2017localization**
Khaki, M., Schumacher, M., Forootan, E., Kuhn, M., Awange, J. L., van Dijk, A. I. J. M. (2017). Accounting for spatial correlation errors in the assimilation of GRACE into hydrological models through localization. *Advances in Water Resources*, 108, 99–112. DOI: 10.1016/j.advwatres.2017.07.024
Treats spatially correlated GRACE errors in ensemble DA through localisation and the full error covariance.
(Summary based on abstract/metadata.)
Status: PAYWALLED

**mehrnegar2020dmda**
Mehrnegar, N., Jones, O., Singer, M. B., Schumacher, M., Bates, P., Forootan, E. (2020). Comparing global hydrological models and combining them with GRACE by dynamic model data averaging (DMDA). *Advances in Water Resources*, 138, 103528. DOI: 10.1016/j.advwatres.2020.103528
Bayesian dynamic model averaging of several GHMs constrained by GRACE. It is a probabilistic multi-model fusion baseline.
(Summary based on abstract/metadata.)
Status: PAYWALLED

**vandijk2014reanalysis**
van Dijk, A. I. J. M., Renzullo, L. J., Wada, Y., Tregoning, P. (2014). A global water cycle reanalysis (2003–2012) merging satellite gravimetry and altimetry observations with a hydrological multi-model ensemble. *HESS*, 18(8), 2955–2973. DOI: 10.5194/hess-18-2955-2014
A sequential DA merging GRACE, water-level altimetry and a multi-model ensemble. Errors come from triple collocation: 11–12 mm for GRACE storage over land, 17–28 mm for model storage changes. It is evaluated against discharge, snow and glacier data, and is a key reference for both DA and TC error estimation.
Status: FULLTEXT_READ (vanDijk2014_waterCycleReanalysis.pdf)

**sun2019dl_mismatch**
Sun, A. Y., Scanlon, B. R., Zhang, Z., Walling, D., Bhanja, S. N., Mukherjee, A., Zhong, Z. (2019). Combining physically based modeling and deep learning for fusing GRACE satellite data: Can we learn from mismatch? *Water Resources Research*, 55(2), 1179–1195. DOI: 10.1029/2018WR023333
CNNs (VGG16, U-Net, SegNet-lite) learn the spatiotemporal mismatch between NOAH and GRACE TWSA over India. The corrected TWSA reaches country-average correlation 0.94 and NSE 0.87 (14% and 52% gains), and the learned residual correlates with in-situ groundwater, i.e., it compensates for missing groundwater. It is the first "learned model-error" approach to GRACE fusion.
Status: FULLTEXT_READ (Sun2019_DLfusingGRACE.pdf; arXiv:1902.01933 preprint)

**gou2024selfsupervised_da**
Gou, J., Soja, B. (2024). Global high-resolution total water storage anomalies from self-supervised data assimilation using deep learning algorithms. *Nature Water*, 2(2), 139–150. DOI: 10.1038/s44221-024-00194-w
A self-supervised "DA" network with a new loss function combines WGHM simulations and GRACE(-FO) to produce global 0.5° TWSA. It keeps GRACE's large-scale accuracy while closing the water balance in small basins, and gives uncertainties from deep ensembles (global median 7.3 mm). It is the state-of-the-art DL fusion and the main competitor and baseline for a diffusion approach.
Status: FULLTEXT_READ (Gou2024_selfSupervisedDA_TWS.pdf)

**kraft2022h2m**
Kraft, B., Jung, M., Körner, M., Koirala, S., Reichstein, M. (2022). Towards hybrid modeling of the global hydrological cycle. *HESS*, 26(6), 1579–1614. DOI: 10.5194/hess-26-1579-2022
H2M is a global hybrid model (1°, daily) with snow, soil moisture and groundwater pools and an embedded RNN for uncertain fluxes. It is trained by multi-task learning against GRACE TWS, runoff, ET and SWE, performs on par with four GHMs, and learns physically plausible responses. It is the reference for GRACE-constrained hybrid (differentiable) hydrology.
Status: FULLTEXT_READ (Kraft2022_hybridHydrology.pdf; HESS discussion-paper version hess-2021-211)

**baghirov2025h2mv**
Baghirov, Z., Jung, M., Reichstein, M., Körner, M., Kraft, B. (2025). H2MV (v1.0): global physically constrained deep learning water cycle model with vegetation. *Geoscientific Model Development*, 18(10), 2921–2943. DOI: 10.5194/gmd-18-2921-2025
Extends H2M: an LSTM predicts uncertain parameters including SMmax and fAPAR, optimised jointly against GRACE TWS, fAPAR, SWE, ET and runoff in 10-fold cross-validation. It explicitly analyses equifinality and under-constrained processes.
Status: FULLTEXT_READ (Baghirov2025_H2MV.pdf)

**shen2023differentiable**
Shen, C., Appling, A. P., Gentine, P., et al. (2023). Differentiable modelling to unify machine learning and physical models for geosciences. *Nature Reviews Earth & Environment*, 4(8), 552–567. DOI: 10.1038/s43017-023-00450-9
A perspective on differentiable modelling: connecting physical priors to NNs through gradients. It argues for better interpretability, generalisation and extrapolation than pure ML, especially when data are scarce. It is the framing reference for physics-ML hybrids that could use GRACE as a constraint.
Status: FULLTEXT_READ (Shen2023_differentiableModelling.pdf; arXiv:2301.04027 preprint version)

### H. Ice-sheet and ocean mass

**velicogna2020continuity**
Velicogna, I., Mohajerani, Y., A, G., Landerer, F., Mouginot, J., Noël, B., et al. (2020). Continuity of ice sheet mass loss in Greenland and Antarctica from the GRACE and GRACE Follow-On missions. *GRL*, 47(8), e2020GL087291. DOI: 10.1029/2020GL087291
GRACE and GRACE-FO are consistent across the gap at continental and regional scales, and the gap is filled with mass-budget-method data. Greenland lost 600 Gt in summer 2019. Antarctic mass-loss acceleration paused after 2016 because of a 980 Gt gain in Queen Maud Land. It provides the independent (mass-budget) validation for ML gap filling over the ice sheets.
Status: FULLTEXT_READ (Velicogna2020_iceSheetContinuity.pdf; ORBi repository copy)

**zhang2022greenland_ann**
Zhang, B., Yao, Y., He, Y. (2022). Bridging the data gap between GRACE and GRACE-FO using artificial neural network in Greenland. *Journal of Hydrology*, 608, 127614. DOI: 10.1016/j.jhydrol.2022.127614
Compares MLR, BPNN and DBN for gap filling over Greenland and six sub-regions. The NN is best (RMSE 1.5 cm ice-sheet-wide); sub-regional skill is limited by leakage.
(Summary based on abstract/metadata.)
Status: PAYWALLED

**shi2024greenland_ml**
Shi, Z., Wang, Z., Zhang, B., et al. (2024). Bridging the spatiotemporal ice sheet mass change data gap between GRACE and GRACE-FO in Greenland using machine learning method. *Journal of Hydrology*, 629, 130622. DOI: 10.1016/j.jhydrol.2024.130622
ML (reportedly SVM driven by climate-model outputs) reconstruction of the spatiotemporal Greenland mass-change gap.
(Summary based on abstract/metadata and secondary summary.)
Status: PAYWALLED

**shi2025gris_ais_gap**
Shi, Z., Wang, Z., Zhang, B., Barrand, N. E., et al. (2025). A unified framework for bridging the data gap between GRACE/GRACE-FO for both Greenland and Antarctica. *IEEE Geoscience and Remote Sensing Letters*, 22. DOI: 10.1109/LGRS.2025.3605913
PLSR combined with a sparrow-search-optimised BP network, giving a single gap-filling framework for both ice sheets.
(Summary based on abstract/metadata.)
Status: PAYWALLED

**gou2025obp_downscaling**
Gou, J., Börger, L., Schindelegger, M., Soja, B. (2025). Downscaling GRACE-derived ocean bottom pressure anomalies using self-supervised data fusion. *Journal of Geodesy*, 99(2), 19. DOI: 10.1007/s00190-025-01943-9
A self-supervised CNN downscales GRACE(-FO) OBP to 0.25° without high-resolution truth. Outputs are constrained to GRACE large-scale mass while spatial detail is learned from two ocean reanalyses. It agrees with GRACE at the mm level over basins and with tide gauges better than GRACE at about 80% of 465 stations. It is a key ML ocean-mass reference with an independent-validation protocol.
Status: FULLTEXT_READ (Gou2025_OBPdownscaling.pdf; arXiv:2404.05818 preprint)

**chambers2012obp**
Chambers, D. P., Bonin, J. A. (2012). Evaluation of Release-05 GRACE time-variable gravity coefficients over the ocean. *Ocean Science*, 8(5), 859–868. DOI: 10.5194/os-8-859-2012
Evaluates RL05 OBP against two ocean models: residual variance is 50–60% lower than RL04, and an optimised destriping helps slightly. The standard error is about 1 cm EWH at low and mid latitudes and 1.5–2 cm in polar and subpolar oceans. It is the reference ocean-error level for validation.
Status: FULLTEXT_READ (Chambers2012_OBP_validation.pdf)

### I. Uncertainty quantification and error characterisation

**wahr2006accuracy**
Wahr, J., Swenson, S., Velicogna, I. (2006). Accuracy of GRACE mass estimates. *GRL*, 33(6), L06401. DOI: 10.1029/2005GL025305
A method for GRACE mass error estimates that depend on latitude and smoothing radius. After adjustment the errors are approximately normally distributed, which allows confidence levels. It is the classical calibration target for predictive uncertainty.
Status: FULLTEXT_READ (Wahr2006_accuracyGRACE.pdf; eScholarship copy)

**sakumura2014ensemble**
Sakumura, C., Bettadpur, S., Bruinsma, S. (2014). Ensemble prediction and intercomparison analysis of GRACE time-variable gravity field models. *GRL*, 41(5), 1389–1397. DOI: 10.1002/2013GL058632
Builds an ensemble mean of CSR, GFZ, JPL and GRGS solutions and uses inter-centre spread as an uncertainty measure. It is a "processing-ensemble" baseline for probabilistic outputs.
(Summary based on abstract/metadata.)
Status: PAYWALLED (publisher OA, automated download blocked)

**kvas2019itsg2018**
Kvas, A., Behzadpour, S., Ellmer, M., et al. (2019). ITSG-Grace2018: Overview and evaluation of a new GRACE-only gravity field time series. *JGR Solid Earth*, 124(8), 9332–9344. DOI: 10.1029/2019JB017415
The ITSG monthly and daily (Kalman) solutions, with full variance-covariance matrices. It is the main source of realistic formal covariances.
(Summary based on abstract/metadata.)
Status: PAYWALLED (publisher OA, automated download blocked)

**kvas2019background_uncertainty**
Kvas, A., Mayer-Gürr, T. (2019). GRACE gravity field recovery with background model uncertainties. *Journal of Geodesy*, 93(12), 2543–2552. DOI: 10.1007/s00190-019-01314-1
Adds AOD background-model uncertainty, as an autoregressive model derived from the ESA ESM, into the least-squares stochastic model. Noise drops at all scales (up to 25% RMS for 250–300 km Gaussian filtering) and formal and empirical errors agree better.
Status: FULLTEXT_READ (Kvas2019_backgroundModelUncertainty.pdf)

**boergens2020covariance**
Boergens, E., Dobslaw, H., Dill, R., Thomas, M., Dahle, C., Murböck, M., Flechtner, F. (2020). Modelling spatial covariances for terrestrial water storage variations verified with synthetic GRACE-FO data. *GEM – Int. J. Geomathematics*, 11, 24. DOI: 10.1007/s13137-020-00160-0
An analytic anisotropic, latitude-dependent covariance model for gridded TWS (a direction-dependent Bessel function with Legendre-polynomial latitude variation), verified on the sphere by Monte Carlo. It is validated against true errors from 5 years of simulated GRACE-FO data. It is a ready-made error covariance for likelihood terms or for evaluating spread.
Status: FULLTEXT_READ (Boergens2020_spatialCovariance.pdf)

**hasan2021triple_collocation**
Hasan, E., Tarhule, A. (2021). Comparison of decadal water storage trends from common GRACE releases (RL05, RL06) using spatial diagnostics and a modified triple collocation approach. *Journal of Hydrology X*, 13, 100108. DOI: 10.1016/j.hydroa.2021.100108
Modified triple collocation with geographically weighted regression scaling across 11 GRACE products, for grid-scale trend uncertainty.
(Summary based on abstract/metadata.)
Status: PAYWALLED (publisher OA, automated download blocked)

**gao2023fusion_uq**
Gao, S., Hao, W., Fan, Y., Li, F., Wang, J. (2023). A multi-source GRACE fusion solution via uncertainty quantification of GRACE-derived terrestrial water storage (TWS) change. *JGR Solid Earth*, 128(11), e2023JB026908. DOI: 10.1029/2023JB026908
Uncertainty-weighted fusion of multiple GRACE solutions, using three-cornered-hat or TC-type UQ.
(Summary based on metadata/title only; method not verified.)
Status: PAYWALLED (publisher OA, automated download blocked)

### J. Benchmark and synthetic datasets for ML evaluation

**dobslaw2015esa_esm**
Dobslaw, H., Bergmann-Wolf, I., Dill, R., Forootan, E., Klemann, V., Kusche, J., Sasgen, I. (2015). The updated ESA Earth System Model for future gravity mission simulation studies. *Journal of Geodesy*, 89(5), 505–513. DOI: 10.1007/s00190-014-0787-8
A synthetic time-variable gravity model with realistic atmosphere, ocean, TWS, ice and solid-Earth components: Stokes coefficients to d/o 180, 6-hourly, 1995–2006. It includes low-degree and eustatic variability. It is the de facto "ground truth" for closed-loop ML experiments (used by Heller-Kaikov et al., Kvas & Mayer-Gürr, and Boergens et al.).
Status: FULLTEXT_READ (Dobslaw2015_ESA_ESM.pdf; GFZpublic copy)

**dobslaw2016dealiasing_errors**
Dobslaw, H., Bergmann-Wolf, I., Forootan, E., Dahle, C., Mayer-Gürr, T., Kusche, J., Flechtner, F. (2016). Modeling of present-day atmosphere and ocean non-tidal de-aliasing errors for future gravity mission simulations. *Journal of Geodesy*, 90(5), 423–436. DOI: 10.1007/s00190-015-0884-3
A realistically perturbed de-aliasing (AOD) error model consistent with the ESA ESM. It covers large-scale errors in the 10–30, 3–10 and 1–3 day bands, the S1 tide and sub-diurnal periods; small-scale errors; and errors from missing processes. Error magnitudes come from a 4-model ensemble and are confirmed by variance-component estimation up to d/o 40. It provides the realistic noise for simulation benchmarks.
Status: FULLTEXT_READ (Dobslaw2016_AODerrors.pdf; GFZpublic copy)

**yin2023gtws_mlrec**
Yin, J., Slater, L. J., Khouakhi, A., et al. (2023). GTWS-MLrec: global terrestrial water storage reconstruction by machine learning from 1940 to present. *Earth System Science Data*, 15(12), 5597–5615. DOI: 10.5194/essd-15-5597-2023
An ML reconstruction (0.25°, 1940–2022) based on the JPL, CSR and GSFC mascons, evaluated against the land–ocean mass budget, 341 basin water budgets and 10,168 streamflow gauges. Its multi-target evaluation protocol is reusable. It is listed here as an ML-ready dataset with an evaluation protocol; the other agent covers reconstruction methods in detail.
Status: FULLTEXT_READ (Yin2023_GTWSMLrec.pdf)

---

## Paywalled

Papers without a local full text (31). Entries marked † are open access at the publisher, PMC or a repository; bot protection blocked automated download, so they can be fetched manually in a browser.

| key | title | DOI |
|---|---|---|
| harvey2022acc_v04 | Modeling GRACE-FO accelerometer data for the version 04 release † (accepted manuscript) | 10.1016/j.asr.2021.10.056 |
| swenson2006destriping | Post-processing removal of correlated errors in GRACE data † | 10.1029/2005GL025285 |
| piretzidis2018ml_correlated | Identifying presence of correlated errors using machine learning algorithms for the selective de-correlation of GRACE harmonic coefficients † | 10.1093/gji/ggy272 |
| hellerkaikov2026signal_separation | Neural network-based framework for signal separation in spatio-temporal gravity data † | 10.1016/j.cageo.2025.106057 |
| save2016csr_mascons | High-resolution CSR GRACE RL05 mascons | 10.1002/2016JB013007 |
| watkins2015jpl_mascons | Improved methods for observing Earth's time variable mass distribution with GRACE using spherical cap mascons † | 10.1002/2014JB011547 |
| martin2022pinn_earthmoon | Physics-informed neural networks for gravity field modeling of the Earth and Moon | 10.1007/s10569-022-10069-5 |
| martin2022pinn_smallbodies | Physics-informed neural networks for gravity field modeling of small bodies | 10.1007/s10569-022-10101-8 |
| loomis2019c20_slr | Improved Earth oblateness rate reveals increased ice sheet losses and mass-driven sea level rise | 10.1029/2019GL082929 |
| cheng2017c20 | The unexpected signal in GRACE estimates of C20 | 10.1007/s00190-016-0995-5 |
| sun2016geocenter | Optimizing estimates of annual variations and trends in geocenter motion and J2 from a combination of GRACE data and geophysical models † | 10.1002/2016JB013073 |
| zammitmangion2015antarctica | Multivariate spatio-temporal modelling for assessing Antarctica's present-day contribution to sea-level rise † | 10.1002/env.2323 |
| uz2022dl_gap | Bridging the gap between GRACE and GRACE-FO missions with deep learning aided water storage simulations † (accepted manuscript) | 10.1016/j.scitotenv.2022.154701 |
| argus2014gps_tws | Seasonal variation in total water storage in California inferred from GPS observations of vertical land motion † | 10.1002/2014GL059570 |
| shen2022lstm_gnss | Improving the inversion accuracy of TWSA by combining GNSS and LSTM algorithm … mainland China † | 10.3390/rs14030535 |
| rietbroek2016sea_level | Revisiting the contemporary sea-level budget on global and regional scales † | 10.1073/pnas.1519132113 |
| reshi2023insar_grace | Estimating land subsidence and gravimetric anomaly … coupling remote sensing with a deep learning neural network model † | 10.3390/w15061206 |
| zaitchik2008grace_da | Assimilation of GRACE terrestrial water storage data into a land surface model: Results for the Mississippi River Basin † | 10.1175/2007JHM951.1 |
| kumar2016grace_da_nldas | Assimilation of gridded GRACE terrestrial water storage estimates in the North American Land Data Assimilation System † | 10.1175/JHM-D-15-0157.1 |
| girotto2017grace_da_india | Benefits and pitfalls of GRACE data assimilation: A case study of terrestrial water storage depletion in India † | 10.1002/2017GL072994 |
| eicker2014watergap_da | Calibration/data assimilation approach for integrating GRACE data into the WaterGAP Global Hydrology Model (WGHM) using an ensemble Kalman filter | 10.1007/s10712-014-9309-8 |
| li2019global_grace_da | Global GRACE data assimilation for groundwater and drought monitoring: Advances and challenges † | 10.1029/2018WR024618 |
| khaki2017localization | Accounting for spatial correlation errors in the assimilation of GRACE into hydrological models through localization † | 10.1016/j.advwatres.2017.07.024 |
| mehrnegar2020dmda | Comparing global hydrological models and combining them with GRACE by dynamic model data averaging (DMDA) † | 10.1016/j.advwatres.2020.103528 |
| zhang2022greenland_ann | Bridging the data gap between GRACE and GRACE-FO using artificial neural network in Greenland | 10.1016/j.jhydrol.2022.127614 |
| shi2024greenland_ml | Bridging the spatiotemporal ice sheet mass change data gap between GRACE and GRACE-FO in Greenland using machine learning method † (accepted manuscript) | 10.1016/j.jhydrol.2024.130622 |
| shi2025gris_ais_gap | A unified framework for bridging the data gap between GRACE/GRACE-FO for both Greenland and Antarctica | 10.1109/LGRS.2025.3605913 |
| sakumura2014ensemble | Ensemble prediction and intercomparison analysis of GRACE time-variable gravity field models † | 10.1002/2013GL058632 |
| kvas2019itsg2018 | ITSG-Grace2018: Overview and evaluation of a new GRACE-only gravity field time series † | 10.1029/2019JB017415 |
| hasan2021triple_collocation | Comparison of decadal water storage trends from common GRACE releases (RL05, RL06) using spatial diagnostics and a modified triple collocation approach † | 10.1016/j.hydroa.2021.100108 |
| gao2023fusion_uq | A multi-source GRACE fusion solution via uncertainty quantification of GRACE-derived TWS change † | 10.1029/2023JB026908 |

## Search notes and gaps (for the survey author)

- No peer-reviewed ML method was found for: (i) KBR/LRI ranging-noise modelling; (ii) learned or ML regularisation priors in mascon inversion; (iii) C20 or degree-1 estimation; (iv) direct GIA estimation from GRACE with ML (only forward-model surrogates exist). Searches surfaced only EGU abstracts and theses (an EGU23 MLP/LSTM GRACE-B accelerometer predictor; TU Delft MSc theses on Swarm-to-GRACE neural networks; a Stuttgart thesis on per-coefficient ANNs). These are not cited here. They do support the claim that these are open gaps.
- An EGU26 abstract by Heller-Kaikov et al. applies the U-Net separator to real Level-2 data as a destriping alternative. Watch for the journal version.
- A 2026 EGU abstract (Zhu & Wang) reports a ResUnet self-supervised GRACE DA into Noah-MP. It is not yet peer-reviewed and is not included.
- Uz et al. 2024 (Scientific Data) and Gou & Soja 2024 overlap with the gridded-TWSA notes; they are kept here because of their multi-sensor (SLR) and DA framing.
