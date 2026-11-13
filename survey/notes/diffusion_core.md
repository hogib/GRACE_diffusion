# Diffusion / score-based generative models: foundations and methods relevant to GRACE/GRACE-FO

Sub-topic notes for the survey "Enhancing GRACE/GRACE-FO satellite gravimetry using deep learning and diffusion models".
BibTeX: `survey/notes/diffusion_core.bib` (61 entries). PDFs: `literature/diffusion_core/`.
Metadata was checked against the arXiv API and Crossref on 2026-11-13.

**Status legend (full-text rule).**
- FULLTEXT_READ: the PDF is in `literature/diffusion_core/` and was extracted with `pdftotext` (text in the scratchpad `dc_txt/`). The abstract and introduction were read, and each specific claim in the summary below (method mechanics, step counts, tasks, metrics, datasets) was checked against the text by targeted search. This is not a line-by-line read of every appendix.
- PAYWALLED: no legal open-access full text; the summary is based on the abstract/metadata only, and the user must fetch the PDF via the DOI.
- No paper is FULLTEXT_AVAILABLE_UNREAD or ABSTRACT_ONLY.

Counts: 61 papers; 56 FULLTEXT_READ, 5 PAYWALLED. Venues for arXiv-only items are the published venues as far as known; items with no peer-reviewed venue are cited as arXiv preprints.

## Summary of the sub-field

**Core math.** A diffusion model defines a forward noising process that gradually maps data $x_0\sim p_{\text{data}}$ to a tractable prior. In discrete time (Sohl-Dickstein 2015; DDPM, Ho 2020) the process is $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$. In continuous time (Song et al. 2021, score SDE) it is an Itô SDE $dx = f(x,t)\,dt + g(t)\,dw$ (variance-preserving or variance-exploding). Anderson (1982) gives the reverse-time SDE $dx = [f - g^2\nabla_x\log p_t(x)]\,dt + g\,d\bar w$. There is also a deterministic "probability-flow" ODE with the same marginals, $dx/dt = f - \tfrac12 g^2 \nabla_x \log p_t(x)$, which allows exact likelihoods and deterministic encoding. Sampling only needs the *score* $\nabla_x\log p_t$. The score is learned by denoising score matching (Hyvärinen 2005; Vincent 2011): a network $s_\theta(x_t,t)$, or equivalently a noise predictor $\epsilon_\theta$ or denoiser $D_\theta$ with $s_\theta=(D_\theta(x_t)-x_t)/\sigma_t^2$, is trained by MSE regression on noisy copies of the data. Song & Ermon (2019, NCSN) and DDPM are the two discretisations that the SDE view unifies. Karras et al. (EDM, 2022) separates the design choices: noise schedule, preconditioning, loss weighting, and the sampler (Heun ODE plus optional stochastic "churn"). The EDM parameterisation is now the default starting point for scientific applications, including GenCast and CorrDiff.

**Generalisations.** Flow matching (Lipman 2023), rectified flow (Liu 2023) and stochastic interpolants (Albergo 2023) regress a velocity field along prescribed paths between two distributions. Diffusion is one special case of these. The general form permits *non-Gaussian source distributions*, such as a coarse or filtered field. Schrödinger bridges (De Bortoli 2021; I2SB, Liu 2023), mean-reverting SDEs (IR-SDE), ResShift, InDI and Cold Diffusion build the process directly between a degraded observation and the clean target. This fits GRACE well: a filtered/truncated GRACE field and a "true" high-resolution TWS field are naturally paired endpoints. Inverse heat dissipation (Rissanen 2023) and Cold Diffusion use deterministic blurring as the corruption. That is close to the Gaussian/DDK smoothing and spherical-harmonic truncation applied to GRACE solutions.

**Conditioning strategies.** (i) *Amortised/conditional training*: train $s_\theta(x_t,t,y)$ on pairs, with the condition $y$ concatenated or cross-attended. Examples are SR3, Palette, Batzolis 2021, CSDI, CorrDiff and LDM. (ii) *Guidance*: classifier guidance (Dhariwal & Nichol 2021) adds $\nabla\log p(y\mid x_t)$, and classifier-free guidance (Ho & Salimans) mixes conditional and unconditional scores. (iii) *Training-free posterior sampling with an unconditional prior*, where Bayes' rule gives $\nabla\log p_t(x\mid y)=\nabla\log p_t(x)+\nabla\log p_t(y\mid x)$. Methods differ in how they approximate the intractable likelihood term:
- projection/replacement onto measurement-consistent sets (Song 2021/2022, RePaint, DDNM);
- SVD-based spectral treatment of a linear operator with noise (SNIPS, DDRM);
- Tweedie-mean likelihood gradients (MCG, DPS), which also handle nonlinear operators;
- Gaussian pseudo-inverse guidance (ΠGDM);
- half-quadratic splitting / plug-and-play (DiffPIR);
- variational regularisation (RED-diff);
- score-based data assimilation over whole trajectories (SDA, Rozet & Louppe 2023).

Daras et al. (2024) survey this family.

**Mapping to GRACE tasks.**
- *Gap filling* (the 2017–2018 GRACE/GRACE-FO gap, missing months): this is temporal inpainting. Relevant methods are RePaint-style mask conditioning, CSDI (conditional time-series imputation) and video diffusion over a (time × lat × lon) cube. SDA can condition a spatiotemporal prior on sparse observations at arbitrary times.
- *Downscaling / super-resolution* (300–500 km effective resolution to about 0.25–0.5°): use SR3/Palette-style conditional models or, more efficiently, residual approaches. CorrDiff (regression mean + diffusion residual) and ResShift/I2SB/InDI start sampling from the coarse field rather than from pure noise.
- *Destriping/denoising*: this is a linear inverse problem $y = A x + n$. Here $A$ is spherical-harmonic truncation to degree ~60–96 plus filtering, and $n$ is correlated, north-south striped noise with known (or estimable) covariance from the Level-2 variance-covariance matrices. DDRM/ΠGDM/DPS/DiffPIR let a single learned prior of realistic TWS fields (e.g., from hydrological models or high-resolution reanalyses) be paired with the *actual* GRACE observation operator and noise covariance at inference time, with no retraining per filter. DDRM's SVD formulation maps cleanly onto the spherical-harmonic basis, because truncation is diagonal there.
- *Uncertainty quantification*: posterior samples yield ensembles. These must be checked for calibration with proper scoring rules (CRPS; Gneiting & Raftery 2007; Hersbach 2000), spread-skill ratios (Fortin 2014) and rank histograms. GenCast reports all three; DYffusion and CSDI report CRPS (DYffusion also spread-skill ratio). Conffusion and BayesDiff offer pixelwise interval/variance estimates. Daras et al. (2024) note that most training-free solvers (DPS, ΠGDM etc.) rely on tractable approximations of the true posterior, unlike asymptotically exact (e.g., SMC) methods. Their ensembles are therefore not guaranteed to be calibrated, and calibration must be assessed explicitly.

**Global/spherical geometry.** GRACE fields live on $S^2$ and are natively represented in spherical harmonics. Options include:
- equirectangular grids with standard U-Nets: simple, but polar distortion and seams at the dateline;
- spherical CNNs (Cohen 2018);
- spherical Fourier neural operators (SFNO, Bonev 2023): convolution in the SH domain, a natural fit for SH-truncated GRACE data;
- graph/mesh networks on icosahedral meshes, as in GenCast;
- Riemannian diffusion (De Bortoli 2022; Huang 2022) and Manifold Diffusion Fields (Elhag 2024), which generate *functions* on manifolds.

Riemannian score models diffuse *points on* the sphere, not *fields over* it, so for gridded TWS the relevant tools are field-valued (SFNO/mesh denoisers, MDF), not point-valued ones. Mascon products (JPL/CSR/GSFC) are another option: they are already gridded with regularisation, and the choice of input representation (SH coefficients vs mascons) should be explicit.

**Practical concerns.**
- *Sampling cost*: naive DDPM needs 1000 steps. DDIM, DPM-Solver, the EDM Heun sampler (~18–50 steps), progressive distillation and consistency models cut this to 1–50 network evaluations, and latent diffusion (Rombach 2022) moves sampling into a compressed space. For GRACE, monthly global grids at 0.25° are about 720×1440, which is cheap compared with imagery. Large posterior ensembles for UQ multiply the cost, though, so few-step samplers matter.
- *Small data*: there are only ~250 monthly GRACE/GRACE-FO fields (2002–present). Priors therefore have to be pre-trained on long hydrological-model or reanalysis TWS simulations, or on synthetic closed-loop data, and adapted via training-free posterior sampling. A purely supervised conditional model trained on 250 samples risks memorisation.
- *Physical constraints*: mass conservation (global mean, ocean/land leakage) can be enforced by projection steps, as in DDNM and RePaint's replacement.
- *Validation*: closed-loop simulations, comparison to independent in-situ data (wells, GNSS loading, altimetry), and probabilistic scores (CRPS, spread-skill, PIT/rank histograms).

**Spatiotemporal models.** Video diffusion (Ho 2022) uses factorised space-time attention. DYffusion couples a learned interpolator with the diffusion time axis for forecasting. PDE-Refiner uses diffusion-style refinement to recover high-frequency spectral content in long rollouts, which is related to recovering spectral power lost to GRACE filtering. GenCast (Nature 2025), CorrDiff (CEE 2025) and SEEDS (Sci. Adv. 2024) are mature geoscience exemplars showing calibrated diffusion ensembles on global/regional grids. They are the closest methodological templates for a GRACE diffusion study.

## Papers

### A. Core foundations

**sohldickstein2015deep** — J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics," ICML 2015 (PMLR 37). arXiv:1503.03585.
The original diffusion probabilistic model: a slow forward Markov diffusion that destroys structure, and a learned reverse process. It already showed inpainting via conditioning, the conceptual root of diffusion-based gap filling.
Status: FULLTEXT_READ (SohlDickstein2015_nonequilibriumThermodynamics.pdf)

**ho2020ddpm** — J. Ho, A. Jain, P. Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020. arXiv:2006.11239.
Introduces the simple ε-prediction loss and its link to denoising score matching / Langevin dynamics, giving the first high-quality diffusion image synthesis. It is the baseline formulation most geoscience diffusion papers start from.
Status: FULLTEXT_READ (Ho2020_DDPM.pdf)

**song2019ncsn** — Y. Song, S. Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution," NeurIPS 2019. arXiv:1907.05600.
Noise-conditional score networks trained with multi-scale denoising score matching and sampled with annealed Langevin dynamics. The score-based view underlies posterior sampling for inverse problems.
Status: FULLTEXT_READ (Song2019_NCSN.pdf)

**song2021scoresde** — Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole, "Score-Based Generative Modeling through Stochastic Differential Equations," ICLR 2021. arXiv:2011.13456.
Unifies DDPM and NCSN as VP/VE SDEs and introduces the reverse SDE, the probability-flow ODE, predictor-corrector samplers, and controllable generation (inpainting, conditional sampling). This is the central theoretical reference for the survey.
Status: FULLTEXT_READ (Song2021_scoreSDE.pdf)

**hyvarinen2005score** — A. Hyvärinen, "Estimation of Non-Normalized Statistical Models by Score Matching," JMLR 6:695–709, 2005. (No DOI; open access at jmlr.org.)
Introduces score matching, the objective for learning $\nabla\log p$ without a normalising constant.
Status: FULLTEXT_READ (Hyvarinen2005_scoreMatching.pdf)

**vincent2011connection** — P. Vincent, "A Connection Between Score Matching and Denoising Autoencoders," Neural Computation 23(7):1661–1674, 2011. DOI:10.1162/NECO_a_00142.
Shows that denoising equals score matching on a noise-smoothed density (denoising score matching), the training objective of all modern diffusion models.
Status: PAYWALLED (summary based on abstract/metadata only; not read)

**anderson1982reverse** — B. D. O. Anderson, "Reverse-Time Diffusion Equation Models," Stochastic Processes and their Applications 12(3):313–326, 1982. DOI:10.1016/0304-4149(82)90051-5.
The classical result that the time reversal of an Itô diffusion is another diffusion whose drift involves the score. This is the mathematical basis for reverse-SDE sampling.
Status: PAYWALLED (summary based on abstract/metadata only; not read)

**song2021ddim** — J. Song, C. Meng, S. Ermon, "Denoising Diffusion Implicit Models," ICLR 2021. arXiv:2010.02502.
A non-Markovian deterministic sampler that reuses DDPM-trained networks with 10–50× fewer steps and gives a near-invertible latent encoding. Used by many inverse-problem solvers (DDRM, ΠGDM, DDNM).
Status: FULLTEXT_READ (Song2021_DDIM.pdf)

**nichol2021improved** — A. Q. Nichol, P. Dhariwal, "Improved Denoising Diffusion Probabilistic Models," ICML 2021 (PMLR 139). arXiv:2102.09672.
Adds learned variances, a cosine noise schedule and a hybrid loss, improving likelihoods and enabling fewer sampling steps. The cosine schedule is commonly used for small-data scientific fields.
Status: FULLTEXT_READ (Nichol2021_improvedDDPM.pdf)

**karras2022edm** — T. Karras, M. Aittala, T. Aila, S. Laine, "Elucidating the Design Space of Diffusion-Based Generative Models," NeurIPS 2022. arXiv:2206.00364.
Separates preconditioning, noise schedule, loss weighting and samplers, and proposes a 2nd-order Heun sampler that reaches state of the art in ~35 evaluations. The EDM recipe is the backbone of GenCast and CorrDiff and is a recommended default for GRACE models.
Status: FULLTEXT_READ (Karras2022_EDM.pdf)

**rombach2022ldm** — R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022, pp. 10674–10685. DOI:10.1109/CVPR52688.2022.01042; arXiv:2112.10752.
Runs diffusion in the latent space of a pretrained autoencoder, with cross-attention conditioning, cutting compute sharply. Relevant for long multi-decadal spatiotemporal TWS cubes or high-resolution downscaling, where pixel-space diffusion is costly.
Status: FULLTEXT_READ (Rombach2022_latentDiffusion.pdf)

**dhariwal2021beat** — P. Dhariwal, A. Nichol, "Diffusion Models Beat GANs on Image Synthesis," NeurIPS 2021. arXiv:2105.05233.
Architecture improvements (the ADM U-Net) and *classifier guidance*, which steers sampling with gradients of an auxiliary classifier. Classifier guidance is the conceptual template for likelihood-gradient guidance in DPS.
Status: FULLTEXT_READ (Dhariwal2021_diffusionBeatGANs.pdf)

**ho2022cfg** — J. Ho, T. Salimans, "Classifier-Free Diffusion Guidance," NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications; arXiv:2207.12598 (2022).
Trains conditional and unconditional models jointly (condition dropout) and extrapolates between their scores to trade diversity for fidelity. Useful when conditioning on auxiliary predictors (precipitation, GNSS, altimetry), with a knob for adherence to them. Strong guidance can harm calibration.
Status: FULLTEXT_READ (Ho2022_classifierFreeGuidance.pdf)

**batzolis2021conditional** — G. Batzolis, J. Stanczuk, C.-B. Schönlieb, C. Etmann, "Conditional Image Generation with Score-Based Diffusion Models," arXiv:2111.13606, 2021 (preprint).
Analyses conditional score estimators (conditional DSM, multi-speed diffusion) and proves consistency of the conditional denoising estimator, applied to SR, inpainting and edge-to-image. It is the theoretical justification for supervised conditional training on paired GRACE/target data.
Status: FULLTEXT_READ (Batzolis2021_conditionalScore.pdf)

### B. Fast sampling and distillation

**salimans2022progressive** — T. Salimans, J. Ho, "Progressive Distillation for Fast Sampling of Diffusion Models," ICLR 2022. arXiv:2202.00512.
Repeatedly halves the number of sampling steps by distilling a deterministic sampler into a student, reaching 4–8 steps. It also introduces v-prediction. Relevant for producing large UQ ensembles cheaply.
Status: FULLTEXT_READ (Salimans2022_progressiveDistillation.pdf)

**lu2022dpmsolver** — C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, J. Zhu, "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps," NeurIPS 2022. arXiv:2206.00927.
A dedicated high-order solver that exploits the semi-linear structure of the probability-flow ODE, giving high-quality samples in 10–20 steps without retraining.
Status: FULLTEXT_READ (Lu2022_DPMSolver.pdf)

**song2023consistency** — Y. Song, P. Dhariwal, M. Chen, I. Sutskever, "Consistency Models," ICML 2023 (PMLR 202). arXiv:2303.01469.
Learns a map from any point on a probability-flow ODE trajectory to its origin, either by distillation or by direct training. Allows one-step generation and zero-shot editing (inpainting, SR). A candidate for operational near-real-time GRACE products.
Status: FULLTEXT_READ (Song2023_consistencyModels.pdf)

### C. Flow matching, interpolants and bridges

**lipman2023flow** — Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le, "Flow Matching for Generative Modeling," ICLR 2023. arXiv:2210.02747.
Simulation-free regression of continuous normalising flow vector fields along conditional probability paths. Diffusion paths and optimal-transport (straight) paths are special cases. Gives faster and more stable training and sampling, and is increasingly used in weather/climate emulators.
Status: FULLTEXT_READ (Lipman2023_flowMatching.pdf)

**liu2023rectified** — X. Liu, C. Gong, Q. Liu, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow," ICLR 2023. arXiv:2209.03003.
Learns straight-line ODE transport between two arbitrary distributions, with "reflow" to straighten paths so that very few steps suffice, plus optional distillation into a one-step model. Supports domain transfer, e.g. a filtered GRACE field mapped to a high-resolution field.
Status: FULLTEXT_READ (Liu2023_rectifiedFlow.pdf)

**albergo2023interpolants** — M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden, "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions," arXiv:2303.08797, 2023 (preprint).
A general framework that bridges any two densities in finite time with stochastic interpolants, unifying score-based diffusion and flow matching. It covers both ODE and SDE generative models with tunable noise. Provides theory for paired coarse-to-fine mappings.
Status: FULLTEXT_READ (Albergo2023_stochasticInterpolants.pdf)

**debortoli2021dsb** — V. De Bortoli, J. Thornton, J. Heng, A. Doucet, "Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling," NeurIPS 2021. arXiv:2106.01357.
Solves the Schrödinger bridge problem with iterative proportional fitting and score-based networks, giving entropic-OT transport between arbitrary distributions in fewer steps.
Status: FULLTEXT_READ (DeBortoli2021_diffusionSchrodingerBridge.pdf)

**liu2023i2sb** — G.-H. Liu, A. Vahdat, D.-A. Huang, E. A. Theodorou, W. Nie, A. Anandkumar, "I²SB: Image-to-Image Schrödinger Bridge," ICML 2023 (PMLR 202). arXiv:2302.05872.
A tractable Schrödinger bridge between paired degraded and clean images, starting generation from the degraded input rather than noise. More efficient and interpretable than conditional diffusion for restoration (SR, deblurring, inpainting, JPEG). A natural template for "filtered GRACE to destriped/high-resolution" mapping.
Status: FULLTEXT_READ (Liu2023_I2SB.pdf)

### D. Image restoration with diffusion (SR, inpainting, editing, residual)

**saharia2023sr3** — C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, M. Norouzi, "Image Super-Resolution via Iterative Refinement," IEEE TPAMI 45(4):4713–4726, 2023. DOI:10.1109/TPAMI.2022.3204461; arXiv:2104.07636.
SR3: a conditional DDPM with the low-resolution input concatenated to the noisy target, giving 4×–8× SR and cascaded generation. This is the direct template for diffusion-based GRACE downscaling.
Status: FULLTEXT_READ (Saharia2022_SR3.pdf)

**saharia2022palette** — C. Saharia, W. Chan, H. Chang, C. Lee, J. Ho, T. Salimans, D. Fleet, M. Norouzi, "Palette: Image-to-Image Diffusion Models," ACM SIGGRAPH 2022 Conf. Proc., pp. 1–10. DOI:10.1145/3528233.3530757; arXiv:2111.05826.
A single general conditional-diffusion framework for colourisation, inpainting, uncropping and JPEG restoration, with a study of L1 vs L2 losses and sample diversity. Supports a multi-task GRACE model (gap filling plus destriping).
Status: FULLTEXT_READ (Saharia2022_Palette.pdf)

**lugmayr2022repaint** — A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, L. Van Gool, "RePaint: Inpainting using Denoising Diffusion Probabilistic Models," CVPR 2022, pp. 11451–11461. DOI:10.1109/CVPR52688.2022.01117; arXiv:2201.09865.
Training-free inpainting with an unconditional DDPM: known pixels are replaced with forward-noised observations at each step, and "resampling" harmonises the result. Maps directly onto filling missing months or regions given a TWS prior.
Status: FULLTEXT_READ (Lugmayr2022_RePaint.pdf)

**meng2022sdedit** — C. Meng, Y. He, Y. Song, J. Song, J. Wu, J.-Y. Zhu, S. Ermon, "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations," ICLR 2022. arXiv:2108.01073.
Perturbs a guide input with noise to an intermediate time and denoises it with a pretrained SDE prior, trading realism against faithfulness. Can project model-simulated or filtered TWS onto the manifold of realistic fields.
Status: FULLTEXT_READ (Meng2022_SDEdit.pdf)

**yue2023resshift** — Z. Yue, J. Wang, C. C. Loy, "ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting," NeurIPS 2023. arXiv:2307.12348.
A Markov chain that shifts the residual between HR and LR images, so sampling starts near the LR input and needs only ~15 steps. Efficient residual SR suited to GRACE downscaling.
Status: FULLTEXT_READ (Yue2023_ResShift.pdf)

**luo2023irsde** — Z. Luo, F. K. Gustafsson, Z. Zhao, J. Sjölund, T. B. Schön, "Image Restoration with Mean-Reverting Stochastic Differential Equations," ICML 2023 (PMLR 202). arXiv:2301.11699.
An Ornstein–Uhlenbeck-type SDE that diffuses a clean image toward a noisy version of its degraded counterpart, with a closed-form score and a max-likelihood training loss. A general restoration approach (denoising, deraining) that could model GRACE destriping.
Status: FULLTEXT_READ (Luo2023_IRSDE.pdf)

**delbracio2023indi** — M. Delbracio, P. Milanfar, "Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration," TMLR 2023. arXiv:2303.11435.
Small-step iterative restoration directly from degraded to clean images, with no Gaussian noise prior. It avoids the "regression to the mean" of single-step methods while staying simple.
Status: FULLTEXT_READ (Delbracio2023_InDI.pdf)

**bansal2023cold** — A. Bansal, E. Borgnia, H.-M. Chu, J. S. Li, H. Kazemi, F. Huang, M. Goldblum, J. Geiping, T. Goldstein, "Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise," NeurIPS 2023. arXiv:2208.09392.
Shows that diffusion-like generation works with deterministic degradations (blur, masking, downsampling). This supports degradation chains built from the GRACE filtering and truncation operators themselves.
Status: FULLTEXT_READ (Bansal2023_coldDiffusion.pdf)

**rissanen2023heat** — S. Rissanen, M. Heinonen, A. Solin, "Generative Modelling With Inverse Heat Dissipation," ICLR 2023. arXiv:2206.13397.
Uses heat-equation blurring as the forward process, so generation proceeds coarse-to-fine in frequency space. Conceptually aligned with spherical-harmonic degree truncation and Gaussian filtering of GRACE fields: recovering high-degree content is the inverse of the heat process.
Status: FULLTEXT_READ (Rissanen2023_inverseHeatDissipation.pdf)

### E. Diffusion for inverse problems

**kawar2021snips** — B. Kawar, G. Vaksman, M. Elad, "SNIPS: Solving Noisy Inverse Problems Stochastically," NeurIPS 2021. arXiv:2105.14951.
Posterior sampling for noisy linear inverse problems via annealed Langevin dynamics in the SVD domain of the forward operator. A precursor of DDRM that explicitly handles measurement noise.
Status: FULLTEXT_READ (Kawar2021_SNIPS.pdf)

**kawar2022ddrm** — B. Kawar, M. Elad, S. Ermon, J. Song, "Denoising Diffusion Restoration Models," NeurIPS 2022. arXiv:2201.11793.
Efficient (~20-step) unsupervised posterior sampling for *noisy linear* inverse problems with a pretrained DDPM, working in the operator's spectral (SVD) space. GRACE SH truncation plus isotropic filtering is diagonal in the SH basis, so DDRM's formulation applies almost directly to destriping and super-resolution.
Status: FULLTEXT_READ (Kawar2022_DDRM.pdf)

**song2022medical** — Y. Song, L. Shen, L. Xing, S. Ermon, "Solving Inverse Problems in Medical Imaging with Score-Based Generative Models," ICLR 2022. arXiv:2111.08005.
Unsupervised CT/MRI reconstruction from undersampled measurements: an unconditional score prior with a data-consistency step in measurement space. It generalises across measurement processes without retraining, analogous to GRACE's varying monthly ground-track coverage and noise.
Status: FULLTEXT_READ (Song2022_medicalInverseScore.pdf)

**chung2022mcg** — H. Chung, B. Sim, D. Ryu, J. C. Ye, "Improving Diffusion Models for Inverse Problems using Manifold Constraints," NeurIPS 2022. arXiv:2206.00941.
Adds a manifold-constraint gradient correction to projection-based solvers so iterates stay on the data manifold. Improves inpainting, colourisation and sparse-view CT.
Status: FULLTEXT_READ (Chung2022_MCG.pdf)

**chung2023dps** — H. Chung, J. Kim, M. T. McCann, M. L. Klasky, J. C. Ye, "Diffusion Posterior Sampling for General Noisy Inverse Problems," ICLR 2023. arXiv:2209.14687.
Approximates the likelihood score using the Tweedie posterior mean $\hat x_0(x_t)$. This handles noisy and *nonlinear* forward operators (Gaussian or Poisson noise). The most widely used generic approach: it could take a differentiable GRACE operator (SH synthesis, filtering, leakage) with its noise model.
Status: FULLTEXT_READ (Chung2023_DPS.pdf)

**song2023pigdm** — J. Song, A. Vahdat, M. Mardani, J. Kautz, "Pseudoinverse-Guided Diffusion Models for Inverse Problems," ICLR 2023. OpenReview id 9_gsMA8MRKQ (no arXiv version).
ΠGDM: a Gaussian approximation of $p(x_0\mid x_t)$ combined with the operator's pseudo-inverse gives a guidance term for noisy, non-linear or even non-differentiable measurements. It matches task-specific models with a problem-agnostic prior.
Status: FULLTEXT_READ (Song2023_PiGDM.pdf, author-hosted camera-ready from jankautz.com)

**wang2023ddnm** — Y. Wang, J. Yu, J. Zhang, "Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model," ICLR 2023. arXiv:2212.00490.
Range-null-space decomposition: keep the range-space component dictated by the measurement and generate only the null-space content. For GRACE this means exactly preserving the observed low-degree SH coefficients while generating the unobserved high-degree content.
Status: FULLTEXT_READ (Wang2023_DDNM.pdf)

**zhu2023diffpir** — Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, L. Van Gool, "Denoising Diffusion Models for Plug-and-Play Image Restoration," CVPR Workshops (NTIRE) 2023. arXiv:2305.08995.
Plugs a diffusion denoiser into half-quadratic splitting, alternating a data-proximal step with a diffusion denoising step. Works in ~20–100 steps for SR, deblurring and inpainting. The plug-and-play structure makes it easy to swap in GRACE-specific data terms.
Status: FULLTEXT_READ (Zhu2023_DiffPIR.pdf)

**mardani2024reddiff** — M. Mardani, J. Song, J. Kautz, A. Vahdat, "A Variational Perspective on Solving Inverse Problems with Diffusion Models," ICLR 2024. arXiv:2305.04391.
RED-diff: variational inference with the diffusion prior acting as a denoising-based regulariser (score distillation) across timesteps. More stable and interpretable than DPS-style guidance, but it returns a point estimate or narrow posterior.
Status: FULLTEXT_READ (Mardani2024_REDdiff.pdf)

**rozet2023sda** — F. Rozet, G. Louppe, "Score-based Data Assimilation," NeurIPS 2023. arXiv:2306.10574.
Learns a score model over short trajectory segments (Markov blanket) and composes them into a whole-trajectory prior. Conditioned on sparse/noisy observations with a DPS-like likelihood, it gives zero-shot data assimilation in geophysical systems. Very relevant for joint gap filling and denoising of monthly TWS time series.
Status: FULLTEXT_READ (Rozet2023_scoreDataAssimilation.pdf)

**daras2024survey** — G. Daras, H. Chung, C.-H. Lai, Y. Mitsufuji, J. C. Ye, P. Milanfar, A. G. Dimakis, M. Delbracio, "A Survey on Diffusion Models for Inverse Problems," arXiv:2410.00083, 2024 (preprint).
Taxonomy of diffusion inverse solvers (explicit approximations, variational, CSGM-type, asymptotically exact/SMC) with an analysis of their approximations and failure modes. The best single reference for choosing a GRACE solver.
Status: FULLTEXT_READ (Daras2024_surveyDiffusionInverse.pdf)

### F. Spheres, manifolds and spherical architectures

**cohen2018spherical** — T. S. Cohen, M. Geiger, J. Köhler, M. Welling, "Spherical CNNs," ICLR 2018. arXiv:1801.10130.
Rotation-equivariant convolution on $S^2$ and SO(3), computed efficiently with generalised (non-commutative) FFTs in the spherical-harmonic domain. The foundational architecture for learning on global fields without projection distortion.
Status: FULLTEXT_READ (Cohen2018_sphericalCNNs.pdf)

**bonev2023sfno** — B. Bonev, T. Kurth, C. Hundt, J. Pathak, M. Baust, K. Kashinath, A. Anandkumar, "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere," ICML 2023 (PMLR 202). arXiv:2306.03838.
Fourier neural operators generalised to the sphere through the spherical-harmonic transform, giving stable long rollouts for atmospheric dynamics. Because GRACE Level-2 data are SH coefficients, SFNO-style layers are a natural denoiser backbone for a spherical diffusion model.
Status: FULLTEXT_READ (Bonev2023_SFNO.pdf)

**debortoli2022riemannian** — V. De Bortoli, E. Mathieu, M. Hutchinson, J. Thornton, Y. W. Teh, A. Doucet, "Riemannian Score-Based Generative Modelling," NeurIPS 2022. arXiv:2202.02763.
Extends score-based models to compact Riemannian manifolds (including $S^2$), with geodesic random walks and heat-kernel approximations, applied to earth-science event data on the sphere. Theoretical reference for diffusion on $S^2$; it models points, not fields.
Status: FULLTEXT_READ (DeBortoli2022_riemannianScoreBased.pdf)

**huang2022riemannian** — C.-W. Huang, M. Aghajohari, A. J. Bose, P. Panangaden, A. Courville, "Riemannian Diffusion Models," NeurIPS 2022. arXiv:2208.07949.
A variational/likelihood framework for continuous-time diffusion on Riemannian manifolds, with a Riemannian ELBO, demonstrated on spherical earth-science datasets.
Status: FULLTEXT_READ (Huang2022_riemannianDiffusion.pdf)

**elhag2024mdf** — A. A. Elhag, Y. Wang, J. M. Susskind, M. A. Bautista, "Manifold Diffusion Fields," ICLR 2024. arXiv:2305.15586.
Learns generative models of *functions defined on* Riemannian manifolds, using Laplace–Beltrami eigenfunctions as intrinsic coordinates. Demonstrated on climate/weather fields on the sphere. The closest "field-on-$S^2$" diffusion formulation for global TWS; on the sphere the eigenfunctions are exactly spherical harmonics.
Status: FULLTEXT_READ (Elhag2024_manifoldDiffusionFields.pdf)

### G. Spatiotemporal / sequence diffusion and geoscience exemplars

**ho2022video** — J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, D. J. Fleet, "Video Diffusion Models," NeurIPS 2022. arXiv:2204.03458.
Extends diffusion to video with factorised space-time U-Nets, joint image-video training and reconstruction-guided conditional sampling for temporal extension and interpolation. The template for monthly TWS sequences as "videos" and for filling gap months.
Status: FULLTEXT_READ (Ho2022_videoDiffusion.pdf)

**tashiro2021csdi** — Y. Tashiro, J. Song, Y. Song, S. Ermon, "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation," NeurIPS 2021. arXiv:2107.03502.
A conditional diffusion model trained with self-supervised masking that imputes missing values in multivariate time series, using temporal and feature transformers and giving probabilistic imputations assessed with CRPS. Directly relevant to probabilistic GRACE gap filling, including the 11-month GRACE/GRACE-FO gap.
Status: FULLTEXT_READ (Tashiro2021_CSDI.pdf)

**ruhlingcachay2023dyffusion** — S. Rühling Cachay, B. Zhao, H. Joren, R. Yu, "DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting," NeurIPS 2023. arXiv:2306.01984.
Ties the diffusion time axis to physical time through a stochastic temporal interpolator and a forecaster, giving efficient probabilistic spatiotemporal forecasts evaluated with CRPS and spread-skill ratio. Relevant to forecasting and temporal interpolation of TWS.
Status: FULLTEXT_READ (RuhlingCachay2023_DYffusion.pdf)

**lippe2023pderefiner** — P. Lippe, B. S. Veeling, P. Perdikaris, R. E. Turner, J. Brandstetter, "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers," NeurIPS 2023. arXiv:2308.05732.
Shows that neural surrogates neglect low-amplitude high-frequency spectral components, and fixes this with a diffusion-inspired multistep refinement. Analogous to recovering high-degree spectral power suppressed by GRACE filtering.
Status: FULLTEXT_READ (Lippe2023_PDERefiner.pdf)

**price2025gencast** — I. Price, A. Sanchez-Gonzalez, F. Alet, T. R. Andersson, A. El-Kadi, D. Masters, T. Ewalds, J. Stott, S. Mohamed, P. Battaglia, R. Lam, M. Willson, "Probabilistic Weather Forecasting with Machine Learning," Nature 637(8044):84–90, 2025. DOI:10.1038/s41586-024-08252-9; arXiv:2312.15796 (as "GenCast").
A conditional EDM-style diffusion model on a refined icosahedral mesh (graph transformer). It generates global 0.25° ensembles that beat ECMWF ENS on CRPS, with spread-skill and rank-histogram calibration analysis. The flagship example of calibrated spherical diffusion ensembles and a UQ evaluation template.
Status: FULLTEXT_READ (Price2025_GenCast.pdf, arXiv version)

**mardani2025corrdiff** — M. Mardani, N. Brenowitz, Y. Cohen, J. Pathak, C.-Y. Chen, C.-C. Liu, A. Vahdat, M. A. Nabian, T. Ge, A. Subramaniam, K. Kashinath, J. Kautz, M. Pritchard, "Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling," Communications Earth & Environment 6:124, 2025. DOI:10.1038/s43247-025-02042-5; arXiv:2309.15214.
CorrDiff: a UNet regression predicts the conditional mean, and an EDM diffusion model generates the residual. This gives 25 km → 2 km downscaling with realistic extremes and spread. The two-stage "mean + stochastic residual" design is a strong blueprint for GRACE downscaling under small-data constraints.
Status: FULLTEXT_READ (Mardani2025_CorrDiff.pdf, open-access journal version)

**li2024seeds** — L. Li, R. Carver, I. Lopez-Gomez, F. Sha, J. Anderson, "Generative Emulation of Weather Forecast Ensembles with Diffusion Models," Science Advances 10(13):eadk4489, 2024. DOI:10.1126/sciadv.adk4489; arXiv:2306.14066 (as "SEEDS").
Diffusion models conditioned on a few operational ensemble members generate large calibrated ensembles, improving extreme-event statistics. Shows how to amplify small ensembles, which is relevant to expanding GRACE uncertainty ensembles.
Status: FULLTEXT_READ (Li2024_SEEDS.pdf, arXiv version)

### H. Uncertainty quantification, calibration and evaluation metrics

**horwitz2022conffusion** — E. Horwitz, Y. Hoshen, "Conffusion: Confidence Intervals for Diffusion Models," arXiv:2211.09795, 2022 (preprint).
Fine-tunes a pretrained diffusion model to predict per-pixel lower/upper bounds in one pass, calibrated with conformal risk control to guarantee coverage. A cheap alternative to sampling ensembles, offering formal coverage guarantees for GRACE products.
Status: FULLTEXT_READ (Horwitz2022_Conffusion.pdf)

**kou2024bayesdiff** — S. Kou, L. Gan, D. Wang, C. Li, Z. Deng, "BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference," ICLR 2024. arXiv:2310.11142.
Propagates last-layer Laplace epistemic uncertainty through the sampling dynamics to obtain pixel-wise uncertainty of generated images. Addresses *model* (epistemic) uncertainty, which posterior ensembles alone do not capture. This matters in small-data GRACE regimes.
Status: FULLTEXT_READ (Kou2024_BayesDiff.pdf)

**gneiting2007scoring** — T. Gneiting, A. E. Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation," J. Amer. Statist. Assoc. 102(477):359–378, 2007. DOI:10.1198/016214506000001437.
The standard reference for proper scoring rules, including CRPS and the energy score, for evaluating probabilistic predictions. Required for evaluating diffusion ensembles of TWS.
Status: PAYWALLED (summary based on abstract/metadata only; not read)

**hersbach2000crps** — H. Hersbach, "Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems," Weather and Forecasting 15(5):559–570, 2000. DOI:10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2.
Gives the ensemble CRPS formula and its decomposition into reliability, resolution and uncertainty. Standard in geoscience ensemble evaluation.
Status: PAYWALLED (summary based on abstract/metadata only; not read)

**fortin2014spread** — V. Fortin, M. Abaza, F. Anctil, R. Turcotte, "Why Should Ensemble Spread Match the RMSE of the Ensemble Mean?," J. Hydrometeorology 15(4):1708–1713, 2014. DOI:10.1175/JHM-D-14-0008.1.
Derives the finite-ensemble correction ($\sqrt{(M+1)/M}$) for the spread-skill relationship used to assess ensemble calibration in hydrology. Directly applicable to diffusion-based TWS ensembles.
Status: PAYWALLED (summary based on abstract/metadata only; not read)

### I. Surveys

**yang2023survey** — L. Yang, Z. Zhang, Y. Song, S. Hong, R. Xu, Y. Zhao, W. Zhang, B. Cui, M.-H. Yang, "Diffusion Models: A Comprehensive Survey of Methods and Applications," ACM Computing Surveys 56(4):1–39, 2023. DOI:10.1145/3626235; arXiv:2209.00796.
A broad taxonomy covering efficient sampling, improved likelihood, data with special structures (manifolds, graphs) and applications, including interdisciplinary science.
Status: FULLTEXT_READ (Yang2023_diffusionSurvey.pdf, arXiv version)

**croitoru2023survey** — F.-A. Croitoru, V. Hondru, R. T. Ionescu, M. Shah, "Diffusion Models in Vision: A Survey," IEEE TPAMI 45(9):10850–10869, 2023. DOI:10.1109/TPAMI.2023.3261988; arXiv:2209.04747.
A vision-focused survey of DDPM, NCSN and SDE formulations and of applications: SR, inpainting, restoration, image-to-image translation and medical imaging.
Status: FULLTEXT_READ (Croitoru2023_diffusionVisionSurvey.pdf, arXiv version)

## Paywalled

| key | title | DOI |
|---|---|---|
| vincent2011connection | A Connection Between Score Matching and Denoising Autoencoders | 10.1162/NECO_a_00142 |
| anderson1982reverse | Reverse-Time Diffusion Equation Models | 10.1016/0304-4149(82)90051-5 |
| gneiting2007scoring | Strictly Proper Scoring Rules, Prediction, and Estimation | 10.1198/016214506000001437 |
| hersbach2000crps | Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems | 10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2 |
| fortin2014spread | Why Should Ensemble Spread Match the RMSE of the Ensemble Mean? | 10.1175/JHM-D-14-0008.1 |

Note: for papers whose journal versions are paywalled (SR3 TPAMI, Croitoru TPAMI, Yang ACM CSUR, LDM/RePaint CVPR, Palette SIGGRAPH, GenCast Nature, CorrDiff CEE, SEEDS Sci. Adv.), the arXiv preprint version was downloaded. Cite the journal DOI in the survey.
