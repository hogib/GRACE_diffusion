# Super-resolution and diffusion models for GRACE terrestrial water storage: a literature review

## Abstract

This review surveys deep-learning methods that increase the spatial resolution of GRACE and GRACE-FO terrestrial water storage anomalies (TWSA), with a focus on approaches inherited from computer vision and on diffusion models. The literature through November 2026 falls into three groups. Deterministic convolutional networks dominate: one study ports named image super-resolution networks directly, and later work converges on U-Nets trained to match GRACE after spatial averaging while taking fine-scale pattern from a hydrological model. Nine studies use generative adversarial networks, none of which samples ensembles or reports uncertainty. Only one diffusion-model paper on GRACE was found, and it could be verified from its abstract alone.

The field differs from image super-resolution for a physical reason. GRACE has an effective resolution of about 300 km, its finer delivery grids are oversampled, and no high-resolution truth exists. Every method therefore imports sub-300 km structure from a prior, and validation below that scale is indirect. Weather, climate and ocean science have meanwhile developed diffusion downscaling methods that address the same constraints: residual formulations, guidance with an explicit measurement operator, hard conservation layers and calibrated ensembles. None has been applied to satellite gravimetry. The principal gap identified is a diffusion model conditioned on GRACE through its actual observation operator and error covariance, evaluated with probabilistic scores in a closed-loop benchmark.

## 1. Introduction

Satellite gravimetry from GRACE and GRACE-FO measures total water storage change globally, but at a spatial scale too coarse for most basin and aquifer management. Raising that resolution is known in hydrology as spatial downscaling and corresponds to super-resolution in computer vision. Since 2021 a growing body of work has applied vision architectures to the problem, and diffusion models have recently become the leading generative approach for downscaling in neighbouring geosciences. This review asks how far those developments have reached GRACE.

### 1.1 Review questions

1. Which computer-vision-inspired deep-learning methods have been applied to GRACE spatial downscaling, and how are they trained without high-resolution truth?
2. Which generative models, and diffusion models in particular, have been applied to GRACE data?
3. Which diffusion downscaling methods from adjacent geosciences are transferable to GRACE?
4. How are downscaled products validated, what critiques exist, and which research gaps remain?

### 1.2 Scope and search method

The review covers journal articles, preprints, theses and conference abstracts available up to November 2026. It includes work on GRACE and GRACE-FO products and on simulated next-generation gravity missions. Temporal downscaling, gap filling and reconstruction are covered only where they inform the spatial problem. Pixel-wise machine learning and data assimilation are treated as baselines.

Sources were located through general web search, the OpenAlex index, the GFZ database of GRACE publications, Copernicus preprint servers and their open reviews, arXiv, and EGU General Assembly abstracts. Each cited work was checked against a retrievable source. Where only an abstract or a bibliographic record could be read, the text says so. Coverage limits are listed in Section 9.

## 2. Background: resolution limits and established baselines

### 2.1 Effective resolution of GRACE products

The input to GRACE super-resolution carries far less information than its pixel count suggests. JPL solves for 4,551 equal-area 3° mascons and delivers them on a 0.5° grid, warning that neighbouring cells are not independent ([JPL GRACE Tellus](https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/)). CSR releases a 0.25° grid with a true resolution of about 200–300 km ([Bonin et al., 2026](https://essd.copernicus.org/articles/18/3481/2026/)). A resolution-operator analysis of the GSFC 1° mascons gives an effective resolution near 520 km outside Antarctica ([Loomis et al., 2019](https://ntrs.nasa.gov/api/citations/20190004972/downloads/20190004972.pdf)).

Reviews converge on a native resolution of roughly 300 km ([Springer et al., 2026](https://hess.copernicus.org/articles/30/985/2026/); [Pascal et al., 2022](https://hess.copernicus.org/articles/26/4169/2022/)). A closed-loop study placed the smallest resolvable catchment at about 63,000 km² for a 2 cm error ([Vishwakarma et al., 2018](https://www.mdpi.com/2072-4292/10/6/852/htm)). Three quantities are therefore called resolution in the literature: basis-function size, delivery grid and effective resolution. Papers usually quote the delivery grid, so nominal scale factors understate the true one.

### 2.2 Downscaling before vision networks

Pascal et al. (2022) divide the field into model-based downscaling, which assimilates GRACE into a land-surface or hydrological model, and data-based downscaling, which fits a statistical relation at coarse scale and applies it to fine-scale predictors. Springer et al. (2026) review about 60 assimilation studies and name resolution mismatch as an open problem. The main global assimilation product, GLWS2.0, agrees better with GNSS vertical loading at more than 1,000 stations than GRACE itself ([Gerdener et al., 2022](https://arxiv.org/abs/2211.17138)).

The statistical branch is dominated by pixel-wise random forests and gradient boosting, according to a systematic review of 80 articles from 2011–2025 ([Al Nadabi et al., 2026](https://www.mdpi.com/2306-5338/13/5/135)). Measured gains can be small. In southern India, random-forest downscaling to 0.5° raised mean correlation with 1,006 wells from 0.76 to 0.79, and the mascon scale-factor product reduced accuracy (Pascal et al., 2022).

### 2.3 Deep learning on the temporal problem

Deep learning reached gap filling and reconstruction before spatial super-resolution. [Sun et al. (2019)](https://ar5iv.arxiv.org/html/1902.01933) trained CNNs to learn the mismatch between a land-surface model and GRACE over India. [Mo et al. (2021)](https://arxiv.org/abs/2101.09361) bridged the gap between the two missions with a Bayesian CNN, and later forecast TWSA one to three months ahead with ensemble inference ([Mo et al., 2025](https://vbn.aau.dk/ws/files/774394102/Mo_DL4TWS_GRL2025.pdf)). Long reconstructions include the statistical GRACE-REC ([Humphrey and Gudmundsson, 2019](https://essd.copernicus.org/articles/11/1153/2019)), GTWS-MLrec ([Yin et al., 2023](https://essd.copernicus.org/articles/15/5597/2023/)) and the BiLSTM-based GRAiCE ([Palazzoli et al., 2025](https://cris.unibo.it/bitstream/11585/1003215/1/s41597-025-04403-3.pdf)). These are driven by meteorological forcing and cannot represent human-driven trends, a weakness that spatial downscalers with similar predictors share.

## 3. Deterministic deep-learning super-resolution

### 3.1 Direct ports of image super-resolution networks

Only one verified study transplants named single-image super-resolution networks to GRACE. [He et al. (2021)](https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/files/2021_he_yang_wang_li_cjrs.pdf) trained SRCNN, VDSR and RCAN over Canada to map 110 km JPL fields to 10 km, an 11× factor. The EALCO land-surface model served as high-resolution truth, and training pairs were built by bicubic degradation. Evaluation was limited to re-aggregated output: VDSR reached 14.4 mm RMSE, RCAN 18.4 mm and SRCNN 22.3 mm, against 82.0 mm for bicubic interpolation.

The ranking is the clearest evidence on capacity against sample size in this domain. RCAN, with 20.6 million parameters and 640 training tiles from 158 months, lost to VDSR with 0.67 million, which the authors attribute to data scarcity. The study has no mass-conservation term, no separate test period and no tree-based baseline. No successor using this model-as-truth recipe was found, and no verified GRACE paper applies EDSR, SwinIR or Restormer.

### 3.2 Self-supervised networks with a dual loss

The field instead adopted encoder-decoder CNNs with a two-term loss: agree with GRACE after spatial averaging, and resemble a hydrological model at fine scale. The line begins with a master's thesis titled as super-resolution of GRACE ([Gou, 2022](https://www.research-collection.ethz.ch/server/api/core/bitstreams/c295321d-28fe-4e40-af13-fdd4dc14f022/content)). It leads to the global 0.5° product of [Gou and Soja (2024)](https://www.nature.com/articles/s44221-024-00194-w), a residual network fed with JPL mascons, WaterGAP (WGHM) storage and GLDAS fluxes. Its area-weighted basin RMSE against GRACE is 21.9 mm, against 49.2 mm for WGHM, over 288 basins. A commentary noted its reliance on WGHM spatial patterns and weak handling of human intervention ([Sun, 2024](https://www.beg.utexas.edu/files/content/beg/ext-aff/2024/02%20Feb/Learning%20to%20downscale%20satellite%20gravimetry%20data%20through%20artificial%20intelligence.pdf)). The same design was extended to ocean bottom pressure ([Gou et al., 2025](https://link.springer.com/article/10.1007/s00190-025-01943-9)).

Later networks vary the weighting and the prior. [Uz et al. (2026)](https://gfzpublic.gfz.de/rest/items/item_5037795_1/component/file_5038113/content) add a variational bottleneck to a global U-Net and weight the GRACE-consistency term by the running correlation between prediction and WGHM. [Li et al. (2026)](https://essd.copernicus.org/preprints/essd-2026-282/) downscale from 3° to 0.1° over China with PCR-GLOBWB 2 as the pattern prior; a referee of this preprint reported mascon-like boundaries in the 0.1° fields. [Goracci et al. (2026)](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4563/) reduce the design to an aggregation-consistency loss alone, with fine structure supplied by ERA5-Land, topography and land-cover channels. They state that the fine-scale solution is underconstrained.

| Study | Architecture and training signal | Resolution and domain | Main result or caveat |
| --- | --- | --- | --- |
| He et al. (2021) | SRCNN, VDSR, RCAN; land model as truth, bicubic degradation | 110 km to 10 km, Canada | VDSR 14.4 mm re-aggregated RMSE; largest network underperformed |
| Gou and Soja (2024) | Residual encoder-decoder; GRACE patch mean plus WGHM pattern | About 3° to 0.5°, global | Basin RMSE 21.9 mm against 49.2 mm for WGHM |
| Uz et al. (2026) | Variational U-Net; dynamically weighted soft constraint | About 300 km to 50 km, global | Alaska trend −13.2 Gt/yr against −21.9 in JPL mascons |
| Li et al. (2026), preprint | U-Net; mascon conservation plus PCR-GLOBWB pattern | 3° to 0.1°, China | Referee reports mascon-like boundaries at 0.1° |
| Goracci et al. (2026), preprint | U-Net; aggregation consistency only | 2–3° to 1°, simulated future missions and real GRACE(-FO) | Fine scale underconstrained; closed-loop truth available |

### 3.3 Attention, transformers and graph networks

Attention appears only as a component: channel attention in RCAN, and temporal attention on CNN-LSTM hybrids over Yunnan ([He, Y. et al., 2026](https://www-app2.gfz.de/grace-references/b2hd-2026CG....20906117H.html)) and Poyang Lake ([Yu et al., 2025](https://www.mdpi.com/2072-4292/17/6/988)). The one transformer-bearing downscaler is adversarial (Section 4.1). The one graph neural network addresses temporal reconstruction ([Arzoumanidis et al., 2026](https://arxiv.org/abs/2606.23833)). No deterministic Swin or vision-transformer super-resolver for GRACE was found.

### 3.4 Comparison with pixel-wise machine learning

Head-to-head evidence is thin and mixed. A ConvLSTM beat random forest by 7% over Iran ([Foroumandi et al., 2023](https://moradkhani.ua.edu/?p=2758)), but random forest was best among four methods over Morocco ([Mercha et al., 2026](https://meetingorganizer.copernicus.org/EGU26/EGU26-645.html)). XGBoost was chosen over a deep network for a global 1 km groundwater product, whose re-aggregated R² of 0.972 falls to 0.538 against held-out wells ([Zhang et al., 2026](https://essd.copernicus.org/articles/18/7593/2026/)). No image-to-image super-resolution paper benchmarks against tree methods on identical data.

Independent comparisons across products have begun. Over 288 basins the Gou and Soja product has a median RMSE of 1.87 cm, against 2.24 cm for a non-learned joint inversion and 3.40 cm for GLWS ([Xiong et al., 2026a](https://essd.copernicus.org/articles/18/4537/2026/)). The two global deep-learning products share a grid, a GRACE source and the WGHM prior, so they are not independent evidence of fine-scale structure.

## 4. Generative models applied to GRACE

### 4.1 Generative adversarial networks

GAN use on GRACE dates from 2023, and nine works were identified. Every one treats the generator as a deterministic regressor with an adversarial sharpening term. None draws multiple samples or reports uncertainty.

| Study | Task | GAN variant | Scale and region | Reported evidence |
| --- | --- | --- | --- | --- |
| [Gu et al. (2024)](https://api.openalex.org/works/doi:10.1109/TGRS.2024.3349548) | Spatial downscaling | GAN with transformer attention | 0.25° to 0.05°, southwestern China | NSE 0.58–0.92 (abstract only) |
| [Hassanibesheli et al. (2024)](https://essopenarchive.org/doi/full/10.22541/essoar.172493940.07693855/v1), preprint | Spatial downscaling | Conditional WGAN-GP | South America | Qualitative; no journal version found |
| [Wang et al. (2025)](https://www-app2.gfz.de/grace-references/b2hd-2025ScTEn.96978874W.html), DownGAN | Spatial downscaling | GAN with a new downscaling scheme | About 300 km to 25 km, Yangtze and Nile | Outperforms a convolutional network in a synthetic test (abstract only) |
| [Dombos et al. (2026)](https://arxiv.org/html/2605.00652v1), D-SHIFT, preprint | Monthly mascon detail onto daily fields | ESRGAN-family generator, U-Net discriminator | Degree-40 harmonics to 0.25°, global | RMSE 2.4 cm (2008) and 4.3 cm (2015–16) |
| [Wang et al. (2024a)](https://isprs-annals.copernicus.org/articles/X-4-2024/389/2024/isprs-annals-X-4-2024-389-2024.pdf), RecGAN | Hindcast | Conditional WGAN-GP | 1°, global land | NSE 0.17 to 0.46 in the Nelson basin |
| [Wang et al. (2024b)](https://www.mdpi.com/2072-4292/16/24/4727), GAIN | Gap filling | Generative adversarial imputation | Basin series, Pearl River | About 51% better than LSTM on synthetic gaps |
| [Heller-Kaikov et al. (2023)](https://www.asg.ed.tum.de/fileadmin/w00cip/iapg/pdf/Heller-Kaikov_Signal_separation_geoAI2023.pdf) | Signal separation, simulated | pix2pix | Global | Errors 27–81% of signal; PCA competitive |
| [Xiong et al. (2026b)](https://api.openalex.org/works/doi:10.2139/ssrn.6254347), preprint | Spatial downscaling | GAN with dense blocks | Tibetan Plateau | Title and keywords only |
| [Fan et al. (2025)](https://api.openalex.org/works/doi:10.2139/ssrn.5583646), preprint | Spatial downscaling | GAN, inferred from keywords | Songhua basin | Title and keywords only |

D-SHIFT is the best documented and shows both the appeal and the risk of adversarial super-resolution. It has no daily ground truth and no formal uncertainty. Its authors report increased high-frequency noise and isolated daily excursions not clearly supported by the monthly reference. The pix2pix experiment is a counterweight: adversarial losses never stabilised, and test errors were about three times training errors.

### 4.2 Probabilistic deep learning

Three approaches give uncertainty, all as per-pixel spread and none as spatially coherent sample fields. Uz et al. (2026) combine latent sampling with Monte Carlo dropout into a 95th-percentile uncertainty of about 3.2–3.3 cm. The Bayesian CNNs of Mo et al. (2021, 2025) use particle ensembles. The U-Net studies use seed ensembles of three to ten members.

Two observations suggest these spreads are too narrow. The Uz et al. uncertainty is lower than that of the JPL mascons it derives from, at six times finer sampling. The three-run spread in Li et al. (2026) was criticised in open review as covering model uncertainty only. No normalizing-flow model and no GRACE-specific foundation model was found.

### 4.3 Diffusion models

One diffusion work on GRACE was found: Deep-GRASP by [Kiani Shahvandi (2026)](https://api.openalex.org/works/doi:10.1016/j.geog.2026.09.010), published in *Geodesy and Geodynamics* in October 2026 with an earlier [preprint](https://api.openalex.org/works/doi:10.22541/essoar.15007597/v1). Its abstract describes a method based on denoising diffusion probabilistic models that couples a U-Net, multilayer perceptrons and a convolutional LSTM. It addresses spatial downscaling, daily reconstruction back to 1980 and prediction up to one year ahead, at global scale.

The abstract reports R² of about 0.64 against independent hydrometeorological observations, and claims 41% higher accuracy and 51% lower cost than existing algorithms. These are author claims against unnamed baselines. The full text could not be read, so the resolutions, conditioning variables, treatment of mass conservation and any ensemble output are unknown.

No other diffusion, score-based, latent-diffusion or flow-matching study on GRACE products was located. The nearest gravity-side work interpolates and denoises exploration survey grids ([Hou et al., 2025](https://api.openalex.org/works/doi:10.1007/s11004-025-10255-9)) or uses altimetric gravity as conditioning for bathymetry ([Chen et al., 2026](https://www.mdpi.com/2072-4292/18/3/496)).

## 5. Diffusion downscaling in adjacent geosciences

Weather, climate and ocean science have converged on a small set of diffusion downscaling designs. This section reviews those that address constraints GRACE shares: few samples, no paired high-resolution target, and a conservation requirement.

### 5.1 Residual diffusion

In residual diffusion a deterministic network predicts the conditional mean and a diffusion model generates only the residual. CorrDiff introduced the design for 25 km to 2 km downscaling over Taiwan ([Mardani et al., 2025](https://arxiv.org/html/2309.15214v4)). R2-D2 applies it to regional climate fields at 45 km to 9 km and lowers CRPS by more than 40% relative to a standard statistical method ([Lopez-Gomez et al., 2025](https://arxiv.org/html/2410.01776v1)). A latent-diffusion variant over Italy beat a GAN with the same backbone ([Tomasi et al., 2025](https://gmd.copernicus.org/articles/18/2051/2025/)).

These models train on far more samples than GRACE offers. Two results narrow that distance: diffusion downscaling of monthly fields at 10× for an East Asian reconstruction ([Ling et al., 2024](https://arxiv.org/abs/2402.06646v2)), and a precipitation model that matched full-data skill with five years of training data by using a frozen pretrained encoder ([Ribeiro et al., 2026](https://arxiv.org/abs/2608.25858)).

### 5.2 Generative priors with guidance

A second family removes the need for paired data. A generative prior is trained on high-resolution fields alone, and the coarse observation is imposed at sampling time. The simplest version adds noise to the coarse field up to an intermediate level and denoises with the prior ([Bischoff and Deck, 2024](https://clima.caltech.edu/wp-content/uploads/2024/06/aies-AIES-D-23-0039.1.pdf)). [Hess et al. (2025)](https://arxiv.org/html/2403.02774v4) set that level where the power spectra of the coarse and fine data intersect, a rule that suits a band-limited signal such as GRACE.

The more rigorous version is posterior sampling with an explicit measurement model. [Tie et al. (2026)](https://arxiv.org/html/2601.21760v2) train on reanalysis only and guide sampling with an area-averaging operator. They also report that supervised networks trained on synthetic pairs degrade on real model inputs, in some cases below the coarse input. Guidance from several observation types at once has been shown ([Chakraborty et al., 2025](https://arxiv.org/html/2506.22780v1)).

The closest analogue of a GRACE recipe is sea-surface-salinity downscaling by [Forestier et al. (2026)](https://os.copernicus.org/articles/22/2357/2026/). It combines an unconditional prior trained on ocean reanalysis, guidance toward a coarse field, and high-resolution auxiliary fields. The diffusion model only slightly outperformed a CNN, and the authors describe their ensemble variance as uncalibrated. [Han et al. (2024)](https://arxiv.org/abs/2408.12632) train on reanalysis, apply the model to observed sea-surface height and validate on independent along-track data. Guidance has a cost: where paired data exist, conditionally trained models reconstruct fine scales better than guided sampling ([Suresh Babu et al., 2025](https://arxiv.org/abs/2507.00719v2)).

### 5.3 Conservation constraints

[Harder et al. (2023)](https://arxiv.org/html/2208.05424v9) designed output layers that reproduce every coarse-cell mean exactly. These hard constraints usually improved accuracy, whereas a soft loss penalty left violations in place and worsened RMSE. Every GRACE network reviewed in Section 3 uses a soft penalty.

Two caveats limit direct transfer. Hard constraints did not help when the coarse and fine data came from mutually inconsistent sources. And the GRACE measurement is not a block average but a truncated, filtered or mascon-averaged field with spatially correlated errors, so a constraint layer would have to be generalised to that operator. No implementation was found in any field.

### 5.4 Evidence from intercomparisons

Benchmarks agree that deterministic CNNs smooth fields and that GANs and diffusion models restore fine-scale variability. They also temper expectations. In a controlled comparison, diffusion gains were small relative to cost, and variability across training seeds exceeded variability across samples ([Singh et al., 2026](https://gmd.copernicus.org/articles/19/7545/2026/)). Ensembles from CorrDiff and R2-D2 are under-dispersive. Generative models lead clearly for precipitation but only marginally for temperature ([Rampal et al., 2026](https://arxiv.org/abs/2606.29172)), and realistic appearance does not guarantee correct change signals ([Rampal et al., 2025](https://arxiv.org/abs/2512.13987v1); [Najafi et al., 2026](https://www.mdpi.com/2073-4441/18/2/271)).

Water storage is a smooth field closer to temperature than to precipitation. The case for diffusion on GRACE therefore rests on spatially coherent uncertainty more than on lower RMSE.

## 6. Validation practice and critiques

### 6.1 How downscaled products are validated

With no fine-scale truth, validation is indirect. Re-aggregation consistency is nearly universal, but it is enforced by the training loss, so high scores at native resolution do not confirm fine-scale accuracy (Goracci et al., 2026). Groundwater wells are the most common independent check, yet few studies quantify improvement over GRACE without downscaling (Pascal et al., 2022). Wells also measure one storage component through an uncertain specific yield.

Other checks include water-balance closure in small basins, streamflow, soil moisture, glacier mass change and closed-loop simulation. GNSS vertical loading has validated the assimilation product GLWS2.0, but no GNSS check of a deep-learning super-resolution product was found. Some checks are partly circular: the Uz et al. (2026) product is compared with its own training target and with one of its inputs.

Third-party evaluation is beginning. [Suryawanshi et al. (2026)](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-5587/) test downscaled products against wells over Indian mascons. In this preprint a mass-conserving statistical method improves on the mascon product more often than the Gou and Soja product does.

### 6.2 Critiques

Five critiques recur across limitations sections, reviews and open referee reports.

- **Oversampled inputs.** The low-resolution field carries no independent information at pixel level (Section 2.1).
- **Inherited fine scale.** In Uz et al. (2026) the Nile trend is 4.6 mm/yr, between 7.0 in the JPL mascons and −0.1 in WGHM. The pattern prior alters even the large-scale signal that gravimetry measures best.
- **Human impacts.** Hydrological models misrepresent irrigation and groundwater depletion (Springer et al., 2026). A trend-sign mismatch in the High Plains Aquifer is reported for a 1 km product (Zhang et al., 2026).
- **Small samples and weak splits.** Studies use random month splits, train on all months, or tune on years that are not a strict test set (Dombos et al., 2026; Yin et al., 2023).
- **Untested hallucination.** No generative GRACE paper tests for invented structure. Known fine-scale truth exists only in the DownGAN synthetic test and in closed-loop simulations.

[Xiong et al. (2026a)](https://essd.copernicus.org/articles/18/4537/2026/) describe their own 0.5° product as a GRACE-constrained reconstruction informed by high-resolution spatial priors. That description fits every learned product reviewed here.

### 6.3 Future missions

GRACE-C is scheduled for 2028 as a continuity mission with laser ranging on a polar orbit ([AEI, Max Planck Society](https://www.aei.mpg.de/1130150/erfolgreiche-deutsch-amerikanische-umweltmission-wird-mit-grace-c-fortgesetzt)). In closed-loop simulation the inclined NGGM pair and the combined MAGIC constellation reduce flood-signal noise over the Danube from 29.3 mm to 17.7 mm and 13.0 mm ([Pfeffer et al., 2026](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4642/egusphere-2026-4642.pdf)). The same U-Net produces better 1° fields from MAGIC-like than from GRACE-C-like input, with Danube water-balance RMSE falling from 14.47 to 7.10 mm (Goracci et al., 2026). Current methods therefore appear limited by input information, not by network capacity.

## 7. Research gaps and future directions

The gaps below follow from Sections 3 to 6. The proposed directions are inferences from the adjacent literature and have not been tested on GRACE.

1. **Observation operator.** No learned GRACE downscaler models the actual measurement operator or its correlated error covariance. Verified degradation operators are bicubic resampling or box averaging. A diffusion prior trained on hydrological-model output and guided by the true operator would match the structure of the problem.
2. **Conservation by construction.** Every GRACE network uses a soft penalty, against evidence that hard constraints are exact and more accurate. A hard constraint for a non-block operator has not been built.
3. **Calibrated uncertainty.** No GRACE downscaling paper reports CRPS, spread-skill ratios or rank histograms. Existing spreads are per pixel and come from single models.
4. **Common benchmark.** No study scores a U-Net, a GAN, a diffusion model and a tree method on the same held-out reference. Closed-loop simulations with known truth are the natural vehicle.
5. **Spectral attribution.** No study separates skill above and below 300 km, or quantifies how much sub-300 km variance comes from the prior and how much from GRACE.
6. **Architectures.** Transformer super-resolution backbones and residual or flow-matching generative designs are unexplored for GRACE. No GAN has been compared with a diffusion model on the same task.
7. **Priors.** No generative prior or foundation model for land water storage exists. Hydrological models are weakest for groundwater abstraction, surface water and glaciers, where GRACE is most informative.
8. **Independent observations.** GNSS vertical loading has not been used to condition or validate a deep-learning super-resolution product.
9. **Replication.** The one diffusion paper on GRACE needs independent reading and replication before its claims can be relied on.

## 8. Conclusion

Super-resolution of GRACE differs in kind from image super-resolution. A vision network recovers detail that existed in a real high-resolution image. A GRACE network distributes a well-measured 300 km mass budget across pixels according to a prior taken from a model or auxiliary data. The choice of architecture matters less than three decisions the literature has mostly left implicit: which prior supplies the fine scale, how strictly the gravimetric constraint is enforced, and whether the remaining ambiguity is reported as a distribution.

Diffusion models are suited to the third decision, because they sample from the set of fine-scale fields consistent with an observation. Adjacent fields have developed the necessary methods, and none has been applied to gravimetry beyond a single recent paper whose details remain unverified. The expected benefit is calibrated, spatially coherent uncertainty and a clear separation between what gravimetry observed and what the prior supplied.

## 9. Limitations of this review

- **Search coverage.** The finding of a single diffusion paper rests on about twenty search phrasings. AGU, IGARSS, IEEE Xplore and ScienceDirect could not be searched systematically, so conference abstracts may have been missed.
- **Abstract-only sources.** Kiani Shahvandi (2026), Gu et al. (2024) and Wang et al. (2025) were read as abstracts or bibliographic records. Xiong et al. (2026b) and Fan et al. (2025) were verified by title and keywords only.
- **Secondary source.** Foroumandi et al. (2023) is cited from a summary page, not the article.
- **Preprints.** Several 2026 sources are unreviewed preprints and are marked as such in the reference list.
- **Conflicting versions.** Two preprint versions of Suryawanshi et al. give different mascon counts, so no tally is quoted here.
- **Unverified mission details.** The programmatic status of NGGM and its expected resolution in kilometres were not confirmed.

## References

Entries give authors, year, venue and a link to the version consulted. Full titles and author lists should be completed from the linked records before formal submission.

- AEI, Max Planck Society. GRACE-C mission announcement. [Link](https://www.aei.mpg.de/1130150/erfolgreiche-deutsch-amerikanische-umweltmission-wird-mit-grace-c-fortgesetzt)
- Al Nadabi et al. (2026). *Hydrology*, 13(5), 135. [Link](https://www.mdpi.com/2306-5338/13/5/135)
- Arzoumanidis et al. (2026). arXiv:2606.23833, preprint. [Link](https://arxiv.org/abs/2606.23833)
- Bischoff and Deck (2024). *Artificial Intelligence for the Earth Systems*. [Link](https://clima.caltech.edu/wp-content/uploads/2024/06/aies-AIES-D-23-0039.1.pdf)
- Bonin et al. (2026). *Earth System Science Data*, 18, 3481. [Link](https://essd.copernicus.org/articles/18/3481/2026/)
- Chakraborty et al. (2025). arXiv:2506.22780, preprint. [Link](https://arxiv.org/html/2506.22780v1)
- Chen et al. (2026). *Remote Sensing*, 18(3), 496. [Link](https://www.mdpi.com/2072-4292/18/3/496)
- Dombos, Gou and Soja (2026). D-SHIFT. arXiv:2605.00652, preprint. [Link](https://arxiv.org/html/2605.00652v1)
- Fan et al. (2025). SSRN preprint, doi:10.2139/ssrn.5583646. [Link](https://api.openalex.org/works/doi:10.2139/ssrn.5583646)
- Forestier et al. (2026). *Ocean Science*, 22, 2357. [Link](https://os.copernicus.org/articles/22/2357/2026/)
- Foroumandi et al. (2023). Cited from a summary page. [Link](https://moradkhani.ua.edu/?p=2758)
- Gerdener et al. (2022). GLWS2.0. arXiv:2211.17138. [Link](https://arxiv.org/abs/2211.17138)
- Goracci, Daras and Longépé (2026). EGUsphere preprint, doi:10.5194/egusphere-2026-4563. [Link](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4563/)
- Gou (2022). Super Resolution of GRACE Total Water Storage Anomaly using Deep Learning Algorithms. Master's thesis, ETH Zurich. [Link](https://www.research-collection.ethz.ch/server/api/core/bitstreams/c295321d-28fe-4e40-af13-fdd4dc14f022/content)
- Gou and Soja (2024). *Nature Water*, doi:10.1038/s44221-024-00194-w. [Link](https://www.nature.com/articles/s44221-024-00194-w)
- Gou et al. (2025). *Journal of Geodesy*, doi:10.1007/s00190-025-01943-9. [Link](https://link.springer.com/article/10.1007/s00190-025-01943-9)
- Gu, Zhou, Zhao, Ma, She, Zhang and Li (2024). A Fast Generative Adversarial Network Combined With Transformer for Downscaling GRACE Terrestrial Water Storage Data in Southwestern China. *IEEE Transactions on Geoscience and Remote Sensing*, 62, doi:10.1109/TGRS.2024.3349548. [Link](https://api.openalex.org/works/doi:10.1109/TGRS.2024.3349548)
- Han et al. (2024). arXiv:2408.12632, preprint. [Link](https://arxiv.org/abs/2408.12632)
- Harder et al. (2023). *Journal of Machine Learning Research*. [Link](https://arxiv.org/html/2208.05424v9)
- Hassanibesheli, Jensen, Dobslaw, Thomas and Saynisch-Wagner (2024). Downscaling Satellite Gravimetry Observation for Terrestrial Hydrology Using a Conditional Generative Adversarial Network. ESS Open Archive, preprint. [Link](https://essopenarchive.org/doi/full/10.22541/essoar.172493940.07693855/v1)
- He et al. (2021). Deep Learning Approaches to Spatial Downscaling of GRACE Terrestrial Water Storage Products Using EALCO Model Over Canada. *Canadian Journal of Remote Sensing*, 47(4), 657–675, doi:10.1080/07038992.2021.1954498. [Link](https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/files/2021_he_yang_wang_li_cjrs.pdf)
- He, Y. et al. (2026). Record in the GFZ GRACE publications database. [Link](https://www-app2.gfz.de/grace-references/b2hd-2026CG....20906117H.html)
- Heller-Kaikov et al. (2023). ACM GeoAI workshop. [Link](https://www.asg.ed.tum.de/fileadmin/w00cip/iapg/pdf/Heller-Kaikov_Signal_separation_geoAI2023.pdf)
- Hess et al. (2025). *Nature Machine Intelligence*. [Link](https://arxiv.org/html/2403.02774v4)
- Hou et al. (2025). *Mathematical Geosciences*, doi:10.1007/s11004-025-10255-9. [Link](https://api.openalex.org/works/doi:10.1007/s11004-025-10255-9)
- Humphrey and Gudmundsson (2019). GRACE-REC. *Earth System Science Data*, 11, 1153. [Link](https://essd.copernicus.org/articles/11/1153/2019)
- JPL GRACE Tellus. JPL global mascons product page. [Link](https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/)
- Kiani Shahvandi (2026). Deep-GRASP: deep generative downscaling-assimilation-prediction for GRACE/GRACE-FO terrestrial water storage anomalies. *Geodesy and Geodynamics*, doi:10.1016/j.geog.2026.09.010. [Link](https://api.openalex.org/works/doi:10.1016/j.geog.2026.09.010)
- Li, Sun, Gu, Wanders, Scanlon and Slater (2026). ESSD preprint, doi:10.5194/essd-2026-282. [Link](https://essd.copernicus.org/preprints/essd-2026-282/)
- Ling et al. (2024). arXiv:2402.06646, preprint. [Link](https://arxiv.org/abs/2402.06646v2)
- Loomis et al. (2019). GSFC mascon resolution analysis, NASA Technical Reports Server. [Link](https://ntrs.nasa.gov/api/citations/20190004972/downloads/20190004972.pdf)
- Lopez-Gomez et al. (2025). *PNAS*, doi:10.1073/pnas.2420288122. [Link](https://arxiv.org/html/2410.01776v1)
- Mardani et al. (2025). CorrDiff. *Communications Earth & Environment*. [Link](https://arxiv.org/html/2309.15214v4)
- Mercha et al. (2026). EGU General Assembly abstract EGU26-645. [Link](https://meetingorganizer.copernicus.org/EGU26/EGU26-645.html)
- Mo et al. (2021). arXiv:2101.09361. [Link](https://arxiv.org/abs/2101.09361)
- Mo et al. (2025). *Geophysical Research Letters*. [Link](https://vbn.aau.dk/ws/files/774394102/Mo_DL4TWS_GRL2025.pdf)
- Najafi et al. (2026). *Water*, 18(2), 271. [Link](https://www.mdpi.com/2073-4441/18/2/271)
- Palazzoli et al. (2025). GRAiCE. *Scientific Data*. [Link](https://cris.unibo.it/bitstream/11585/1003215/1/s41597-025-04403-3.pdf)
- Pascal et al. (2022). *Hydrology and Earth System Sciences*, 26, 4169. [Link](https://hess.copernicus.org/articles/26/4169/2022/)
- Pfeffer et al. (2026). EGUsphere preprint. [Link](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4642/egusphere-2026-4642.pdf)
- Rampal et al. (2025). arXiv:2512.13987, preprint. [Link](https://arxiv.org/abs/2512.13987v1)
- Rampal et al. (2026). CORDEX-ML-Bench. arXiv:2606.29172, preprint. [Link](https://arxiv.org/abs/2606.29172)
- Ribeiro et al. (2026). arXiv:2608.25858, preprint. [Link](https://arxiv.org/abs/2608.25858)
- Singh et al. (2026). *Geoscientific Model Development*, 19, 7545. [Link](https://gmd.copernicus.org/articles/19/7545/2026/)
- Springer et al. (2026). *Hydrology and Earth System Sciences*, 30, 985–1022. [Link](https://hess.copernicus.org/articles/30/985/2026/)
- Sun (2024). Learning to downscale satellite gravimetry data through artificial intelligence. *Nature Water*, News & Views. [Link](https://www.beg.utexas.edu/files/content/beg/ext-aff/2024/02%20Feb/Learning%20to%20downscale%20satellite%20gravimetry%20data%20through%20artificial%20intelligence.pdf)
- Sun et al. (2019). arXiv:1902.01933. [Link](https://ar5iv.arxiv.org/html/1902.01933)
- Suresh Babu et al. (2025). arXiv:2507.00719, preprint. [Link](https://arxiv.org/abs/2507.00719v2)
- Suryawanshi et al. (2026). EGUsphere preprint 2026-5587. [Link](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-5587/)
- Tie et al. (2026). arXiv:2601.21760, preprint. [Link](https://arxiv.org/html/2601.21760v2)
- Tomasi et al. (2025). *Geoscientific Model Development*, 18, 2051. [Link](https://gmd.copernicus.org/articles/18/2051/2025/)
- Uz, Atman, Akyılmaz and Shum (2026). A dynamic soft-constrained deep learning paradigm for spatial downscaling of satellite gravimetry terrestrial water storage. *Journal of Hydrology*, 668, 135015. [Link](https://gfzpublic.gfz.de/rest/items/item_5037795_1/component/file_5038113/content)
- Vishwakarma et al. (2018). *Remote Sensing*, 10(6), 852. [Link](https://www.mdpi.com/2072-4292/10/6/852/htm)
- Wang, Shen and Awange (2024a). RecGAN. *ISPRS Annals*, X-4-2024, 389. [Link](https://isprs-annals.copernicus.org/articles/X-4-2024/389/2024/isprs-annals-X-4-2024-389-2024.pdf)
- Wang et al. (2024b). GAIN. *Remote Sensing*, 16(24), 4727. [Link](https://www.mdpi.com/2072-4292/16/24/4727)
- Wang, Shen, Awange, Tabatabaeiasl, Song and Liu (2025). A novel generative adversarial network and downscaling scheme for GRACE/GRACE-FO products: Exemplified by the Yangtze and Nile River Basins. *Science of the Total Environment*, 969, 178874. [Link](https://www-app2.gfz.de/grace-references/b2hd-2025ScTEn.96978874W.html)
- Xiong et al. (2026a). *Earth System Science Data*, 18, 4537. [Link](https://essd.copernicus.org/articles/18/4537/2026/)
- Xiong et al. (2026b). SSRN preprint, doi:10.2139/ssrn.6254347. [Link](https://api.openalex.org/works/doi:10.2139/ssrn.6254347)
- Yin et al. (2023). GTWS-MLrec. *Earth System Science Data*, 15, 5597. [Link](https://essd.copernicus.org/articles/15/5597/2023/)
- Yu et al. (2025). *Remote Sensing*, 17(6), 988. [Link](https://www.mdpi.com/2072-4292/17/6/988)
- Zhang et al. (2026). *Earth System Science Data*, 18, 7593. [Link](https://essd.copernicus.org/articles/18/7593/2026/)
