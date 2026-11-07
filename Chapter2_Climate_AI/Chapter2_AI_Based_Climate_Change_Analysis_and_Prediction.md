# AI-Based Climate Change Analysis and Prediction: Hybrid Modelling, Explainable Forecasting and Uncertainty Quantification

**Uday Pratap Singh^1,\*^, Ebtasam Ahmad Siddiqui^2^**

^1^ Department of AI & DS, Poornima Institute of Engineering & Technology, Jaipur, India

^2^ Department of CE, Poornima Institute of Engineering & Technology, Jaipur, India

\*Corresponding email: udaypratap@allduniv.ac.in

ORCID IDs: Author 1: [16-digit ORCID]; Author 2: [16-digit ORCID]

## Abstract

Climate research increasingly depends on extracting information from large, heterogeneous and rapidly growing environmental datasets. Artificial intelligence offers new ways to find nonlinear relationships, recognise spatial and temporal structure, forecast environmental variables and support climate-risk assessment. Prediction accuracy alone is, however, an inadequate measure of usefulness. The climate system obeys conservation laws, exhibits large irreducible uncertainty and varies strongly across space and time, which makes purely data-driven approaches fragile outside the conditions they were trained on. This chapter examines AI for climate analysis and prediction with emphasis on the connection between statistical learning and physical knowledge. It reviews climate datasets and pre-processing, classical machine learning, and deep spatiotemporal architectures including convolutional, recurrent and attention-based models. Applications to temperature, precipitation, drought, floods, heatwaves, cyclones and greenhouse-gas emissions are discussed, followed by statistical and machine-learning downscaling and bias correction. Hybrid and physics-informed models are developed in detail, including an advection–diffusion physics-informed neural network and the replacement of sub-grid parameterisations. Explainability methods, including SHAP and integrated gradients, are assessed for their reliability on climate problems. Uncertainty is treated through aleatoric and epistemic decomposition, deep ensembles, Bayesian approximations and proper scoring rules, and evaluation practice is set out with block cross-validation, the anomaly correlation coefficient and extreme-event metrics. The chapter closes with applications to mitigation and adaptation, risks of irresponsible use, and emerging directions. It argues for climate-AI frameworks that judge models jointly by skill, physical consistency, calibrated uncertainty and decision value.

**Keywords:** Climate-AI; hybrid modelling; physics-informed neural networks; explainable AI; uncertainty quantification; climate downscaling; Earth-system prediction

## 1. Introduction

### 1.1 Climate change as a data and modelling problem

Climate change is usually framed as a physical problem, but for the working scientist it is equally a problem of information. The state of the climate system is observed incompletely, by instruments with changing coverage and bias. It is simulated by models whose resolution is too coarse to represent clouds, convection and much of the hydrological cycle explicitly. And it must be projected under scenarios of human activity that are themselves uncertain. Every step from raw measurement to a decision about a sea wall or a power-system investment involves statistical inference. Machine learning is the branch of statistics that scales best to the volume and dimensionality of this information, which is why Rolnick et al. could catalogue applications from electricity systems to climate-model emulation in a single review [1].

### 1.2 Growth of climate observations and computational modelling

The volume of climate data has grown by orders of magnitude. Satellite constellations deliver continuous global fields; reanalyses such as ERA5 reconstruct the atmosphere hourly from 1940 onward on a roughly 31 km grid with 137 vertical levels [2]; and the Coupled Model Intercomparison Project Phase 6 (CMIP6) coordinates dozens of modelling centres running common experiments for the assessment of future change [3]. The resulting archives hold petabytes. They are the raw material from which data-driven models are trained, and they are also the reason such models are needed, because the archives are too large to interrogate with hand-crafted diagnostics alone.

### 1.3 Why conventional approaches alone can be computationally demanding

Numerical models integrate the primitive equations of atmospheric and oceanic motion. Their cost grows steeply with resolution: halving the horizontal grid spacing in three dimensions multiplies the number of grid columns by four, and the Courant–Friedrichs–Lewy stability condition then halves the permissible time step, so the total work rises roughly eightfold before any increase in vertical levels is counted. Producing large ensembles, which are essential for uncertainty quantification, multiplies the cost again. This is the computational bottleneck that learned emulators aim to relieve. Pathak et al. and Lam et al. report data-driven global forecasts that run in seconds to minutes on accelerators once trained [4, 5], and Rasp et al. provide the benchmark framework against which such systems can be compared [6].

### 1.4 Opportunities and limitations of AI

The opportunities are real: speed, the ability to exploit nonlinear multivariate structure, and the possibility of learning processes that physics cannot yet resolve. The limitations are equally real and are the organising theme of this chapter. A neural network has no built-in knowledge of conservation of mass, energy or momentum. It interpolates well within the distribution of its training data and may extrapolate poorly beyond it, which matters acutely for a problem whose defining feature is a changing distribution. Reichstein et al. summarise the position succinctly: deep learning offers major opportunities for Earth-system science but needs to be integrated with process understanding, and interpretability, physical consistency and the scarcity of labelled data are open challenges [7].

### 1.5 Scope of the chapter

Section 2 describes climate data and pre-processing. Section 3 treats classical machine learning and Section 4 deep learning. Section 5 reviews applications to specific hazards, Section 6 downscaling, and Section 7 hybrid and physics-informed approaches. Sections 8 and 9 address explainability and uncertainty, Section 10 evaluation, Section 11 mitigation and adaptation, Section 12 risks and responsible use, and Section 13 emerging directions. Section 14 concludes.

## 2. Climate Data and Computational Foundations

### 2.1 Observational climate datasets

Station records of temperature, precipitation, pressure, humidity and wind form the longest instrumental baseline, in places extending back more than a century. Radiosondes provide vertical profiles of temperature, humidity and wind twice daily at hundreds of sites. Observational networks are unevenly distributed: the Northern Hemisphere midlatitudes are densely sampled, while the tropics, polar regions, high mountains and oceans are sparse. Station data also suffer from instrument changes, relocation and urbanisation, so that homogenisation is a prerequisite for trend analysis. For machine learning, the practical consequence is that training data reflect where people measured, not where the climate varies.

### 2.2 Satellite and remote-sensing observations

Satellites provide near-global coverage of variables such as sea-surface temperature, land-surface temperature, vegetation indices, precipitation estimates, aerosol optical depth, cloud properties and trace-gas columns. Instruments on the MODIS platforms, for example, supply daily and multi-day composites of land and ocean surface properties, and the Copernicus programme provides open Earth-observation data and services. The record length of any individual sensor is short, typically one to two decades for modern products, and retrievals depend on algorithms with their own assumptions. Merging products across sensor generations introduces discontinuities that a learning algorithm may mistake for climate signals.

### 2.3 Reanalysis datasets

Reanalyses assimilate historical observations into a fixed modern forecasting system, producing gridded, physically consistent fields of many variables with no gaps. ERA5, the fifth-generation reanalysis of the European Centre for Medium-Range Weather Forecasts, offers hourly output of atmospheric, land-surface and ocean-wave variables from 1940, with uncertainty information from a lower-resolution ensemble [2]. Other widely used products include NCEP/NCAR Reanalysis 1 and MERRA-2. Reanalysis is the default training target for data-driven weather and climate models, as in WeatherBench [6], FourCastNet [4] and GraphCast [5]. It must nevertheless be recognised that reanalysis is model output constrained by data: variables that are weakly observed, such as precipitation and surface fluxes, inherit the biases of the underlying model. A model trained to reproduce ERA5 is learning ERA5's version of reality.

### 2.4 Climate-model simulations

CMIP6 provides standardised simulations from historical forcing to future scenarios, including the Scenario Model Intercomparison Project that pairs radiative-forcing pathways with socio-economic narratives [3]. Regional downscaling experiments, coordinated for example under CORDEX, supply higher-resolution simulations nested in global models. Climate-model output is attractive for machine learning because it is long, complete and spans scenarios beyond the observed climate. It is also the only source of data on states the real system has not yet visited. Climate-model emulators and foundation models trained on CMIP6 exploit exactly this property [8]. The corresponding danger is that the emulator inherits structural errors of the parent model.

### 2.5 Socio-environmental and emissions data

Many applications require non-climatic predictors or targets: population, land use, energy demand, emissions inventories, crop yields and economic indicators. Future emissions pathways are encoded in the Shared Socioeconomic Pathways (SSPs), which combine narratives of development with radiative-forcing levels and are the scenario framework used in CMIP6 [3]. These data are typically annual or coarser, spatially aggregated by administrative units, and revised retrospectively, so they must be harmonised with gridded climate fields through explicit spatial and temporal aggregation.

### 2.6 Spatial and temporal resolution

Datasets differ by orders of magnitude in resolution: CMIP6 models commonly run at 50 to 250 km, ERA5 at about 31 km, regional models at 10 to 50 km, and convection-permitting models at a few kilometres. The effective resolution, the smallest scale actually resolved, is several grid lengths and therefore coarser than the nominal grid. Temporal resolution ranges from sub-hourly station data to monthly and annual model output. Matching resolutions across predictors and targets is the first design decision in any climate-AI pipeline. When a high-resolution target is predicted from coarse predictors, the problem becomes downscaling (Section 6); when a coarse quantity is predicted from fine inputs, information is aggregated and the problem is better posed.

### 2.7 Data quality check and pre-processing

Pre-processing determines more of a model's behaviour than architecture choice does. The essential steps are as follows. *Quality control* removes physically impossible values and flags suspect ones. *Homogenisation* corrects non-climatic discontinuities. *Regridding* maps fields to a common grid using conservative remapping for flux-like quantities and bilinear interpolation for smooth state variables. *Deseasonalisation* removes the climatological annual cycle, defining the anomaly of variable $x$ at location $s$ and time $t$ as

$$
x'(s,t) = x(s,t) - \bar{x}(s,\tau(t)), \qquad (1)
$$

where $\bar{x}(s,\tau)$ is the climatological mean for calendar day or month $\tau$ over a reference period. *Detrending* removes secular change where the task is to learn variability rather than the trend. *Normalisation* scales each variable by climatological mean and standard deviation so that gradient-based training is well conditioned. All statistics for these steps must be computed from the training period only, otherwise information from the test period leaks into the model and skill is overstated. Because geographic grid cells shrink toward the poles, loss functions should be weighted by $\cos\phi$, where $\phi$ is latitude, so that errors are weighted by area.

## 3. Machine Learning Methods for Climate Analysis

### 3.1 Regression methods

Linear and regularised regression remain the baseline against which more complex models must justify themselves. Ridge regression minimises $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|^2$ and handles collinear predictors, which are ubiquitous in climate data because neighbouring grid cells and related variables are strongly correlated. The lasso replaces the squared penalty with an $\ell_1$ norm and thereby performs variable selection. Regression is used for trend estimation, for statistical post-processing of model output and as the skeleton of many downscaling schemes. A model that cannot beat a well-specified regression on the same predictors has found nothing that regression could not.

### 3.2 Decision tree models and ensemble learning

Tree ensembles handle nonlinearity, interactions and mixed variable types with little tuning. Random forests average many decorrelated trees, each grown on a bootstrap sample with a random subset of predictors at each split, which reduces variance without increasing bias much [9]. Gradient-boosted trees, as implemented in XGBoost, build trees sequentially to correct the residual errors of the ensemble, using second-order gradient information and regularisation, and are frequently the strongest method on tabular data [10]. In climate applications tree ensembles are used for drought and crop-yield prediction, bias correction and attribution of regional change. They cannot extrapolate: predictions are averages of training targets, so a tree model asked about a warmer-than-observed regime returns values bounded by the warmest it has seen.

### 3.3 Support-vector methods

Support-vector regression finds a function within an $\varepsilon$-tube of the targets while maximising margin, with kernels such as the radial basis function encoding similarity. Support-vector machines were widely applied to rainfall and streamflow forecasting in the 2000s. Their cost grows at least quadratically with the number of samples, which limits their use on modern reanalysis-sized datasets, and they have largely been displaced by tree ensembles and neural networks, except on small station-based problems where they remain competitive.

### 3.4 Clustering and unsupervised learning

Unsupervised methods summarise structure without labels. $K$-means partitions daily circulation fields into $K$ weather regimes by minimising within-cluster variance, and the resulting regime frequencies are used to describe how the large-scale circulation, and hence regional weather, is changing. Self-organising maps (SOMs) arrange circulation patterns on a low-dimensional lattice that preserves topology, giving a continuum rather than discrete clusters and making transitions between regimes visible. Because cluster number is a free parameter and regimes are not guaranteed to exist as separated modes in a continuous system, stability under resampling should be checked.

### 3.5 Dimensionality reduction

Empirical orthogonal function (EOF) analysis, equivalent to principal component analysis of the anomaly field, is the classical tool for extracting dominant spatial patterns. Stacking $T$ anomaly maps of $n$ grid points into $\mathbf{X}\in\mathbb{R}^{T\times n}$ with area weights applied, the singular-value decomposition $\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{T}$ gives EOFs as columns of $\mathbf{V}$ and principal-component time series as columns of $\mathbf{U}\boldsymbol{\Sigma}$; the fraction of variance in mode $k$ is $\sigma_k^2/\sum_j\sigma_j^2$. Leading EOFs of sea-surface temperature and sea-level pressure correspond to well-known modes such as the El Niño–Southern Oscillation and the North Atlantic Oscillation. Truncated EOF expansions compress inputs for downstream models, but EOFs are orthogonal by construction, not physical, and the sampling errors on individual modes can be large when adjacent eigenvalues are close.

### 3.6 Feature selection

With thousands of candidate predictors and a few decades of annual or seasonal samples, the risk of selecting spurious predictors is severe. Filter methods rank predictors by correlation or mutual information; wrapper methods search subsets using model performance; embedded methods, including the lasso and tree-based importances, select during training. Selection must occur inside the cross-validation loop, not before it, and should be guided by physical plausibility. A predictor with no plausible mechanism linking it to the target should be treated as suspect however strongly it correlates.

### 3.7 Strengths and limitations of conventional machine learning

Conventional methods are cheap, robust on small samples, often interpretable and easy to validate. Their limits are the need for hand-crafted features, an inability to exploit spatial and temporal structure directly, and, for tree-based and kernel methods, poor extrapolation. They remain the right choice for small tabular problems and are a necessary baseline for any deep model. Section 4 turns to architectures that learn spatial and temporal features directly from gridded data.

## 4. Deep Learning for Climate Prediction

### 4.1 Artificial neural networks

A feed-forward network composes affine maps and nonlinear activations, $\mathbf{h}^{(l)}=\sigma(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)}+\mathbf{b}^{(l)})$, and is trained by stochastic gradient descent on a loss between predictions and targets. Sufficiently wide networks can approximate any continuous function on a compact set, but this universality says nothing about how much data or what inductive bias is needed to learn a particular function. Climate fields have strong structure, namely locality, translation-like regularity on the sphere and temporal continuity, and architectures that encode this structure learn far more efficiently than fully connected networks.

### 4.2 Convolutional neural networks

Convolutional neural networks (CNNs) apply shared local filters, so that a pattern, such as a front or a convective cell, is detected wherever it appears. For a field $\mathbf{X}$ with $C$ channels, a convolutional layer computes $\mathbf{Y}_{k}=\sigma\big(\sum_{c=1}^{C}\mathbf{W}_{k,c}\ast\mathbf{X}_{c}+b_k\big)$. Stacked layers, pooling and skip connections build hierarchical features from local gradients to synoptic patterns. On the sphere, plain convolutions on a latitude–longitude grid distort near the poles and need periodic padding in longitude; spherical and graph-based variants remove this distortion. CNNs underlie most downscaling and nowcasting networks, including super-resolution models for climate fields [11].

### 4.3 Recurrent neural networks

Recurrent networks process sequences by carrying a hidden state, $\mathbf{h}_t=\sigma(\mathbf{W}_h\mathbf{h}_{t-1}+\mathbf{W}_x\mathbf{x}_t+\mathbf{b})$. They suit climate time series whose present depends on the past, such as soil moisture memory or ocean heat content. Their weakness is that gradients propagated through many steps either vanish or explode, so simple recurrent networks cannot learn dependencies across long lags.

### 4.4 LSTM and GRU architectures

Long short-term memory (LSTM) cells address the gradient problem with a cell state $\mathbf{c}_t$ regulated by gates [12]:

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f), \quad
\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i),\\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c),\quad
\mathbf{c}_t = \mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t,\\
\mathbf{o}_t &= \sigma(\mathbf{W}_o[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o),\quad
\mathbf{h}_t = \mathbf{o}_t\odot\tanh(\mathbf{c}_t),
\end{aligned} \qquad (2)
$$

where $\mathbf{f}_t$, $\mathbf{i}_t$ and $\mathbf{o}_t$ are forget, input and output gates and $\odot$ is element-wise multiplication. The additive update of $\mathbf{c}_t$ lets gradients flow across many steps when the forget gate is near one. The gated recurrent unit (GRU) merges cell and hidden state and uses two gates, giving fewer parameters at similar accuracy in many tasks. In climate applications LSTMs encode temporal memory in streamflow, soil moisture, sea-surface temperature and index forecasting.

### 4.5 Autoencoders

An autoencoder learns an encoder $f_\phi$ and decoder $g_\theta$ minimising reconstruction error $\|\mathbf{x}-g_\theta(f_\phi(\mathbf{x}))\|^2$ through a low-dimensional bottleneck. It is a nonlinear generalisation of EOF analysis (Section 3.5): with linear activations the optimal bottleneck spans the leading EOF subspace. Autoencoders are used for compressing fields, for anomaly detection by reconstruction error, and as the latent space in which a forecast model operates, which reduces cost when the model predicts latent states and decodes them to the physical grid. Variational autoencoders add a probabilistic latent space useful for generating ensembles.

### 4.6 Attention mechanisms and transformers

Attention computes, for each position, a weighted sum of values with weights set by query–key similarity [13]:

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{d_k}}\right)\mathbf{V}. \qquad (3)
$$

Because every position can attend to every other, transformers capture long-range dependence, such as teleconnections between the tropical Pacific and mid-latitude circulation, without the stepwise propagation of convolution or recurrence. Vision transformers split an image into patches, embed each as a token and apply attention across tokens [14]; for climate fields, patches become tiles of a gridded multivariate state. The price of global attention is cost quadratic in the number of tokens, which motivates patch embedding and efficient variants. FourCastNet addresses it with adaptive Fourier neural operators that mix tokens in the frequency domain at lower cost [4]. Pre-trained transformer backbones are the basis of recent foundation models for weather and climate [8].

### 4.7 Spatiotemporal deep-learning models

Climate prediction requires models that treat space and time jointly. The convolutional LSTM (ConvLSTM) replaces the matrix products of Equation (2) with convolutions, so that the cell and hidden states are spatial fields and the gates respond to local spatial context; it was introduced for precipitation nowcasting from radar sequences and outperformed fully connected LSTMs on that task [15]. PredRNN extends the idea by passing memory both across time and up through layers. Transformer-based models operate on tokenised fields and relate positions at any distance. Graph neural networks represent the globe as a mesh, with message passing along edges, which avoids the polar distortion of regular grids; GraphCast uses an icosahedral multi-mesh encoder–processor–decoder to produce 10-day global forecasts at 0.25° resolution [5].

A hybrid architecture combines the strengths of convolutional and attention components, as shown in Figure 1. A ConvLSTM encoder captures local spatiotemporal dynamics from the input sequence; its feature maps are tokenised into patches and processed by a transformer encoder that captures global interactions; a decoder reconstructs the target field, and a probabilistic head outputs both a mean and a variance to support uncertainty estimation.

![Figure 1. Hybrid ConvLSTM–Transformer architecture for gridded climate prediction. Local spatiotemporal features are extracted by a ConvLSTM encoder, global teleconnections are modelled by a transformer encoder over patch tokens, and a decoder with a probabilistic head outputs the predictive mean and variance.](figures/Figure1_ConvLSTM_Transformer.png)

## 5. AI Applications in Climate Prediction

### 5.1 Temperature prediction

Near-surface and upper-air temperature are the most predictable variables because they are smooth in space and strongly constrained by large-scale dynamics. Global data-driven models trained on ERA5 now produce multi-day temperature forecasts that compete with operational numerical systems [4, 5]. WeatherBench defines the standard evaluation, in which 500 hPa geopotential height and 850 hPa temperature are forecast at lead times of several days and scored by latitude-weighted root-mean-square error [6]. At seasonal to decadal scales, skill derives from slowly varying boundary conditions, chiefly ocean temperature, and recurrent or convolutional networks are used to exploit that memory. Whatever the horizon, a forecast must be compared with the right baseline: persistence and climatology at short leads, and damped persistence or linear inverse models at seasonal leads.

### 5.2 Precipitation and rainfall forecasting

Precipitation is intermittent, skewed and spatially discontinuous, so squared-error losses applied to it produce blurred forecasts that underestimate extremes. ConvLSTM was developed for radar-based nowcasting over a few hours and demonstrated gains over optical-flow extrapolation on that task [15]. At longer leads, global models learn precipitation less reliably than state variables because precipitation is a diagnosed, parameterised quantity in the reanalysis they imitate [2, 6]. Remedies include zero-inflated or gamma likelihoods, quantile and distributional losses, and adversarial training, all aimed at preserving the sharpness and tail of the distribution.

### 5.3 Drought prediction

Drought is a slow, cumulative anomaly of the water balance, and is quantified through indices. The Standardised Precipitation Index (SPI) fits a probability distribution, typically a gamma, to accumulated precipitation over a chosen window and transforms the cumulative probability to a standard normal variate. The Standardised Precipitation–Evapotranspiration Index (SPEI) applies the same procedure to the climatic water balance $D=P-\mathrm{PET}$, which includes the temperature-driven evaporative demand and therefore responds to warming [16]. Because indices are autocorrelated and driven by accumulated anomalies, tree ensembles and LSTMs given antecedent index values, soil-moisture and teleconnection predictors perform well at one-to-few-month leads. Skill beyond that depends largely on the predictability of the driving ocean state.

### 5.4 Flood and hydrological extremes

Flood prediction links meteorological forcing to catchment response. Recurrent networks trained on rainfall–runoff data have been shown to perform strongly in streamflow prediction, in part because the LSTM cell state acts as a learnable store of snow and soil-water memory [12]. The difficulty is extremes: floods are rare, so training sets contain few examples, and the largest events lie outside the observed range. Hybrid approaches embed a conceptual water balance inside the network, or use the network to predict parameters of a hydrological model, trading some flexibility for plausible behaviour at high flows.

### 5.5 Heatwave prediction

Heatwaves are defined by thresholds on temperature persistence, for instance exceedance of a local percentile for several consecutive days. They are therefore a classification or event-probability problem built on a continuous variable. Predictability comes from persistent anticyclonic blocking, soil-moisture deficits that amplify surface heating, and sea-surface temperature patterns. Models that take soil moisture and 500 hPa geopotential anomalies as inputs and predict exceedance probability can provide advance warning at lead times of days to weeks, and the main evaluation concern is calibration of those probabilities and the balance between missed events and false alarms.

### 5.6 Cyclones and severe weather

Tropical cyclone track forecasting has benefited from learned global models. GraphCast, evaluated on tropical cyclone tracks, reports lower track error than a leading operational model across lead times [5]. Intensity is harder, since it depends on inner-core processes that 0.25° grids do not resolve and on rapid intensification, a notoriously sensitive phenomenon. Learned detectors of cyclones and atmospheric rivers in reanalysis and climate-model output provide census statistics of how events change under warming, though detection thresholds must be consistent between the datasets compared.

### 5.7 Greenhouse-gas and carbon-emission prediction

Predicting concentrations and emissions combines physical and socio-economic data. Atmospheric concentration depends on emissions, land and ocean sinks and transport, while emissions depend on energy demand, fuel mix and policy. Machine-learning models are used to nowcast sectoral emissions from activity data, to estimate fluxes from satellite column retrievals, and to forecast electricity demand and carbon intensity for grid operators [1]. Long-horizon emissions projections are better treated as scenario analysis, using the SSP framework [3], than as statistical extrapolation, because policy and technology shifts break the historical relationships a model has learned.

Table 1 summarises the main hazard types, the targets and metrics that govern their evaluation, representative model classes, and the principal obstacle to skill. No numerical skill scores are compared across rows, because the studies use different datasets, domains and metrics, and quoting them side by side would imply a comparability that does not exist.

**Table 1. Prediction tasks, evaluation metrics, representative approaches and principal difficulties across climate hazard types.**

| Hazard / variable | Typical target | Primary metrics | Representative model classes | Principal difficulty | Sources |
|---|---|---|---|---|---|
| Temperature | Gridded state variable, daily to seasonal | Latitude-weighted RMSE, ACC | Graph and Fourier neural operators, transformers | Beating strong numerical baselines at short lead; weak signal at long lead | [4–6] |
| Precipitation | Accumulation or intensity field, hours to days | RMSE, threat score, CRPS | ConvLSTM, generative and distributional models | Intermittency; blurring of extremes; reanalysis bias | [2, 15] |
| Drought | SPI / SPEI at 1–12 month scales | RMSE, correlation, categorical hit rates | Random forest, gradient boosting, LSTM | Dependence on predictable ocean forcing | [9, 10, 16] |
| Floods | Streamflow, discharge exceedance | NSE-type efficiency, peak-flow error, recall | LSTM, hybrid conceptual–neural models | Rare extremes outside training range | [12] |
| Heatwaves | Binary or probabilistic event occurrence | Brier score, reliability, false-alarm ratio | Gradient boosting, CNN with soil-moisture inputs | Calibration of event probabilities | [10, 17] |
| Cyclones | Track and intensity | Track error, intensity error | Graph neural networks, detection CNNs | Intensity and rapid intensification | [5] |
| Greenhouse-gas / emissions | Concentration, sectoral emissions, carbon intensity | MAPE, RMSE | Gradient boosting, LSTM, scenario models | Non-stationarity from policy and technology | [1, 3] |

Two points emerge from Table 1. First, the metric must match the decision: RMSE rewards smooth forecasts that miss extremes, whereas threat scores and the continuous ranked probability score reward discrimination and calibration. Second, difficulty follows predictability, not model sophistication. Where the physical signal is weak or the target is rare, a more elaborate network does not create information that is not in the data.

## 6. Climate Downscaling and Regional Prediction

### 6.1 Global versus regional climate information

Adaptation decisions are local, but global models resolve scales of tens to hundreds of kilometres. Mountains, coastlines, lakes and cities modulate temperature and precipitation at scales below the grid, and extreme events are controlled by processes that coarse models parameterise. Downscaling bridges the gap, either dynamically, by running a regional model nested in the global one, or statistically, by learning a mapping from coarse predictors to fine-scale predictands.

### 6.2 Statistical downscaling

Classical statistical downscaling assumes that local weather is conditioned by the large-scale state. It takes three forms: regression relating predictors to local variables, weather-typing approaches that resample local weather conditional on circulation regime, and stochastic weather generators. All rest on the stationarity assumption that the relationship estimated from historical data remains valid under future forcing. This assumption cannot be verified with observations alone and is the main conceptual risk of the approach.

### 6.3 Machine-learning-based downscaling

Machine learning replaces linear transfer functions with flexible nonlinear ones. A downscaling network maps a coarse field $\mathbf{X}_c$ and static high-resolution covariates $\mathbf{S}$ (orography, land-sea mask, land cover) to a fine field $\hat{\mathbf{Y}}_f$:

$$
\hat{\mathbf{Y}}_f = f_\theta(\mathbf{X}_c,\mathbf{S}), \qquad \theta^{\ast}=\arg\min_\theta \; \mathbb{E}\big[\ell(\hat{\mathbf{Y}}_f,\mathbf{Y}_f)\big]. \qquad (4)
$$

The choice of loss determines the character of the output. A mean-squared-error loss yields the conditional mean, which is smooth, and underestimates variance and extremes. Likelihood-based, adversarial or diffusion-style losses sample from the conditional distribution and restore realistic variability, at the price of outputs that are not unique for a given input.

### 6.4 Spatial super-resolution approaches

Super-resolution treats a coarse field as a low-resolution image to be upsampled. DeepSD applied stacked super-resolution CNNs to downscale precipitation over the United States, learning successive resolution increases with topography as auxiliary input and comparing favourably with several statistical downscaling benchmarks [11]. Generative adversarial networks add a discriminator that penalises unrealistic texture and so produce sharper fields. Figure 2 shows the processing chain: coarse predictors and static covariates enter a convolutional encoder, an upsampling decoder generates the fine field, and a bias-correction stage aligns its distribution with observations.

![Figure 2. Machine-learning downscaling pipeline. Coarse-resolution predictors and static high-resolution covariates are mapped by a super-resolution network to a fine-resolution field, which is then bias-corrected by quantile mapping and evaluated against station and gridded observations.](figures/Figure2_Downscaling_Pipeline.png)

### 6.5 Bias correction

Climate-model output systematically departs from observations, and downstream impact models are sensitive to such departures. Quantile mapping corrects the distribution by transforming model values through the mapping

$$
x_{\text{corr}} = F_{o}^{-1}\big(F_{m}(x_{m})\big), \qquad (5)
$$

where $F_m$ is the cumulative distribution function of the model in the reference period and $F_o^{-1}$ the quantile function of observations. Cannon et al. showed that simple quantile mapping can distort the climate-change signal in quantiles and extremes, since it forces the corrected historical distribution to match observations whatever the projected change, and proposed trend-preserving variants [18]. Bias correction should therefore be applied with awareness that it modifies not only the mean state but the projected change itself.

### 6.6 Regional transferability

A model trained on one region is often applied to another with sparser data. Transferability depends on whether the physical regime is shared: a network trained over temperate mid-latitudes may fail in the tropics, where convective rather than frontal precipitation dominates. Common practice is to fine-tune a pre-trained model on a small regional dataset, to include static covariates that let the network condition on local geography, or to train on pooled multi-region data. Transfer should be tested by withholding entire regions, not random grid cells (Section 10.2).

### 6.7 Limitations of AI-generated high-resolution climate information

A downscaled field has more pixels but not necessarily more information. The network can only add fine-scale structure that is predictable from the coarse input and static covariates; beyond that it adds plausible texture. Users must distinguish *deterministic* detail, such as orographic enhancement of rainfall, from *stochastic* detail, which is statistically realistic but not tied to any particular event. Further limits are stationarity, since the learned relationship may change under warming, and physical consistency, since independently downscaled variables need not satisfy the relations that bind them. High resolution should never be mistaken for high confidence.

## 7. Hybrid and Physics-Informed Climate AI

### 7.1 Why purely data-driven prediction can fail

Free-running neural emulators are unconstrained by physics. Small violations of mass or energy conservation can accumulate over thousands of autoregressive steps and produce drift, spurious trends or unphysical states such as negative humidity. Models trained for short-lead accuracy may produce physically implausible fields on long rollouts even when each step looks accurate. The governing primitive equations make the constraints explicit. For an atmosphere in hydrostatic balance on a sphere, mass continuity and momentum conservation read

$$
\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0,\qquad
\frac{D\mathbf{u}}{Dt}=-\frac{1}{\rho}\nabla p-2\boldsymbol{\Omega}\times\mathbf{u}+\mathbf{g}+\mathbf{F}, \qquad (6)
$$

together with the thermodynamic energy equation $c_p\,DT/Dt-\alpha\,Dp/Dt=Q$ and the equation of state, where $\boldsymbol{\Omega}$ is Earth's rotation vector, $\mathbf{g}$ gravity, $\mathbf{F}$ friction and $Q$ diabatic heating. A network that ignores these relations has no reason to satisfy them.

### 7.2 Incorporating physical constraints

Constraints may be *soft* or *hard*. A soft constraint adds a penalty to the loss, for example the squared violation of column-integrated moisture or energy balance; it is easy to implement but the constraint is satisfied only approximately and only on the training distribution. A hard constraint builds satisfaction into the architecture, for instance by predicting a stream function so that the resulting velocity is divergence-free by construction, or by adding a final layer that rescales outputs to enforce a conserved budget exactly. Hard constraints guarantee the property but may restrict expressiveness, and designing them demands knowledge of which invariants matter [7, 19].

### 7.3 Hybrid numerical–AI models

Hybrid models keep the dynamical core of a numerical model and replace uncertain components with learned ones. The most developed example is the sub-grid parameterisation. Rasp et al. trained a deep neural network on high-resolution simulations with explicitly resolved convection to predict sub-grid heating and moistening tendencies from the coarse-grained state, and showed that the network, embedded in a coarse model, reproduced the mean climate and aspects of variability at a fraction of the cost [20]. The principle is to learn the closure term $\mathbf{S}_\theta$ in

$$
\frac{\partial \bar{\mathbf{x}}}{\partial t}=\mathcal{F}_{\text{dyn}}(\bar{\mathbf{x}})+\mathbf{S}_\theta(\bar{\mathbf{x}}), \qquad (7)
$$

where $\bar{\mathbf{x}}$ is the resolved state, $\mathcal{F}_{\text{dyn}}$ the numerical dynamical core and $\mathbf{S}_\theta$ the learned sub-grid tendency. The chief difficulty is coupled stability: a closure that is accurate offline can destabilise the model when its errors feed back through the dynamics, and a learned scheme trained on one climate may misrepresent another [7, 20].

### 7.4 Physics-informed neural networks

Physics-informed neural networks (PINNs) enforce differential equations through the training loss [21]. Consider transport of a scalar $C$, for example a pollutant or tracer, by a known wind field $\mathbf{u}$:

$$
\frac{\partial C}{\partial t}+\mathbf{u}\cdot\nabla C=D\nabla^{2}C+S. \qquad (8)
$$

Let $\hat{C}(x,y,z,t;\theta)$ be a network approximation. The PDE residual is

$$
R(x,y,z,t;\theta)=\frac{\partial \hat{C}}{\partial t}+\mathbf{u}\cdot\nabla\hat{C}-D\nabla^{2}\hat{C}-S, \qquad (9)
$$

evaluated by automatic differentiation at $N_r$ collocation points. The training loss combines data misfit, residual and boundary or initial-condition terms:

$$
\mathcal{L}(\theta)=\frac{w_d}{N_d}\sum_{i=1}^{N_d}\big|\hat{C}(\mathbf{x}_i,t_i;\theta)-C_i\big|^{2}+\frac{w_p}{N_r}\sum_{j=1}^{N_r}\big|R(\mathbf{x}_j,t_j;\theta)\big|^{2}+\frac{w_b}{N_b}\sum_{l=1}^{N_b}\big|\mathcal{B}[\hat{C}](\mathbf{x}_l,t_l)-g_l\big|^{2}. \qquad (10)
$$

Figure 3 shows how the terms are assembled. The same machinery solves inverse problems: if $D$ or $S$ is unknown it can be added to the trainable parameters and inferred from sparse concentration measurements [21]. In practice the weights $w_d$, $w_p$ and $w_b$ strongly influence convergence, and PINNs become difficult to train for stiff, chaotic or highly multiscale problems, which includes most atmospheric flows at full complexity [19].

![Figure 3. Physics-informed neural network (PINN) coupling flowchart for advection–diffusion. Network inputs are space–time coordinates; automatic differentiation yields the PDE residual; data, physics and boundary losses are combined and minimised to update the network parameters.](figures/Figure3_PINN_Flowchart.png)

### 7.5 AI-assisted numerical weather and climate models

Learned global weather models are the most visible application. FourCastNet uses adaptive Fourier neural operators to forecast global fields at 0.25° resolution orders of magnitude faster than a conventional ensemble [4]. GraphCast uses a graph neural network on a multi-scale icosahedral mesh and was reported to outperform the operational high-resolution deterministic system of the European Centre for Medium-Range Weather Forecasts on a large majority of verification targets for 10-day forecasts [5]. Both are trained on ERA5 and therefore reproduce the statistics of that reanalysis [2]. Two cautions apply. First, these models are validated at weather time scales, where initial-value predictability matters; their behaviour over decades under changing forcing is a separate and unresolved question. Second, they provide a single deterministic trajectory unless extended with perturbations or generative components, so ensemble and probabilistic capability must be added deliberately.

### 7.6 Surrogate modelling

Surrogates, or emulators, approximate the input–output map of an expensive model. A climate emulator can learn the response of regional temperature and precipitation to forcing scenarios from a modest number of CMIP6 runs and then evaluate arbitrary scenarios in seconds [3, 8]. Emulators are valuable for integrated-assessment and impact studies, which require thousands of scenario evaluations. Their validity is bounded by the range of forcings in the training ensemble, and extrapolation beyond that range must be flagged.

### 7.7 Data assimilation with machine learning

Machine learning enters data assimilation in several ways: as a fast surrogate for the forecast model inside ensemble methods, as a learned observation operator mapping model state to satellite radiances, as a bias-correction scheme for model error, and as a replacement for parts of the analysis step. Because reanalyses such as ERA5 are themselves products of assimilation [2], improved assimilation improves the training targets of every data-driven model. This connection suggests tight coupling of learning and assimilation, in which learned models forecast and are corrected by observations in a continual loop.

## 8. Explainable AI for Climate Science

### 8.1 Why interpretability matters

A forecast that is accurate but unexplained invites distrust from practitioners and, more seriously, hides failure. A network may achieve high skill by exploiting an artefact, such as a satellite-era discontinuity or a spurious correlation, and no accuracy metric on in-distribution data will reveal it. Explainability methods are tools for scientific scrutiny: they allow a researcher to ask whether the model's reasoning agrees with known physics, and to discover when it does not [7].

### 8.2 Feature importance

Global importance ranks predictors by their overall contribution. Permutation importance measures the drop in skill when a predictor's values are shuffled; tree-based importances measure impurity reduction. Both are easy to compute and both mislead when predictors are correlated, since a shuffled predictor may be partly replaced by its correlated neighbours, spreading or hiding importance. In gridded climate data, where adjacent cells and related variables are strongly correlated, importance should be computed for physically meaningful groups of features.

### 8.3 SHAP and related approaches

SHAP assigns each input a contribution derived from Shapley values of cooperative game theory, giving a unique additive attribution that satisfies local accuracy, missingness and consistency [22]. For a model $f$ and input $\mathbf{x}$ with feature set $F$,

$$
f(\mathbf{x})=\phi_0+\sum_{i\in F}\phi_i,\qquad
\phi_i=\sum_{S\subseteq F\setminus\{i\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[f_{S\cup\{i\}}(\mathbf{x}_{S\cup\{i\}})-f_S(\mathbf{x}_S)\Big], \qquad (11)
$$

where $\phi_0$ is the baseline expectation. KernelSHAP estimates the values by weighted linear regression over sampled feature subsets and is model-agnostic. DeepSHAP combines DeepLIFT-style back-propagation with Shapley values for neural networks, and TreeSHAP computes exact values for tree ensembles efficiently [22]. Integrated gradients attribute the prediction by integrating the gradient along a straight path from a baseline $\mathbf{x}'$ to the input [23]:

$$
\operatorname{IG}_i(\mathbf{x})=(x_i-x_i')\int_{0}^{1}\frac{\partial f\big(\mathbf{x}'+\alpha(\mathbf{x}-\mathbf{x}')\big)}{\partial x_i}\,d\alpha. \qquad (12)
$$

IG satisfies completeness, in that attributions sum to $f(\mathbf{x})-f(\mathbf{x}')$, and sensitivity. Both families depend on a baseline: the choice of reference climatology or zero field changes what an attribution means.

### 8.4 Local versus global explanations

Local explanations account for one prediction, for example why a model forecast extreme rainfall for a particular storm. Global explanations describe average behaviour across a dataset. The two are linked, since a global importance can be computed as the mean absolute local attribution, but they answer different questions. In climate science local explanations support case studies of individual extremes, while global ones support statements about which processes the model has learned to rely on.

### 8.5 Physical interpretation of learned relationships

An attribution map is a hypothesis, not a finding. If a model forecasting European heatwaves assigns high attribution to soil-moisture deficit and blocking-related geopotential anomalies, the result is consistent with established mechanisms and raises confidence. If it assigns it to a variable with no plausible pathway, the explanation signals either a discovery to be tested or a shortcut the model is exploiting. Composite maps of attribution for a class of events, compared with composite physical anomalies, provide a systematic check.

### 8.6 Avoiding misleading explanations

Attribution methods can be unreliable. They depend on baselines, can be unstable to small input changes, and conflate correlation with causation because they describe the model, not the world. Correlated predictors can split credit arbitrarily, and different methods applied to one model can disagree. Good practice is to use several methods, to check that attributions pass sanity tests such as degrading when model weights are randomised, to test explanations by intervention, and to avoid causal language. Table 2 compares the main methods on the criteria that matter in climate applications.

**Table 2. Comparative evaluation of explainability methods for climate analysis.**

| Method | Scope | Model access needed | Handles correlated features | Main climate use | Principal weakness | Sources |
|---|---|---|---|---|---|---|
| Permutation importance | Global | Black box | Poorly; credit is split or hidden | Ranking predictors for seasonal forecasts | Unrealistic inputs when features are correlated | [9] |
| Tree-based impurity importance | Global | Tree ensembles | Poorly; biased toward high-cardinality features | Quick screening in drought and yield models | Bias; no local explanation | [9, 10] |
| KernelSHAP | Local and global | Black box | Depends on background distribution | Explaining individual extreme-event forecasts | Computational cost; sampling variance | [22] |
| DeepSHAP | Local and global | Neural network internals | Approximate | Attribution for CNN and recurrent models | Approximation of Shapley values | [22] |
| TreeSHAP | Local and global | Tree ensembles | Exact for the model, but credit still shared | Attribution in gradient-boosted climate models | Describes model, not causal effect | [10, 22] |
| Integrated gradients | Local (pixel-level) | Differentiable network | Not addressed; baseline-dependent | Attribution maps over gridded fields | Sensitivity to baseline choice | [23] |

Table 2 shows that no method is universally superior. Model-agnostic methods are flexible but expensive, model-specific methods are efficient but tied to an architecture, and all inherit the limitations of the model being explained. For gridded fields, pixel-level methods such as integrated gradients give spatial maps that can be compared with physical composites, whereas SHAP applied to aggregated predictors gives the ranking of drivers that a policy audience can read.

## 9. Uncertainty and Reliability

### 9.1 Sources of uncertainty in climate prediction

Climate prediction is uncertain for three reasons that must be kept distinct. *Internal variability* arises from the chaotic nature of the system, so that nearly identical initial states diverge. *Scenario uncertainty* arises from unknown future emissions [3]. *Model uncertainty* arises from imperfect representation of processes. In the language of statistical learning, uncertainty is split into *aleatoric* uncertainty, the irreducible noise in the data-generating process, and *epistemic* uncertainty, which reflects limited knowledge of the model and shrinks with more data. For a predictive distribution with mean $\mu_\theta(\mathbf{x})$ and variance $\sigma_\theta^2(\mathbf{x})$, and an ensemble of $M$ models, the total predictive variance decomposes as

$$
\operatorname{Var}[y\mid\mathbf{x}]=\underbrace{\frac{1}{M}\sum_{m=1}^{M}\sigma_m^{2}(\mathbf{x})}_{\text{aleatoric}}+\underbrace{\frac{1}{M}\sum_{m=1}^{M}\big(\mu_m(\mathbf{x})-\bar{\mu}(\mathbf{x})\big)^{2}}_{\text{epistemic}}, \qquad (13)
$$

where $\bar{\mu}=M^{-1}\sum_m\mu_m$ [24]. The distinction has practical consequences: epistemic uncertainty flags inputs unlike the training data and can be reduced by collecting more of them, whereas aleatoric uncertainty sets the floor on achievable skill.

### 9.2 Data uncertainty

Observations and reanalyses carry measurement error, representativeness error and, for reanalysis, model-dependent bias. ERA5 provides an ensemble-based uncertainty estimate precisely to support users who need to propagate this error [2]. Training on a single deterministic product treats it as truth, so the model's skill is conditional on that product. Where possible, training and evaluation should use several independent products and report the spread.

### 9.3 Model uncertainty

Model uncertainty includes structural error in numerical models, evident in the spread across CMIP6 members [3], and for learned models it includes the arbitrary choices of architecture, hyperparameters, random initialisation and training sample. Retraining the same architecture from different seeds can give different predictions in regions of sparse data, and this seed-to-seed variation is a direct, if partial, measure of epistemic uncertainty.

### 9.4 Predictive uncertainty

Predictive uncertainty is what the user ultimately needs: a distribution over the quantity of interest, conditional on the inputs. For a heteroscedastic Gaussian head, training minimises the negative log-likelihood

$$
\mathcal{L}_{\text{NLL}}=\frac{1}{N}\sum_{i=1}^{N}\left[\frac{\big(y_i-\mu_\theta(\mathbf{x}_i)\big)^2}{2\sigma_\theta^2(\mathbf{x}_i)}+\frac{1}{2}\ln\sigma_\theta^2(\mathbf{x}_i)\right], \qquad (14)
$$

which lets the network predict larger variance where it is less certain. A Gaussian is a poor description of precipitation and many extremes, so skewed or mixture distributions, or direct quantile prediction, are frequently needed.

### 9.5 Ensemble approaches

Deep ensembles train $M$ networks from different random initialisations, optionally on resampled data, and combine their predictions as a uniformly weighted mixture; they have been shown to give well-calibrated uncertainty that often matches or beats approximate Bayesian methods, at the cost of $M$ trainings [24]. In weather forecasting, ensembles are also generated by perturbing initial conditions and, for learned models, by injecting noise during rollout. Ensemble spread should be tested against actual error: a well-calibrated ensemble has a spread-skill ratio near one, and ensembles of weather models are commonly under-dispersive.

### 9.6 Probabilistic forecasting

Bayesian neural networks place a distribution over weights and compute the posterior predictive by integrating over it, which is intractable for deep networks and is approximated variationally or by sampling. Monte Carlo dropout interprets dropout at test time as approximate variational inference, so that repeated stochastic forward passes yield a predictive distribution at little extra cost [25]. Its estimates can be overconfident and depend on the dropout rate, which limits its reliability. Evidential deep learning instead predicts the parameters of a higher-order distribution over the likelihood parameters in a single pass; it is cheap, but its uncertainty estimates have been found to be sensitive to regularisation and are not guaranteed to be calibrated. For climate applications, the safest course is to treat any single method's uncertainty as provisional until verified on independent data.

### 9.7 Calibration and reliability assessment

A probabilistic forecast is *reliable* if events forecast with probability $p$ occur with frequency $p$. A reliability diagram bins forecast probabilities and plots the observed frequency in each bin against the bin's mean forecast probability, and points on the diagonal indicate perfect calibration. Figure 4 shows the construction with illustrative data. Proper scoring rules assess calibration and sharpness together; the continuous ranked probability score (CRPS) is the integrated squared difference between the forecast cumulative distribution function $F$ and the step function at the observation $y$ [17]:

$$
\operatorname{CRPS}(F,y)=\int_{-\infty}^{\infty}\big(F(z)-\mathbb{1}\{z\ge y\}\big)^{2}\,dz, \qquad (15)
$$

and for an ensemble of $M$ members $\{x_m\}$ can be estimated by $\frac{1}{M}\sum_m|x_m-y|-\frac{1}{2M^{2}}\sum_{m,n}|x_m-x_n|$. CRPS reduces to the absolute error for a deterministic forecast, so it permits direct comparison of deterministic and probabilistic systems. Because it is a proper score, it is minimised in expectation only by the true predictive distribution [17].

![Figure 4. Reliability diagram for a probabilistic event forecast (illustrative synthetic data). The diagonal denotes perfect reliability; the over-confident forecast departs from it at both ends, and recalibration moves it toward the diagonal. The inset histogram shows forecast sharpness.](figures/Figure4_Reliability_Diagram.png)

## 10. Evaluating AI-Based Climate Models

### 10.1 Training, validation and independent testing

Data are split into training, validation and test sets, the first for fitting, the second for choosing hyperparameters and stopping, and the third, touched once, for the final estimate. The test set must be independent in the sense that matters for the application. Splitting climate data randomly in time gives neighbouring days to training and test sets, and because consecutive days are strongly autocorrelated the test set then reproduces the training set; skill estimated this way is optimistic. Splits should be contiguous blocks, ideally by year.

### 10.2 Spatial and temporal cross-validation

Block cross-validation withholds contiguous blocks, in time, in space or both, with buffers between training and test blocks wider than the autocorrelation length. Spatial block cross-validation withholds whole regions and estimates how a model will perform where it has no data, which is the question that matters for transferability (Section 6.6). Leave-one-year-out and leave-one-model-out schemes, the latter withholding an entire climate model, test generalisation across years and across model structures. The block size should be set from the empirical variogram or autocorrelation function of the residuals.

### 10.3 Regression and classification metrics

Regression is scored by root-mean-square error, $\mathrm{RMSE}=\sqrt{N^{-1}\sum_i w_i(f_i-o_i)^2}$ with area weights $w_i\propto\cos\phi_i$, and by mean absolute error, which is less sensitive to outliers. Event forecasts are scored from the contingency table of hits $a$, false alarms $b$, misses $c$ and correct negatives $d$: the probability of detection is $a/(a+c)$, the false-alarm ratio $b/(a+b)$, and the threat score (critical success index) $a/(a+b+c)$. For rare events, accuracy $(a+d)/N$ is uninformative because the correct-negative count dominates. Probabilistic event forecasts are scored with the Brier score, the mean squared difference between forecast probability and the binary outcome, which is a proper score [17].

### 10.4 Forecast skill

Skill is always relative to a reference. The skill score $\mathrm{SS}=1-\mathrm{MSE}_{\text{forecast}}/\mathrm{MSE}_{\text{ref}}$ measures improvement over climatology or persistence. The anomaly correlation coefficient (ACC) measures the pattern agreement between forecast and observed anomalies relative to climatology:

$$
\mathrm{ACC}=\frac{\sum_i w_i\,(f_i-\bar{f})(o_i-\bar{o})}{\sqrt{\sum_i w_i\,(f_i-\bar{f})^{2}\;\sum_i w_i\,(o_i-\bar{o})^{2}}}, \qquad (16)
$$

where $f_i$ and $o_i$ are forecast and observed anomalies, with anomalies defined relative to the same climatology as in Equation (1) (so that $\bar f$ and $\bar o$ are area-weighted spatial means of the anomalies), and $w_i$ are area weights. ACC is insensitive to systematic bias in amplitude, so it must be reported alongside RMSE. A common practical threshold is an ACC of 0.6, below which a forecast is usually considered to have lost useful skill relative to climatology, and the lead time at which ACC drops below it is a headline measure in the weather-forecast benchmarks used by WeatherBench and GraphCast [5, 6].

### 10.5 Extreme-event performance

Average metrics hide tail behaviour. A model can achieve low RMSE while systematically underpredicting the largest values. Tail performance should be assessed with metrics conditioned on extremes: error restricted to values above a high percentile, quantile scores at the 95th and 99th percentiles, threat scores at extreme thresholds, and comparison of the upper tail of the predicted and observed distributions using quantile–quantile plots. Extreme value theory can provide return-level estimates against which a model's extrapolation is judged. Because events are rare, confidence intervals from bootstrap resampling in blocks are essential.

### 10.6 Physical consistency

Skill scores do not test whether predictions obey physics. Complementary diagnostics include the global budget of mass, water and energy over long rollouts, the spectral energy distribution (a model that is too smooth will lose energy at small scales), the balance relations such as geostrophic and hydrostatic balance, and the preservation of known correlations between variables. A model whose RMSE is competitive but whose kinetic-energy spectrum collapses at high wavenumbers is blurring, not forecasting [19].

### 10.7 Generalization to unseen climate regimes

The ultimate test for climate applications is performance in a climate unlike the training period. Proxy tests include training on early decades and testing on recent ones, training on a subset of climate models or scenarios and testing on withheld ones, and training on cool-climate regions and testing on warm ones [7]. These tests can reveal whether the model has learned stable physical relationships or period-specific correlations. Perfect-model experiments, in which a climate model is treated as truth, allow evaluation under a warmer climate for which no observations exist, with the caveat that the model's own errors define the truth.

Table 3 collects the metrics and protocols discussed above and indicates the question each answers and its main limitation.

**Table 3. Evaluation metrics and validation protocols for AI-based climate models.**

| Metric / protocol | Question answered | Applies to | Strength | Main limitation | Sources |
|---|---|---|---|---|---|
| Latitude-weighted RMSE | How large are errors, on average? | Continuous fields | Standard, comparable across studies | Rewards smoothing; hides tail errors | [6] |
| ACC (Eq. 16) | Does the forecast anomaly pattern match observations? | Gridded anomaly fields | Removes climatological mean; pattern-oriented | Insensitive to amplitude bias | [5, 6] |
| Threat score / POD / FAR | Are events detected without too many false alarms? | Binary events | Directly tied to warning decisions | Depends on threshold; poor for very rare events | [17] |
| Brier score and reliability diagram | Are forecast probabilities calibrated? | Probabilistic events | Proper score; decomposable | Needs many events to resolve high-probability bins | [17] |
| CRPS (Eq. 15) | How good is the full predictive distribution? | Probabilistic continuous | Proper; reduces to MAE for deterministic forecasts | Dominated by bulk, not tails | [17] |
| Block spatial / temporal CV | Will the model generalise to new regions and years? | All | Honest estimate under autocorrelation | Requires choice of block size | [7] |
| Spectral and budget diagnostics | Is the model physically consistent? | Autoregressive emulators | Detects blurring and drift | Does not measure accuracy | [19] |

Table 3 shows that evaluation is a portfolio, not a single number. Accuracy, calibration, tail behaviour and physical consistency are distinct properties, and a model can excel at one and fail at another, so a credible assessment reports metrics from each row appropriate to the application.

## 11. Climate Change Mitigation and Adaptation

### 11.1 Energy and emissions forecasting

Forecasts of electricity demand, net load and grid carbon intensity underpin demand-side flexibility and storage dispatch. Learned models combine weather forecasts with calendar and consumption data, and the accuracy of the weather input, particularly of temperature and cloud cover, is often the dominant error source [1]. The same forecasts support carbon-aware scheduling of flexible loads.

### 11.2 Renewable-energy prediction

Wind and solar output depend directly on near-surface wind and irradiance, so improved weather prediction translates into improved generation forecasts. Learned global models offer fast, probabilistic input ensembles for such forecasting [4, 5], and statistical post-processing corrects local biases. Because the cost of forecast error is asymmetric, since imbalance penalties and reserve costs differ in sign, quantile and probabilistic forecasts, scored by CRPS or pinball loss, are more useful to operators than point forecasts [17].

### 11.3 Carbon-emission modelling

Machine learning supports bottom-up estimation of emissions from activity data and top-down inversion of fluxes from satellite and ground observations of concentrations. For inversion, a transport model links surface fluxes to observed concentrations, and learned surrogates of the transport model accelerate the estimation. The result is an uncertain estimate that must carry its uncertainty into any policy accounting [1].

### 11.4 Water-resource management

Seasonal streamflow, snowpack and reservoir-inflow forecasts support allocation and flood control. Downscaled precipitation and temperature projections [11, 18] drive hydrological impact models, and drought indices such as SPEI frame the risk [16]. Decision-makers value forecasts with calibrated uncertainty, which allows them to weigh the cost of acting against the probability and severity of the event.

### 11.5 Agricultural adaptation

Crop-yield models driven by seasonal forecasts, soil moisture and satellite vegetation indices guide planting, irrigation and insurance. Tree ensembles are widely used here because they cope with mixed tabular inputs [9, 10]. A distinct need is projecting yields under future climate, where models trained on historical relationships must extrapolate, and this is the setting in which hybrid crop-physiology models are most defensible.

### 11.6 Climate-risk assessment

Risk combines hazard, exposure and vulnerability. AI contributes to the hazard component through improved event statistics, downscaled projections and large synthetic ensembles of extremes, and to exposure and vulnerability through remote-sensing-based mapping of assets and populations. Compound and cascading risks require joint modelling of hazards, for which learned multivariate distributions are useful, but the extreme-value dependence structure they must capture is data-hungry.

### 11.7 Early-warning systems

Early-warning systems translate forecasts into actionable alerts. Their value depends on lead time, on calibrated probabilities that support threshold-based action, and on communication. Fast learned forecasts allow larger ensembles within operational time limits and so better probability estimates [4, 5]. A warning system should be evaluated by decision outcomes, including false-alarm cost and missed-event cost, not forecast skill alone.

## 12. Limitations, Risks and Responsible Climate AI

### 12.1 Data scarcity and geographical imbalance

Observations are densest where wealth is greatest, so models trained on them perform best for regions that are already best served, and may fail for regions most vulnerable to climate impacts [1]. Reanalysis fills gaps but with model-dependent quality in poorly observed regions [2]. Reporting regional performance, not global averages alone, is the minimum safeguard.

### 12.2 Dataset bias

Biases are inherited from the training targets. A model trained on a reanalysis reproduces its precipitation bias; a model trained on a single climate model reproduces its sensitivity [2, 3]. Training on multi-model ensembles reduces dependence on any one structure, but ensembles are not independent because models share code and parameterisations.

### 12.3 Overfitting and spurious correlations

With short records and many predictors, flexible models readily fit noise. Climate time series are autocorrelated, so the effective sample size is much smaller than the number of time steps, and ordinary cross-validation overstates generalisation. Remedies are block validation (Section 10.2), regularisation, physically motivated predictor selection and comparison with simple baselines.

### 12.4 Extrapolation beyond the training climate

Climate change is by definition a shift in the distribution of inputs. A model that has never seen temperatures or moisture states above those in its training record may behave unpredictably, and tree-based methods cannot extrapolate at all (Section 3.2). Physics-informed constraints and hybrid designs retain correct behaviour more reliably out of distribution [7, 19]. Epistemic uncertainty estimates should be used to flag out-of-distribution inputs, with the caveat that they are themselves unreliable far from training data.

### 12.5 Interpretability

Opaque models are difficult to audit. Explanation methods help but are imperfect (Section 8.6) and should be treated as hypotheses to be tested, not as evidence of mechanism [22, 23].

### 12.6 Reproducibility

Deep-learning results depend on random seeds, software versions, hardware and preprocessing details. Reproducibility requires published code, fixed data splits, documented normalisation statistics and reporting of results across multiple seeds with spread, in the manner that benchmarks such as WeatherBench make possible [6].

### 12.7 Computational and energy costs of AI

Training large models consumes substantial energy, and a model that saves compute at inference after a large training cost must be used enough to amortise it. Rolnick et al. note that machine learning has its own emissions that should be weighed against its benefits [1]. Efficient architectures, transfer from pre-trained models [8] and reporting of training compute are practical responses.

### 12.8 Physical plausibility and scientific trust

Trust in a climate model rests on understanding why it works. Data-driven systems can earn trust through long-rollout stability, conservation diagnostics, correct response in perturbation experiments and transparent uncertainty [7, 19]. They should be regarded as complements to, not replacements for, physics-based models until their behaviour under changed forcing has been demonstrated.

## 13. Emerging Directions

### 13.1 Foundation models for Earth-system science

Foundation models are pre-trained on large and diverse data and then fine-tuned for many tasks. ClimaX is a transformer pre-trained on CMIP6 data with variable tokenisation and aggregation, allowing it to handle heterogeneous variables and resolutions and to be fine-tuned for forecasting, downscaling and climate projection [8]. Such models promise reuse and data efficiency, but their generalisation to novel regimes, and the biases they inherit from the pre-training simulations, need systematic study.

### 13.2 Multimodal climate AI

Future systems will fuse gridded fields, satellite imagery, station time series, text reports and socio-economic data. Multimodal fusion raises the alignment problems of Section 2.6 and the risk that a model learns to rely on whichever modality is most convenient, not most informative.

### 13.3 AI-assisted Earth-system modelling

Learned components are entering operational Earth-system models as parameterisations, bias correctors and fast emulators of sub-models [7, 20]. The research priority is stable, conservative coupling that remains valid under climate change.

### 13.4 Physics–AI co-modelling

Beyond hybrids that replace single components, co-modelling treats physical and learned models as partners that exchange information during training and inference, with constraints and differentiable solvers permitting end-to-end optimisation [19, 21]. Differentiable physics enables gradient-based calibration of parameters against observations.

### 13.5 Probabilistic and uncertainty-aware AI

Generative models, including diffusion and score-based approaches, offer a route to large calibrated ensembles at low cost, and could address the under-dispersion of deterministic learned forecasters [17, 24]. Verification with proper scores must accompany them.

### 13.6 AI for compound climate hazards

Compound events, such as concurrent heat and drought or storm surge coinciding with heavy rainfall, are hard to study because their joint probabilities need long records. Large emulator ensembles and learned dependence models offer a means of sampling them, provided that tail dependence is validated [3, 8].

### 13.7 Toward decision-oriented climate intelligence

The aim is to move from variables to decisions: systems that couple calibrated forecasts, impact models and cost–loss analysis to recommend actions, with explanation and uncertainty attached [1]. This requires co-design with the people who use the information.

## 14. Conclusions

AI has become a serious instrument of climate analysis and prediction. Classical learners such as random forests and gradient boosting remain effective for small tabular problems [9, 10]. Convolutional, recurrent and attention-based networks learn spatial and temporal structure from gridded data [12–15], and learned global models now produce weather forecasts at a fraction of the cost of numerical ensembles [4, 5]. Downscaling networks add fine-scale detail to coarse fields, though they add plausible texture as well as information and require careful bias correction [11, 18].

The central conclusion of this chapter is that accuracy is not enough. Climate prediction is an extrapolation problem under changing forcing, and models that ignore conservation laws can drift or fail outside their training regime. Hybrid and physics-informed designs, from learned sub-grid parameterisations to PINNs for advection–diffusion, give a principled route to physical consistency [19–21]. Explainability tools such as SHAP and integrated gradients help scrutinise models but yield hypotheses, not proofs [22, 23]. Calibrated uncertainty, assessed with proper scores such as CRPS and with reliability diagrams, is a requirement, not an extra [17, 24].

Evaluation must be a portfolio: block cross-validation, anomaly correlation, extreme-event metrics and physical diagnostics together (Table 3). Responsible practice requires attention to geographic bias, reproducibility, energy cost and the limits of extrapolation [1]. Foundation models and generative methods open new possibilities [8], and their use will be most valuable where they are coupled to physical understanding and judged by their contribution to decisions on mitigation and adaptation.

## References

[1] D. Rolnick *et al.*, “Tackling climate change with machine learning,” *ACM Comput. Surv.*, vol. 55, no. 2, Art. no. 42, 2022, doi: [10.1145/3485128](https://doi.org/10.1145/3485128)

[2] H. Hersbach *et al.*, “The ERA5 global reanalysis,” *Q. J. R. Meteorol. Soc.*, vol. 146, no. 730, pp. 1999–2049, 2020, doi: [10.1002/qj.3803](https://doi.org/10.1002/qj.3803)

[3] V. Eyring *et al.*, “Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) experimental design and organization,” *Geosci. Model Dev.*, vol. 9, no. 5, pp. 1937–1958, 2016, doi: [10.5194/gmd-9-1937-2016](https://doi.org/10.5194/gmd-9-1937-2016)

[4] J. Pathak *et al.*, “FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators,” arXiv:2202.11214, 2022, doi: [10.48550/arXiv.2202.11214](https://doi.org/10.48550/arXiv.2202.11214)

[5] R. Lam *et al.*, “Learning skillful medium-range global weather forecasting,” *Science*, vol. 382, no. 6677, pp. 1416–1421, 2023, doi: [10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336)

[6] S. Rasp, P. D. Dueben, S. Scher, J. A. Weyn, S. Mouatadid, and N. Thuerey, “WeatherBench: A benchmark data set for data-driven weather forecasting,” *J. Adv. Model. Earth Syst.*, vol. 12, no. 11, Art. no. e2020MS002203, 2020, doi: [10.1029/2020MS002203](https://doi.org/10.1029/2020MS002203)

[7] M. Reichstein, G. Camps-Valls, B. Stevens, M. Jung, J. Denzler, N. Carvalhais, and Prabhat, “Deep learning and process understanding for data-driven Earth system science,” *Nature*, vol. 566, no. 7743, pp. 195–204, 2019, doi: [10.1038/s41586-019-0912-1](https://doi.org/10.1038/s41586-019-0912-1)

[8] T. Nguyen, J. Brandstetter, A. Kapoor, J. K. Gupta, and A. Grover, “ClimaX: A foundation model for weather and climate,” in *Proc. 40th Int. Conf. Mach. Learn. (ICML)*, 2023, doi: [10.48550/arXiv.2301.10343](https://doi.org/10.48550/arXiv.2301.10343)

[9] L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001, doi: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

[10] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, pp. 785–794, doi: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

[11] T. Vandal, E. Kodra, S. Ganguly, A. Michaelis, R. Nemani, and A. R. Ganguly, “DeepSD: Generating high resolution climate change projections through single image super-resolution,” in *Proc. 23rd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2017, pp. 1663–1672, doi: [10.1145/3097983.3098004](https://doi.org/10.1145/3097983.3098004)

[12] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997, doi: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)

[13] A. Vaswani *et al.*, “Attention is all you need,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 5998–6008, doi: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)

[14] A. Dosovitskiy *et al.*, “An image is worth 16x16 words: Transformers for image recognition at scale,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021, doi: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)

[15] X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, and W.-C. Woo, “Convolutional LSTM network: A machine learning approach for precipitation nowcasting,” in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 28, 2015, pp. 802–810, doi: [10.48550/arXiv.1506.04214](https://doi.org/10.48550/arXiv.1506.04214)

[16] S. M. Vicente-Serrano, S. Beguería, and J. I. López-Moreno, “A multiscalar drought index sensitive to global warming: The standardized precipitation evapotranspiration index,” *J. Climate*, vol. 23, no. 7, pp. 1696–1718, 2010, doi: [10.1175/2009JCLI2909.1](https://doi.org/10.1175/2009JCLI2909.1)

[17] T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and estimation,” *J. Am. Stat. Assoc.*, vol. 102, no. 477, pp. 359–378, 2007, doi: [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)

[18] A. J. Cannon, S. R. Sobie, and T. Q. Murdock, “Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes?” *J. Climate*, vol. 28, no. 17, pp. 6938–6959, 2015, doi: [10.1175/JCLI-D-14-00754.1](https://doi.org/10.1175/JCLI-D-14-00754.1)

[19] K. Kashinath *et al.*, “Physics-informed machine learning: Case studies for weather and climate modelling,” *Philos. Trans. R. Soc. A*, vol. 379, no. 2194, Art. no. 20200093, 2021, doi: [10.1098/rsta.2020.0093](https://doi.org/10.1098/rsta.2020.0093)

[20] S. Rasp, M. S. Pritchard, and P. Gentine, “Deep learning to represent subgrid processes in climate models,” *Proc. Natl. Acad. Sci. USA*, vol. 115, no. 39, pp. 9684–9689, 2018, doi: [10.1073/pnas.1810286115](https://doi.org/10.1073/pnas.1810286115)

[21] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *J. Comput. Phys.*, vol. 378, pp. 686–707, 2019, doi: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

[22] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 4765–4774, doi: [10.48550/arXiv.1705.07874](https://doi.org/10.48550/arXiv.1705.07874)

[23] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 3319–3328, doi: [10.48550/arXiv.1703.01365](https://doi.org/10.48550/arXiv.1703.01365)

[24] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 6402–6413, doi: [10.48550/arXiv.1612.01474](https://doi.org/10.48550/arXiv.1612.01474)

[25] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,” in *Proc. 33rd Int. Conf. Mach. Learn. (ICML)*, 2016, pp. 1050–1059, doi: [10.48550/arXiv.1506.02142](https://doi.org/10.48550/arXiv.1506.02142)
