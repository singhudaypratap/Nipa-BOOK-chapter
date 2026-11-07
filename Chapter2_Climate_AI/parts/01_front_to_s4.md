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

Climate change is usually framed as a physical problem, but for the working scientist it is equally a problem of information. The state of the climate system is observed incompletely, by instruments with changing coverage and bias. It is simulated by models whose resolution is too coarse to represent clouds, convection and much of the hydrological cycle explicitly. And it must be projected under scenarios of human activity that are themselves uncertain. Every step from raw measurement to a decision about a sea wall or a power-system investment involves statistical inference. Machine learning is the branch of statistics that scales best to the volume and dimensionality of this information, which is why Rolnick et al. could catalogue applications from electricity systems to climate-model emulation in a single review [@rolnick].

### 1.2 Growth of climate observations and computational modelling

The volume of climate data has grown by orders of magnitude. Satellite constellations deliver continuous global fields; reanalyses such as ERA5 reconstruct the atmosphere hourly from 1940 onward on a roughly 31 km grid with 137 vertical levels [@hersbach]; and the Coupled Model Intercomparison Project Phase 6 (CMIP6) coordinates dozens of modelling centres running common experiments for the assessment of future change [@eyring]. The resulting archives hold petabytes. They are the raw material from which data-driven models are trained, and they are also the reason such models are needed, because the archives are too large to interrogate with hand-crafted diagnostics alone.

### 1.3 Why conventional approaches alone can be computationally demanding

Numerical models integrate the primitive equations of atmospheric and oceanic motion. Their cost grows steeply with resolution: halving the horizontal grid spacing in three dimensions multiplies the number of grid columns by four, and the Courant–Friedrichs–Lewy stability condition then halves the permissible time step, so the total work rises roughly eightfold before any increase in vertical levels is counted. Producing large ensembles, which are essential for uncertainty quantification, multiplies the cost again. This is the computational bottleneck that learned emulators aim to relieve. Pathak et al. and Lam et al. report data-driven global forecasts that run in seconds to minutes on accelerators once trained [@pathak; @lam], and Rasp et al. provide the benchmark framework against which such systems can be compared [@rasp2020].

### 1.4 Opportunities and limitations of AI

The opportunities are real: speed, the ability to exploit nonlinear multivariate structure, and the possibility of learning processes that physics cannot yet resolve. The limitations are equally real and are the organising theme of this chapter. A neural network has no built-in knowledge of conservation of mass, energy or momentum. It interpolates well within the distribution of its training data and may extrapolate poorly beyond it, which matters acutely for a problem whose defining feature is a changing distribution. Reichstein et al. summarise the position succinctly: deep learning offers major opportunities for Earth-system science but needs to be integrated with process understanding, and interpretability, physical consistency and the scarcity of labelled data are open challenges [@reichstein].

### 1.5 Scope of the chapter

Section 2 describes climate data and pre-processing. Section 3 treats classical machine learning and Section 4 deep learning. Section 5 reviews applications to specific hazards, Section 6 downscaling, and Section 7 hybrid and physics-informed approaches. Sections 8 and 9 address explainability and uncertainty, Section 10 evaluation, Section 11 mitigation and adaptation, Section 12 risks and responsible use, and Section 13 emerging directions. Section 14 concludes.

## 2. Climate Data and Computational Foundations

### 2.1 Observational climate datasets

Station records of temperature, precipitation, pressure, humidity and wind form the longest instrumental baseline, in places extending back more than a century. Radiosondes provide vertical profiles of temperature, humidity and wind twice daily at hundreds of sites. Observational networks are unevenly distributed: the Northern Hemisphere midlatitudes are densely sampled, while the tropics, polar regions, high mountains and oceans are sparse. Station data also suffer from instrument changes, relocation and urbanisation, so that homogenisation is a prerequisite for trend analysis. For machine learning, the practical consequence is that training data reflect where people measured, not where the climate varies.

### 2.2 Satellite and remote-sensing observations

Satellites provide near-global coverage of variables such as sea-surface temperature, land-surface temperature, vegetation indices, precipitation estimates, aerosol optical depth, cloud properties and trace-gas columns. Instruments on the MODIS platforms, for example, supply daily and multi-day composites of land and ocean surface properties, and the Copernicus programme provides open Earth-observation data and services. The record length of any individual sensor is short, typically one to two decades for modern products, and retrievals depend on algorithms with their own assumptions. Merging products across sensor generations introduces discontinuities that a learning algorithm may mistake for climate signals.

### 2.3 Reanalysis datasets

Reanalyses assimilate historical observations into a fixed modern forecasting system, producing gridded, physically consistent fields of many variables with no gaps. ERA5, the fifth-generation reanalysis of the European Centre for Medium-Range Weather Forecasts, offers hourly output of atmospheric, land-surface and ocean-wave variables from 1940, with uncertainty information from a lower-resolution ensemble [@hersbach]. Other widely used products include NCEP/NCAR Reanalysis 1 and MERRA-2. Reanalysis is the default training target for data-driven weather and climate models, as in WeatherBench [@rasp2020], FourCastNet [@pathak] and GraphCast [@lam]. It must nevertheless be recognised that reanalysis is model output constrained by data: variables that are weakly observed, such as precipitation and surface fluxes, inherit the biases of the underlying model. A model trained to reproduce ERA5 is learning ERA5's version of reality.

### 2.4 Climate-model simulations

CMIP6 provides standardised simulations from historical forcing to future scenarios, including the Scenario Model Intercomparison Project that pairs radiative-forcing pathways with socio-economic narratives [@eyring]. Regional downscaling experiments, coordinated for example under CORDEX, supply higher-resolution simulations nested in global models. Climate-model output is attractive for machine learning because it is long, complete and spans scenarios beyond the observed climate. It is also the only source of data on states the real system has not yet visited. Climate-model emulators and foundation models trained on CMIP6 exploit exactly this property [@nguyen]. The corresponding danger is that the emulator inherits structural errors of the parent model.

### 2.5 Socio-environmental and emissions data

Many applications require non-climatic predictors or targets: population, land use, energy demand, emissions inventories, crop yields and economic indicators. Future emissions pathways are encoded in the Shared Socioeconomic Pathways (SSPs), which combine narratives of development with radiative-forcing levels and are the scenario framework used in CMIP6 [@eyring]. These data are typically annual or coarser, spatially aggregated by administrative units, and revised retrospectively, so they must be harmonised with gridded climate fields through explicit spatial and temporal aggregation.

### 2.6 Spatial and temporal resolution

Datasets differ by orders of magnitude in resolution: CMIP6 models commonly run at 50 to 250 km, ERA5 at about 31 km, regional models at 10 to 50 km, and convection-permitting models at a few kilometres. The effective resolution, the smallest scale actually resolved, is several grid lengths and therefore coarser than the nominal grid. Temporal resolution ranges from sub-hourly station data to monthly and annual model output. Matching resolutions across predictors and targets is the first design decision in any climate-AI pipeline. When a high-resolution target is predicted from coarse predictors, the problem becomes downscaling (Section 6); when a coarse quantity is predicted from fine inputs, information is aggregated and the problem is better posed.

### 2.7 Data quality check and pre-processing

Pre-processing determines more of a model's behaviour than architecture choice does. The essential steps are as follows. *Quality control* removes physically impossible values and flags suspect ones. *Homogenisation* corrects non-climatic discontinuities. *Regridding* maps fields to a common grid using conservative remapping for flux-like quantities and bilinear interpolation for smooth state variables. *Deseasonalisation* removes the climatological annual cycle, defining the anomaly of variable $x$ at location $s$ and time $t$ as

$$
x'(s,t) = x(s,t) - \bar{x}(s,\tau(t)), \tag{1}
$$

where $\bar{x}(s,\tau)$ is the climatological mean for calendar day or month $\tau$ over a reference period. *Detrending* removes secular change where the task is to learn variability rather than the trend. *Normalisation* scales each variable by climatological mean and standard deviation so that gradient-based training is well conditioned. All statistics for these steps must be computed from the training period only, otherwise information from the test period leaks into the model and skill is overstated. Because geographic grid cells shrink toward the poles, loss functions should be weighted by $\cos\phi$, where $\phi$ is latitude, so that errors are weighted by area.

## 3. Machine Learning Methods for Climate Analysis

### 3.1 Regression methods

Linear and regularised regression remain the baseline against which more complex models must justify themselves. Ridge regression minimises $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|^2$ and handles collinear predictors, which are ubiquitous in climate data because neighbouring grid cells and related variables are strongly correlated. The lasso replaces the squared penalty with an $\ell_1$ norm and thereby performs variable selection. Regression is used for trend estimation, for statistical post-processing of model output and as the skeleton of many downscaling schemes. A model that cannot beat a well-specified regression on the same predictors has found nothing that regression could not.

### 3.2 Decision tree models and ensemble learning

Tree ensembles handle nonlinearity, interactions and mixed variable types with little tuning. Random forests average many decorrelated trees, each grown on a bootstrap sample with a random subset of predictors at each split, which reduces variance without increasing bias much [@breiman]. Gradient-boosted trees, as implemented in XGBoost, build trees sequentially to correct the residual errors of the ensemble, using second-order gradient information and regularisation, and are frequently the strongest method on tabular data [@chen]. In climate applications tree ensembles are used for drought and crop-yield prediction, bias correction and attribution of regional change. They cannot extrapolate: predictions are averages of training targets, so a tree model asked about a warmer-than-observed regime returns values bounded by the warmest it has seen.

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

Convolutional neural networks (CNNs) apply shared local filters, so that a pattern, such as a front or a convective cell, is detected wherever it appears. For a field $\mathbf{X}$ with $C$ channels, a convolutional layer computes $\mathbf{Y}_{k}=\sigma\big(\sum_{c=1}^{C}\mathbf{W}_{k,c}\ast\mathbf{X}_{c}+b_k\big)$. Stacked layers, pooling and skip connections build hierarchical features from local gradients to synoptic patterns. On the sphere, plain convolutions on a latitude–longitude grid distort near the poles and need periodic padding in longitude; spherical and graph-based variants remove this distortion. CNNs underlie most downscaling and nowcasting networks, including super-resolution models for climate fields [@vandal].

### 4.3 Recurrent neural networks

Recurrent networks process sequences by carrying a hidden state, $\mathbf{h}_t=\sigma(\mathbf{W}_h\mathbf{h}_{t-1}+\mathbf{W}_x\mathbf{x}_t+\mathbf{b})$. They suit climate time series whose present depends on the past, such as soil moisture memory or ocean heat content. Their weakness is that gradients propagated through many steps either vanish or explode, so simple recurrent networks cannot learn dependencies across long lags.

### 4.4 LSTM and GRU architectures

Long short-term memory (LSTM) cells address the gradient problem with a cell state $\mathbf{c}_t$ regulated by gates [@hochreiter]:

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f), \quad
\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i),\\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c),\quad
\mathbf{c}_t = \mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t,\\
\mathbf{o}_t &= \sigma(\mathbf{W}_o[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o),\quad
\mathbf{h}_t = \mathbf{o}_t\odot\tanh(\mathbf{c}_t),
\end{aligned} \tag{2}
$$

where $\mathbf{f}_t$, $\mathbf{i}_t$ and $\mathbf{o}_t$ are forget, input and output gates and $\odot$ is element-wise multiplication. The additive update of $\mathbf{c}_t$ lets gradients flow across many steps when the forget gate is near one. The gated recurrent unit (GRU) merges cell and hidden state and uses two gates, giving fewer parameters at similar accuracy in many tasks. In climate applications LSTMs encode temporal memory in streamflow, soil moisture, sea-surface temperature and index forecasting.

### 4.5 Autoencoders

An autoencoder learns an encoder $f_\phi$ and decoder $g_\theta$ minimising reconstruction error $\|\mathbf{x}-g_\theta(f_\phi(\mathbf{x}))\|^2$ through a low-dimensional bottleneck. It is a nonlinear generalisation of EOF analysis (Section 3.5): with linear activations the optimal bottleneck spans the leading EOF subspace. Autoencoders are used for compressing fields, for anomaly detection by reconstruction error, and as the latent space in which a forecast model operates, which reduces cost when the model predicts latent states and decodes them to the physical grid. Variational autoencoders add a probabilistic latent space useful for generating ensembles.

### 4.6 Attention mechanisms and transformers

Attention computes, for each position, a weighted sum of values with weights set by query–key similarity [@vaswani]:

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{d_k}}\right)\mathbf{V}. \tag{3}
$$

Because every position can attend to every other, transformers capture long-range dependence, such as teleconnections between the tropical Pacific and mid-latitude circulation, without the stepwise propagation of convolution or recurrence. Vision transformers split an image into patches, embed each as a token and apply attention across tokens [@dosovitskiy]; for climate fields, patches become tiles of a gridded multivariate state. The price of global attention is cost quadratic in the number of tokens, which motivates patch embedding and efficient variants. FourCastNet addresses it with adaptive Fourier neural operators that mix tokens in the frequency domain at lower cost [@pathak]. Pre-trained transformer backbones are the basis of recent foundation models for weather and climate [@nguyen].

### 4.7 Spatiotemporal deep-learning models

Climate prediction requires models that treat space and time jointly. The convolutional LSTM (ConvLSTM) replaces the matrix products of Equation (2) with convolutions, so that the cell and hidden states are spatial fields and the gates respond to local spatial context; it was introduced for precipitation nowcasting from radar sequences and outperformed fully connected LSTMs on that task [@shi]. PredRNN extends the idea by passing memory both across time and up through layers. Transformer-based models operate on tokenised fields and relate positions at any distance. Graph neural networks represent the globe as a mesh, with message passing along edges, which avoids the polar distortion of regular grids; GraphCast uses an icosahedral multi-mesh encoder–processor–decoder to produce 10-day global forecasts at 0.25° resolution [@lam].

A hybrid architecture combines the strengths of convolutional and attention components, as shown in Figure 1. A ConvLSTM encoder captures local spatiotemporal dynamics from the input sequence; its feature maps are tokenised into patches and processed by a transformer encoder that captures global interactions; a decoder reconstructs the target field, and a probabilistic head outputs both a mean and a variance to support uncertainty estimation.

![Figure 1. Hybrid ConvLSTM–Transformer architecture for gridded climate prediction. Local spatiotemporal features are extracted by a ConvLSTM encoder, global teleconnections are modelled by a transformer encoder over patch tokens, and a decoder with a probabilistic head outputs the predictive mean and variance.](figures/Figure1_ConvLSTM_Transformer.png)
