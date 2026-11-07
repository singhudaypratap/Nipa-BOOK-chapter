# Digital Twins for Sustainable Systems: Multiscale Modeling, Data Assimilation, Prediction and Decision Support

**Uday Pratap Singh\***

Department of AI & DS, Poornima Institute of Engineering & Technology, Jaipur, Rajasthan, India

\*Corresponding email: udaypratap@allduniv.ac.in

ORCID ID: [Insert Valid 16-digit ORCID ID]

## Abstract

Digital twins are computational representations of physical systems that are kept consistent with those systems through observation and are used to evaluate how the systems would behave under alternative operating conditions. For sustainability, their value lies less in visualisation or real-time dashboards than in the disciplined combination of heterogeneous observations, physical models, data-driven forecasting and decision-making under explicit uncertainty. This chapter examines digital twins for environmental, energy, water, urban and resource systems. It first separates static digital models, digital shadows and fully coupled digital twins, and proposes a six-level Digital Twin Coupling Level (DTCL) taxonomy that grades twins by the direction, latency and statistical rigour of the information exchanged between the physical and virtual domains. A six-tier reference architecture is then described, followed by a treatment of the data foundation, including the ensemble Kalman filter and its covariance and rank properties. Physics-based, data-driven, reduced-order and physics-informed modelling strategies are compared, and the mathematics of proper orthogonal decomposition and physics-informed neural-network losses is set out. Prediction, calibration, uncertainty propagation and drift management are treated as continuous processes rather than one-off steps. Applications in water, air quality, circular resources, ecosystems, agriculture, buildings, renewable energy and grids are reviewed, together with multi-objective optimisation for resource efficiency and carbon reduction. The chapter closes with a structured analysis of interoperability, data quality, computational cost, cybersecurity, validation, explainability, infrastructure sustainability and human oversight, and outlines research directions toward trustworthy, adaptive and climate-resilient twins.

**Keywords:** Digital twins; sustainability; data assimilation; physics-informed machine learning; environmental monitoring; energy systems; decision support

## 1. Introduction

### 1.1 From digital models to digital twins

Engineering has relied on mathematical models for as long as it has had calculus, and on computer models for more than half a century. A finite-element mesh of a dam, a hydraulic network model of a city's water mains, or an energy simulation of an office tower are all digital models: abstractions that answer "what if" questions about a design. What such models lack is a persistent, two-way relationship with one specific physical asset as it ages, degrades and is operated. The digital-twin concept was formulated precisely to supply that relationship. Grieves and Vickers describe the twin as a virtual counterpart linked to its physical product across the whole lifecycle, intended to expose emergent and undesirable behaviour before it materialises [@grieves]. The aerospace community adopted a similar view, defining the twin as an integrated, multiphysics, multiscale and probabilistic simulation of an as-built vehicle that mirrors the life of its flying twin using the best available models and sensor data [@glaessgen].

The distinction is therefore not one of fidelity. A coarse model that is continuously corrected by data may be a twin; a high-resolution simulation run once at the design stage is not. What makes a twin is the closure of an information loop between a particular physical entity and its virtual representation [@kritzinger; @tao].

### 1.2 Why sustainability requires dynamic system representations

Sustainability problems share three features that defeat static analysis. First, they are non-stationary: catchment hydrology shifts with land use and climate, building loads change with occupancy patterns, and electricity networks evolve as variable renewable generation displaces dispatchable plant. Second, they are coupled across scales, so that a local intervention such as a pumping schedule or a demand-response signal propagates to regional water balances or grid stability. Third, they are judged on several objectives at once, including cost, emissions, resource use, resilience and equity. A design-stage model, frozen at commissioning, drifts away from the real system and becomes a poor guide to operational decisions. The Earth-system community has drawn the same conclusion at planetary scale, arguing that a digital twin of the Earth that fuses kilometre-scale simulation with observations is needed to support the green transition [@bauer]. A dynamic representation, continuously reconciled with observations, is the minimum condition for credible operational sustainability decisions.

### 1.3 Digital twins versus conventional simulation and monitoring systems

Conventional simulation is forward and open-loop: inputs are specified, a model is run, outputs are interpreted. Conventional monitoring, typically realised as supervisory control and data acquisition (SCADA) systems or Internet-of-Things (IoT) dashboards, is observational and model-free: it reports what sensors measure but cannot infer unobserved states or project future behaviour. A digital twin joins the two. Observations constrain the model state, the model fills gaps between sensors and extends them into the future, and the resulting estimates inform actions that change the physical system, which is then observed again [@rasheed; @fuller]. Three properties follow from this coupling: (i) the twin can estimate quantities that are never directly measured, such as pollutant concentration between monitoring stations; (ii) its predictions carry quantified uncertainty derived from both model and observation errors; and (iii) its value can be judged by the quality of the decisions it enables, not only by its accuracy.

### 1.4 The purpose and scope of this chapter

The literature on digital twins is large but fragmented. Systematic reviews report inconsistent use of the term and frequent conflation of twins with ordinary simulation or monitoring tools [@jones; @kritzinger]. This chapter aims to give a coherent, mathematically grounded treatment oriented toward sustainable systems. Section 2 establishes definitions and introduces the Digital Twin Coupling Level taxonomy. Section 3 presents a six-tier reference architecture. Sections 4 to 6 develop the data, modelling and uncertainty foundations, including data assimilation and physics-informed learning. Sections 7 and 8 survey environmental, resource, energy and built-environment applications. Section 9 addresses optimisation and decision support, Section 10 compares documented implementations, Section 11 analyses open challenges, Section 12 discusses future directions and Section 13 concludes.

## 2. Foundations of Digital Twin Technology

### 2.1 Concept and evolution of digital twins

The intellectual lineage of the digital twin runs through product lifecycle management, where Grieves introduced the triad of a physical product, a virtual product and the data flows connecting them [@grieves]. Aerospace engineering then gave the concept its probabilistic, multiphysics character, motivated by structural-health management of airframes whose individual load histories differ from fleet averages [@glaessgen]. Manufacturing extended it to production lines and factories under the banner of cyber-physical production systems [@tao]. Environmental and Earth-system science adopted the concept later, but brought with it decades of experience in data assimilation for numerical weather prediction, which in many respects is the oldest operational digital twin in existence [@bauer; @evensen2009]. This convergence matters: the manufacturing tradition contributes architecture and lifecycle thinking, while the geoscience tradition contributes rigorous statistical state estimation.

### 2.2 Physical entity, virtual representation and data connection

Every twin has three constituents. The *physical entity* is the bounded real-world system of interest: a pump station, a watershed, a building, a district heating network. Its boundary must be defined explicitly, because exchanges of mass, energy and information across that boundary become forcing terms or boundary conditions in the virtual model. The *virtual representation* is a set of models, often of different types and resolutions, that together describe the state, dynamics and response of the entity. The *data connection* comprises sensors, communication links, data pipelines and the algorithms that translate observations into state corrections and decisions into actuator commands [@tao; @fuller]. Jones et al. identified thirteen characteristics recurring in the literature, among them the physical-to-virtual and virtual-to-physical connections, twinning rate and fidelity [@jones]. For sustainable systems, the twinning rate, meaning the frequency with which the virtual state is synchronised, must be chosen relative to the characteristic time scale of the process: minutes for grid frequency, hours for building thermal response, days to weeks for groundwater.

### 2.3 Digital model, digital shadow and digital twin

Kritzinger et al. proposed a classification based on the automation of data flows [@kritzinger]. In a *digital model*, data exchange between physical and virtual objects is manual in both directions. In a *digital shadow*, data flow automatically from the physical to the virtual object, while any influence in the reverse direction is manual. In a *digital twin*, flows are automatic in both directions. This classification is valuable because it is operational: it can be checked by inspecting a system's interfaces. It is also coarse. It does not distinguish a shadow that merely displays sensor values from one that performs statistically optimal state estimation, nor a twin that issues advisory recommendations from one that actuates equipment autonomously.

### 2.4 Levels of coupling between physical and digital systems

To resolve that coarseness, this chapter proposes a Digital Twin Coupling Level (DTCL) taxonomy with six graded levels, defined by three measurable attributes: the direction of automated information flow, the synchronisation latency $\tau_s$ relative to the dominant process time scale $T_p$, and the statistical treatment of the mismatch between model and observation.

- **DTCL-0, Disconnected model.** No automated data link; the model is parameterised from design data or one-off surveys.
- **DTCL-1, Batch shadow.** Observations are ingested periodically and used for visual comparison or offline recalibration; $\tau_s \gg T_p$.
- **DTCL-2, Streaming shadow.** Observations are ingested continuously and displayed beside model output, but the model state is not corrected; $\tau_s \lesssim T_p$.
- **DTCL-3, Assimilating twin.** The model state, and possibly its parameters, are updated by a formal estimator such as a Kalman-type filter or variational method, yielding a posterior state with uncertainty [@evensen2009].
- **DTCL-4, Advisory twin.** An assimilating twin that also evaluates candidate interventions and returns ranked recommendations to a human operator, who closes the loop.
- **DTCL-5, Autonomous twin.** The loop is closed automatically: the twin selects and executes control actions within governed limits and monitors their effect.

The ladder is cumulative; each level presupposes the capabilities of the one below. It maps onto Kritzinger's classes, since DTCL-0 corresponds to a digital model, DTCL-1 and DTCL-2 to shadows and DTCL-3 to DTCL-5 to twins, but it separates engineering maturity from interface automation. It also makes clear that progressing from DTCL-2 to DTCL-3 demands statistical machinery, not only connectivity. The maturity pathway is summarised later in Figure 4.

### 2.5 Static, predictive and adaptive twins

A second, orthogonal axis concerns temporal capability. A *static* twin describes the current or recent state of the asset. A *predictive* twin propagates that state forward to produce forecasts with uncertainty. An *adaptive* twin additionally revises its own structure or parameters in response to evidence that the physical system has changed, for example after sensor replacement, retrofitting or a regime shift in climate forcing [@rasheed]. Sustainability applications almost always require predictive capability, because interventions such as reservoir releases, storage dispatch or ventilation schedules must be planned before the conditions they respond to occur. Adaptive capability becomes essential over the long asset lifetimes typical of infrastructure.

### 2.6 From monitoring to simulation and decision support

Taken together, the coupling and temporal axes define a progression from monitoring through simulation to decision support. Monitoring answers "what is happening?", estimation answers "what is the full state, including unmeasured variables?", prediction answers "what will happen?", and decision support answers "what should be done, and with what expected consequences and risks?". Each step adds value and also adds obligations: a twin that recommends actions must justify them, quantify their uncertainty and record their provenance [@fuller; @rasheed]. The remainder of the chapter treats these obligations as design requirements rather than afterthoughts.

## 3. Digital Twin Architecture for Sustainable Systems

Architectures for digital twins have been described in many ways, from three-dimensional models comprising physical, virtual and connection parts to five-dimensional variants that add data and services [@tao]. For sustainable systems, a six-tier layering, shown in Figure 1, has the advantage of separating concerns whose technologies, owners and failure modes differ.

![Figure 1. Six-tier architectural layering for sustainable digital twins. Bidirectional arrows indicate that information flows upward as observations and state estimates, and downward as model requests, control set-points and actuation commands.](figures/Figure1_DigitalTwin_Architecture.png)

### 3.1 Physical layer

The base tier is the physical system itself, which may be engineered (a treatment plant, a microgrid), natural (a river reach, a forest stand) or socio-technical (a city district). Defining this tier means stating the system boundary, the state variables of interest, the available control inputs and the exogenous drivers such as weather, demand and markets. In environmental twins the controllable fraction of the system is often small: a basin operator controls reservoir gates but not rainfall. The twin must therefore represent uncontrolled drivers probabilistically.

### 3.2 Sensing and IoT layer

The second tier converts physical quantities into data. It includes in-situ sensors for flow, pressure, temperature, water quality, particulate matter and power; remote-sensing platforms such as satellites, aircraft and drones; and human-generated observations such as inspection reports. Sensor placement determines observability: a state component that does not influence any measurement cannot be corrected by assimilation and is constrained only by the model. Placement design is therefore an optimisation in its own right, trading sensor cost against expected reduction in posterior uncertainty.

### 3.3 Communication and data-management layer

The third tier moves, stores and curates data, and IoT connectivity and data management are recurrently identified as core enabling technologies of twins [@fuller]. Industrial protocols such as OPC Unified Architecture (OPC UA) and lightweight publish–subscribe protocols such as MQTT are widely used for transport, while time-series databases, object stores and data lakes provide persistence. Edge nodes may pre-process and compress data before transmission, which reduces bandwidth and energy use. This tier is also where metadata, units, coordinate reference systems, timestamps and quality flags must be attached and preserved, because errors introduced here propagate silently into every model above.

### 3.4 Modelling and simulation layer

The fourth tier hosts the virtual representation: mechanistic models such as computational fluid dynamics (CFD), finite-element, hydraulic, hydrological or building-energy simulations, together with reduced-order surrogates that make them fast enough for operational use [@rasheed]. Multiscale coupling is often required, for instance a catchment rainfall–runoff model driving a one-dimensional river model that in turn supplies boundary conditions to a two-dimensional inundation model.

### 3.5 Analytics and artificial intelligence layer

The fifth tier contains statistical and machine-learning components: data assimilation, anomaly detection, forecasting, surrogate modelling, physics-informed learning and explainability tools [@karniadakis; @reichstein]. The boundary with the modelling tier is deliberately permeable, since hybrid models combine mechanistic and learned components within a single computational graph.

### 3.6 Decision and control layer

The sixth tier turns estimates and forecasts into actions. It includes optimisation engines, rule-based policies, model-predictive controllers and human–machine interfaces. Its outputs range from advisory reports at DTCL-4 to automated set-points at DTCL-5. Governance rules, including operating envelopes, approval workflows and fail-safe defaults, belong in this tier.

### 3.7 Feedback between physical and virtual environments

The defining property of the architecture is closure of the loop. Observations ascend from tiers 1 to 5; decisions descend from tier 6 to tier 1; the consequences of those decisions are observed again. Two latencies govern the loop: the *sensing-to-estimate* latency, which bounds how current the twin's state is, and the *decision-to-effect* latency, which bounds how quickly interventions act. Stable operation requires the total loop latency to be short relative to the process time scale, and requires that the twin anticipate rather than chase disturbances. In that sense the architecture is a feedback controller, and the familiar concerns of control engineering, namely stability, robustness and delay, apply to it in full.

## 4. Data as the Foundation of Environmental and Sustainable Digital Twins

### 4.1 Sensor and IoT observations

In-situ sensors supply the highest temporal resolution but sparse spatial coverage. Low-cost IoT devices have increased coverage substantially, for instance in urban air-quality networks, but at the price of drift, cross-sensitivity to humidity and temperature, and calibration offsets that vary between units. Treating such observations as error-free is a common and serious mistake. Each observation should carry an error model, at minimum a variance, that the assimilation layer can use to weight it against the model forecast.

### 4.2 Remote sensing and geospatial data

Satellite and airborne platforms provide spatially continuous fields such as land-surface temperature, soil moisture, vegetation indices, surface-water extent and atmospheric composition. Their temporal sampling is limited by revisit times and cloud cover, and their retrievals are themselves model-dependent inferences. For large-scale environmental twins, gridded reanalysis products are often used as background states and boundary forcing. ERA5, for example, provides hourly estimates of atmospheric, land and ocean-wave variables at about 31 km horizontal resolution on 137 vertical levels, produced by assimilating observations into a numerical weather prediction model [@hersbach].

### 4.3 Historical databases and simulation-generated data

Historical records such as utility billing data, maintenance logs, past hydrographs and archived SCADA streams provide the long baselines needed to train statistical models and characterise rare events. Simulation-generated data, produced by running high-fidelity models across designed parameter sweeps, are the principal training resource for surrogates and reduced-order models [@rasheed]. Mixing observed and simulated data requires care, because simulated data inherit the structural biases of the model that produced them.

### 4.4 Data quality, missing observations and sensor uncertainty

Operational data streams contain gaps, spikes, flat-lines, clock errors and unit inconsistencies. A robust twin applies automated quality control, including range, rate-of-change, persistence and spatial-consistency checks, before assimilation, and records the result as a quality flag rather than silently deleting data. Missing observations are handled naturally in a Bayesian framework: when a sensor is absent, the corresponding row of the observation operator is removed, and the analysis reverts toward the model forecast with correspondingly larger uncertainty [@evensen2009].

### 4.5 Spatial and temporal data integration

Heterogeneous sources differ in support, the spatial and temporal footprint over which a measurement averages. A point gauge, a 10 m satellite pixel and a 31 km reanalysis cell do not measure the same quantity even when they report the same variable. Integration therefore requires explicit observation operators that map the model state to the support of each observation, together with representativeness-error terms that account for unresolved variability. Temporal alignment poses parallel problems: asynchronous observations are best assimilated at their true observation times, either by four-dimensional methods or by time-windowed sequential updates.

### 4.6 Data assimilation and state estimation

Data assimilation is the formal mechanism by which a twin moves from DTCL-2 to DTCL-3. Consider a discrete-time state vector $\mathbf{x}_k \in \mathbb{R}^{n}$ evolving under a possibly nonlinear model $\mathcal{M}$ with additive model error, and observed through an operator $\mathbf{H}_k$:

$$
\mathbf{x}_k = \mathcal{M}_{k-1\rightarrow k}(\mathbf{x}_{k-1}) + \boldsymbol{\eta}_k, \qquad \boldsymbol{\eta}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k), \tag{1}
$$

$$
\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \boldsymbol{\epsilon}_k, \qquad \boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k), \tag{2}
$$

where $\mathbf{z}_k \in \mathbb{R}^{m}$ is the observation vector, $\mathbf{H}_k \in \mathbb{R}^{m\times n}$, $\mathbf{Q}_k \in \mathbb{R}^{n\times n}$ is the model-error covariance and $\mathbf{R}_k \in \mathbb{R}^{m\times m}$ is the observation-error covariance. The ensemble Kalman filter (EnKF), introduced by Evensen [@evensen1994], represents the forecast distribution with an ensemble of $N$ members $\{\mathbf{x}_k^{f,(i)}\}_{i=1}^{N}$, each propagated through the full nonlinear model. The ensemble mean $\bar{\mathbf{x}}_k^f$ and the anomaly matrix

$$
\mathbf{A}_k = \frac{1}{\sqrt{N-1}}\left[\mathbf{x}_k^{f,(1)}-\bar{\mathbf{x}}_k^f,\; \ldots,\; \mathbf{x}_k^{f,(N)}-\bar{\mathbf{x}}_k^f\right] \in \mathbb{R}^{n\times N} \tag{3}
$$

define the sample forecast covariance $\mathbf{P}_k^f = \mathbf{A}_k\mathbf{A}_k^{T}$. Each member is then updated by

$$
\mathbf{x}_k^{a,(i)} = \mathbf{x}_k^{f,(i)} + \mathbf{K}_k\left(\mathbf{z}_k + \boldsymbol{\epsilon}_k^{(i)} - \mathbf{H}_k \mathbf{x}_k^{f,(i)}\right), \tag{4}
$$

with the Kalman gain

$$
\mathbf{K}_k = \mathbf{P}_k^f \mathbf{H}_k^{T}\left(\mathbf{H}_k \mathbf{P}_k^f \mathbf{H}_k^{T} + \mathbf{R}_k\right)^{-1} \in \mathbb{R}^{n\times m}. \tag{5}
$$

Averaging (4) over the ensemble yields the familiar mean update $\mathbf{x}_k^a = \mathbf{x}_k^f + \mathbf{K}_k(\mathbf{z}_k - \mathbf{H}_k\mathbf{x}_k^f)$, in which the innovation $\mathbf{z}_k - \mathbf{H}_k\mathbf{x}_k^f$ is weighted by the relative confidence in model and data. The perturbations $\boldsymbol{\epsilon}_k^{(i)} \sim \mathcal{N}(\mathbf{0},\mathbf{R}_k)$ in (4) are needed so that the analysis ensemble has the correct posterior covariance, $\mathbf{P}_k^a \approx (\mathbf{I}-\mathbf{K}_k\mathbf{H}_k)\mathbf{P}_k^f$ [@evensen2009]. Between analyses, covariance is propagated implicitly by advancing each member through $\mathcal{M}$, which avoids the explicit tangent-linear and adjoint models required by the extended Kalman filter.

Three properties deserve emphasis. First, the matrix to be inverted in (5) is only $m\times m$, so the cost scales with the number of observations rather than the state dimension; for very large $m$, observations can be processed serially or in local batches. Second, $\mathbf{P}_k^f$ is never formed explicitly when $n$ is large: the products $\mathbf{P}_k^f\mathbf{H}_k^T = \mathbf{A}_k(\mathbf{H}_k\mathbf{A}_k)^T$ are evaluated with ensemble matrices. Third, and most consequential, $\operatorname{rank}(\mathbf{P}_k^f) \le N-1$. Because environmental states often have $n \sim 10^5$ to $10^8$ while affordable ensembles have $N \sim 10^1$ to $10^2$, the sample covariance is severely rank-deficient. The analysis increment is confined to the subspace spanned by the ensemble anomalies, and sampling noise produces spurious long-range correlations that can corrupt distant state variables. Practical twins mitigate this with covariance localisation, which tapers correlations beyond a physically motivated radius, and with multiplicative or additive inflation, which counteracts the systematic underestimation of spread that finite ensembles cause [@evensen2009]. Parameters can be estimated jointly by augmenting the state vector with uncertain coefficients, which turns the EnKF into an online calibration engine, a capability exploited in Section 6.
