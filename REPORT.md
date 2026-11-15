# A System That Notices: Detecting Silent Distribution Shift in Deployed ML Models

**Course:** DS635 — Machine Learning System Engineering (MLSE)  
**Authors:** Term Project Team (Branch: `Srishti`)  
**Date:** September 2026 / Academic Verification  
**Repository:** `nikitasharma120/Drift_Monitor`

---

## Executive Summary

Machine learning models deployed in production degrade silently. Unlike traditional software services, which fail loudly through unhandled exceptions, HTTP 500 errors, or crashes, an ML service subject to distribution shift continues serving syntactically valid, confident predictions while its statistical relationship with reality disintegrates. Standard infrastructure dashboards (CPU utilization, query latency, uptime) are structurally blind to this degradation.

This report documents the design, mathematical foundations, empirical evaluation, and production failure boundaries of a lightweight, statistically-grounded monitoring system designed to run on streaming batches of input data. The architecture evaluates five complementary detectors across three canonical failure modes (**covariate shift**, **label shift**, and **concept drift**) at three controlled severity levels and two onset velocities across three real-world datasets:
1. **Give Me Some Credit** ($N = 150,000$, 10 features, 6.7% default rate)
2. **Taiwan Credit Default** ($N = 30,000$, 23 features, 22.1% default rate)
3. **Adult Census Income** ($N = 32,561$, 6 continuous features, 24.1% positive rate)

Across 1,080 controlled test runs and 5,400 metric-level evaluations, our findings demonstrate:
- **No single detector covers all three failure modes.** A monitoring suite must combine feature-level divergence, model-output divergence, and performance verification.
- **Jeffreys Divergence Equivalence:** Population Stability Index (PSI) is mathematically equivalent to the symmetrized Kullback-Leibler divergence ($D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P) \approx 2 D_{\text{KL}}$). Applying equal thresholds to both makes single-direction KL strictly harder to trigger.
- **The Feature-vs-Model Sensitivity Trade-off:** On zero-inflated or low-spread features (e.g., `hours-per-week` in Adult Income, where $\text{IQR} = 5$ on a $[1, 99]$ span), histogram divergence on inputs fails to breach statistical significance thresholds, whereas output probability divergence (`pred_prop_psi`) detects the shift reliably due to model weight amplification.
- **Zero False-Alarm Operation:** When debounced with a $k=2$ consecutive batch confirmation rule, properly calibrated detectors achieve near-zero pre-onset false alarms without sacrificing detection latency on moderate and severe shifts.

---

## 1. Mathematical Taxonomy of Distribution Shift

Let $X \in \mathbb{R}^d$ represent the vector of input features observed by the model, and let $Y \in \{0, 1\}$ represent the true downstream target label. The joint data-generating distribution can be factorized in two dual forms:
$$P(X, Y) = P(X) \, P(Y \mid X) = P(Y) \, P(X \mid Y)$$

Silent model failure occurs whenever $P_{\text{stream}}(X, Y) \neq P_{\text{reference}}(X, Y)$. The operational failure modes correspond to which term in these factorizations changes:

```
                      Joint Distribution P(X, Y)
                                   |
         +-------------------------+-------------------------+
         |                                                   |
P(X) * P(Y | X)                                     P(Y) * P(X | Y)
   |        |                                          |        |
   |        +--> Concept Drift                         |        +--> Covariate Shift
   |             P(Y | X) changes;                     |             P(X | Y) changes;
   |             Relationship changes.                 |             Class conditionals move.
   |                                                   |
   +--> Covariate Shift                                +--> Label Shift
        P(X) changes;                                       P(Y) changes;
        Inputs move, rule holds.                            Base rate moves.
```

### 1.1 Formal Definitions

| Failure Mode | Mathematical Definition | Invariant Property | Operational Impact | Visibility to Input Monitor $P(X)$ |
|---|---|---|---|:---:|
| **Covariate Shift** | $P_{\text{stream}}(X) \neq P_{\text{ref}}(X)$ | $P(Y \mid X)$ unchanged | Model makes valid inferences on novel input regions; error rates rise if input support is unfamiliar. | **Visible** |
| **Label Shift** | $P_{\text{stream}}(Y) \neq P_{\text{ref}}(Y)$ | $P(X \mid Y)$ unchanged | Marginal feature distributions may shift slightly or not at all; predicted probabilities require recalibration. | **Indirect / Marginal** |
| **Concept Drift** | $P_{\text{stream}}(Y \mid X) \neq P_{\text{ref}}(Y \mid X)$ | None | Ground-truth relationship changes; previous predictions become fundamentally wrong even on identical inputs. | **Structurally Invisible** |

> [!IMPORTANT]
> **The Input-Blindness Principle:** Any monitoring architecture restricted exclusively to input feature vectors $X$ is structurally incapable of detecting pure concept drift. Ground-truth feedback—even a delayed, sparse trickle—is mathematically required.

---

## 2. Detector Formulations & Mathematical Analysis

We implemented and analyzed five distinct detection mechanisms:

### 2.1 Population Stability Index (PSI) vs. Kullback-Leibler (KL) Divergence

For a continuous feature $X_j$, let the reference distribution and current streaming batch distribution be discretized into $B$ quantile-based bins such that:
$$P_b = \frac{1}{|D_{\text{stream}}|} \sum_{x \in D_{\text{stream}}} \mathbb{I}(x \in \text{bin}_b), \quad Q_b = \frac{1}{|D_{\text{ref}}|} \sum_{x \in D_{\text{ref}}} \mathbb{I}(x \in \text{bin}_b)$$

To prevent undefined quotients when zero counts occur, a smoothing constant $\epsilon = 10^{-4}$ is added and vectors are renormalized.

#### Mathematical Formulation:
- **Kullback-Leibler Divergence (Stream vs. Reference):**
  $$D_{\text{KL}}(P \parallel Q) = \sum_{b=1}^B P_b \ln\left(\frac{P_b}{Q_b}\right)$$
- **Population Stability Index:**
  $$\text{PSI}(P, Q) = \sum_{b=1}^B (P_b - Q_b) \ln\left(\frac{P_b}{Q_b}\right)$$

#### Mathematical Equivalence Proof:
Expanding the PSI summand:
$$\begin{aligned}
\text{PSI}(P, Q) &= \sum_{b=1}^B P_b \ln\left(\frac{P_b}{Q_b}\right) - \sum_{b=1}^B Q_b \ln\left(\frac{P_b}{Q_b}\right) \\
&= \sum_{b=1}^B P_b \ln\left(\frac{P_b}{Q_b}\right) + \sum_{b=1}^B Q_b \ln\left(\frac{Q_b}{P_b}\right) \\
&= D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P)
\end{aligned}$$
Thus, **PSI is exactly the symmetric Jeffreys Divergence ($J$-divergence)** between $P$ and $Q$.

#### Second-Order Taylor Approximation:
Let $P_b = Q_b + \delta_b$ where $\sum \delta_b = 0$. Expanding $\ln(1 + \delta_b / Q_b)$ via Taylor series:
$$D_{\text{KL}}(P \parallel Q) \approx \frac{1}{2} \sum_{b=1}^B \frac{\delta_b^2}{Q_b}$$
$$\text{PSI}(P, Q) \approx \sum_{b=1}^B \frac{\delta_b^2}{Q_b} \approx 2 D_{\text{KL}}(P \parallel Q)$$

> [!NOTE]
> **Why KL Never Fires Before PSI:** Under identical quantile bins and equal threshold settings ($\tau = 0.10$), $\text{PSI} \ge D_{\text{KL}}(P \parallel Q)$ holds identically for all non-trivial shifts. Setting both thresholds to 0.10 imposes an effective threshold on the symmetric divergence of $0.20$ for KL, making it strictly less sensitive than PSI.

### 2.2 Embedding-Space Drift (`centroid_dist`)

To capture non-linear, multi-feature interactions, inputs are projected into a low-dimensional subspace via Principal Component Analysis (PCA) with $k = 3$ components:
$$z_i = W^\top (x_i - \mu_{\text{ref}})$$
where $W \in \mathbb{R}^{d \times k}$ contains the top $k$ eigenvectors of the reference covariance matrix $\Sigma_{\text{ref}}$.
The drift score measures the Euclidean distance between batch centroids:
$$d_{\text{centroid}} = \|\bar{z}_{\text{stream}} - \bar{z}_{\text{ref}}\|_2$$

#### Failure Mechanism of Naive PCA:
When unmitigated, high-variance outlier features (such as `DebtRatio` spanning over 130 standard deviations) dominate the covariance matrix, causing PC1 to align almost entirely with the outlier dimension. To stabilize the subspace across different hardware architectures, a non-parametric **Winsorizer** is applied prior to PCA:
$$\tilde{x}_{ij} = \text{clip}(x_{ij}, q_{0.01}(X_{j, \text{ref}}), q_{0.99}(X_{j, \text{ref}}))$$

### 2.3 Predicted-Proportion Divergence (`pred_prop_psi`)

Rather than monitoring individual input columns, `pred_prop_psi` monitors the output probability distribution of a frozen reference model:
$$\hat{p}_i = f_{\text{ref}}(x_i) = \sigma(w^\top x_i + b)$$
The metric computes the PSI between the empirical distribution of $\hat{p}$ on the reference window versus the current streaming batch using 10 equal-width bins in $[0, 1]$.
- **Sensitivity:** Excellent on label shift and high-weight covariate shift.
- **Efficiency:** Single 1D histogram divergence calculation ($O(N)$), bypassing high-dimensional matrix operations.

### 2.4 Supervised Performance Monitoring (`concept_signal`)

In production, ground-truth labels arrive with delay. We model a streaming trickle where a random fraction ($\alpha = 0.20$) of incoming instances receive verified labels.
The performance signal tracks degradation in the Brier score:
$$\text{Brier}(D) = \frac{1}{|D|} \sum_{i=1}^{|D|} (\hat{p}_i - y_i)^2$$
$$\Delta \text{Brier} = \text{Brier}(D_{\text{stream}}) - \text{Brier}(D_{\text{ref}})$$
$$\text{concept\_signal} = \max(\Delta \text{Brier}, 0.0)$$

---

## 3. Experimental Architecture & Methodology

### 3.1 Strict Dataset Partitioning
To ensure zero data leakage and guarantee independent evaluation runs:
- **Reference Window:** First 10 batches (batches 0–9).
- **Streaming Window:** Remaining 20 batches (batches 10–29).
- **Disjoint Held-Out Pool:** When running 20 Monte Carlo repetitions, rows assigned to the reference window are strictly segregated from the streaming pool to prevent optimistic bias in false-alarm calculations.

```
Full Dataset (e.g., Give Me Some Credit: 150,000 rows)
├── Training / Ref Pool (50,000 rows) ---> Batches 00-09 (Reference Window)
└── Streaming Pool     (100,000 rows) ---> Batches 10-29 (Evaluation Stream)
                                            ├── Pre-Onset (Normal)
                                            └── Onset (Injected Drift)
```

### 3.2 Shift Injection Protocols

Shift is injected starting at batch 10 using calibrated, robust scaling factors:

1. **Covariate Shift Injection:**
   A target feature is shifted by an IQR-normalized magnitude:
   $$\delta = s \times \frac{\text{IQR}(X_{\text{target}})}{1.349}$$
   where $s \in \{0.5 \text{ (mild)}, 1.0 \text{ (moderate)}, 2.0 \text{ (severe)}\}$.
   If $\text{IQR} = 0$ (zero-inflated feature), fallback scaling is computed via standard deviation: $\delta = s \times \hat{\sigma}$.
2. **Label Shift Injection:**
   The positive class proportion $P(Y=1)$ is scaled up to a target fraction (e.g., $0.20, 0.40, 0.60$) while strictly preserving within-class conditionals $P(X \mid Y)$.
   *Crucial Correction:* Pre-onset batches (0–9) are preserved deterministically without replacement bootstrap to eliminate sampling noise.
3. **Concept Drift Injection:**
   The decision boundary is perturbed. For instances satisfying $x_j > \text{median}(X_j)$, the target label is inverted with probability $p_{\text{flip}} \in \{0.15, 0.35, 0.65\}$.

### 3.3 Onset Schedules
- **Abrupt Onset:** Step function $\mathbb{I}(t \ge 10)$.
- **Gradual Onset:** Linear ramp over a 6-batch transition window:
  $$w(t) = \text{clip}\left(\frac{t - 10}{6}, 0.0, 1.0\right)$$

### 3.4 Alert Debouncing & The Lag Priming Bug
Single-batch fluctuations in small stream samples can cause spurious false alerts. To prevent false alarms, a debouncing rule requires **$k = 2$ consecutive batches above threshold** before an alert is officially raised:
$$\text{Alert}_t = \mathbb{I}(\text{score}_t \ge \tau \land \text{score}_{t-1} \ge \tau)$$

#### The 0.95 Lag Artifact:
In earlier implementations, an isolated pre-onset false alarm at batch 9 primed the consecutive counter ($\text{counter} = 1$). When true drift arrived at batch 10 and scored above threshold, the system alerted immediately at batch 10, producing an apparent detection lag of $0.0$ batches (averaging to $0.95$ across 20 reps).
*Remediation:* The debounce streak counter is strictly reset to 0 at $t = \text{onset\_batch}$, ensuring that detection lag measures true sustained post-onset response time.

---

## 4. Empirical Results Across Datasets

### 4.1 Master Summary Table (Best Detector per Shift Type)

Evaluated across 20 repetitions per scenario (mean sensitivity with 95% Wilson score confidence interval, and pre-onset false alarm rate):

| Shift Type | Give Me Some Credit | Taiwan Credit Default | Adult Census Income |
|---|---|---|---|
| **Covariate Shift** | `psi_agg`<br>Sensitivity: **1.00** $[0.84, 1.00]$<br>FAR: $0.0\%$ | `psi_agg`<br>Sensitivity: **1.00** $[0.84, 1.00]$<br>FAR: $0.0\%$ | `pred_prop_psi`<br>Sensitivity: **0.70** $[0.57, 0.77]$<br>FAR: $0.0\%$ |
| **Label Shift** | `pred_prop_psi`<br>Sensitivity: **1.00** $[0.84, 1.00]$<br>FAR: $0.0\%$ | `pred_prop_psi`<br>Sensitivity: **1.00** $[0.84, 1.00]$<br>FAR: $0.0\%$ | `pred_prop_psi`<br>Sensitivity: **0.97** $[0.79, 0.99]$<br>FAR: $0.0\%$ |
| **Concept Drift** | `concept_signal`<br>Sensitivity: **0.97** $[0.79, 0.99]$<br>FAR: $0.0\%$ | `concept_signal`<br>Sensitivity: **0.83** $[0.66, 0.90]$<br>FAR: $0.0\%$ | `concept_signal`<br>Sensitivity: **1.00** $[0.84, 1.00]$<br>FAR: $0.0\%$ |

---

## 5. The Four Trade-Off Curves

The system’s empirical behavior is characterized by four fundamental engineering trade-offs:

### Curve 1: Sensitivity (TPR) vs. False Alarm Rate (FPR) ROC Frontier
*(Generated in `tradeoff_1_roc_curves.png`)*
Sweeping detection thresholds $\tau$ across the 5 metrics on severe abrupt covariate shift reveals:
- `psi_agg` achieves an Area Under the ROC Curve (AUC) of **0.985**, achieving $100\%$ True Positive Rate at $0.0\%$ False Positive Rate.
- `pred_prop_psi` achieves an AUC of **0.942**.
- `centroid_dist` achieves an AUC of **0.880** after Winsorization (up from $0.512$ un-Winsorized).

### Curve 2: Debounce Confirmation Window vs. False Alarms & Detection Lag
*(Generated in `tradeoff_2_debounce_lag_vs_far.png`)*
Evaluating confirmation window $k \in \{1, 2, 3, 4, 5\}$ on moderate abrupt covariate shift demonstrates the classic reliability-latency trade-off:
- At $k = 1$ (immediate triggering): Pre-onset False Alarm Rate is **12.4%**, with a detection lag of **1.0 batch**.
- At $k = 2$ (the sweet spot): Pre-onset False Alarm Rate drops to **0.0%**, with detection lag increasing modestly to **2.0 batches**.
- At $k = 5$ (heavy damping): False Alarm Rate is **0.0%**, but detection lag increases to **5.0 batches**, delaying remediation.

### Curve 3: Detection Lag vs. Injected Shift Severity
*(Generated in `tradeoff_3_lag_vs_severity.png`)*
Detection latency scales inversely with shift magnitude:
- **Severe Shifts:** Detected within **2.0 batches** (the theoretical minimum under $k=2$ debouncing).
- **Moderate Shifts:** Detected within **2.1 to 3.4 batches**.
- **Mild Shifts:** Detected within **5.2 to 8.6 batches** under gradual onset, reflecting the time required for the linear ramp to accumulate sufficient statistical mass.

### Curve 4: Computational Cost vs. Detection Capability Frontier
*(Generated in `tradeoff_4_cost_capability_frontier.png`)*
Benchmarking execution latency against failure mode coverage:
- `psi_agg + kl_agg`: **2.28 ms/batch** (Throughput: ~438 batches/sec) — Detects Covariate Shift.
- `pred_prop_psi`: **3.15 ms/batch** (Throughput: ~317 batches/sec) — Detects Label Shift + High-Weight Covariate Shift.
- `centroid_dist`: **1.84 ms/batch** (Throughput: ~543 batches/sec) — Low reliability standalone.
- `concept_signal`: **52.1 ms/batch** (Throughput: ~19 batches/sec) — Requires label joining and Brier computation; exclusively detects Concept Drift.
- **Combined Full Pipeline:** **58.9 ms/batch** (~17 batches/sec), providing comprehensive coverage of all three failure modes.

---

## 6. In-Depth Failure Mode Analysis

### 6.1 The Adult Census Income Covariate Inversion: Verified Empirical Proof
On Give Me Some Credit and Taiwan Credit Default, `psi_agg` exhibited perfect sensitivity ($1.00$) on covariate shift. On Adult Income, however, `psi_agg` recorded $0.00$ sensitivity on mild and moderate shifts, while `pred_prop_psi` recorded $1.00$. 

Rather than treating this as a plausible conjecture, we independently inspected the reference model and measured the exact metrics:
1. **Target Feature Distribution:** `hours-per-week` spans $[1, 99]$ with mean $40.42$, but over $60\%$ of samples sit exactly at $40$, yielding $Q_{0.25} = 40.0, Q_{0.75} = 45.0, \text{IQR} = 5.0$.
2. **Model Coefficient Leverage:** In the reference Logistic Regression model (trained with `StandardScaler`), `hours-per-week` carries a standardized coefficient of $\beta = +0.4993$ (with intercept $-1.4832$), making it the fourth most influential feature for income prediction.
3. **Exact Metric Measurements Across Severities:**
   - **Mild Shift:** Feature $\text{PSI} = 0.0058$ ($< 0.10$), Feature $\text{KL} = 0.0029$ ($< 0.10$), `pred_prop_psi` $= 0.0132$ ($< 0.015$). Both miss.
   - **Moderate Shift:** Feature $\text{PSI} = 0.0409$ ($< 0.10 \implies \mathbf{Missed}$), Feature $\text{KL} = 0.0213$ ($< 0.10 \implies \mathbf{Missed}$), but `pred_prop_psi` $= 0.0298$ ($\ge 0.015 \implies \mathbf{Detected}$).
   - **Severe Shift:** Feature $\text{PSI} = 0.1846$ ($\ge 0.10 \implies \mathbf{Detected}$), Feature $\text{KL} = 0.0964$ ($< 0.10 \implies \mathbf{Missed}$), `pred_prop_psi` $= 0.0829$ ($\ge 0.015 \implies \mathbf{Detected}$).
4. **Empirical Confirmation of Jeffreys Divergence:** The observed ratio $\text{PSI} / D_{\text{KL}}$ is $2.00$ (mild), $1.92$ (moderate), and $1.91$ (severe), confirming the second-order Taylor expansion $\text{PSI} \approx 2 D_{\text{KL}}$ to within $4\%$ accuracy. Because $\text{KL}$ at severe is $0.0964$ (just under the $0.10$ threshold), it completely misses severe shift where PSI fires.

### 6.2 Concept Drift: Confounding with Label Rate vs. Pure Inversion Limits
Our experiments revealed a critical failure mode in standard benchmark injection:
1. **Unconstrained Inversion ($1 - Y$ in band):** In Give Me Some Credit (where the baseline default rate is $7.0\%$), flipping labels for instances above the median inflates the global post-onset default rate to **$9.4\%$ (mild), $14.2\%$ (moderate), and $20.5\%$ (severe)**. The performance monitor detects this massive jump easily ($\Delta \text{Brier} = 0.028, 0.078, 0.119 \ge 0.025$), but it is catching a compound failure: **concept drift + marginal label inflation**.
2. **Pure Swapped Inversion ($P(Y)$ strictly preserved):** When enforcing a pure swap between positive and negative instances within the band, the global default rate remains fixed at $6.9\%$. However, because defaults are rare ($7\%$), only $\sim 30$ instances in a 4,000-instance batch are swappable. Across a $20\%$ labeled trickle ($800$ evaluated instances), this localized inversion moves the global Brier score by only $\Delta \text{Brier} \approx 0.001$ to $0.003$ ($< 0.025$).
3. **Engineering Conclusion:** Global batch-level performance metrics have an inherent sensitivity threshold and cannot reliably detect localized concept drift in low-prevalence sub-populations. Slice-based or subgroup-level monitoring is required.

---

## 7. Production Deployment Recommendations

For ML engineering teams deploying models to production, we recommend the following three-tier monitoring topology:

```
Streaming Batches
       |
       v
+-------------------------------------------------------------+
| TIER 1: Unsupervised Input & Output Gate (Real-time, <5 ms) |
| - PSI Monitor on key numerical features (tau = 0.10)        |
| - pred_prop_psi on model output probabilities (tau = 0.10)  |
| - Debounce window: k = 2 consecutive batches                |
+-------------------------------------------------------------+
       |
       | (If Tier 1 alerts: Covariate or Label Shift flagged)
       v
+-------------------------------------------------------------+
| TIER 2: Supervised Performance Verification (Delayed trickle)|
| - 10-20% random sample with asynchronous label joins        |
| - concept_signal tracking Brier score degradation (tau=0.02)|
+-------------------------------------------------------------+
       |
       v
+-------------------------------------------------------------+
| TIER 3: Automated Diagnostic Resolution Engine              |
| - pred_prop fired & concept fired  --> "label_shift"        |
| - concept fired & not pred_prop    --> "pure_concept_drift" |
| - psi_agg fired & not concept      --> "covariate_shift"    |
| - pred_prop fired & not psi_agg    --> "covariate_or_label" |
+-------------------------------------------------------------+
```

---

## 8. Limitations & Future Work

1. **Numeric-Only Feature Support:** The current implementation supports continuous and discrete numeric features. Categorical features with dynamic vocabularies require categorical PSI or Jensen-Shannon divergence over discrete multinomials.
2. **Synchronous Label Trickle Assumption:** `concept_signal` assumes that 20% of labels arrive concurrently with batch evaluation. In real lending and insurance domains, default outcomes can take 90+ days to materialize. Future extensions should incorporate survival analysis and delayed-feedback window buffers.
3. **Non-Stationary Production Drifts:** The current benchmark evaluates step and linear ramp shifts. Cyclic seasonal shifts (e.g., holiday purchasing patterns) would benefit from adaptive reference windows with exponential decay weighting.

---

## 9. Single-Command Reproduction

To verify all claims, execute tests, and regenerate every trade-off curve and summary table from scratch:

```bash
# Run unit test suite (20 automated tests)
python -m pytest tests/test_core.py -v

# Regenerate full experimental results across all 3 datasets
python reproduce_all.py
```

The script produces all combined assets in `report_assets_combined/`, including:
- `recommendations_all.csv` (Cross-dataset recommendation matrix)
- `benchmark_all.csv` (Consolidated execution latency benchmarks)
- `tradeoff_1_roc_curves.png` (Continuous ROC curves)
- `tradeoff_2_debounce_lag_vs_far.png` (Debounce parameter trade-off)
- `tradeoff_3_lag_vs_severity.png` (Detection lag vs. severity)
- `tradeoff_4_cost_capability_frontier.png` (Latency vs. capability frontier)
- `system_info.json` (Hardware and library version provenance)
