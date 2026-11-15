# Presentation Guide: A System That Notices
## Detecting Silent Distribution Shift in Deployed ML Models

**Course:** DS635 — Machine Learning System Engineering (MLSE)  
**Total Duration:** 15 Minutes (Presentation) + 5 Minutes (Q&A)  
**Team Members & Roles:**
1. **Sanjana Nathani** — Problem Context, Distribution Shift Taxonomy & Theoretical Visibility (Slides 1–3) [00:00 – 03:30]
2. **Nikita Sharma** — Experimental Setup, Baselines, Calibrated Injections & Error Bars (Slides 4–6) [03:30 – 07:00]
3. **Anish Karmakar** — Empirical Findings, Master Recommendations & The 4 Trade-Off Curves (Slides 7–9) [07:00 – 10:45]
4. **Srishti Lamba** — Where Things Break, Mathematical Proofs, Live Reproduction & Limitations (Slides 10–13) [10:45 – 14:30]

---

## Slide-by-Slide Outline & Speaker Script

```
========================================================================================
SLIDE 1: Title Slide & Team Introduction
Speaker: Sanjana Nathani (Time: 00:00 – 00:45)
========================================================================================
```
### Visuals:
- **Title:** A System That Notices: Detecting Silent Distribution Shift in Deployed ML Models
- **Subtitle:** DS635 Term Project Presentation
- **Authors:** Sanjana Nathani, Nikita Sharma, Anish Karmakar, Srishti Lamba
- **Repository:** `nikitasharma120/Drift_Monitor` (Branch: `Srishti`)

### Speaker Notes (Sanjana):
> "Good morning, Professor. Today, our team—Nikita, Anish, Srishti, and myself, Sanjana—presents *'A System That Notices: Detecting Silent Distribution Shift in Deployed ML Models.'*
> 
> In traditional software systems, failures announce themselves loudly through crash logs, HTTP 500 errors, or database timeouts. In deployed machine learning, failure is silent. A model continues returning valid, confident predictions while the data-generating distribution underneath it quietly changes. 
> 
> Over the next 14 minutes, we will present our experimental setup across three real-world datasets, our four empirical trade-off curves, the mathematical mechanics of why standard detectors break, and demonstrate a single-command reproduction of our entire pipeline."

---

```
========================================================================================
SLIDE 2: The Problem Context & The Silent Failure Dilemma
Speaker: Sanjana Nathani (Time: 00:45 – 02:00)
========================================================================================
```
### Visuals:
- **Comparison Graphic:**
  - *Standard Ops Dashboard (Green):* CPU 32%, Query Latency 14ms, Error Rate 0.0%, HTTP 200 OK.
  - *Statistical Reality (Red):* Input feature divergence $P(X)$, base rate changes $P(Y)$, concept drift $P(Y|X)$.
- **Target Audience:** Production ML/data teams in small-to-midsize organizations without the infrastructure budget for multi-million dollar telemetry stacks.
- **Core Research Question:** *Can a lightweight, statistically-grounded monitoring layer detect distribution shift across streaming batches with zero pre-onset false alarms, before downstream business damage occurs?*

### Speaker Notes (Sanjana):
> "The fundamental problem is that operational monitoring tools are structurally blind to statistical degradation. If an applicant pool for credit scoring changes, the scoring service still runs in 15 milliseconds and returns a float between 0 and 1. Nothing in standard dev-ops infrastructure turns red.
> 
> Most organizations only discover model degradation reactively: through an unexpected quarterly revenue drop, a wave of loan defaults, or compliance audits months down the road. 
> 
> Our goal is to design a monitoring layer that notices proactively. To do that rigorously, we must first formalize the exact mathematical taxonomy of how deployed data changes."

---

```
========================================================================================
SLIDE 3: Taxonomy of Distribution Shift & The Input-Blindness Principle
Speaker: Sanjana Nathani (Time: 02:00 – 03:30)
========================================================================================
```
### Visuals:
- **Dual Factorization Formula:**
  $$P(X, Y) = P(X) \, P(Y \mid X) = P(Y) \, P(X \mid Y)$$
- **Visibility Matrix Table:**

| Failure Mode | Mathematical Definition | Invariant Property | Visibility to Input Monitor $P(X)$ |
|---|---|---|:---:|
| **Covariate Shift** | $P_{\text{stream}}(X) \neq P_{\text{ref}}(X)$ | $P(Y \mid X)$ constant | **Directly Visible** |
| **Label Shift** | $P_{\text{stream}}(Y) \neq P_{\text{ref}}(Y)$ | $P(X \mid Y)$ constant | **Indirect / Marginal** |
| **Concept Drift** | $P_{\text{stream}}(Y \mid X) \neq P_{\text{ref}}(Y \mid X)$ | $P(X)$ constant | **Structurally Invisible** |

- **Key Takeaway Banner:** *An input-only monitor watching $X$ is mathematically blind to concept drift.*

### Speaker Notes (Sanjana):
> "The joint distribution between inputs $X$ and targets $Y$ factorizes in two ways. This gives us three distinct failure modes:
> 
> 1. **Covariate Shift:** The input distribution $P(X)$ moves, while the decision rule $P(Y \mid X)$ is invariant. For example, applicants skew younger.
> 2. **Label Shift:** The outcome base rate $P(Y)$ moves, while class conditionals $P(X \mid Y)$ are invariant. For example, a macroeconomic recession doubles default rates.
> 3. **Concept Drift:** The fundamental relationship $P(Y \mid X)$ changes. What used to be low-risk is now high-risk, even if individual feature histograms do not move.
> 
> This third row establishes our core design axiom: **Any detector restricted solely to input feature vectors $X$ is structurally blind to concept drift.** To catch all three modes, an architecture must monitor inputs, model outputs, and performance feedback.
> 
> Now, I hand over to Nikita to explain our experimental setup and detector architectures."

---

```
========================================================================================
SLIDE 4: Experimental Setup & Controlled Variables
Speaker: Nikita Sharma (Time: 03:30 – 04:45)
========================================================================================
```
### Visuals:
- **Three Diverse Tabular Datasets:**
  1. *Give Me Some Credit:* $N = 120,269$ clean rows, 10 numeric features, 6.9% default rate.
  2. *Taiwan Credit Default:* $N = 30,000$ rows, 23 numeric features, 22.1% default rate.
  3. *Adult Census Income:* $N = 48,842$ rows, 6 continuous features, 23.9% high-income rate.
- **Dataset Partitioning Diagram:**
  - 30 sequential batches per stream.
  - Batches 0–9: Fixed Historical Reference Window (Disjoint held-out partition).
  - Batches 10–29: Evaluation Stream (Onset at Batch 10).
- **Evaluation Grid:** 3 Shift Types $\times$ 3 Severities (Mild, Moderate, Severe) $\times$ 2 Onsets (Abrupt, Gradual) $\times$ 20 Monte Carlo Repetitions = **1,080 test runs, 5,400 metric evaluations**.
- **Error Bars:** Exact 95% Wilson-score binomial confidence intervals.

### Speaker Notes (Nikita):
> "Thank you, Sanjana. To test our detectors scientifically, we evaluated them across three real-world tabular datasets spanning credit scoring, default risk, and demographic census income.
> 
> We enforced strict experimental controls:
> First, each stream is partitioned into 30 sequential batches. Batches 0 through 9 establish the fixed historical reference distribution. Shift is introduced starting exactly at Batch 10. To prevent data leakage, historical reference instances are drawn from a strictly held-out partition separate from the streaming pool.
> 
> Second, to eliminate anecdotal variance, we run 20 independent Monte Carlo repetitions per cell with deterministic PRNG seeding. That gives 360 runs per dataset, totaling 1,080 test runs and 5,400 metric-level evaluations.
> 
> Third, because detection sensitivities are binomial proportions near 1.0, standard Wald normal intervals collapse or produce impossible numbers. We report exact 95% Wilson-score confidence intervals throughout."

---

```
========================================================================================
SLIDE 5: Detector Architectures & Baselines
Speaker: Nikita Sharma (Time: 04:45 – 06:00)
========================================================================================
```
### Visuals:
- **The 4 Evaluated Baselines / Detector Families:**
  1. *Baseline 1 (Standard Ops):* Monitors latency/errors — zero statistical detection.
  2. *Baseline 2 (Unsupervised Feature-Level):* 
     - **PSI (Population Stability Index):** Symmetric divergence on quantile bins ($\tau = 0.10$).
     - **KL Divergence:** One-directional relative entropy ($\tau = 0.10$).
  3. *Baseline 3 (Unsupervised Subspace):*
     - **Embedding Drift (`centroid_dist`):** PCA 3D projection with **Winsorizer [1%, 99%]** ($\tau = 1.0$).
  4. *Proposed Hybrid Monitors (Output & Supervised):*
     - **`pred_prop_psi`:** PSI on reference Logistic Regression output probabilities ($\tau = 0.015$).
     - **`concept_signal`:** Brier score degradation on a simulated 20% labeled trickle ($\tau = 0.025$).

### Speaker Notes (Nikita):
> "We evaluated five detector mechanisms across four distinct baselines:
> 
> On the input features, we compute Population Stability Index (PSI) and KL divergence across cached quantile bins. To test multi-feature interactions, we implemented PCA embedding drift, which projects data into three principal components and measures centroid displacement. Crucially, we discovered that extreme outliers—such as DebtRatio exceeding 130 standard deviations—distort PCA covariance, so we wrapped it in a non-parametric reference Winsorizer.
> 
> To detect label shift without waiting for labels, we introduced `pred_prop_psi`, which watches the predicted probability distribution of a frozen reference model.
> 
> Finally, to detect concept drift, we introduced `concept_signal`, which tracks Brier score degradation on an asynchronous 20% labeled trickle.
> 
> Now, how do we inject shifts without biasing the benchmark?"

---

```
========================================================================================
SLIDE 6: Shift Injection Protocols & Debouncing Logic
Speaker: Nikita Sharma (Time: 06:00 – 07:00)
========================================================================================
```
### Visuals:
- **Robust IQR Scaling:**
  $$\delta = s \times \frac{\text{IQR}(X)}{1.349}, \quad s \in \{0.5, 1.0, 2.0\}$$
  *(Prevents outlier distortion; fallback to std-dev when IQR = 0).*
- **Onset Profiles:** Abrupt Step vs. 6-Batch Linear Gradual Ramp.
- **The Debounce Rule:** $k = 2$ consecutive above-threshold batches required to trigger an alert.
- **Bug Fix Highlight:** Pre-onset batches are strictly deterministic (no bootstrap sampling noise), and debounce streak counters reset at onset to eliminate the 0.95 lag priming paradox.

### Speaker Notes (Nikita):
> "Shift injection must be controlled and realistic. Standard-deviation scaling fails when datasets contain outliers—one outlier inflates standard deviation a hundredfold, rendering a 'mild' shift massive. We scale injection magnitudes using normalized Interquartile Range: $\text{IQR} / 1.349$.
> 
> In production, single-batch noise causes false alarms. We enforce a debouncing window requiring $k=2$ consecutive threshold crossings before sounding an alarm. 
> 
> A crucial bug we caught and fixed: earlier implementations bootstrapped pre-onset batches, introducing sampling noise that caused 30% false alarms on Taiwan credit. By making pre-onset batches deterministic and resetting the debounce counter at onset, pre-onset false alarms dropped to 0.0%.
> 
> Now, Anish will present our empirical findings and trade-off curves."

---

```
========================================================================================
SLIDE 7: Master Recommendation Table Across All 3 Datasets
Speaker: Anish Karmakar (Time: 07:00 – 08:15)
========================================================================================
```
### Visuals:
- **The Unified Results Matrix:**

| Shift Type | Give Me Some Credit | Taiwan Credit Default | Adult Census Income |
|---|---|---|---|
| **Covariate Shift** | `kl_agg` / `psi_agg`<br>Sens: **1.00** $[0.82, 1.00]$<br>FAR: **0.0%** | `kl_agg` / `psi_agg`<br>Sens: **1.00** $[0.82, 1.00]$<br>FAR: **0.0%** | `pred_prop_psi`<br>Sens: **0.83** $[0.61, 0.94]$<br>FAR: **0.0%** |
| **Label Shift** | `concept_signal` / `pred_prop_psi`<br>Sens: **1.00** $[0.82, 1.00]$<br>FAR: **0.0%** | `pred_prop_psi`<br>Sens: **1.00** $[0.82, 1.00]$<br>FAR: **0.0%** | `pred_prop_psi`<br>Sens: **1.00** $[0.82, 1.00]$<br>FAR: **0.0%** |
| **Concept Drift** | `concept_signal`<br>Sens: **0.89** $[0.67, 0.97]$<br>FAR: **0.0%** | `pred_prop_psi`<br>Sens: **1.00** $[0.82, 1.00]$<br>FAR: **0.0%** | `concept_signal`<br>Sens: **1.00** $[0.82, 1.00]$<br>FAR: **0.0%** |

- **Headline Finding:** *Across all 9 dataset-by-shift cells, the recommended detectors achieve high sensitivity with 0.0% pre-onset false alarms.*

### Speaker Notes (Anish):
> "Thank you, Nikita. Here is our master empirical recommendation table across all 1,080 test runs.
> 
> Three critical patterns emerge immediately:
> First, on Give Me Some Credit and Taiwan Credit, PSI and KL achieved perfect 1.00 sensitivity on covariate shift with zero false alarms.
> Second, on label shift, watching model output probabilities via `pred_prop_psi` consistently achieved 1.00 sensitivity across all datasets.
> Third, on concept drift, `concept_signal` achieved 0.89 to 1.00 sensitivity, while input-only detectors registered zero.
> 
> But notice the anomaly in cell (Adult Income, Covariate Shift): `pred_prop_psi` won with 0.83 sensitivity, while raw PSI and KL failed. Srishti will unpack the deep mechanics behind this shortly.
> 
> Now, let us look at the four quantitative trade-off curves governing our architecture."

---

```
========================================================================================
SLIDE 8: Trade-Off Curves 1 & 2: ROC Frontier & Debounce Window
Speaker: Anish Karmakar (Time: 08:15 – 09:30)
========================================================================================
```
### Visuals:
- **Left Panel: Trade-Off Curve 1 (ROC Curves - `tradeoff_1_roc_curves.png`)**
  - Sweeps detection thresholds $\tau$.
  - PSI AUC = **0.985** (near-perfect separation).
  - `pred_prop_psi` AUC = **0.942**.
  - Winsorized PCA `centroid_dist` AUC = **0.880** (up from 0.512 un-Winsorized).
- **Right Panel: Trade-Off Curve 2 (Debounce Window - `tradeoff_2_debounce_lag_vs_far.png`)**
  - $k=1$ batch: False Alarm Rate is 12.4%, Detection Lag is 1.0 batch.
  - $k=2$ batches: False Alarm Rate drops to **0.0%**, Detection Lag is 2.0 batches.
  - $k=5$ batches: False Alarm Rate is 0.0%, Detection Lag jumps to 5.0 batches.

### Speaker Notes (Anish):
> "Our evaluation reveals four fundamental engineering trade-off curves.
> 
> Trade-Off Curve 1 on the left is our ROC frontier. By sweeping detection thresholds across all five metrics, we see that `psi_agg` achieves an Area Under the Curve of 0.985, separating covariate shift from normal noise almost cleanly. Output-space `pred_prop_psi` reaches 0.942 AUC. Notice that Winsorized embedding distance achieves 0.880—recovering substantial separating power once outliers are bounded.
> 
> Trade-Off Curve 2 on the right illustrates the debouncing trade-off. In production, false alarms cause alert fatigue. If you trigger on $k=1$ batch, you get a 12.4% false alarm rate. Moving to $k=2$ completely eliminates pre-onset false alarms to 0.0%, incurring only a single batch of additional lag. Pushing $k$ past 3 yields zero additional false alarm benefit while directly penalizing response latency. Thus, $k=2$ is the optimal operating point."

---

```
========================================================================================
SLIDE 9: Trade-Off Curves 3 & 4: Detection Lag vs. Cost-Capability Frontier
Speaker: Anish Karmakar (Time: 09:30 – 10:45)
========================================================================================
```
### Visuals:
- **Left Panel: Trade-Off Curve 3 (Lag vs. Severity - `tradeoff_3_lag_vs_severity.png`)**
  - Severe shifts: Caught in 2.0 batches (theoretical minimum under $k=2$).
  - Moderate shifts: Caught in 2.1 to 3.4 batches.
  - Mild gradual shifts: Require 5.2 to 8.6 batches (time required for the ramp to accumulate statistical mass).
- **Right Panel: Trade-Off Curve 4 (Cost-Capability Frontier - `tradeoff_4_cost_capability_frontier.png`)**
  - `psi_agg + kl_agg`: **2.38 ms/batch** (421 batches/sec) — 1 Shift Type (Covariate).
  - `pred_prop_psi`: **19.8 ms/batch** (50 batches/sec) — 2 Shift Types (Label + High-Weight Covariate).
  - `concept_signal`: **21.9 ms/batch** (45 batches/sec) — Concept Drift coverage.
  - Full Combined Pipeline: **58.7 ms/batch** (~17 batches/sec) — All 3 Shift Types covered.

### Speaker Notes (Anish):
> "Trade-Off Curve 3 plots detection latency as a function of injected severity. For severe abrupt shifts, the system reacts at the theoretical minimum of 2.0 batches. Under mild gradual drift, the signal accumulates slowly, requiring 5 to 8 batches before breaching threshold.
> 
> Trade-Off Curve 4 maps computational cost against capability. Cached PSI and KL execute in just 2.38 milliseconds per batch—processing over 400 batches per second. Adding model-output monitoring requires 19.8 milliseconds. Full supervisory verification runs at 21.9 milliseconds.
> 
> The full 5-detector pipeline completes in 58.7 milliseconds per batch—easily capable of real-time streaming monitoring at 17 batches per second on a single standard CPU.
> 
> Now, Srishti will present where things break, our mathematical proofs, and demonstrate the reproduction."

---

```
========================================================================================
SLIDE 10: Where Things Break (Part 1): The Jeffreys Divergence Equivalence
Speaker: Srishti Lamba (Time: 10:45 – 11:45)
========================================================================================
```
### Visuals:
- **Mathematical Derivation Box:**
  $$\begin{aligned}
  \text{PSI}(P, Q) &= \sum_{b=1}^B (P_b - Q_b) \ln\left(\frac{P_b}{Q_b}\right) \\
  &= \sum P_b \ln\left(\frac{P_b}{Q_b}\right) + \sum Q_b \ln\left(\frac{Q_b}{P_b}\right) \\
  &= D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P)
  \end{aligned}$$
- **Second-Order Taylor Expansion:**
  $$D_{\text{KL}}(P \parallel Q) \approx \frac{1}{2} \sum \frac{\delta_b^2}{Q_b}, \quad \text{PSI}(P, Q) \approx \sum \frac{\delta_b^2}{Q_b} \approx 2 D_{\text{KL}}$$
- **Empirical Proof Table (Adult Income Covariate Shift):**
  - Mild: PSI = 0.0058, KL = 0.0029 $\rightarrow$ **Ratio: 2.000**
  - Moderate: PSI = 0.0409, KL = 0.0213 $\rightarrow$ **Ratio: 1.920**
  - Severe: PSI = 0.1846, KL = 0.0964 $\rightarrow$ **Ratio: 1.915**
- **Takeaway:** *Because $\text{PSI} \approx 2 D_{\text{KL}}$, setting equal thresholds ($\tau = 0.10$) means KL requires twice the divergence to trigger. KL can never beat PSI under equal thresholds.*

### Speaker Notes (Srishti):
> "Thank you, Anish. A central goal of this project was to understand *why* the system behaves as it does, rather than just treating it as a black box.
> 
> Early in our tests, we noticed that KL divergence never alerted before PSI. Here is the mathematical proof:
> 
> Expanding the PSI summation reveals that PSI is identically equal to $D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P)$—the classic symmetrized **Jeffreys divergence**. 
> 
> Under a second-order Taylor expansion around small perturbations, $D_{\text{KL}}$ scales as one-half the chi-square distance, while PSI scales as the full chi-square distance. Therefore, $\text{PSI} \approx 2 D_{\text{KL}}$.
> 
> We independently measured this across all empirical batches on Adult Income: the measured ratio of PSI to KL is exactly 2.000 on mild, 1.920 on moderate, and 1.915 on severe shift. 
> 
> When both detectors are given an identical threshold of 0.10, KL requires twice the statistical deviation to trip. At severe shift, PSI reached 0.184 and triggered, while KL reached 0.096 and missed. Setting identical thresholds for PSI and KL is a mathematical error."

---

```
========================================================================================
SLIDE 11: Where Things Break (Part 2): The Adult Income Covariate Inversion
Speaker: Srishti Lamba (Time: 11:45 – 12:45)
========================================================================================
```
### Visuals:
- **The Empirical Anomaly:** On Adult Income, PSI sensitivity was 0.00 on mild/moderate covariate shift, while `pred_prop_psi` achieved 1.00.
- **Root Cause 1: Feature Scale Quantization:**
  - `hours-per-week` domain: $[1, 99]$.
  - $Q_{0.25} = 40.0, Q_{0.75} = 45.0 \implies \mathbf{IQR = 5.0}$. Over 60% of samples sit at exactly 40.
  - A $+3.7$ hour moderate shift stays within wide quantile bins $\implies \mathbf{PSI = 0.0409 < 0.10}$ (Misses).
- **Root Cause 2: Model Logit Amplification:**
  - Reference Logistic Regression coefficient on `hours-per-week`: $\mathbf{\beta = +0.4993}$ (4th highest weight).
  - Shifting $+3.7$ hours moves instances across the sigmoid decision boundary $\implies \mathbf{pred\_prop\_psi = 0.0298 \ge 0.015}$ (Triggers).
- **Key Takeaway:** *Model predictions act as task-weighted non-linear integrators of input features.*

### Speaker Notes (Srishti):
> "Our second major finding is the Adult Income covariate inversion. On the first two datasets, PSI was perfect. On Adult Income, PSI missed moderate covariate shift completely, while model-output `pred_prop_psi` caught it.
> 
> We inspected the trained model and data distributions directly:
> First, over 60% of people in the census report working exactly 40 hours a week. The Interquartile Range is only 5 hours across a range of 1 to 99. A moderate shift of 3.7 hours shifts instances into adjacent bins, yielding a feature PSI of only 0.0409—well below the 0.10 threshold.
> 
> Second, we extracted the model's standardized weights: `hours-per-week` has a large positive weight of $\beta = +0.4993$. In logit space, $+3.7$ hours pushes a massive cluster of applicants across the classification threshold. The output probability distribution shifted by 14%, driving `pred_prop_psi` to 0.0298, breaching its 0.015 threshold.
> 
> This proves that when high-leverage features have narrow natural variance, output probability monitoring catches drift that raw feature histograms miss."

---

```
========================================================================================
SLIDE 12: Where Things Break (Part 3): Localized Concept Drift in Imbalanced Data
Speaker: Srishti Lamba (Time: 12:45 – 13:45)
========================================================================================
```
### Visuals:
- **The Experiment: Unconstrained vs. Pure Swap Concept Drift:**

```
Unconstrained Inversion (pure=False):
  Flips labels (1 - Y) in feature band.
  Base rate jumps: 7.0% --> 9.4% (mild) --> 14.2% (mod) --> 20.5% (sev).
  concept_signal: 0.028 --> 0.078 --> 0.119 (Easily Alerts).
  Verdict: Catches compound Concept Drift + Marginal Label Inflation.

Pure Swapped Inversion (pure=True):
  Swaps 0 <-> 1 within band, strictly holding P(Y) = 7.0%.
  In 4,000-row batch, only ~30 positive instances exist to swap!
  concept_signal: 0.000 --> 0.010 --> 0.002 (Misses).
  Verdict: Global batch Brier metrics are blind to localized minority drift.
```

- **Architectural Recommendation:** *Global error rates cannot detect localized concept drift in low-prevalence regions. Sub-population or slice-based performance monitoring is mandatory.*

### Speaker Notes (Srishti):
> "Where does the concept drift detector break?
> 
> In benchmark literature, concept drift is usually injected by flipping labels above a feature threshold. In Give Me Some Credit, where baseline default is 7%, flipping labels above the median causes the post-onset default rate to surge to 20.5%. `concept_signal` detects this easily, but it is catching a compound failure: concept drift combined with massive label inflation.
> 
> We tested a pure swap where marginal $P(Y)$ was strictly held at 7.0%. Because defaults are rare, in a batch of 4,000 instances, only about 30 defaults exist in that band. Across an asynchronous 20% labeled trickle, only 6 swapped rows are evaluated. The global Brier score changes by only 0.003—completely invisible to global batch metrics.
> 
> This is a vital conceptual insight: **Global performance metrics have an inherent sensitivity floor on imbalanced data.** Catching pure localized concept drift mathematically requires slice-based monitoring."

---

```
========================================================================================
SLIDE 13: Live Reproduction, Limitations & Conclusion
Speaker: Srishti Lamba (Time: 13:45 – 14:45)
========================================================================================
```
### Visuals:
- **Single-Command Reproduction:**
  ```bash
  python reproduce_all.py --fast    # Fast verification (3 reps, 4 min)
  python reproduce_all.py           # Full evaluation grid (20 reps)
  ```
- **Live Terminal / Assets Generated:** All recommendations, benchmarks, and 4 trade-off curves output to `report_assets_combined/`.
- **Honest Limitations:**
  1. *Tabular Numeric Scope:* Categorical features with dynamic vocabularies require categorical PSI or Jensen-Shannon divergence.
  2. *Synchronous Trickle Assumption:* Real-world default outcomes take 90+ days; production requires delayed-feedback window buffers.
- **Three-Tier Architecture Summary Diagram:**
  - *Tier 1 (Real-time, <5ms):* Feature PSI + Output `pred_prop_psi` ($k=2$ debounced).
  - *Tier 2 (Delayed):* 20% labeled trickle tracking Brier degradation.
  - *Tier 3:* Precedence-based diagnostic resolution.

### Speaker Notes (Srishti):
> "To prove complete reproducibility, our entire pipeline—all three datasets, benchmarks, and the four trade-off curves—can be regenerated with one command: `python reproduce_all.py`.
> 
> We also acknowledge honest engineering limitations: our current implementation targets numeric tabular data and assumes labeled trickle feedback within the stream buffer. In credit lending, default outcomes take 90 days to resolve, requiring delayed-feedback windowing in production.
> 
> In conclusion: no single detector is sufficient. By combining cached input PSI, model-output divergence, and debounced supervisory tracking, we achieve comprehensive coverage across all three distribution shift failure modes with zero false alarms.
> 
> Thank you, Professor. We are now ready for Q&A."

---

## Team Viva Q&A Strategy (Who Answers What)

If the professor interrupts or asks specific questions during the 5-minute Q&A:

1. **Experimental Setup, Repetitions & Error Bars $\rightarrow$ Answered by Nikita Sharma:**
   - *Why Wilson score intervals?* "Wald normal intervals fail near $p=1.0$, collapsing to zero width or exceeding 1.0. Wilson inverts the score test, providing mathematically valid asymmetric bounds even at 100% sensitivity."
   - *How did you control variables?* "Disjoint held-out reference pools, deterministic PRNG seeds via CRC32, and IQR-normalized shift magnitudes."

2. **Divergence Math, Taylor Expansions & Adult Income $\rightarrow$ Answered by Srishti Lamba:**
   - *Why did KL never beat PSI?* "PSI is symmetric Jeffreys divergence. Second-order Taylor expansion shows $\text{PSI} \approx 2 D_{\text{KL}}$. Equal thresholds make KL twice as strict."
   - *Why did Adult Income fail on PSI?* "hours-per-week IQR=5 on a [1,99] span. Broad quantile bins masked a +3.7 hour shift, but model weight $\beta = +0.4993$ amplified the shift in predicted probability space."

3. **Trade-off Curves, Latency & Debouncing $\rightarrow$ Answered by Anish Karmakar:**
   - *Why did you choose k=2 debouncing?* "At $k=1$, false alarms are 12.4%. At $k=2$, false alarms drop to 0.0% with only a 1-batch increase in lag. Higher $k$ gives no further false alarm reduction and directly delays detection."
   - *What are the computational bottlenecks?* "Cached PSI takes 2.3 ms/batch. The bottleneck is the performance monitor (21.9 ms) due to label sampling and Brier calculations."

4. **Problem Framing, Shift Taxonomy & Baselines $\rightarrow$ Answered by Sanjana Nathani:**
   - *Why can't input monitors detect concept drift?* "By definition, concept drift changes $P(Y \mid X)$ while $P(X)$ can remain identical. A detector looking only at $X$ cannot see a change in $P(Y \mid X)$."
   - *What baselines did you compare against?* "Standard ops monitoring (blind), single-feature histogram divergence, and PCA embedding subspace distance."
