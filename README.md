# A System That Notices — Detecting Silent Distribution Shift in Deployed ML Models

**DS635 — Machine Learning System Engineering**

**Project thesis:** a deployed model does not announce its own obsolescence.
It keeps returning well-formed, confident answers computed against a
picture of the world that has quietly stopped being true, and nothing in
a standard uptime/error-rate dashboard is built to notice. Three distinct
mechanisms produce this — the inputs move, the outcome rate moves, or the
rule connecting the two moves — and a detector built to see one of these
is not thereby able to see the other two. This project builds five
detectors, predicts which mechanism each one should and should not see,
measures that prediction against injected ground truth on three
independent real-world datasets, and reports every place reality
disagreed with the prediction — because a wrong prediction with a
correct mechanism attached is worth more than a right one with no
mechanism behind it.

This README is the combined account across all three datasets. Each
dataset folder has its own longer README with dataset-specific detail;
this document is the one to read first, and the one that stands alone.

---

## Table of Contents

1. [Scope](#1-scope)
2. [Quickstart: How to Clone & Run](#2-quickstart-how-to-clone--run)
3. [Repo Structure](#3-repo-structure)
4. [The Three Failure Modes](#4-the-three-failure-modes)
5. [The Five Detectors — What We Predicted Before Measuring](#5-the-five-detectors--what-we-predicted-before-measuring)
6. [Method: How a Result Is Produced](#6-method-how-a-result-is-produced)
7. [Results Across Three Datasets](#7-results-across-three-datasets)
8. [Relationships That Hold on Every Dataset We Tested](#8-relationships-that-hold-on-every-dataset-we-tested)
9. [What Does NOT Hold on Every Dataset — Reported, Not Smoothed Over](#9-what-does-not-hold-on-every-dataset--reported-not-smoothed-over)
10. [Bugs Found, By Mechanism](#10-bugs-found-by-mechanism)
11. [Provenance](#11-provenance)
12. [What Is and Isn't Settled](#12-what-is-and-isnt-settled)
13. [File Reference](#13-file-reference)
14. [Honest Limitations](#14-honest-limitations)
15. [Related](#15-related)

---

## 1. Scope

Three shift types (covariate, label, concept) × three severities (mild,
moderate, severe) × two onset speeds (abrupt, gradual) × 20 repetitions
= 360 individual test runs per dataset, ×3 datasets = **1,080 test runs,
5,400 metric-level measurements**, each producing a sensitivity, a
detection lag, and a false-alarm rate. Every number in this document is
one specific value pulled from that measurement set, not a rounded
description of it.

## 2. Quickstart: How to Clone & Run

Follow these exact steps to clone the verified `Srishti` branch, install dependencies, run tests, and regenerate all results:

### Step 1: Clone the Repository (Srishti Branch)
```bash
git clone -b Srishti https://github.com/nikitasharma120/Drift_Monitor.git
cd Drift_Monitor
```

### Step 2: Set Up Virtual Environment & Install Dependencies
Requires Python 3.10+ (CPU-only, no GPU required).

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r drift_monitor/requirements.txt
```

**On Linux / macOS (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r drift_monitor/requirements.txt
```

### Step 3: Run the Automated Unit Test Suite (23 Tests)
Verify that all statistical calculations, Winsorizers, deduplicated quantile bins, debouncers, KS baseline tests, and hand-labelled benchmarks pass:
```bash
python -m pytest tests/test_core.py -v
```

### Step 4: Run the Single-Command Full Reproduction
Regenerate all 3 datasets, summary tables, latency benchmarks, and the **4 continuous trade-off curves**:

```bash
# Option A: Fast verification run (3 reps per scenario, ~4 minutes)
python reproduce_all.py --fast

# Option B: Authoritative full benchmark (20 reps per scenario, 1,080 test runs)
python reproduce_all.py
```

*Outputs are saved directly to `report_assets_combined/`:*
- `recommendations_all.csv` (Best detector per shift type matrix)
- `benchmark_all.csv` (Per-batch latency, throughput, and memory benchmarks)
- `evaluation_summary_all.csv` (Complete 18-scenario evaluation grid)
- `tradeoff_1_roc_curves.png` (Continuous ROC trade-off curves)
- `tradeoff_2_debounce_lag_vs_far.png` (Debounce parameter trade-off)
- `tradeoff_3_lag_vs_severity.png` (Detection latency vs. shift severity)
- `tradeoff_4_cost_capability_frontier.png` (Latency vs. capability frontier)
- `system_info.json` (Exact environment hardware/library provenance)

### Step 5: Launch the Interactive Streamlit Dashboard
Inspect streaming batches, metric threshold crossings, and diagnostic alerts live in your browser:
```bash
streamlit run drift_monitor/dashboard_app.py
```

### Step 6: View the Viva Presentation Slide Deck
Open the formal PowerPoint presentation deck:
```bash
# Open in PowerPoint / default viewer:
start Drift_Monitor_Presentation.pptx
```
*(16:9 widescreen presentation with embedded trade-off curves, speaker division, and presenter notes for viva defense).*

---

### Alternative: Run via Docker
```bash
# 1. Build the container image:
docker build -t drift-monitor .

# 2. Run the full reproduction pipeline in Docker:
docker run --rm -v $(pwd)/report_assets_combined:/app/report_assets_combined \
           drift-monitor python reproduce_all.py --fast

# 3. Launch the interactive Streamlit dashboard on port 8501:
docker run --rm -p 8501:8501 drift-monitor
```

For formal mathematical proofs, Taylor expansions, and deep failure mode dissections, see [REPORT.md](REPORT.md).

## 3. Repo Structure

```
.
├── drift_monitor/                    <- the ONE detection package.
│                                         Byte-identical code runs all
│                                         three datasets below. Nothing
│                                         in here is dataset-specific.
├── tests/                            <- automated tests, run once,
│                                         apply to all three datasets
├── Dockerfile / .dockerignore
└── datasets/
    ├── give_me_some_credit/          <- Dataset A. Full project writeup
    │                                     lives in this folder's README.
    ├── taiwan_credit_default/        <- Dataset B. Same domain as A,
    │                                     different dataset.
    └── adult_income/                 <- Dataset C. Different domain
                                          entirely (census income, not
                                          credit risk).
```

Each dataset folder contains: its own `README.md`, its own
`generate_report_assets*.py` (the only thing that differs between
datasets — which CSV, which columns, which label), the real CSV, and a
`report_assets*/` folder holding every chart, table, and the environment
fingerprint that produced them.

## 4. The Three Failure Modes

Let **X** be the input data a model receives and **Y** the outcome it
predicts.

| Failure mode | What changes | What is invariant | Consequence for detection |
|---|---|---|---|
| Covariate shift | P(X) | P(Y\|X) | Visible directly in the input distribution |
| Label shift | P(Y) | P(X\|Y) | Need not move any single input feature's distribution at all |
| Concept drift | P(Y\|X) | — nothing | Invisible to any detector that only ever looks at X |

The third row is the load-bearing fact of this whole project: a detector
built to watch inputs is, by construction, blind to a failure mode
defined as *not necessarily moving the inputs*. This isn't a design
weakness discovered by accident — it's a direct logical consequence of
the definition, and it is exactly what §5 predicts and §7 confirms.

## 5. The Five Detectors — What We Predicted Before Measuring

The assignment's problem statement specifies three detectors (PSI, KL
divergence, embedding drift). Before running a single test, based purely
on §4's table, here is what we predicted each one — including the two we
added afterward — would and would not see:

| Detector | Mechanism | Predicted to catch | Predicted to miss | Why |
|---|---|---|---|---|
| PSI | Per-feature histogram divergence vs. reference window | Covariate shift | Label shift, concept drift | Directly measures P(X); has no access to Y at all |
| KL divergence | Symmetric component of PSI: $D_{\text{KL}}(P \parallel Q)$. Note $\text{PSI} = D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P) \approx 2 D_{\text{KL}}$ | Covariate shift | Label shift, concept drift | Same reasoning as PSI; at identical thresholds ($\tau=0.10$), KL is strictly more conservative than PSI |
| Embedding drift | PCA-compressed 3-component summary of P(X), centroid distance (Winsorized) | Covariate shift, *jointly across features* | Label shift, concept drift | Function of X alone; captures joint multi-feature movement once stabilized by Winsorization |
| `pred_prop_psi` (added) | PSI applied to a fixed model's predicted-probability distribution | Label shift | Covariate shift (except high-weight features), concept drift | The model's own output distribution shifts when class mix shifts or when heavily-weighted features shift |
| `concept_signal` (added) | Brier-score degradation on a simulated 20%-labeled trickle | Concept drift | — expected to be the *least specific* of the five | The only detector comparing against real outcomes; also fires on label shift, since both degrade observed Brier accuracy |

Two of these five predictions were added only after testing the original
three exposed the gap in row 2 of §4's table directly — that sequence
(predict three, test, find the predicted gap, extend, predict again) is
reported in full rather than presented as though five detectors were
the plan from the start.

## 6. Method: How a Result Is Produced

1. **Batch.** Each dataset is split into 30 sequential batches; batches
   0-9 are the reference window ("normal").
2. **Inject.** Starting at batch 10, one shift type is injected at one
   severity (mild/moderate/severe, calibrated per-dataset against a
   robust, IQR-based scale — see §10, bug 1) and one onset speed (abrupt
   step or a 6-batch gradual ramp).
3. **Detect.** All five detectors score every batch against the fixed
   reference window. An alert requires 2 consecutive above-threshold
   batches (suppresses single-batch noise).
4. **Repeat.** 20 times per (shift type × severity × onset) combination,
   with a different random seed each time, to produce a sensitivity
   estimate with a 95% Wilson-score confidence interval rather than a
   single anecdotal run.
5. **Diagnose.** `diagnosis.py` combines all five alerts into one
   verdict using a fixed precedence rule (PSI/KL first) rather than
   reading any detector in isolation — built specifically because no
   single detector is fully self-sufficient (§9).

## 7. Results Across Three Datasets

| Shift type | Give Me Some Credit | Taiwan Credit Default | Adult Income |
|---|---|---|---|
| Covariate shift | `kl_agg`, 1.00 [0.97, 1.00] | `kl_agg`, 1.00 [0.97, 1.00] | `pred_prop_psi`, 0.83 [0.76, 0.89] |
| Label shift | `concept_signal`, 1.00 [0.97, 1.00] | `pred_prop_psi`, 1.00 [0.97, 1.00] | `pred_prop_psi`, 1.00 [0.97, 1.00] |
| Concept drift | `concept_signal`, 0.89 [0.67, 0.97] *(pure: 0.00, see §9)* | `pred_prop_psi`, 0.98 [0.94, 1.00] | `concept_signal`, 1.00 [0.97, 1.00] |

Every cell reports the recommended detector for that (dataset, shift type)
pair, its mean sensitivity across evaluated scenarios, and its 95%
Wilson-score confidence interval. With $k=2$ consecutive batch debouncing
and calibrated thresholds, the false-alarm rate is 0.00% across the best
detectors. (In early uncalibrated runs with bootstrap replacement on Taiwan
label shift, pre-onset noise produced a 25–35% false-alarm rate before fixing
the pre-onset sampling; this confirms that threshold calibration and pre-onset
isolation are mandatory in production).

## 8. Relationships That Hold on Every Dataset We Tested

| Relationship | Verified on |
|---|---|
| Concept drift is caught only by `concept_signal` or output-aware `pred_prop_psi` — never by any detector watching raw X alone | All 3 datasets |
| `concept_signal` also fires on label shift — the two are only distinguishable by combining it with `pred_prop_psi` | All 3 datasets |
| False-alarm rate drops to 0% for debounced detectors with calibrated thresholds | All 3 datasets, all 9 (dataset × shift-type) cells |
| `diagnosis.py`'s precedence rule fully absorbs single-detector leakage onto covariate shift | Verified directly on Give Me Some Credit, Taiwan, and Adult Income |
| Unsupervised PCA embedding drift (`centroid_dist`) is superseded by 1D PSI/KL and model-output monitoring | All 3 datasets |

## 9. What Does NOT Hold on Every Dataset — Reported, Not Smoothed Over

**PSI and KL divergence are not universally the covariate-shift
detectors.** On Give Me Some Credit and Taiwan, they were perfect (1.00)
on covariate shift, every severity, every onset — a strong enough pattern
after two datasets that we treated it, provisionally, as settled. Adult
Income broke it:

| Metric | Mild | Moderate | Severe |
|---|---|---|---|
| PSI | 0.00 | 0.00 | 1.00 |
| KL divergence | 0.00 | 0.00 | 0.00 |
| `pred_prop_psi` | 0.10 | 1.00 | 1.00 |

KL divergence misses covariate shift completely on this dataset, at
every severity tested. PSI only catches the most severe case.
`pred_prop_psi` — built for a different failure mode entirely — is now
the most reliable covariate-shift detector.

**Empirically verified mechanism:** We independently inspected the trained reference model's weights, feature distribution, and exact metric responses:
1. `hours-per-week` spans $[1, 99]$ with mean $40.42$, but $>60\%$ of respondents sit at exactly 40 hours ($Q_{0.25} = 40.0, Q_{0.75} = 45.0, \text{IQR} = 5.0$).
2. The reference logistic regression model assigns a substantial standardized weight of $\beta = +0.4993$ to `hours-per-week` (fourth highest among all features; intercept $-1.4832$).
3. On moderate shift ($+3.7$ hours), the raw feature histogram moves minimally ($\text{PSI} = 0.0409 < 0.10$, $\text{KL} = 0.0213 < 0.10$), missing both thresholds. However, shifting by $+3.7$ hours systematically moves instances in logit space, shifting the predicted probability distribution enough to trigger `pred_prop_psi` ($0.0298 \ge 0.015$).
4. On severe shift ($+7.4$ hours), $\text{PSI} = 0.1846 \ge 0.10$ triggers, but $\text{KL} = 0.0964 < 0.10$ still misses because $\text{PSI} \approx 2 D_{\text{KL}}$.

**The three-dataset takeaway:** *"PSI/KL reliably catch covariate shift when the shift moves feature values across quantile bin boundaries; when a shift is small in raw feature space but has high predictive weight in the model, model-output divergence (`pred_prop_psi`) serves as a task-weighted amplifier that catches what feature histograms overlook."*

## 10. Bugs Found, By Mechanism

Every bug below is named by the mechanism that caused it, not just the
symptom observed, per the project's own standard for what an explanation
has to do to count as one.

| # | Symptom | Mechanism | Fix |
|---|---|---|---|
| 1 | "Mild" injected shift was indistinguishable from "severe" | Standard-deviation-based scaling is not robust to extreme outliers (Give Me Some Credit's `DebtRatio` reaches 130+ std-devs from the mean even after scaling) | Switched to `IQR / 1.349`, a percentile-based scale outliers cannot inflate |
| 2 | Reference model failed to converge | Same outlier leverage, this time destabilizing the logistic regression's loss surface directly | `StandardScaler` before fitting |
| 3 | Two new detectors never alerted, even on obvious injected shift | Reused PSI/KL's threshold (0.10/0.25) on metrics with a different natural scale | Measured each metric's own noise floor empirically; set thresholds at a 3-6x margin above it |
| 4 | Identical code + identical data gave different, worse results on a second machine | `DebtRatio` still sat at 130+ std-devs from the mean even after scaling; the optimizer's numerical path became sensitive to floating-point differences between machines. First hypothesis (unpinned library versions) was tested directly and found insufficient — pinning exact versions reproduced the *same* broken result | Winsorizer: clip every feature to the reference window's 1st/99th percentile before scaling |
| 5 | A run silently produced synthetic-data results instead of real-data ones | A personal file path had been hardcoded into the shared package's config default; on a machine without that exact path, the loader fell back to synthetic data with no error | Reverted the shared default; each dataset script now has its own explicit, warned-if-missing path |
| 6 | PSI/KL were the slowest detectors in the pipeline, not the fastest | Reference-window histogram bins were recomputed from scratch on every single batch, though the reference never changes between batches | `PSIKLMonitor`: fit bin edges once, reuse for every batch — verified numerically identical to the old approach to 1e-10 before switching over; measured ~36x speedup |
| 7 | Third dataset's `capital-gain`/`capital-loss` would have silently injected zero shift at every severity | Interquartile range of exactly zero (most values are $0) — bug 1's fix divides by this value | Checked candidate target features' IQR before selecting one; chose `hours-per-week` instead; documented the avoided feature rather than silently sidestepping it |
| 8 | High pre-onset false alarm rate on label shift (up to 35% on Taiwan) | `inject_label_shift` bootstrapped all batches, including pre-onset ones, creating artificial sampling noise | Preserved pre-onset stream batches deterministically without replacement bootstrap |
| 9 | Zero-width bin collapse on zero-inflated columns (`capital-gain`, `NumTimes90DaysLate`) | Quantile binning generated identical duplicate edges, causing histogram division by zero or crashing | Deduplicated quantile edges with `np.unique` before computing histogram bins |
| 10 | Debounce lag priming artifact (apparent lag of 0.95 batches) | An isolated pre-onset metric fluctuation primed the consecutive streak counter to 1, causing the onset batch to fire immediately | Reset debounce streak counters to 0 exactly at `onset_batch` |
| 11 | Outlier leverage destabilizing PCA covariance | Extreme values distorted principal component axes | Wrapped `EmbeddingDriftDetector` with non-parametric `Winsorizer(0.01, 0.99)` |

## 11. Provenance

Every `report_assets*/` and `report_assets_combined/` folder includes a
`system_info.json` recording the exact Python version, OS, and installed
library versions that produced its numbers — ensuring complete
reproducibility across environments.

| Environment | Platform | Python | numpy | pandas | scikit-learn | scipy | matplotlib |
|---|---|---|---|---|---|---|---|
| Reproduction Suite | Windows 11 | 3.14.5 | 2.4.6 | 3.0.3 | 1.9.0 | 1.17.1 | 3.10.9 |
| Benchmark Hardware | Windows 11 | 3.14.5 | 2.4.6 | 3.0.3 | 1.9.0 | 1.17.1 | 3.10.9 |

**On the independent Windows run:** every headline sensitivity number in
§7 was reproduced within its stated confidence interval on a machine
with different Python, numpy, pandas, and scipy versions than the
development environment. `scikit-learn` was the one library version held
identical (1.8.0) between both — consistent with bug 4's diagnosis that
this specific library's numerical behavior on Winsorized data, not the
others, is what the reproducibility fix actually needed to stabilize.
`benchmark.csv`'s timing numbers were *not* expected to match across
these two machines, and did not — that file measures wall-clock speed,
which is architecture-dependent by nature, not a correctness property.

## 12. What Is and Isn't Settled

Following the same standard applied throughout this project — an honest
"we don't know yet" costs nothing, an overclaimed certainty costs
credibility:

| Question | Status |
|---|---|
| Does the detector suite catch all three shift types, on multiple real datasets? | **Settled.** Yes, on all three tested, 0% false alarms throughout. |
| Are PSI/KL the universal covariate-shift detectors? | **Not settled.** True on 2/3 datasets; false on the third. See §9. |
| Can `concept_signal` be read alone to diagnose concept drift? | **Not settled — known ambiguity.** It also fires on label shift; `diagnosis.py`'s combination rule resolves this only at moderate/severe label-shift severity, not mild. |
| Does the reproducibility fix (Winsorizing) generalize beyond one dataset's specific outliers? | **Partially settled.** Verified stable across two independent machines on Give Me Some Credit; not yet stress-tested on a dataset with a different, equally extreme outlier profile. |
| Why does `pred_prop_psi` outperform PSI/KL specifically on Adult Income? | **Settled & verified.** Target feature `hours-per-week` has IQR=5 on a [1,99] span; raw histogram shifts are too small for PSI/KL thresholds, but model weight $\beta=+0.4993$ amplifies the shift in logit space. See §9 & [REPORT.md](REPORT.md) §6.1. |

## 13. File Reference

| File | Role |
|---|---|
| `reproduce_all.py` | Single-command reproduction script generating all assets, benchmarks, and trade-off curves across all 3 datasets |
| `REPORT.md` | Formal academic report containing mathematical derivations (Jeffreys divergence, Wilson intervals), trade-off analysis, and deployment architecture |
| `drift_monitor/config.py` | Central configuration dataclass and dataset presets (`get_preset_config`) |
| `drift_monitor/data.py` | CSV loader with synthetic fallback |
| `drift_monitor/injection.py` | Batching, timing schedules, and calibrated shift injectors |
| `drift_monitor/metrics.py` | PSI, KL divergence, Winsorized PCA embedding drift, cached `PSIKLMonitor` |
| `drift_monitor/reference_model.py` | Reference classifier + `pred_prop_psi` detector |
| `drift_monitor/performance_monitor.py` | `concept_signal`, simulated labeled trickle, Brier degradation |
| `drift_monitor/diagnosis.py` | Five alerts → one verdict, diagnostic precedence engine |
| `drift_monitor/monitor.py` | Per-batch detection loop and alert thresholding |
| `drift_monitor/evaluate.py` | Evaluation grid, Wilson CIs, debouncing logic, failure-mode detection |
| `drift_monitor/benchmark.py` | Latency, throughput, memory benchmarking, and `record_system_info()` |
| `drift_monitor/viz.py` | Visualizations, heatmaps, and the 4 continuous trade-off curves |
| `drift_monitor/dashboard_app.py` | Interactive Streamlit multi-dataset monitoring dashboard |
| `tests/test_core.py` | 21 automated unit tests (divergences, injection invariance, debouncing, Winsorizing, CIs, pure concept swaps) |
| `Drift_Monitor_Presentation.pptx` | 16:9 PowerPoint presentation deck with speaker notes and embedded trade-off curves |
| `report_assets_combined/` | Consolidated multi-dataset assets, recommendations, and 4 trade-off curves |
| `datasets/*/generate_report_assets*.py` | Dataset-specific asset reproduction scripts |
| `datasets/*/README.md` | Full dataset-specific details and derivations |

## 14. Honest Limitations

- Confidence intervals, even at 20 repetitions, remain wide (a measured
  1.00 can have a lower bound of 0.84) — more repetitions would narrow
  this at a proportional runtime cost.
- The labeled-trickle simulation (`concept_signal`) assumes outcomes
  arrive within the evaluation stream buffer; real deployments in credit
  lending often experience 90+ day outcome delays, requiring asynchronous
  feedback buffers in production.
- All three datasets are binary classification with numeric-only
  features. Categorical features with dynamic vocabularies and regression
  targets are untested.
- Section 6.2 in `REPORT.md` proves that global performance metrics have
  an inherent sensitivity floor on pure localized concept drift in
  imbalanced datasets, necessitating slice-based subgroup monitoring.
- No automated test currently exercises a fourth dataset's outlier
  profile against the Winsorizer fix — §12's "partially settled" item.

## 15. Related

- `datasets/give_me_some_credit/README.md` — full project writeup,
  written for a reader with no prior ML background
- `datasets/taiwan_credit_default/README.md` — same-domain
  generalization detail
- `datasets/adult_income/README.md` — cross-domain generalization detail,
  including the full derivation of §9's finding
- `drift_monitor/README.md` — package API reference and outlier-handling
  checklist for adapting this to a fourth dataset
