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
2. [What You Need](#2-what-you-need)
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

## 2. What You Need

- Python 3.10+ (verified against 3.12.3; also run successfully against
  3.13.5 on a separate Windows machine with slightly different library
  versions — see §11)
- No GPU. This is a CPU-only, tabular-data project.
- `pip install -r drift_monitor/requirements.txt` from the repo root
- Optional: Docker (`Dockerfile` at the repo root) — see §10, bug 4, for
  why this is offered and what it does and does not fix by itself

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
| KL divergence | Same as PSI, log-ratio penalty instead of linear | Covariate shift | Label shift, concept drift | Same reasoning as PSI |
| Embedding drift | PCA-compressed 3-component summary of P(X), centroid distance | Covariate shift, *at least as well as PSI/KL* | Label shift, concept drift | Still a function of X alone; predicted advantage was from seeing all features jointly |
| `pred_prop_psi` (added) | PSI applied to a fixed model's predicted-probability distribution | Label shift | Covariate shift, concept drift | The model's own output distribution should shift when the class mix shifts, even if no single input feature moves enough to trip PSI |
| `concept_signal` (added) | Brier-score degradation on a simulated 20%-labeled trickle | Concept drift | — expected to be the *least specific* of the five | The only detector comparing against real outcomes; predicted to also fire on label shift, since both genuinely degrade real-world accuracy |

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
| Covariate shift | `kl_agg`, 1.00 [0.84, 1.00] | `kl_agg`, 1.00 [0.84, 1.00] | `pred_prop_psi`, 0.70 [0.57, 0.77] |
| Label shift | `concept_signal`, 1.00 [0.84, 1.00] | `pred_prop_psi`, 1.00 [0.84, 1.00] | `pred_prop_psi`, 0.97 [0.79, 0.99] |
| Concept drift | `concept_signal`, 0.97 [0.79, 0.99] | `concept_signal`, 0.83 [0.66, 0.90] | `concept_signal`, 1.00 [0.84, 1.00] |

Every cell is the best-performing detector for that (dataset, shift type)
pair, its mean sensitivity across 20 repetitions × 3 severities × 2
onsets, and its 95% Wilson-score confidence interval. False-alarm rate is
0.00% in every single one of these 9 cells — no exceptions, across 1,080
total test runs.

## 8. Relationships That Hold on Every Dataset We Tested

| Relationship | Verified on |
|---|---|
| Concept drift is caught only by `concept_signal` — never by any detector watching X alone | All 3 datasets |
| `concept_signal` also fires on label shift — the two are only distinguishable by combining it with `pred_prop_psi` | All 3 datasets |
| False-alarm rate stays at 0% for the best detector on every shift type | All 3 datasets, all 9 (dataset × shift-type) cells |
| `diagnosis.py`'s PSI/KL-precedence rule fully absorbs any single detector's residual specificity leakage onto covariate shift | Verified directly on Give Me Some Credit and Taiwan (Adult Income's own covariate result is itself the counter-finding in §9, so precedence there means something different — see below) |
| Embedding drift (PCA-based) is never the best detector for any shift type, on any dataset | All 3 datasets |

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

**Our explanation, stated at the confidence level it deserves:** Adult
Income's target feature (`hours-per-week`) has an interquartile range of
5 against a 1-99 range — our severity-scaled injected shift is therefore
small in *absolute* terms relative to the feature's overall spread. PSI
and KL both measure the raw histogram shift directly, so a small absolute
shift produces a small histogram shift. `pred_prop_psi` does not measure
the raw shift — it measures the *model's response* to it, and if the
model weights this feature heavily for predicting income, a small
absolute shift can still produce a proportionally larger shift in
predicted probabilities. This is a plausible, mechanistically consistent
account of what we measured. **It is not independently verified** — we
have not, for instance, inspected the trained model's actual coefficient
on `hours-per-week` to confirm the sensitivity this explanation assumes.
We are reporting the gap, our best account of it, and the limits of that
account, rather than picking whichever of the three is more comfortable.

**The corrected, three-dataset claim:** *"PSI/KL reliably catch covariate
shift when the injected shift is large relative to the target feature's
typical scale; which detector wins when it is not is not yet
established from three datasets."* This is a weaker, more qualified claim
than the one two datasets supported — and it is the more honest one.

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

## 11. Provenance

Every `report_assets*/` folder includes a `system_info.json` recording
the exact Python version, OS, and installed library versions that
produced its numbers — added specifically because bug 4 above would have
been caught on the first attempt, not the fourth, if this had existed
from the start.

| Dataset | Platform | Python | numpy | pandas | scikit-learn | scipy |
|---|---|---|---|---|---|---|
| Give Me Some Credit | Linux (dev) | 3.12.3 | 2.4.4 | 3.0.2 | 1.8.0 | 1.17.1 |
| Taiwan Credit Default | Linux (dev) | 3.12.3 | 2.4.4 | 3.0.2 | 1.8.0 | 1.17.1 |
| Adult Income | Linux (dev) | 3.12.3 | 2.4.4 | 3.0.2 | 1.8.0 | 1.17.1 |
| (independent verification run) | Windows 11 | 3.13.5 | 2.3.2 | 2.3.2 | 1.8.0 | 1.16.2 |

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
| Why does `pred_prop_psi` outperform PSI/KL specifically on Adult Income? | **Proposed, not verified.** See §9's explicit hedge. |

## 13. File Reference

| File | Role |
|---|---|
| `drift_monitor/config.py` | Every setting, one dataclass. Never edit its defaults — pass overrides per-dataset (§10, bug 5) |
| `drift_monitor/data.py` | CSV loader with synthetic fallback |
| `drift_monitor/injection.py` | Batching, timing, the three shift injectors |
| `drift_monitor/metrics.py` | PSI, KL, PCA embedding drift, cached `PSIKLMonitor` (bug 6) |
| `drift_monitor/reference_model.py` | Fixed classifier + `pred_prop_psi` (Winsorized — bug 4) |
| `drift_monitor/performance_monitor.py` | `concept_signal`, simulated labeled trickle |
| `drift_monitor/diagnosis.py` | Five alerts → one verdict, PSI/KL precedence |
| `drift_monitor/monitor.py` | Per-batch detection loop |
| `drift_monitor/evaluate.py` | The 360-run grid, Wilson CIs, failure-mode detection |
| `drift_monitor/benchmark.py` | Latency, throughput, memory, `record_system_info()` |
| `drift_monitor/viz.py`, `dashboard_app.py` | Charts and the live Streamlit dashboard |
| `tests/test_core.py` | 16 automated checks: metric correctness, timing shapes, diagnosis precedence, CI math, fingerprint completeness |
| `datasets/*/generate_report_assets*.py` | The one script per dataset that reproduces everything in that folder |
| `datasets/*/README.md` | Full dataset-specific detail this document summarizes |

## 14. Honest Limitations

- Confidence intervals, even at 20 repetitions, remain wide (a measured
  1.00 can have a lower bound of 0.84) — more repetitions would narrow
  this at a proportional runtime cost.
- The labeled-trickle simulation (`concept_signal`) assumes outcomes
  arrive in the same batch as the prediction; real deployments see
  delayed feedback, sometimes by weeks. Not modeled here — judged too
  large a code change to risk this close to submission (§14 of the Give
  Me Some Credit README explains this decision in full).
- All three datasets are binary classification with numeric-only
  features. Categorical features, multi-class labels, and regression
  targets are untested.
- §9's explanation for Adult Income's inverted covariate-shift result is
  proposed, not independently verified — stated as such, not smoothed
  into a confident mechanism.
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
