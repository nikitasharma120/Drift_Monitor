# Generalization Test #2 — Adult / Census Income Dataset

**A second companion package to the main "A System That Notices" project (DS635).**

If you haven't read the main project's `README.md` yet, read that first.
If you've also read `datasets/taiwan_credit_default/README.md`, even
better — this document assumes you know both, and focuses on what a
*third*, deliberately different dataset reveals.

> **The question this package answers:** the Taiwan test showed the
> design holds up on a second dataset in the *same domain* (credit risk).
> Does it hold up on a dataset that isn't about credit at all?

**Short answer: mostly, but this dataset broke something neither of the
first two did, and inverted one of our most confident earlier findings.**
Both are reported here in full, because that's more useful than a third
clean success story would have been.

## 1. Why a Different Domain, Not Another Credit Dataset

Give Me Some Credit and Taiwan Credit Default both predict financial
default from financial features (debt ratios, bill amounts, payment
history). Even though Taiwan is a different dataset, it's arguably the
same *kind* of problem wearing different numbers. A much stronger test of
generalization is a dataset that shares no feature semantics with either.

## 2. The Dataset

**UCI "Adult" / "Census Income"** (Becker & Kohavi, 1996) —
`adult_income.csv` in this folder. Predicts whether a person's annual
income exceeds $50,000 from census/demographic data: age, education,
hours worked per week, capital gains/losses, and a sampling weight
(`fnlwgt`).

- 48,842 rows, no missing values in the numeric columns used here
- 23.9% positive rate (>$50K) — a different class balance than either
  earlier dataset (7% and 22.1%)
- The original label column (`"income"`, text: `"<=50K"` / `">50K"`) was
  converted to a binary `label` column before use; `drift_monitor`'s
  loader expects an already-binary label, so this one conversion was done
  once outside the shared package rather than adding string-label
  handling into it

**What changed vs. both earlier datasets:** again, nothing in
`drift_monitor/`. Only this folder's `generate_report_assets_adult.py`
config differs — which columns, which label, which target feature.

## 3. A New Failure Mode This Dataset Exposed: Zero-IQR Features

Before running anything, we checked this dataset's numeric features the
same way the main README's bug #1 taught us to. Two of them —
`capital-gain` and `capital-loss` — have an **interquartile range of
exactly zero**: most people report $0 in either category, so both the
25th and 75th percentiles land on zero.

Our severity scaling (main README, bug #1) computes injected shift as
`severity_multiplier × (IQR / 1.349)`. If IQR is exactly zero, **every
severity level — mild, moderate, severe — injects exactly zero shift**,
silently. This is a sharper, more literal version of the outlier problem
Give Me Some Credit had; that dataset's problem was an IQR that was too
*small relative to extreme outliers*, not an IQR that is *exactly zero*.

**We avoided this by choosing `hours-per-week`** (a well-behaved,
non-degenerate continuous feature) as the target feature instead, and
are documenting the avoided failure explicitly rather than silently
picking a feature that happens to work. If you adapt this pipeline to a
new dataset, **check your candidate target feature's IQR before choosing
it** — a zero (or even very small) IQR relative to the feature's actual
range is now an explicit item on the pre-flight checklist this project's
README recommends.

## 4. Results

| Shift type | Best detector | Sensitivity | Matches earlier datasets? |
|---|---|---|---|
| Covariate shift | **`pred_prop_psi`** | 0.70 | ❌ **No — this is new** |
| Label shift | `pred_prop_psi` | 0.97 | ✅ Yes (same detector as Taiwan) |
| Concept drift | `concept_signal` | 1.00 | ✅ Yes (same detector, even stronger) |

### The genuinely new finding: PSI/KL are no longer the reliable covariate-shift detectors

On both earlier datasets, PSI and KL divergence moved in lockstep and
were *always* the strongest, most reliable detectors for covariate shift
(1.00 sensitivity, every severity, every onset, on both datasets). On
this dataset, that pattern doesn't hold:

| Metric | Mild | Moderate | Severe |
|---|---|---|---|
| PSI | 0.00 | 0.00 | **1.00** |
| KL divergence | 0.00 | 0.00 | **0.00** — misses even severe shift entirely |
| `pred_prop_psi` | 0.10 | **1.00** | **1.00** |

**PSI only catches the most severe case. KL divergence misses covariate
shift completely, at every severity we tested — the first time either
original detector has gone completely blind to the shift type they were
specifically built for.** The detector that ends up most reliable here is
`pred_prop_psi` — previously our label-shift specialist, now unexpectedly
the best covariate-shift detector on this dataset.

**Our best explanation (not fully certain — flagged as such):**
`hours-per-week`'s IQR is only 5 (on a 1-99 range, median 40). Our
severity-scaling approach — the same one that worked well on the first
two datasets — produces an injected shift of roughly 2-7 hours here,
which is a small *absolute* change relative to the feature's overall
range. PSI and KL both measure the raw shift in the feature's histogram
directly, so a small absolute shift is a small histogram shift, and both
struggle to detect it except at the largest severity. `pred_prop_psi`
doesn't measure the raw feature shift at all — it measures how much the
*model's predictions* change, and if the model has learned that
`hours-per-week` matters a lot for predicting income, even a small shift
in this feature can produce a *proportionally larger* shift in predicted
probabilities. This is a plausible, mechanistically consistent
explanation for what we measured, but we haven't independently verified
it (e.g. by checking the model's actual feature-importance for
`hours-per-week`) — flagging that limitation honestly rather than
presenting the explanation as more certain than it is.

**Why PSI and KL diverge from each other here (not just from expectations):**
less certain still. Our best guess is that this dataset's severity-scaled
shift is closer to an additive mean-shift than a shape change, and PSI's
linear per-bin penalty vs. KL's log-ratio penalty may weight that
differently — but we would want to inspect the actual per-bin
probabilities directly before stating this as settled, and haven't done
so here.

### Label shift and concept drift: the pattern holds

Both of these matched the earlier datasets closely: `pred_prop_psi`
remains the strongest label-shift detector (0.97, similar to Taiwan's
1.00), and `concept_signal` remains the only real concept-drift detector,
here even stronger (1.00) than on either earlier dataset.

## 5. What This Means for the Project's Overall Claims

**The claim that gets stronger:** "the system as a whole (all five
detectors plus `diagnosis.py`) generalizes across domains" — still holds.
Some detector correctly caught every shift type on every dataset tested,
across three genuinely different datasets now.

**The claim that needs revising:** "PSI/KL are reliably *the* covariate-shift
detectors" no longer holds universally. On this dataset, they're the
*weakest* link for covariate shift, and the detector we built to solve an
entirely different problem (label shift) turned out to be the strongest.
**We think the more accurate, more defensible claim is: "which detector
best catches covariate shift depends on how large the calibrated shift
ends up being relative to that feature's typical scale — and that
relationship is not yet fully understood from three datasets."** This is
a more honest, more useful thing to write in a report than the cleaner
but now-contradicted claim we made after only two datasets.

## 6. System Cost

The full five-detector pipeline runs fastest on this dataset of the
three tested (~15ms/batch, ~65 batches/sec) — this dataset has only 6
numeric features (vs. 10 and 20 on the other two), so PSI/KL/embedding
drift all have less work to do. Same relative ranking otherwise:
`concept_signal` is the most expensive single detector.

`report_assets_adult/system_info.json` records the exact environment this
was run in, per the main README's §8.2.

## 7. How to Run This Yourself

**This README lives in `datasets/adult_income/`.** The detection code
(`drift_monitor/`) is two folders up, at the repo root, shared unmodified
with both other datasets — see the repo's top-level `README.md`.

From the **repo root**, install dependencies once (covers all three datasets):

```bash
pip install -r drift_monitor/requirements.txt
```

Then, from **this folder**:

```bash
cd datasets/adult_income    # if not already here
python generate_report_assets_adult.py
```

`adult_income.csv` already sits in this same folder. Outputs land in
`report_assets_adult/` — same file structure as the other two dataset
folders. Takes roughly 3-5 minutes at 20 repetitions per scenario (this
is the smallest of the three datasets by feature count, hence the
fastest run).
