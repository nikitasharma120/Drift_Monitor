# Generalization Test — Taiwan Credit Card Default Dataset

**A companion package to the main "A System That Notices" project (DS635).**

If you haven't read the main project's `README.md` yet, read that first —
it explains what distribution shift is, what the three failure types are,
and what each of the five detectors does. This document assumes you
already know that, and focuses on one specific question:

> **Do the design choices we made — severity levels, alert thresholds,
> which detector to trust for which failure type — actually generalize to
> a different dataset, or were they secretly just tuned to fit one
> dataset's specific quirks?**

This is exactly the kind of question a good engineer should ask about
their own work before trusting it, so we tested it properly, on a second,
completely different real-world dataset — without changing a single line
of the actual detection code.

## 1. Why This Test Matters

Anyone can tune a system until it works perfectly on the one dataset in
front of them. That doesn't prove the system is any good — it might just
mean the numbers were quietly adjusted to fit. The only real way to check
is to point the *exact same, unmodified* code at a dataset it has never
seen, with no new tuning allowed, and see what happens.

**The rule we set for ourselves:** the `drift_monitor/` package folder
must be byte-for-byte identical to the main project's. Only the
*configuration* (which columns to use, what the dataset's label column is
called) is allowed to change. If we had to touch the detection logic
itself to make this dataset work, that would be a sign our original
design didn't really generalize — it would just mean we retuned it a
second time.

## 2. The Dataset

**UCI "Default of Credit Card Clients"** (Taiwan, 2005) — publicly
available, `UCI_Credit_Card.csv` in this package.

| | Give Me Some Credit (main project) | Taiwan Credit Default (this test) |
|---|---|---|
| Rows | 150,000 (120,269 after cleaning) | 30,000 (all usable, no missing values) |
| Default rate | ~7% | ~22% |
| Extreme outliers? | Yes — some values 100-1000× a normal value | Much milder — the largest unusual values are only about 2× a typical value |

We picked this dataset *specifically because* it's much cleaner than the
original one. If our outlier-handling fixes (see the main README, bugs
1, 2, and 4) were really just band-aids for one dataset's specific mess,
this is the dataset that would expose that — a "clean" dataset shouldn't
need any outlier protection to work, so if our system still behaves
sensibly here, that's meaningful evidence the underlying approach is
sound, not just a patch.

*(Retrieved from a public GitHub mirror, since the official UCI
repository wasn't reachable from the environment this project was built
in. The row count (30,000) and column structure match the official
dataset, which is a reasonable spot-check on authenticity.)*

## 3. What We Changed (and What We Deliberately Did Not)

**Changed:** only the dataset configuration — which columns count as
"features," which column is the yes/no outcome, and which single feature
stands in for the role `DebtRatio` played in the original dataset (here,
we used `BILL_AMT1`, the customer's most recent bill amount, since it's
the closest equivalent — a continuous financial number).

**Not changed, on purpose:** every severity level (what counts as "mild"
vs. "severe" injected shift), every alert threshold, and all five
detectors' actual code. If any of these needed adjusting to make this
dataset work, we would say so plainly — see §5.

## 4. Results

### The headline result: it mostly just worked

| Shift type | Best detector | Sensitivity here | Matches the original dataset? |
|---|---|---|---|
| Covariate shift | PSI, KL, or `pred_prop_psi` (three-way tie) | 1.00 | ✅ Yes |
| Label shift | `pred_prop_psi` (`concept_signal` nearly tied at 0.99) | 1.00 | ✅ Yes |
| Concept drift | `concept_signal` | 0.83 | ✅ Yes (weaker, but same detector wins) |

With **zero retuning**, the system correctly identified which detector to
trust for which failure type, on a dataset it had never seen, with a
completely different class balance (22% vs. 7%) and much cleaner data.
That's a genuinely positive, non-trivial result.

### The honest part: not everything transferred perfectly

**`pred_prop_psi` lost some of its precision.** On the original dataset,
this detector was cleanly silent (0.00 sensitivity) whenever covariate
shift happened — it correctly minded its own business. On this dataset,
it now also fires at 1.00 sensitivity on covariate shift, which isn't
supposed to be its job:

| Detector | Covariate shift (should be near 0) | Concept drift (should be near 0) |
|---|---|---|
| `pred_prop_psi` | **1.00** (leaking) | 0.20 (some leaking) |
| `concept_signal` | 0.03 (mostly fine) | — (this is its actual job) |

This means the prediction model's calibration — and therefore how
"specific" `pred_prop_psi` stays to label shift alone — is itself
somewhat dependent on the exact dataset it's trained on. This is a real
limitation, not something we're hiding.

**The good news: this doesn't actually cause wrong diagnoses in practice.**
The main project's `diagnosis.py` file doesn't read any one detector in
isolation — it gives PSI and KL divergence *priority* when deciding
whether something is covariate shift, precisely because they stayed
perfectly reliable (1.00 sensitivity, zero false alarms) on both
datasets. We re-ran that combined-diagnosis check on this dataset
specifically to confirm it still works:

```
                        Diagnosed as:
True shift type    "concept or mild label"   "covariate"   "label"   "no alert"
Concept drift              31.7%                 0.0%        0.0%      68.3%
Covariate shift             0.0%                95.0%        0.0%       5.0%
Label shift                10.8%                 0.0%       51.7%      37.5%
```

**Zero covariate-shift batches were ever misdiagnosed as concept drift or
label shift**, even though the individual detector leaked. The
combined-diagnosis approach absorbed the imperfection completely — which
is itself a meaningful result: the *system* is more robust than any one
of its individual parts.

### Concept drift is genuinely harder to catch on this dataset

Across all severities, 68.3% of concept-drift test batches got no alert
at all here, versus 34% on the original dataset. Looking at the live
chart for this scenario (`report_assets_taiwan/dashboard_concept.png`),
the detection signal is visibly noisier and sits right at the edge of the
alert threshold, rather than making a clean, obvious jump the way it did
on the original dataset. We believe this is a genuine property of this
dataset (its higher, 22% default rate and cleaner features change how
much a shift in prediction accuracy stands out against normal noise) —
not a mistake in our code, since nothing in the detection logic changed
between the two datasets.

## 5. Did We Have to Change Any Severity Levels or Thresholds?

**No.** This is worth stating plainly because it's the actual point of
this whole test. Every severity level and every alert threshold is
identical to the main project's. The results above are what you get by
pointing the unmodified system at new data — not the result of a second
round of tuning made to look successful after the fact.

## 6. System Cost on This Dataset

Every run now also saves `report_assets_taiwan/system_info.json` — a
record of the exact Python/library versions and machine that produced
these numbers (see the main project README, §8.2, for why this exists:
it's the record that would have caught our cross-environment
reproducibility bug on the first try instead of the fourth). If you
re-run this yourself, compare your `system_info.json` against the one in
this package before assuming any numeric difference is a real
generalization finding rather than an environment difference — `benchmark.csv`'s
timing numbers are expected to vary by machine; the sensitivity/lag/false-alarm
numbers in `evaluation_summary.csv` are not.

The full five-detector pipeline runs slightly *faster* here (~20
milliseconds per batch) than on the original dataset — likely because
there are no extreme outlier values to slow down the internal
calculations. The same relative pattern holds either way:
`concept_signal` is the most computationally expensive detector, and PSI
+ KL divergence (after the caching fix described in the main README) are
by far the cheapest.

## 7. What to Take Away From This Test

**For a claim of "this generalizes":** the core design — robust,
percentile-based severity scaling, and trusting different detectors for
different failure types — held up on a second, structurally different
dataset with zero retuning, and the combined-diagnosis logic proved even
more robust than any single detector alone.

**For a claim of "we're not overselling this":** one detector's
precision (`pred_prop_psi`) turned out to be somewhat dataset-dependent,
and concept drift is measurably harder to catch here than on the original
dataset. We're reporting both the success and the imperfection, because a
report that only shows the clean result on a second dataset chosen and
retuned to make that result clean wouldn't actually tell you anything
useful.

## 8. How to Run This Yourself

**This README lives in `datasets/taiwan_credit_default/`.** The detection
code (`drift_monitor/`) is two folders up, at the repo root, shared
unmodified with the main dataset — see the repo's top-level `README.md`.

From the **repo root**, install dependencies once (covers both datasets):

```bash
pip install -r drift_monitor/requirements.txt
```

Then, from **this folder**:

```bash
cd datasets/taiwan_credit_default    # if not already here
python generate_report_assets_taiwan.py
```

`UCI_Credit_Card.csv` already sits in this same folder. Outputs land in
`report_assets_taiwan/` (inside this dataset folder) — same file
structure as the main project's `report_assets/` (dashboards, heatmaps,
CSVs, `benchmark.csv`, `system_info.json`). Takes roughly 10-15 minutes
at the default 20 repetitions per scenario.
