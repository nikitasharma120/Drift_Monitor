# A System That Notices — Detecting Silent Distribution Shift in Deployed ML Models

**DS635 — Machine Learning System Engineering**

This README explains the entire project from scratch — what problem it solves,
why it's built the way it is, what every file does, what the results mean,
and what its limits are. No prior machine learning background is assumed;
every technical term is explained the first time it's used.

---

## Table of Contents

1. [The Problem, in Plain Language](#1-the-problem-in-plain-language)
2. [The Three Ways a Model Can Go Wrong](#2-the-three-ways-a-model-can-go-wrong)
3. [The Dataset](#3-the-dataset)
4. [How We Test the System (Since We Can't Wait for Real Drift)](#4-how-we-test-the-system-since-we-cant-wait-for-real-drift)
5. [The Five Detectors](#5-the-five-detectors)
6. [Every File in This Project, Explained](#6-every-file-in-this-project-explained)
7. [How to Run This Project](#7-how-to-run-this-project)
8. [Results](#8-results)
   - [8.1 What We Predicted Before We Measured](#81-what-we-predicted-before-we-measured-and-where-reality-disagreed)
   - [8.2 What Should and Shouldn't Reproduce Exactly](#82-what-should-and-shouldnt-reproduce-exactly-across-machines)
9. [Bugs We Found and Fixed](#9-bugs-we-found-and-fixed)
10. [Honest Limitations](#10-honest-limitations)
11. [What We Deliberately Did Not Do, and Why](#11-what-we-deliberately-did-not-do-and-why)
12. [Summary for a Reader in a Hurry](#12-summary-for-a-reader-in-a-hurry)

---

## 1. The Problem, in Plain Language

Imagine a security camera that's supposed to alert a guard whenever
something unusual happens. Now imagine that camera has a broken alarm —
it still records video perfectly, it still "sees" everything, but no
matter what happens in front of it, it never rings the alarm. That's what
happens to most machine learning models after they're deployed.

A machine learning model is a program trained on past data to make
predictions — for example, "will this loan applicant default on their
loan?" Once deployed, the model keeps making predictions, and on the
surface everything looks fine: the program runs, it returns an answer for
every request, no errors appear in any log. But the world the model was
trained on keeps changing — the economy shifts, customer behavior changes,
new types of applicants show up — and the model has no built-in way of
knowing that the ground has moved under it. It just keeps confidently
answering questions based on a picture of the world that may no longer be
true. This silent failure is called **distribution shift**, or informally,
**model drift**.

**Why this matters:** most companies only discover their model has drifted
after something bad has already happened — a spike in customer complaints,
a quarter of bad financial decisions, a compliance audit. By then the
damage is done. Large tech companies can afford dedicated monitoring teams
and infrastructure to catch this early. Most companies cannot.

**What this project builds:** a lightweight system that watches a model's
incoming data continuously and raises an alert the moment the data starts
looking meaningfully different from what the model was trained on — before
that difference has a chance to cause real damage. That's the "System That
Notices" in the title.

## 2. The Three Ways a Model Can Go Wrong

Not all drift looks the same. This project studies three distinct failure
patterns, because — as we discovered — each one requires a genuinely
different detection strategy.

### Covariate shift — "the inputs change"

The kinds of people/situations coming in change, but the underlying rule
connecting a situation to its outcome stays the same.

> *Example:* Your loan applicants suddenly skew younger on average. A
> young, high-debt applicant is still exactly as risky as it always was —
> nothing about *risk itself* has changed, but the *mix* of applicants has.

### Label shift — "the outcome rate changes"

The overall rate of the thing you're predicting changes, but people who
do have that outcome still look the same as they always did.

> *Example:* A recession hits and the overall default rate jumps from 7%
> to 15%. But a defaulting borrower still has the same kind of profile
> (high debt, missed payments) as before — there are just more of them.

### Concept drift — "the rule itself changes"

The exact same person, with the exact same numbers on paper, now means
something different than before.

> *Example:* A new regulation changes what counts as risky. A debt ratio
> that used to signal "safe" now signals "risky" — nothing about the
> applicant has changed, but what their numbers *mean* has.

These map to a simple statistical shorthand used throughout this project:
if **X** is the input data and **Y** is the outcome being predicted,

| Shift type | What changes | What stays the same |
|---|---|---|
| Covariate shift | P(X) — the distribution of inputs | P(Y\|X) — the rule mapping inputs to outcomes |
| Label shift | P(Y) — the overall outcome rate | P(X\|Y) — what each outcome group looks like |
| Concept drift | P(Y\|X) — the rule itself | — nothing; this is the hardest to detect |

## 3. The Dataset

We used **Give Me Some Credit**, a real, publicly available dataset from
Kaggle used for credit-risk prediction: given someone's financial history,
predict whether they'll experience serious payment delinquency within two
years.

- **150,000 people**, one row each
- After removing rows with missing income/dependents data: **120,269 usable rows**
- **10 numeric features** per person: things like their debt-to-income
  ratio, age, monthly income, how many credit lines they have open, and
  how many times they've been late on payments
- **~7% of people** in the dataset actually experienced delinquency (a
  realistic, imbalanced rate — most people don't default, which is
  exactly what makes this a hard, realistic prediction problem)

**One important, real-world messiness in this data** that ended up
shaping much of this project: two of the features (`DebtRatio` and
`RevolvingUtilizationOfUnsecuredLines`) contain a handful of extremely
large, almost certainly erroneous values — numbers in the hundreds of
thousands, when a normal, sensible value is well under 1. This is a
known, documented quirk of this specific dataset, not something we
introduced. It caused several of the bugs described in §9, and dealing
with it honestly (rather than just quietly deleting the bad rows) became
a real engineering theme of this project.

## 4. How We Test the System (Since We Can't Wait for Real Drift)

We can't wait months for real drift to happen in order to test whether
our system notices it. Instead, we **simulate** drift on purpose, in a
controlled way, so we know exactly what "the right answer" is and can
measure how well the system finds it.

**The setup:** we take the dataset and split it into 30 sequential
"batches" — imagine 30 days of incoming loan applications, one batch per
day. For the first 10 batches, everything is normal (this is called the
**reference window** — what "normal" looks like). Starting at batch 11,
we deliberately inject one of the three shift types described above.

**We vary two things about the injected shift, to make the test realistic:**

- **Severity** — mild, moderate, or severe. Real drift doesn't always
  announce itself loudly; sometimes it's subtle.
- **Onset speed** — *abrupt* (drift appears suddenly, like a policy
  change taking effect overnight) or *gradual* (drift creeps in slowly
  over about a week's worth of batches, like a slowly shifting economy).

This gives us **18 distinct test scenarios** (3 shift types × 3 severities
× 2 onset speeds), and we repeat each one **20 times** with different
random seeds, so we get a reliable, not-just-lucky answer — **360
configurations, 1,800 individual measurements** in total.

**For every single test, we measure three things:**

1. **Sensitivity** — did the system notice at all? (as a percentage of
   the 20 repeats)
2. **Detection lag** — how many batches did it take to notice, once the
   shift started?
3. **False-alarm rate** — did the system cry wolf before anything had
   actually changed?

## 5. The Five Detectors

The original plan for this project (and the literal wording of the
assignment's problem statement) called for **three** detectors. We ended
up building **five**, because testing the original three honestly showed
they weren't enough — a story explained fully in §8 and §9.

### The original three

1. **PSI (Population Stability Index)** — a standard statistic that
   compares two sets of numbers and outputs a single score representing
   "how different is this new batch's distribution from what I saw
   before?" A bigger number means more difference.
2. **KL divergence** — the one-directional component of PSI ($D_{\text{KL}}(P \parallel Q)$).
   Mathematically, $\text{PSI} = D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P)$
   (the symmetrized Jeffreys divergence, approximately $2 D_{\text{KL}}$). Because
   $\text{PSI} \ge D_{\text{KL}}$ always, under identical thresholds KL requires roughly
   twice the distribution shift to trigger, making it strictly more conservative.
3. **Embedding drift** — instead of checking one feature at a time, this
   compresses all 10 features into a simplified 3-number summary (using a
   technique called **PCA**, Principal Component Analysis) and checks
   whether that summary has moved. The idea was that this might catch
   more complex, multi-feature changes that PSI/KL (which check one
   feature at a time) could miss.

**These three all have one thing in common: they only ever look at the
input data (X).** That turns out to be their fundamental limitation.

### The two we added

4. **Predicted-proportion detector (`pred_prop_psi`)** — instead of
   watching the raw input data, this watches what a simple, fixed
   prediction model *thinks* about each incoming batch. If the model
   starts predicting "risky" more or less often than it used to, that's a
   sign something about the makeup of the incoming data has changed —
   even if no single input feature moved enough to trip PSI or KL.
5. **Performance monitor (`concept_signal`)** — this is the only detector
   that ever checks against **real, actual outcomes**. In a real deployment,
   you eventually learn whether a loan actually defaulted or not — just
   with a delay. This detector simulates that: a small, randomly-sampled
   slice of each batch (20%) "reveals" its true outcome, and we check
   whether the model's predictions are getting worse compared to real
   life, not just compared to its own past predictions.

**Why we needed these two:** the three original detectors only ever look
at input data. Label shift and concept drift, by definition, don't have
to change the input data at all — so a detector that only watches inputs
is structurally blind to them, no matter how well-tuned it is. This isn't
a hypothesis; we proved it by testing (§8).

## 6. Every File in This Project, Explained

```
repo-root/
├── README.md                          <- repo overview, links to both datasets
├── Dockerfile / .dockerignore         <- optional: run this project in a
│                                          pre-configured, guaranteed-to-work
│                                          environment (see §7)
├── tests/
│   └── test_core.py                   <- automated checks, shared by both datasets
├── drift_monitor/                     <- the actual reusable package —
│   ├── config.py                         SHARED, unmodified, across both
│   ├── data.py                           dataset folders below
│   ├── injection.py
│   ├── metrics.py
│   ├── reference_model.py
│   ├── performance_monitor.py
│   ├── diagnosis.py
│   ├── monitor.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── viz.py
│   ├── dashboard_app.py
│   ├── requirements.txt
│   └── README.md                      <- technical usage manual (API details)
└── datasets/
    ├── give_me_some_credit/           <- YOU ARE HERE
    │   ├── README.md                      <- this file
    │   ├── generate_report_assets.py      <- the one script that runs everything
    │   ├── cs-training.csv                <- the real dataset
    │   └── report_assets/                 <- every chart, table, and number
    │                                          this README refers to, already
    │                                          generated, including
    │                                          system_info.json (see §8.2)
    └── taiwan_credit_default/         <- the second-dataset generalization
        └── ...                            test — see its own README
```

| File | In plain terms, what it does | Why it exists |
|---|---|---|
| `drift_monitor/config.py` | One place holding every setting: which dataset, which columns, how severe "severe" means, etc. | So the whole system can be pointed at a brand-new dataset by changing one thing, not hunting through every file |
| `drift_monitor/data.py` | Loads the dataset from a CSV file. If no file is given, it makes up a fake but realistic-looking dataset instead. | Lets the whole project still run and be demoed even without the real data file on hand |
| `drift_monitor/injection.py` | Splits data into batches and deliberately injects the three kinds of fake shift described in §4 | This *is* the test — it creates the "answer key" the detectors are graded against |
| `drift_monitor/metrics.py` | Calculates PSI, KL divergence, and the PCA-based embedding drift (detectors #1-3) | The original three detectors |
| `drift_monitor/reference_model.py` | Trains a simple prediction model and builds detector #4 (`pred_prop_psi`) on top of it | Closes the label-shift blind spot |
| `drift_monitor/performance_monitor.py` | Builds detector #5 (`concept_signal`), which checks predictions against a simulated trickle of real outcomes | Closes the concept-drift blind spot |
| `drift_monitor/diagnosis.py` | Looks at all five detectors' alerts together and decides: "this looks like covariate shift" / "label shift" / "concept drift" / "nothing" | One detector alone can be ambiguous (see §9); combining them resolves most of that |
| `drift_monitor/monitor.py` | The main loop: for every incoming batch, run all five detectors and decide whether to sound an alarm | This is the actual "monitoring system" |
| `drift_monitor/evaluate.py` | Runs the full 360-configuration, 1,800-repeat test described in §4, and calculates how confident we can be in each result | Turns "it seems to work" into a precise, honest, defensible number |
| `drift_monitor/benchmark.py` | Times how long each detector takes to run, and how much memory it uses | A monitoring system that's too slow or too expensive to run isn't actually useful in practice — this checks that ours isn't |
| `drift_monitor/viz.py` | Draws the charts (dashboards, heatmaps) | Shared drawing code used by both the report and the live dashboard |
| `drift_monitor/dashboard_app.py` | An interactive web page where you can pick a scenario and watch the system detect it live | The "lightweight dashboard" this project promised to deliver |
| `tests/test_core.py` | Automatically checks that basic building blocks work correctly (e.g., "if nothing changed, the drift score should be exactly zero") | Catches mistakes automatically instead of relying on a human eyeballing a chart every time |
| `generate_report_assets.py` | Runs absolutely everything above, end to end, and saves every chart/table you see in this README | The one script to run if you want to reproduce this entire project from scratch |

## 7. How to Run This Project

**This README lives in `datasets/give_me_some_credit/`, inside the repo.**
The detection code itself (`drift_monitor/`) lives two folders up, at the
repo root, shared with the second dataset — see the repo's top-level
`README.md` for why it isn't duplicated per dataset. Everything below
assumes you're running from inside this folder.

### Option A: Plain Python (what we used throughout development)

From the **repo root**, install dependencies once (this covers both datasets):

```bash
pip install -r drift_monitor/requirements.txt
```

Then, from **this folder** (`datasets/give_me_some_credit/`):

```bash
python generate_report_assets.py
```

The `cs-training.csv` file already sits in this same folder, so
`generate_report_assets.py`'s `CSV_PATH` setting works as-is. This takes
roughly 10-15 minutes at the default 20 repetitions per scenario, prints
live progress, and leaves every chart and table in a new `report_assets/`
folder (also inside this dataset folder, not at the repo root).

**Important:** if you want to point this at a *different* CSV file later,
edit the `CSV_PATH` line inside `generate_report_assets.py` — do **not**
edit the similar-looking default inside `drift_monitor/config.py`. That
file's default is shared by the whole reusable package (used by both
datasets in this repo); editing it caused a real, confusing bug during
this project's development (explained in §9, bug #5).

To run the automated tests (from the **repo root** — they test the shared
package, not this dataset specifically):

```bash
cd ../..          # back to repo root, if you're inside this folder
pip install pytest
pytest tests/ -v
```

To launch the interactive dashboard (also from the **repo root**):

```bash
streamlit run drift_monitor/dashboard_app.py
```

(You must use `streamlit run`, not plain `python`, or it won't launch a
browser window properly.)

### Option B: Docker (recommended if you can — see why in §9, bug #4)

From the **repo root** (the Dockerfile lives there, alongside `drift_monitor/`):

```bash
docker build -t drift-monitor .
docker run -p 8501:8501 -v $(pwd)/datasets/give_me_some_credit/cs-training.csv:/app/cs-training.csv drift-monitor
```

## 8. Results

### Detection results (20 repeats per scenario, 95% confidence intervals)

| Shift type | Best detector | Sensitivity | Detection lag (severe case) | False alarms |
|---|---|---|---|---|
| Covariate shift | PSI or KL | 1.00 (CI: 0.84-1.00) | 1 batch | 0% |
| Label shift | `concept_signal` | 1.00 (CI: 0.84-1.00) | 1 batch | 0% |
| Concept drift | `concept_signal` | 0.97 (CI: 0.79-0.99) | 1 batch | 0% |

**In plain terms:** the finished, five-detector system reliably catches
all three kinds of drift, usually within a single batch of it starting,
and essentially never raises a false alarm. That's a strong, honest result.

**But here's the important, honest caveat:** if you only build the
*original three* detectors that the assignment's problem statement
literally describes, here's what actually happens:

| Shift type | PSI | KL divergence | Embedding drift |
|---|---|---|---|
| Covariate shift | 1.00 (good) | 1.00 (good) | 0.00 (misses it) |
| Label shift | 0.00 (misses it) | 0.00 (misses it) | 0.00 (misses it) |
| Concept drift | 0.00 (misses it) | 0.00 (misses it) | 0.00 (misses it) |

**The original three-detector design only ever catches covariate shift.**
It completely misses the other two failure modes, every single time,
regardless of how severe the drift is. This is the single most important
finding of the whole project — and it's *why* the system grew to five
detectors instead of three. We consider discovering this gap, and being
upfront about it, to be a more valuable result than if the original
three-detector design had simply worked as hoped.

### System cost (does this actually run fast enough to be useful?)

The complete five-detector pipeline processes one batch of data in
roughly 20 milliseconds (about 20-50 batches per second, depending on the
machine). For context, a real monitoring system typically only needs to
check new data every few minutes or hours, not many times per second — so
this system has enormous headroom and would run comfortably even on
modest hardware.

The most computationally expensive detector is `concept_signal` (the
performance monitor); PSI and KL divergence, after a performance fix
described in §9, are now the cheapest by a wide margin — about 35 times
faster than before that fix.

### 8.1 What We Predicted Before We Measured, and Where Reality Disagreed

Before running the evaluation grid, based purely on the theory in §2 (what
each shift type does and doesn't change), we wrote down what we expected
each detector to do — before looking at a single result. Here is that
prediction, next to what we actually measured:

| Detector | We predicted | We measured | Prediction correct? |
|---|---|---|---|
| PSI / KL | Catches covariate shift (it directly watches input distributions); misses label shift and concept drift (neither has to move the input distribution) | Exactly this: 1.00 on covariate, 0.00 on label and concept | ✅ Yes, cleanly |
| Embedding drift | At least as good as PSI/KL, since PCA retains information from *all* features at once, not just one at a time | The **weakest** detector throughout — never once the best choice for any shift type | ❌ No — and this is the most useful wrong prediction in the whole project |
| `pred_prop_psi` | Would catch label shift, since a model's predictions should shift when the input mix shifts | Confirmed on the original dataset; also confirmed, but *less precisely* (see §4 of the Taiwan generalization package), on a second dataset | ✅ Mostly — with an honest asterisk |
| `concept_signal` | Would catch concept drift, since it's the only detector comparing against real outcomes; expected it to be the LEAST specific of the five, since performance can degrade for any reason, not just concept drift | Confirmed on both points: it does catch concept drift, and it does also fire on label shift (has to be combined with `pred_prop_psi` via `diagnosis.py` to disambiguate) | ✅ Yes, on both counts |

**Why the embedding-drift prediction was wrong, and what that's actually
worth:** the theory-driven guess was that a representation capturing *all*
features should never do worse than looking at features one at a time.
The measured result contradicts that. The mechanism: PCA's 3 components
are fit to capture the directions of *largest overall variance* across
all 10 features — which is not the same thing as being sensitive to a
shift concentrated in *one* feature. When only `DebtRatio` shifts, that
shift has to compete for representation against the other 9 features'
ordinary variance, and it mostly loses. A correct-but-boring prediction
would have told us nothing; this wrong one told us exactly why compressing
inputs isn't automatically a stronger monitoring strategy than watching
them individually — which is the more useful thing to have learned.

### 8.2 What Should and Shouldn't Reproduce Exactly Across Machines

Not every number in this report is expected to come out identical if you
re-run it on a different computer, and it's worth being explicit about
which is which, rather than treating "the numbers matched" as the only
sign of a working system:

| Category | Should reproduce exactly? | Why |
|---|---|---|
| Detection sensitivity, detection lag, false-alarm rate | **Yes** — same numbers every time | Every random process here (batch shuffling, shift injection, the labeled-trickle sample) is seeded explicitly. Given the same code, same data, and same pinned library versions, these are deterministic. |
| `benchmark.csv` timing numbers (milliseconds, throughput) | **No** — expected to vary by machine | These measure wall-clock speed, which depends on CPU, current system load, and background processes. Compare the *ranking* of detectors (which one is fastest/slowest) across machines, not the raw millisecond values. |
| `system_info.json` | Deliberately different on every machine | This is the point of the file — see below. |

**On `system_info.json`:** every run of `generate_report_assets.py` now
saves a small environment fingerprint (Python version, OS, and the exact
installed version of every pinned library) alongside the results. We
added this *after* spending real time diagnosing bug #4 below the hard
way — if this fingerprint had existed from the start, the mismatch
between two machines' results would have been visible immediately by
comparing two `system_info.json` files side by side, instead of requiring
a separate multi-step investigation to discover. We're including it going
forward not because we expect to need it again, but because "record what
produced this number" should have been there from the first result, not
added after the fourth bug.



Every single bug listed here traces back to the same root cause: this
dataset has a small number of extreme, unrealistic values in two of its
features. Below is a plain-language walk-through, in the order we found
them.

**Bug 1 — "Mild" shift was just as extreme as "severe" shift.**
Our first version measured how big a shift to inject using a standard
statistical measure called "standard deviation." Because of the extreme
outlier values mentioned in §3, this measure was wildly inflated — so
even our smallest ("mild") injected shift ended up being enormous and
unrealistic, indistinguishable from our largest ("severe") one.
*Fix:* switched to a more outlier-resistant way of measuring typical
spread (based on percentiles rather than every single value equally).

**Bug 2 — The prediction model refused to train properly.**
The same extreme outlier values also confused the model-training process
itself, causing it to fail to converge to a sensible answer.
*Fix:* rescaled all the input numbers to a consistent range before
training.

**Bug 3 — New detectors seemed to never trigger, even on huge shifts.**
When we first added detectors #4 and #5, we reused the same alert
threshold as the original three. But the new detectors' scores turned out
to naturally live on a much smaller numeric scale, so the borrowed
threshold was set far too high to ever be crossed.
*Fix:* measured each new detector's typical "quiet, nothing's-wrong"
value directly, and set thresholds based on that, rather than reusing
someone else's number.

**Bug 4 — The exact same code gave different results on a different computer.**
This was the trickiest one. Running the identical code on the identical
data, but on a different machine, gave meaningfully different — and
noticeably worse — results. Our first guess was that the two machines had
different versions of the software libraries installed, so we pinned
every library to an exact version number. That did *not* fix it — we
re-tested and got the exact same wrong results again. Digging deeper, we
found the real cause: the extreme outlier values mentioned in §3 put the
model-training process in a fragile, borderline state where tiny,
essentially invisible differences in how two computers do arithmetic
could push the result to a meaningfully different (and much less
accurate) outcome.
*Fix:* we now deliberately cap ("clip") those extreme outlier values to a
sensible range before training the model, which removes the fragility at
its actual source. We verified the fix by running the full test twice and
getting byte-for-byte identical results both times.

**Bug 5 — The system silently used fake data instead of the real dataset.**
During development, someone (us) temporarily hardcoded a personal file
path into a settings file as a shortcut. Later, running the project on a
different computer — where that exact file path didn't exist — caused the
whole system to silently fall back to made-up demo data, with no error or
warning, producing confusing, inconsistent results.
*Fix:* removed the hardcoded personal path, and added a loud, impossible-to-miss
warning message that now prints if the dataset file can't be found.

**Bug 6 — Two of the fastest detectors were actually needlessly slow.**
PSI and KL divergence were recalculating a reference value from scratch
every single time they ran — even though that reference value never
actually changes between checks. This is like re-reading a textbook cover
to cover every time you want to check one fact in it.
*Fix:* calculate that reference value once, and simply reuse it — verified
this produces mathematically identical answers, just about 35 times faster.

## 10. Honest Limitations

- **Confidence intervals are still fairly wide.** Even at 20 repeats, a
  measured "1.00" sensitivity has a 95% confidence range as low as 0.84 —
  meaning we can't be fully certain the true rate isn't occasionally
  lower. More repeats would narrow this further, at the cost of more
  runtime.
- **The system sometimes can't tell mild label shift apart from concept
  drift.** Our combined-diagnosis logic (`diagnosis.py`) resolves this for
  moderate and severe label shift, but at the mildest severity, both
  failure types can look identical to the detectors we've built.
- **Tested on one dataset, one target feature, and one (simple) prediction
  model.** We did test a second, different dataset (see the separate
  Taiwan Credit Default package) to check generalization — results mostly
  held up, but one detector's precision noticeably weakened on the new
  dataset, which is itself an honest, useful finding rather than a clean
  success story.
- **The simulated "real outcomes" arrive instantly, which isn't realistic.**
  In a real deployment, you'd typically wait weeks before learning whether
  a loan actually defaulted. Our performance monitor currently assumes
  that information arrives in the very same batch as the prediction,
  which is a simplification.
- **A small amount of cross-talk between detectors remains.** Detector #5
  occasionally shows a little bit of sensitivity to covariate shift too
  (which isn't its job). Our combined-diagnosis logic fully compensates
  for this in practice — we measured zero resulting misdiagnoses — but the
  underlying cross-talk itself hasn't been eliminated at the source.

## 11. What We Deliberately Did Not Do, and Why

Given a real deadline, we made a conscious choice not to attempt a fix for
the "instant outcomes" simplification (limitation above) inside this
final version. Making that fix realistic would mean changing how the
performance monitor handles data across multiple batches at once — a
meaningfully bigger, riskier code change — and we judged that the risk of
introducing a new bug shortly before submission outweighed the benefit of
attempting it under time pressure. We're noting this decision explicitly,
rather than silently leaving it out, because we think knowing when *not*
to make a risky change is itself part of good engineering judgment.

## 12. Summary for a Reader in a Hurry

We built a system to catch three ways a deployed ML model can silently
become wrong. The originally-planned three-part design only ever caught
one of the three failure types when we tested it honestly — so we built
two more detectors specifically to close that gap, verified all five work
together on 1,800 separate test runs with 0% false alarms, benchmarked
that the system is fast enough to actually be useful, generalized it to a
second real-world dataset, and documented — rather than hid — every bug
and every remaining limitation we found along the way.
