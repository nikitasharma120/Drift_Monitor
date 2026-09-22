# A System That Notices — Detecting Silent Distribution Shift in Deployed ML Models

**DS635 — Machine Learning System Engineering**

A monitoring system that detects three kinds of ML model degradation
(covariate shift, label shift, concept drift) in streaming data, tested
against two independent, real-world datasets to check that its design
choices generalize rather than being tuned to fit just one.

## Repo Structure

```
.
├── drift_monitor/                       <- the ONE shared detection package.
│                                            Identical code powers both
│                                            datasets below -- nothing here
│                                            is dataset-specific.
├── tests/                                <- automated tests for drift_monitor/
├── Dockerfile / .dockerignore            <- reproducible environment (see
│                                            datasets/give_me_some_credit/README.md
│                                            §9, bug #4, for why this exists)
└── datasets/
    ├── give_me_some_credit/              <- PRIMARY dataset & full project writeup
    │   ├── README.md                        <- start here — explains the whole
    │   │                                        project from scratch
    │   ├── generate_report_assets.py        <- run this to reproduce everything
    │   ├── cs-training.csv                  <- the real dataset
    │   └── report_assets/                   <- every chart/table already generated
    │
    ├── taiwan_credit_default/            <- SECOND dataset — same-domain generalization
    │   ├── README.md                        <- read after the one above — assumes
    │   │                                        you already know the main project
    │   ├── generate_report_assets_taiwan.py
    │   ├── UCI_Credit_Card.csv
    │   └── report_assets_taiwan/
    │
    └── adult_income/                     <- THIRD dataset — cross-domain generalization
        ├── README.md                        <- a genuinely different domain (census
        │                                        income, not credit risk) — reveals a
        │                                        finding that revises one of the main
        │                                        project's confident claims
        ├── generate_report_assets_adult.py
        ├── adult_income.csv
        └── report_assets_adult/
```

## Why Multiple Datasets Share One Package

Anyone can tune a detection system until it works perfectly on the one
dataset in front of them — that doesn't prove the system is any good. The
only real test is pointing the *exact same, unmodified* detection code at
a dataset it has never seen and checking whether the same design choices
(severity levels, alert thresholds, which detector to trust for which
failure type) still hold up.

**The rule enforced by this repo's structure:** `drift_monitor/` is not
duplicated per dataset. Only each dataset folder's own script — which
just tells the shared package which CSV, which columns, and which label
to use — differs. If the detection code itself had to change to make a
second or third dataset work, that would mean the original design didn't
really generalize; it would just mean it got tuned again. It didn't need
to.

**Three datasets, two domains, one surprising result:** Give Me Some
Credit and Taiwan Credit Default are both credit-risk problems; Adult
Income is a genuinely different domain (census/demographic data). The
first two datasets agreed that PSI and KL divergence are the reliable
covariate-shift detectors. The third dataset broke that: PSI/KL turned
out to be the *weakest* covariate-shift detectors there, and a detector
built for an entirely different purpose (label shift) took over as the
best one. See `datasets/adult_income/README.md` §4-5 for the full,
honest account — including our best (not fully certain) explanation for
why, and what it means for the project's overall claims.

## Quick Start

```bash
pip install -r drift_monitor/requirements.txt

cd datasets/give_me_some_credit
python generate_report_assets.py       # ~10-15 min at 20 reps/scenario

cd ../taiwan_credit_default
python generate_report_assets_taiwan.py

cd ../adult_income
python generate_report_assets_adult.py # ~3-5 min — smallest of the three datasets
```

Or with Docker (see `datasets/give_me_some_credit/README.md` §7 for why
this is recommended):

```bash
docker build -t drift-monitor .
docker run -v $(pwd)/datasets/give_me_some_credit/cs-training.csv:/app/cs-training.csv \
           -p 8501:8501 drift-monitor
```

To run the shared test suite (from the repo root):

```bash
pytest tests/ -v
```

## Where to Read First

1. **`datasets/give_me_some_credit/README.md`** — the complete project
   writeup: the problem, every file explained from scratch, results,
   every bug found and fixed, and honest limitations. Read this first,
   even if you only care about the other two datasets.
2. **`datasets/taiwan_credit_default/README.md`** — the first
   generalization test (same domain, different dataset). Assumes you've
   read #1.
3. **`datasets/adult_income/README.md`** — the second generalization test
   (different domain entirely). Assumes you've read #1, and is more
   interesting if you've also read #2 — it's the one that overturns a
   claim the first two datasets agreed on.
