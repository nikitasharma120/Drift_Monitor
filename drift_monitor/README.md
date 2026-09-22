# drift_monitor

**A System That Notices** — a reusable pipeline for injecting and detecting
distribution shift (covariate shift, label shift, concept drift) in a batched
data stream, evaluating detector performance, and viewing results on a
lightweight dashboard.

Built for DS635 (Machine Learning System Engineering). Everything in this
package is dataset-agnostic — the "Give Me Some Credit" defaults are just
the demo configuration.

## Install

```bash
pip install -r requirements.txt
```

## Recommended: run via Docker instead

This project hit the same reproducibility bug four separate times across
different machines (see Phase 5 report Appendix A) — different installed
library versions producing different results from identical code and data.
A container removes that variable permanently. From the top-level folder
(the one with `Dockerfile` and `drift_monitor/` both visible):

```bash
docker build -t drift-monitor .

# Dashboard (dataset optional — falls back to synthetic data if not mounted):
docker run -p 8501:8501 -v /path/to/cs-training.csv:/app/cs-training.csv drift-monitor

# Regenerate report_assets/ instead of the dashboard:
docker run -v /path/to/cs-training.csv:/app/cs-training.csv \
           -v $(pwd)/report_assets:/app/report_assets \
           drift-monitor python generate_report_assets.py
```

The dataset is mounted at runtime, not baked into the image (keeps the
image small and lets you swap datasets without rebuilding) — set
`csv_path` in `DriftMonitorConfig` to `/app/cs-training.csv` to match the
mount point above. The build step verifies every pinned library version
installed exactly as expected and fails loudly if not, rather than
silently continuing with a substituted version.

## Quick start (synthetic demo data)

```python
from drift_monitor import DriftMonitorConfig, load_dataset
from drift_monitor.evaluate import run_full_grid, summarize_grid, best_detector_per_shift_type

config = DriftMonitorConfig()          # defaults to synthetic demo data
df = load_dataset(config)
records = run_full_grid(df, config)
summary = summarize_grid(records)
print(best_detector_per_shift_type(summary))
```

## Plugging in your own dataset/model

Everything lives in one place: `DriftMonitorConfig`. You do **not** need to
edit any other file in this package to point it at a new dataset.

```python
from drift_monitor import DriftMonitorConfig

config = DriftMonitorConfig(
    csv_path="my_data.csv",
    label_col="target",                 # your binary (0/1) label column
    features=["f1", "f2", "f3", "f4"],  # your numeric feature columns
    target_feature="f1",                # feature used for covariate shift + concept drift
    column_rename={},                   # map real CSV column names -> the names above, if they differ
)
```

**Pass `csv_path` when you construct `DriftMonitorConfig`, as above — do
not edit `config.py`'s default `csv_path` field.** That default is shared
by anyone importing this package (including `generate_report_assets.py`
if you don't override it there); hardcoding your own machine's path into
it means the package silently falls back to synthetic data for everyone
else, with no error. `generate_report_assets.py` has its own script-local
`CSV_PATH` variable at the top specifically to keep this override separate
from the shared library default — edit that one, not `config.py`.

If your label isn't binary 0/1, binarize it before passing the dataframe in,
or extend `injection.inject_label_shift` — it currently assumes two classes.

### Recalibrating severities

The default `severity_covariate` / `severity_label` / `severity_concept`
dicts in `config.py` were tuned against the demo dataset's quantiles. **They
will very likely need re-tuning against your real data** — the concept-drift
fractions especially, since they're expressed relative to your
`target_feature`'s IQR. If mild/moderate/severe don't separate cleanly (e.g.
moderate and severe both saturate to the same detection rate), narrow the
fractions; if none of them register at all, widen them. Re-run
`run_full_grid` after any change — it's fast (a minute or two for the default
grid on a laptop).

## Package layout

| File | Purpose |
|---|---|
| `config.py` | `DriftMonitorConfig` — every knob, in one dataclass |
| `data.py` | Load a real CSV (via config) or fall back to synthetic demo data |
| `injection.py` | Batching, timing control (abrupt/gradual onset), the three shift injectors |
| `metrics.py` | PSI, KL divergence (cached — see below), PCA-based embedding drift |
| `reference_model.py` | A fixed reference classifier + predicted-class-proportion detector (targets label shift specifically) |
| `performance_monitor.py` | Brier-score-based performance monitor with a simulated labeled trickle (targets concept drift specifically) |
| `diagnosis.py` | Combines all 5 alerts into one shift-type diagnosis, resolving `concept_signal`'s cross-detector ambiguity |
| `benchmark.py` | Per-detector latency/throughput/memory cost — see "System cost" below |
| `monitor.py` | Runs metrics over a batch stream, applies alert thresholds + consecutive-batch confirmation |
| `evaluate.py` | Full evaluation grid (shift type × severity × onset × reps), summary stats with Wilson-score confidence intervals, failure-mode detection, best-detector recommendation |
| `viz.py` | Dashboard panel + evaluation-grid heatmaps (shared by the report generator and the Streamlit app) |
| `dashboard_app.py` | Polished interactive Streamlit dashboard |

## Performance: PSI/KL are cached, not recomputed per batch

`PSIKLMonitor` (in `metrics.py`) precomputes bin edges and the reference
window's own histogram once at `.fit()` time, and reuses them for every
future batch. The plain `multi_feature_psi()`/`multi_feature_kl()`
functions still exist for one-off comparisons but recompute from scratch
every call — `monitor.py` and `evaluate.py` both use the cached version.
Benchmarked: this cut PSI+KL from ~23ms to ~0.6ms per batch (a ~36x
speedup on the combined cost), and the full 5-detector pipeline from
~44ms to ~22ms per batch — verified numerically identical to the
uncached version to 1e-12 before switching over.

## System cost (not just detection quality)

```python
from drift_monitor.benchmark import run_benchmark
print(run_benchmark(df, config))
```

Reports per-detector latency (ms/batch), implied throughput
(batches/sec), one-time fit cost, and an approximate memory footprint.
Also runs automatically as part of `generate_report_assets.py`, writing
`report_assets/benchmark.csv`. On the Give Me Some Credit setup (10
features, ~4000-row batches): the full 5-detector pipeline processes a
batch in ~22ms (~45 batches/sec) — `concept_signal` (the performance
monitor) is now the single most expensive detector per batch, not PSI/KL.

## Statistical confidence: sensitivity estimates include Wilson-score CIs

Every sensitivity number in `evaluate.py`'s output (`summarize_grid()`,
`best_detector_per_shift_type()`) now comes with a 95% Wilson-score
confidence interval (`sensitivity_ci_low`/`sensitivity_ci_high`) — a
proper interval for a binomial proportion, not a normal approximation
that breaks down at n=5 reps and near 0/1 proportions (which is most of
our cells). A reported "1.0" with n=5 has a CI of roughly [0.57, 1.0] —
report the interval, not just the point estimate, when the exact number
matters.

## Outlier-robustness checklist (read before adding a new statistic)

This dataset's `DebtRatio`/`RevolvingUtilizationOfUnsecuredLines` have extreme
outliers (values in the tens of thousands against medians near 0.3). We hit
the same bug twice from this:

1. **Covariate-shift injection** originally scaled by raw `std()`, which the
   outliers inflated 100-1000x, making "mild" shift unrealistically huge.
   Fixed by switching to IQR / 1.349 (see `injection.py`).
2. **The reference model** (`reference_model.py`) originally fit an unscaled
   `LogisticRegression` directly on raw features, which failed to converge
   for the same reason and produced a silently miscalibrated model. Fixed
   by standardizing features first.

**Before adding any new statistic to this pipeline, check whether it uses
`std()`, an unscaled model, or any other non-robust measure of scale on a
feature that might contain outliers** — this class of bug will not throw an
error, it will just silently produce wrong numbers.

## Detectors and what they actually catch

Evaluated on the real Give Me Some Credit dataset (see the Phase 5 report
for full numbers):

| Shift Type | Best Detector | Mean Sensitivity |
|---|---|---|
| Covariate shift | PSI or KL | 1.00 |
| Label shift | concept_signal + pred_prop_psi | 1.00 (concept_signal alone) |
| Concept drift | concept_signal (performance monitor) | 0.70 – 1.00 |

PSI/KL/embedding drift all operate on **X** and are blind to label shift
and concept drift by construction. Two additional detectors close most of
that gap:

- **`reference_model.py`** (predicted-proportion detector) watches a fixed
  model's predictions instead of raw features — catches label shift at
  moderate/severe severity.
- **`performance_monitor.py`** (Brier-score performance monitor) simulates
  a delayed, partial labeled trickle and compares real outcomes against
  predictions — the only detector with real sensitivity to concept drift,
  and it also independently catches label shift at every severity.

**Important caveat:** `concept_signal` is sensitive but not *selective* —
it fires on both label shift and concept drift, since both genuinely
degrade real-world accuracy. It also has a small amount of genuine leakage
onto covariate shift at moderate/severe severity (a fixed model's
calibration can degrade under covariate shift too — that's real model
behavior, not a bug). Don't read `concept_signal` in isolation — use
`drift_monitor/diagnosis.py`, which combines all five alerts into a single
diagnosis with PSI/KL given precedence:

```python
from drift_monitor.diagnosis import add_diagnosis
results = add_diagnosis(run_monitor(batches, reference_df, emb_detector, config,
                                     pred_detector=pred_detector, perf_detector=perf_detector))
print(results[["batch", "diagnosis"]])
```

Validated across the full evaluation grid: this precedence rule resolves
100% of the covariate-shift leakage (0% of covariate batches are
misdiagnosed as concept drift, vs. 20-40% if you read `concept_signal`
alone). The one ambiguity it does NOT resolve — mild label shift looking
identical to concept drift — is a real, currently-unsolved limit of the
detector suite, not a bug in the diagnosis logic; see the Phase 5 report,
Section 3.4, for why.

## Running the dashboard

```bash
streamlit run drift_monitor/dashboard_app.py
```

Lets you pick a shift type / severity / onset from the sidebar and watch the
monitor's response live, plus an evaluation-grid explorer tab.

## Known limitations (see the Phase 5 report for the full write-up)

- All three detectors here operate on **X** (raw features or their PCA
  embedding). They're strong on covariate shift, weak-to-blind on label
  shift and concept drift by construction — that's not a threshold-tuning
  problem, see the report's Findings section.
- The embedding-drift detector underperforms plain PSI/KL in this
  evaluation; treat it as a secondary signal, not the concept-drift catcher
  it was originally intended to be.
- `zlib.crc32` is used for grid-seed generation instead of Python's built-in
  `hash()`, which is randomized per-process by default and silently breaks
  reproducibility across runs — worth remembering if you extend the grid.
