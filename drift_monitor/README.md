# drift_monitor

**A System That Notices** - a reusable pipeline that injects and detects distribution shift
(covariate shift, label shift, concept drift) in a batched data stream, evaluates the detectors
against injected ground truth, and shows the result on a dashboard.
Built for DS635 (Machine Learning System Engineering). Nothing in this package is dataset-specific;
the three datasets are configured in `presets.py`.

This file documents the **API**. Every result number lives in [`../REPORT.md`](../REPORT.md) and in each
dataset's `RESULTS.md`; both are generated from the CSV/JSON files, so this file quotes none.

## Install and run

```bash
pip install -r requirements.txt          # from the repo root: pip install -r drift_monitor/requirements.txt
python run_all.py                        # regenerate everything (all 3 datasets + docs)
python run_all.py --quick                # 3-rep smoke test into a temp folder
streamlit run drift_monitor/dashboard_app.py
python -m pytest tests/ -q
```

Docker (the pinned environment; the CSVs are inside the image):

```bash
docker build -t drift-monitor .
docker run -p 8501:8501 drift-monitor                                  # dashboard
docker run --rm -v "$(pwd)":/app drift-monitor python run_all.py       # regenerate everything
docker run --rm -v "$(pwd)":/app drift-monitor \
       python datasets/give_me_some_credit/generate_report_assets.py   # one dataset (path is from /app)
```

## The pipeline in four calls

```python
from drift_monitor.presets import get_preset
from drift_monitor import load_dataset
from drift_monitor.injection import make_scenario
from drift_monitor.evaluate import calibrate_thresholds
from drift_monitor.monitor import run_monitor

config = get_preset("taiwan")                       # or DriftMonitorConfig(csv_path=..., features=[...], ...)
df = load_dataset(config)
thresholds, noise = calibrate_thresholds(df, config)              # 1. per-dataset thresholds from CLEAN runs
reference_df, batches = make_scenario("label", "severe", "abrupt", 42, df, config)   # 2. disjoint reference + shifted stream
results = run_monitor(batches, reference_df, config, thresholds)  # 3. fit on reference, score, alert
# 4. python -m drift_monitor.report taiwan   -> the full grid, curves, baseline, benchmark, provenance
```

## Design rules

| Rule | Where | Why |
|---|---|---|
| The reference window and the stream are **disjoint** rows, and every seed draws a **fresh** split | `injection.make_reference_and_stream` | Prevents reference rows leaking into the stream and a single reference serving all repetitions |
| Thresholds are **calibrated per dataset** from clean runs (never below the default floor) | `evaluate.calibrate_thresholds` | Noise depends on batch size; a threshold that suits 3,000-row batches can sit below the noise on 750-row batches |
| Concept drift is **pure**: P(X) and P(Y) unchanged | `injection.inject_concept_drift` | A band flip would also move the label rate |
| Label shift **never touches pre-onset batches** | `injection.inject_label_shift` | Resampling clean batches would add noise before any shift exists |
| Raw scores are saved; thresholds are applied afterwards | `monitor.score_batches` / `apply_alerts` | Lets the same runs trace sensitivity-vs-false-alarm curves (`tradeoff.py`) |
| A **named baseline** is run on every run | `baselines.KSBaseline` | Per-feature KS test, Bonferroni-corrected, fixed alpha, untuned |
| A **hand-labelled** check independent of the injectors | `../validation/` | See that file's docstring for what it does and does not show |

## PSI versus KL

For identical bins, `PSI = KL(cur||ref) + KL(ref||cur)` - the symmetrised KL (Jeffreys) divergence. Both use a
log-ratio; PSI is **not** "linear where KL is logarithmic". `metrics.kl_divergence` is only the forward term
`KL(cur||ref)`, so for the same data it is **never larger than PSI**. With the same threshold (0.10 / 0.25), KL
therefore cannot fire when PSI does not, and its scores are roughly half of PSI's. `tests/test_design.py` checks the identity
and a hand-computed 2-bin example.

## Package layout

| File | Purpose |
|---|---|
| `config.py` | `DriftMonitorConfig`: every knob. Never edit its defaults; pass overrides |
| `presets.py` | The three dataset configurations; refuses to fall back silently to synthetic data |
| `data.py` | CSV loader (synthetic fallback only when no `csv_path` is set) |
| `injection.py` | Disjoint split, onset schedules, the three injectors, `make_scenario` |
| `metrics.py` | PSI, KL, cached `PSIKLMonitor`, PCA embedding drift |
| `reference_model.py` | Fixed reference classifier + `pred_prop_psi` detector |
| `performance_monitor.py` | `concept_signal`: Brier degradation on a simulated labelled trickle |
| `baselines.py` | The named KS + Bonferroni baseline |
| `monitor.py` | `fit_detectors` -> `score_batches` -> `apply_alerts` (2-consecutive-batch rule) |
| `diagnosis.py` | Five alerts -> one verdict (covariate / label / concept_or_mild_label / label_or_subtle_covariate / none) |
| `evaluate.py` | Calibration, the shift x severity x onset x rep grid, Wilson CIs, baseline comparison |
| `tradeoff.py` | Sensitivity-vs-false-alarm curves, score AUC, sensitivity at fixed false-alarm budgets |
| `report.py` | Regenerates every file in a dataset's `report_assets*/` |
| `benchmark.py` | Latency / throughput / memory, and the environment fingerprint |
| `viz.py`, `dashboard_app.py` | Charts and the Streamlit dashboard |

## Adding a fourth dataset

1. Add an entry to `presets.PRESETS` and a branch in `get_preset` (CSV path, label, numeric features, target feature).
2. **Check the target feature's IQR first.** Injection is scaled by `IQR / 1.349`; a zero IQR (e.g. Adult's `capital-gain`) injects nothing at any severity.
3. Check every feature for extreme outliers before trusting any statistic that uses `std()` or an unscaled model (the reference model winsorises to the 1st/99th reference percentile for this reason).
4. `python -m drift_monitor.report <key>`, then `python tools/build_docs.py`.

Severity levels are per-config dicts (`severity_covariate`, `severity_label`, `severity_concept`); concept severity is the
fraction of the positives in the region `x >= median` that are swapped with negatives in `x < median`.

## Known limitations

- Labels are assumed to arrive in the same batch (a 20 % random trickle); real feedback is delayed.
- Binary labels and numeric features only.
- `concept_signal` is sensitive but not selective (it also responds to label shift); `diagnosis.py` cannot separate mild label shift from concept drift.
- `zlib.crc32` (not `hash()`) seeds the grid: `hash()` is randomised per process and silently breaks reproducibility.
