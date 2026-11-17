# Viva plan: 15 min presentation + 5 min Q&A

Rubric: experimental rigour 35 %, conceptual understanding 30 % (individual), results and insight 20 %, reproducibility and report 15 %.
Be crisp, breadth first, let the examiner interrupt. Have the repo and `python run_all.py --demo` ready in a terminal BEFORE you start.

| Min | Slide | What to say | Where in the repo |
|---|---|---|---|
| 0-2 | Context | Models drift silently. Three mechanisms: P(X), P(Y), P(Y\|X). X-only detectors cannot see concept drift. | README "three failure modes" |
| 2-5 | Experimental setup | Baseline = KS + Bonferroni. Varied: shift x severity x onset x detector x dataset. Fixed: 2-batch rule, disjoint reference, batch = rows//40, model, seeds. Measured: sensitivity + Wilson CI (20 reps), lag, false alarms, AUC. 1,080 runs. Hand-labelled check. | REPORT.md section 2 |
| 5-10 | Key findings | Show `tradeoff_curves.png` (Taiwan) and `severity_curves.png`. Covariate: PSI = KS. Label: model-based beat KS. Concept: only `concept_signal`, weak when positives are rare. Calibration took Taiwan false alarms from 25-37 % to <= 0.1 %. | `datasets/*/report_assets*/`, REPORT.md |
| 10-12 | Where it breaks and why | Rare positives, small batches, fixed PSI/KL threshold, mild label vs concept ambiguity, conservative operating points. | REPORT.md section 3 |
| 12-14 | Demo | `python run_all.py --demo` (about 90 s). Start it at minute 10 and talk while it runs; show the table it prints. | run_all.py |
| 14-15 | Recommendations + limitations | Six recommendations; 3 datasets, binary labels, same-batch labels, we wrote the hand labels. | REPORT.md sections 4, 8 |

## Individual viva: concepts you must be able to explain
- Why X-only detectors cannot see concept drift; why they sometimes catch label shift (P(X) = sum over y of P(X|y)P(y)).
- PSI = KL(cur||ref) + KL(ref||cur); our KL is one direction so it never exceeds PSI.
- Why batch size sets the noise floor and therefore the threshold; how calibration works and why seeds are disjoint.
- Why the concept injector is pure (swap equal numbers, P(X) and P(Y) unchanged) and why that made detection harder.
- Why reference and stream must be disjoint, and why a fresh split per run.
- Wilson CI vs normal approximation; what 20 reps can and cannot tell you.
- What each metric means: sensitivity, lag, batch-level false alarm vs runs-with-a-false-alarm, score AUC, sensitivity @ FAR.
- What would change the result: more labels, bigger batches, a lower PSI threshold, a different severity scale.
- Be honest on limits: our own hand labels, same pool of rows for calibration and evaluation, 3 datasets.

## Checklist before the slot
- [ ] Code pushed to GitHub and the link submitted in the Google Form (steps at the end of this file).
- [ ] `pip install -r drift_monitor/requirements.txt` done; `python run_all.py --demo` tested on the machine you will present from.
- [ ] `datasets/taiwan_credit_default/report_assets_taiwan/tradeoff_curves.png` and `severity_curves.png` open in tabs.
- [ ] Offline slot (FB-3, Room 3209)? Edit the calendar request description to say "offline"; you still join the Gmeet and screenshare.
- [ ] Questions about the process go to the CRs.


---

# Pushing to GitHub

From the repo root (the folder containing `run_all.py`):

```bash
git init
git add .
git commit -m "Drift monitor: calibrated thresholds, pure concept drift, trade-off curves, baseline, hand-labelled check"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Create the empty repository on GitHub first (no README, no .gitignore). Then paste the repo URL in the Google Form.
Largest committed files are the dataset CSVs (well under GitHub's 100 MB limit) and `raw_scores.csv.gz` (a few MB each).
After cloning on another machine: `pip install -r drift_monitor/requirements.txt && python run_all.py --demo`.


Note: the deck (Drift_Monitor_Presentation.pptx) uses the final numbers. Key extra point to make: with calibrated thresholds, k = 1 (instead of 2 consecutive batches) raises sensitivity a lot at under 0.5% false alarms (Taiwan label 0.56 to 0.91; GMSC concept 0.12 to 0.49).
