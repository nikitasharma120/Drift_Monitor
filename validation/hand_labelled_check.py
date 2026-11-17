"""
validation/hand_labelled_check.py - the HAND-LABELLED check.

The evaluation grid scores detectors against ground truth produced by our own
injectors, so an error shared by an injector and a detector could hide. This
check removes the injectors from the loop: four small streams are built BY
HAND from an explicit generative recipe written out below, and every batch
carries a hand-written truth label (the lists below). Nothing in
drift_monitor/injection.py is used to make them.

  stream      batches 0-9        batches 10-19                         hand label for 10-19   verdict we accept
  ---------   ----------------   ----------------------------------    --------------------   ------------------------
  clean       recipe             recipe (unchanged)                    none                   none
  covariate   recipe             f1 mean moved +2.0, rule unchanged    covariate              covariate
  label       recipe             resampled to 45% positive             label                  label
  label_mild  recipe             resampled to 35% positive             label                  label or concept_or_mild_label
  concept     recipe             sign of f1 in the rule flipped, the   concept                concept_or_mild_label
                                 intercept re-solved so P(Y) is kept
                                 (P(X) identical)

The accepted verdicts were written down BEFORE running the check, from
diagnosis.py's documented behaviour (concept drift alone can only reach
"concept_or_mild_label"). Batches 10 and 11 are excluded from scoring: the
2-consecutive-batch confirmation rule cannot fire earlier, so they are a
detection-lag allowance, not a miss.

Two label streams are included: `label_mild` (35% positive, only +8 points
over the recipe's ~27% base rate, so it is genuinely hard) and `label` (45%,
+18 points, comparable to the grid's "severe" +20). Accepted verdicts for
`label_mild` include concept_or_mild_label, which diagnosis.py documents as
the mild-label verdict. Both streams are reported, including where the suite
misses.

Limits, stated plainly: these labels are hand-written from how each stream was
constructed, not annotations of real production incidents -- none of the
three public datasets has a timestamped, human-confirmed drift event. This
makes the check independent of the injectors, not independent of us.

Run:  python -m validation.hand_labelled_check     (writes validation/hand_labelled_check.csv)
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drift_monitor.config import DriftMonitorConfig
from drift_monitor.diagnosis import classify_shift
from drift_monitor.evaluate import calibrate_thresholds
from drift_monitor.monitor import run_monitor

FEATURES = ["f1", "f2"]
B = 600                       # rows per batch
N_BATCH, ONSET = 20, 10
REF_ROWS = 10 * B

HAND_TRUTH = {                # hand-written, per batch index 0..19
    "clean":     ["none"] * 20,
    "covariate": ["none"] * 10 + ["covariate"] * 10,
    "label":     ["none"] * 10 + ["label"] * 10,
    "label_mild": ["none"] * 10 + ["label"] * 10,
    "concept":   ["none"] * 10 + ["concept"] * 10,
}
ACCEPTED = {  # per STREAM (not per truth label), written down before running
    "clean": {"none"}, "covariate": {"covariate"}, "label": {"label"},
    "label_mild": {"label", "concept_or_mild_label"}, "concept": {"concept_or_mild_label"}}
LABEL_RATE = {"label": 0.45, "label_mild": 0.35}
SETTLE = 2                    # confirmation-rule allowance (batches after onset not scored)


def _recipe(n, rng, f1_shift=0.0, coef=1.5, intercept=-1.5):
    f1 = rng.normal(f1_shift, 1.0, n)
    f2 = rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(coef * f1 - 1.0 * f2 + intercept)))
    return pd.DataFrame({"f1": f1, "f2": f2, "label": (rng.random(n) < p).astype(int)})


def _solve_intercept(rng, coef, target_rate, n=400_000):
    """bisection: intercept so the rule with this coef gives P(Y)=target_rate when f1 ~ N(0,1)."""
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(coef * f1 - f2 + mid)))).mean()
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    return (lo + hi) / 2


def build_streams(seed=7):
    rng = np.random.default_rng(seed)
    reference = _recipe(REF_ROWS, rng)
    base_rate = reference["label"].mean()
    flipped_intercept = _solve_intercept(rng, -1.5, base_rate)
    streams = {}
    for name in HAND_TRUTH:
        batches = []
        for i in range(N_BATCH):
            post = i >= ONSET
            if name == "covariate" and post:
                b = _recipe(B, rng, f1_shift=2.0)
            elif name == "concept" and post:
                b = _recipe(B, rng, coef=-1.5, intercept=flipped_intercept)
            elif name in LABEL_RATE and post:
                b = _recipe(B * 3, rng)
                pos, neg = b[b.label == 1], b[b.label == 0]
                n_pos = int(round(LABEL_RATE[name] * B))
                b = pd.concat([pos.sample(n_pos, replace=n_pos > len(pos), random_state=int(rng.integers(1e6))),
                               neg.sample(B - n_pos, replace=False, random_state=int(rng.integers(1e6)))])
                b = b.sample(frac=1.0, random_state=int(rng.integers(1e6))).reset_index(drop=True)
            else:
                b = _recipe(B, rng)
            batches.append(b)
        streams[name] = batches
    return reference, streams


def run_hand_labelled_check(seed=7) -> pd.DataFrame:
    config = DriftMonitorConfig(features=FEATURES, label_col="label", target_feature="f1",
                                n_batches=20, n_reference_batches=10, onset_batch=ONSET,
                                calibration_reps=30, embedding_components=2)
    clean_df = _recipe(30 * B, np.random.default_rng(seed + 1))     # B = 18000 // 30 = 600: same batch size
    thresholds, _ = calibrate_thresholds(clean_df, config)
    reference, streams = build_streams(seed)
    rows = []
    for name, batches in streams.items():
        res = run_monitor(batches, reference, config, thresholds=thresholds, seed=seed)
        for i, r in res.iterrows():
            truth = HAND_TRUTH[name][i]
            diag = classify_shift(r)
            scored = not (ONSET <= i < ONSET + SETTLE)
            any_alert = bool(r["any_alert"])
            rows.append({"stream": name, "batch": i, "hand_label": truth, "diagnosis": diag,
                         "scored": scored, "agree": (diag in ({"none"} if truth == "none" else ACCEPTED[name])) if scored else None,
                         "any_alert": any_alert,
                         # detection-only: did the suite alert exactly when the hand label says something is wrong?
                         "alert_ok": (any_alert == (truth != "none")) if scored else None,
                         "psi_alert": r["psi_agg_alert"], "pred_prop_psi_alert": r["pred_prop_psi_alert"],
                         "concept_signal_alert": r["concept_signal_alert"], "ks_baseline_alert": r["ks_baseline_alert"]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per stream: how often the suite alerted exactly when the hand label says something is
    wrong (alert_ok), and how often diagnosis.py's verdict was one of the pre-declared accepted
    verdicts (agree). Both over the scored batches (pre-onset + post-onset after the settle window)."""
    s = df[df["scored"]].copy()
    s["agree"] = s["agree"].astype(bool); s["alert_ok"] = s["alert_ok"].astype(bool)
    return s.groupby("stream").agg(batches_scored=("agree", "size"), alert_ok=("alert_ok", "mean"),
                                    diagnosis_agrees=("agree", "mean")).round(3).reset_index()


if __name__ == "__main__":
    df = run_hand_labelled_check()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_labelled_check.csv")
    df.to_csv(out, index=False)
    print(summarize(df).to_string(index=False))
    print("wrote", out)
