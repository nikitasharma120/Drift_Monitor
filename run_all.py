"""run_all.py - ONE command regenerates every result and every number in the docs.

    python run_all.py              # full run: 3 datasets x 360 runs, then REPORT.md / RESULTS.md / README tables
    python run_all.py --quick      # 3-rep smoke test into a temp folder (does not touch the committed assets)
    python run_all.py --datasets taiwan adult
    python run_all.py --docs-only  # rebuild the markdown from the existing CSV/JSON files
    python run_all.py --demo       # viva demo: regenerate the main (Taiwan) result live in 1-2 minutes

Order: for each dataset -> drift_monitor.report (calibrate, grid, curves, benchmark, provenance);
then tools/build_docs.py reads ONLY those CSV/JSON files to write the numbers into the docs.
"""

import argparse
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from drift_monitor.presets import PRESETS
from drift_monitor.report import generate_report_assets


def demo():
    """One command that regenerates the main result live: Taiwan (the dataset where calibration matters most),
    5 reps instead of 20, written to a temp folder so the committed 20-rep results are untouched."""
    import pandas as pd
    out = os.path.join(tempfile.mkdtemp(prefix="drift_demo_"), "taiwan")
    generate_report_assets("taiwan", n_reps=5, out_dir=out, verbose=False)
    s = pd.read_csv(os.path.join(out, "evaluation_summary.csv"))
    t = s.groupby(["shift_type", "metric"]).agg(sensitivity=("sensitivity", "mean"),
                                                false_alarm=("mean_false_alarm_rate", "mean")).round(3).reset_index()
    sens = t.pivot(index="shift_type", columns="metric", values="sensitivity")
    far = t.pivot(index="shift_type", columns="metric", values="false_alarm")
    print("\n=== MAIN RESULT (Taiwan, 5 reps): sensitivity ===\n" + sens.to_string())
    print("\n=== false-alarm rate (clean pre-onset batches) ===\n" + far.to_string())
    print("\ncharts + curves written to", out, "(tradeoff_curves.png, severity_curves.png)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=list(PRESETS), default=list(PRESETS))
    ap.add_argument("--quick", action="store_true", help="3 reps, temp output, docs not rebuilt")
    ap.add_argument("--docs-only", action="store_true")
    ap.add_argument("--n-reps", type=int, default=None)
    ap.add_argument("--demo", action="store_true",
                    help="viva demo: Taiwan, 5 reps (~1-2 min), temp folder; prints the main result table")
    a = ap.parse_args()
    t0 = time.time()
    if a.demo:
        demo(); return
    if not a.docs_only:
        for key in a.datasets:
            out = os.path.join(tempfile.mkdtemp(prefix="drift_quick_"), key) if a.quick else None
            generate_report_assets(key, n_reps=a.n_reps, quick=a.quick, out_dir=out)
    if not a.quick:
        from tools.build_docs import build_all
        build_all()
    print(f"run_all finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
