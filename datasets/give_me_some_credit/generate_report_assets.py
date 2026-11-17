"""generate_report_assets.py - regenerate everything in report_assets/ for the Give Me Some Credit dataset.

    python datasets/give_me_some_credit/generate_report_assets.py      (from the repo root, or from this folder)

This is now a thin wrapper: the dataset's configuration lives in
drift_monitor/presets.py and all the work happens in drift_monitor/report.py,
so the three datasets are guaranteed to be processed by identical code.
Add --quick for a 3-repetition smoke test that writes to a temp folder.
For all three datasets plus the docs in one command: `python run_all.py`.
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from drift_monitor.report import generate_report_assets

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 reps, written to a temp folder")
    ap.add_argument("--n-reps", type=int, default=None)
    a = ap.parse_args()
    out = tempfile.mkdtemp(prefix="drift_quick_") if a.quick else None
    generate_report_assets("gmsc", n_reps=a.n_reps, quick=a.quick, out_dir=out)
