"""
drift_monitor
=============
A System That Notices — a reusable distribution-shift monitoring
pipeline: batch a data stream, inject/detect covariate shift, label
shift, and concept drift, evaluate detector performance, and view
results on a lightweight dashboard.

Quick start
-----------
    from drift_monitor import DriftMonitorConfig, load_dataset
    from drift_monitor.evaluate import run_full_grid, summarize_grid

    config = DriftMonitorConfig(csv_path="my_data.csv", label_col="target",
                                 features=["f1", "f2", "f3"], target_feature="f1")
    df = load_dataset(config)
    records = run_full_grid(df, config)
    summary = summarize_grid(records)

See README.md for the full plug-in-a-new-dataset guide.
"""

from .config import DriftMonitorConfig
from .data import load_dataset

__all__ = ["DriftMonitorConfig", "load_dataset"]
__version__ = "1.0.0"
