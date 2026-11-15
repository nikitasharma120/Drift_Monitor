# Pins the exact environment this project's results were verified against.
# Built to resolve, permanently, the reproducibility saga documented in the
# Phase 5 report's Appendix A: four separate times, differences in the host
# machine's Python/library setup produced different results from identical
# code and data. A container removes that variable entirely -- "works on my
# machine" becomes "works in this exact machine, always."

FROM python:3.12-slim

WORKDIR /app

# System build tools needed by scipy/scikit-learn/matplotlib wheels on some
# platforms; kept minimal and removed from the final layer where possible.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY drift_monitor/requirements.txt ./drift_monitor/requirements.txt
RUN pip install --no-cache-dir -r drift_monitor/requirements.txt

COPY . .

# Verify the pinned versions actually installed correctly at build time,
# not just that pip exited 0 -- catches silent resolver substitutions.
RUN python -c "\
import matplotlib, sklearn, numpy, pandas, scipy; \
assert sklearn.__version__ == '1.8.0', sklearn.__version__; \
assert numpy.__version__ == '2.4.4', numpy.__version__; \
assert pandas.__version__ == '3.0.2', pandas.__version__; \
assert scipy.__version__ == '1.17.1', scipy.__version__; \
assert matplotlib.__version__ == '3.10.0', matplotlib.__version__; \
print('All pinned versions verified.')"

EXPOSE 8501

# Default: launch the dashboard. Override with `docker run drift-monitor python reproduce_all.py`
# to regenerate all combined report assets and trade-off curves.
CMD ["streamlit", "run", "drift_monitor/dashboard_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
