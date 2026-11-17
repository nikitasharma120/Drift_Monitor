# Pins the exact library versions this project's results were generated with.
# Everything runs from /app (the repo root). The dataset CSVs are copied into the image so
# results can be regenerated inside the container.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY drift_monitor/requirements.txt ./drift_monitor/requirements.txt
RUN pip install --no-cache-dir -r drift_monitor/requirements.txt

COPY . .

# Fail the build if pip silently resolved different versions than the ones in requirements.txt.
RUN python -c "\
import matplotlib, sklearn, numpy, pandas, scipy; \
assert sklearn.__version__ == '1.8.0', sklearn.__version__; \
assert numpy.__version__ == '2.4.4', numpy.__version__; \
assert pandas.__version__ == '3.0.2', pandas.__version__; \
assert scipy.__version__ == '1.17.1', scipy.__version__; \
assert matplotlib.__version__ == '3.10.0', matplotlib.__version__; \
print('All pinned versions verified.')"

EXPOSE 8501

# Default: the dashboard (all three datasets are selectable in the sidebar).
#   docker build -t drift-monitor .
#   docker run -p 8501:8501 drift-monitor
#
# Regenerate EVERY result and doc number (outputs land in ./datasets/*/report_assets*, REPORT.md, README.md):
#   docker run --rm -v "$(pwd)":/app drift-monitor python run_all.py
# One dataset only (paths are relative to /app, not to the dataset folder):
#   docker run --rm -v "$(pwd)":/app drift-monitor python datasets/give_me_some_credit/generate_report_assets.py
# Tests:
#   docker run --rm drift-monitor python -m pytest tests/ -q
CMD ["streamlit", "run", "drift_monitor/dashboard_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
