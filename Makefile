.PHONY: all demo quick docs test dashboard
all:        ## regenerate every result + every number in the docs (about 15 minutes on one CPU)
	python run_all.py
demo:       ## viva demo: main result live, Taiwan, 5 reps
	python run_all.py --demo
quick:      ## 3-rep smoke test, writes to a temp folder
	python run_all.py --quick
docs:       ## rebuild markdown from the existing CSV/JSON files
	python run_all.py --docs-only
test:
	python -m pytest tests/ -q
dashboard:
	streamlit run drift_monitor/dashboard_app.py
