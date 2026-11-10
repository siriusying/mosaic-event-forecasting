.PHONY: test verify forward rolling

test:
	python -m pytest -q tests

verify:
	python scripts/verify_manifest.py

forward:
	python scripts/run_forward.py --output runs/forward_sample

rolling:
	python scripts/run_rolling.py --output runs/rolling_sample
