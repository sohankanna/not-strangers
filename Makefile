.PHONY: data results test

data:
	bash scripts/download_data.sh

results:
	python -m src.run_pipeline

test:
	pytest tests/
