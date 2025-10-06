.PHONY: help setup data features train predict eval test dashboard clean lint format

help:
	@echo "Anomalous Market Behavior Recognition - Make Targets"
	@echo "======================================================"
	@echo "setup       - Install dependencies and setup environment"
	@echo "data        - Download and ingest data to SQL"
	@echo "features    - Engineer features from raw data"
	@echo "train       - Train LSTM model"
	@echo "predict     - Generate predictions using trained model"
	@echo "eval        - Evaluate model and compute metrics"
	@echo "dashboard   - Launch Streamlit dashboard"
	@echo "test        - Run all tests"
	@echo "lint        - Run linters (flake8, mypy)"
	@echo "format      - Format code (black, isort)"
	@echo "clean       - Remove generated files and artifacts"
	@echo "all         - Run full pipeline (data → features → train → predict → eval)"

setup:
	pip install -r requirements.txt
	pre-commit install

data:
	python -m src.data.download
	python -m src.data.ingest_sql

features:
	python -m src.data.features

train:
	python -m src.pipelines.train

predict:
	python -m src.pipelines.predict

eval:
	python -m src.pipelines.evaluate

dashboard:
	streamlit run src/viz/dashboard.py

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf models/ artifacts/ __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete

all: data features train predict eval
	@echo "✅ Full pipeline completed successfully!"

