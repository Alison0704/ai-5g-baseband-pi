PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup test dataset train regression benchmark dashboard clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -v

dataset:
	$(PYTHON) -m scripts.generate_dataset
	$(PYTHON) -m scripts.generate_fault_dataset

train:
	$(PYTHON) -m src.ml.train_link_adapter
	$(PYTHON) -m src.ml.failure_classifier

regression:
	$(PYTHON) -m scripts.run_regression

benchmark:
	$(PYTHON) -m scripts.benchmark_link
	$(PYTHON) -m scripts.benchmark_system

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py \
		--server.address 0.0.0.0 \
		--server.port 8501

clean:
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: demo

demo:
	$(PYTHON) -m scripts.demo
