install:
	pip install --upgrade pip && pip install -r requirements/requirements.txt
format:
	black *.py
lint:
	pylint *.py
test:
	python -m pytest tests/test*.py
all: install lint test format
