.PHONY: clean build test


clean:  ## clean all build, python, and testing files
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .coverage
	rm -fr coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache

test: ## run pytest with coverage report
	pytest

lint: ## run a lint check / report
	flake8 src/allencell_ml_segmenter --count --verbose --show-source --statistics
	black --check --exclude vendor src/allencell_ml_segmenter

format: ## reformat files with black
	black --exclude vendor src/allencell_ml_segmenter -l 120