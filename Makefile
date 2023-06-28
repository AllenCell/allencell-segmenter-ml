.DEFAULT_GOAL := test

# See https://tech.davis-hansson.com/p/make/
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

##############################################

PYTHON_VERSION = python3.10
VENV_NAME := venv
VENV_BIN := $(VENV_NAME)/bin
ACTIVATE = $(VENV_NAME)/bin/activate
PYTHON = $(VENV_NAME)/bin/python3

$(PYTHON):
> $(PYTHON_VERSION) -m venv --upgrade-deps $(VENV_NAME)

venv: $(PYTHON)

# Necessary to supply private package source until lkaccess is removed or made public
install: venv
> $(PYTHON) -m pip install --index-url='https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-virtual/simple' --extra-index-url='https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-snapshot-local/simple' . 
.PHONY: install

install-dev: venv install
> $(PYTHON) -m pip install .[dev]
.PHONY: install-dev

install-test-lint: venv install
> $(PYTHON) -m pip install .[test_lint]
.PHONY: install-test-lint

clean:
> rm -fr build/
> rm -fr dist/
> rm -fr .eggs/
> find . -name '*.egg-info' -exec rm -fr {} +
> find . -name '*.egg' -exec rm -f {} +
> find . -name '*.pyc' -exec rm -f {} +
> find . -name '*.pyo' -exec rm -f {} +
> find . -name '*~' -exec rm -f {} +
> find . -name '__pycache__' -exec rm -fr {} +
> rm -fr .coverage
> rm -fr coverage.xml
> rm -fr htmlcov/
> rm -fr .pytest_cache
> rm -fr ./venv
.PHONY: clean

test: install-test-lint ## run pytest
> pytest
.PHONY: test

test-cov: install-test-lint ## run pytest with coverage report
> pytest --cov=allencell_ml_segmenter
.PHONY: test

lint: ## run a lint check / report
> flake8 src/allencell_ml_segmenter --count --verbose --show-source --statistics
> black --check --exclude vendor src/allencell_ml_segmenter
.PHONY: lint

format: ## reformat files with black
> black --exclude vendor src/allencell_ml_segmenter
.PHONY: format

bumpversion-release: venv
> bumpversion --list release
.PHONY: bumpversion-release

bumpversion-major: venv
> bumpversion --list major
.PHONY: bumpversion-major

bumpversion-minor: venv
> bumpversion --list minor
.PHONY: bumpversion-minor

bumpversion-patch: venv
> bumpversion --list --allow-dirty patch
.PHONY: bumpversion-patch

bumpversion-dev: venv
> bumpversion --list devbuild
.PHONY: bumpversion-dev

version: venv
> $(PYTHON) setup.py --version
.PHONY: version