.PHONY: install test verify-assets compile help

install:
	python -m pip install -e .

test:
	python -m pytest -q

verify-assets:
	python scripts/verify_assets.py --require-all

compile:
	python -m compileall -q src scripts tests examples

help:
	@echo "Targets: install test verify-assets compile"
