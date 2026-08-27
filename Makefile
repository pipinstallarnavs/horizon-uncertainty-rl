.PHONY: test lint

test:
	pytest -q

lint:
	ruff check horizon tests

.PHONY: verify calibrate
verify:
	python -m horizon.report

calibrate:
	python -m horizon.calibrate.search

.PHONY: dryrun
dryrun:
	python -m horizon.dryrun
