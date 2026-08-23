.PHONY: test lint

test:
	pytest -q

lint:
	ruff check horizon tests
