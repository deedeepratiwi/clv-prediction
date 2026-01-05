.PHONY: format lint test run api

format:
	uv run black .
	uv run ruff check . --fix

lint:
	uv run ruff check .

test:
	uv run pytest -v

run:
	python src/train.py

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000
