.PHONY: install dev pipeline dashboard test lint format docker-up docker-down

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

pipeline:
	python -m src.pipeline

dashboard:
	streamlit run dashboard/app.py

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

docker-up:
	docker compose up --build

docker-down:
	docker compose down
