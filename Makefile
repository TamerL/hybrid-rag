.PHONY: install ingest serve eval-retrieval eval-ragas test docker

install:
	pip install -r requirements.txt

ingest:
	python scripts/ingest.py

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

eval-retrieval:
	python scripts/evaluate.py retrieval

eval-ragas:
	python scripts/evaluate.py ragas

test:
	pytest tests/ -v

docker:
	docker compose up --build
