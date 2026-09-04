.PHONY: demo data train test serve docker-up

# Full demo pipeline: generate data -> quality check -> train best config -> run tests
demo: data train test

data:
	python3 src/data/synthetic_generator.py
	python3 -m src.data.prepare

train:
	python3 -m src.training.train --config configs/densenet.yaml

experiments:
	python3 -m src.training.train --config configs/baseline.yaml
	python3 -m src.training.train --config configs/densenet.yaml
	python3 -m src.training.train --config configs/efficientnet.yaml
	python3 -m src.training.train --config configs/vit.yaml
	python3 -m src.evaluation.build_experiment_table

test:
	pytest -v

serve:
	uvicorn app.main:app --reload

docker-up:
	docker compose up --build
