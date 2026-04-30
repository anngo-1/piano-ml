.PHONY: install app sample train prepare hf-files

install:
	uv sync --extra app

app:
	uv run python app.py

sample:
	uv run python -m src.generate --output outputs/sample.mid

prepare:
	uv run python -m src.data --config configs/config.json

train:
	uv run python -m src.train --config configs/config.json

hf-files:
	find huggingface/space -maxdepth 4 -type f | sort
