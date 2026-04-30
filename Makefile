.PHONY: install app sample train prepare hf-files

install:
	uv pip install --python .uv-venv/bin/python -e .

app:
	uv pip install --python .uv-venv/bin/python -e ".[app]"
	.uv-venv/bin/python app.py

sample:
	.uv-venv/bin/pianogen sample --output outputs/sample.mid

prepare:
	.uv-venv/bin/pianogen prepare

train:
	.uv-venv/bin/pianogen train

hf-files:
	find huggingface/space -maxdepth 4 -type f | sort
