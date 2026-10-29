.PHONY: install download test train evaluate app format lint

install:
	pip install -r requirements.txt

download:
	python -m data.download

test:
	pytest -q

train:
	python -m src.train --model best

evaluate:
	python -m src.evaluate

app:
	streamlit run app/streamlit_app.py

format:
	ruff format .

lint:
	ruff check .