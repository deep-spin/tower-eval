.PHONY: setup install publish flake8-test black-test tests

setup:
	pip install "poetry"
	poetry config virtualenvs.create false

install: setup
	poetry install

publish: install
	poetry publish --build
