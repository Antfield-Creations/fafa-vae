.PHONY: tests

check:
	pipenv run flake8 .
	pipenv run mypy .
	pipenv run yamllint .


tests:
	pipenv run python -m unittest discover .
