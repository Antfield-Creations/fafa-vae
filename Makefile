.PHONY: tests

check:
	pipenv run flake8 .
	pipenv run mypy .
	pipenv run yamllint .
	helm lint charts/argo-mlops-operator

tests:
	pipenv run python -m unittest discover .
