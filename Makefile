NAMESPACE ?= data

.PHONY: tests

check:
	pipenv run flake8 .
	pipenv run mypy .
	pipenv run yamllint .
	helm lint charts/argo-mlops-operator

tests:
	pipenv run python -m unittest discover .

workflow:
	kubectl -n ${NAMESPACE} apply -f config.yaml

delete-workflow:
	kubectl -n ${NAMESPACE} delete -f config.yaml
