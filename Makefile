NAMESPACE ?= data
GOOGLE_APPLICATION_CREDENTIALS ?= ~/Nextcloud/Documents/fafa-vae-0102b4da1611.json

.PHONY: tests

check:
	pipenv run flake8 .
	pipenv run mypy .
	pipenv run yamllint .
	helm lint charts/argo-mlops-operator

tests:
	GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS} pipenv run python -m unittest discover .

workflow:
	kubectl -n ${NAMESPACE} apply -f config.yaml

delete-workflow:
	kubectl -n ${NAMESPACE} delete -f config.yaml
