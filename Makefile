.PHONY: test docker

test:
	pytest -v --cov=jointly_hic --cov-report=term-missing --cov-fail-under=80 tests/

docker:
	docker build -t jointly_hic .
