.PHONY: demo demo-check demo-download demo-offline demo-clean
demo:
	python -m tools.quickstart
demo-check:
	python -m tools.quickstart --check-only
demo-download:
	python -m tools.quickstart --download-only
demo-offline:
	python -m tools.quickstart --offline
demo-clean:
	python -m tools.quickstart --clean-output --check-only
