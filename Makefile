PYTHON ?= python
WORK ?= /tmp/orbit_all

.PHONY: install test lint smoke discovery-list tune-list

install:
	$(PYTHON) -m pip install -e ".[data,dev]"

test:
	pytest -q

lint:
	ruff check src scripts tests

smoke:
	$(PYTHON) scripts/campaign.py --phase smoke --work-dir $(WORK) --device cpu --synthetic --steps-override 2

discovery-list:
	$(PYTHON) scripts/campaign.py --phase discovery --work-dir $(WORK) --list-tasks

tune-list:
	$(PYTHON) scripts/campaign.py --phase matched_tune --work-dir $(WORK) --list-tasks
