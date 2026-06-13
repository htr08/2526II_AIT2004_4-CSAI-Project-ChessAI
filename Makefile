VENV_DIR := .venv
PYTHON := python3
VENV_PY := $(VENV_DIR)/bin/python

.PHONY: all venv check clean

all: venv check

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
	else \
		echo "Virtual environment already exists, skipping."; \
	fi

check: venv
	@$(VENV_PY) env_check.py

clean:
	rm -rf $(VENV_DIR)