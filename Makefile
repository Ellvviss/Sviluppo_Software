IS_GITHUB_ACTIONS := $(findstring true,$(CI))
ifeq ($(OS),Windows_NT)
    R_PYTHON = venv\Scripts\python.exe
    R_PIP = venv\Scripts\pip.exe
	SET_PYTHONPATH = set PYTHONPATH=. &
else
    R_PYTHON = ./venv/bin/python
    R_PIP = ./venv/bin/pip
    SET_PYTHONPATH = PYTHONPATH=.
endif
ifeq ($(IS_GITHUB_ACTIONS),true)
    R_PYTHON = python
    R_PIP = pip
endif

venv:
	python -m venv venv

install:
	$(R_PIP) install --upgrade pip
	$(R_PIP) install -r requirements.txt
	@echo Installazione delle dipendenze terminata.
lint:
	$(SET_PYTHONPATH) $(R_PYTHON) -m pylint --disable=R,C src/*.py tests/*.py
	@echo Linting complete.
test:
	$(SET_PYTHONPATH) $(R_PYTHON) -m pytest -vv --cov=src tests/
	@echo Testing complete.

init-poetry:
	$(R_PIP) install poetry
	-poetry init --no-interaction
	$(R_PYTHON) -c "import os; [os.system(f'poetry add {line.strip()}') for line in open('requirements.txt') if line.strip()]"

build:
	$(R_PYTHON) -m build
	@echo "Build complete. Check dist/ directory."

clean-build:
	rm -rf dist/ build/ *.egg-info .pytest_cache .coverage __pycache__

ifeq ($(OS),Windows_NT)
    DOCKER_PWD := $(subst \,/,${CURDIR})
else
    DOCKER_PWD := $(CURDIR)
endif
docker:
	docker build -t dogorcat .

docker_run: docker
	docker run --rm --name dogorcat -v "$(DOCKER_PWD)/persistent_data:/app/results" dogorcat python src/ai.py
	docker image prune -f