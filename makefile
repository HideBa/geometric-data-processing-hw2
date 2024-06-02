PACKAGE_DIR := .

.PHONY: install
install:
	poetry install

.PHONY: update
update:
	poetry update

.PHONY: lint
lint:
	poetry run flake8 $(PACKAGE_DIR)/**/*.py

.PHONY: type
type:
	poetry run mypy $(PACKAGE_DIR)/**/*.py

.PHONY: format
format:
	poetry run black $(PACKAGE_DIR)/**/*.py

.PHONY: sort
sort:
	poetry run isort $(PACKAGE_DIR)/**/*.py

.PHONY: test
test:
	poetry run pytest

.PHONY: check
check: lint type test	format sort

.PHONY: blender
blender:
	zsh -i -c 'blender --python ${PACKAGE_DIR}/run.py'

.PHONY: blender-test
blender-test:
	zsh -i -c 'blender --background --python ${PACKAGE_DIR}/test.py'