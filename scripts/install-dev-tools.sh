#!/bin/sh

pip install poetry
poetry config virtualenvs.in-project true 
poetry config cache-dir ${WORKSPACE_DIR}/.cache
poetry install
