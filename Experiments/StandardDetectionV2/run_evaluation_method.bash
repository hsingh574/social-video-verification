#!/usr/bin/env bash



PYTHON_LOCATION=${1:-""}
SCRIPT_LOCATION=${2:-""}
RESULTS_DIR=${3:-""}
DATA_DIR=${4:-""}
SAVE_DIR=${5:-""}


"${PYTHON_LOCATION}/wav2lip/bin/python3" "${SCRIPT_LOCATION}/make_plots_v2.py" --results-dir ${RESULTS_DIR} \
--data-dir ${DATA_DIR} --save-dir ${SAVE_DIR}