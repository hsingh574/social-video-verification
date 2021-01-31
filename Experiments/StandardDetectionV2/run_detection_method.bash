#!/usr/bin/env bash



PYTHON_LOCATION=${1:-""}
SCRIPT_LOCATION=${2:-""}
DATA_DIR=${3:-""}
NUM_PCS=${4:-""}
NUM_PARTICIPANTS=${5:-""}
SAVE_DIR=${6:-""}


"${PYTHON_LOCATION}/wav2lip/bin/python3" "${SCRIPT_LOCATION}/window_acc_exp_v2.py" --data-dir ${DATA_DIR} \
--num_pcs ${NUM_PCS} --num_participants ${NUM_PARTICIPANTS} --save-dir ${SAVE_DIR}