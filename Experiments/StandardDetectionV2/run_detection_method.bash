#!/usr/bin/env bash



PYTHON_LOCATION=${1:-""}
SCRIPT_LOCATION=${2:-""}
DATA_DIR=${3:-""}
NUM_PCS=${4:-""}
NUM_CAMS=${5:-""}
ZERO_START=${6:-""}
SAVE_DIR=${7:-""}

#If ZERO_START is 0 then don't use zero-start 

if [ ${ZERO_START} = "0" ]; then
    "${PYTHON_LOCATION}/wav2lip/bin/python3" "${SCRIPT_LOCATION}/window_acc_exp_v2.py" --data-dir ${DATA_DIR} \
--num_pcs ${NUM_PCS} --num-cams ${NUM_CAMS} --save-dir ${SAVE_DIR}
else
    "${PYTHON_LOCATION}/wav2lip/bin/python3" "${SCRIPT_LOCATION}/window_acc_exp_v2.py" --data-dir ${DATA_DIR} \
--num_pcs ${NUM_PCS} --num-cams ${NUM_CAMS} --zero-start --save-dir ${SAVE_DIR}
fi

