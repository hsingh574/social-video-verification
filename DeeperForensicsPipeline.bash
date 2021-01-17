#!/usr/bin/env bash


# 1. Get 2D landmarks on original camera files
# 2. Create the 3 random fakes per ID
# 3. Get 2D landmarks on fakes


## Note: 7 videos per ID here ##




VIDEO_LOCATION_BASE=${1:-""}
DEEPFAKE_LOCATION_BASE=${2:-""}
SCRIPT_LOCATION=${3:-""}
WAV2LIP_LOCATION=${4:-""}
WAV2LIP_CHECKPOINT=${5:-""}
AUDIO_FILENAME=${6:-""}
NUM_CAMS=${7:-""}
NUM_FAKES=${8:-""}

# If you pass in a single ID here (e.g. ID2) script will be run for only that ID. 
# To iterate over all IDs, pass in the string "ALL" and will iterate over VIDEO_LOCATION_BASE_DIRECTORY
SINGLE_ID=${9:-""}



# Exit the moment a command fails
set -euo pipefail


function singleID {

mkdir -p "${1}/bounding-boxes"


for ((i=0; i< $7; i++));
do

    ### Make frame directory to speed up subsequent steps and then remove it to save memory ###
    
    mkdir -p "${1}/cam${i}-frames"
    echo "Getting frames for cam $i at ${1}"
    ffmpeg -i "${1}/camera${i}.mp4" -loglevel quiet -q:v 2 "${1}/cam${i}-frames/frames%04d.jpg" 
    

    ### Get face bounding boxes for each real video ####
    echo "Getting bounding boxes for cam $i at ${1}"
    "${4}/wav2lip/bin/python3" "${3}/cnn_face_detector_v2.py" "${3}/mmod_human_face_detector.dat" "${1}/cam${i}-frames" "${1}/bounding-boxes/cam${i}-bounding-boxes.txt"
    
    
    ### Get face bounding boxes for each real video ####
    echo "Getting 2d landmarks for cam $i at ${1}"
    
    mkdir -p "${1}/cam${i}-landmarks"
    
    "${4}/wav2lip/bin/python3" "${3}/detectFeatures_v2.py" "${1}/cam${i}-frames" "${1}/cam${i}-landmarks" "${1}/bounding-boxes/cam${i}-bounding-boxes.txt"
    
    ### Delete frame directory as we don't need it anymore ###
    
    rm -rf "${1}/cam${i}-frames"

done

mkdir -p "${2}/bounding-boxes"

fakes=($(shuf -i 0-6 -n $8))

for i in "${fakes[@]}"; 
do


    ### Create the deepfake ###
    echo "Creating deepfake for video $i  at ${1} with Wav2Lip..."
    
    "${4}/wav2lip/bin/python3" "${4}/inference.py" --checkpoint_path "${5}" --face "${1}/camera${i}.mp4" --audio "${4}/audio/${6}.wav" --pads 40 10 15 15 --bboxFile "${1}/bounding-boxes/cam${i}-bounding-boxes.txt" 
    mv "${4}/results/result_voice.mp4" "${2}/cam${i}-wav2lip.mp4"
    
    
    ### Process deepfake in identical manner to real video ###
    
    mkdir -p "${2}/cam${i}-frames"
    echo "Getting frames for cam $i at ${2}"
    ffmpeg -i "${2}/cam${i}-wav2lip.mp4" -loglevel quiet -q:v 2 "${2}/cam${i}-frames/frames%04d.jpg" 
    

    echo "Getting bounding boxes for cam $i at ${2}"
    "${4}/wav2lip/bin/python3" "${3}/cnn_face_detector_v2.py" "${3}/mmod_human_face_detector.dat" "${2}/cam${i}-frames" "${2}/bounding-boxes/cam${i}-bounding-boxes.txt"
    
    
    echo "Getting 2d landmarks for cam $i at ${1}"
    mkdir -p "${2}/cam${i}-landmarks"
    "${4}/wav2lip/bin/python3" "${3}/detectFeatures_v2.py" "${2}/cam${i}-frames" "${2}/cam${i}-landmarks" "${2}/bounding-boxes/cam${i}-bounding-boxes.txt"
    
    
    rm -rf "${2}/cam${i}-frames"
done

}


if [[ ${SINGLE_ID} =~ ID([1-9]|[1-9]{1}[0-9]{1}|[1-9]{1}[0-9]{2}|[1-9]{1}[0-9]{3})$ ]]; then
    mkdir -p "${DEEPFAKE_LOCATION_BASE}/${SINGLE_ID}"
    DEEPFAKE_LOCATION="${DEEPFAKE_LOCATION_BASE}/${SINGLE_ID}"
    VIDEO_LOCATION="${VIDEO_LOCATION_BASE}/${SINGLE_ID}"
    
    singleID ${VIDEO_LOCATION} ${DEEPFAKE_LOCATION} ${SCRIPT_LOCATION} ${WAV2LIP_LOCATION} ${WAV2LIP_CHECKPOINT} ${AUDIO_FILENAME} ${NUM_CAMS} ${NUM_FAKES}
        
else
    cd ${VIDEO_LOCATION_BASE}
    
# loop over all ID directories in base directory
    for dir in */ ;
    do
        echo "Working on directory ${dir%?}"
        
        
        ### Make the fake directory if it does not exist
        
        mkdir -p "${DEEPFAKE_LOCATION_BASE}/${dir%?}"
        DEEPFAKE_LOCATION="${DEEPFAKE_LOCATION_BASE}/${dir%?}"
        VIDEO_LOCATION="${VIDEO_LOCATION_BASE}/${dir%?}"
        singleID ${VIDEO_LOCATION} ${DEEPFAKE_LOCATION} ${SCRIPT_LOCATION} ${WAV2LIP_LOCATION} ${WAV2LIP_CHECKPOINT} ${AUDIO_FILENAME} ${NUM_CAMS} ${NUM_FAKES}
            
    done


fi




