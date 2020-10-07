#!/usr/bin/env bash
# Run format: bash -i data-pipelineV2 <Video folder> <DeepFake Save folder> <Script folder (with data processing scripts)> 
# <Location of Wav2Lip clone> <Checkpoint for Wav2Lip model> <Audio Filename> <Number of videos (cameras)> <Single ID>
#
# Assumes that downsampled and retricted videos have already been generated 
# (and are of the format VIDEO_LOCATION/cam1-lipgan/cam1-first-4k.mp4)
#
# Assumes audio file is present in the audio folder of WAV2LIP_LOCATION
#
#
# Script will:
# 1. Create DeepFake for each camera view of a video
# 2. Split video into frames
# 3. Crop around the faces in these frames, and save the bounding box coordinates to a txt file
# 4. Run 2D landmark detection on these cropped images
#
# Assumes 6 cameras with 29.9fps mp4 video, named camera1.MP4, ..., camera6.MP4
#
# Can be run in two modes, either for a single ID, or over all IDs at once
#
#
# Written by Eleanor Tursman
# Modified by Adam Pikielny, Harman Suri September 2020

VIDEO_LOCATION_BASE=${1:-""}
DEEPFAKE_LOCATION_BASE=${2:-""}
SCRIPT_LOCATION=${3:-""}
WAV2LIP_LOCATION=${4:-""}
WAV2LIP_CHECKPOINT=${5:-""}
AUDIO_FILENAME=${6:-""}
NUM_VIDEOS=${7:-""}

# If you pass in a single ID here (e.g. ID2) script will be run for only that ID. 
# To iterate over all IDs, pass in the string "ALL" and will iterate over VIDEO_LOCATION_BASE_DIRECTORY
SINGLE_ID=${8:-""}


# Exit the moment a command fails
set -euo pipefail


# To run lipGan on our machine, we needed to downsample & restrict the video length to the first 4000 frames
# TODO: Double check this is still true for Wav2Lip, if not true, then can to re-generate 


# Function singleID: runs deepfake generation and processing for NUM_VIDEOS for a single ID
# Args:
# $1: DEEPFAKE_LOCATION
# $2: NUM_VIDEOS
# $3: WAV2LIP_LOCATION
# $4: WAV2LIP_CHECKPOINT
# $5: VIDEO_LOCATION
# $6: AUDIO_FILENAME
# $7: SCRIPT_LOCATION


function singleID {

mkdir -p "${1}/bounding-boxes"

for ((i=1; i<= $2; i++));
do
    
################## Create deepfake ##################
   echo "Creating deepfake for video $i with Wav2Lip..."
    
   mkdir -p "${1}/cam${i}-wav2lip"

   cd "${3}"
   
   wav2lip/bin/python3 inference.py --checkpoint_path "${4}" --face "${5}/cam${i}-lipgan/cam${i}-first-4k.mp4" --audio "audio/${6}.wav" --pads 0 40 0 0 --blending
   mv results/result_voice.mp4 "${1}/cam${i}-wav2lip/cam${i}-wav2lip.mp4"


################## Process deepfake ##################

   echo "Converting deepfake video to frames..."


   mkdir -p "${1}/cam${i}-wav2lip/frames"
   ffmpeg -i "${1}/cam${i}-wav2lip/cam${i}-wav2lip.mp4" -loglevel quiet -q:v 2 "${1}/cam${i}-wav2lip/frames/frames%04d.jpg" 
   echo "Deepfake for Video ${i} frame generation done."
   
   echo "Cropping faces..."
   
   mkdir -p "${1}/cam${i}-wav2lip/cropped"   
   wav2lip/bin/python3 "${7}/cnn_face_detector.py" "${7}/mmod_human_face_detector.dat" "${1}/cam${i}-wav2lip/frames/" "${1}/cam${i}-wav2lip/cropped/" "${1}/bounding-boxes/cam${i}-post-wav2lip-"
   echo "Cropping faces for Video ${i} done."
   
   echo "Running 2D landmark detection..."
   
   mkdir -p "${1}/cam${i}-wav2lip/landmarks"
   wav2lip/bin/python3 "${7}/detectFeatures.py" "${1}/cam${i}-wav2lip/frames/" "${1}/cam${i}-wav2lip/landmarks" "${1}/bounding-boxes/cam${i}-post-wav2lip-bounding-boxes.txt"
   echo "Running 2D landmark detection for Video ${i} done."
   
done

}



if [[ ${SINGLE_ID} =~ ID([1-9]|[1-9]{1}[0-9]{1})$ ]]; then
    DEEPFAKE_LOCATION="${DEEPFAKE_LOCATION_BASE}/${SINGLE_ID}/"
    VIDEO_LOCATION="${VIDEO_LOCATION_BASE}/${SINGLE_ID}/"

    singleID ${DEEPFAKE_LOCATION} ${NUM_VIDEOS} ${WAV2LIP_LOCATION} ${WAV2LIP_CHECKPOINT} ${VIDEO_LOCATION} ${AUDIO_FILENAME} ${SCRIPT_LOCATION}
    
    
else
    cd ${VIDEO_LOCATION_BASE}
    
# loop over all ID directories in base directory
    for dir in */ ;
    do
        echo ${dir%?}
        
        
        
        #hardcoded for now because generation stopped midway
        #TODO: remove this for general case
        
        if [[ ${dir%?} =~ ID(1)$ || ${dir%?} =~ ID(10)$ || ${dir%?} =~ ID(11)$ || ${dir%?} =~ ID(12)$ || ${dir%?} =~ ID(13)$ || ${dir%?} =~ ID(14)$ || ${dir%?} =~ ID(17)$ || ${dir%?} =~ ID(18)$ || ${dir%?} =~ ID(19)$ || ${dir%?} =~ ID(20)$ || ${dir%?} =~ ID(21)$ || ${dir%?} =~ ID(22)$ || ${dir%?} =~ ID(23)$ || ${dir%?} =~ ID(2)$ || ${dir%?} =~ ID(15)$ || ${dir%?} =~ ID(16)$ ]]; then
        continue
        fi
        
        
        
        
        if [[ ${dir%?} =~ ID([1-9]|[1-9]{1}[0-9]{1})$ ]]; then
            echo ${dir%?}
            DEEPFAKE_LOCATION="${DEEPFAKE_LOCATION_BASE}/${dir%?}"
            VIDEO_LOCATION="${VIDEO_LOCATION_BASE}/${dir%?}"
            singleID ${DEEPFAKE_LOCATION} ${NUM_VIDEOS} ${WAV2LIP_LOCATION} ${WAV2LIP_CHECKPOINT} ${VIDEO_LOCATION} ${AUDIO_FILENAME} ${SCRIPT_LOCATION}
        fi
            
    done


fi









