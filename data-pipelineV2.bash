#!/usr/bin/env bash
# Run format: bash -i data-pipelineV2 <Video folder> <DeepFake Save folder> <Script folder (with data processing scripts)> 
# <Location of Wav2Lip clone> <Checkpoint for Wav2Lip model> <Audio Filename> <Number of videos (cameras)>
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
# Written by Eleanor Tursman
# Modified by Adam Pikielny, Harman Suri September 2020

VIDEO_LOCATION=${1:-""}
DEEPFAKE_LOCATION=${2:-""}
SCRIPT_LOCATION=${3:-""}
WAV2LIP_LOCATION=${4:-""}
WAV2LIP_CHECKPOINT=${5:-""}
AUDIO_FILENAME=${6:-""}
NUM_VIDEOS=${7:-""}

# Exit the moment a command fails
set -euo pipefail


# To run lipGan on our machine, we needed to downsample & restrict the video length to the first 4000 frames
# TODO: Double check this is still true for Wav2Lip, if not true, then can to re-generate 


for i in {1..${NUM_VIDEOS}}
do
    
################## Create deepfake ##################
   echo "Creating deepfake for video ${i} with Wav2Lip..."
    
   mkdir -p "${DEEPFAKE_LOCATION}/cam${i}-wav2lip"
#  ffmpeg -i "${VIDEO_LOCATION}/camera${i}.MP4" -ss 0.0 -frames:v 4000 -framerate 29.97 -vf scale=1280:720 -crf 0 "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/cam${i}-first-4k.mp4"
#  echo "Video ${i} downsamling and restriction done"
#  python3 "${SCRIPT_LOCATION}/get-new-bboxes.py" ${i} "${DEEPFAKE_LOCATION}"

   echo "${WAV2LIP_LOCATION}"
   
   cd "${WAV2LIP_LOCATION}"
   
   source wav2lip/bin/activate
   
   echo "${VIDEO_LOCATION}/cam${i}-lipgan/cam${i}-first-4k.mp4"
   echo "audio/${AUDIO_FILENAME}.wav"
   
   
   python3 inference.py --checkpoint_path "${WAV2LIP_CHECKPOINT}" --face "${VIDEO_LOCATION}/cam${i}-lipgan/cam${i}-first-4k.mp4" --audio "audio/${AUDIO_FILENAME}.wav"
   mv results/result_voice.mp4 "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/cam${i}-wav2lip.mp4"
   
   deactivate
   
# conda deactivate

################## Process deepfake ##################

   echo "Converting deepfake video to frames..."


   mkdir -p "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/frames"
   ffmpeg -i "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/cam${i}-wav2lip.mp4" -loglevel quiet -q:v 2 "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/frames/frames%04d.jpg" 
   echo "Deepfake for Video ${i} frame generation done."
   
   echo "Cropping faces..."
   
   mkdir -p "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/cropped"
   python3 "${SCRIPT_LOCATION}/cnn_face_detector.py" "${SCRIPT_LOCATION}/mmod_human_face_detector.dat" "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/frames/" "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/cropped/" "${DEEPFAKE_LOCATION}/bounding-boxes/cam${i}-post-wav2lip-"
   echo "Cropping faces for Video ${i} done."
   
   echo "Running 2D landmark detection..."
   
   mkdir -p "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/landmarks"
   python3 "${SCRIPT_LOCATION}/detectFeatures.py" "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/frames/" "${DEEPFAKE_LOCATION}/cam${i}-wav2lip/landmarks" "${DEEPFAKE_LOCATION}/bounding-boxes/cam${i}-post-wav2lip-bounding-boxes.txt"
   echo "Running 2D landmark detection for Video ${i} done."
   
done








