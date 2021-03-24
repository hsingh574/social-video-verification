#!/usr/bin/env bash
# Run format: bash -i preprocess_inference.bash <Path to ID folder> <Path to DeepForensics ID> <Path to preprocessing folder> <CCV username>
# Script will:
# 1. Setup dataset structure for ID
# 2. Copy over videos from DeepForensicsReal
# 3. Split videos contained in each <Path to ID folder>/real/cam{fake-id} into frames
# 4. Preprocess each frame for the cameras that will be faked
# 5. Transfer files to Brown CCV for inference

ID_LOCATION=${1:-""}
DATA_LOCATION=${2:-""}
PREPROCESS_LOCATION=${3:-""}
CCV_USERNAME=${4:-""}

# Exit the moment a command fails
set -euo pipefail

echo "Setting up dataset structure..."

mkdir -p "${ID_LOCATION}"
mkdir -p "${ID_LOCATION}/real"
mkdir -p "${ID_LOCATION}/real/cam0"
mkdir -p "${ID_LOCATION}/real/cam1"
mkdir -p "${ID_LOCATION}/real/cam2"
mkdir -p "${ID_LOCATION}/real/cam3"
mkdir -p "${ID_LOCATION}/real/cam4"
mkdir -p "${ID_LOCATION}/real/cam5"
mkdir -p "${ID_LOCATION}/real/cam6"
mkdir -p "${ID_LOCATION}/real/cam0/frames"
mkdir -p "${ID_LOCATION}/real/cam1/frames"
mkdir -p "${ID_LOCATION}/real/cam2/frames"
mkdir -p "${ID_LOCATION}/real/cam3/frames"
mkdir -p "${ID_LOCATION}/real/cam4/frames"
mkdir -p "${ID_LOCATION}/real/cam5/frames"
mkdir -p "${ID_LOCATION}/real/cam6/frames"

mkdir -p "${ID_LOCATION}/fake"
mkdir -p "${ID_LOCATION}/fake/cam1"
mkdir -p "${ID_LOCATION}/fake/cam3"
mkdir -p "${ID_LOCATION}/fake/cam5"

echo "Copying over videos..."

cp "${DATA_LOCATION}/camera0.mp4" "${ID_LOCATION}/real/cam0"
cp "${DATA_LOCATION}/camera1.mp4" "${ID_LOCATION}/real/cam1"
cp "${DATA_LOCATION}/camera2.mp4" "${ID_LOCATION}/real/cam2"
cp "${DATA_LOCATION}/camera3.mp4" "${ID_LOCATION}/real/cam3"
cp "${DATA_LOCATION}/camera4.mp4" "${ID_LOCATION}/real/cam4"
cp "${DATA_LOCATION}/camera5.mp4" "${ID_LOCATION}/real/cam5"
cp "${DATA_LOCATION}/camera6.mp4" "${ID_LOCATION}/real/cam6"

echo "Converting videos to frames..."

ffmpeg -i "${ID_LOCATION}/real/cam0/camera0.mp4" -loglevel quiet -q:v 2 "${ID_LOCATION}/real/cam0/frames/frames%04d.jpg"
echo "Camera 0 done."
ffmpeg -i "${ID_LOCATION}/real/cam1/camera1.mp4" -loglevel quiet -q:v 2 "${ID_LOCATION}/real/cam1/frames/frames%04d.jpg"
echo "Camera 1 done."
ffmpeg -i "${ID_LOCATION}/real/cam2/camera2.mp4" -loglevel quiet -q:v 2 "${ID_LOCATION}/real/cam2/frames/frames%04d.jpg"
echo "Camera 2 done."
ffmpeg -i "${ID_LOCATION}/real/cam3/camera3.mp4" -loglevel quiet -q:v 2 "${ID_LOCATION}/real/cam3/frames/frames%04d.jpg"
echo "Camera 3 done."
ffmpeg -i "${ID_LOCATION}/real/cam4/camera4.mp4" -loglevel quiet -q:v 2 "${ID_LOCATION}/real/cam4/frames/frames%04d.jpg"
echo "Camera 4 done."
ffmpeg -i "${ID_LOCATION}/real/cam5/camera5.mp4" -loglevel quiet -q:v 2 "${ID_LOCATION}/real/cam5/frames/frames%04d.jpg"
echo "Camera 5 done."
ffmpeg -i "${ID_LOCATION}/real/cam6/camera6.mp4" -loglevel quiet -q:v 2 "${ID_LOCATION}/real/cam6/frames/frames%04d.jpg"
echo "Camera 6 done."

echo "Preprocessing cameras 1, 3, and 5..."

#sudo docker rm dlib
sudo docker run -it --ipc host -v ${PREPROCESS_LOCATION}:/workspace -v ${ID_LOCATION}/real/cam1/frames:/DATA -v ${ID_LOCATION}/real/cam1/cropped:/RESULT --name dlib -t dlib:0.0 python preprocess_inference_multiprocess.py --frames_dir /DATA --output_dir /RESULT
echo "Camera 1 done."

sudo docker rm dlib
sudo docker run -it --ipc host -v ${PREPROCESS_LOCATION}:/workspace -v ${ID_LOCATION}/real/cam3/frames:/DATA -v ${ID_LOCATION}/real/cam3/cropped:/RESULT --name dlib -t dlib:0.0 python preprocess_inference_multiprocess.py --frames_dir /DATA --output_dir /RESULT
echo "Camera 3 done."

sudo docker rm dlib
sudo docker run -it --ipc host -v ${PREPROCESS_LOCATION}:/workspace -v ${ID_LOCATION}/real/cam5/frames:/DATA -v ${ID_LOCATION}/real/cam5/cropped:/RESULT --name dlib -t dlib:0.0 python preprocess_inference_multiprocess.py --frames_dir /DATA --output_dir /RESULT
echo "Camera 5 done."

sudo docker rm dlib

echo "Transfering files to Brown CCV..."

scp -r ${ID_LOCATION} ${CCV_USERNAME}@transfer.ccv.brown.edu:~/data/${CCV_USERNAME}/inference_videos

echo "Done!"







