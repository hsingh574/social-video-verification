DEEPFAKE_LOCATION=${1:-""}
VIDEO_NAME=${2:-""}
SCRIPT_LOCATION=${3:-""}

set -euo pipefail

mkdir -p "${DEEPFAKE_LOCATION}/bounding-boxes"

echo "Converting deepfake video to frames..."
mkdir -p "${DEEPFAKE_LOCATION}/${VIDEO_NAME}/frames"
ffmpeg -i "${DEEPFAKE_LOCATION}/${VIDEO_NAME}/${VIDEO_NAME}.mp4" -loglevel quiet -q:v 2 "${DEEPFAKE_LOCATION}/${VIDEO_NAME}/frames/frames%04d.jpg"
echo "Frame generation done."

echo "Cropping faces..."
mkdir -p "${DEEPFAKE_LOCATION}/${VIDEO_NAME}/cropped"
/home/socialvv/social-video-verification-v2/Wav2Lip/wav2lip/bin/python3 "${SCRIPT_LOCATION}/cnn_face_detector_copy.py" "${SCRIPT_LOCATION}/mmod_human_face_detector.dat" "${DEEPFAKE_LOCATION}/${VIDEO_NAME}/frames/" "${DEEPFAKE_LOCATION}/${VIDEO_NAME}/cropped/" "${DEEPFAKE_LOCATION}/bounding-boxes/${VIDEO_NAME}-"
echo "Cropping faces done."

echo "Running 2D landmark detection..."
mkdir -p "${DEEPFAKE_LOCATION}/${VIDEO_NAME}/landmarks"
/home/socialvv/social-video-verification-v2/Wav2Lip/wav2lip/bin/python3 "${SCRIPT_LOCATION}/detectFeatures.py" "${DEEPFAKE_LOCATION}/${VIDEO_NAME}/frames/" "${DEEPFAKE_LOCATION}/${VIDEO_NAME}/landmarks" "${DEEPFAKE_LOCATION}/bounding-boxes/${VIDEO_NAME}-bounding-boxes.txt"
echo "Running 2D landmark detection done."





