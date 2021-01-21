#!/usr/bin/env bash

#flow:

# iterate through ids
# for each id:
#   create relit fakes (how many? different lighting conditions or angles?)
# then run accuracy experiment

IMAGE_RELIGHT_BASE_DIR=${1:-"~/social-video-verification-v2/image-relighting"}

IMAGE_RELIGHT_LIGHTING_DIR=${2:-"/home/socialvv/social-video-verification-v2/image-relighting/data/test/light"}

echo ${IMAGE_RELIGHT_BASE_DIR}

# cd ${IMAGE_RELIGHT_BASE_DIR} #TODO: why isn't this working?

cd ..

source lighting-env/bin/activate

cd image-relighting

"python" "live_lighting_transfer.py" "--light_text" "${IMAGE_RELIGHT_LIGHTING_DIR}/rotate_light_04.txt" "--input_path" "/home/socialvv/Dataset/ID1/camera3.MP4" "--output_path" "/media/eleanor/New-Volume/socialvv/social-video-verification-v2/image-relighting-output/ID1/bashScriptTest.avi"