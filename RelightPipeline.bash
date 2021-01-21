#!/usr/bin/env bash

#flow:

# iterate through ids
# for each id:
#   create relit fakes (how many? different lighting conditions or angles?)
# then run accuracy experiment

IMAGE_RELIGHT_DIR=${1:-"~/social-video-verification-v2/image-relighting"}

echo ${IMAGE_RELIGHT_DIR}

# cd ${IMAGE_RELIGHT_DIR} #TODO: why isn't this working?

cd ..

source lighting-env/bin/activate

cd image-relighting

"python" "live_lighting_transfer.py"
"--light_text" "/home/socialvv/social-video-verification-v2/image-relighting/data/test/light/rotate_light_04.txt"

# "--output_path "/media/eleanor/New-Volume/socialvv/social-video-verification-v2/image-relighting-output/ID1/cam3.avi""