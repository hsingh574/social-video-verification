#!/usr/bin/env bash

#flow:

# iterate through ids
# for each id:
#   create relit fakes (how many? different lighting conditions or angles?)
# then run accuracy experiment

IMAGE_RELIGHT_DIR = '~/social-video-verification-v2/image-relighting'

python "${IMAGE_RELIGHT_DIR}/live_lighting_transfer.py" "--light_text /home/socialvv/social-video-verification-v2/image-relighting/data/test/light/rotate_light_04.txt" "--input_path "/home/socialvv/Dataset/ID1/camera3.MP4"" "--output_path "/media/eleanor/New-Volume/socialvv/social-video-verification-v2/image-relighting-output/ID1/cam3.avi""