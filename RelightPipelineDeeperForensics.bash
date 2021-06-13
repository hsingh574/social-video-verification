#!/usr/bin/env bash

#flow:

# iterate through ids
# for each id:
#   create relit fakes (how many? different lighting conditions or angles?)
# then run accuracy experiment

IMAGE_RELIGHT_BASE_DIR=${1:-"~/social-video-verification-v2/image-relighting"}

IMAGE_RELIGHT_LIGHTING_DIR=${2:-"/home/socialvv/social-video-verification-v2/image-relighting/data/test/light"}

IMAGE_RELIGHT_OUTPUT=${3:-"/media/eleanor/New-Volume/relighting-data"}

# echo ${IMAGE_RELIGHT_BASE_DIR}

# cd ${IMAGE_RELIGHT_BASE_DIR} #TODO: why isn't this working?

function relight_id { #params: id, camera to relight

    ### to be changed/randomized:
    # ROTATE_LIGHT=$(( RANDOM % 7 )) #random number (0 to 6)
    ROTATE_LIGHT=4
    CAMERA_NUM=${2}

    cd ..

    source lighting-env/bin/activate

    cd image-relighting

    mkdir -p "/${IMAGE_RELIGHT_OUTPUT}/ID${1}"

    echo 'Creating deep fake for id '${1} 'and camera' ${CAMERA_NUM}
    "python" "live_lighting_transfer.py" "--light_text" "${IMAGE_RELIGHT_LIGHTING_DIR}/rotate_light_0${ROTATE_LIGHT}.txt" "--input_path" "/home/socialvv/Dataset/DeepForensicsReal/ID${1}/camera${CAMERA_NUM}.mp4" "--output_path" "/${IMAGE_RELIGHT_OUTPUT}/ID${1}/fake/cam${CAMERA_NUM}/camera${CAMERA_NUM}.avi"

    mkdir -p "/${IMAGE_RELIGHT_OUTPUT}/mat_files"

    echo 'Getting sh coords for id '${1}
    "python" "analyze_lighting_multiple.py" "--videos_path" "/home/socialvv/Dataset/DeepForensicsReal/ID${1}/" "--frames" "1500" "--mat_path" "/${IMAGE_RELIGHT_OUTPUT}/mat_files/fake${CAMERA_NUM}-ID${1}.mat" "--fake_path" "/${IMAGE_RELIGHT_OUTPUT}/ID${1}/fake/cam${CAMERA_NUM}/camera${CAMERA_NUM}.avi"

}

# ids=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
ids=(100)

for id in ${ids[@]};
do
    relight_id ${id} 1 #relight most head-on IDs for this dataset
    relight_id ${id} 3
    relight_id ${id} 5
done

cd ..

echo 'finished relighting and analysis'

echo pwd

# cd social-video-verification
# cd Experiments

# matlab -nosplash -nodesktop -r "relight_windowed_acc_v2"

# matlab -nosplash -nodesktop -r "relight_make_plots"