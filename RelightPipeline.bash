#!/usr/bin/env bash

#flow:

# iterate through ids
# for each id:
#   create relit fakes (how many? different lighting conditions or angles?)
# then run accuracy experiment

IMAGE_RELIGHT_BASE_DIR=${1:-"~/social-video-verification-v2/image-relighting"}

IMAGE_RELIGHT_LIGHTING_DIR=${2:-"/home/socialvv/social-video-verification-v2/image-relighting/data/test/light"}

IMAGE_RELIGHT_OUTPUT=${3:-"/media/eleanor/New-Volume/socialvv/social-video-verification-v2/image-relighting-output"}

# echo ${IMAGE_RELIGHT_BASE_DIR}

# cd ${IMAGE_RELIGHT_BASE_DIR} #TODO: why isn't this working?

function relight_id { #params: id, camera to relight

    ### to be changed/randomized:
    # ROTATE_LIGHT=$(( RANDOM % 7 )) #random number (0 to 6)
    ROTATE_LIGHT=0
    CAMERA_NUM=${2}

    cd ..

    source lighting-env/bin/activate

    cd image-relighting

    mkdir -p "/${IMAGE_RELIGHT_OUTPUT}/ID${1}"

    # echo 'Creating deep fake for id '${1} 'and camera' ${CAMERA_NUM}
    # "python" "live_lighting_transfer.py" "--light_text" "${IMAGE_RELIGHT_LIGHTING_DIR}/rotate_light_0${ROTATE_LIGHT}.txt" "--input_path" "/home/socialvv/Dataset/EleanorTrain/ID${1}/camera${CAMERA_NUM}.MP4" "--output_path" "/${IMAGE_RELIGHT_OUTPUT}/ID${1}/light${ROTATE_LIGHT}_camera${CAMERA_NUM}_id${1}.avi"

    echo 'Getting sh coords for id '${1}
    "python" "analyze_lighting_multiple.py" "--videos_path" "/home/socialvv/Dataset/EleanorTrain/ID${1}/" "--frames" "1000" "--mat_path" "/home/socialvv/social-video-verification-v2/social-video-verification/Experiments/DataSHCoords/fake${CAMERA_NUM}-ID${1}.mat" "--fake_path" "/home/socialvv/socialvv/social-video-verification-v2/image-relighting-output/ID${1}/light${ROTATE_LIGHT}_camera${CAMERA_NUM}_id${1}.avi"

}

ids=(1 2 3)

for id in ${ids[@]};
do
    relight_id ${id} 2
    relight_id ${id} 3
    relight_id ${id} 4
done

cd ..

echo 'finished relighting and analysis'

echo pwd

cd social-video-verification
cd Experiments

# matlab -nosplash -nodesktop -r "relight_windowed_acc_v2"

# matlab -nosplash -nodesktop -r "relight_make_plots"