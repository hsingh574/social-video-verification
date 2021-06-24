#!/usr/bin/env bash

# iterate through ids
# for each id:
#   move videos


EXPRESSION_OUTPUT=${3:-"/media/eleanor/New-Volume/expression-data"}

function move_id { #params: id, camera to relight

    echo 'Copying over reals for id ' ${1}
    # cams=(0 1 2 3 4 5 6)
    # for cam_num in ${cams[@]};
    # do
    mkdir -p "/${EXPRESSION_OUTPUT}/ID${1}/real/cam${cam_num}/"
    cp "/home/socialvv/Dataset/DeepForensicsReal/ID${1}/"*.mp4 "/${EXPRESSION_OUTPUT}/ID${1}/real/cam${cam_num}/"

    #TODO this won't work because it should only copy over for cams that actually have a fake
    mkdir -p "/${EXPRESSION_OUTPUT}/ID${1}/fake/cam${cam_num}/"
    cp "/home/socialvv/Dataset/DeepForensicsFake/ID${1}/"*.mp4 "/${EXPRESSION_OUTPUT}/ID${1}/fake/cam${cam_num}/"

    # done


}

# ids=(1020 1084 150 219 419 501 63 679 685 785 800 875 908 949)
ids=(1020)

for id in ${ids[@]};
do
    move_id ${id}
done

#TODO copy mat files