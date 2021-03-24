#!/usr/bin/env bash
sudo docker rm dlib
sudo docker run -it --ipc host -v /media/eleanor/New-Volume/faceshifter/preprocess:/workspace -v /media/eleanor/New-Volume/identity-data/ID63/real/cam5/frames:/DATA -v /media/eleanor/New-Volume/identity-data/ID63/real/cam5/cropped:/RESULT --name dlib -t dlib:0.0 python preprocess_inference.py --frames_dir /DATA --output_dir /RESULT
