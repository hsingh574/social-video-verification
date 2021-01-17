#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dlib
import sys
import cv2
import os

print("Dlib using cuda?")
print(dlib.DLIB_USE_CUDA)


face_detector_path = sys.argv[1]
frames_path = sys.argv[2]
save_path = sys.argv[3]

cnn_face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)


print('Reading video frames...')

with open(save_path, 'w+') as out:
    for f in os.listdir(frames_path):
        frame = cv2.imread(os.path.join(frames_path, f))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = cnn_face_detector(frame, 1)
        sortedDets = sorted(dets, key=lambda a: a.confidence, reverse=True)
        if(len(dets) == 0):
            print('No faces detected. Using last detection result.')
        else:
            d = sortedDets[0]
        out.write('%d, %d, %d, %d\n' % (d.rect.left(), 
                                                    d.rect.top(), 
                                                    d.rect.right(), 
                                                   d.rect.bottom()))
        
