#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# From: https://github.com/1adrianb/face-alignment

import face_alignment
from skimage import io
import numpy as np
import os
import sys

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)

frames_path = sys.argv[1]
save_path = sys.argv[2]
boundingBoxFile = sys.argv[3]


numImg = len([name for name in os.listdir(frames_path) if 
                  os.path.isfile(os.path.join(frames_path, name))])

with open(boundingBoxFile, 'r') as out:
    for f in range(1,numImg+1):
        number = '{0:04d}'.format(f)
        filename = os.path.join(frames_path, "frames" + number + ".jpg")
        box = out.readline().split(',')
        if box[0]=='':
            continue
        boxInt = [[int(box[0]),int(box[1]), int(box[2]), int(box[3])]]
        img = io.imread(filename)
        pred = fa.get_landmarks_from_image(img,detected_faces=boxInt)
        pred = np.array(pred)
        if (pred.size != 1):
            curData = np.reshape(pred,(68,-1))  
        np.save(os.path.join(save_path, 'landmarks2D-' + '{0:04d}'.format(f) + '.npy'),curData)
