#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dlib
import sys
import os
import face_detection

# print("Dlib using cuda?")
# print(dlib.DLIB_USE_CUDA)


face_detector_path = sys.argv[1]
frames_path = sys.argv[2]
save_path = sys.argv[3]

#cnn_face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)


fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D)



print('Reading video frames...')

numImg = len([name for name in os.listdir(frames_path) if 
                  os.path.isfile(os.path.join(frames_path, name))])
from_first= True

with open(save_path, 'w+') as out:
    for f in range(1,numImg+1):
        number = '{0:04d}'.format(f)
        filename = os.path.join(frames_path, "frames" + number + ".jpg")       
        img = dlib.load_rgb_image(filename)
        dets = fa.get_detections_for_image(img)
        
        #dets = cnn_face_detector(img, 1)
        sortedDets = sorted(dets, key=lambda a: a[-1], reverse=True)
        if(len(dets) == 0):
            print('No faces detected. Using last detection result.')
            if from_first:
                print('Not detected on first frame, second frame will be used twice')
                continue
        else:
            d = sortedDets[0]
        if from_first:
            out.write('%d, %d, %d, %d\n' % (d[0], d[1],d[2], d[3]))
            out.write('%d, %d, %d, %d\n' % (d[0], d[1],d[2], d[3]))
            
        else:
            out.write('%d, %d, %d, %d\n' % (d[0], d[1],d[2], d[3]))
        from_first = False
        
        
        


#try the different face detectors here and see what happens
