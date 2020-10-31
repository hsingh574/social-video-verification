#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to run a CNN based face detector using dlib.  The
#   example loads a pretrained model and uses it to find faces in images.  The
#   CNN model is much more accurate than the HOG based model shown in the
#   face_detector.py example, but takes much more computational power to
#   run, and is meant to be executed on a GPU to attain reasonable speed.
#
#   You can download the pre-trained model from:
#       http://dlib.net/files/mmod_human_face_detector.dat.bz2
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy


import dlib
import os
from joblib import Parallel, delayed
import time
from joblib import wrap_non_picklable_objects

print("Dlib using cuda?")
print(dlib.DLIB_USE_CUDA)



face_detector_path = ("/home/socialvv/social-video-verification-v2/"
                      "social-video-verification/Data-Processing/"
                      "mmod_human_face_detector.dat")

baseDir = "/home/socialvv/socialvv"

def parallel_detection(cam, ID):
    
    #loading in model every time is an annoying slowdown, 
    #but I can't find a way to pickle it and I think this is an overall speedup
    cnn_face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
    
    
    frameDir = os.path.join(baseDir, f'ID{ID}',f'cam{cam}-wav2lip', 'frames')
    boundingBoxFile = os.path.join(baseDir, f'ID{ID}','bounding-boxes',f'cam{cam}-post-wav2lipv2-bounding-boxes.txt')
    #count number of frames in directory
    numImg = len([name for name in os.listdir(frameDir) if os.path.isfile(os.path.join(frameDir, name))])
    
    with open(boundingBoxFile, 'w+') as out:
        #temporary for testing
        for f in range(1, min(numImg + 1, 50)):
            number = '{0:04d}'.format(f)
            filename = os.path.join(frameDir, "frames" + number + ".jpg")
            img = dlib.load_rgb_image(filename)
            dets = cnn_face_detector(img, 1)
            sortedDets = sorted(dets, key=lambda a: a.confidence, reverse=True)
            if(len(dets) == 0):
                print('No faces detected. Using last detection result.')
            else:
                d = sortedDets[0]
            out.write('%d, %d, %d, %d\n' % (d.rect.left(), 
                                                d.rect.top(), 
                                                d.rect.right(), 
                                               d.rect.bottom()))
    

if __name__ == '__main__':
    
    numCams = 6
    numParticipants = 25
    exclude_list  = [17]
    #ids = [i for i in range(1, numParticipants+1) if i not in exclude_list]   
    ids = [1,2,3]
    
    n_jobs = -1
    
    start = time.time()
    Parallel(n_jobs=n_jobs)(delayed(parallel_detection)(cam,ID) for cam in range(1,numCams+1) for ID in ids)
    end = time.time()
    print('{:.4f} s'.format(end-start))