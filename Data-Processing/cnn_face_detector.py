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

'''
This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
These objects can be accessed by simply iterating over the mmod_rectangles object
The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.
    
It is also possible to pass a list of images to the detector.
- like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

In this case it will return a mmod_rectangless object.
This object behaves just like a list of lists and can be iterated over.
'''

import sys
import dlib
import os
from joblib import Parallel, delayed
from multiprocessing import Process, JoinableQueue
import time

print("Dlib using cuda?")
print(dlib.DLIB_USE_CUDA)

cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.argv[1])
numImg = len([name for name in os.listdir(sys.argv[2]) if os.path.isfile(os.path.join(sys.argv[2], name))])
padding = 100


def helper(d):
    with open(sys.argv[3] + "bounding-boxes.txt","w+") as out:
        while True:
            val = q.get()
            if val is None:
                break
            out.write(val)
        q.task_done()
        # Finish up
        q.task_done()


def processor(f):
    number = '{0:04d}'.format(f)
    filename = sys.argv[2] + "frames" + number + ".jpg"
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(filename)
    dets = cnn_face_detector(img, 1)
    h, w = img.shape[:2]

    #print("Number of faces detected: {}".format(len(dets)))
    sortedDets = sorted(dets, key=lambda a: a.confidence, reverse=True)
    
    # Only keep most confident face-- we only expect one face per frame
    if(len(dets) == 0):
        print('No faces detected. Using last detection result.')
    else:
        d = sortedDets[0]
        
    #put the rectangles in the q
    to_put = '%d, %d, %d, %d\n' % (d.rect.left(), 
                                            d.rect.top(), 
                                            d.rect.right(), 
                                           d.rect.bottom())
    q.put(to_put)
    

start = time.time()
q = JoinableQueue()
p = Process(target=helper, args=(q,))
p.start()
Parallel(n_jobs=-1, verbose=0)(delayed(processor)(f) for f in range(1,numImg + 1))
q.put(None) # Poison pill
p.join()
end = time.time()
print('{:.4f} s'.format(end-start)) 
