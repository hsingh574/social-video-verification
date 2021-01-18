#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io as sio
import sys

def landmark2mat(inPathReal, inPathFake,numCams,outPath,numF,shift):
# Prep landmark data for analysis in Matlab
# Lip points are: 49-68

    numLip = 20
    dataMat = np.zeros((numF,numLip*2,numCams+1))

    # Calculate data matrix of landmarks for each camera
    for i in range(0,numCams+1):
        
        means = np.zeros((numLip*2,numF))

        # Get feature location per frame in image space
        # can start from a specific frame number in order to make this faster
        
        for f in range(1,numF+1):
            # Load data
            # format: x1,y1, x2,y2, ... xn,yn, for lip points 1,...,n
            if (i == numCams):
                data = np.load(os.path.join(inPathFake, "landmarks2D-" + "{0:04d}".format(f) + ".npy"), allow_pickle = True)
            else:
                data = np.load(os.path.join(inPathReal, f"cam{i}-landmarks", "landmarks2D-" + 
                                            "{0:04d}".format(f+shift) + ".npy"), allow_pickle = True)
                    
            means[:,f-1] = np.squeeze(np.reshape(data[-numLip:,:],(-1,1)))

        # Need it to be in format where observations are rows, vars are cols        
        camData = means.T

        # Normalize data for camera i, which 0-indexed is i-1
        camMean = np.mean(camData,axis=0)
        camStd = np.std(camData,axis=0)
        
        dataMat[:,:,i-1] = (camData - camMean) / camStd

    # Save data matrix as mat
    sio.savemat(outPath,{'cam0':dataMat[:,:,0],'cam1':dataMat[:,:,1],'cam2':dataMat[:,:,2],'cam3':dataMat[:,:,3],
                         'cam4':dataMat[:,:,4],'cam5': dataMat[:,:,5],'cam6': dataMat[:,:,6],'fake':dataMat[:,:,7]})
    
    
    
inPathReal = sys.argv[1]
inPathFake = sys.argv[2]
outPath = sys.argv[3]

shift = 0
numCams = 7

lengthLst = [(len(os.listdir(os.path.join(inPathReal, f'cam{i}-landmarks'))) - shift) for i in range(0,numCams)]
lengthLst.append(len(os.listdir(inPathFake)))
numF = min(lengthLst)
landmark2mat(inPathReal, inPathFake, numCams,outPath,numF, shift)
print(f"Done! Saved output to: {outPath}")
