import os
import numpy as np
import scipy.io as sio

def landmark2mat(inPathReal, inPathFake,numCams,outPath,numF,fakeCam,shift):
# Prep landmark data for analysis in Matlab
# Lip points are: 49-68

    numLip = 20
    dataMat = np.zeros((numF,numLip*2,numCams))

    # Calculate data matrix of landmarks for each camera
    for i in range(1,numCams+1):
        
        means = np.zeros((numLip*2,numF))

        # Get feature location per frame in image space
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
    sio.savemat(outPath,{'cam1':dataMat[:,:,0],'cam2':dataMat[:,:,1],'cam3':dataMat[:,:,2],'cam4':dataMat[:,:,3],
                         'cam5':dataMat[:,:,4],'cam6': dataMat[:,:,5],'fake':dataMat[:,:,6]})

if __name__ == '__main__':

    fakeCam = 4          
    numCams = 7 # six real cameras + one fake
    numParticipants = 25
    exclude_list  = [14,23]
    ids = [i for i in range(1, numParticipants+1) if i not in exclude_list]
    
    inPathBase = "media/eleanor/New-Volume/deepfake-data/Dataset"
    outPathBase = "/media/eleanor/New-Volume/socialvv"
    
    if not(os.path.isdir(os.path.join(outPathBase, "wav2lip_landmarks"))):
        os.makedirs(os.path.join(outPathBase, "wav2lip_landmarks"))
        
    landmarkBase = os.path.join(outPathBase, "wav2lip_landmarks")
    
    
    for ID in ids:
        inPathReal = os.path.join(inPathBase, f"ID{ID}")
        inPathFake = os.path.join(outPathBase, f"ID{ID}", f"cam{fakeCam}-wav2lip", "landmarks")
        outPath = os.path.join(landmarkBase, f"mouth-data-fake{fakeCam}-ID{ID}.mat")
        print(f"Saving output to: {outPath}")
    
        # LipGAN's output is shifted from the input by five frames
        shift = 0        
    
        lengthLst = [(len(os.listdir(os.path.join(inPathReal, f'cam{i}-landmarks'))) - shift) for i in range(1,7)]
        lengthLst.append(len(os.listdir(inPathFake)))
        numF = min(lengthLst)

        landmark2mat(inPathReal, inPathFake, numCams,outPath,numF,fakeCam, shift)
