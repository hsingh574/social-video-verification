# From: https://github.com/1adrianb/face-alignment

import face_alignment
from skimage import io
import numpy as np
import os
import time
from joblib import Parallel, delayed
import shutil

baseDir = "/home/socialvv/socialvv"

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)


def delete_helper(cam, ID):
    delDir = os.path.join(baseDir,f'ID{ID}',f'cam{cam}-wav2lip', 'landmarks')
    outDir = os.path.join(baseDir,f'ID{ID}',f'cam{cam}-wav2lip', 'landmarksv2')
    
    if os.path.isdir(delDir):
        shutil.rmtree(delDir)
    
    if os.path.isdir(outDir):
        shutil.rmtree(outDir)
    
    
    


def parallel_generation(cam,ID):
    frameDir = os.path.join(baseDir, f'ID{ID}',f'cam{cam}-wav2lip', 'frames')
    boundingBoxFile = os.path.join(baseDir, f'ID{ID}','bounding-boxes',
                                   f'cam{cam}-post-wav2lipv2-bounding-boxes.txt')
    
    delDir = os.path.join(baseDir,f'ID{ID}',f'cam{cam}-wav2lip', 'landmarks')
    outDir = os.path.join(baseDir,f'ID{ID}',f'cam{cam}-wav2lip', 'landmarksv2')
    
    print(outDir)
    
    
    if os.path.isdir(delDir):
        shutil.rmtree(delDir)
    
    if os.path.isdir(outDir):
        shutil.rmtree(outDir)
    
        
    
    if not(os.path.isdir(outDir)):
        os.makedirs(outDir)
    #count number of frames in directory
    numImg = len([name for name in os.listdir(frameDir) if 
                  os.path.isfile(os.path.join(frameDir, name))])
    with open(boundingBoxFile, 'r') as out:
        for f in range(1,numImg+1):
            number = '{0:04d}'.format(f)
            filename = os.path.join(frameDir, "frames" + number + ".jpg")
            box = out.readline().split(',')
            if box[0]=='':
                continue
            boxInt = [[int(box[0]),int(box[1]), int(box[2]), int(box[3])]]
            img = io.imread(filename)
            pred = fa.get_landmarks_from_image(img,detected_faces=boxInt)
            pred = np.array(pred)
            if (pred.size != 1):
                curData = np.reshape(pred,(68,-1))  
            np.save(os.path.join(outDir, 'landmarks2D-' + '{0:04d}'.format(f) + '.npy'),curData)
                
                
    



if __name__ == '__main__':
    
    numCams = 6
    numParticipants = 25
    exclude_list  = [17]
    ids = [i for i in range(6, numParticipants+1) if i not in exclude_list]   
    
    #n_jobs = -1
    
    start = time.time()
    
    
    for ID in ids:
        for cam in range(1, numCams+1):
            delete_helper(cam, ID)
    
    
    for ID in ids:
        for cam in range(1, numCams+1):
            parallel_generation(cam, ID)
            
    
    #Parallel(n_jobs=n_jobs)(delayed(parallel_generation)(cam,ID) for cam in range(1,numCams+1) for ID in ids)
    end = time.time()
    print('{:.4f} s'.format(end-start))
