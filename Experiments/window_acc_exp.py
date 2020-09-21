#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet
from scipy.cluster.hierarchy import linkage, fcluster


def detectFakesTree(link, thresh):
    ratio = link[-1][-2] / link[-2][-2]
    if ratio > thresh:
        c = fcluster(link, 2,criterion='maxclust')
        partition1 = len(np.argwhere(c==1))
        partition2 = len(np.argwhere(c==2))
        if (partition1 > partition2):
            numFakes = partition2
        else:
            numFakes = partition1
    else:
        numFakes = 0
        c = 0
    return numFakes, c

def mahalanobis_calculate(data, num_pcs):
    pca = PCA(num_pcs)
    T = pca.fit_transform(StandardScaler().fit_transform(data))
    # fit a Minimum Covariance Determinant (MCD) robust estimator to data 
    robust_cov = MinCovDet().fit(T)
    # Get the Mahalanobis distance
    m = robust_cov.mahalanobis(T)
    return m


def onlyPCA(cam1, cam2, cam3, cam4, cam5, cam6, fake2, 
            fake3, fake4, start, end, num_pcs, thresh):
    
    cam1Out = mahalanobis_calculate(cam1[start:end,:], num_pcs)
    cam2Out = mahalanobis_calculate(cam2[start:end,:], num_pcs)
    cam3Out = mahalanobis_calculate(cam3[start:end,:], num_pcs)
    cam4Out = mahalanobis_calculate(cam4[start:end,:], num_pcs)
    cam5Out = mahalanobis_calculate(cam5[start:end,:], num_pcs)
    cam6Out = mahalanobis_calculate(cam6[start:end,:], num_pcs)
    
    camFake1 = mahalanobis_calculate(fake2[start:end,:], num_pcs)
    camFake2 = mahalanobis_calculate(fake3[start:end,:], num_pcs)
    camFake3 = mahalanobis_calculate(fake4[start:end,:], num_pcs)
    
    X0 = np.array([cam1Out, cam2Out, cam3Out, cam4Out, cam5Out, cam6Out])
    X1 = np.array([cam1Out, cam2Out, cam3Out, camFake3, cam5Out, cam6Out])
    X2 = np.array([cam1Out, cam2Out, camFake2, camFake3, cam5Out, cam6Out])
    X3 = np.array([cam1Out, camFake1, camFake2, camFake3, cam5Out, cam6Out])
    
    #Test for tracking failures and remove
    badInds = []
    
    for i, row in enumerate(X0.T):
        if np.max(row) >= 10:
            badInds.append(i)
    
    X0 = np.delete(X0, badInds, axis = 1)
    X1 = np.delete(X1, badInds, axis = 1)
    X2 = np.delete(X2, badInds, axis = 1)
    X3 = np.delete(X3, badInds, axis = 1)
    
    link0 = linkage(X0)
    link1 = linkage(X1)
    link2 = linkage(X2)
    link3 = linkage(X3)
    
    numFakes0, _ = detectFakesTree(link0, thresh)
    numFakes1, c1 = detectFakesTree(link1, thresh)
    numFakes2, c2 = detectFakesTree(link2, thresh)
    numFakes3, c3 = detectFakesTree(link3, thresh)
    
    return numFakes0, numFakes1, numFakes2, numFakes3, c1, c2, c3
    
    
    


def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')

    parser.add_argument('--data-dir', type=str, default='Data',
                    help='Directory where processed landmark files live')
    parser.add_argument('--num_pcs', type=int, default=5,
                    help='Number of principal components to use')
    parser.add_argument('--num_participants', type=int, default=1,
                    help='Number of participants')
    
    
    args = parser.parse_args()
    return args




def main():
    args = parse_args()
        
    
    for i in range(args.num_participants):
        if i == 16:
            continue
        
        
        data2 = loadmat(os.path.join(args.data_dir, f'mouth-data-fake2-ID{i+1}.mat'))
        data3 = loadmat(os.path.join(args.data_dir, f'mouth-data-fake3-ID{i+1}.mat'))
        data4 = loadmat(os.path.join(args.data_dir, f'mouth-data-fake4-ID{i+1}.mat'))
        
        fullLen = min(data2['cam1'].shape[0], data3['cam1'].shape[0], data4['cam1'].shape[0])
        
        
        cam1 = data3['cam1'][:fullLen,:]
        cam2 = data3['cam2'][:fullLen,:]
        cam3 = data3['cam3'][:fullLen,:]
        cam4 = data3['cam4'][:fullLen,:]
        cam5 = data3['cam5'][:fullLen,:]
        cam6 = data3['cam6'][:fullLen,:]
        
        #split into two thirds (fake, real, fake)
        intervalWin = fullLen // 3
        fake2 = np.vstack([data2['fake'][:intervalWin,:], 
                           cam2[intervalWin:(2*intervalWin),:], 
                           data2['fake'][(2*intervalWin):fullLen,:]])
        fake3 = np.vstack([data3['fake'][:intervalWin,:], 
                           cam3[intervalWin:(2*intervalWin),:], 
                           data3['fake'][(2*intervalWin):fullLen,:]])
        fake4 = np.vstack([data4['fake'][:intervalWin,:], 
                           cam4[intervalWin:(2*intervalWin),:], 
                           data4['fake'][(2*intervalWin):fullLen,:]])
    
        #calculate L2 between real/fake mouth landmarks for all frames
        #baseline = np.linalg.norm(cam4 - fake4, axis = 1)
    
        #Iterate over thresholds too
        #threshes = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
        #window_sizes = [50,150,250,350]
        
        threshes = [1.3]
        window_sizes = [1000]
        
        
# =============================================================================
#         (1) TP if window contains a faked frame & fake is detected
#         (2) TN if window does not have fake & fake is not detected
#         (3) FP if window does not have fake & fake is detected
#         (4) FN if window contains a faked frame & fake is not detected
# =============================================================================
    
        for t in threshes:
            for j in window_sizes:
                numWin = fullLen - j
                acc0 = acc1 = acc2 = acc3 = np.zeros((4,numWin))
                for start in range(fullLen):
                    end = start + j
                    if end > fullLen-1:
                        continue
                    
                    numFakes0, numFakes1, numFakes2, numFakes3, c1, c2, c3 = onlyPCA(cam1, cam2, cam3, cam4, cam5, cam6, fake2, 
            fake3, fake4, start, end, args.num_pcs, t)
                    
                    if len(set(range(start, end)).intersection(set(range(intervalWin, 2*intervalWin)))) == 0:
                        isFake = 1
                    else:
                        isFake = 0
                        
                    #0 fakes case
                    if numFakes0 ==0:
                        acc0[1][start] = 1 
                    else:
                        acc0[2][start] = 1
                        
                        
                    #1 fake case
                    if numFakes1 ==1:
                        if isFake==0:
                            acc1[2][start] = 1
                        else:
                            if (np.all(c1 == np.array([1,1,1,2,1,1])) or np.all(c1 == np.array([2,2,2,1,2,2]))):
                                acc1[0][start] = 1
                            else:
                                acc1[3][start] = 1
                    elif numFakes1 > 1:
                        acc1[2][start] = 1
                    else:
                        if isFake == 0:
                            acc1[1][start] = 1
                        else:
                            acc1[3][start] = 1
                            
                    #2 fakes case
                    if numFakes2 ==2:
                        if isFake==0:
                            acc2[2][start] = 1
                        else:
                            if (np.all(c2 == np.array([1,1,2,2,1,1])) or np.all(c2 == np.array([2,2,1,1,2,2]))):
                                acc2[0][start] = 1
                            else:
                                acc2[3][start] = 1
                    elif ((numFakes2 == 1) or (numFakes2 > 2)):
                        acc2[2][start] = 1
                    else:
                        if isFake == 0:
                            acc2[1][start] = 1
                        else:
                            acc2[3][start] = 1
                            
                    #3 fakes case
                    
                    if numFakes3 ==3:
                        if isFake==0:
                            acc3[2][start] = 1
                        else:
                            if (np.all(c3 == np.array([1,2,2,2,1,1])) or np.all(c3 == np.array([2,1,1,1,2,2]))):
                                acc3[0][start] = 1
                            else:
                                acc3[3][start] = 1
                    elif ((numFakes3 == 1) or (numFakes3 == 2) or (numFakes3 > 3)):
                        acc3[2][start] = 1
                    else:
                        if isFake == 0:
                            acc3[1][start] = 1
                        else:
                            acc3[3][start] = 1
                            
                    print(f'Window Start: {start}')
                            
                    
                print(f'ID: {i}. Threshold: {t}. Window size: {j}.'
                          f'Acc0: {np.mean(acc0, axis = 1)}. Acc1: {np.mean(acc1, axis = 1)}.'
                          f'Acc2: {np.mean(acc2, axis = 1)}. Acc3: {np.mean(acc3, axis = 1)}.')
                    
                    
            
                    
                    
                    


if __name__ == "__main__":
    main()
    

