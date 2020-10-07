#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from scipy.cluster.hierarchy import linkage, fcluster
from pywt import wavedec2, dwt_max_level

def mahalanobis_calculate(data, num_pcs):
    pca = PCA(num_pcs)
    T = pca.fit_transform(data)
    # fit a Minimum Covariance Determinant (MCD) robust estimator to data 
    robust_cov = MinCovDet().fit(T)
    # Get the Mahalanobis distance
    m = robust_cov.mahalanobis(T)
    return m

# numFakes will be the number of
# fakes detected, and c will be a vector of numCams integers, which are 
# partitioned into two sets. We assume the larger partition is real.

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


def resultsHelper(numFakes, cLst, result):
    if numFakes[0] == 0:
        result[0][0] = 1
    
    if numFakes[1] == 1:
        if (np.all(cLst[0] == np.array([1,1,1,2,1,1])) or np.all(cLst[0] == np.array([2,2,2,1,2,2]))):
            result[0][1] = 1
            
    if numFakes[2] == 2:
        if (np.all(cLst[1] == np.array([1,1,2,2,1,1])) or np.all(cLst[1] == np.array([2,2,1,1,2,2]))):
            result[0][2] = 1
            
    if numFakes[3] == 3:
        if (np.all(cLst[2] == np.array([1,2,2,2,1,1])) or np.all(cLst[2] == np.array([2,1,1,1,2,2]))):
            result[0][3] = 1
    return result



def clusterHelper(X0, X1, X2, X3, thresh, result):
    #Test for tracking failures and remove
    badInds = []
    
    for i, row in enumerate(X0.T):
        if np.max(row) >= 10:
            badInds.append(i)
    
    X0 = np.delete(X0, badInds, axis = 1)
    X1 = np.delete(X1, badInds, axis = 1)
    X2 = np.delete(X2, badInds, axis = 1)
    X3 = np.delete(X3, badInds, axis = 1)
    
    #generate the linkage matrices with euclidean metric, we will cluster data ourselves
    link0 = linkage(X0)
    link1 = linkage(X1)
    link2 = linkage(X2)
    link3 = linkage(X3)
    
    numFakes0, _ = detectFakesTree(link0, thresh)
    numFakes1, c1 = detectFakesTree(link1, thresh)
    numFakes2, c2 = detectFakesTree(link2, thresh)
    numFakes3, c3 = detectFakesTree(link3, thresh)
    
    return resultsHelper([numFakes0, numFakes1, numFakes2, numFakes3], [c1,c2,c3], result)
    
    
def createDecompVector(coeff):
    output = []
    for i in range(len(coeff)):
        for j in range(len(coeff[i])):
            output.append(coeff[i][j].flatten())
    return np.hstack(output)

def socialVerificationNoPCA(data2,data3,data4, thresh):
    result = np.zeros((1,4))
    fullLen = min(data2['cam1'].shape[0], data3['cam1'].shape[0], data4['cam1'].shape[0])
    
    level = dwt_max_level(fullLen, 'haar')
    
    cam1_dist = createDecompVector(wavedec2(data2['cam1'][:fullLen,:], level = level, wavelet ='haar'))
    cam2_dist = createDecompVector(wavedec2(data2['cam2'][:fullLen,:], level = level, wavelet ='haar'))
    cam3_dist = createDecompVector(wavedec2(data2['cam3'][:fullLen,:], level = level, wavelet ='haar'))
    cam4_dist = createDecompVector(wavedec2(data2['cam4'][:fullLen,:], level = level, wavelet ='haar'))
    cam5_dist = createDecompVector(wavedec2(data2['cam5'][:fullLen,:], level = level, wavelet ='haar'))
    cam6_dist = createDecompVector(wavedec2(data2['cam6'][:fullLen,:], level = level, wavelet ='haar'))
    fake2_dist = createDecompVector(wavedec2(data2['fake'][:fullLen,:], level = level, wavelet ='haar'))
    fake3_dist = createDecompVector(wavedec2(data3['fake'][:fullLen,:], level = level, wavelet ='haar'))
    fake4_dist = createDecompVector(wavedec2(data4['fake'][:fullLen,:], level = level, wavelet ='haar'))
    
    X0 = np.array([cam1_dist, cam2_dist, cam3_dist, cam4_dist, cam5_dist, cam6_dist])
    X1 = np.array([cam1_dist, cam2_dist, cam3_dist, fake4_dist, cam5_dist, cam6_dist])
    X2 = np.array([cam1_dist, cam2_dist, fake3_dist, fake4_dist, cam5_dist, cam6_dist])
    X3 = np.array([cam1_dist, fake2_dist, fake3_dist, fake4_dist, cam5_dist, cam6_dist])
    
    return clusterHelper(X0, X1, X2, X3, thresh, result)
    




def socialVerificationOnlyPCA(data2,data3,data4, thresh, num_pcs):
    #storing the accuracy for the 0,1,2,3 fake cases
    result = np.zeros((1,4))
    fullLen = min(data2['cam1'].shape[0], data3['cam1'].shape[0], data4['cam1'].shape[0])
    
    cam1_dist = mahalanobis_calculate(data2['cam1'][:fullLen,:], num_pcs)
    cam2_dist = mahalanobis_calculate(data2['cam2'][:fullLen,:], num_pcs)
    cam3_dist = mahalanobis_calculate(data2['cam3'][:fullLen,:], num_pcs)
    cam4_dist = mahalanobis_calculate(data2['cam4'][:fullLen,:], num_pcs)
    cam5_dist = mahalanobis_calculate(data2['cam5'][:fullLen,:], num_pcs)
    cam6_dist = mahalanobis_calculate(data2['cam6'][:fullLen,:], num_pcs)
    fake2_dist = mahalanobis_calculate(data2['fake'][:fullLen,:], num_pcs)
    fake3_dist = mahalanobis_calculate(data3['fake'][:fullLen,:], num_pcs)
    fake4_dist = mahalanobis_calculate(data4['fake'][:fullLen,:], num_pcs)
    
    
    X0 = np.array([cam1_dist, cam2_dist, cam3_dist, cam4_dist, cam5_dist, cam6_dist])
    X1 = np.array([cam1_dist, cam2_dist, cam3_dist, fake4_dist, cam5_dist, cam6_dist])
    X2 = np.array([cam1_dist, cam2_dist, fake3_dist, fake4_dist, cam5_dist, cam6_dist])
    X3 = np.array([cam1_dist, fake2_dist, fake3_dist, fake4_dist, cam5_dist, cam6_dist])
    
    return clusterHelper(X0, X1, X2, X3, thresh, result)
    
    
    



def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')

    parser.add_argument('--data-dir', type=str, default='Data',
                    help='Directory where processed landmark files live')
    parser.add_argument('--num_pcs', type=int, default=5,
                    help='Number of principal components to use')
    parser.add_argument('--threshold', type=float, default=1.3,
                    help='Cluster threshold')
    parser.add_argument('--num_participants', type=int, default=25,
                    help='Number of participants')
    
    
    args = parser.parse_args()
    return args




def main():
    args = parse_args()
    
    #there is no data for ID 17
    if args.num_participants >= 17:
        averagePCA = np.zeros((args.num_participants-1, 4))
        averageSimple = np.zeros((args.num_participants-1, 4))
    else:
        averagePCA = np.zeros((args.num_participants, 4))
        averageSimple = np.zeros((args.num_participants, 4))
        
    
    for i in range(args.num_participants):
        
        #there is no data for ID 17
        if i == 16:
            continue
        
        
        data2 = loadmat(os.path.join(args.data_dir, f'mouth-data-fake2-ID{i+1}.mat'))
        data3 = loadmat(os.path.join(args.data_dir, f'mouth-data-fake3-ID{i+1}.mat'))
        data4 = loadmat(os.path.join(args.data_dir, f'mouth-data-fake4-ID{i+1}.mat'))
        
        resultPCA = socialVerificationOnlyPCA(data2,data3,data4,args.threshold, args.num_pcs)
        
        resultSimple = socialVerificationNoPCA(data2,data3,data4, args.threshold)
        
        print(f'Iteration: {i+1}. PCA Result: {resultPCA}')
        print(f'Iteration: {i+1}. SimpleMethod Result: {resultSimple}')
        
        
        if i > 16:
            averagePCA[i-1] = resultPCA
            averageSimple[i-1] = resultSimple
            
        else:
            averagePCA[i] = resultPCA
            averageSimple[i] = resultSimple
        
    print(f'Average accuracy PCA: {np.mean(averagePCA, axis = 0)}')
    print(f'Average accuracy No PCA: {np.mean(averageSimple, axis = 0)}')



if __name__ == "__main__":
    main()
    