#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import argparse
import os
import re
import numpy as np
from sklearn.decomposition import PCA
from scipy.io import loadmat, savemat
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict, Counter

from joblib import Parallel, delayed

from matplotlib import pyplot as plt


zerodist = []
onedist = []
twodist = []
threedist = []

        
def l2_calculate(data, upper_lip_start, lower_lip_start, num_points):
    
    length = len(data)
    upper_lip = data[:, upper_lip_start:(upper_lip_start+num_points)]
    lower_lip = data[:, lower_lip_start:(lower_lip_start+num_points)]
    
    #upper_lip = np.reshape(upper_lip, (length, -1, 2))
    #lower_lip = np.reshape(lower_lip, (length, -1, 2))
    
    return np.linalg.norm(upper_lip - lower_lip, axis = -1)


def mahalanobis(T, eigenval):
    
    cov = np.linalg.pinv(np.diag(eigenval.T))
    return np.sqrt(np.sum(np.multiply(np.matmul(T, cov), T), axis=1)) #T @ cov, T), axis = 1))


def mahalanobis_calculate(data, num_pcs):
    pca = PCA(num_pcs)
    T = pca.fit_transform(data)
    eigenval = pca.explained_variance_
    return mahalanobis(T, eigenval)

#for a cam, sum the L2 distances across all other cams in the list
def L2_sum(cams, index):
    print("len cams: ", len(cams))
    print("index: ", index)
    curr_cam = cams[index]
    other_cams = cams.copy()
    other_cams.pop(index)
    sum = 0
    for cam in cams:
        sum += np.linalg.norm(cam - curr_cam)
    return sum

#determine L2 difference between clusters
def linkage_L2(linkage_matrix):
    print("Linkage matrix: ")
    print(linkage_matrix)

#previous version: This takes longer and 
#returns a different value from the matlab version   
# =============================================================================
#     T = pca.fit_transform(StandardScaler(with_std=False).fit_transform(data))
#     # fit a Minimum Covariance Determinant (MCD) robust estimator to data 
#     robust_cov = MinCovDet().fit(T)
#     # Get the Mahalanobis distance
#     m = robust_cov.mahalanobis(T)
#     return m
# =============================================================================

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



def cluster_helper(X0, X1, X2, X3, thresh):
    
    #Test for tracking failures and remove
    #delete the columns which have an element greater than 10
    badInds = []
    for i, row in enumerate(X0.T):
        if np.max(row) >= 10:
            badInds.append(i)
    
    X0 = np.delete(X0, badInds, axis = 1)
    X1 = np.delete(X1, badInds, axis = 1)
    X2 = np.delete(X2, badInds, axis = 1)
    X3 = np.delete(X3, badInds, axis = 1)

    # print("dims X0: ", X0.shape)
    
    link0 = linkage(X0)
    link1 = linkage(X1)
    link2 = linkage(X2)
    link3 = linkage(X3)

    # print("0:", link0)
    # print("0 one elt:", link0[-1, 2])

    zerodist.append(link0[-1, 2])
    onedist.append(link1[-1, 2])
    twodist.append(link2[-1, 2])
    threedist.append(link3[-1, 2])

    # print("1:", link1[-1, 2])
    # print("2:", link2[-1, 2])
    # print("3:", link3[-1, 2])

    # linkage_L2(link1)
    
    numFakes0, _ = detectFakesTree(link0, thresh)
    numFakes1, c1 = detectFakesTree(link1, thresh)
    numFakes2, c2 = detectFakesTree(link2, thresh)
    numFakes3, c3 = detectFakesTree(link3, thresh)
    
    return numFakes0, numFakes1, numFakes2, numFakes3, c1, c2, c3

def build_test_arrays(camsOut, fake0Out, fake1Out, fake2Out):
    #X0 is no fakes, X1 is 1 fake, etc.
    X0 = np.array(camsOut)
    temp = X0.copy()
    temp[3] = fake0Out
    X1 = temp
    temp = X1.copy()
    temp[2] = fake1Out
    X2 = temp
    temp = X2.copy()
    temp[1] = fake2Out
    X3 = temp
    
    return X0, X1, X2, X3
    
    
# =============================================================================
# Take the l2 distance between the
# upper and lower lip for each frame of each camera 
# and cluster (as described in Tursman et al. 2020)
# =============================================================================
def onlyL2(cams, fake0, fake1, fake2, start, end, num_pcs, thresh):
    
    #Can tweak these later to check performance on other facial landmarks
    upper_lip_start = 0
    lower_lip_start = 14
    num_points = 4
    
    camsOut = []
    for c in cams:
        camsOut.append(l2_calculate(c[start:end,:], upper_lip_start, lower_lip_start, num_points))
    
    fake0Out = l2_calculate(fake0[start:end,:], upper_lip_start, lower_lip_start, num_points)
    fake1Out = l2_calculate(fake1[start:end,:], upper_lip_start, lower_lip_start, num_points)
    fake2Out = l2_calculate(fake2[start:end,:], upper_lip_start, lower_lip_start, num_points)
    
    X0, X1, X2, X3 = build_test_arrays(camsOut, fake0Out, fake1Out, fake2Out)
    
    return cluster_helper(X0, X1, X2, X3, thresh)

def weighted_SH_coords_sum(coords):
    weights = np.array([9,8,7,6,5,4,3,2,1])
    return np.sum(coords * np.power(10, weights), axis = 1)

## for debugging lighting results: don't use PCA/mahalanobis for clustering. Trying difference methods such as summing SH coords
def noPCA(cams, fake0, fake1, fake2, start, end, num_pcs, thresh):
    
    camsOut = []
    # camsOutPCA = []
    allCamsTrim = []

    for c in cams:
        # camsOut.append(weighted_SH_coords_sum(c))
        # camsOut.append(c[start:end, 0])
        # camsOut.append(c[start:end, 5])

        # camsOut.append(c[start:end,:])
        # camsOutPCA.append(mahalanobis_calculate(c[start:end,:], num_pcs))
        allCamsTrim.append(c[start:end, :])
    allCamsTrim.append(fake0[start:end, :])
    allCamsTrim.append(fake1[start:end, :])
    allCamsTrim.append(fake2[start:end, :])

    cam0_norm = L2_sum(allCamsTrim, 0)
    cam1_norm = L2_sum(allCamsTrim, 1)
    cam2_norm = L2_sum(allCamsTrim, 2)
    cam3_norm = L2_sum(allCamsTrim, 3)
    cam4_norm = L2_sum(allCamsTrim, 4)
    cam5_norm = L2_sum(allCamsTrim, 5)
    fake0_norm = L2_sum(allCamsTrim, 6)

    print("got norm!", cam0_norm)

    plt.title("cams plotted with L2 distance to all others, red is fake")
    plt.plot(cam0_norm, 'tab:blue')
    plt.plot(cam1_norm, 'tab:orange')
    plt.plot(cam2_norm, 'tab:green')
    plt.plot(cam3_norm, 'tab:purple')
    plt.plot(cam4_norm, 'tab:brown')
    plt.plot(cam5_norm, 'tab:pink')
    plt.plot(fake0_norm, 'tab:red')
    plt.show()
    

    # cam0PCA = mahalanobis_calculate(cams[0][start:end,:], num_pcs)
    # cam1PCA = mahalanobis_calculate(cams[1][start:end,:], num_pcs)
    # cam2PCA = mahalanobis_calculate(cams[2][start:end,:], num_pcs)
    # cam3PCA = mahalanobis_calculate(cams[3][start:end,:], num_pcs)
    # cam4PCA = mahalanobis_calculate(cams[4][start:end,:], num_pcs)
    # cam5PCA = mahalanobis_calculate(cams[5][start:end,:], num_pcs)
    # fake0PCA = mahalanobis_calculate(fake0[start:end,:], num_pcs)

    # cam0_weighted = weighted_SH_coords_sum(cams[0][start:end,:])
    # cam1_weighted = weighted_SH_coords_sum(cams[1][start:end,:])
    # cam2_weighted = weighted_SH_coords_sum(cams[2][start:end,:])
    # cam3_weighted = weighted_SH_coords_sum(cams[3][start:end,:])
    # cam4_weighted = weighted_SH_coords_sum(cams[4][start:end,:])
    # cam5_weighted = weighted_SH_coords_sum(cams[5][start:end,:])
    # fake_weighted = weighted_SH_coords_sum(fake0[start:end,:])

    # plt.title("cams plotted with weighted_sum distance, red is fake")
    # plt.plot(cam0_weighted, 'tab:blue')
    # plt.plot(cam1_weighted, 'tab:orange')
    # plt.plot(cam2_weighted, 'tab:green')
    # plt.plot(cam3_weighted, 'tab:purple')
    # plt.plot(cam4_weighted, 'tab:brown')
    # plt.plot(cam5_weighted, 'tab:pink')
    # plt.plot(fake_weighted, 'tab:red')
    # plt.show()

    # fake1Out = mahalanobis_calculate(fake1[start:end,:], num_pcs)
    # fake2Out = mahalanobis_calculate(fake2[start:end,:], num_pcs)


    # fake0Out = fake0[start:end,0]
    # fake1Out = fake1[start:end,0]
    # fake2Out = fake2[start:end,0]

    # fake0Out = np.sum(fake0[start:end], axis = 1)
    # fake1Out = np.sum(fake1[start:end], axis = 1)
    # fake2Out = np.sum(fake2[start:end], axis = 1)
    # fake0Out = weighted_SH_coords_sum(fake0)
    # fake1Out = weighted_SH_coords_sum(fake1)
    # fake2Out = weighted_SH_coords_sum(fake2)

    # plt.title("cams plotted with PCA mahalanobis distance, red is fake")
    # plt.plot(cam0PCA, 'tab:blue')
    # plt.plot(cam1PCA, 'tab:orange')
    # plt.plot(cam2PCA, 'tab:green')
    # plt.plot(cam3PCA, 'tab:purple')
    # plt.plot(cam4PCA, 'tab:brown')
    # plt.plot(cam5PCA, 'tab:pink')
    # plt.plot(fake0PCA, 'tab:red')
    # plt.show()

    # cam0out = np.linalg.norm(cams[0][start:end, :], axis = 1)
    # cam1out = np.linalg.norm(cams[1][start:end, :], axis = 1)
    # cam2out = np.linalg.norm(cams[2][start:end, :], axis = 1)
    # cam3out = np.linalg.norm(cams[3][start:end, :], axis = 1)
    # cam4out = np.linalg.norm(cams[4][start:end, :], axis = 1)
    # cam5out = np.linalg.norm(cams[5][start:end, :], axis = 1)

    # fake0Out = np.linalg.norm(fake0[start:end, :], axis = 1)
    # fake1Out = np.linalg.norm(fake1[start:end, :], axis = 1)
    # fake2Out = np.linalg.norm(fake2[start:end, :], axis = 1)

    # plt.title("cams plotted in no PCA, only sh coord 5, red is fake")
    # plt.plot(cams[0][start:end, 5], 'tab:blue')
    # plt.plot(cams[1][start:end, 5], 'tab:orange')
    # plt.plot(cams[2][start:end, 5], 'tab:green')
    # plt.plot(cams[3][start:end, 5], 'tab:purple')
    # plt.plot(cams[4][start:end, 5], 'tab:brown')
    # plt.plot(cams[5][start:end, 5], 'tab:pink')
    # plt.plot(fake1[start:end, 5], 'tab:red')
    # plt.show()


    # cam0PreNorm = cams[5][start:end, :]
    # cam1PreNorm = cams[1][start:end, :]
    # fake0PreNorm = fake2[start:end, :]

    # print("cam0 dims", cam0PreNorm.shape)
    # print("fake0 dims", fake0PreNorm.shape)

    # print("cams pre norm diff", np.mean(cam0PreNorm - cam1PreNorm))
    # print("fake0 vs. cam pre norm diff", np.mean(cam0PreNorm - fake0PreNorm))
    # print("fake1 vs. cam diff", np.mean(camsOut[0] - fake1Out))

    # print("fake pre norm dims", fake2[start:end].shape)
    # print("fake dims", fake2Out.shape)

    # print("PCA dims: ", fake0OutPCA.shape)
    # print("landmark dims: ", fake0Out.shape)

    # print("cam0pca dims: ", camsOutPCA[0].shape)
    # print("cam0 dims: ", camsOut[0].shape)

    # fake0Out = fake0[start:end, 5]
    # fake1Out = fake1[start:end, 5]
    # fake2Out = fake2[start:end, 5]
    
    X0, X1, X2, X3 = build_test_arrays(camsOut, fake0Out, fake1Out, fake2Out)
    
    return cluster_helper(X0, X1, X2, X3, thresh)

def onlyPCA(cams, fake0, fake1, fake2, start, end, num_pcs, thresh):
    
    camsOut = []
    for c in cams:
        camsOut.append(mahalanobis_calculate(c[start:end,:], num_pcs))
    
    fake0Out = mahalanobis_calculate(fake0[start:end,:], num_pcs)
    fake1Out = mahalanobis_calculate(fake1[start:end,:], num_pcs)
    fake2Out = mahalanobis_calculate(fake2[start:end,:], num_pcs)
    
    X0, X1, X2, X3 = build_test_arrays(camsOut, fake0Out, fake1Out, fake2Out)
    
    return cluster_helper(X0, X1, X2, X3, thresh)

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')

    parser.add_argument('--data-dir', type=str, default='Data',
                    help='Directory where processed landmark files live')
    parser.add_argument('--num_pcs', type=int, default=5,
                    help='Number of principal components to use')
    parser.add_argument('--save-dir', type=str, default='results_v2',
                    help='Directory to save results')
    parser.add_argument('--zero-start', action='store_true',
                    help='Whether or not there is a cam0')
    parser.add_argument("--num-cams", type=int, default=6)
    # parser.add_argument("--thresholds", nargs="+", default=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    # parser.add_argument("--thresholds", nargs="+", default=[1.3, 1.5, 1.7, 1.9, 2.1])
    parser.add_argument("--thresholds", nargs="+", default=[2.1, 2.3, 2.5, 2.7, 2.9])
    # parser.add_argument("--thresholds", nargs="+", default=[7.0, 7.2, 7.4, 7.6, 7.8])



    parser.add_argument("--window-sizes", nargs="+", default=[50,100,150,200,250,300])
    # parser.add_argument("--window-sizes", nargs="+", default=[300])

    parser.add_argument("--num-jobs", type=int, default=-1)
    
    
    
    args = parser.parse_args()
    return args


def calculate_acc_helper(option1, option2, numFakes, c, correctAnswer, isFake):
    acc = np.zeros((4,))
    
    if numFakes == correctAnswer:
        if isFake==0:
            acc[2] = 1  #FP, detected some number of fakes where there was none
            # print("FP!")
        else:
            if (np.all(c == option1) or np.all(c == option2)):
                acc[0] = 1 #TP, detected some number of fakes where there was some
                # print("TP!")
            else:
                acc[3] = 1 #FN, failed to detect some number of fakes where there was some
    elif not(numFakes == 0):
        acc[2] = 1 #FP deteected some number of fakes, but was wrong number of fakes
        # print("FP!")
    else:
        if isFake == 0:
            acc[1] = 1 #TN, correctly got no fakes 
        else:
            acc[3] = 1 #FN incorrectly got no fakes
    return acc



def split_procedure(data0, data1, data2, real_cam0, real_cam1, 
                    real_cam2, alternative, fullLen, intervalWin):
    
    #split into two thirds (fake, real, fake)
    if not alternative:
        
        
        fake0 = np.vstack([data0['fake'][:intervalWin,:], 
                           real_cam0[intervalWin:(2*intervalWin),:], 
                           data0['fake'][(2*intervalWin):fullLen,:]])
    
        fake1 = np.vstack([data1['fake'][:intervalWin,:], 
                           real_cam1[intervalWin:(2*intervalWin),:], 
                           data1['fake'][(2*intervalWin):fullLen,:]])
    
        fake2 = np.vstack([data2['fake'][:intervalWin,:], 
                           real_cam2[intervalWin:(2*intervalWin),:], 
                           data2['fake'][(2*intervalWin):fullLen,:]])
    else:
        #alternative procedure: keep the fake as a fake throughout all frames
        fake0 = data0['fake'][:fullLen,:]
        fake1 = data1['fake'][:fullLen,:]
        fake2 = data2['fake'][:fullLen,:]
        
    return fake0, fake1, fake2
    
    
#Note: zero_start is whether or not there is a cam0    

def get_cams(data, num_cams, zero_start, fullLen):
    
    cams_list = []
    if zero_start:
        for i in range(num_cams+1):
            cams_list.append(data['cam{}'.format(i)][:fullLen,:])
    else:
        for i in range(1, num_cams+1):
            cams_list.append(data['cam{}'.format(i)][:fullLen,:])
    
    return cams_list



#Note: fake_cams is 0-indexed

#Note: first fake at index 3, second at index 2, third at index 1


def gen_results(i, fake_cams, num_cams, zero_start, data_dir, 
                alternative, threshes, window_sizes, num_pcs, save_dir):
    #print("Processing ID", str(i))
    
    
    data0 = loadmat(os.path.join(data_dir, "fake{}-ID{}.mat".format(fake_cams[0], i)))
    data1 = loadmat(os.path.join(data_dir, "fake{}-ID{}.mat".format(fake_cams[1], i)))
    data2 = loadmat(os.path.join(data_dir, "fake{}-ID{}.mat".format(fake_cams[2], i)))
    
    fullLen = min(data0['cam1'].shape[0], data1['cam1'].shape[0], data2['cam1'].shape[0])
    intervalWin = fullLen // 3
    
    #print("Total number of frames to work over: {}".format(fullLen))
    
    #Get the non-faked camera views. Arbitrarily pick data1
    #because the non-faked views should be the same across all 
    #files for a given ID    
    cams = get_cams(data1, num_cams, zero_start, fullLen)

    print("cams lens: ", len(cams))
    print("cams shape: ", (cams[0].shape))


    
    
    #Get the real camera to weave in with the fake camera
    try:
        real_cam0 = cams[fake_cams[0]]
        real_cam1 = cams[fake_cams[1]]
        real_cam2 = cams[fake_cams[2]]
    except IndexError:
        print("Number of cams",len(cams))
        print('First Fake cam: ',fake_cams[0])
        print('Second Fake cam: ',fake_cams[1])
        print('Third Fake cam: ',fake_cams[2])
        
    
    
    #Generate the fakes, using a standard or alternative procedure
    fake0, fake1, fake2 = split_procedure(data0, data1, data2, real_cam0, 
                                          real_cam1, real_cam2, alternative, fullLen, intervalWin)
    
    # plt.title("cams plotted in gen results, only SH coord 5, red is fake")
    # plt.plot(cams[0][:,5], 'tab:blue')
    # plt.plot(cams[1][:,5], 'tab:orange')
    # plt.plot(cams[2][:,5], 'tab:green')
    # plt.plot(cams[3][:,5], 'tab:purple')
    # plt.plot(cams[4][:,5], 'tab:brown')
    # plt.plot(cams[5][:,5], 'tab:pink')
    # plt.plot(fake2[:,5], 'tab:red')
    # plt.show()
    
    
    
# =============================================================================
#         (1) TP if window contains a faked frame & fake is detected
#         (2) TN if window does not have fake & fake is not detected
#         (3) FP if window does not have fake & fake is detected
#         (4) FN if window contains a faked frame & fake is not detected
# =============================================================================
    
    for ind, t in enumerate(threshes):
        #print("Processing threshold {}".format(t))
        for ind2, j in enumerate(window_sizes):
            #print('Processing window size ', str(j))
            numWin = fullLen - j
            if numWin <= 0:
                print("Skipping  window size " + str(j) + " for ID " + str(i) + " , as it is larger than the total number of available frames")
                continue
            acc0 = np.zeros((4, numWin))
            acc1 = np.zeros((4, numWin))
            acc2 = np.zeros((4, numWin))
            acc3 = np.zeros((4, numWin))
            for start in range(fullLen):
                end = start + j
                if end > fullLen-1:
                    continue
                
                # print("cams0 dims", cams[0].shape)
                # print("fake dims", fake0.shape)

                # print("cams diff pre norm", np.mean(cams[0] - cams[1]))
                # print("cam vs fake0 diff pre norm", np.mean(cams[0] - fake0))
                # print("cam vs fake0 diff pre norm", np.mean(cams[1] - fake1))

                numFakes0, numFakes1, numFakes2, numFakes3, c1, c2, c3 = noPCA(cams, fake0, fake1, fake2, start, end, num_pcs, t)
                # print("numFake0", numFakes0)
                # print("numFake1", numFakes1)
                # print("numFake2", numFakes2)
                # print("numFake3", numFakes3)

                if not alternative:
                    isFake = (len(set(range(start, end)).intersection(set(range(intervalWin, 2*intervalWin)))) == 0)
                else:
                    isFake = True
                
                #0 fakes case
                if numFakes0 ==0:
                    acc0[1][start] = 1 #TN
                else:
                    acc0[2][start] = 1 #FP
                    
                if zero_start:    
                    all_ones = np.ones((num_cams+1,))
                    all_twos = 2*np.ones((num_cams+1,))
                else:
                    all_ones = np.ones((num_cams,))
                    all_twos = 2*np.ones((num_cams,))
                    
                single_fake_ones = all_ones.copy()
                single_fake_ones[3] = 2
                single_fake_twos = all_twos.copy()
                single_fake_twos[3] = 1
                double_fake_ones = single_fake_ones.copy()
                double_fake_ones[2] = 2
                double_fake_twos = single_fake_twos.copy()
                double_fake_twos[2] = 1
                triple_fake_ones = double_fake_ones.copy()
                triple_fake_ones[1] = 2
                triple_fake_twos = double_fake_twos.copy()
                triple_fake_twos[1] = 1
                
 
                acc1[:, start] = calculate_acc_helper(single_fake_ones, 
                    single_fake_twos, numFakes1, c1, 1, isFake)
                
                acc2[:,start] = calculate_acc_helper(double_fake_ones, 
                    double_fake_twos, numFakes2, c2, 2, isFake)
                
                
                acc3[:,start] = calculate_acc_helper(triple_fake_ones, 
                    triple_fake_twos, numFakes3, c3, 3, isFake)
                
            # print("acc0 dims", acc0.shape)
            # print("acc1", acc1)
            # print("acc2", acc2)
            # print("acc3", acc3)
                
            #save the results to do more statistics with later
            saveDir = os.path.join(save_dir,"ID{}".format(i),"thresh_{}".format(ind))
            if not(os.path.isdir(saveDir)):
                os.makedirs(saveDir)
            saveDict = {'acc0':acc0, 'acc1': acc1, 
                        'acc2':acc2, 'acc3': acc3, 
                        'thresh': t, 'window_size':j }
            savemat(os.path.join(saveDir,"window_{}.mat".format(j)), saveDict)

    # print("zerodist: ", np.mean(np.array(zerodist)))
    # print("onedist: ", np.mean(np.array(onedist)))
    # print("twodist: ", np.mean(np.array(twodist)))
    # print("threedist: ", np.mean(np.array(threedist)))    




def main():  
    
    args = parse_args()
    
    ids = set()
    fake_cams_dict = defaultdict(list)
    
    for f in os.listdir(args.data_dir):
        try:
            x = re.search(r"fake([0-9]*)-ID([0-9]*).mat", f)
            ID = int(x.group(2))
            fake_cam = int(x.group(1))
            fake_cams_dict[ID].append(fake_cam)
            ids.add(ID)
        except AttributeError:
            continue
        
    exclude_list  = [17]
    ids = [i for i in ids if i not in exclude_list] 
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    #whether or not to use alternative procedure for fakes#
    alternative = False
    
# =============================================================================
#     for i in ids:
#         gen_results(i, args.data_dir, alternative, threshes, window_sizes, args.num_pcs, args.save_dir)
# =============================================================================
    
    Parallel(n_jobs=args.num_jobs)(delayed(gen_results)(i, fake_cams_dict[i], 
             args.num_cams, args.zero_start, args.data_dir, alternative, args.thresholds, 
             args.window_sizes, args.num_pcs, args.save_dir) for i in ids)


             

if __name__ == "__main__":
    main()
