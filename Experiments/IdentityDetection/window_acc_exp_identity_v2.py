#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import argparse
import os
import re
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from rpca import *
from scipy.io import loadmat, savemat
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import kmeans, whiten
from scipy.special import softmax
from collections import defaultdict, Counter

from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')

    parser.add_argument('--data-dir', type=str, default='Data',
                    help='Directory where processed landmark files live')
    parser.add_argument('--num_pcs', type=int, default=5,
                    help='Number of principal components to use')
    parser.add_argument('--save-dir', type=str, default='Results',
                    help='Directory to save results')
    parser.add_argument('--zero-start', action='store_false',
                    help='Whether or not there is a cam0')
    parser.add_argument("--num-cams", type=int, default=6)
    parser.add_argument("--thresholds", nargs="+", default=[1.2, 1.3, 1.5, 1.7, 1.9, 2.1, 2.5])
    parser.add_argument("--window-sizes", nargs="+", default=[10, 20, 30, 40, 50, 60])
    parser.add_argument("--num-jobs", type=int, default=-1)
    
    args = parser.parse_args()
    return args

def mahalanobis(T, eigenval):
    cov = np.linalg.pinv(np.diag(eigenval.T))
    return np.sqrt(np.sum(np.multiply(np.matmul(T, cov), T), axis=1)) #T @ cov, T), axis = 1))

def mahalanobis_calculate(data, num_pcs, useRPCA = False):
    if useRPCA:
        data, _, _, _ = stoc_rpca(data, 100)
    pca = PCA(num_pcs)
    T = pca.fit_transform(data)
    eigenval = pca.explained_variance_
    return mahalanobis(T, eigenval)

def cluster_helper(X0, X1, X2, X3, thresh, mode='kmeans'):   
    if mode == 'kmeans':
        X0 = whiten(X0)
        X1 = whiten(X1)
        X2 = whiten(X2)
        X3 = whiten(X3)
        clusters0 = kmeans(X0, 2)
        clusters1 = kmeans(X1, 2)
        clusters2 = kmeans(X2, 2)
        clusters3 = kmeans(X3, 2)

        num_fakes0, c0, p0 = detect_fakes(clusters0, X0, thresh, mode='kmeans', correct=0)
        num_fakes1, c1, p1 = detect_fakes(clusters1, X1, thresh, mode='kmeans', correct=1)
        num_fakes2, c2, p2 = detect_fakes(clusters2, X2, thresh, mode='kmeans', correct=2)
        num_fakes3, c3, p3 = detect_fakes(clusters3, X3, thresh, mode='kmeans', correct=3)
    else:
        link0 = linkage(X0)
        link1 = linkage(X1)
        link2 = linkage(X2)
        link3 = linkage(X3)

        num_fakes0, c0, p0 = detect_fakes(link0, X0, thresh, mode='linkage')
        num_fakes1, c1, p1 = detect_fakes(link1, X1, thresh, mode='linkage')
        num_fakes2, c2, p2 = detect_fakes(link2, X2, thresh, mode='linkage')
        num_fakes3, c3, p3 = detect_fakes(link3, X3, thresh, mode='linkage')

    return num_fakes0, num_fakes1, num_fakes2, num_fakes3, c0, c1, c2, c3, p0, p1, p2, p3

def detect_fakes(clusters, X, thresh, mode='kmeans', correct=-1):
    if mode == 'kmeans':
        centroids, mean_distance = clusters

        ratio = np.linalg.norm(centroids[0]-centroids[1]) / mean_distance

        if ratio > thresh:
            # Determine which cluster each camera belongs 
            c = np.zeros(len(X))
            confidence = np.ones(len(X))

            average_dist_0 = 0
            average_dist_1 = 0
            for i in range(len(X)):
                dist_0 = np.linalg.norm(centroids[0]-X[i]) 
                dist_1 = np.linalg.norm(centroids[1]-X[i])
                if dist_0 > dist_1:
                    c[i] = 1
                    confidence[i] *= dist_0 / (dist_1 if dist_1 > 0 else 1)
                else:
                    c[i] = 0
                    confidence[i] *= dist_1 / (dist_0 if dist_0 > 0 else 1)
            partition0 = len(np.argwhere(c==0))
            partition1 = len(np.argwhere(c==1))
            numFakes = min(partition0, partition1)

            # Negate the confidence of the real class so they reflect our confidence
            # that they belong to the fake class
            if partition1 > partition0:
                c[c==1] = 2
                c[c==0] = 1
                c[c==2] = 0

            confidence[c==0] *= -1
        else:
            numFakes = 0
            c = np.ones(len(X))

            confidence = np.ones(len(X))

            for i in range(len(X)):
                dist_0 = np.linalg.norm(centroids[0]-X[i]) 
                dist_1 = np.linalg.norm(centroids[1]-X[i])
                if dist_0 > dist_1:
                    # Confidence we don't belong to the other cluster
                    confidence[i] *= -1 * dist_0 / (dist_1 if dist_1 > 0 else 1)
                    # Confidence compared to other cameras in this cluster
                    #confidence[i] *= average_dist_1 / (dist_1 if dist_1 > 0 else 1)
                else:
                    # Confidence we don't belong to the other cluster
                    confidence[i] *= -1* dist_1 / (dist_0 if dist_0 > 0 else 1)
                    # Confidence compared to other cameras in this cluster
                    #confidence[i] *= average_dist_0 / (dist_0 if dist_0 > 0 else 1)
    else:
        link = clusters
        ratio = link[-1][2] / link[-2][-2]

        if ratio > thresh:
            c = fcluster(link, 2, criterion='maxclust')
            c[c==2] = 0
            partition0 = len(np.argwhere(c==0))
            partition1 = len(np.argwhere(c==1))
            numFakes = min(partition0, partition1)

            if partition1 > partition0:
                c[c==1] = 2
                c[c==0] = 1
                c[c==2] = 0

            confidence = np.ones(len(c)) * ratio
            confidence[c==0] *= -1 
        else:
            numFakes = 0
            c = np.zeros(len(X))
            confidence = np.ones(len(c))*-1
    return numFakes, c, confidence
    
def build_test_arrays(camsOut, fake0Out, fake1Out, fake2Out):
    #X0 is no fakes, X1 is 1 fake, etc.
    X0 = np.array(camsOut)
    #print(X0)
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

def only_PCA(cams, fake0, fake1, fake2, start, end, num_pcs, thresh):
    
    camsOut = []
    for c in cams:
        camsOut.append(mahalanobis_calculate(c[start:end,:], num_pcs))
    
    fake0Out = mahalanobis_calculate(fake0[start:end,:], num_pcs)
    fake1Out = mahalanobis_calculate(fake1[start:end,:], num_pcs)
    fake2Out = mahalanobis_calculate(fake2[start:end,:], num_pcs)
    
    X0, X1, X2, X3 = build_test_arrays(camsOut, fake0Out, fake1Out, fake2Out)
    
    return cluster_helper(X0, X1, X2, X3, thresh, mode='linkage')

def no_PCA(cams, fake0, fake1, fake2, start, end, num_pcs, thresh):
    camsOut = []

    for c in cams:
        camsOut.append(np.mean(c[start:end], axis=0))
    fake0Out = np.mean(fake0[start:end], axis=0)
    fake1Out = np.mean(fake1[start:end], axis=0)
    fake2Out = np.mean(fake2[start:end], axis=0)

    X0, X1, X2, X3 = build_test_arrays(camsOut, fake0Out, fake1Out, fake2Out)
    return cluster_helper(X0, X1, X2, X3, thresh)

def adam_method(cams, fake0, fake1, fake2, start, end, num_pcs, thresh):
    allCamsTrim = []

    for c in cams:
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
    # cam6_norm = L2_sum(allCamsTrim, 6)
    fake0_norm = L2_sum(allCamsTrim, 6)
    fake1_norm = L2_sum(allCamsTrim, 7)
    fake2_norm = L2_sum(allCamsTrim, 8)

    all_cams = np.vstack([cam0_norm, cam1_norm, cam2_norm, cam3_norm, cam4_norm, cam5_norm, fake0_norm, fake1_norm, fake2_norm])

    mean_all_cams = np.mean(all_cams)
    std_all_cams = np.std(all_cams)

    all_cams -= mean_all_cams
    all_cams /= std_all_cams

    X0, X1, X2, X3 = build_test_arrays(all_cams[0:7], all_cams[7], all_cams[8], all_cams[9])
    
    return cluster_helper(X0, X1, X2, X3, thresh, mode='linkage')

def L2_sum(cams, index):
    curr_cam = cams[index]
    other_cams = cams.copy()
    other_cams.pop(index)
    sum = 0
    for cam in other_cams:
        sum += np.linalg.norm(cam - curr_cam, axis = 1)
    return sum

def chance_performance_test(cams, fake0, fake1, fake2, start, end, num_pcs, thresh):
    c = np.random.randint(2, size=4*len(cams))
    confidence = np.ones(4*len(cams))
    partition0 = len(np.argwhere(c==0))
    partition1 = len(np.argwhere(c==1))

    if partition1 > partition0:
        c[c==1] = 2
        c[c==0] = 1
        c[c==2] = 0

    confidence[c==0] *= -1

    c = np.reshape(c, (4, -1))
    confidence = np.reshape(confidence, (4, -1))

    return None, None, None, None, c[0], c[1], c[2], c[3], \
        confidence[0], confidence[1], confidence[2], confidence[3]



# option1 (1 == real), option0 (0 == real)
def calculate_acc_helper(option0, option1, c):
    """
    Required: from scipy.special import softmax
    
    Justifying examples:
    C = [1  1  1  2  1  1  1]   (2 == fake)
    c = [1  1  1  1  1  1  1]

        Of the real videos, we have gotten all of them correct,
        giving us a preadjustment TN rating of 1. 
        Of the fake videos we have gotten them all wrong, giving
        us a preadjustment FN rating of 1.

        [0  1  0  1] ==> softmax[-inf  1  -inf  1] ==> [0  0.5  0  0.5]

    C = [1  2  2  2  1  1  1]
    c = [1  1  1  2  1  1  2]

        Of the real videos, we have correctly classified 3/4 (TN) of them 
        and incorrectly classified 1/4 of them (FP).
        Of the fake videos, we have correctly classified 1/3 (TP) of them
        and incorrectly classified 2/3 of them (FN).

        [.33  .75  .25  .66] ==> softmax[.33  .75  .25  .66] ~~> [.21  .31  .19  .29]
    """
    acc = np.zeros((4,))

    real = np.argmax(np.bincount(c.astype('int64')))
    fake = 1 if real == 0 else 0
    C = (option1 if real == 0 else option0)
    number_of_fakes = np.count_nonzero(C == fake)

    for correct_guess, guess in zip(C, c):
        if correct_guess == guess == fake:
            acc[0] += 1/number_of_fakes #TP
        elif correct_guess == guess == real:
            acc[1] += 1/(len(c)-number_of_fakes) #TN
        elif correct_guess != guess == fake:
            acc[2] += 1/(len(c)-number_of_fakes) #FP
        elif correct_guess != guess == real:
            acc[3] += 1/number_of_fakes #FN
   
    acc[acc==0] = -np.inf

    return softmax(acc)
    #return acc

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
    data0 = loadmat(os.path.join(data_dir, "fake{}-ID{}.mat".format(fake_cams[0], i))) 
    data1 = loadmat(os.path.join(data_dir, "fake{}-ID{}.mat".format(fake_cams[1], i)))
    data2 = loadmat(os.path.join(data_dir, "fake{}-ID{}.mat".format(fake_cams[2], i)))

    fullLen = min(data0['cam1'].shape[0], data1['cam1'].shape[0], data2['cam1'].shape[0])
    intervalWin = fullLen // 3
    
    #Get the non-faked camera views. Arbitrarily pick data1
    #because the non-faked views should be the same across all 
    #files for a given ID    
    cams = get_cams(data1, num_cams, zero_start, fullLen)

    #Get the real camera to weave in with the fake camera
    try:
        if zero_start:
            real_cam0 = cams[fake_cams[0]]
            real_cam1 = cams[fake_cams[1]]
            real_cam2 = cams[fake_cams[2]]
        else:
            real_cam0 = cams[fake_cams[0]-1]
            real_cam1 = cams[fake_cams[1]-1]
            real_cam2 = cams[fake_cams[2]-1]
    except IndexError:
        print("Number of cams",len(cams))
        print('First Fake cam: ',fake_cams[0])
        print('Second Fake cam: ',fake_cams[1])
        print('Third Fake cam: ',fake_cams[2])
    
    #Generate the fakes, using a standard or alternative procedure
    fake0, fake1, fake2 = split_procedure(data0, data1, data2, real_cam0, 
                                          real_cam1, real_cam2, alternative, fullLen, intervalWin)
    
    #total = 0
    #for i in window_sizes:
    #    total += max(fullLen-i, 0)
    #pbar = tqdm(total=total, position=pos)

    for ind, t in enumerate(threshes):
        for ind2, j in enumerate(window_sizes):
            numWin = fullLen - j
            if numWin <= 0:
                print("Skipping  window size " + str(j) + " for ID " + str(i) + " , as it is larger than the total number of available frames")
                continue
            acc0 = np.zeros((4, numWin))
            acc1 = np.zeros((4, numWin))
            acc2 = np.zeros((4, numWin))
            acc3 = np.zeros((4, numWin))

            p0_total = None
            p1_total = None
            p2_total = None
            p3_total = None

            for start in range(fullLen):
                end = start + j
                #pbar.update(1)
                if end > fullLen-1:
                    continue
                
                if not alternative:
                    isFake = (len(set(range(start, end)).intersection(set(range(intervalWin, 2*intervalWin)))) == 0)
                else:
                    isFake = True
                
                numFakes0, numFakes1, numFakes2, numFakes3, c0, c1, c2, c3, p0, p1, p2, p3 = \
                    adam_method(cams, fake0, fake1, fake2, start, end, num_pcs, t)
                    
                if zero_start:    
                    all_ones = np.ones((num_cams+1,))
                    all_zeros = np.zeros((num_cams+1,))
                else:
                    all_ones = np.ones((num_cams,))
                    all_zeros = np.zeros((num_cams,))
                    
                single_fake_ones = all_ones.copy()
                single_fake_ones[3] = 0
                single_fake_zeros = all_zeros.copy()
                single_fake_zeros[3] = 1
                double_fake_ones = single_fake_ones.copy()
                double_fake_ones[2] = 0
                double_fake_zeros = single_fake_zeros.copy()
                double_fake_zeros[2] = 1
                triple_fake_ones = double_fake_ones.copy()
                triple_fake_ones[1] = 0
                triple_fake_zeros = double_fake_zeros.copy()
                triple_fake_zeros[1] = 1
                
                acc0[:, start] = calculate_acc_helper(all_ones, all_zeros, c0)
                acc1[:, start] = calculate_acc_helper(single_fake_ones, 
                    single_fake_zeros, c1)
                acc2[:,start] = calculate_acc_helper(double_fake_ones, 
                    double_fake_zeros, c2)
                acc3[:,start] = calculate_acc_helper(triple_fake_ones, 
                    triple_fake_zeros, c3)

                p = np.hstack([p0, p1, p2, p3])
                c = np.hstack([all_zeros, single_fake_zeros, double_fake_zeros, triple_fake_zeros])

                y = np.vstack([p, c])

                if start == 0:
                    p0_total = np.vstack([p0, all_zeros])
                    p1_total = np.vstack([p1, single_fake_zeros])
                    p2_total = np.vstack([p2, double_fake_zeros])
                    p3_total = np.vstack([p3, triple_fake_zeros])
                else:
                    p0_total = np.hstack([p0_total, np.vstack([p0, all_zeros])])
                    p1_total = np.hstack([p1_total, np.vstack([p1, single_fake_zeros])])
                    p2_total = np.hstack([p2_total, np.vstack([p2, double_fake_zeros])])
                    p3_total = np.hstack([p3_total, np.vstack([p3, triple_fake_zeros])])

            saveDir = os.path.join(save_dir,"ID{}".format(i),"thresh_{}".format(ind))
            if not(os.path.isdir(saveDir)):
                os.makedirs(saveDir)
            saveDict = {'acc0':acc0, 'acc1': acc1, 
                        'acc2':acc2, 'acc3': acc3, 
                        'thresh': t, 'window_size':j }
            savemat(os.path.join(saveDir,"window_{}.mat".format(j)), saveDict)
            pDict = {'p0': p0_total, 'p1':p1_total, 'p2':p2_total, 'p3':p3_total}
            savemat(os.path.join(saveDir, "p_window_{}.mat".format(j)), pDict)

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

    print(len(fake_cams_dict.keys()))
    
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
