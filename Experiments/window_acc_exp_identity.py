#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from sklearn.decomposition import PCA
from scipy.io import loadmat, savemat
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt



def mahalanobis(T, eigenval):
    
    cov = np.linalg.pinv(np.diag(eigenval.T))
    return np.sqrt(np.sum(np.multiply(np.matmul(T, cov), T), axis=1)) #T @ cov, T), axis = 1))



def mahalanobis_calculate(data, num_pcs):

    pca = PCA(num_pcs)
    T = pca.fit_transform(data)
    eigenval = pca.explained_variance_
    return mahalanobis(T, eigenval)

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
    
    #X0 is no fakes, X1 is 1 fake, etc.
    X0 = np.array([cam1Out, cam2Out, cam3Out, cam4Out, cam5Out, cam6Out])
    X1 = np.array([cam1Out, cam2Out, cam3Out, camFake3, cam5Out, cam6Out])
    X2 = np.array([cam1Out, cam2Out, camFake2, camFake3, cam5Out, cam6Out])
    X3 = np.array([cam1Out, camFake1, camFake2, camFake3, cam5Out, cam6Out])
    
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

    parser.add_argument('--data-dir', type=str, default='DataIdentity',
                    help='Directory where processed landmark files live')
    parser.add_argument('--num_pcs', type=int, default=5,
                    help='Number of principal components to use')
    parser.add_argument('--num_participants', type=int, default=1,
                    help='Number of participants')
    parser.add_argument('--save-dir', type=str, default='results',
                    help='Directory to save results')
    parser.add_argument('--rocOn', action = 'store_false')
    parser.add_argument('--roc-window-size', type=int, default=50, help = "Window size to use when generating ROC curve")
    parser.add_argument('--acc-threshold', type=float, default=1.3, help = "Threshold to use when generating ACC curve")
    parser.add_argument('--accOn', action = 'store_true')
    parser.add_argument("--thresholds", nargs="+", default=[1.3,1.5,1.7])
    parser.add_argument("--window-sizes", nargs="+", default=[50])
    
    
    args = parser.parse_args()
    return args


def calculate_acc_helper(option1, option2, numFakes, c, correctAnswer, isFake):
    acc = np.zeros((4,))
    
    if numFakes == correctAnswer:
        if isFake==0:
            acc[2] = 1  #FP, detected some number of fakes where there was none
        else:
            if (np.all(c == option1) or np.all(c == option2)):
                acc[0] = 1 #TP, detected some number of fakes where there was some
            else:
                acc[3] = 1 #FN, failed to detect some number of fakes where there was some
    elif not(numFakes == 0):
        acc[2] = 1 #FP deteected some number of fakes, but was wrong number of fakes
    else:
        if isFake == 0:
            acc[1] = 1 #TN, correctly got no fakes 
        else:
            acc[3] = 1 #FN incorrectly got no fakes
    return acc



def main():  
    
    args = parse_args()
    
    exclude_list  = [17]
    ids = [i for i in range(1, args.num_participants+1) if i not in exclude_list] 
    n = len(ids)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #Iterate over diffrent thresholds and window sizes
    threshes = args.thresholds
    window_sizes = args.window_sizes
    threshNum = len(threshes)  
    if args.rocOn:
        tpResults = np.zeros((threshNum,3,n))
        fpResults = np.zeros((threshNum,3,n))
    if args.accOn:
        accResults = np.zeros((4,len(window_sizes),n))
        
    person = 0
    
    for i in ids:  
        print("Processing ID", str(i))          
        if args.accOn:
            accs = np.zeros((4, len(args.window_sizes)))
        
        data2 = loadmat(os.path.join(args.data_dir, "fake2-ID{}.mat".format(i)))
        data3 = loadmat(os.path.join(args.data_dir, "fake3-ID{}.mat".format(i)))
        data4 = loadmat(os.path.join(args.data_dir, "fake4-ID{}.mat".format(i)))
        
        fullLen = min(data2['cam1'].shape[0], data3['cam1'].shape[0], data4['cam1'].shape[0])
        
        print("Total number of frames to work over: {}".format(fullLen))
        
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
        
        
# =============================================================================
#         (1) TP if window contains a faked frame & fake is detected
#         (2) TN if window does not have fake & fake is not detected
#         (3) FP if window does not have fake & fake is detected
#         (4) FN if window contains a faked frame & fake is not detected
# =============================================================================
        
        for ind, t in enumerate(threshes):
            print("Processing threshold {}".format(t))
            for ind2, j in enumerate(window_sizes):
                print('Processing window size ', str(j))
                numWin = fullLen - j
                acc0 = np.zeros((numWin,4))
                acc1 = np.zeros((numWin,4))
                acc2 = np.zeros((numWin,4))
                acc3 = np.zeros((numWin,4))
                for start in range(fullLen):
                    end = start + j
                    if end > fullLen-1:
                        continue
                    print('Start: ', str(start))
                    print('End: ', str(end))
                    
                    numFakes0, numFakes1, numFakes2, numFakes3, c1, c2, c3 = onlyPCA(cam1, cam2, cam3, cam4, cam5, cam6, fake2, 
            fake3, fake4, start, end, args.num_pcs, t)
                    
                    isFake = (len(set(range(start, end)).intersection(set(range(intervalWin, 2*intervalWin)))) == 0)
                    
                    #0 fakes case
                    if numFakes0 ==0:
                        acc0[start][1] = 1 #TN
                    else:
                        acc0[start][2] = 1 #FP

                    acc1[start,:] = calculate_acc_helper(np.array([1,1,1,2,1,1]), 
                        np.array([2,2,2,1,2,2]), numFakes1, c1, 1, isFake)
                    
                    acc2[start,:] = calculate_acc_helper(np.array([1,1,2,2,1,1]), 
                        np.array([2,2,1,1,2,2]), numFakes2, c2, 2, isFake)
                    
                    
                    acc3[start,:] = calculate_acc_helper(np.array([1,2,2,2,1,1]), 
                        np.array([2,1,1,1,2,2]), numFakes3, c3, 3, isFake)
                    
                    
                #save the results to do more statistics with later
                saveDir = os.path.join(args.save_dir,"ID{}".format(i),"thresh_{}".format(ind))
                if not(os.path.isdir(saveDir)):
                    os.makedirs(saveDir)
                saveDict = {'acc0x':acc0, 'acc1x': acc1, 
                            'acc2x':acc2, 'acc3x': acc3, 
                            'threshx': t, 'window_size':j }
                savemat(os.path.join(saveDir,"window_{}.mat".format(j)), saveDict)
                
                            
                if (args.rocOn and j == args.roc_window_size):
                    tpResults[ind,0,person] = np.sum(acc1[:,0]) / (np.sum(acc1[:,0]) + np.sum(acc1[:,3]) + 1e-7)
                    tpResults[ind,1,person] = np.sum(acc2[:,0]) / (np.sum(acc2[:,0]) + np.sum(acc2[:,3]) + 1e-7)
                    tpResults[ind,2,person] = np.sum(acc3[:,0]) / (np.sum(acc3[:,0]) + np.sum(acc3[:,3]) + 1e-7)
                    
                    fpResults[ind,0,person] = np.sum(acc1[:,2]) / (np.sum(acc1[:,2]) + np.sum(acc1[:,1]) +1e-7)
                    fpResults[ind,1,person] = np.sum(acc2[:,2]) / (np.sum(acc2[:,2]) + np.sum(acc2[:,1]) +1e-7)
                    fpResults[ind,2,person] = np.sum(acc3[:,2]) / (np.sum(acc3[:,2]) + np.sum(acc3[:,1]) +1e-7)

                if (args.accOn and t == args.acc_threshold):
                    
                    accs[0,ind2] = (np.sum(acc0[0,:]) + np.sum(acc0[1,:])) / (np.sum(acc0[0,:]) + np.sum(acc0[1,:]) + np.sum(acc0[2,:]) + np.sum(acc0[3,:]))
                    accs[1,ind2] = (np.sum(acc1[0,:]) + np.sum(acc1[1,:])) / (np.sum(acc1[0,:]) + np.sum(acc1[1,:]) + np.sum(acc1[2,:]) + np.sum(acc1[3,:]))
                    accs[2,ind2] = (np.sum(acc2[0,:]) + np.sum(acc2[1,:])) / (np.sum(acc2[0,:]) + np.sum(acc2[1,:]) + np.sum(acc2[2,:]) + np.sum(acc2[3,:]))
                    accs[3,ind2] = (np.sum(acc3[0,:]) + np.sum(acc3[1,:])) / (np.sum(acc3[0,:]) + np.sum(acc3[1,:]) + np.sum(acc3[2,:]) + np.sum(acc3[3,:]))
            
        if args.accOn:
            accResults[:,:,person] = accs
            
        person += 1
                        
    if args.rocOn:
        
        meanTP = np.mean(tpResults, axis = 2)     
        meanFP = np.mean(fpResults,axis = 2)
        stdTP = np.std(tpResults,axis = 2)
        stdFP = np.std(fpResults,axis = 2)
        
        #meanTP of shape: [num_thresholds, 3]
        
        oneFake = np.array([meanFP[:,0],meanTP[:,0] ])
        twoFake = np.array([meanFP[:,1],meanTP[:,1] ])
        threeFake = np.array([meanFP[:,2],meanTP[:,2] ])        
        
        oneFake = oneFake[oneFake[:,0].argsort(),]
        twoFake = twoFake[twoFake[:,0].argsort(),]
        threeFake = threeFake[threeFake[:,0].argsort(),]
        
        
        print(max(stdTP[:,0]))
        print(max(stdFP[:,0]))
        
        
        
        
        plt.errorbar(oneFake[0],oneFake[1], stdTP[:,0],stdFP[:,0], label = 'One Fake')
        plt.errorbar(twoFake[0],twoFake[1], stdTP[:,1],stdFP[:,1], label = 'Two Fakes')
        plt.errorbar(threeFake[0],threeFake[1], stdTP[:,2],stdFP[:,2], label = 'Three Fakes')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.legend()
        plt.title("ROC Curve. Window size = {}".format(args.roc_window_size))
        plt.savefig(os.path.join(args.save_dir, "roc_window_size_{}.jpg".format(args.roc_window_size)))
        
        
    if args.accOn:
        meanRes = np.mean(accResults,axis = 2)
        stdRes = np.std(accResults,axis = 2)
    
        plt.errorbar(args.window_sizes, meanRes[0,:], yerr = stdRes[0,:], label = "No Fakes")
        plt.errorbar(args.window_sizes, meanRes[1,:],yerr = stdRes[1,:], label = "One Fake")
        plt.errorbar(args.window_sizes, meanRes[2,:], yerr = stdRes[2,:],label = "Two Fakes")
        #plt.plot(args.window_sizes, meanRes[3,:], label = "Three Fakes")
        plt.xlim([min(args.window_sizes),max(args.window_sizes) + 50])
        plt.ylim([0,1])
        plt.xlabel("Window Size")
        plt.ylabel("Accuracy")
        plt.title("Detection Accuracy vs. Window Size with Threshold = {}".format(args.acc_threshold))
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, "acc_vs_window_size_thresh{}.jpg".format(args.acc_threshold)))
        
                    

if __name__ == "__main__":
    main()
    

