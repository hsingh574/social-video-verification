#!/usr/bin/env python3


import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from scipy.io import loadmat



def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')

    parser.add_argument('--rocOn', action='store_true',
                    help='Whether to plot ROC curve')
    
    parser.add_argument('--accOn', action='store_true',
                    help='Whether to plot Accuracy curve')
    
    parser.add_argument('--prOn', action='store_true',
                    help='Whether to plot Precision Recall curve')
    
    parser.add_argument('--num_participants', type=int, default=25,
                    help='Number of participants')
    
    parser.add_argument('--results-dir', type=str, default='results_v2',
                    help='Directory where results have been saved')
    
    parser.add_argument('--save-dir', type=str, default='plots_v2',
                    help='Directory to save results plots')
    
    
    parser.add_argument("--thresholds", nargs="+", default=[1.3, 1.5, 1.7, 1.9, 2.1])
    parser.add_argument("--window-sizes", nargs="+", default=[50, 150, 250, 350])
    
    
    args = parser.parse_args()
    return args

def plot_PR(ids, threshes, window_size, results_dir, save_dir):
    threshNum = len(threshes)
    numPeople = len(ids)
    
    pResults = np.zeros((threshNum,3,numPeople))
    rResults = np.zeros((threshNum,3,numPeople))
    
    for i,ID in enumerate(ids):
        for j, t in enumerate(threshes):
            results = loadmat(os.path.join(results_dir, 'ID{}'.format(ID),
                                           'thresh_{}'.format(j), 
                                           "window_{}.mat".format(window_size)))

            if (np.sum(results['acc1'][0,:]) == 0 and np.sum(results['acc1'][2,:]) == 0):
                pResults[j,0,i] = 1
            else:
                pResults[j,0,i] = (np.sum(results['acc1'][0,:]) / 
                        (np.sum(results['acc1'][0,:]) + np.sum(results['acc1'][2,:])))
                        
            if (np.sum(results['acc2'][0,:]) == 0 and np.sum(results['acc2'][2,:]) == 0):
                pResults[j,1,i] = 1
            else:
                pResults[j,1,i] = (np.sum(results['acc2'][0,:]) / 
                        (np.sum(results['acc2'][0,:]) + np.sum(results['acc2'][2,:])))
                
            if (np.sum(results['acc3'][0,:]) == 0 and np.sum(results['acc3'][2,:]) == 0):
                pResults[j,2,i] = 1
            else:
                pResults[j,2,i] = (np.sum(results['acc3'][0,:]) / 
                        (np.sum(results['acc3'][0,:]) + np.sum(results['acc3'][2,:])))
                
            rResults[j,0,i] = (np.sum(results['acc1'][0,:]) / 
                            (np.sum(results['acc1'][0,:]) + np.sum(results['acc1'][3,:])))
            
            rResults[j,1,i] = (np.sum(results['acc2'][0,:]) / 
                            (np.sum(results['acc2'][0,:]) + np.sum(results['acc2'][3,:])))
            
            rResults[j,2,i] = (np.sum(results['acc3'][0,:]) / 
                            (np.sum(results['acc3'][0,:]) + np.sum(results['acc3'][3,:])))
            
    meanP = np.mean(pResults,axis = 2)
    meanR = np.mean(rResults,axis = 2)
    stdP = np.std(pResults,axis = 2)
    stdR = np.std(rResults, axis = 2)

    #reformat & order by recall values
    #sort rows by first column
    oneFake = np.hstack([meanR[:,0], meanP[:,0]])
    twoFake = np.hstack([meanR[:,1], meanP[:,1]])
    thrFake = np.hstack([meanR[:,2], meanP[:,2]])
    
    oneFake = oneFake[np.argsort(oneFake[:, 0])]
    twoFake = twoFake[np.argsort(twoFake[:, 0])]
    thrFake = thrFake[np.argsort(thrFake[:, 0])]


    plt.errorbar(oneFake[:,0], oneFake[:,1], stdP[:,0], stdR[:,0], label = 'One Fake')
    plt.errorbar(twoFake[:,1], twoFake[:,1], stdP[:,1], stdR[:,1], label = 'Two Fakes')
    plt.errorbar(thrFake[:,2], thrFake[:,2], stdP[:,2], stdR[:,2], label = 'Three Fakes')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc = 'lower right')
    plt.title('Precision v Recall, Window size = {}'.format(window_size))
    plt.savefig(os.path.join(save_dir, 'pr_plot_window_size_{}'.format(window_size)))

                   
def acc_helper(results):
    
    numerator = np.sum(results[0,:]) + np.sum(results[1,:])
    denominator = np.sum(results, axis = (0,1))
    return numerator / denominator
            
def plot_acc(ids, window_sizes, threshold, threshold_idx, results_dir, save_dir):
    
    numPeople = len(ids)
    numWin = len(window_sizes)
    
    accResults = np.zeros((4,4,numPeople))
    
    for i,ID in enumerate(ids):
        accs = np.zeros((4,numWin))
        
        for j, w in enumerate(window_sizes):
            results = loadmat(os.path.join(results_dir, 'ID{}'.format(ID),
                                           'thresh_{}'.format(threshold_idx), 
                                           "window_{}.mat".format(w)))
            
            accs[0,j] = acc_helper(results['acc0'])
            accs[1,j] = acc_helper(results['acc1'])
            accs[2,j] = acc_helper(results['acc2'])
            accs[3,j] = acc_helper(results['acc3'])
        accResults[:,:,i] = accs
    
    meanRes = np.mean(accResults, axis = 3)
    stdRes = np.std(accResults, axis = 3)
    
    plt.errorbar(window_sizes, meanRes[0,:], stdRes[0,:], label = 'Zero Fakes')
    plt.errorbar(window_sizes, meanRes[1,:], stdRes[1,:], label = 'One Fake')
    plt.errorbar(window_sizes, meanRes[2,:], stdRes[2,:], label = 'Two Fakes')
    plt.errorbar(window_sizes, meanRes[3,:], stdRes[3,:], label = 'Three Fakes')
    
    plt.xlim([0, 400])
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.xlabel('Window Size')
    plt.legend(loc = 'lower right')
    plt.title('Detection Accuracy v Window Size. Threshold = {}'.format(threshold))
    plt.savefig(os.path.join(save_dir, 'acc_plot_thresh_{}'.format(threshold)))
        
            

def plot_ROC(ids, threshes, window_size, results_dir, save_dir):
    
    threshNum = len(threshes)
    numPeople = len(ids)
    
    tpResults = np.zeros((threshNum,3,numPeople))
    fpResults = np.zeros((threshNum,3,numPeople))
    fpZeroFake = np.zeros((threshNum,1,numPeople))
    
    for i,ID in enumerate(ids):
        for j, t in enumerate(threshes):
            results = loadmat(os.path.join(results_dir, 'ID{}'.format(ID),
                                           'thresh_{}'.format(j), 
                                           "window_{}.mat".format(window_size)))
    
            
            tpResults[j,0,i] = (np.sum(results['acc1'][0,:]) / (np.sum(results['acc1'][0,:]) + 
                                                                 np.sum(results['acc1'][3,:])))
            
            tpResults[j,1,i] = (np.sum(results['acc2'][0,:]) / (np.sum(results['acc2'][0,:]) + 
                                                                 np.sum(results['acc2'][3,:])))
            
            tpResults[j,2,i] = (np.sum(results['acc3'][0,:]) / (np.sum(results['acc3'][0,:]) + 
                                                                 np.sum(results['acc3'][3,:])))
            
            fpResults[j,0,i] = (np.sum(results['acc1'][3,:]) / (np.sum(results['acc1'][3,:]) + 
                                                                 np.sum(results['acc1'][2,:])))
            
            fpResults[j,1,i] = (np.sum(results['acc2'][3,:]) / (np.sum(results['acc2'][3,:]) + 
                                                                 np.sum(results['acc2'][2,:])))
            
            fpResults[j,2,i] = (np.sum(results['acc3'][3,:]) / (np.sum(results['acc3'][3,:]) + 
                                                                 np.sum(results['acc3'][2,:])))
            
            fpZeroFake[j,0,i] = (np.sum(results['acc0'][3,:]) / (np.sum(results['acc0'][3,:]) + 
                                                                 np.sum(results['acc0'][2,:])))
    
    meanTP = np.mean(tpResults,axis = 2)
    meanFP = np.mean(fpResults,axis = 2)
    stdTP = np.std(tpResults,axis = 2)
    stdFP = np.std(fpResults, axis = 2)

    #zeroFakeMean = np.mean(fpZeroFake,axis = 2)
    #zeroFakeStd = np.std(fpZeroFake,axis = 2)

    #reformat & order by recall values
    #sort rows by first column
    oneFake = np.hstack([meanFP[:,0], meanTP[:,0]])
    twoFake = np.hstack([meanFP[:,1], meanTP[:,1]])
    thrFake = np.hstack([meanFP[:,2], meanTP[:,2]])
    
    oneFake = oneFake[np.argsort(oneFake[:, 0])]
    twoFake = twoFake[np.argsort(twoFake[:, 0])]
    thrFake = thrFake[np.argsort(thrFake[:, 0])]


    plt.errorbar(oneFake[:,0], oneFake[:,1], stdTP[:,0], stdFP[:,0], label = 'One Fake')
    plt.errorbar(twoFake[:,1], twoFake[:,1], stdTP[:,1], stdFP[:,1], label = 'Two Fakes')
    plt.errorbar(thrFake[:,2], thrFake[:,2], stdTP[:,2], stdFP[:,2], label = 'Three Fakes')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right')
    plt.title('ROC Curve, Window size = {}'.format(window_size))
    plt.savefig(os.path.join(save_dir, 'roc_plot_window_size_{}'.format(window_size)))



def main():
    
    args = parse_args()
    
    exclude_list  = [17]
    ids = [i for i in range(1, args.num_participants+1) if i not in exclude_list] 
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    threshes = args.thresholds
    window_sizes = args.window_sizes
    
    
    
    window_size = 250
    threshold = 1.5
    threshold_idx = 1
    
    if args.rocOn:
        plot_ROC(ids, threshes, window_size, args.results_dir, args.save_dir)
        
    if args.accOn:
        plot_acc(ids, window_sizes, threshold, threshold_idx, args.results_dir, args.save_dir)
        
    if args.prOn:
        plot_PR(ids, threshes, window_size, args.results_dir, args.save_dir)



if __name__ == "__main__":
    main()