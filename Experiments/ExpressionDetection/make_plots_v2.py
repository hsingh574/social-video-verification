#!/usr/bin/env python3


import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from scipy.io import loadmat
import re
from sklearn.metrics import roc_curve

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')

    parser.add_argument('--rocOn', action='store_false',
                    help='Whether to plot ROC curve')
    
    parser.add_argument('--accOn', action='store_false',
                    help='Whether to plot Accuracy curve')
    
    parser.add_argument('--prOn', action='store_true',
                    help='Whether to plot Precision Recall curve')
    
    parser.add_argument('--data-dir', type=str, default='DataEleanorShort',
                    help='Directory where data has been saved')
    
    parser.add_argument('--results-dir', type=str, default='L2 method Results',
                    help='Directory where results have been saved')
    
    parser.add_argument('--save-dir', type=str, default='Plots',
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
   
    skipID = False
    for i,ID in enumerate(ids):
        if skipID:
            continue
        for j, t in enumerate(threshes):
            try:
                results = loadmat(os.path.join(results_dir, 'ID{}'.format(ID),
                                           'thresh_{}'.format(j), 
                                           "window_{}.mat".format(window_size)))
                skipID = False
            except FileNotFoundError:
                skipID = True
                break
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
    stdP = np.std(pResults,axis = 2, ddof = 1)
    stdR = np.std(rResults, axis = 2, ddof = 1)

    #reformat & order by recall values
    #sort rows by first column
    oneFake = np.column_stack([meanR[:,0], meanP[:,0]])
    twoFake = np.column_stack([meanR[:,1], meanP[:,1]])
    thrFake = np.column_stack([meanR[:,2], meanP[:,2]])
    
    oneFake = oneFake[np.argsort(oneFake[:, 0])]
    twoFake = twoFake[np.argsort(twoFake[:, 0])]
    thrFake = thrFake[np.argsort(thrFake[:, 0])]
    
    plt.figure(3)
    
    plt.errorbar(oneFake[:,0], oneFake[:,1], yerr = stdP[:,0], xerr = stdR[:,0], label = 'One Fake')
    plt.errorbar(twoFake[:,0], twoFake[:,1], yerr = stdP[:,1], xerr = stdR[:,1], label = 'Two Fakes')
    plt.errorbar(thrFake[:,0], thrFake[:,1], yerr = stdP[:,2], xerr = stdR[:,2], label = 'Three Fakes')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc = 'lower right')
    plt.title('Precision v Recall, Window size = {}'.format(window_size))
    plt.savefig(os.path.join(save_dir, 'pr_plot_window_size_{}.png'.format(window_size)))

                   
def acc_helper(results):
    numerator = np.sum(results[0,:]) + np.sum(results[1,:])
    denominator = np.sum(results, axis = (0,1))
    return numerator / (1 if denominator == 0 else denominator)
            
def plot_acc(ids, window_sizes, threshold, threshold_idx, results_dir, save_dir):
    print('Detection Accuracy v Window Size. Threshold = {}'.format(threshold))
    numPeople = len(ids)
    numWin = len(window_sizes)
    skipID = False 
    accResults = np.zeros((4,numWin,numPeople))
    
    for i,ID in enumerate(ids):
        if skipID:
            continue
        accs = np.zeros((4,numWin))
        
        for j, w in enumerate(window_sizes):
            try:  
                results = loadmat(os.path.join(results_dir, 'ID{}'.format(ID),
                                           'thresh_{}'.format(threshold_idx), 
                                           "window_{}.mat".format(w)))
                skipID = False
            except FileNotFoundError:
                skipID = True
                break

            accs[0,j] = acc_helper(results['acc0'])
            accs[1,j] = acc_helper(results['acc1'])
            accs[2,j] = acc_helper(results['acc2'])
            accs[3,j] = acc_helper(results['acc3'])
        accResults[:,:,i] = accs
    
    meanRes = np.mean(accResults, axis = 2)
    print("mean: ", np.mean(meanRes))

    stdRes = np.std(accResults, axis = 2, ddof=1)
    print("std: ", np.std(stdRes))

    print("\n")
    
    plt.figure()
    
    plt.errorbar(window_sizes, meanRes[0,:], yerr = stdRes[0,:], label = 'Zero Fakes', elinewidth=0.5, capsize=1)
    plt.errorbar(window_sizes, meanRes[1,:], yerr = stdRes[1,:], label = 'One Fake', elinewidth=0.5, capsize=1)
    plt.errorbar(window_sizes, meanRes[2,:], yerr = stdRes[2,:], label = 'Two Fakes', elinewidth=0.5, capsize=1)
    plt.errorbar(window_sizes, meanRes[3,:], yerr = stdRes[3,:], label = 'Three Fakes', elinewidth=0.5, capsize=1)
    
    plt.xlim([0, max(window_sizes)+10])
    plt.ylim([0, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Window Size')
    plt.legend(loc = 'lower right')
    plt.title('Detection Accuracy v Window Size. Threshold = {}'.format(threshold))
    plt.savefig(os.path.join(save_dir, 'acc_plot_thresh_{}.png'.format(threshold)))

def acc_helper_v2(p):
    y_score = p[0, :]
    y_pred = y_score.copy()
    y_pred[y_score > 0] = 1
    y_pred[y_score <= 0] = 0
    y_true = p[1, :]

    num_correct = np.sum((y_pred-y_true==0).astype('int32'))
    num_pred = len(y_pred)

    return num_correct / num_pred



def plot_acc_v2(ids, window_sizes, threshold, threshold_idx, results_dir, save_dir):
    print('Detection Accuracy v Window Size. Threshold = {}'.format(threshold))
    numPeople = len(ids)
    numWin = len(window_sizes)
    skipID = False 
    accResults = np.zeros((4,numWin,numPeople))
    
    for i,ID in enumerate(ids):
        if skipID:
            continue
        accs = np.zeros((4,numWin))
        
        for j, w in enumerate(window_sizes):
            try:  
                results = loadmat(os.path.join(results_dir, 'ID{}'.format(ID),
                                           'thresh_{}'.format(threshold_idx), 
                                           "p_window_{}.mat".format(w)))
                skipID = False
            except FileNotFoundError:
                skipID = True
                break

            accs[0,j] = acc_helper_v2(results['p0'])
            accs[1,j] = acc_helper_v2(results['p1'])
            accs[2,j] = acc_helper_v2(results['p2'])
            accs[3,j] = acc_helper_v2(results['p3'])
        accResults[:,:,i] = accs
    
    meanRes = np.mean(accResults, axis = 2)
    print("mean: ", np.mean(meanRes))

    stdRes = np.std(accResults, axis = 2, ddof=1)
    print("std: ", np.std(stdRes))

    print("\n")
    
    plt.figure()
    
    plt.errorbar(window_sizes, meanRes[0,:], yerr = stdRes[0,:], label = 'Zero Fakes', elinewidth=0.5, capsize=1)
    plt.errorbar(window_sizes, meanRes[1,:], yerr = stdRes[1,:], label = 'One Fake', elinewidth=0.5, capsize=1)
    plt.errorbar(window_sizes, meanRes[2,:], yerr = stdRes[2,:], label = 'Two Fakes', elinewidth=0.5, capsize=1)
    plt.errorbar(window_sizes, meanRes[3,:], yerr = stdRes[3,:], label = 'Three Fakes', elinewidth=0.5, capsize=1)
    
    plt.xlim([0, max(window_sizes)+10])
    plt.ylim([0, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Window Size')
    plt.legend(loc = 'lower right')
    plt.title('Detection Accuracy v Window Size. Threshold = {}'.format(threshold))
    plt.savefig(os.path.join(save_dir, 'acc_plot_thresh_{}.png'.format(threshold)))


def plot_ROC_v2(ids, threshes, window_size, results_dir, save_dir):
    y_true_0 = None
    y_score_0 = None

    y_true_1 = None
    y_score_1 = None

    y_true_2 = None
    y_score_2 = None

    y_true_3 = None
    y_score_3 = None
    for i,ID in enumerate(ids):
        for j, threshold in enumerate(threshes):
            try:
                results = loadmat(os.path.join(results_dir, 'ID{}'.format(ID),
                                            'thresh_{}'.format(j), 
                                            "p_window_{}.mat".format(window_size)))
            except FileNotFoundError:
                print("not found")
                continue

            if i == 0 and j == 0:
                y_score_0 = results['p0'][0, :]
                y_true_0 = results['p0'][1, :]

                y_score_1 = results['p1'][0, :]
                y_true_1 = results['p1'][1, :]

                y_score_2 = results['p2'][0, :]
                y_true_2 = results['p2'][1, :]

                y_score_3 = results['p3'][0, :]
                y_true_3 = results['p3'][1, :]
            else:
                y_score_0 = np.hstack([y_score_0, results['p0'][0, :]])
                y_true_0 = np.hstack([y_true_0, results['p0'][1, :]])

                y_score_1 = np.hstack([y_score_1, results['p1'][0, :]])
                y_true_1 = np.hstack([y_true_1, results['p1'][1, :]])

                y_score_2 = np.hstack([y_score_2, results['p2'][0, :]])
                y_true_2 = np.hstack([y_true_2, results['p2'][1, :]])

                y_score_3 = np.hstack([y_score_3, results['p3'][0, :]])
                y_true_3 = np.hstack([y_true_3, results['p3'][1, :]])
        
    #fpr0, tpr0, _  = roc_curve(y_true_0, y_score_0)
    fpr1, tpr1, _  = roc_curve(y_true_1, y_score_1)
    fpr2, tpr2, _  = roc_curve(y_true_2, y_score_2)
    fpr3, tpr3, _  = roc_curve(y_true_3, y_score_3)

    plt.figure()

    #plt.plot(fpr0, tpr0, label = 'Zero Fake')
    plt.plot(fpr1, tpr1, label = 'One Fake')
    plt.plot(fpr2, tpr2, label = 'Two Fake')
    plt.plot(fpr3, tpr3, label = 'Three Fake')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right')
    plt.title('ROC Curve, Window size = {}'.format(window_size))
    plt.savefig(os.path.join(save_dir, 'roc_plot_window_size_{}.png'.format(window_size)))

def plot_ROC_helper(acc):
    tpr = int(np.sum(acc[0,:])) # number of TP
    tnr = int(np.sum(acc[1,:])) # number of TN
    fpr = int(np.sum(acc[2,:])) # number of FP
    fnr = int(np.sum(acc[3,:])) # number of FN

    # TP
    y_true = np.ones(tpr)
    y_score = np.ones(tpr)

    # TN
    y_true = np.hstack([y_true, np.zeros(tnr)])
    y_score = np.hstack([y_score, -1*np.ones(tnr)])

    # FP
    y_true = np.hstack([y_true, np.zeros(fpr)])
    y_score = np.hstack([y_score, np.ones(fpr)])

    # FN
    y_true = np.hstack([y_true, np.ones(fnr)])
    y_score = np.hstack([y_score, -1*np.ones(fnr)])

    return y_score, y_true

def plot_ROC(ids, threshes, window_size, results_dir, save_dir):
    
    threshNum = len(threshes)
    numPeople = len(ids)
    skipID = False

    y_true_1 = None
    y_score_1 = None

    y_true_2 = None
    y_score_2 = None

    y_true_3 = None
    y_score_3 = None
    
    for i,ID in enumerate(ids):
        if skipID:
            continue
        for j, t in enumerate(threshes):
            try:
                results = loadmat(os.path.join(results_dir, 'ID{}'.format(ID),
                                           'thresh_{}'.format(j), 
                                           "window_{}.mat".format(window_size)))
                skipID = False
            except FileNotFoundError:
                skipID = True
                break

            if i == 0 and j == 0:
                y_score_1, y_true_1 = plot_ROC_helper(results['acc1'])

                y_score_2, y_true_2 = plot_ROC_helper(results['acc2'])

                y_score_3, y_true_3 = plot_ROC_helper(results['acc3'])
            else:
                y_score_1_update, y_true_1_update = plot_ROC_helper(results['acc1'])
                y_score_1 = np.hstack([y_score_1, y_score_1_update])
                y_true_1 = np.hstack([y_true_1, y_true_1_update])

                y_score_2_update, y_true_2_update = plot_ROC_helper(results['acc2'])
                y_score_2 = np.hstack([y_score_2, y_score_2_update])
                y_true_2 = np.hstack([y_true_2, y_true_2_update])

                y_score_3_update, y_true_3_update = plot_ROC_helper(results['acc3'])
                y_score_3 = np.hstack([y_score_3, y_score_3_update])
                y_true_3 = np.hstack([y_true_3, y_true_3_update])
            
    fpr1, tpr1, _  = roc_curve(y_true_1, y_score_1)
    fpr2, tpr2, _  = roc_curve(y_true_2, y_score_2)
    fpr3, tpr3, _  = roc_curve(y_true_3, y_score_3)

    plt.figure()

    plt.plot(fpr1, tpr1, label = 'One Fake')
    plt.plot(fpr2, tpr2, label = 'Two Fake')
    plt.plot(fpr3, tpr3, label = 'Three Fake')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right')
    plt.title('ROC Curve, Window size = {}'.format(window_size))
    plt.savefig(os.path.join(save_dir, 'roc_plot_window_size_{}.png'.format(window_size)))

def main():
    
    args = parse_args()
    
    
    ids = set()
    for f in os.listdir(args.data_dir):
        try:
            x = re.search(r"fake([0-9]*)-ID([0-9]*).mat", f)
            ID = int(x.group(2))
            ids.add(ID)
        except AttributeError:
            continue
    
    exclude_list  = [17]
    ids = [i for i in ids if i not in exclude_list] 
    
    print(ids) 
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    threshes = args.thresholds
    window_sizes = args.window_sizes
    
    if args.rocOn:
        for window_size in args.window_sizes:
            plot_ROC_v2(ids, threshes, window_size, args.results_dir, args.save_dir)
        
    if args.accOn:
        for i, threshold in enumerate(args.thresholds):
            plot_acc(ids, window_sizes, threshold, i, args.results_dir, args.save_dir)
        
    #if args.prOn:
    #    plot_PR(ids, threshes, window_size, args.results_dir, args.save_dir)



if __name__ == "__main__":
    main()
