#!/usr/bin/env python3
# Script to save a ROC and accuracy plot per 0-3 fakes across baselines
# Eg for how to run:
# python3 make_comparison_plots_for_paper.py --data-dir ./dataset/ --results-dir ./results_ours/ --results-dir-base1 ./results_base1/ --results-dir-base2 ./results_base2/ --save-dir ./out/

import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from scipy.io import loadmat
import re

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')
    
    parser.add_argument('--data-dir', type=str, default='data_v2',
                    help='Directory where data has been saved')
    
    parser.add_argument('--results-dir', type=str, default='results_v2',
                    help='Directory where results have been saved')

    parser.add_argument('--results-dir-base1', type=str, default='results_v2',
                    help='Directory where baseline 1 results have been saved')

    parser.add_argument('--results-dir-base2', type=str, default='results_v2',
                    help='Directory where baseline 2 results have been saved')

    parser.add_argument('--save-dir', type=str, default='plots_v2',
                    help='Directory to save results plots')
    
    
    parser.add_argument("--thresholds", nargs="+", default=[1.3, 1.5, 1.7, 1.9, 2.1])
    parser.add_argument("--window-sizes", nargs="+", default=[10,20,30,40,50])
    
    
    args = parser.parse_args()
    return args
                   
def acc_helper(results):
    
    numerator = np.sum(results[0,:]) + np.sum(results[1,:])
    denominator = np.sum(results, axis = (0,1))
    return numerator / denominator
            
def plot_acc(ids, window_sizes, threshold, threshold_idx, results_dir, save_dir):
    
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
    stdRes = np.std(accResults, axis = 2, ddof=1)

    return meanRes, stdRes    
 
def plot_ROC(ids, threshes, window_size, results_dir, save_dir):
    
    threshNum = len(threshes)
    numPeople = len(ids)
    skipID = False
    tpResults = np.zeros((threshNum,3,numPeople))
    fpResults = np.zeros((threshNum,3,numPeople))
    fpZeroFake = np.zeros((threshNum,1,numPeople))
    
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
            
            tpResults[j,0,i] = (np.sum(results['acc1'][0,:]) / (np.sum(results['acc1'][0,:]) + 
                                                                 np.sum(results['acc1'][3,:])))
            
            tpResults[j,1,i] = (np.sum(results['acc2'][0,:]) / (np.sum(results['acc2'][0,:]) + 
                                                                 np.sum(results['acc2'][3,:])))
            
            tpResults[j,2,i] = (np.sum(results['acc3'][0,:]) / (np.sum(results['acc3'][0,:]) + 
                                                                 np.sum(results['acc3'][3,:])))
            
            fpResults[j,0,i] = (np.sum(results['acc1'][2,:]) / (np.sum(results['acc1'][2,:]) + 
                                                                 np.sum(results['acc1'][1,:])))
            
            fpResults[j,1,i] = (np.sum(results['acc2'][2,:]) / (np.sum(results['acc2'][2,:]) + 
                                                                 np.sum(results['acc2'][1,:])))
            
            fpResults[j,2,i] = (np.sum(results['acc3'][2,:]) / (np.sum(results['acc3'][2,:]) + 
                                                                 np.sum(results['acc3'][1,:])))
            
            if (np.sum(results['acc0'][2,:]) + np.sum(results['acc0'][1,:])==0):
                fpZeroFake[j,0,i] = 0
            else:
                fpZeroFake[j,0,i] = (np.sum(results['acc0'][2,:]) / (np.sum(results['acc0'][2,:]) + 
                                                                 np.sum(results['acc0'][1,:])))
    
    meanTP = np.mean(tpResults,axis = 2)
    meanFP = np.mean(fpResults,axis = 2)
    stdTP = np.std(tpResults,axis = 2, ddof=1)
    stdFP = np.std(fpResults, axis = 2, ddof=1)

    #zeroFakeMean = np.mean(fpZeroFake,axis = 2)
    #zeroFakeStd = np.std(fpZeroFake,axis = 2)

    #reformat & order by recall values
    #sort rows by first column
    
    oneFake = np.column_stack([meanFP[:,0], meanTP[:,0]])
    twoFake = np.column_stack([meanFP[:,1], meanTP[:,1]])
    thrFake = np.column_stack([meanFP[:,2], meanTP[:,2]])
    
    oneFake = oneFake[np.argsort(oneFake[:, 0])]
    twoFake = twoFake[np.argsort(twoFake[:, 0])]
    thrFake = thrFake[np.argsort(thrFake[:, 0])]

    return oneFake,twoFake,thrFake,stdTP,stdFP

def plot_vs_baselines_acc(ids, window_sizes, threshold, threshold_idx, results, save_dir):

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
 
    # For each method, plot 0-3 fakes
    for method in results:
        curMeans,curStds = plot_acc(ids, window_sizes, threshold, threshold_idx, results[method], save_dir)

        ax0.errorbar(window_sizes, curMeans[0,:]+np.random.normal(0,0.1,5), yerr = curStds[0,:], label = method)
        ax1.errorbar(window_sizes, curMeans[1,:]+np.random.normal(0,0.1,5), yerr = curStds[1,:], label = method)
        ax2.errorbar(window_sizes, curMeans[2,:]+np.random.normal(0,0.1,5), yerr = curStds[2,:], label = method)
        ax3.errorbar(window_sizes, curMeans[3,:]+np.random.normal(0,0.1,5), yerr = curStds[3,:], label = method)


    # Setting all the graph stuff
    ax0.set_xlim([0, max(window_sizes)+10])
    ax0.set_ylim([0, 1.1])
    ax0.set_ylabel('Accuracy')
    ax0.set_xlabel('Window Size')
    ax0.legend(loc = 'lower right')
    ax0.set_title('Detection Accuracy v Window Size, {} Fakes, Threshold = {}'.format(0,threshold))

    ax1.set_xlim([0, max(window_sizes)+10])
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Window Size')
    ax1.legend(loc = 'lower right')
    ax1.set_title('Detection Accuracy v Window Size, {} Fakes, Threshold = {}'.format(1,threshold))

    ax2.set_xlim([0, max(window_sizes)+10])
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Window Size')
    ax2.legend(loc = 'lower right')
    ax2.set_title('Detection Accuracy v Window Size, {} Fakes, Threshold = {}'.format(2,threshold))

    ax3.set_xlim([0, max(window_sizes)+10])
    ax3.set_ylim([0, 1.1])
    ax3.set_ylabel('Accuracy')
    ax3.set_xlabel('Window Size')
    ax3.legend(loc = 'lower right')
    ax3.set_title('Detection Accuracy v Window Size, {} Fakes, Threshold = {}'.format(3,threshold))

    # Saving figures
    fig0.savefig(os.path.join(save_dir, 'acc_plot_{}_fakes_thresh_{}.png'.format(0,threshold)))         
    fig1.savefig(os.path.join(save_dir, 'acc_plot_{}_fakes_thresh_{}.png'.format(1,threshold)))         
    fig2.savefig(os.path.join(save_dir, 'acc_plot_{}_fakes_thresh_{}.png'.format(2,threshold)))         
    fig3.savefig(os.path.join(save_dir, 'acc_plot_{}_fakes_thresh_{}.png'.format(3,threshold)))            

def plot_vs_baselines_ROC(ids, threshes, window_size, results, save_dir):
    
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    # For each method, plot 0-3 fakes
    for method in results:
        oneFake,twoFake,thrFake,stdTP,stdFP = plot_ROC(ids, threshes, window_size, results[method], save_dir)
        
        ax0.errorbar(oneFake[:,0], oneFake[:,1], yerr = stdTP[:,0], xerr = stdFP[:,0], label = method)
        ax1.errorbar(twoFake[:,0], twoFake[:,1], yerr = stdTP[:,1], xerr = stdFP[:,1], label = method)
        ax2.errorbar(thrFake[:,0], thrFake[:,1], yerr = stdTP[:,2], xerr = stdFP[:,2], label = method)

    # Graphing settings
    ax0.set_xlim([0, 1])
    ax0.set_ylim([0, 1])
    ax0.set_ylabel('True Positive Rate')
    ax0.set_xlabel('False Positive Rate')
    ax0.legend(loc = 'lower right')
    ax0.set_title('ROC Curve, {} Fakes, Window size = {}'.format(1, window_size))

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlabel('False Positive Rate')
    ax1.legend(loc = 'lower right')
    ax1.set_title('ROC Curve, {} Fakes, Window size = {}'.format(2, window_size))

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xlabel('False Positive Rate')
    ax2.legend(loc = 'lower right')
    ax2.set_title('ROC Curve, {} Fakes, Window size = {}'.format(3, window_size))
 
    # Save plots
    fig0.savefig(os.path.join(save_dir, 'roc_plot_{}_fakes_window_size_{}.png'.format(1,window_size)))
    fig1.savefig(os.path.join(save_dir, 'roc_plot_{}_fakes_window_size_{}.png'.format(2,window_size)))
    fig2.savefig(os.path.join(save_dir, 'roc_plot_{}_fakes_window_size_{}.png'.format(3,window_size)))

def main():
    
    args = parse_args()
    
    # Populate list of IDs for the current data
    ids = set()
    for f in os.listdir(args.data_dir):
        try:
            x = re.search(r"([0-9]*)-ID([0-9]*).mat", f)
            ID = int(x.group(2))
            ids.add(ID)
        except AttributeError:
            continue
        
    exclude_list  = [17]
    ids = [i for i in ids if i not in exclude_list] 
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    threshes = args.thresholds
    window_sizes = args.window_sizes
    
    window_size = 60
    threshold = 1.5
    threshold_idx = 1

    # Change dictionary entry titles depending on your baselines/baseline order input
    results = {'Ours':args.results_dir, 'Baseline 1': args.results_dir_baseline1, 'Baseline 2': args.results_dir_baseline2}

    plot_vs_baselines_acc(ids, window_sizes, threshold, threshold_idx, results, args.save_dir)
    plot_vs_baselines_ROC(ids, threshes, window_size, results, args.save_dir)      
        

if __name__ == "__main__":
    main()
