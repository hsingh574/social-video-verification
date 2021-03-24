#!/usr/bin/env python3
# Script to save a ROC and accuracy plot per 0-3 fakes across baselines
# Eg for how to run:
# python3 make_comparison_plots_for_paper.py --data-dir ./dataset/ --results-dir ./results_ours/ --results-dir-base1 ./results_base1/ --results-dir-base2 ./results_base2/ --save-dir ./out/

import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_curve
import re

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')
    
    parser.add_argument('--data-dir', type=str, default='Data',
                    help='Directory where data has been saved')
    
    parser.add_argument('--results-dir', type=str, default='Adam method Results',
                    help='Directory where results have been saved')
    parser.add_argument('--thresholds-main', nargs="+", default=[1.2, 1.3, 1.5, 1.7, 1.9, 2.1, 2.5])
    parser.add_argument('--threshold-idx-main', type=int, default=6)

    parser.add_argument('--results-dir-base1', type=str, default='Mean feature method Results',
                    help='Directory where baseline 1 results have been saved')
    parser.add_argument('--thresholds-base1', nargs="+", default=[1.2, 1.3, 1.5, 1.7, 1.9, 2.1, 2.5])
    parser.add_argument('--threshold-idx-base1', type=int, default=2)

    parser.add_argument('--results-dir-base2', type=str, default='PCA method Results',
                    help='Directory where baseline 2 results have been saved')
    parser.add_argument('--thresholds-base2', nargs="+", default=[1.2, 1.3, 1.5, 1.7, 1.9, 2.1, 2.5])
    parser.add_argument('--threshold-idx-base2', type=int, default=2)

    parser.add_argument('--save-dir', type=str, default='Comparison Plots',
                    help='Directory to save results plots')

    parser.add_argument("--window-sizes", nargs="+", default=[10,20,30,40,50,60])
    
    
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

    return fpr1, tpr1, fpr2, tpr2, fpr3, tpr3

def plot_vs_baselines_acc(ids, window_sizes, thresholds, threshold_idx, results, save_dir):

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
 
    # For each method, plot 0-3 fakes
    for method in results:
        curMeans,curStds = plot_acc(ids, window_sizes, thresholds[method][threshold_idx[method]], threshold_idx[method], results[method], save_dir)

        ax0.errorbar(window_sizes, curMeans[0,:], yerr = curStds[0,:], label = method, elinewidth=0.5, capsize=1)
        ax1.errorbar(window_sizes, curMeans[1,:], yerr = curStds[1,:], label = method, elinewidth=0.5, capsize=1)
        ax2.errorbar(window_sizes, curMeans[2,:], yerr = curStds[2,:], label = method, elinewidth=0.5, capsize=1)
        ax3.errorbar(window_sizes, curMeans[3,:], yerr = curStds[3,:], label = method, elinewidth=0.5, capsize=1)


    # Setting all the graph stuff
    ax0.set_xlim([0, max(window_sizes)+10])
    ax0.set_ylim([0, 1.1])
    ax0.set_ylabel('Accuracy')
    ax0.set_xlabel('Window Size')
    ax0.legend(loc = 'lower right')
    ax0.set_title('Detection Accuracy v Window Size, {} Fakes'.format(0))

    ax1.set_xlim([0, max(window_sizes)+10])
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Window Size')
    ax1.legend(loc = 'lower right')
    ax1.set_title('Detection Accuracy v Window Size, {} Fakes'.format(1))

    ax2.set_xlim([0, max(window_sizes)+10])
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Window Size')
    ax2.legend(loc = 'lower right')
    ax2.set_title('Detection Accuracy v Window Size, {} Fakes'.format(2))

    ax3.set_xlim([0, max(window_sizes)+10])
    ax3.set_ylim([0, 1.1])
    ax3.set_ylabel('Accuracy')
    ax3.set_xlabel('Window Size')
    ax3.legend(loc = 'lower right')
    ax3.set_title('Detection Accuracy v Window Size, {} Fakes'.format(3))

    # Saving figures
    fig0.savefig(os.path.join(save_dir, 'acc_plot_{}_fakes.png'.format(0)))         
    fig1.savefig(os.path.join(save_dir, 'acc_plot_{}_fakes.png'.format(1)))         
    fig2.savefig(os.path.join(save_dir, 'acc_plot_{}_fakes.png'.format(2)))         
    fig3.savefig(os.path.join(save_dir, 'acc_plot_{}_fakes.png'.format(3)))            

def plot_vs_baselines_ROC(ids, threshes, window_size, results, save_dir):
    
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    # For each method, plot 0-3 fakes
    for method in results:
        fpr1, tpr1, fpr2, tpr2, fpr3, tpr3 = plot_ROC_v2(ids, threshes, window_size, results[method], save_dir)
        
        ax0.plot(fpr1, tpr1, label = method)
        ax1.plot(fpr2, tpr2, label = method)
        ax2.plot(fpr3, tpr3, label = method)


    # Graphing settings
    ax0.set_xlim([0, 1])
    ax0.set_ylim([0, 1.1])
    ax0.set_ylabel('True Positive Rate')
    ax0.set_xlabel('False Positive Rate')
    ax0.legend(loc = 'lower right')
    ax0.set_title('ROC Curve, {} Fakes, Window size = {}'.format(1, window_size))

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlabel('False Positive Rate')
    ax1.legend(loc = 'lower right')
    ax1.set_title('ROC Curve, {} Fakes, Window size = {}'.format(2, window_size))

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.1])
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
        
    window_sizes = args.window_sizes
    
    window_size = 20
    thresholds = {'Ours':args.thresholds_main, 'Mean feature': args.thresholds_base1, 'PCA': args.thresholds_base2}
    threshold_idx = {'Ours':args.threshold_idx_main, 'Mean feature': args.threshold_idx_base1, 'PCA': args.threshold_idx_base2}

    # Change dictionary entry titles depending on your baselines/baseline order input
    results = {'Ours':args.results_dir, 'Mean feature': args.results_dir_base1, 'PCA': args.results_dir_base2}
    thresholds

    plot_vs_baselines_acc(ids, window_sizes, thresholds, threshold_idx, results, args.save_dir)
    plot_vs_baselines_ROC(ids, thresholds, window_size, results, args.save_dir)      
        

if __name__ == "__main__":
    main()
