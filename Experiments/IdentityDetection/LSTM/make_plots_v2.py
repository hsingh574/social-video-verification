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
    
    parser.add_argument('--data-dir', type=str, default='Data',
                    help='Directory where data has been saved')
    
    parser.add_argument('--results-dir', type=str, default='Results',
                    help='Directory where results have been saved')
    
    parser.add_argument('--save-dir', type=str, default='Plots',
                    help='Directory to save results plots')
   
    parser.add_argument("--window-sizes", nargs="+", default=[30])
    parser.add_argument("--title", type=str, required=True)
    
    
    args = parser.parse_args()
    return args


def plot_ROC_v2(window_size, results_dir, save_dir, title):
    y_true_0 = None
    y_score_0 = None

    y_true_1 = None
    y_score_1 = None

    y_true_2 = None
    y_score_2 = None

    y_true_3 = None
    y_score_3 = None
    try:
        results = loadmat(os.path.join(results_dir, "p_window_{}.mat".format(window_size)))
        y_score_0 = results['p0'][0, :]
        y_true_0 = results['p0'][1, :]

        y_score_1 = results['p1'][0, :]
        y_true_1 = results['p1'][1, :]

        y_score_2 = results['p2'][0, :]
        y_true_2 = results['p2'][1, :]

        y_score_3 = results['p3'][0, :]
        y_true_3 = results['p3'][1, :]
    except FileNotFoundError:
        print("not found")

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
    plt.title('ROC Curve, Window size = {}, {}'.format(window_size, title))
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

def main():
    
    args = parse_args()
    

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.rocOn:
        for window_size in args.window_sizes:
            plot_ROC_v2(window_size, args.results_dir, args.save_dir, args.title)


if __name__ == "__main__":
    main()
