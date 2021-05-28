import argparse
import os
import re
from more_itertools.more import interleave
import networkx as nx
import itertools
import more_itertools
import numpy as np
import scipy.stats as st
from dip import diptst
from unidip import *
from scipy.io import loadmat, savemat
from collections import defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm

from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')

    parser.add_argument('--data-dir', type=str, default='Data',
                    help='Directory where processed landmark files live')
    parser.add_argument('--save-dir', type=str, default='Results',
                    help='Directory to save results')
    parser.add_argument("--num-cams", type=int, default=6)
    parser.add_argument('--zero-start', action='store_false',
                    help='Whether or not there is a cam0')
    
    args = parser.parse_args()
    return args

def get_cams(data, num_cams, fullLen, zero_start):
    cams_list = []
    if zero_start:
        for i in range(num_cams+1):
            cams_list.append(data['cam{}'.format(i)][:fullLen,:])
    else:
        for i in range(1, num_cams+1):
            cams_list.append(data['cam{}'.format(i)][:fullLen,:])
    
    return cams_list

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

def gen_results(ID, fake_cams, data_dir, num_cams, zero_start, save_dir):
    data0 = loadmat(os.path.join(data_dir, "fake{}-ID{}.mat".format(fake_cams[0], ID))) 
    data1 = loadmat(os.path.join(data_dir, "fake{}-ID{}.mat".format(fake_cams[1], ID)))
    data2 = loadmat(os.path.join(data_dir, "fake{}-ID{}.mat".format(fake_cams[2], ID)))

    num_frames = min(data0['cam1'].shape[0], data1['cam1'].shape[0], data2['cam1'].shape[0])

    fake0 = data0['fake'][:num_frames,:]
    fake1 = data1['fake'][:num_frames,:]
    fake2 = data2['fake'][:num_frames,:]
    
    #Get the non-faked camera views. Arbitrarily pick data1
    #because the non-faked views should be the same across all 
    #files for a given ID    
    cams = get_cams(data1, num_cams, num_frames, zero_start)

    X0, X1, X2, X3 = build_test_arrays(cams, fake0, fake1, fake2)

    if zero_start:
        U = set(range(num_cams+1))
    else:
        U = set(range(num_cams))
    
    L0 = {}
    L1 = {3}
    L2 = {2,3}
    L3 = {1,2,3}

    correct = [
        graph_cut_partition(num_frames, X0, L0, U),
        graph_cut_partition(num_frames, X1, L1, U),
        graph_cut_partition(num_frames, X2, L2, U),
        graph_cut_partition(num_frames, X3, L3, U),
    ]

    total = num_frames

    return correct, total
    
def graph_cut_partition(num_frames, X, L, U):
    correct = 0
    for frame in range(num_frames): #tqdm(range(num_frames)):
        G = nx.Graph()
        G.add_nodes_from(U)

        # Find the mean and standard deviation of the l2 distances between all the 
        # cameras. We will use this to find the probablity that each l2 value could have
        # been greater than or equal to that value.

        l2s = []
        for (i, j) in itertools.combinations(U, 2):
            l2 = np.linalg.norm(X[i, frame, :] - X[j, frame, :], axis = 0)
            l2s.insert(0, l2)

        # Calculate the probability that the distribution of l2s is unimodal. When there are 0
        # fakes we expect the distribution to be a single normal curve. When there are some number
        # of fakes we expect to see another area of density to the far right which represents the
        # distribution of l2s for edges connecting real and fake nodes.

        dat = sorted(l2s)
        _, pval, _ = diptst(dat, False, 100)
        intervals = UniDip(dat, alpha=0.25).run()

        if intervals:
            leftmost_dist = intervals[0]
            l2s_slice = l2s[leftmost_dist[0]:leftmost_dist[1]]
            mean = np.mean(l2s_slice)
            std = np.std(l2s_slice) + 0.00001
        else:
            mean = np.mean(l2s)
            std = np.std(l2s)

        #plt.hist(sorted(l2s), 20, facecolor='blue', alpha=0.5)
        #plt.show()

        # Generate the edge weights for the graph.

        for (i, j) in itertools.combinations(U, 2):
            l2 = l2s.pop()
            z = (l2 - mean) / std

            # Probability that this l2 value could have taken on its value or greater assuming its
            # in the leftmost distribution

            prob = 1 - st.norm.cdf(z) + 0.00001
            weighted_l2 = (1/prob**0.5)**4

            G.add_edge(i, j, weight=weighted_l2)

        # Find the max cut of the graph to partition the nodes.

        cuts = []
        max_cut_value = -1
        partition = None
        for p, c in more_itertools.set_partitions(U, 2):
            cut_value = 0.0
            for i in p:
                for j in c:
                    cut_value += G[i][j]['weight']
            cuts.append(cut_value)
            if cut_value > max_cut_value:
                max_cut_value = cut_value
                partition = p

        #plt.hist(sorted(cuts), 20, facecolor='blue', alpha=0.5)
        #plt.show()

        partition = set(partition)

        # Assuming the values of the cuts are normally distributed, determine the probability
        # that the max cut would take on the its value or higher.

        z = (max_cut_value - np.mean(cuts)) / np.std(cuts)
        cut_prob = 1 - st.norm.cdf(z)

        if cut_prob*pval > 0.01:
            partition = {}

        # Determine if our answer was correct.

        if partition == L or partition == U.difference(L):
            correct += 1
    return correct

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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(len(fake_cams_dict.keys()))

    """
    correct = [0,0,0,0]
    total = 0
    for id in tqdm(ids):
        c, t = gen_results(id, fake_cams_dict[id], args.data_dir, args.num_cams, args.zero_start)
        for i in range(len(c)):
            correct[i] += c[i]
        total += t
    print(correct[0]/total, correct[1]/total, correct[2]/total, correct[3]/total)
    """
    corrects, totals = zip(*Parallel(n_jobs=-1)(delayed(gen_results)(
        id, fake_cams_dict[id], args.data_dir, args.num_cams, args.zero_start, args.save_dir) for id in ids))
    
    correct = [0,0,0,0]
    total = 0
    for c, t in zip(corrects, totals):
        for i in range(len(c)):
            correct[i] += c[i]
        total += t
    print(correct[0]/total, correct[1]/total, correct[2]/total, correct[3]/total)

if __name__ == "__main__":
    main()


 