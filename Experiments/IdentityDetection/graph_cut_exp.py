import argparse
import os
import re
import sys
import math
import networkx as nx
from numpy.core.numeric import full
import dimod
import dwave_networkx as dnx
import itertools
import more_itertools
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from collections import defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')

    parser.add_argument('--data-dir', type=str, default='Data',
                    help='Directory where processed landmark files live')
    parser.add_argument('--save-dir', type=str, default='Results',
                    help='Directory to save results')
    parser.add_argument("--num-cams", type=int, default=7)
    
    args = parser.parse_args()
    return args

def get_cams(data, num_cams, fullLen):
    cams_list = []
    for i in range(num_cams):
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

def gen_results(ID, fake_cams, data_dir, num_cams):
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
    cams = get_cams(data1, num_cams, num_frames)

    X0, X1, X2, X3 = build_test_arrays(cams, fake0, fake1, fake2)

    U = set(range(num_cams))
    L0 = {}
    L1 = {3}
    L2 = {2,3}
    L3 = {1,2,3}

    correct = [
        graph_cut_partition(num_frames, num_cams, X0, L0, U),
        graph_cut_partition(num_frames, num_cams, X1, L1, U),
        graph_cut_partition(num_frames, num_cams, X2, L2, U),
        graph_cut_partition(num_frames, num_cams, X3, L3, U),
    ]

    total = num_frames

    return correct, total
    
def graph_cut_partition(num_frames, num_cams, X, L, U):
    correct = 0
    for frame in range(num_frames):
        G = nx.Graph()
        G.add_nodes_from(U)

        # Find the mean and standard deviation of the l2 distances between all the 
        # cameras. We will use this to find the probablity that each l2 value could have
        # been greater than or equal to that value.

        l2s = []
        for (i, j) in itertools.combinations(U, 2):
            l2 = np.linalg.norm(X[i, frame, :] - X[j, frame, :], axis = 0)
            l2s.insert(0, l2)
        mean = np.mean(l2s)
        std = np.std(l2s)

        # Weight each l2 value according to the its probablity and add its value to the
        # appropriate edge in the graph.

        for (i, j) in itertools.combinations(U, 2):
            l2 = l2s.pop()
            z = (l2 - mean) / std
            prob = 1 - st.norm.cdf(z)
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

        partition = set(partition)

        # Assuming the values of the cuts are normally distributed, determine the probability
        # that the max cut would take on the its value or higher. If this probability is greater
        # than 0.025 (that the cut doesn't seem too extraordinary), we assume that we are looking
        # at the 0 fake case.

        z = (max_cut_value - np.mean(cuts)) / np.std(cuts)
        cut_prob = 1 - st.norm.cdf(z)

        if cut_prob > 0.025:
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

    correct = [0,0,0,0]
    total = 0
    for id in tqdm(ids):
        c, t = gen_results(id, fake_cams_dict[219], args.data_dir, args.num_cams)
        for i in range(len(c)):
            correct[i] += c[i]
        total += t
    print(correct[0]/total, correct[1]/total, correct[2]/total, correct[3]/total)

if __name__ == "__main__":
    main()


 