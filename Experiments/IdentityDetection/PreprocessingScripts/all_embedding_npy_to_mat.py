import os
import numpy as np
import scipy.io as sio
import time
import argparse
import re
import sys

parser = argparse.ArgumentParser(description='Convert the .npy embedding files for all cameras into a single .mat file for testing.')
parser.add_argument('--root', help='the ID folder')
parser.add_argument('--num_cams', type=int, help='the number of real viewpoint cameras (this means NOT including the faked viewpoint)', default=7)
parser.add_argument('--faked_cams', metavar='N', type=int, nargs='+')
parser.add_argument('--experiment_dir', help='place to save the mat files for use.')
parser.add_argument('--ID', type=int, help='the ID of the data being used')
args = parser.parse_args()

# Get number of frames and the shift

#for category, idx in zip((['real']*args.num_cams)+(['fake']*len(args.faked_cams)), list(range(1, args.num_cams+1)) + args.faked_cams):
#    print(category + ' ' + str(idx))

fake_npy_paths = [os.path.join(args.root, 'fake', 'cam'+str(i), 'identity_embeddings') for i in args.faked_cams]
real_npy_paths = [os.path.join(args.root, 'real', 'cam'+str(i), 'identity_embeddings') for i in range(0, args.num_cams)]

fakes = []
for path in fake_npy_paths:
    fakes_npy = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    fakes_npy.sort()
    fakes.append(fakes_npy)

reals = []
for path in real_npy_paths:
    reals_npy = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name))]
    reals_npy.sort()
    reals.append(reals_npy)

shift = 0
num_frames = sys.maxsize
for l in fakes+reals:
    shift = max(int(re.search('\d+', l[0]).group()), shift)
    num_frames = min(len(l), num_frames)

print('Shift:', shift)
print('Frames:', num_frames)

embedding_size = 512

for faked_cam in args.faked_cams:
    # Generate .mat file from the .npy files
    dataMat = np.zeros((num_frames, embedding_size, args.num_cams+1))
    print('Processing fake ', faked_cam)
    # Calculate data matrix of identity embeddings for each camera
    for i in range(0, args.num_cams+1): # Number of true cameras plus one fake camera
        print("Processing camera", i)
        means = np.zeros((embedding_size, num_frames))

        # Get identity embedding per frame
        for f in range(0, num_frames):
            # Load data
            if i == args.num_cams:
                input_dir = os.path.join(args.root, 'fake', 'cam'+str(faked_cam), 'identity_embeddings')
                if not os.path.exists(os.path.join(input_dir, "identity_embedding-" + "{0:04d}".format(f+shift) + ".npy")):
                    data = means[:, f-1]
                else:
                    data = np.load(os.path.join(input_dir, "identity_embedding-" + "{0:04d}".format(f+shift) + ".npy"), allow_pickle=True)
            else:
                input_dir = os.path.join(args.root, 'real', 'cam'+str(i), 'identity_embeddings')
                if not os.path.exists(os.path.join(input_dir, "identity_embedding-" + "{0:04d}".format(f+shift) + ".npy")):
                    data = means[:, f-1]
                else:
                    data = np.load(os.path.join(input_dir, "identity_embedding-" + "{0:04d}".format(f+shift) + ".npy"), allow_pickle=True) 
            means[:,f] = np.squeeze(np.reshape(data, (-1, 1)))
    
        # Need the data to be in format where observations are rows, vars are columns
        camData = means.T

        # Normalize data for camera i, which is 0-indexed is i-1
        #camMean = np.mean(camData, axis=0)
        #camStd = np.std(camData, axis=0)

        dataMat[:,:,i] = camData#(camData - camMean) / camStd

    # Save data matrix as .mat
    mat_dictionary = {}
    # Add each real camera to the dictionary
    for i in range(0, args.num_cams):
        mat_dictionary["cam{}".format(i)] = dataMat[:,:,i]
    # Add the fake camera to the dictionary
    mat_dictionary["fake"] = dataMat[:,:,args.num_cams]
    output_dir = os.path.join(args.root, 'fake', 'cam'+str(faked_cam))
    sio.savemat(os.path.join(output_dir, "fake{}-ID{}.mat".format(faked_cam, args.ID)), mat_dictionary)
    sio.savemat(os.path.join(args.experiment_dir, "fake{}-ID{}.mat".format(faked_cam, args.ID)), mat_dictionary) 
    print("Done! Saved output to:", os.path.join(output_dir, "fake{}-ID{}.mat".format(faked_cam, args.ID)))


