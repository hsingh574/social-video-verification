import os
import numpy as np
import scipy.io as sio
import time
import argparse

parser = argparse.ArgumentParser(description='Convert the .npy embedding files for all cameras into a single .mat file for testing.')
parser.add_argument('--input_dir_fake', help='a path to a folder containing identity embeddings in the format identity_embedding-0001.npy ... identity_embedding-XXXX.npy')
parser.add_argument('--input_dir_real', help='a path containing folders formatted as cam{i}-identity_embeddings which have identity embedding .npy files inside')
parser.add_argument('--output_dir', help='the path where the final .mat final will be placed')
parser.add_argument('--num_cams', type=int, help='the number of real viewpoint cameras (this means NOT including the faked viewpoint)', default=6)
parser.add_argument('--faked_cam', type=int, help='which camera was faked (i.e 1,2, or 3)')
parser.add_argument('--num_frames', type=int, help='the number of frames to process')
parser.add_argument('--shift', type=int, default=0, help='the frame to start processing at')
parser.add_argument('--ID', type=int, help='the ID of the data being used')
args = parser.parse_args()

# Create output directory if it does not already exist
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# Generate .mat file from the .npy files
embedding_size = 512
dataMat = np.zeros((args.num_frames, embedding_size, args.num_cams+1))

# Calculate data matrix of identity embeddings for each camera
for i in range(1, args.num_cams+1+1): # Number of true cameras plus one fake camera
    print("Processing camera", i)
    means = np.zeros((embedding_size, args.num_frames))

    # Get identity embedding per frame
    for f in range(1, args.num_frames+1):
        # Load data
        if i == args.num_cams+1:
            data = np.load(os.path.join(\
                    args.input_dir_fake, "identity_embedding-" + "{0:04d}".format(f+args.shift) + ".npy"), allow_pickle=True)
        else:
            data = np.load(os.path.join(\
                    args.input_dir_real, "cam{}-identity_embeddings".format(i), "identity_embedding-" + "{0:04d}".format(f+args.shift) + ".npy"), allow_pickle=True) 
        means[:,f-1] = np.squeeze(np.reshape(data, (-1, 1)))
    
    # Need the data to be in format where observations are rows, vars are columns
    camData = means.T

    # Normalize data for camera i, which is 0-indexed is i-1
    camMean = np.mean(camData, axis=0)
    camStd = np.std(camData, axis=0)

    dataMat[:,:,i-1] = (camData - camMean) / camStd

# Save data matrix as .mat
mat_dictionary = {}
# Add each real camera to the dictionary
for i in range(1, args.num_cams+1):
    mat_dictionary["cam{}".format(i)] = dataMat[:,:,i-1]
# Add the fake camera to the dictionary
mat_dictionary["fake"] = dataMat[:,:,args.num_cams]
sio.savemat(os.path.join(args.output_dir, "fake{}-ID{}.mat".format(args.faked_cam, args.ID)), mat_dictionary)

print("Done! Saved output to:", os.path.join(args.output_dir, "fake{}-ID{}.mat".format(args.faked_cam, args.ID)))


