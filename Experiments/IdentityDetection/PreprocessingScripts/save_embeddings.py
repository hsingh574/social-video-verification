import insightface
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import shutil
import os
from PIL import Image

parser = argparse.ArgumentParser(description='Calculate the embedding for the face in each frame and save them to .npy files.')
parser.add_argument('--input_dir', help='a path to a folder containing frames in the format frames0001.jpg ... framesXXXX.jpg')
parser.add_argument('--output_dir', help='place to save the .npy files which are formatted as identity_embedding-XXXX.npy')
args = parser.parse_args()

if os.path.isdir(args.output_dir):
    shutil.rmtree(args.output_dir)
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
num_frames = len([name for name in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, name))])

# Prepare model
model = insightface.app.FaceAnalysis()
ctx_id = 0
model.prepare(ctx_id=ctx_id, nms=0.4)

# Iterate through each frame and find the embedding of its face then save it in a .npy file
for i in range(1, num_frames+1):
    print("Processing frame ", i)
    img = np.asarray(Image.open(args.input_dir + "/frames" + "{0:04d}".format(i) + ".jpg"))
    output = model.get(img)
    if len(output) > 0:
        face = output[0]
        embedding = face.embedding
        np.save(os.path.join(args.output_dir, 'identity_embedding-' + '{0:04d}'.format(i) + '.npy'), embedding)

















