import insightface
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import shutil
import os
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Calculate the embedding for the face in each frame and save them to .npy files.')
parser.add_argument('--input_dir', help='the ID folder to process')
#parser.add_argument('--output_dir', help='place to save the .npy files which are formatted as identity_embedding-XXXX.npy')
args = parser.parse_args()

def process_frame():
    pass

# Prepare model
model = insightface.app.FaceAnalysis()
ctx_id = 0
model.prepare(ctx_id=ctx_id, nms=0.4)

# Iterate through each frame and find the embedding of its face then save it in a .npy file
for category, idx in zip((['real']*7)+(['fake']*3), list(range(0, 7)) + list(range(1, 6, 2))):

    frames_path = os.path.join(args.input_dir, category, 'cam'+str(idx), 'frames') 
    output_path = os.path.join(args.input_dir, category, 'cam'+str(idx), 'identity_embeddings')

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    num_frames = len([name for name in os.listdir(frames_path) if os.path.isfile(os.path.join(frames_path, name))])

    for i in tqdm(range(1, num_frames+1)):
        img = np.asarray(Image.open(frames_path + "/frames" + "{0:04d}".format(i) + ".jpg"))
        output = model.get(img)
        if len(output) > 0:
            face = output[0]
            embedding = face.embedding
            np.save(os.path.join(output_path, 'identity_embedding-' + '{0:04d}'.format(i) + '.npy'), embedding)


















