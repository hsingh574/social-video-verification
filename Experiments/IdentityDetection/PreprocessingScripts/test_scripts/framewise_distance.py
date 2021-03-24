import matplotlib
matplotlib.use('Agg')

import insightface
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser(description='Create a plot of the difference between each frame\'s identity embedding and the mean embedding over the given window.')
parser.add_argument('--path_view_1_real', help='a path to a folder containing frames in the format frames0001.jpg ... framesXXXX.jpg')
parser.add_argument('--path_view_1_fake', help='a path to a folder containing frames in the format frames0001.jpg ... framesXXXX.jpg')
parser.add_argument('--path_view_2', help='a path to a folder containing frames in the format frames0001.jpg ... framesXXXX.jpg')
parser.add_argument('--num_frames', type=int, help='the number of frames to process')
parser.add_argument('--shift', type=int, default=0)
args = parser.parse_args()

videos = []
videos.append([args.path_view_1_real + "/frames" + "{0:04d}".format(i) + ".jpg" for i in range(1, args.num_frames + 1)])
videos.append([args.path_view_1_fake + "/frames" + "{0:04d}".format(i) + ".jpg" for i in range(1, args.num_frames + 1)])
videos.append([args.path_view_2 + "/frames" + "{0:04d}".format(i) + ".jpg" for i in range(1, args.num_frames + 1)])

# Prepare model
model = insightface.app.FaceAnalysis()
ctx_id = 0
model.prepare(ctx_id=ctx_id, nms=0.4)

# Keep track of the distances between the embeddings in corresponding frames from each view  
distances_real = []
distances_fake = []
frames = []

# Iterate through each frame and find the embeddings of the faces in both videos, comparing them
# with euclidean distance
for i in range(args.shift, args.num_frames+args.shift):
    print("Processing frames ", i)
    img = np.asarray(Image.open(videos[0][i]))
    
    output = model.get(img)

    if len(output) <= 0:
        continue

    face = output[0]
    embedding_view_1_real = face.embedding

    img = np.asarray(Image.open(videos[1][i]))
    
    output = model.get(img)

    if len(output) <= 0:
        continue

    face = output[0]
    embedding_view_1_fake = face.embedding

    img = np.asarray(Image.open(videos[2][i]))
    
    output = model.get(img)
    
    if len(output) <= 0:
        continue

    face = output[0]
    embedding_view_2 = face.embedding

    distances_real.append(np.linalg.norm(embedding_view_1_real - embedding_view_2))
    distances_fake.append(np.linalg.norm(embedding_view_1_fake - embedding_view_2))
    frames.append(i)

# Plot the data

fig, ax = plt.subplots()
fig.suptitle('Euclidean distance between viewpoints')
ax.plot(frames, distances_real, color='blue', label='Real vs. Real')
ax.plot(frames, distances_fake, color='red', label='Fake vs. Real')
ax.set(xlabel='Frame', ylabel='L2', title='L2 Across view points')
ax.grid()
ax.legend()
fig.savefig('framewise_distance.jpg')
plt.close()



















