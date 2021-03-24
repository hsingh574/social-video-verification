import insightface
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser(description='Create a plot of the difference between each frame\'s identity embedding and the mean embedding over the given window.')
parser.add_argument('--path', help='a path to a folder containing frames in the format frames0001.jpg ... framesXXXX.jpg')
parser.add_argument('--num_frames', type=int, help='the number of frames to process')
args = parser.parse_args()

frames = [args.path + "/frames" + "{0:04d}".format(i) + ".jpg" for i in range(1, args.num_frames + 1)]

# Prepare model
model = insightface.app.FaceAnalysis()
ctx_id = 0
model.prepare(ctx_id=ctx_id, nms=0.4)

# Keep a running sum total of the embeddings of each frame to calculate the mean at the end
# and keep track of the embeddings themselves 
total = np.zeros(512)
embeddings = []

# Iterate through each frame and find the embedding of the face
for i in range(0, args.num_frames):
    print("Processing frame ", i)
    img = np.asarray(Image.open(frames[i]))
    face = model.get(img)[0]
    embedding = face.embedding
    embeddings.append(embedding)
    total += embedding

# Find the mean of the embeddings 
mean = total / args.num_frames
distances = []

# Calculate the L2 distance between each embedding and the mean and plot this value for each frame
for i in range(0, args.num_frames):
    print("Calculating L2 distance for frame ", i)
    distance = np.linalg.norm(embeddings[i] - mean)
    distances.append(distance)

# Plot the data
frames = list(range(0, args.num_frames))

fig, ax = plt.subplots()
ax.plot(frames, distances)
ax.set(xlabel='Frame', ylabel='L2', title='Euclidean distance from mean embedding across frames')
ax.grid()
fig.savefig('mean_difference.jpg')
plt.close()



















