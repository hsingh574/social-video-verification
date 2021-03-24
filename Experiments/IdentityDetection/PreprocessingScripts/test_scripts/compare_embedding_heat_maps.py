import matplotlib
matplotlib.use('Agg')

import insightface
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from PIL import Image

# Frames to ccompare
fake_frame_1 = Image.open('/media/eleanor/New-Volume/identity-data/ID908/fake/cam3/frames/frames0010.jpg')
fake_frame_2 = Image.open('/media/eleanor/New-Volume/identity-data/ID908/fake/cam3/frames/frames0100.jpg')

real_frame_1 = Image.open('/media/eleanor/New-Volume/identity-data/ID908/real/cam3/frames/frames0010.jpg')
real_frame_2 = Image.open('/media/eleanor/New-Volume/identity-data/ID908/real/cam3/frames/frames0100.jpg')

# Load Face Model
model = insightface.app.FaceAnalysis()
ctx_id = 0
model.prepare(ctx_id = ctx_id, nms = 0.4)

# Get embeddings for each frame
fake_frame_1_face = model.get(np.asarray(fake_frame_1))[0]
fake_frame_2_face = model.get(np.asarray(fake_frame_2))[0]

real_frame_1_face = model.get(np.asarray(real_frame_1))[0]
real_frame_2_face = model.get(np.asarray(real_frame_2))[0]

# Construct figure
f, axarr = plt.subplots(2, 4)
f.suptitle('Visualizing embeddings')
axarr[0, 0].imshow(fake_frame_1.crop(fake_frame_1_face.bbox.astype(np.int).flatten()))
axarr[0, 0].set_axis_off()
axarr[1, 0].pcolor(fake_frame_1_face.embedding.reshape((32, 16)), cmap=plt.cm.hot)
axarr[1, 0].set_axis_off()

axarr[0, 1].imshow(fake_frame_2.crop(fake_frame_2_face.bbox.astype(np.int).flatten()))
axarr[0, 1].set_axis_off()
axarr[1, 1].pcolor(fake_frame_2_face.embedding.reshape((32, 16)), cmap=plt.cm.hot)
axarr[1, 1].set_axis_off()

axarr[0, 2].imshow(real_frame_1.crop(real_frame_1_face.bbox.astype(np.int).flatten()))
axarr[0, 2].set_axis_off()
axarr[1, 2].pcolor(real_frame_1_face.embedding.reshape((32, 16)), cmap=plt.cm.hot)
axarr[1, 2].set_axis_off()

axarr[0, 3].imshow(real_frame_2.crop(real_frame_2_face.bbox.astype(np.int).flatten()))
axarr[0, 3].set_axis_off()
axarr[1, 3].pcolor(real_frame_2_face.embedding.reshape((32, 16)), cmap=plt.cm.hot)
axarr[1, 3].set_axis_off()

plt.savefig('heatmap_comparison.jpg', pad_inches=0)
plt.close()




