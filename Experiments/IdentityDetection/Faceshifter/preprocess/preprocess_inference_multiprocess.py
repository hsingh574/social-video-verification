import os
import PIL
import dlib
import random
import argparse
import numpy as np
import scipy.ndimage
import multiprocessing as multiprocessing

from PIL import Image, ImageDraw
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--frames_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()


def draw_overlay(img_file, quad, fn):
    original_img = PIL.Image.open(img_file)
    original_img = original_img.convert('RGBA')

    TINT_COLOR = (255,255,255)
    TRANSPARENCY = 0.25
    OPACITY = int(255 * TRANSPARENCY)

    overlay = Image.new('RGBA', original_img.size, TINT_COLOR+(0,))
    draw = ImageDraw.Draw(overlay)
    
    quad_adj = quad + 0.5
    quad_adj = quad_adj.flatten()
    points = quad_adj.astype('int32')

    draw.polygon(list(points), fill=TINT_COLOR+(OPACITY,))

    quad_img = Image.alpha_composite(original_img, overlay)
    quad_img.convert("RGB")
    quad_img.save("temp/"+fn)


output_size = 256
transform_size=4096
enable_padding=True
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

torch.backends.cudnn.benchmark = False

os.makedirs(args.output_dir, exist_ok=True)
img_files = [
            ("cropped_"+filename, os.path.join(path, filename))
            for path, dirs, files in os.walk(args.frames_dir)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]

img_files.sort()

bounding_boxes = np.zeros((len(img_files), 8))

def preprocess_img(tup):
    idx, (fn, img_file) = tup
    output_img = os.path.join(args.output_dir, fn)
    
    img = dlib.load_rgb_image(img_file)
    dets = detector(img, 1)
    if len(dets) <= 0:
        img = Image.new('RGB', (output_size, output_size), color=0)
        img.save(output_img)
        return [-1,-1,-1,-1,-1,-1,-1,-1]
    else:
        shape = sp(img, dets[0])
        points = np.empty([68, 2], dtype=int)
        for b in range(68):
            points[b, 0] = shape.part(b).x
            points[b, 1] = shape.part(b).y
        lm = points
    # lm = fa.get_landmarks(input_img)[-1]
    # lm = np.array(item['in_the_wild']['face_landmarks'])
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = PIL.Image.open(img_file)
    img = img.convert('RGB')

    #draw_overlay(img_file, quad, 'beginning_quad.png')

    # Save bounding box information
    bounding_box = quad.flatten()
    
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    #draw_overlay(img_file, quad, 'shrink_quad.png')

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    
    #draw_overlay(img_file, quad, 'crop_quad.png')

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    #draw_overlay(img_file, quad, 'pad_quad.png')

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    
    #draw_overlay(img_file, quad, 'transform_quad.png')
    # Save aligned image.
    img.save(output_img)
    return bounding_box

bounding_boxes = process_map(preprocess_img, list(enumerate(img_files)), max_workers=os.cpu_count()-2)
bounding_boxes = np.stack(bounding_boxes)

num_skipped = np.sum((bounding_boxes == -1).astype('int32'))//8
print('Skipped', str(num_skipped), 'frames because faces weren\'t detected.')

np.save(os.path.join(args.output_dir,"bounding_boxes.npy"), np.stack(bounding_boxes))

