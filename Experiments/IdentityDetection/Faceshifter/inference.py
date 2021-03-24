import argparse
from PIL import Image, ImageDraw, ImageFilter
import PIL
from omegaconf import OmegaConf

import dlib
import numpy as np
import scipy, cv2, os, sys
from os import listdir, path
from glob import glob
import string
from tqdm import tqdm
import torch, face_detection
from torchvision import transforms
import math
from aei_net import AEINet

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                    help="path of configuration yaml file")
parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="path of aei-net pre-trained file")
parser.add_argument("--source_image", type=str, required=True,
                    help="path of source face image")
parser.add_argument("--target_dir", type=str, required=True,
                    help="path of folder containing folders for faces and frames.")
parser.add_argument("--output_path", type=str, default="output.mp4",
                    help="path of output video")
parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu")
parser.add_argument("--blending", default=False, action="store_true",
                    help="Blends generated frames onto the original image if set to true.")
parser.add_argument("--fps", default=30, type=int,
                    help="FPS of the target video.")
args = parser.parse_args()
args.image_size = 256

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

def main():

    # Load preprocessed images and frames from folders and bounding box information from np file   
     
    print("Gathering faces, frames, and bounding boxes...")
    faces = [
        os.path.join(args.target_dir, "cropped", filename)
        for _, _, files in os.walk(os.path.join(args.target_dir, "cropped"))
        for filename in files
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
    ]    
    faces.sort()

    frames = [
        os.path.join(args.target_dir, "frames", filename)
        for _, _, files in os.walk(os.path.join(args.target_dir, "frames"))
        for filename in files
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
    ]
    frames.sort()

    boxes = np.load(os.path.join(args.target_dir, "cropped", "bounding_boxes.npy"), allow_pickle=True)

    if len(frames) > len(faces):
        frames = frames[0:len(faces)]

    print("Loading AEINet model...")
    
    hp = OmegaConf.load(args.config)
    model = AEINet(hp=hp)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu")["state_dict"])
    model.eval()
    model.freeze()
    model.to(device)

    source_img = transforms.ToTensor()(Image.open(args.source_image)).unsqueeze(0).to(device)
   
    frame_h, frame_w = cv2.imread(frames[0]).shape[:-1]
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (frame_w, frame_h))
    
    for i, (face, frame, box) in enumerate(tqdm(zip(faces, frames, boxes))):
        # If preprocessing didn't detect a face in this frame, skip it
        if np.all(box == -1):
            out.write(cv2.imread(frame))
            continue

        target_img = transforms.ToTensor()(Image.open(face)).unsqueeze(0).to(device)
        with torch.no_grad():
            output, _, _, _, _ = model.forward(target_img, source_img)
        pred = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))

        if args.blending: 
            x1, y1, x2, y2, x3, y3, x4, y4 = box
            left = min(x1, x2, x3, x4)
            top = min(y1, y2, y3, y4)
 
 
            length = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            p = pred.convert('RGBA')
            p = p.resize((int(length), int(length)), resample=Image.BILINEAR)
          
            f = Image.open(frame)
            f = np.asarray(f)
            
            alphaArr = np.zeros((p.size[0], p.size[1]))
            blends = np.array((0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0))
            blends_len = len(blends)

            # Top
            for col in range(blends_len):
                alphaArr[:, col] = np.repeat(blends[col], p.size[0])
            # Bottom
            i = 0
            for col in reversed(range(p.size[1]-blends_len, p.size[1])):
                alphaArr[:, col] = np.repeat(blends[i], p.size[0])
                i = i+1
            # Left
            for row in range(blends_len):
                alphaArr[row, :] = np.repeat(blends[row], p.size[1])
            # Right
            i = 0
            for row in reversed(range(p.size[0]-blends_len, p.size[0])):
                alphaArr[row, :] = np.repeat(blends[i], p.size[1])
                i = i+1
            
            if left == x2:
                theta = -1*np.arctan((y4-y1)/(x4-x1)) * 180 / np.pi
            elif left == x1:
                theta = -1*np.arctan((y1-y4)/(x1-x4)) * 180 / np.pi
            
            p = p.rotate(theta, resample=Image.BILINEAR, expand=True, fillcolor=(0,0,0,0))
           
            alphaArr = Image.fromarray(alphaArr)
            alphaArr = alphaArr.rotate(theta, expand=True, fillcolor=1, resample=Image.BILINEAR)
            alphaArr = np.asarray(alphaArr)

            x1, y1, x2, y2, x3, y3, x4, y4 = box.astype('int32')
           
            left = min(x1, x2, x3, x4)
            top = min(y1, y2, y3, y4)
 
            p = np.asarray(p) 


            square = f[top:top+p.shape[0], left:left+p.shape[1], :] 
            alphaArr_square = alphaArr[0:square.shape[0], 0:square.shape[1]]
            p_square = p[0:square.shape[0], 0:square.shape[1], :]

            # Blended channels
            pBlendR = (alphaArr_square * square[:,:,0]) + ((1-alphaArr_square) * p_square[:,:,0])
            pBlendG = (alphaArr_square * square[:,:,1]) + ((1-alphaArr_square) * p_square[:,:,1])
            pBlendB = (alphaArr_square * square[:,:,2]) + ((1-alphaArr_square) * p_square[:,:,2])
            
            result = f.copy()           
 
            result[top:top+square.shape[0],left:left+square.shape[1],0] = pBlendR
            result[top:top+square.shape[0],left:left+square.shape[1],1] = pBlendG
            result[top:top+square.shape[0],left:left+square.shape[1],2] = pBlendB
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = box
            
            left = min(x1, x2, x3, x4)
            top = min(y1, y2, y3, y4)
          
            p = pred.convert('RGBA')
            f = Image.open(frame)
 
            if left == x2:
                theta = -1*np.arctan((y4-y1)/(x4-x1)) * 180 / np.pi
            elif left == x1:
                theta = -1*np.arctan((y1-y4)/(x1-x4)) * 180 / np.pi
            length = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            p = p.resize((int(length), int(length)), resample=Image.BILINEAR)
            p = p.rotate(theta, resample=Image.BILINEAR, expand=True, fillcolor=(0,0,0,0))

            f.paste(p, box=(int(left), int(top)), mask=p)
            result = np.asarray(f)

        out.write(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    out.release()
    print("Done generating video!")

if __name__ == "__main__":
    main()
