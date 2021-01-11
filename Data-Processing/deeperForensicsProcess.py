#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:22:02 2021

@author: harman
"""

import os
import argparse
from shutil import copy2

def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Experiment')

    parser.add_argument('--data-dir', type=str,
                    help='Directory where DeeperForensics dataset directories live')
    parser.add_argument('--num-parts', type=int, default=2, help = 'Number of parts of the dataset')
    
    
    args = parser.parse_args()
    
    return args


def main():
    
    args = parse_args()
    
    id_count = 1
    
    #iterate over dataset parts
    for i in range(args.num_parts):
        source_video_dir = os.path.join(args.data_dir, 'source_videos' + str(i))
        
        #iterate over M names
        for name in os.listdir(source_video_dir):
            name_video_dir = os.path.join(source_video_dir, name)
            
            #iterate over types of change (BlendShape, light, etc)
            for dir_type in os.listdir(name_video_dir):
                type_name_video_dir = os.path.join(name_video_dir, dir_type)
                
                #If in BlendShape situation
                if 'camera_down' in list(os.listdir(type_name_video_dir)):
                    ID_dir = os.path.join(args.data_dir, 'ID' + str(id_count))
                    os.makedirs(ID_dir)
                    id_count += 1
                    for cam_num, angle in enumerate(os.listdir(type_name_video_dir)):
                        temp = os.path.join(type_name_video_dir, angle)
                        file = os.path.join(temp, list(os.listdir(temp))[0])
                        os.rename(file, os.path.join(ID_dir, 'camera' + str(cam_num)+'.mp4'))
                else:
                    #Iterate over emotions in lighting conditions
                    for emotion in os.listdir(type_name_video_dir):
                        emotion_video_dir = os.path.join(type_name_video_dir, emotion)
                        ID_dir = os.path.join(args.data_dir, 'ID' + str(id_count))
                        os.makedirs(ID_dir)
                        id_count += 1
                        
                        #iterate over camer angles and tranfer files
                        for cam_num, angle in enumerate(os.listdir(emotion_video_dir)):
                            temp = os.path.join(emotion_video_dir, angle)
                            file = os.path.join(temp, list(os.listdir(temp))[0])
                            copy2(file, os.path.join(ID_dir, 'camera' + str(cam_num)+'.mp4'))
          
                
            


if __name__ == "__main__":
    main()