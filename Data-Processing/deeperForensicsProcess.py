#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:22:02 2021

@author: harman
"""

import os
import argparse

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
    
    for i in range(args.num_parts):
        source_video_dir = os.path.join(args.data_dir, 'souce_videos' + str(i))
        for name in os.listdir(source_video_dir):
            name_video_dir = os.path.join(source_video_dir, name)
            for dir_type in os.listdir(name_video_dir):
                type_name_video_dir = os.path.join(name_video_dir, dir_type)
                ID_dir = os.path.join(args.data_dir, 'ID' + str(id_count))
                os.makedirs(ID_dir)
                id_count += 1
                for cam_num, angle in enumerate(os.listdir(type_name_video_dir)):
                    file = os.path.join(type_name_video_dir, angle)
                    os.rename(file, os.path.join(ID_dir, 'camera' + str(cam_num)+'.mp4'))
            


if __name__ == "__main__":
    main()