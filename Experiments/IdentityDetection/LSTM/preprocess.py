import os
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset

def load_dataset(window_size=1):
    ID = ["ID63",  "ID100", "ID150", "ID219", "ID418", "ID419", "ID501", 
          "ID679", "ID685", "ID785", "ID800", "ID875", "ID887", "ID908",
          "ID949", "ID995", "ID1020", "ID1084", "ID1093", "ID1105"]
    random.shuffle(ID)

    train_ID = ID[3:]
    validation_ID = ID[:3]

    #train_ID = ["ID419", "ID150", "ID219", "ID679", "ID685",
    #       "ID785", "ID800", "ID949", "ID1020", "ID1084"]
    #validation_ID = ["ID875", "ID63"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    directory = os.path.join(os.getcwd(),"expression-data")
    train_dataset = WindowedDataset(transform, directory, window_size, 100, train_ID)
    validation_dataset = WindowedDataset(transform, directory, window_size, 100, validation_ID)#, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    return train_loader, validation_loader, validation_ID

class Event():
    """
    A set of fake videos and a set of real videos that all encode the
    same event 
    """
    def __init__(self, transform, directory):
        self.length = sys.maxsize
        self.fake_set = {}
        for i in [0, 1, 2]:
            cropped_directory = os.path.join(directory, "fake", "cam"+str(i), "cropped")
            images = []
            for filename in os.listdir(cropped_directory):
                if filename.endswith(".jpg"):
                    image = Image.open(os.path.join(cropped_directory, filename))
                    image = transform(image)
                    images.append(image)
            if len(images) < self.length:
                self.length = len(images) 
            self.fake_set[i] = images
        self.real_set = {}
        for i in range(0, 7):
            cropped_directory = os.path.join(directory, "real", "cam"+str(i), "cropped")
            images = []
            for filename in os.listdir(cropped_directory):
                if filename.endswith(".jpg"):
                    image = Image.open(os.path.join(cropped_directory, filename))
                    image = transform(image)
                    images.append(image)
            if len(images) < self.length:
                self.length = len(images) 
            self.real_set[i] = images    
            
    def __len__(self):
        return self.length

    def get_stream(self, video_frames, start, window_size, transform):
        stream = []
        for i in range(start, start+window_size):
            image = video_frames[i]
            if not (transform is None):
                image = transform(image)
            stream.append(image)
        return torch.stack(stream)

    def get_streams(self, num_fakes, start, window_size, transform):
        streams = []
        labels = [-1, -1, -1, -1]

        streams.append(self.get_stream(self.real_set[3], start, window_size, transform))
        streams.append(self.get_stream(self.real_set[4], start, window_size, transform))
        streams.append(self.get_stream(self.real_set[5], start, window_size, transform))
        streams.append(self.get_stream(self.real_set[6], start, window_size, transform))

        if num_fakes >= 1:
            streams.append(self.get_stream(self.fake_set[0], start, window_size, transform))
            labels.append(1)
        else:
            streams.append(self.get_stream(self.real_set[0], start, window_size, transform))
            labels.append(-1)

        if num_fakes >= 2:
            streams.append(self.get_stream(self.fake_set[1], start, window_size, transform))
            labels.append(1)
        else:
            streams.append(self.get_stream(self.real_set[1], start, window_size, transform))
            labels.append(-1)

        if num_fakes == 3:
            streams.append(self.get_stream(self.fake_set[2], start, window_size, transform))
            labels.append(1)
        else:
            streams.append(self.get_stream(self.real_set[2], start, window_size, transform))
            labels.append(-1)

        # Shuffle streams and labels
        temp = list(zip(streams, labels))
        random.shuffle(temp)
        streams, labels = zip(*temp)

        return streams, torch.tensor(labels)

class WindowedDataset(Dataset):
    def __init__(self, transform, dataset_directory, window_size, windows_per_event, ID, is_test = False):
        self.window_size = window_size
        self.events = []
        self.is_test = is_test
        for filename in tqdm(os.listdir(dataset_directory)):
            if filename in ID:
                self.events.append(Event(transform, os.path.join(dataset_directory, filename)))
        
        self.windows = []
        for event_idx in range(len(self.events)):
            for start in range(0, len(self.events[event_idx]) - self.window_size):
                fake_cases = [0, 1, 2, 3]
                weights = [0.1, 0.2, 0.3, 0.4]
                num_fakes = random.choices(fake_cases, weights=weights, k=1)[0]
                self.windows.append((event_idx, num_fakes, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        event, num_fakes, start = self.windows[idx]
        if random.randint(0,1) == 0 and not self.is_test:
            transform = transforms.RandomHorizontalFlip(p=1)
        else:
            transform = None
  
        streams, labels = self.events[event].get_streams(num_fakes, start, self.window_size, transform)
        item = {
            "inputs": streams,
            "y": labels,
        }
        return item
