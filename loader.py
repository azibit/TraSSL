import glob
import os
from PIL import Image, ImageFilter

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import cv2

class RotationLoader(Dataset):
    def __init__(self, is_train=True, transform=None, dataset_name='CIFAR'):
        self.is_train = is_train
        self.transform = transform
        self.img_path = glob.glob(f'./DATA/{dataset_name}/train/*/*') + glob.glob(f'./DATA/organsmnist/train/*/*') + glob.glob(f'./DATA/dermamnist/train/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        if self.is_train:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3]
        else:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]

class Loader2(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list

        if self.is_train: # train
            self.img_path = path_list
        else:
            if path_list is None:
                self.img_path = glob.glob('./DATA/train/*/*') # for loss extraction
            else:
                self.img_path = path_list
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.is_train:
            img = cv2.imread(self.img_path[idx][:-1])
        else:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                img = cv2.imread(self.img_path[idx][:-1])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        return img, label
    
class Loader_Cold(Dataset):
    def __init__(self, is_train=True, transform=None, data_list = [], dataset_name = 'CIFAR'):
        self.classes = 10
        self.is_train = is_train
        self.transform = transform
        self.list = data_list

        if self.is_train: # train
           self.img_path = self.list
        else:
            self.img_path = glob.glob(f'./DATA/{dataset_name}/test/*/*')

        if not self.img_path:
            raise ValueError(f"No images found. Check the path or dataset_name.")

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx].strip())
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        return img, label
    
class Loader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', n_views = 2):
        self.classes = 10 
        self.is_train = is_train
        self.transform = transform
        self.n_views = n_views
        if self.is_train: # train
            self.img_path = glob.glob(path + '/train/*/*')
        else:
            self.img_path = glob.glob(path + '/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        # label = int(self.img_path[idx].split('/')[-2])

        return img, [idx for i in range(self.n_views)]
    
class LoaderUniversalRepresentation(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', n_views = 2):
        self.classes = 10 
        self.is_train = is_train
        self.transform = transform
        self.n_views = n_views
        if self.is_train: # train
            self.img_path = glob.glob(path + '/train/*/*')
        else:
            self.img_path = glob.glob(path + '/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        rotations = [0,1,2,3]

        rotation_labels = []
        imgs = [] # Images after applying rotation

        # Iterate over the img and rotate each one and get a label for the rotation.
        # For each im in img
        for im in img:

            # Get a random rotation angle
            random_angle = random.choice(rotations)
            rotation_labels.append(random_angle)

            # Rotate the image
            new_im = torch.rot90(im, random_angle, [1,2])
            imgs.append(new_im)

        return imgs, [idx for i in range(self.n_views)], rotation_labels
    
class SampleDataLoader(Dataset):
    def __init__(self, is_train=True, transform=None, count = 100):
        self.classes = 10 
        self.is_train = is_train
        self.transform = transform
        if self.is_train: # train
            imgs_paths = glob.glob('./DATA/train/*/*')
        else:
            imgs_paths = glob.glob('./DATA/test/*/*')
        random.shuffle(imgs_paths)
        self.img_path = imgs_paths[:count]
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        img1 = torch.rot90(img, 1, [1,2])
        img2 = torch.rot90(img, 2, [1,2])
        img3 = torch.rot90(img, 3, [1,2])
        imgs = [img, img1, img2, img3]
        rotations = [0,1,2,3]
        random.shuffle(rotations)
        return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], img_path