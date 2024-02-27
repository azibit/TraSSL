import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import glob 
from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models import ResNetGenerator
from lightly.models.modules.heads import MoCoProjectionHead
from pl_bolts.datamodules import CIFAR10DataModule
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
from lightly.transforms import MoCoV2Transform, utils

import sys
from model_loaders import *
from helper_methods import *

import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--DATASET_NAME', required=True, help='Name of the dataset')
parser.add_argument('--TRAINING_OPTION', type=int, required=True, help='Option to use for running the experiment')
parser.add_argument('--TRANSFER_DATASET_NAME', default="", required=False, help='The transfer dataset to use')
# Option 1 - Standard Training
# Option 2 - Self Supervised Training
# Option 3 - Self-Supervised Training with Transfer Data
# parser.add_argument('--sort_order', type=int, default=-1, help='Sort order (default: -1)')

args = parser.parse_args()

num_workers = int(os.cpu_count() / 2)
batch_size = 256 if torch.cuda.is_available() else 64
memory_bank_size = 4096
seed = 1
finetuning_epochs = 40
self_supervised_epochs = 100

dataset_name = args.DATASET_NAME # "pneumoniamnist"
training_option = args.TRAINING_OPTION
transfer_dataset_name = args.TRANSFER_DATASET_NAME


dataset_folder_path = f"./DATA/MedMNIST/{dataset_name}"
# path_to_train_self_supervised = "./DATA/CIFAR10/train_self_sup/"

path_to_val = f"{dataset_folder_path}/val/"
path_to_test = f"{dataset_folder_path}/test/"
path_to_train_classifier = f"{dataset_folder_path}/train/"
path_to_train_self_supervised = path_to_train_classifier # Train with same as training data

if training_option == 3:
    if transfer_dataset_name:
        path_to_train_self_supervised = f"{dataset_folder_path}/{dataset_name}_{transfer_dataset_name}/"
        path_to_train_data_for_transfer_dataset = f"./DATA/MedMNIST/{transfer_dataset_name}/train"

        # Copy the train of the dataset and the transfer dataset into a new folder inside of the dataset folder
        #1. Create new folder with name - path_to_train_self_supervised
        #2. Copy files from  path_to_train_classifier into the new folder - path_to_train_self_supervised
        #3. Copy files from path_to_train_data_for_transfer_dataset into the new folder - path_to_train_self_supervised
        #4. If folder names are duplicated, rename the files or folders so that all are copied
        #5. Before copying, specify a number of files to copy from each folder into the new folder
        copy_files_with_limit(path_to_train_classifier, path_to_train_self_supervised, 0) # Original dataset
        copy_files_with_limit(path_to_train_data_for_transfer_dataset, path_to_train_self_supervised, 0) # Transfer dataset

    else:
        path_to_train_self_supervised = f"{dataset_folder_path}/train_self_sup/" # Train with same as training data

# pl.seed_everything(seed)

# disable blur because we're working with tiny images
transform = MoCoV2Transform(
    input_size=32,
    gaussian_blur=0.0,
)

# Augmentations typically used to train on cifar-10
train_classifier_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)

# We use the moco augmentations for training moco
dataset_train_moco = LightlyDataset(input_dir=path_to_train_self_supervised, transform=transform)

# Since we also train a linear classifier on the pre-trained moco model we
# reuse the test augmentations here (MoCo augmentations are very strong and
# usually reduce accuracy of models which are not used for contrastive learning.
# Our linear layer will be trained using cross entropy loss and labels provided
# by the dataset. Therefore we chose light augmentations.)
dataset_train_classifier = LightlyDataset(
    input_dir=path_to_train_classifier, transform=train_classifier_transforms
)

dataset_val = LightlyDataset(input_dir=path_to_val, transform=test_transforms)

dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)

dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)
    
model = MocoModel(max_epochs=self_supervised_epochs)

if training_option != 1:
    trainer = pl.Trainer(max_epochs=self_supervised_epochs, devices=1, accelerator="gpu")
    trainer.fit(model, dataloader_train_moco)

# model.eval()
classifier = LitResnet(backbone=model.backbone, data_loader=dataloader_train_classifier, num_classes=count_folders(path_to_test))

# Save the classifier model
# torch.save(classifier.state_dict(), 'classifier_model.pth')

trainer = pl.Trainer(max_epochs=finetuning_epochs, devices=1, accelerator="gpu")
trainer.fit(classifier, dataloader_train_classifier, dataloader_val)

# # test (pass in the loader)
trainer.test(dataloaders=dataloader_test)