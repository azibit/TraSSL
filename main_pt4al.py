import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, time, glob
import argparse

# from models import *
from utils import progress_bar
from loader import Loader_Cold
from sklearn.metrics import precision_score, recall_score, f1_score  # For precision and recall
import pandas as pd

import numpy as np

from auto_augment import RandAugmentPolicy

from torch.optim.swa_utils import AveragedModel, SWALR
from helper_methods import select_samples, count_folders

from model_loaders import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--DATASET_NAME', required=True, help='Name of the dataset')
# parser.add_argument('--sort_order', type=int, default=-1, help='Sort order (default: -1)')

args = parser.parse_args()

dataset_name = args.DATASET_NAME

total_epochs = 100
swa_start = int(0.75 * total_epochs)
total_experiments = 1
num_classes = count_folders(f'./DATA/{dataset_name}/train')

def run_experiment(num_exps):
    output_result = [] # Save all results
    data_result = [] # Save result for each round of data sample selected
    output_file_name = f'output_{dataset_name}_.csv'

    for _ in range(num_exps):
        for num_samples in [1]:

            # Data
            print('==> Preparing data..')
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugmentPolicy(N=1,M=5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_data_list = glob.glob(f"./DATA/{dataset_name}/train/*/*")
            trainset = Loader_Cold(is_train=True, transform=transform_train, dataset_name=dataset_name, data_list=train_data_list)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

            testset = Loader_Cold(is_train=False, transform=transform_test, dataset_name=dataset_name)
            testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

            # Model
            print('==> Building model..')
            # net = ResNet18(num_classes=num_classes)
            # net = net.to(device)
            # Load the classifier model
            model = MocoModel()
            net = Classifier(model.backbone,  num_classes=num_classes)
            net.load_state_dict(torch.load('classifier_model.pth'))
            net = net.to(device)

            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

            swa_model = AveragedModel(net)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])
            swa_scheduler = SWALR(optimizer, swa_lr=0.05)

            # Training
            def train(epoch):
                print('\nEpoch: %d' % epoch)

                net.train()
                train_loss = 0
                correct = 0
                total = 0

                train_preds = []
                train_targets = []

                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                    
                    # Accumulate predictions and targets during training
                    train_preds.extend(predicted.cpu().numpy())
                    train_targets.extend(targets.cpu().numpy())

                    if epoch > swa_start:
                        swa_model.update_parameters(net)
                        swa_scheduler.step()

                    
                return (train_loss/(batch_idx+1)), 100.*correct/total, train_preds, train_targets

            def test_swa_model():
                torch.optim.swa_utils.update_bn(trainloader, swa_model)
                swa_model.eval()
                test_loss = 0
                correct = 0
                total = 0
                test_preds = []
                test_targets = []
                batch_idx = 0

                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(testloader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = swa_model(inputs)
                        loss = criterion(outputs, targets)

                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                        
                        # Accumulate predictions and targets during training
                        test_preds.extend(predicted.cpu().numpy())
                        test_targets.extend(targets.cpu().numpy())
                    

                # Save checkpoint.
                acc = 100. * correct / total if total != 0 else 0
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt.pth')
                return (test_loss/(batch_idx+1)), acc, test_preds, test_targets

            start_time = time.time()
            for epoch in range(0, total_epochs):
                train_loss, train_acc, train_preds, train_targets = train(epoch)
                scheduler.step()

            test_loss, test_acc, test_preds, test_targets = test_swa_model()

            train_prec = precision_score(train_targets, train_preds, average='macro')
            train_recall = recall_score(train_targets, train_preds, average='macro')

            test_prec = precision_score(test_targets, test_preds, average='macro')
            test_recall = recall_score(test_targets, test_preds, average='macro')

            # Compute F1 scores
            f1_train = f1_score(train_targets, train_preds, average='macro')
            f1_test = f1_score(test_targets, test_preds, average='macro')

            data_result = [num_samples, train_loss, train_acc, train_prec, train_recall, f1_train, test_loss, test_acc, test_prec, test_recall, f1_test, epoch]
            output_result.append(data_result)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time} seconds for {num_samples} samples to run {total_epochs} epochs")

            column_headers = [
                'num_samples',
                'train_loss',
                'train_acc',
                'train_prec',
                'train_recall',
                'f1_train',
                'test_loss',
                'test_acc',
                'test_prec',
                'test_recall',
                'f1_test',
                'epoch'
            ]

            # Create a DataFrame
            df = pd.DataFrame(output_result, columns = column_headers)

            # Save the DataFrame to a CSV file
            df.to_csv(output_file_name, index=False,  mode='w') 
    print(output_result)


run_experiment(total_experiments)