import torch
import torchvision
from torchvision import transforms
import os
import numpy as np

from utils import full_train_VGG11, full_train_resnet, get_boundary_subset_from_net
from attacks import SubsetTransformDataset, ReplaceWithDataset, Deepfool, Pseudoinverse, NaiveMaxEMD
from scoring import get_curv_scores_for_net

cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

basecifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, transform=default_transform, download=False)

BASE_DIR = './cifar10_curv_scores'
num_runs = 5

sizes = [10, 100, 1000]

def replace_attack(dir_name, replace_dataset, net_type='VGG', mode='random', resize=(32, 32)):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')
            if mode == 'random':
                subset_idx = torch.randperm(len(cifar10))[:size]
            elif mode == 'boundary':
                if net_type == 'VGG':
                    basenet = full_train_VGG11(basecifar10)
                elif net_type == 'Resnet':
                    basenet = full_train_resnet(basecifar10)
                subset_idx = get_boundary_subset_from_net(basecifar10, basenet, size)
            new_dset = SubsetTransformDataset(cifar10, subset_idx, 
                                              transforms.Compose([
                                                transforms.ToTensor(), 
                                                ReplaceWithDataset(replace_dataset, resize),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]), 
                                                default_transform)
            if net_type == 'VGG':
                net = full_train_VGG11(new_dset)
            elif net_type == 'Resnet':
                net = full_train_resnet(new_dset)
            scores = get_curv_scores_for_net(new_dset, net)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict)

def deepfool_attack(dir_name, overshoot=0.02, net_type='VGG', mode='random'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')
            if net_type == 'VGG':
                basenet = full_train_VGG11(basecifar10)
            elif net_type == 'Resnet':
                basenet = full_train_resnet(basecifar10)
            if mode == 'random':
                subset_idx = torch.randperm(len(cifar10))[:size]
            elif mode == 'boundary':
                subset_idx = get_boundary_subset_from_net(basecifar10, basenet, size)
            new_dset = SubsetTransformDataset(cifar10, subset_idx, 
                                              transforms.Compose([
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                                                Deepfool(basenet, overshoot)]),
                                                default_transform)
            if net_type == 'VGG':
                net = full_train_VGG11(new_dset)
            elif net_type == 'Resnet':
                net = full_train_resnet(new_dset)
            scores = get_curv_scores_for_net(new_dset, net)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict)


def pinv_attack(dir_name, net_type='VGG', mode='random'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')
            
            if mode == 'random':
                subset_idx = torch.randperm(len(cifar10))[:size]
            elif mode == 'boundary':
                if net_type == 'VGG':
                    basenet = full_train_VGG11(basecifar10)
                elif net_type == 'Resnet':
                    basenet = full_train_resnet(basecifar10)
                subset_idx = get_boundary_subset_from_net(basecifar10, basenet, size)
            new_dset = SubsetTransformDataset(cifar10, subset_idx, 
                                              transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                                                Pseudoinverse()]),
                                                default_transform)
            if net_type == 'VGG':
                net = full_train_VGG11(new_dset)
            elif net_type == 'Resnet':
                net = full_train_resnet(new_dset)
            scores = get_curv_scores_for_net(new_dset, net)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict)

def naive_emd_attack(dir_name, net_type='VGG', mode='random'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')
            
            if mode == 'random':
                subset_idx = torch.randperm(len(cifar10))[:size]
            elif mode == 'boundary':
                if net_type == 'VGG':
                    basenet = full_train_VGG11(basecifar10)
                elif net_type == 'Resnet':
                    basenet = full_train_resnet(basecifar10)
                subset_idx = get_boundary_subset_from_net(basecifar10, basenet, size)
            new_dset = SubsetTransformDataset(cifar10, subset_idx, 
                                              transforms.Compose([
                                                transforms.ToTensor(), 
                                                NaiveMaxEMD(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]),
                                                default_transform)
            if net_type == 'VGG':
                net = full_train_VGG11(new_dset)
            elif net_type == 'Resnet':
                net = full_train_resnet(new_dset)
            scores = get_curv_scores_for_net(new_dset, net)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict)

svhn = torchvision.datasets.SVHN(root='./data', split='train', transform=transforms.ToTensor(), download=False)

replace_attack('svhn_resnet', svhn, net_type='Resnet')
deepfool_attack('deepfool02_resnet', net_type='Resnet')
pinv_attack('pinv_resnet', net_type='Resnet')
naive_emd_attack('naiveemd_resnet', net_type='Resnet')