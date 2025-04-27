import torch
import torchvision
from torchvision import transforms
import os
import numpy as np

from utils import full_train_VGG11
from attacks import SubsetTransformDataset, ReplaceWithDataset, Deepfool, Pseudoinverse, NaiveMaxEMD
from scoring import get_curv_scores_for_net

mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

BASE_DIR = './mnist_curv_scores'
num_runs = 5

sizes = [10, 100, 1000]

def replace_attack(dir_name, replace_dataset, net_type='VGG', mode='random'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')
            if mode == 'random':
                subset_idx = torch.randperm(len(mnist))[:size]
                new_dset = SubsetTransformDataset(mnist, subset_idx, ReplaceWithDataset(replace_dataset))
                if net_type == 'VGG':
                    net = full_train_VGG11(new_dset, 2)
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
            if mode == 'random':
                if net_type == 'VGG':
                    basenet = full_train_VGG11(mnist, 2)
                subset_idx = torch.randperm(len(mnist))[:size]
                new_dset = SubsetTransformDataset(mnist, subset_idx, Deepfool(basenet, overshoot))
                if net_type == 'VGG':
                    net = full_train_VGG11(new_dset, 2)
                scores = get_curv_scores_for_net(new_dset, net)
                score_dict = dict(subset=subset_idx, scores=scores)
                np.savez(f'{dir_path}/run_{i+1}', **score_dict)


def pinv_attack(dir_name, net_type='VGG', mode='random'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue # delete or rename old score directory if new ones are to be created

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')
            if mode == 'random':
                subset_idx = torch.randperm(len(mnist))[:size]
                new_dset = SubsetTransformDataset(mnist, subset_idx, Pseudoinverse())
                if net_type == 'VGG':
                    net = full_train_VGG11(new_dset, 2)
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
                subset_idx = torch.randperm(len(mnist))[:size]
                new_dset = SubsetTransformDataset(mnist, subset_idx, NaiveMaxEMD())
                if net_type == 'VGG':
                    net = full_train_VGG11(new_dset, 2)
                scores = get_curv_scores_for_net(new_dset, net)
                score_dict = dict(subset=subset_idx, scores=scores)
                np.savez(f'{dir_path}/run_{i+1}', **score_dict)

fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
kmnist = torchvision.datasets.KMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)

replace_attack('fashion_vgg', fashion)
replace_attack('kmnist_vgg', kmnist)
deepfool_attack('deepfool02_vgg')
pinv_attack('pinv_vgg')
naive_emd_attack('naiveemd_vgg')