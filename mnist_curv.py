import torch
import torchvision
from torchvision import transforms
import os
import numpy as np

from utils import full_train_VGG11_MNIST, full_train_resnet_MNIST, full_train_mobilenet_MNIST
from attacks import SubsetTransformDataset, ReplaceWithDataset, Deepfool, Pseudoinverse, NaiveMaxEMD
from scoring import get_curv_scores_for_net

from mat import full_train_resnet_mat_MNIST

default_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=default_transform, download=True)

BASE_DIR = './mnist_curv_scores'
num_runs = 5

sizes = [10, 100, 1000]

def replace_attack(dir_name, replace_dataset, net_type='VGG'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')
            subset_idx = torch.randperm(len(mnist))[:size]
            new_dset = SubsetTransformDataset(mnist, subset_idx, ReplaceWithDataset(replace_dataset))
            if net_type == 'VGG':
                net = full_train_VGG11_MNIST(new_dset)
            elif net_type == 'Resnet':
                net = full_train_resnet_MNIST(new_dset)
            elif net_type == 'Mobile':
                net = full_train_mobilenet_MNIST(new_dset)
            scores = get_curv_scores_for_net(new_dset, net)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict)

def replace_attack_mat(dir_name, replace_dataset, net_type='Resnet'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')
            subset_idx = torch.randperm(len(mnist))[:size]
            new_dset = SubsetTransformDataset(mnist, subset_idx, ReplaceWithDataset(replace_dataset))
            if net_type == 'Resnet':
                net = full_train_resnet_mat_MNIST(new_dset)
            scores = get_curv_scores_for_net(new_dset, net)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict) 

def deepfool_attack(dir_name, overshoot=0.02, net_type='VGG'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')
            if net_type == 'VGG':
                basenet = full_train_VGG11_MNIST(mnist)
            elif net_type == 'Resnet':
                basenet = full_train_resnet_MNIST(mnist)
            elif net_type == 'Mobile':
                basenet = full_train_mobilenet_MNIST(mnist)
            subset_idx = torch.randperm(len(mnist))[:size]
            new_dset = SubsetTransformDataset(mnist, subset_idx, Deepfool(basenet, overshoot))
            if net_type == 'VGG':
                net = full_train_VGG11_MNIST(new_dset)
            elif net_type == 'Resnet':
                net = full_train_resnet_MNIST(new_dset)
            elif net_type == 'Mobile':
                net = full_train_mobilenet_MNIST(new_dset)
            scores = get_curv_scores_for_net(new_dset, net)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict)


def pinv_attack(dir_name, net_type='VGG'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue # delete or rename old score directory if new ones are to be created

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')           
            subset_idx = torch.randperm(len(mnist))[:size]
            new_dset = SubsetTransformDataset(mnist, subset_idx, Pseudoinverse())
            if net_type == 'VGG':
                net = full_train_VGG11_MNIST(new_dset)
            elif net_type == 'Resnet':
                net = full_train_resnet_MNIST(new_dset)
            elif net_type == 'Mobile':
                net = full_train_mobilenet_MNIST(new_dset)
            scores = get_curv_scores_for_net(new_dset, net)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict)

def pinv_mat_attack(dir_name, net_type='Resnet'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue # delete or rename old score directory if new ones are to be created

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')           
            subset_idx = torch.randperm(len(mnist))[:size]
            new_dset = SubsetTransformDataset(mnist, subset_idx, Pseudoinverse())
            # if net_type == 'VGG':
            #     net = full_train_VGG11_MNIST(new_dset)
            if net_type == 'Resnet':
                net = full_train_resnet_mat_MNIST(new_dset)
            # elif net_type == 'Mobile':
            #     net = full_train_mobilenet_MNIST(new_dset)
            scores = get_curv_scores_for_net(new_dset, net)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict)

def naive_emd_attack(dir_name, net_type='VGG'):
    for size in sizes:
        dir_path = f'{BASE_DIR}/{dir_name}_{size}'
        try:
            os.mkdir(dir_path) # make directory to keep scores if not already created
        except:
            continue

        for i in range(num_runs):
            print(f'Saving scores at {dir_name} for size {size} run {i+1}...')
            
            subset_idx = torch.randperm(len(mnist))[:size]
            new_dset = SubsetTransformDataset(mnist, subset_idx, NaiveMaxEMD())
            if net_type == 'VGG':
                net = full_train_VGG11_MNIST(new_dset)
            elif net_type == 'Resnet':
                net = full_train_resnet_MNIST(new_dset)
            elif net_type == 'Mobile':
                net = full_train_mobilenet_MNIST(new_dset)
            scores = get_curv_scores_for_net(new_dset, net)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict)

fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=default_transform, download=True)

replace_attack('fashion_vgg', fashion, net_type='VGG')
replace_attack('fashion_resnet', fashion, net_type='Resnet')
replace_attack('fashion_mobile', fashion, net_type='Mobile')

deepfool_attack('deepfool_vgg', net_type='VGG')
deepfool_attack('deepfool_resnet', net_type='Resnet')
deepfool_attack('deepfool_mobile', net_type='Mobile')

pinv_attack('pinv_vgg', net_type='VGG')
pinv_attack('pinv_resnet', net_type='Resnet')
pinv_attack('pinv_mobile', net_type='Mobile')

naive_emd_attack('naiveemd_vgg', net_type='VGG')
naive_emd_attack('naiveemd_resnet', net_type='Resnet')
naive_emd_attack('naiveemd_mobile', net_type='Mobile')
