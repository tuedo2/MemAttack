import torch
import torchvision
from torchvision import transforms
import os
import numpy as np

from utils import full_train_VGG11_MNIST, full_train_resnet_MNIST, full_train_mobilenet_MNIST
from attacks import SubsetTransformDataset, ReplaceWithDataset, Deepfool, Pseudoinverse, NaiveMaxEMD
from scoring import get_memorization_scores_MNIST

default_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=default_transform, download=True)

BASE_DIR = './mnist_mem_scores'
num_runs = 1

sizes = [100]

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
            scores = get_memorization_scores_MNIST(new_dset, net_type)
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
            scores = get_memorization_scores_MNIST(new_dset, net_type)
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
            scores = get_memorization_scores_MNIST(new_dset, net_type)
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
            scores = get_memorization_scores_MNIST(new_dset, net_type)
            score_dict = dict(subset=subset_idx, scores=scores)
            np.savez(f'{dir_path}/run_{i+1}', **score_dict)

fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=default_transform, download=False)
# kmnist = torchvision.datasets.KMNIST(root='./data', train=True, transform=default_transform, download=False)
# emnist = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, transform=default_transform, download=False)

replace_attack('fashion_vgg1', fashion, net_type='VGG')
replace_attack('fashion_vgg2', fashion, net_type='VGG')
replace_attack('fashion_vgg3', fashion, net_type='VGG')
replace_attack('fashion_vgg4', fashion, net_type='VGG')
replace_attack('fashion_vgg5', fashion, net_type='VGG')

pinv_attack('pinv_vgg1', net_type='VGG')
pinv_attack('pinv_vgg2', net_type='VGG')
pinv_attack('pinv_vgg3', net_type='VGG')
pinv_attack('pinv_vgg4', net_type='VGG')
pinv_attack('pinv_vgg5', net_type='VGG')

replace_attack('fashion_resnet1', fashion, net_type='Resnet')
replace_attack('fashion_resnet2', fashion, net_type='Resnet')
replace_attack('fashion_resnet3', fashion, net_type='Resnet')
replace_attack('fashion_resnet4', fashion, net_type='Resnet')
replace_attack('fashion_resnet5', fashion, net_type='Resnet')

pinv_attack('pinv_resnet1', net_type='Resnet')
pinv_attack('pinv_resnet2', net_type='Resnet')
pinv_attack('pinv_resnet3', net_type='Resnet')
pinv_attack('pinv_resnet4', net_type='Resnet')
pinv_attack('pinv_resnet5', net_type='Resnet')

replace_attack('fashion_mobile1', fashion, net_type='Mobile')
replace_attack('fashion_mobile2', fashion, net_type='Mobile')
replace_attack('fashion_mobile3', fashion, net_type='Mobile')
replace_attack('fashion_mobile4', fashion, net_type='Mobile')
replace_attack('fashion_mobile5', fashion, net_type='Mobile')

pinv_attack('pinv_mobile1', net_type='Mobile')
pinv_attack('pinv_mobile2', net_type='Mobile')
pinv_attack('pinv_mobile3', net_type='Mobile')
pinv_attack('pinv_mobile4', net_type='Mobile')
pinv_attack('pinv_mobile5', net_type='Mobile')

# deepfool_attack('deepfool_vgg', net_type='VGG')
# deepfool_attack('deepfool_resnet', net_type='Resnet')
# deepfool_attack('deepfool_mobile', net_type='Mobile')

# naive_emd_attack('naiveemd_vgg', net_type='VGG')
# naive_emd_attack('naiveemd_resnet', net_type='Resnet')
# naive_emd_attack('naiveemd_mobile', net_type='Mobile')