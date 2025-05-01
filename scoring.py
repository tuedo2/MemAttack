import torch
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np

from utils import full_train_VGG11, full_train_resnet, full_train_mobilenet
from utils import full_train_VGG11_MNIST, full_train_resnet_MNIST, full_train_mobilenet_MNIST

from utils import get_correctness_from_net

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

def get_regularized_curvature_for_batch(net, batch_data, batch_labels, h=1e-3, niter=10, temp=1):
    """
    Helper function that generates curvature scores for a batch
    """
    num_samples = batch_data.shape[0]
    net.eval()
    net.zero_grad()
    regr = torch.zeros(num_samples)

    for _ in range(niter):
        v = torch.randint_like(batch_data, high=2).cuda()
        # Generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        v = h * (v + 1e-7)

        batch_data.requires_grad_()
        outputs_pos = net(batch_data + v)
        outputs_orig = net(batch_data)
        loss_pos = criterion(outputs_pos / temp, batch_labels)
        loss_orig = criterion(outputs_orig / temp, batch_labels)
        grad_diff = torch.autograd.grad((loss_pos-loss_orig), batch_data )[0]

        regr += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()

        net.zero_grad()
        if batch_data.grad is not None:
            batch_data.grad.zero_()

    curv_estimate = regr / niter
    return curv_estimate

def get_curv_scores_for_net(dataset, net):
    """
    Args:
        dataset (Dataset): The dataset that is being scored.
        net (Model): a model trained on the dataset parameter
    """
    scores = torch.zeros(len(dataset))
    total = 0

    trainloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=512, shuffle=False)

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        start_idx = total
        stop_idx = total + len(targets)
        idxs = [j for j in range(start_idx, stop_idx)]
        total = stop_idx

        inputs.requires_grad = True
        
        curv_estimate = get_regularized_curvature_for_batch(net, inputs, targets)
        scores[idxs] = curv_estimate.detach().clone().cpu()

    return scores

def get_memorization_scores(dataset, net_type="VGG", num_runs=100, subset_ratio=0.7):
    full_length = len(dataset)
    subset_length = subset_ratio * full_length
    masks = []
    correctnesses = []

    for _ in num_runs:
        subset_idx = torch.randperm(full_length)[:subset_length]
        subset_dset = Subset(dataset, subset_idx)
        if net_type == "VGG":
            subset_net = full_train_VGG11(subset_dset, 10)
        elif net_type == "Resnet":
            subset_net = full_train_resnet(subset_net, 10)
        elif net_type == "Mobile":
            subset_net = full_train_mobilenet(subset_dset, 10)
        
        mask = np.zeros(full_length, dtype=torch.bool)
        mask[subset_idx] = True
        correctness = get_correctness_from_net(dataset, subset_net)
        
        masks.append(mask)
        correctnesses.append(correctness)
    
    def _masked_avg(x, mask, axis=0, esp=1e-10):
        return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

    full_mask = np.vstack(mask for mask in masks)
    inv_mask = np.logical_not(full_mask)
    full_correctness = np.vstack(cor for cor in correctnesses)
    mem_est = _masked_avg(full_correctness, full_mask) - _masked_avg(full_correctness, inv_mask)

    return mem_est


def get_memorization_scores_MNIST(dataset, net_type="VGG", num_runs=100, subset_ratio=0.7):
    full_length = len(dataset)
    subset_length = subset_ratio * full_length
    masks = []
    correctnesses = []

    for _ in range(num_runs):
        subset_idx = torch.randperm(full_length)[:subset_length]
        subset_dset = Subset(dataset, subset_idx)
        if net_type == "VGG":
            subset_net = full_train_VGG11_MNIST(subset_dset, 5)
        elif net_type == "Resnet":
            subset_net = full_train_resnet_MNIST(subset_net, 5)
        elif net_type == "Mobile":
            subset_net = full_train_mobilenet_MNIST(subset_dset, 5)
        
        mask = np.zeros(full_length, dtype=torch.bool)
        mask[subset_idx] = True
        correctness = get_correctness_from_net(dataset, subset_net)
        
        masks.append(mask)
        correctnesses.append(correctness)
    
    def _masked_avg(x, mask, axis=0, esp=1e-10):
        return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

    full_mask = np.vstack(mask for mask in masks)
    inv_mask = np.logical_not(full_mask)
    full_correctness = np.vstack(cor for cor in correctnesses)
    mem_est = _masked_avg(full_correctness, full_mask) - _masked_avg(full_correctness, inv_mask)
    
    return mem_est
