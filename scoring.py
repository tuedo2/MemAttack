import torch
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt

from utils import full_train_VGG11, full_train_resnet, full_train_mobilenet
from utils import full_train_VGG11_MNIST, full_train_resnet_MNIST, full_train_mobilenet_MNIST

from vgg import VGG, VGG_MNIST
from resnet import ResNet18, ResNet18_MNIST
from mobilenet import MobileNetV2

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
    subset_length = int(subset_ratio * full_length)
    masks = []
    correctnesses = []

    for _ in range(num_runs):
        subset_idx = torch.randperm(full_length)[:subset_length]
        subset_dset = Subset(dataset, subset_idx)
        if net_type == "VGG":
            subset_net = full_train_VGG11(subset_dset, 10)
        elif net_type == "Resnet":
            subset_net = full_train_resnet(subset_dset, 10)
        elif net_type == "Mobile":
            subset_net = full_train_mobilenet(subset_dset, 10)
        
        mask = np.zeros(full_length, dtype=bool)
        mask[subset_idx] = True
        correctness = get_correctness_from_net(dataset, subset_net)
        
        masks.append(mask)
        correctnesses.append(correctness)
    
    def _masked_avg(x, mask, axis=0, esp=1e-10):
        return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

    full_mask = np.vstack([mask for mask in masks])
    inv_mask = np.logical_not(full_mask)
    full_correctness = np.vstack([cor for cor in correctnesses])
    mem_est = _masked_avg(full_correctness, full_mask) - _masked_avg(full_correctness, inv_mask)

    return mem_est


def get_memorization_scores_MNIST(dataset, net_type="VGG", num_runs=100, subset_ratio=0.7):
    full_length = len(dataset)
    subset_length = int(subset_ratio * full_length)
    masks = []
    correctnesses = []

    for _ in range(num_runs):
        subset_idx = torch.randperm(full_length)[:subset_length]
        subset_dset = Subset(dataset, subset_idx)
        if net_type == "VGG":
            subset_net = full_train_VGG11_MNIST(subset_dset, 5)
        elif net_type == "Resnet":
            subset_net = full_train_resnet_MNIST(subset_dset, 5)
        elif net_type == "Mobile":
            subset_net = full_train_mobilenet_MNIST(subset_dset, 5)
        
        mask = np.zeros(full_length, dtype=bool)
        mask[subset_idx] = True
        correctness = get_correctness_from_net(dataset, subset_net)
        
        masks.append(mask)
        correctnesses.append(correctness)
    
    def _masked_avg(x, mask, axis=0, esp=1e-10):
        return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

    full_mask = np.vstack([mask for mask in masks])
    inv_mask = np.logical_not(full_mask)
    full_correctness = np.vstack([cor for cor in correctnesses])
    mem_est = _masked_avg(full_correctness, full_mask) - _masked_avg(full_correctness, inv_mask)
    
    return mem_est


# def distrs_compute(tr_values, te_values, tr_labels, te_labels, num_bins=5, log_bins=True, plot_name=None):
    
#     ### function to compute and plot the normalized histogram for both training and test values class by class.
#     ### we recommand using the log scale to plot the distribution to get better-behaved distributions.
#     plt.ioff()
    
#     num_classes = len(set(tr_labels))
#     sqr_num = np.ceil(np.sqrt(num_classes))
#     tr_distrs, te_distrs, all_bins = [], [], []
    
#     # plt.figure(figsize = (15,15))
#     # plt.rc('font', family='serif', size=10)
#     # plt.rc('axes', linewidth=2)
    
#     for i in range(num_classes):
#         tr_list, te_list = tr_values[tr_labels==i], te_values[te_labels==i]
#         if log_bins:
#             # when using log scale, avoid very small number close to 0
#             small_delta = 1e-10
#             tr_list[tr_list<=small_delta] = small_delta
#             te_list[te_list<=small_delta] = small_delta
#         # n1, n2 = np.sum(tr_labels==i), np.sum(te_labels==i)
#         all_list = np.concatenate((tr_list, te_list))
#         max_v, min_v = np.amax(all_list), np.amin(all_list)
        
#         # plt.subplot(sqr_num, sqr_num, int(i+1))
#         if log_bins:
#             bins = np.logspace(np.log10(min_v), np.log10(max_v),num_bins+1)
#             weights = np.ones_like(tr_list)/float(len(tr_list))
#             h1, _,_ = plt.hist(tr_list,bins=bins,facecolor='b',weights=weights,alpha = 0.5)
#             # plt.gca().set_xscale("log")
#             weights = np.ones_like(te_list)/float(len(te_list))
#             h2, _, _ = plt.hist(te_list,bins=bins,facecolor='r',weights=weights,alpha = 0.5)
#             # plt.gca().set_xscale("log")
#         else:
#             bins = np.linspace(min_v, max_v,num_bins+1)
#             weights = np.ones_like(tr_list)/float(len(tr_list))
#             h1, _,_ = plt.hist(tr_list,bins=bins,facecolor='b',weights=weights,alpha = 0.5)
#             weights = np.ones_like(te_list)/float(len(te_list))
#             h2, _, _ = plt.hist(te_list,bins=bins,facecolor='r',weights=weights,alpha = 0.5)
#         tr_distrs.append(h1)
#         te_distrs.append(h2)
#         all_bins.append(bins)
#     # if plot_name == None:
#     #     plot_name='./tmp'
#     # plt.savefig(plot_name+'.png', bbox_inches='tight')
#     tr_distrs, te_distrs, all_bins = np.array(tr_distrs), np.array(te_distrs), np.array(all_bins)
#     return tr_distrs, te_distrs, all_bins


# def risk_score_compute(tr_distrs, te_distrs, all_bins, data_values, data_labels):
    
#     ### Given training and test distributions (obtained from the shadow classifier), 
#     ### compute the corresponding privacy risk score for training points (of the target classifier).
    
#     def find_index(bins, value):
#         # for given n bins (n+1 list) and one value, return which bin includes the value
#         if value>=bins[-1]:
#             return len(bins)-2 # when value is larger than any bins, we assign the last bin
#         if value<=bins[0]:
#             return 0  # when value is smaller than any bins, we assign the first bin
#         return np.argwhere(bins<=value)[-1][0]
    
#     def score_calculate(tr_distr, te_distr, ind): 
#         if tr_distr[ind]+te_distr[ind] != 0:
#             return tr_distr[ind]/(tr_distr[ind]+te_distr[ind])
#         else: # when both distributions have 0 probabilities, we find the nearest bin with non-zero probability
#             for t_n in range(1, len(tr_distr)):
#                 t_ind = ind-t_n
#                 if t_ind>=0:
#                     if tr_distr[t_ind]+te_distr[t_ind] != 0:
#                         return tr_distr[t_ind]/(tr_distr[t_ind]+te_distr[t_ind])
#                 t_ind = ind+t_n
#                 if t_ind<len(tr_distr):
#                     if tr_distr[t_ind]+te_distr[t_ind] != 0:
#                         return tr_distr[t_ind]/(tr_distr[t_ind]+te_distr[t_ind])
                    
#     risk_score = []   
#     for i in range(len(data_values)):
#         c_value, c_label = data_values[i], data_labels[i]
#         c_tr_distr, c_te_distr, c_bins = tr_distrs[c_label], te_distrs[c_label], all_bins[c_label]
#         c_index = find_index(c_bins, c_value)
#         c_score = score_calculate(c_tr_distr, c_te_distr, c_index)
#         risk_score.append(c_score)
#     return np.array(risk_score)

# def calculate_risk_score(tr_values, te_values, tr_labels, te_labels, data_values, data_labels, 
#                          num_bins=5, log_bins=True):
    
#     ########### tr_values, te_values, tr_labels, te_labels are from shadow classifier's training and test data
#     ########### data_values, data_labels are from target classifier's training data
#     ########### potential choice for the value -- entropy, or modified entropy, or prediction loss (i.e., -np.log(confidence))
    
#     tr_distrs, te_distrs, all_bins = distrs_compute(tr_values, te_values, tr_labels, te_labels, 
#                                                     num_bins=num_bins, log_bins=log_bins)
#     risk_score = risk_score_compute(tr_distrs, te_distrs, all_bins, data_values, data_labels)
#     return risk_score

# def get_risk_scores(dset, net_type='VGG'):
#     if net_type == "VGG":
#         net = full_train_VGG11(dset, 5)
#     elif net_type == "Resnet":
#         net = full_train_resnet(dset, 5)
#     elif net_type == "Mobile":
#         net = full_train_mobilenet(dset, 5)

def get_proxies(dset, net_type='VGG', num_epochs=5):
    if net_type == 'VGG':
        net = VGG('VGG11').to(device)
    elif net_type == 'Resnet':
        net = ResNet18().to(device)
    elif net_type == 'Mobile':
        net = MobileNetV2().to(device)
    
    trainloader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)
    baseloader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    full_confidence = []
    full_max_confidence = []
    full_entropy = []
    full_correctness = []

    for _ in range(num_epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        
        confidence_list = []
        max_confidence_list = []
        entropy_list = []
        correctness_list = []

        with torch.no_grad():
            for inputs, labels in baseloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                probs = nn.functional.softmax(outputs, dim=1)
                
                confidence = probs[torch.arange(len(labels)), labels].cpu()
                confidence_list.append(confidence)
                max_confidence = probs.max(dim=1).values.cpu()
                max_confidence_list.append(max_confidence)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).cpu()
                entropy_list.append(entropy)
                preds = outputs.argmax(dim=1)
                correct = (preds == labels).cpu()
                correctness_list.append(correct)
        

        full_confidence.append(torch.cat(confidence_list))
        full_max_confidence.append(torch.cat(max_confidence_list))
        full_entropy.append(torch.cat(entropy_list))
        full_correctness.append(torch.cat(correctness_list))

    full_confidence = np.stack(full_confidence)
    full_max_confidence = np.stack(full_max_confidence)
    full_entropy = np.stack(full_entropy)
    full_correctness = np.stack(full_correctness)

    return full_confidence, full_max_confidence, full_entropy, full_correctness

def get_proxies_MNIST(dset, net_type='VGG', num_epochs=5):
    if net_type == 'VGG':
        net = VGG_MNIST('VGG11').to(device)
    elif net_type == 'Resnet':
        net = ResNet18_MNIST().to(device)
    elif net_type == 'Mobile':
        net = MobileNetV2(num_channels=1).to(device)
    
    trainloader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)
    baseloader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    full_confidence = []
    full_max_confidence = []
    full_entropy = []
    full_correctness = []

    for _ in range(num_epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        
        confidence_list = []
        max_confidence_list = []
        entropy_list = []
        correctness_list = []

        with torch.no_grad():
            for inputs, labels in baseloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                probs = nn.functional.softmax(outputs, dim=1)
                
                confidence = probs[torch.arange(len(labels)), labels].cpu()
                confidence_list.append(confidence)
                max_confidence = probs.max(dim=1).values.cpu()
                max_confidence_list.append(max_confidence)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).cpu()
                entropy_list.append(entropy)
                preds = outputs.argmax(dim=1)
                correct = (preds == labels).cpu()
                correctness_list.append(correct)
        

        full_confidence.append(torch.cat(confidence_list))
        full_max_confidence.append(torch.cat(max_confidence_list))
        full_entropy.append(torch.cat(entropy_list))
        full_correctness.append(torch.cat(correctness_list))

    full_confidence = np.stack(full_confidence)
    full_max_confidence = np.stack(full_max_confidence)
    full_entropy = np.stack(full_entropy)
    full_correctness = np.stack(full_correctness)

    return full_confidence, full_max_confidence, full_entropy, full_correctness
