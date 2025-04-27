import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from cnn import CNN
from vgg import VGG
from resnet import ResNet18

def full_train_CNN(train, num_epochs=2, device=torch.device('cuda')):
    lr = 0.1
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = CNN(10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
    
    return net

def full_train_VGG11(train, num_epochs=5, device=torch.device('cuda')):
    lr = 0.001
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = VGG('VGG11').to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
    
    return net


def full_train_resnet(train, num_epochs=5, device=torch.device('cuda')):
    lr = 0.01
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = ResNet18().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
    
    return net


def get_boundary_scores_for_net(train, net):
    '''
    Returns confidence scoring over dataset
    '''
    total = 0
    scores = torch.zeros(len(train))
    loader = torch.utils.data.DataLoader(dataset=train, batch_size=512, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs, targets = inputs.to('cuda'), targets.to('cuda')

            start_idx = total
            stop_idx = total + len(targets)
            idxs = [j for j in range(start_idx, stop_idx)]
            total = stop_idx

            logits = net(inputs)
            softmax_probs = F.softmax(logits, dim=1)
            max_softmax, _ = torch.max(softmax_probs, dim=1)
            
            scores[idxs] = max_softmax.cpu()

    return scores

def get_boundary_subset_from_net(train, net, k):
    '''
    Returns the indices of the k lowest confidence points from net on train
    '''
    scores = get_boundary_scores_for_net(train, net)
    _, subset_idx = torch.topk(scores, k=k, largest=False)

    return subset_idx