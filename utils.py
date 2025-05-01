import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from vgg import VGG, VGG_MNIST
from mobilenet import MobileNetV2
from resnet import ResNet18, ResNet18_MNIST

def full_train_VGG11(train, num_epochs=5, device=torch.device('cuda')):
    lr = 0.01
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

def full_train_VGG11_MNIST(train, num_epochs=2, device=torch.device('cuda')):
    lr = 0.01
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = VGG_MNIST('VGG11').to(device)
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

def full_train_resnet_MNIST(train, num_epochs=2, device=torch.device('cuda')):
    lr = 0.01
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = ResNet18_MNIST().to(device)
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


def full_train_mobilenet(train, num_epochs=5, device=torch.device('cuda')):
    lr = 0.01
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = MobileNetV2().to(device)
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

def full_train_mobilenet_MNIST(train, num_epochs=2, device=torch.device('cuda')):
    lr = 0.01
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = MobileNetV2(num_channels=1).to(device)
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

def get_correctness_from_net(dset, net, device=torch.device('cuda')):
    net.eval()
    loader = torch.utils.data.DataLoader(dset, batch_size=512, shuffle=False)
    correctness_list = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            preds = outputs.argmax(dim=1)
            correct = (preds == labels)
            correctness_list.append(correct.cpu())
    
    correctness_tensor = torch.cat(correctness_list)
    return correctness_tensor