import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy, softmax

from vgg import VGG, VGG_MNIST
from mobilenet import MobileNetV2
from resnet import ResNet18, ResNet18_MNIST

def XRM(dset, net_a, net_b, opt_a, opt_b, batch_size=512, num_classes=10, num_epochs=5):
    net_a = net_a.cuda()
    net_b = net_b.cuda()
    def balanced_cel(preds, targets):
        losses = nn.functional.cross_entropy(preds, targets, reduction='none')
        return sum([losses[targets==yi].mean() for yi in targets.unique()])

    indices_a = torch.zeros(len(dset)).bernoulli(0.5).long().cuda()
    trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
    for _ in range(num_epochs):
        # pbar = tqdm(trainloader)
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.cuda(), labels.cuda()

            pred_a = net_a(inputs)
            pred_b = net_b(inputs)

            opt_a.zero_grad()
            opt_b.zero_grad()

            pred_hi = pred_a * (indices_a[i * batch_size:(i + 1) * batch_size]).unsqueeze(1) + pred_b * ((1 - indices_a[i * batch_size:(i + 1) * batch_size])).unsqueeze(1)
            balanced_cel(pred_hi, labels).backward()
            opt_a.step()
            opt_b.step()
    
    net_a.eval()
    net_b.eval()
    p_ho_full = torch.zeros(len(dset), num_classes)

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.cuda(), labels.cuda()

        pred_a = net_a(inputs)
        pred_b = net_b(inputs)

        pred_ho = pred_a * ((1 - indices_a[i * batch_size:(i + 1) * batch_size])).unsqueeze(1) + pred_b * (indices_a[i * batch_size:(i + 1) * batch_size]).unsqueeze(1)
        p_ho_full[i*batch_size:(i+1)*batch_size] = pred_ho.detach().cpu()
    
    return p_ho_full

def get_shift(y, ho, temp=1.0):
    # 1st dim: x, 2nd dim: yho
    p_yho_given_x = (ho / temp).softmax(1)
    # 1st dim: yho, 2nd dim: y
    p_y_yho = torch.cat([
        p_yho_given_x[y.eq(y_i)].sum(0).unsqueeze(1) / len(y)
        for y_i in y.unique()], 1)
    p_y_given_yho = p_y_yho / p_y_yho.sum(1, keepdim=True)
    # calibrated p_yho_given_x
    return torch.log(torch.mm(p_yho_given_x, p_y_given_yho) + 1e-6).detach()

def full_train_resnet_mat(train, num_epochs=5, device=torch.device('cuda')):
    lr = 0.01
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    net_a, net_b = ResNet18().to(device), ResNet18().to(device)
    opt_a, opt_b = optim.SGD(net_a.parameters(), lr=lr, momentum=0.9), optim.SGD(net_b.parameters(), lr=lr, momentum=0.9)

    p_ho_full = XRM(train, net_a, net_b, opt_a, opt_b, batch_size=batch_size, num_epochs=num_epochs).to(device)

    del net_a, net_b, opt_a, opt_b

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = ResNet18().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            shift = get_shift(labels, p_ho_full[i*batch_size:(i+1)*batch_size])
            loss = criterion(outputs + shift, labels)

            loss.backward()
            optimizer.step()
    
    return net

def full_train_resnet_mat_MNIST(train, num_epochs=5, device=torch.device('cuda')):
    lr = 0.01
    batch_size = 512
    criterion = nn.CrossEntropyLoss()

    net_a, net_b = ResNet18_MNIST().to(device), ResNet18_MNIST().to(device)
    opt_a, opt_b = optim.SGD(net_a.parameters(), lr=lr, momentum=0.9), optim.SGD(net_b.parameters(), lr=lr, momentum=0.9)

    p_ho_full = XRM(train, net_a, net_b, opt_a, opt_b, batch_size=batch_size, num_epochs=num_epochs).to(device)

    del net_a, net_b, opt_a, opt_b

    trainloader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    
    net = ResNet18_MNIST().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            shift = get_shift(labels, p_ho_full[i*batch_size:(i+1)*batch_size])
            loss = criterion(outputs + shift, labels)

            loss.backward()
            optimizer.step()
    
    return net