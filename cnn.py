import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel (grayscale), 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7*7*64, 128)  # Fully connected layer after flattening the feature map
        self.fc2 = nn.Linear(128, num_classes)  # 10 output classes for MNIST digits

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply relu after convolution
        x = torch.max_pool2d(x, 2)    # Max pooling with kernel size 2
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.reshape(-1, 7*7*64)  # Flatten the feature map into a vector
        x = torch.relu(self.fc1(x))  # Apply relu activation after the fully connected layer
        x = self.fc2(x)  # Final output layer
        return x
