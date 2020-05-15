import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
%matplotlib inline

# Implementation of the ResNet-50 Architecture

class ResidualBlock(torch.nn.Module):

  def __init__(self, in_channels, out_channels, downsample = None, stride = 1):

    super(ResidualBlock, self).__init__()

    self.expansion = 4
    self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
    self.BatchNorm1 = torch.nn.BatchNorm2d(out_channels)
    self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
    self.BatchNorm2 = torch.nn.BatchNorm2d(out_channels)
    self.conv3 = torch.nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size = 1, stride = 1, padding = 0)
    self.BatchNorm3 = torch.nn.BatchNorm2d(out_channels * self.expansion)
    self.relu = torch.nn.ReLU()
    self.downsample = downsample
  
  def forward(self, x):

    identity = x

    x = self.conv1(x)
    x = self.BatchNorm1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.BatchNorm2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.BatchNorm3(x)

    # After Each Block we will add the identity 
    # Change the shape in some way so that the input and the output have the same shape for successful addition

    if self.downsample is not None:

      identity = self.downsample(identity)
    
    x += identity
    x = self.relu(x)

    return x

class ResNet(torch.nn.Module):

  def __init__(self, residual_block, layers, image_channels, num_classes = 10):

    super(ResNet, self).__init__()
    self.in_channels = 64
    # The initial layer(Not the Resnet Layer)
    self.conv1 = torch.nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3)
    self.BatchNorm1 = torch.nn.BatchNorm2d(64)
    self.relu = torch.nn.ReLU()
    self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

    # Defining the ResNet Layers
    self.layer1 = self.make_layer(residual_block, layers[0], out_channels = 64, stride = 1) # In the end it will be 64*4
    self.layer2 = self.make_layer(residual_block, layers[1], out_channels = 128, stride = 2) # In the end it will be 128*4
    self.layer3 = self.make_layer(residual_block, layers[2], out_channels = 256, stride = 2) # In the end it will be 256*4
    self.layer4 = self.make_layer(residual_block, layers[3], out_channels = 512, stride = 2) # In the end it will be 512*4

    # Average Pooling
    self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
    
    # Fully Connected Layers
    self.fc = torch.nn.Linear(512*4, num_classes)

  def forward(self, x):

    x = self.conv1(x)
    x = self.BatchNorm1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avg_pool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)

    return x
  # A helper function to build the ResNet Layer
  def make_layer(self, residual_block, num_residual_block, out_channels, stride):

    downsample = None
    layers = []

    if stride != 1 or self.in_channels != out_channels * 4:

      downsample = torch.nn.Sequential(torch.nn.Conv2d(self.in_channels, out_channels * 4, kernel_size = 1, stride = stride),
                                       torch.nn.BatchNorm2d(out_channels*4))
    
    layers.append(ResidualBlock(self.in_channels, out_channels, downsample, stride))

    self.in_channels = out_channels * 4 # 256 = 64 * 4

    for i in range(num_residual_block - 1):

      layers.append(ResidualBlock(self.in_channels, out_channels)) # 256->64, 64->(256)
    
    return torch.nn.Sequential(*layers) # Will Unpack the list

def ResNet50(img_channels = 3, num_classes = 1000):

  	return ResNet(ResidualBlock, [3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels = 3, num_classes = 1000):
	return ResNet(ResidualBlock, [3,4,23,3], img_channels, num_classes)