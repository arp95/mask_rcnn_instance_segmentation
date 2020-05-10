# header files
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import cv2
import glob
from torch.autograd import Variable
import torchvision.models as models
import sys


train_path = ""
val_path = ""
args = sys.argv
if(len(args) > 2):
    train_path = args[1]
    val_path = args[2]

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_path, transform=train_transforms)
val_data = datasets.ImageFolder(val_path, transform=train_transforms)
print(len(train_data))
print(len(val_data))

# loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)

# define VGG16 network
class VGG16(nn.Module):
    
    # init method
    def __init__(self, num_classes = 1000):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            
            # first cnn block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # second cnn block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # third cnn block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # fourth cnn block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # fifth cnn block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    # forward step
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

# define loss
criterion = nn.NLLLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16()
model.to(device)

# optimizer to be used
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

train_losses = []
train_acc = []
val_losses = []
val_acc = []

# train and validate
for epoch in range(0, 40):
    
    # train
    model.train()
    training_loss = 0.0
    total = 0
    correct = 0
    for i, (input, target) in enumerate(train_loader):
        
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        training_loss = training_loss + loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    training_loss = training_loss / float(len(train_loader))
    training_accuracy = str(100.0 * (float(correct) / float(total)))
    train_losses.append(training_loss)
    train_acc.append(training_accuracy)
    
    # validate
    model.eval()
    valid_loss = 0.0
    total = 0
    correct = 0
    for i, (input, target) in enumerate(val_loader):
        
        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        valid_loss = valid_loss + loss.item()
    valid_loss = valid_loss / float(len(val_loader))
    valid_accuracy = str(100.0 * (float(correct) / float(total)))
    val_losses.append(valid_loss)
    val_acc.append(valid_accuracy)
    
    print()
    print("Epoch" + str(epoch) + ":")
    print("Training Accuracy: " + str(training_accuracy) + "    Validation Accuracy: " + str(valid_accuracy))
    print("Training Loss: " + str(training_loss) + "    Validation Loss: " + str(valid_loss))
    print()

#save model
torch.save(model.state_dict(), 'model.pth')
