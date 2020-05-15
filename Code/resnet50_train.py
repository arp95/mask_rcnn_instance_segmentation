from resnet import *
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
%matplotlib inline

# Data Augmentation for the training data
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # After converting to PyTorch tensor the [0,255] range is converted to [0,1] range
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]) # Mean and Standard deviation for each colour channel
])

# Data Augmentation for the testing data
# In the test set transformation we dont need to perform random rotation or random horizontal flips

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

root = 'drive/My Drive/CATS_DOGS/CATS_DOGS'

# Location of the training data
train_data = datasets.ImageFolder(os.path.join(root,'train'), transform = train_transform)

# Location of the testing data
test_data = datasets.ImageFolder(os.path.join(root,'test'), transform = test_transform)

# Set seed
torch.manual_seed(42)

train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
test_loader = DataLoader(test_data,batch_size=10)

class_names = train_data.classes

# Setup the Loss and the Optimization Parameters
torch.manual_seed(101)

# create device variable
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Object for the ResNet class
resnet = ResNet50(img_channels = 3, num_classes = 2) 

resnet.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# The optimizer. We use Adam Optimizer
optimizer = torch.optim.Adam(CNNModel.parameters(),lr=0.001)

import time 

# The start time of execution
start_time = time.time()

epochs = 3

max_train_batch = 800 # Batch of 10 images --> 8000 Images
max_test_batch = 300 # Batch of 10 images --> 300 Images

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):

  trn_corr = 0
  tst_corr = 0

  for b, (x_train, y_train) in enumerate(train_loader):

    x_train = x_train.cuda()
    y_train = y_train.cuda()

    # Limit the number of batches
    if b == max_train_batch:
      break
    b += 1
        
    y_pred = resnet(x_train)
    loss = criterion(y_pred,y_train)
        
    # Tally number of correct predictions
    #predicted = torch.max(y_pred.data,1)[1]
    #batch_corr = (predicted == y_train).sum()
    #trn_corr += batch_corr
        
    # Update the Parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    if b%200 == 0:

      print(f'Epoch {i} LOSS {loss.item()}')
    
    #train_losses.append(loss)
    #train_correct.append(trn_corr)
    
    #TEST SET  
total_time = time.time() - start_time
print(f'Total Time: {total_time/60} minutes')

with torch.no_grad():
   for b , (x_Test,y_test) in enumerate(test_loader):
     x_Test = x_Test.cuda()
     y_test = y_test.cuda()
     
     y_val = resnet(x_Test)

     predicted = torch.max(y_pred.data, 1)[1] 
     batch_corr = (predicted == y_test).sum()
     tst_corr = batch_corr
            
loss = criterion(y_val,y_test)
test_losses.append(loss)
test_correct.append(tst_corr)