#Author: Mohamed Abdelkader
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable 
from operator import itemgetter
import sys
import random

#Network method
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16*5*5,120)
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84,10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1,16*5*5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def uploadData(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='latin1')
    data = dict['data'].reshape((len(dict['data']), 32, 3, 32),order='F').transpose(0, 2, 3, 1)
    data=data/255
    label = dict['labels']
  return label,data

if __name__ == "__main__": 
  img= []

  for i in range(1,6):
    file = "data_batch_" + str(i)
    x,y = uploadData(file) 

    for f in range (10000):
      img.append([x[f],y[f]]) #all 50,000 images
    random.shuffle(img)

  
  ranged_images = []
  range = int(sys.argv[1])

  random.shuffle(img)

  counters = [0,0,0,0,0,0,0,0,0,0]
  count = 0
  while count < 50000: 
    intLabel = (img[count])[0]
    if(counters[intLabel]<range):
      ranged_images.append(img[count])
      counters[intLabel] = counters[intLabel] + 1
    count = count + 1
  
  arr1 = []
  arr2 = []
  arr3 = []
  miniBatches = []
  for i in ranged_images:
    if(i[0]==0 or i[0]==1 or i[0]==2 or i[0]==3):
      arr1.append(i)
    elif(i[0]==4 or i[0]==5 or i[0]==6 or i[0]==7):
      arr2.append(i)
    else:
      arr3.append(i)

  miniBatches.append(arr1)
  miniBatches.append(arr2)
  miniBatches.append(arr3)

  
  net = Net()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)  
 
#Train Data
  epoch = 0
  while epoch < 200: 
    for i in miniBatches:
      for x in i:
        labels = torch.Tensor([x[0]])    
        items = torch.Tensor([x[1]])
        optimizer.zero_grad() # Clear off the gradients from any past operation
        out = net(items)  # Do the forward pass
        
        loss = criterion(out, labels.long()) # Calculate the loss
        loss.backward()       # Calculate the gradients with help of back propagation
        optimizer.step()      # Ask the optimizer to adjust the parameters based on the gradients 
    epoch = epoch + 1

  print("Training Completed, network saved to myNet.pth")
  torch.save(net.state_dict(), "myNet.pth")
