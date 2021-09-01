#Author: Mohamed Abdelkader
#Single image input
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable 
from matplotlib.image import imread
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

if __name__ == "__main__": 
  img = imread(sys.argv[2])
  img = img/255

  img = np.array(img).reshape(1,32,3,32,order='F').transpose(0, 2, 3, 1)
  net = Net() 
  net.load_state_dict(torch.load(sys.argv[1]))

  pred = 0

  #Test the network
  with torch.no_grad():
      
    items = torch.Tensor(img)
    out = net(items)
    _,pred = torch.max(out,1)
    print("Predicted: {} \n".format(pred.item()))