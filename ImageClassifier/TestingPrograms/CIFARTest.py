#Author: Mohamed Abdelkader
#Tested using test_batch_osu
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
    data = np.array(dict['data']).reshape((len(dict['data']), 32, 3, 32),order='F').transpose(0, 2, 3, 1)
    data=data/255
    label = dict['labels']
  return label,data

if __name__ == "__main__": 
  img= []
  x,y = uploadData(sys.argv[2]) 

  for f in range (len(x)):
    img.append([x[f],y[f]]) #all 50,000 images


  net = Net() 
  net.load_state_dict(torch.load(sys.argv[1]))

  correct = 0
  total = 0
  pred = 0
  #File to write predicted class labels
  f = open("predicted.txt", "w")
  #Test the network
  with torch.no_grad():
    for i in img:
      labels = torch.Tensor([i[0]])
      items = torch.Tensor([i[1]])
      out = net(items)
      _,pred = torch.max(out,1)
      f.write("Predicted: {} \n".format(pred))
      if(pred == labels):
        correct += 1
      total += 1

  #Find accuracy
  acc = 100*(correct/total)
  print("Test Completed. Accuracy: {}%".format(acc))
  f.close()