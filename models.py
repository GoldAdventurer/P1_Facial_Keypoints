## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)

        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        #self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        
        self.mp1 = nn.MaxPool2d(2, 2)
        self.mp2 = nn.MaxPool2d(2, 2)  
        self.mp3 = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(p=0.2)
        #self.fc1 = nn.Linear(64*53*53, 272)
        #self.fc1 = nn.Linear(64*54*54, 272)
        self.fc1 = nn.Linear(64*26*26, 272)
        self.fc2 = nn.Linear(272, 136)
            
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
        #self.bn1 = nn.BatchNorm1d(64*26*26)
        #self.bn2 = nn.BatchNorm1d(272)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        
        #nn.init.kaiming_uniform_(self.conv1.weight.data)
        #nn.init.kaiming_uniform_(self.conv2.weight.data)   
        #nn.init.kaiming_uniform_(self.conv3.weight.data)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = self.mp3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        #x = self.bn1(x)        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #ßßx = self.bn2(x)
        x = self.fc2(x)

                
        # a modified x, having gone through all the layers of your model, should be returned
        return x

