import torch
from torch import nn
from torch.nn import functional

#creating convolution neural network

class convolution_neural_network(nn.Module):

    def __init__(self):
        super(convolution_neural_network,self).__init__()

        #creating the convolution layer
        self.conv_1 = nn.Conv2d(3,48,3)
        self.conv_2 = nn.Conv2d(48,48,3)
        self.conv_3 = nn.Conv2d(48,96,3)
        self.conv_4 = nn.Conv2d(96,96,3,stride=2)

        #flatterning the layer
        self.flat = nn.Flatten()
        self.batch_normal = nn.BatchNorm1d(96*12*12)

        #Fully connected layer
        self.fc1 = nn.Linear(96*12*12,256)
        self.fc2 = nn.Linear(256,10)

        #Activation function
        self.relu = nn.ReLU()

    def Forward(self,signal):
        #Connecting the network and Forwarding the input

        #Convolution layer forwarding
        signal = self.relu(self.conv_1(signal))
        signal = self.relu(self.conv_2(signal))
        signal = functional.dropout(signal,0.1)
        signal = self.relu(self.conv_3(signal))
        signal = self.relu(self.conv_4(signal))
        signal = functional.dropout(signal,0.5)

        #Flaterning the input
        signal = self.flat(signal)
        signal = self.relu(self.batch_normal(signal))

        #Fully connected layer
        signal = self.relu(self.fc1(signal))
        signal = self.fc2(signal)

        return signal

