import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    def __init__(self,input_height,input_width,input_channels):
        super(Model,self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.batch_norm = nn.BatchNorm2d(input_channels)
        self.conv_layer = nn.Conv2d(input_channels,16,(6,6),stride=2,bias=True)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.dropout1 =nn.Dropout2d(p=0.2)
        self.conv_layer1 = nn.Conv2d(16,8,(6,6),stride =2,padding=1,bias = True)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.dropout2 =nn.Dropout2d(p=0.2)
        self.conv_layer2 = nn.Conv2d(8,5,(4,5),stride=2,bias=True)
        self.batch_norm3 = nn.BatchNorm2d(5)
        self.max_pool2 = nn.MaxPool2d((3,4),1)
        self.dropout3 =nn.Dropout2d(p=0.2)
        self.conv_layer3 = nn.Conv2d(5,3,4,stride=1,bias=True)
        self.batch_norm4 = nn.BatchNorm2d(3)
        self.fc_layer1 = nn.Linear(264,128,bias=True)
        self.fc_layer2 = nn.Linear(128,32,bias=True)
        self.fc_layer3 = nn.Linear(32,4,bias=True)
    
    def forward(self,input_img):
        #print(input_img.shape)
        output = self.batch_norm(input_img)
        output = self.conv_layer(output)
        output = F.relu(output)
        output = self.batch_norm1(output)
        output = self.dropout1(output)
        output = self.conv_layer1(output)
        output = F.relu(output)
        output = self.batch_norm2(output)
        output = self.dropout2(output)
        #print(output.shape)
        output = self.conv_layer2(output)
        output  = F.relu(output)
        output = self.batch_norm3(output) 
        #print(output.shape)
        output = self.max_pool2(output)
        #print(output.shape)
        output = self.dropout3(output)
        output = self.conv_layer3(output)
        output = F.relu(output)
        output = self.batch_norm4(output)
        #print(output.shape)
        output = output.reshape(output.size(0),-1)
        output = self.fc_layer1(output)
        output = F.sigmoid(output)
        output = self.fc_layer2(output)
        output = F.sigmoid(output)
        output = self.fc_layer3(output)
        output = F.sigmoid(output)
        return output

#input size = 120*160, weight decay = 1e-5