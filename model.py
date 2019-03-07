import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torch.nn.functional as F
#from resnet import resnet34
from torchvision.models import resnet34
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    def __init__(self,input_height,input_width,input_channels):
        super(Model,self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(input_channels)
        self.resnet = resnet34(pretrained=False)
        self.layer = nn.Sequential(nn.ReLU(),nn.Linear(1000,512,bias=True),nn.ReLU(),nn.Linear(512,128,bias=True),nn.ReLU(),nn.Linear(128,4,bias=True),nn.Sigmoid())
        #self.layer = nn.Sequential(nn.ReLU(),nn.Linear(512,128,bias=True),nn.ReLU(),nn.Linear(128,64,bias=True),nn.ReLU(),nn.Linear(64,4,bias=True),nn.Sigmoid())                  
    def forward(self,x):
        x = self.batch_norm1(x)
        x = self.resnet(x)
        x = self.layer(x)
        return x
