import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchsummary import summary
from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import torch.utils.data as data
import pandas as pd
import copy
import time

class vGG16(nn.Module):
    def __init__(self):
        super(vGG16, self).__init__()
        self.block1 = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

            )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),        
            nn.ReLU(),  
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )   

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )   

        self.block6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 256),   
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )

        for m in self.modules():#权重初始化
            if isinstance(m, nn.Conv2d):#如果是卷积层，则使用Kaiming正态分布初始化权重，偏置初始化为0
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')#Kaiming正态分布初始化权重，适用于ReLU激活函数
                if m.bias is not None:#如果卷积层有偏置，则将偏置初始化为0
                    nn.init.constant_(m.bias, 0)#将偏置初始化为0
            elif isinstance(m, nn.Linear):#如果是全连接层，则使用正态分布初始化权重，均值为0，标准差为0.01，偏置初始化为0
                nn.init.normal_(m.weight, 0, 0.01)#正态分布初始化权重，均值为0，标准差为0.01
                if m.bias is not None:#如果全连接层有偏置，则将偏置初始化为0
                    nn.init.constant_(m.bias, 0)#将偏置初始化为0



    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        return x
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vGG16().to(device)
    summary(model, (1, 224, 224))
