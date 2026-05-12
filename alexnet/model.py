import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F



class aLexNet (nn.Module):
    def __init__(self):
        super(aLexNet,self).__init__()

        self.ReLU = nn.ReLU()

        self.juanji1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2)

        self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.juanji2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)

        self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.juanji3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)

        self.juanji4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)

        self.juanji5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        self.maxpooling3 = nn.MaxPool2d(kernel_size=3, stride=2)    

        self.flatten = nn.Flatten()

        self.fullconnect1 = nn.Linear(in_features=256*6*6, out_features=4096)

        self.fullconnect2 = nn.Linear(in_features=4096, out_features=4096)

        self.fullconnect3 = nn.Linear(in_features=4096, out_features=10)        

    def forward(self, x):

        x = self.juanji1(x)

        x = self.ReLU(x)

        x = self.maxpooling1(x)

        x = self.juanji2(x)

        x = self.ReLU(x)

        x = self.maxpooling2(x)

        x = self.juanji3(x)

        x = self.ReLU(x)

        x = self.juanji4(x)

        x = self.ReLU(x)

        x = self.juanji5(x)

        x = self.ReLU(x)

        x = self.maxpooling3(x)

        x = self.flatten(x)

        x = self.fullconnect1(x)

        x = F.dropout(x, p=0.5, training=self.training) #在训练过程中以0.5的概率随机丢弃一些神经元，防止过拟合

        x = self.ReLU(x)

        x = self.fullconnect2(x)

        x = F.dropout(x, p=0.5, training=self.training) #在训练过程中以0.5的概率随机丢弃一些神经元，防止过拟合

        x = self.ReLU(x)

        x = self.fullconnect3(x)

        return x
    

if __name__ == "__main__":#运行主函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#创建设备
    model = aLexNet().to(device)# 实例化模型并将其移动到设备上（GPU或CPU）
    summary(model, (1, 224, 224))#(1,224,224)是输入的图像的维度，1是通道数，224是图像的宽和高