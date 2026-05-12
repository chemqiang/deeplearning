import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary



class leNet(nn.Module):
    def __init__(self):
        super(leNet, self).__init__()
        self.juanji1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.sigmod = nn.Sigmoid()
        self.pooling1 = nn.AvgPool2d(kernel_size=2, stride = 2)
        self.juanji2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pooling2 = nn.AvgPool2d(kernel_size=2, stride = 2)
        self.flatten = nn.Flatten()
        self.fullconnect1 = nn.Linear(16*5*5, 120)
        self.fullconnect2 = nn.Linear(120, 84)
        self.fullconnect3 = nn.Linear(84, 10)



    def forward(self, x):
        x = self.juanji1(x)
        x = self.sigmod(x)
        x = self.pooling1(x)
        x = self.juanji2(x)
        x = self.sigmod(x)
        x = self.pooling2(x)
        x = self.flatten(x)
        x = self.fullconnect1(x)
        x = self.sigmod(x)
        x = self.fullconnect2(x)
        x = self.sigmod(x)
        x = self.fullconnect3(x)

        return x  

if __name__ == "__main__":#运行主函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    the_first_model = leNet().to(device)
    summary(the_first_model, (1, 28, 28))#(1,28,28)是输入的图像的维度，1是通道数，28是图像的宽和高

    import netron

        # 准备一个“诱饵”输入，让 PyTorch 追踪一下路径
    dummy_input = torch.randn(1, 1, 28, 28).to(device)

        # 导出！这一步是关键
    torch.onnx.export(the_first_model, dummy_input, "my_model.onnx")

    # 接下来，你只需要在 Netron 里打开 my_model.onnx
    netron.start("my_model.onnx")
    netron.wait()
