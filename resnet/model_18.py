# resnet-50
import torch  
import torch.nn as nn   
import torchvision   
import numpy as np   
import os, PIL, pathlib 
from torchsummary import summary    

import torch
import torch.nn as nn
import torch.nn.functional as F
import netron 

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 卷积层"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    """ResNet-18/34 使用的基础残差块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # 如果步长不为 1 或输入输出维度不匹配，使用 1x1 卷积对齐维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 核心：捷径连接
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 初始输入层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个 Stage 的残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 全连接输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=1000):
    """实例化 ResNet-18: 每个 stage 有 2 个 BasicBlock"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# 测试模型
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=1000).to(device)
    summary(model, (1, 224, 224))

    # 准备一个“诱饵”输入，让 PyTorch 追踪一下路径
    dummy_input = torch.randn(1, 1, 224, 224).to(device)

    # 导出！这一步是关键
    torch.onnx.export(model, dummy_input, "my_model.onnx")

    # 接下来，你只需要在 Netron 里打开 my_model.onnx
    netron.start("my_model.onnx")
    netron.wait()
    