from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch
import matplotlib.pyplot as plt



train_data = FashionMNIST(root="./data",#数据集的存储路径
                        train=True,#指定加载训练集
                        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),#对图像进行预处理，先将图像调整为224x224的大小，然后将其转换为张量    
                        download=True)#下载数据集

train_data_loader = torch.utils.data.DataLoader(train_data, #加载数据集
                                                batch_size=64,#每次训练的样本数量
                                                shuffle=True)#数据是否打乱

