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


#导入之前写好的用法和模型
from model import vGG16


def test_data_progress():#测试集的划分
    test_data = FashionMNIST(root="./data",    #数据集的存储路径
                            train=False,     #指定加载测试集
                            transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]),      #对图像进行预处理，先将图像调整为224x224的大小，然后将其转换为张量    
                            download=True)      #下载数据集
    test_data_loader = torch.utils.data.DataLoader(test_data,      #加载测试集
                                                batch_size=1,     #每次测试的样本数量
                                                num_workers=0,      #加载数据的线程数，0表示不使用多线程
                                                shuffle=True)       #数据是否打乱
    return test_data_loader

def train_model_progress(model, test_dataloader, num_epochs=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#判断是否有GPU可用，如果有则使用GPU，否则使用CPU

    model.to(device)#将模型移动到设备上

    #初始化参数

    test_correct = 0.0
    test_num = 0


    with torch.no_grad():#在测试过程中不计算梯度，节省内存和计算资源,此处只进行前向传播，不进行梯度计算和反向传播
        
        for inputs, labels in test_dataloader:#遍历测试数据加载器中的每个批次，获取输入数据和对应的标签

            inputs = inputs.to(device)#将输入数据移动到设备上
            labels = labels.to(device)#将标签移动到设备上

            outputs = model(inputs)#将输入数据传入模型，得到输出结果

            model.eval()#将模型设置为评估模式，关闭dropout和batch normalization等训练时特有的层的行为

            _, preds = torch.max(outputs, dim = 1)#获取输出结果中每行的最大值的索引，即预测的类别

            test_correct += torch.sum(preds == labels.data).item()#统计预测正确的数量

            test_num += inputs.size(0)#统计测试样本的总数量


    test_acc = test_correct / test_num#计算测试准确率  
    
    print("测试集的准确率为：{:.4f}".format(test_acc))#输出测试集的准确率


if __name__ == "__main__":#运行主函数
    model = vGG16()#实例化模型

    model.load_state_dict(torch.load("./best_model.pth"))#加载之前的训练的最好的模型的参数，注意这里容易出错

    test_dataloader = test_data_progress()#获取测试数据加载器

    # train_model_progress(model, test_dataloader)  #这个是用上面写的函数来测试模型的准确率，下面写的是能够看到过程的模型推理的测试过程


    #模型的推理
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#创建设备
    model.to(device)#将模型放到设备上
    with torch.no_grad():#在测试过程中不计算梯度，节省内存和计算资源,此处只进行前向传播，不进行梯度计算和反向传播
        for inputs, labels in test_dataloader:#遍历测试数据加载器中的每个批次，获取输入数据和对应的标签

            inputs = inputs.to(device)#将输入数据移动到设备上
            labels = labels.to(device)#将标签移动到设备上

            outputs = model(inputs)#将输入数据传入模型，得到输出结果

            model.eval()#将模型设置为评估模式，关闭dropout和batch normalization等训练时特有的层的行为

            _, preds = torch.max(outputs, dim = 1)#获取输出结果中每行的最大值的索引，即预测的类别

            result = preds.item()

            print("预测类别：{}  ------------- 真实类别：{}".format(result, labels.item()))#输出预测的类别和真实的类别

