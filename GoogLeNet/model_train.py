import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchsummary import summary
from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time


from model import GoogLeNet,Inception#导入模型(重要)


def train_val_data_progress():#训练集和验证集的划分
    train_data = FashionMNIST(root="./data",    #数据集的存储路径
                            train=True,     #指定加载训练集
                            transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]),      #对图像进行预处理，先将图像调整为224x224的大小，然后将其转换为张量    
                            download=True)      #下载数据集
    


    train_data, val_data = torch.utils.data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data) )])#将训练集划分为训练集和验证集，80%作为训练集，20%作为验证集      

    train_data_loader = torch.utils.data.DataLoader(train_data,      #加载训练集
                                                batch_size=64,     #每次训练的样本数量
                                                num_workers=3,      #加载数据的线程数，0表示不使用多线程
                                                shuffle=True)       #数据是否打乱

    val_data_loader = torch.utils.data.DataLoader(val_data,         #加载验证集
                                                batch_size=64,     #每次训练的样本数量
                                                num_workers=3,      #加载数据的线程数，0表示不使用多线程
                                                shuffle=False)      #数据是否打乱

    return train_data_loader, val_data_loader

train_data_loader, val_data_loader = train_val_data_progress()

def train_model_progress(model, train_dataloader, val_data_loader, num_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#判断是否有GPU可用，如果有则使用GPU，否则使用CPU

    criterion = nn.CrossEntropyLoss()#定义损失函数为交叉熵损失函数，适用于多分类问题

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#定义优化器为Adam优化器，学习率为0.001

    model.to(device)#将模型移动到设备上

    best_model_wts = copy.deepcopy(model.state_dict())#保存模型的初始权重

    #初始化参数

    #最高准确度
    best_acc = 0.0

    #训练损失列表
    train_loss_list = []

    #验证损失列表
    val_loss_list = []  
    
    #训练准确度列表
    train_acc_list = []

    #验证准确度列表
    val_acc_list = []

    #记录训练开始时间 
    since = time.time()


    for epoch in range(num_epochs):     #循环训练指定的轮数
        print(f"Epoch {epoch+1}/{num_epochs}")      #打印当前的轮数
        print("-"*10)

    #初始化每轮的损失和正确分类的数量

        train_loss = 0.0        #训练损失
        train_corrects = 0      #训练正确分类的数量
        val_loss = 0.0          #验证损失
        val_corrects = 0        #验证正确分类的数量
        train_num = 0           #训练样本数量
        val_num = 0             #验证样本数量

        #训练阶段


        for step, (inputs, labels) in enumerate(train_dataloader):      #循环遍历训练数据加载器，获取输入和标签
            inputs = inputs.to(device)      #将输入移动到设备上
            labels = labels.to(device)      #将标签移动到设备上

            model.train()       #将模型设置为训练模式

            optimizer.zero_grad()       #清零优化器的梯度

            outputs = model(inputs)     #将输入传递给模型，得到输出

            loss = criterion(outputs, labels)       #计算损失

            loss.backward()     #反向传播计算梯度

            optimizer.step()        #更新模型参数

            _, preds = torch.max(outputs, 1)    #获取预测结果

            train_loss += loss.item() * inputs.size(0)      #累加训练损失
            train_corrects += torch.sum(preds == labels.data)       #累加正确分类的数量
            train_num += inputs.size(0)     #累加训练样本数量(训练过的样本的数量)    
    



        for step, (inputs, labels) in enumerate(val_data_loader):       #循环遍历验证数据加载器，获取输入和标签
            inputs = inputs.to(device)      #将输入移动到设备上
            labels = labels.to(device)      #将标签移动到设备上

            model.eval()        #将模型设置为评估模式

            with torch.no_grad():       #在验证阶段不计算梯度
                outputs = model(inputs)     #将输入传递给模型，得到输出

                loss = criterion(outputs, labels)       #计算损失

                _, preds = torch.max(outputs, 1)        #获取预测结果

                val_loss += loss.item() * inputs.size(0)        #累加验证损失
                val_corrects += torch.sum(preds == labels.data) #累加正确分类的数量
                val_num += inputs.size(0)       #累加验证样本数量(验证过的样本的数量)

    

        train_loss_list.append(train_loss/train_num)   #计算并保存训练损失的平均值

        val_loss_list.append(val_loss/val_num)     #计算并保存验证损失的平均值

        train_acc_list.append((train_corrects.double()/train_num).item())     #计算并保存训练准确度,double()将正确分类的数量转换为浮点数，train_num是训练样本数量

        val_acc_list.append((val_corrects.double()/val_num).item())       #计算并保存验证准确度,double()将正确分类的数量转换为浮点数，val_num是验证样本数量  

        print(f"Train Loss: {train_loss_list[-1]:.4f} Acc: {train_acc_list[-1]:.4f}")     #打印训练损失和准确度
        print(f"Val Loss: {val_loss_list[-1]:.4f} Acc: {val_acc_list[-1]:.4f}")       #打印验证损失和准确度    


        if val_acc_list[-1] > best_acc:      #如果当前的验证准确度比之前的最高准确度更高

            best_acc = val_acc_list[-1]      #更新最高准确度

            best_model_wts = copy.deepcopy(model.state_dict())      #保存当前模型的权重
        
        time_elapsed = time.time() - since#计算训练的总时间 
        print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")       #打印训练的总时间
        print(f"Best Val Acc: {best_acc:.4f}")      #打印最高验证

        #选择最优参数
        #加载最高准确率下的模型参数
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), "best_model.pth")      #保存模型参数到文件



    train_progress = pd.DataFrame({"train_loss": train_loss_list,
                                        "val_loss": val_loss_list, 
                                        "train_acc": train_acc_list, 
                                        "val_acc": val_acc_list})      #将训练损失、验证损失、训练准确度和验证准确度保存到一个DataFrame中
        
    return train_progress
    

#画图

def matplot_acc_loss(train_progress):
    plt.figure(figsize=(12, 5))     #设置图像大小

    plt.subplot(1, 2, 1)       #创建一个1行2列的子图，选择第一个子图
    plt.plot(train_progress["train_loss"], label="Train Loss")     #绘制训练损失曲线
    plt.plot(train_progress["val_loss"], label="Val Loss")         #绘制验证损失曲线
    plt.xlabel("Epoch")       #设置x轴标签
    plt.ylabel("Loss")        #设置y轴标签
    plt.title("Train and Val Loss")      #设置图像标题
    plt.legend()     #显示图例

    plt.subplot(1, 2, 2)       #选择第二个子图
    plt.plot(train_progress["train_acc"], label="Train Acc")       #绘制训练准确度曲线
    plt.plot(train_progress["val_acc"], label="Val Acc")           #绘制验证准确度曲线
    plt.xlabel("Epoch")       #设置x轴标签
    plt.ylabel("Accuracy")        #设置y轴标签
    plt.title("Train and Val Accuracy")      #设置图像标题
    plt.legend()     #显示图例

    plt.tight_layout()      #调整子图之间的间距
    plt.show()



#模型训练

if __name__ == "__main__":
    model = GoogLeNet(num_classes=10)     #创建模型实例
    train_data_loader, val_data_loader = train_val_data_progress()     #获取训练数据加载器和验证数据加载器
    train_progress = train_model_progress(model, train_data_loader, val_data_loader, num_epochs=20)      #训练模型并获取训练进度
    matplot_acc_loss(train_progress)     #绘制训练损失和准确度曲线

