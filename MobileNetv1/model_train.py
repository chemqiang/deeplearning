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


from model import MobileNet


def train_val_data_progress():
    train_data = FashionMNIST(root="./data",
                            train=True,
                            transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]),
                            download=True)
    

    train_data, val_data = torch.utils.data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])

    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=32,
                                                num_workers=0,
                                                shuffle=True)

    val_data_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=32,
                                                num_workers=0,
                                                shuffle=False)

    return train_data_loader, val_data_loader

train_data_loader, val_data_loader = train_val_data_progress()

def train_model_progress(model, train_dataloader, val_data_loader, num_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    train_loss_list = []

    val_loss_list = []
    
    train_acc_list = []

    val_acc_list = []

    since = time.time()


    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-"*10)

        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.train()

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            _, preds = torch.max(outputs, 1)

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            train_num += inputs.size(0)
    

        for step, (inputs, labels) in enumerate(val_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.eval()

            with torch.no_grad():
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_num += inputs.size(0)

    

        train_loss_list.append(train_loss/train_num)

        val_loss_list.append(val_loss/val_num)

        train_acc_list.append((train_corrects.double()/train_num).item())

        val_acc_list.append((val_corrects.double()/val_num).item())

        print(f"Train Loss: {train_loss_list[-1]:.4f} Acc: {train_acc_list[-1]:.4f}")
        print(f"Val Loss: {val_loss_list[-1]:.4f} Acc: {val_acc_list[-1]:.4f}")


        if val_acc_list[-1] > best_acc:

            best_acc = val_acc_list[-1]

            best_model_wts = copy.deepcopy(model.state_dict())
        
        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        print(f"Best Val Acc: {best_acc:.4f}")

        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), "best_model.pth")



    train_progress = pd.DataFrame({"train_loss": train_loss_list,
                                        "val_loss": val_loss_list,
                                        "train_acc": train_acc_list,
                                        "val_acc": val_acc_list})
        
    return train_progress
    

def matplot_acc_loss(train_progress):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_progress["train_loss"], label="Train Loss")
    plt.plot(train_progress["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Val Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_progress["train_acc"], label="Train Acc")
    plt.plot(train_progress["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train and Val Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = MobileNet(num_classes=10)
    train_data_loader, val_data_loader = train_val_data_progress()
    train_progress = train_model_progress(model, train_data_loader, val_data_loader, num_epochs=20)
    matplot_acc_loss(train_progress)