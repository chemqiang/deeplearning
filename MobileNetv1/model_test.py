import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import MobileNet


def get_data_loaders(batch_size=128, num_workers=4):
    """加载 FashionMNIST 数据"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST 的均值和标准差
    ])

    train_dataset = FashionMNIST(
        root="./data", train=True, transform=transform, download=True
    )

    # 划分训练集和验证集 (90% / 10%)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += inputs.size(0)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct / total:.4f}"
        })

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="[Val]"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, num_epochs=30, lr=0.001, device='cuda'):
    """完整训练流程"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 学习率调度：每 10 个 epoch 衰减为 0.1 倍
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    since = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 学习率调整
        scheduler.step(val_loss)

        # 打印
        print(f"\n📊 Epoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "best_model.pth")
            print(f"💾 保存最佳模型 (Val Acc: {best_val_acc:.4f})")

    time_elapsed = time.time() - since
    print(f"\n✅ 训练完成！总耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 最佳验证准确率: {best_val_acc:.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    return model, pd.DataFrame(history)


def plot_history(history_df):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history_df['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history_df['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history_df['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history_df['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 针对 RTX 4060 8GB 优化的参数
    BATCH_SIZE = 128      # 4060 8GB 可以轻松跑 128，甚至 256
    NUM_WORKERS = 4       # 充分利用 CPU 加载数据
    NUM_EPOCHS = 30       # 足够收敛
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 数据
    print("📦 加载数据...")
    train_loader, val_loader = get_data_loaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    print(f"   训练样本: {len(train_loader.dataset)}")
    print(f"   验证样本: {len(val_loader.dataset)}")

    # 模型
    print("🧠 初始化 MobileNet...")
    model = MobileNet(num_classes=10).to(device)

    # 训练
    model, history = train(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device
    )

    # 绘图
    plot_history(history)
