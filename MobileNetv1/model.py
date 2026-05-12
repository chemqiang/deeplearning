import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import netron

class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()

        self.ReLU = nn.ReLU()

        # 标准卷积层（第一层）
        self.juanji1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # 深度可分离卷积块 1 (32 → 64)
        self.depthwise_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)#深度卷积，groups=输入通道数
        self.bn2 = nn.BatchNorm2d(32)
        self.pointwise_conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        # 深度可分离卷积块 2 (64 → 128, stride=2)
        self.depthwise_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pointwise_conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        # 深度可分离卷积块 3 (128 → 128)
        self.depthwise_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.pointwise_conv4 = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)

        # 深度可分离卷积块 4 (128 → 256, stride=2)
        self.depthwise_conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.pointwise_conv5 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.bn9 = nn.BatchNorm2d(256)

        # 深度可分离卷积块 5 (256 → 256)
        self.depthwise_conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.pointwise_conv6 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.bn11 = nn.BatchNorm2d(256)

        # 深度可分离卷积块 6 (256 → 512, stride=2)
        self.depthwise_conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        self.pointwise_conv7 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)

        # 深度可分离卷积块 7-10 (512 → 512, stride=1，重复5次)
        self.depthwise_conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        self.pointwise_conv8 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.bn15 = nn.BatchNorm2d(512)

        self.depthwise_conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn16 = nn.BatchNorm2d(512)
        self.pointwise_conv9 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.bn17 = nn.BatchNorm2d(512)

        self.depthwise_conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn18 = nn.BatchNorm2d(512)
        self.pointwise_conv10 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.bn19 = nn.BatchNorm2d(512)

        self.depthwise_conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn20 = nn.BatchNorm2d(512)
        self.pointwise_conv11 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.bn21 = nn.BatchNorm2d(512)

        self.depthwise_conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn22 = nn.BatchNorm2d(512)
        self.pointwise_conv12 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.bn23 = nn.BatchNorm2d(512)

        # 深度可分离卷积块 11 (512 → 1024, stride=2)
        self.depthwise_conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512, bias=False)
        self.bn24 = nn.BatchNorm2d(512)
        self.pointwise_conv13 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False)
        self.bn25 = nn.BatchNorm2d(1024)

        # 深度可分离卷积块 12 (1024 → 1024)
        self.depthwise_conv14 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, groups=1024, bias=False)
        self.bn26 = nn.BatchNorm2d(1024)
        self.pointwise_conv14 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False)
        self.bn27 = nn.BatchNorm2d(1024)

        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fullconnect1 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        # 标准卷积
        x = self.juanji1(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        # 深度可分离卷积块 1
        x = self.depthwise_conv2(x)
        x = self.bn2(x)
        x = self.ReLU(x)
        x = self.pointwise_conv2(x)
        x = self.bn3(x)
        x = self.ReLU(x)

        # 深度可分离卷积块 2
        x = self.depthwise_conv3(x)
        x = self.bn4(x)
        x = self.ReLU(x)
        x = self.pointwise_conv3(x)
        x = self.bn5(x)
        x = self.ReLU(x)

        # 深度可分离卷积块 3
        x = self.depthwise_conv4(x)
        x = self.bn6(x)
        x = self.ReLU(x)
        x = self.pointwise_conv4(x)
        x = self.bn7(x)
        x = self.ReLU(x)

        # 深度可分离卷积块 4
        x = self.depthwise_conv5(x)
        x = self.bn8(x)
        x = self.ReLU(x)
        x = self.pointwise_conv5(x)
        x = self.bn9(x)
        x = self.ReLU(x)

        # 深度可分离卷积块 5
        x = self.depthwise_conv6(x)
        x = self.bn10(x)
        x = self.ReLU(x)
        x = self.pointwise_conv6(x)
        x = self.bn11(x)
        x = self.ReLU(x)

        # 深度可分离卷积块 6
        x = self.depthwise_conv7(x)
        x = self.bn12(x)
        x = self.ReLU(x)
        x = self.pointwise_conv7(x)
        x = self.bn13(x)
        x = self.ReLU(x)

        # 深度可分离卷积块 7-10（重复5次）
        x = self.depthwise_conv8(x)
        x = self.bn14(x)
        x = self.ReLU(x)
        x = self.pointwise_conv8(x)
        x = self.bn15(x)
        x = self.ReLU(x)

        x = self.depthwise_conv9(x)
        x = self.bn16(x)
        x = self.ReLU(x)
        x = self.pointwise_conv9(x)
        x = self.bn17(x)
        x = self.ReLU(x)

        x = self.depthwise_conv10(x)
        x = self.bn18(x)
        x = self.ReLU(x)
        x = self.pointwise_conv10(x)
        x = self.bn19(x)
        x = self.ReLU(x)

        x = self.depthwise_conv11(x)
        x = self.bn20(x)
        x = self.ReLU(x)
        x = self.pointwise_conv11(x)
        x = self.bn21(x)
        x = self.ReLU(x)

        x = self.depthwise_conv12(x)
        x = self.bn22(x)
        x = self.ReLU(x)
        x = self.pointwise_conv12(x)
        x = self.bn23(x)
        x = self.ReLU(x)

        # 深度可分离卷积块 11
        x = self.depthwise_conv13(x)
        x = self.bn24(x)
        x = self.ReLU(x)
        x = self.pointwise_conv13(x)
        x = self.bn25(x)
        x = self.ReLU(x)

        # 深度可分离卷积块 12
        x = self.depthwise_conv14(x)
        x = self.bn26(x)
        x = self.ReLU(x)
        x = self.pointwise_conv14(x)
        x = self.bn27(x)
        x = self.ReLU(x)

        # 分类头
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fullconnect1(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNet(num_classes=10).to(device)
    summary(model, (1, 224, 224))
    # 准备一个“诱饵”输入，让 PyTorch 追踪一下路径
    dummy_input = torch.randn(1, 1, 224, 224).to(device)

    # 导出！这一步是关键
    torch.onnx.export(model, dummy_input, "my_model.onnx")

    # 接下来，你只需要在 Netron 里打开 my_model.onnx
    netron.start("my_model.onnx")
    netron.wait()
    
