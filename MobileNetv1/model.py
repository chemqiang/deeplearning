import torch
import torch.nn as nn
from torchsummary import summary
import netron


class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU"""
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DWConvPWConv(nn.Module):
    """深度可分离卷积：DWConv + PWConv"""
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.dw = ConvBNReLU(in_c, in_c, kernel=3, stride=stride, padding=1, groups=in_c)
        self.pw = ConvBNReLU(in_c, out_c, kernel=1, stride=1, padding=0)

    def forward(self, x):
        return self.pw(self.dw(x))


class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # 第一层：标准卷积
            ConvBNReLU(1, 32, stride=2),

            # 块1: 32 → 64, stride=1
            DWConvPWConv(32, 64, stride=1),

            # 块2: 64 → 128, stride=2 (下采样)
            DWConvPWConv(64, 128, stride=2),

            # 块3: 128 → 128, stride=1
            DWConvPWConv(128, 128, stride=1),

            # 块4: 128 → 256, stride=2 (下采样)
            DWConvPWConv(128, 256, stride=2),

            # 块5: 256 → 256, stride=1
            DWConvPWConv(256, 256, stride=1),

            # 块6: 256 → 512, stride=2 (下采样)
            DWConvPWConv(256, 512, stride=2),

            # 块7-11: 512 → 512, stride=1 (重复5次)
            DWConvPWConv(512, 512, stride=1),
            DWConvPWConv(512, 512, stride=1),
            DWConvPWConv(512, 512, stride=1),
            DWConvPWConv(512, 512, stride=1),
            DWConvPWConv(512, 512, stride=1),

            # 块12: 512 → 1024, stride=2 (下采样)
            DWConvPWConv(512, 1024, stride=2),

            # 块13: 1024 → 1024, stride=1
            DWConvPWConv(1024, 1024, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNet(num_classes=10).to(device)
    summary(model, (1, 224, 224))

    dummy = torch.randn(1, 1, 224, 224).to(device)
    torch.onnx.export(model, dummy, "my_model.onnx")
    print("✅ ONNX 模型已导出")
    netron.start("my_model.onnx")
    netron.wait()