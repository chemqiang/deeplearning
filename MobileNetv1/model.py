import torch
import torch.nn as nn
from torchsummary import summary
import netron


class ConvBlock(nn.Module):
    """标准卷积块：Conv + BN + ReLU"""
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块：DWConv + PWConv"""
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.depthwise = ConvBlock(in_c, in_c, kernel=3, stride=stride, padding=1, groups=in_c)
        self.pointwise = ConvBlock(in_c, out_c, kernel=1, stride=1, padding=0)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MobileNet(nn.Module):
    """MobileNet-V1 — 极致简洁版"""
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        
        def _ch(c):
            return max(1, int(c * width_mult))

        # 网络配置：[输入通道, 输出通道, stride, 重复次数]
        config = [#
            (_ch(32),  _ch(64),   1, 1),   # block 1
            (_ch(64),  _ch(128),  2, 1),   # block 2
            (_ch(128), _ch(128),  1, 1),   # block 3
            (_ch(128), _ch(256),  2, 1),   # block 4
            (_ch(256), _ch(256),  1, 1),   # block 5
            (_ch(256), _ch(512),  2, 1),   # block 6
            (_ch(512), _ch(512),  1, 5),   # block 7-11 (重复5次)
            (_ch(512), _ch(1024), 2, 1),   # block 12
            (_ch(1024),_ch(1024), 1, 1),   # block 13
        ]

        # 动态构建所有层
        layers = []
        layers.append(ConvBlock(1, _ch(32), stride=2))
        
        for in_c, out_c, stride, repeat in config:
            for _ in range(repeat):
                layers.append(DepthwiseSeparableConv(in_c, out_c, stride))
                in_c = out_c
                stride = 1
        
        self.features = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(_ch(1024), num_classes)
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