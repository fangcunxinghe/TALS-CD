from torch import nn


class BasicConv(nn.Module):
    """三合一基础卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class BasicConvTranspose(nn.Module):
    """三合一基础反卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConvTranspose, self).__init__()
        self.convT = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convT(x)


class BasicDepthWiseSeparableConv(nn.Module):
    """三合一深度可分离卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicDepthWiseSeparableConv, self).__init__()
        # 深度卷积
        self.DWConv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels  # 分组数等于输入通道数
            ),
        )
        # 逐点卷积
        self.PWConv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.DWConv(x)
        y = self.PWConv(y)
        return y