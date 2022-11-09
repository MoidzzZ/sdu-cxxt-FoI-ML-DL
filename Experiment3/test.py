import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    print(torch.cuda.get_device_properties(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 只加载test数据
def load_data(batch_size):
    test_augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010))])

    data_test = torchvision.datasets.CIFAR10(root="data", train=False, transform=test_augs, download=False)
    return data.DataLoader(data_test, batch_size, shuffle=False, num_workers=0)


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, change=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if change:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())

block2 = nn.Sequential(Residual(64, 64), Residual(64, 64))

block3 = nn.Sequential(Residual(64, 128, change=True, strides=2), Residual(128, 128))

block4 = nn.Sequential(Residual(128, 256, change=True, strides=2), Residual(256, 256))

block5 = nn.Sequential(Residual(256, 512, change=True, strides=2), Residual(512, 512))

net = nn.Sequential(block1, block2, block3, block4, block5,
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 10))

net.load_state_dict(torch.load('test.pth'))
net = net.to(device)


def test(net, test_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式，停止dropout和batchnorm
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    # 同样要阻挡梯度的传播
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)
            # 后者为数组中元素的个数
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


test_iter = load_data(256)

test_acc = test(net, test_iter, device)
print("准确率：%.03f" % test_acc)
