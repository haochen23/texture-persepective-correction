import torch
import torch.nn as nn
import torch.nn.functional as F


class plainEncoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
        super(plainEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class plainDecoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):

        super(plainDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChannel)

        if stride == 1:
            self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outChannel)
        else:
            self.conv2 = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2)
            self.bn2 = nn.BatchNorm2d(outChannel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x





