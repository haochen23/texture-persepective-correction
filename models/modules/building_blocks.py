import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, inChannel, outChannel, kernel_size,
                 stride, padding, apply_norm=False,
                 norm_type='BatchNorm', apply_dropout=False, drop_out=0.4):
        super(ConvBlock, self).__init__()
        conv_block = []
        conv = nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=kernel_size, stride=stride,
                         padding=padding)
        nn.init.xavier_normal_(conv.weight)
        conv_block += [conv]
        if apply_norm:
            if norm_type.lower() == "batchnorm":
                norm_layer = nn.BatchNorm2d(outChannel)
            elif norm_type.lower() == "instancenorm":
                norm_layer = nn.InstanceNorm2d(outChannel)
            else:
                raise NotImplementedError(f"{norm_type} is not implemented in ConvBlock")
            conv_block += [norm_layer]

        conv_block += [nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                       nn.ReLU()]

        if apply_dropout:
            conv_block += [nn.Dropout(drop_out)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


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


class resEncoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):

        super(resEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outChannel))

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


class resDecoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):

        super(resDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChannel)

        self.downsample = None

        if stride == 1:
            self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outChannel)
        else:
            self.conv2 = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2)
            self.bn2 = nn.BatchNorm2d(outChannel)

            self.downsample = nn.Sequential(
                nn.ConvTranspose2d(inChannel, outChannel, kernel_size=1, stride=2, output_padding=1),
                nn.BatchNorm2d(outChannel))

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out
