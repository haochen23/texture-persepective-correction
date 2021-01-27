# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from matplotlib import pyplot as plt
from models.modules.building_blocks import ConvBlock

class HomographyNet(nn.Module):

    def __init__(self, apply_norm=False, norm_type='BatchNorm',
                 apply_dropout=False, drop_out=0.4, out_len=5):
        super(HomographyNet, self).__init__()
        pretrained_model = models.resnet50(True)
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.resnet_bridge = nn.Sequential(*list(pretrained_model.children())[:-2])

        self.conv1 = ConvBlock(inChannel=2048, outChannel=128, kernel_size=3,
                               stride=1, padding=1, apply_norm=apply_norm,
                               norm_type=norm_type, apply_dropout=apply_dropout,
                               drop_out=drop_out)

        self.conv2 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                               stride=1, padding=1, apply_norm=apply_norm,
                               norm_type=norm_type, apply_dropout=apply_dropout,
                               drop_out=drop_out)

        self.conv3 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                               stride=1, padding=1, apply_norm=apply_norm,
                               norm_type=norm_type, apply_dropout=apply_dropout,
                               drop_out=drop_out)

        self.conv4 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                               stride=1, padding=1, apply_norm=apply_norm,
                               norm_type=norm_type, apply_dropout=apply_dropout,
                               drop_out=drop_out)

        self.conv5 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                               stride=1, padding=1, apply_norm=apply_norm,
                               norm_type=norm_type, apply_dropout=apply_dropout,
                               drop_out=drop_out)

        self.conv6 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                               stride=1, padding=1, apply_norm=apply_norm,
                               norm_type=norm_type, apply_dropout=apply_dropout,
                               drop_out=drop_out)

        self.conv7 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                               stride=1, padding=1, apply_norm=apply_norm,
                               norm_type=norm_type, apply_dropout=apply_dropout,
                               drop_out=drop_out)

        self.conv8 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                               stride=1, padding=1, apply_norm=apply_norm,
                               norm_type=norm_type, apply_dropout=apply_dropout,
                               drop_out=drop_out)

        self.conv9 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                               stride=1, padding=1, apply_norm=apply_norm,
                               norm_type=norm_type, apply_dropout=apply_dropout,
                               drop_out=drop_out)

        self.fc_block = nn.Sequential(
            nn.Linear(6272, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_len),
            nn.Sigmoid())

    def forward(self, x):
        x = self.resnet_bridge(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = torch.flatten(x, 1)
        x = self.fc_block(x)

        return x

    # def print_grad(self, module, grad_input, grad_output):
    #     print('Inside ' + module.__class__.__name__ + ' backward')
    #     # print('grad_input size:', grad_input[0].size())
    #     # print('grad_output size:', grad_output[0].size())
    #     print('grad_input norm:', grad_input[0].norm())
    #
    # def visualize_activation(self, module, input, output):
    #     feature_map = output.cpu().data.numpy()
    #     a, filter_range, x, y = np.shape(feature_map)
    #     fig = plt.figure(figsize=(y * 0.07, x * 2))
    #     # fig = plt.figure()
    #
    #     for i in range(filter_range):
    #         ax = fig.add_subplot(filter_range, 3, i + 1)
    #         ax.set_xticklabels([])
    #         ax.set_yticklabels([])
    #         ax.set_aspect('equal')
    #
    #         activation = feature_map[0, i]
    #         if (i == 0):
    #             print("Activation shape :", np.shape(activation))
    #
    #         # = cv2.resize(activation, (y * resize_scale, x * resize_scale), interpolation = cv2.INTER_CUBIC)  # scale image up
    #         ax.imshow(np.squeeze(activation), cmap='gray')
    #         ax.set_title('%s' % str(i + 1))
    #
    #     plt.subplots_adjust(wspace=0, hspace=0.35)
    #     plt.show()


if __name__ == '__main__':
    model = HomographyNet()
    dummy_input = torch.randn([5, 3, 512, 512])
    output = model(dummy_input)
    print(output.shape)
    torch.save(model, 'saved.pt')
