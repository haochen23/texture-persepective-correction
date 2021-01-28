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
        # self.conv_branchs = []
        # self.fc_blocks = []
        self.out_len = out_len
        # for i in range(out_len):
        #     self.conv_branchs.append(self.make_conv_branch(apply_norm=apply_norm,
        #                                                    norm_type=norm_type,
        #                                                    apply_dropout=apply_dropout,
        #                                                    drop_out=drop_out))
        #     self.fc_blocks.append(self.make_fc_block())
        self.conv1 = self.make_conv_branch(apply_norm=apply_norm,
                                           norm_type=norm_type,
                                           apply_dropout=apply_dropout,
                                           drop_out=drop_out)
        self.fc1 = self.make_fc_block()

        self.conv2 = self.make_conv_branch(apply_norm=apply_norm,
                                           norm_type=norm_type,
                                           apply_dropout=apply_dropout,
                                           drop_out=drop_out)
        self.fc2 = self.make_fc_block()

        self.conv3 = self.make_conv_branch(apply_norm=apply_norm,
                                           norm_type=norm_type,
                                           apply_dropout=apply_dropout,
                                           drop_out=drop_out)
        self.fc3 = self.make_fc_block()

    def make_conv_branch(self, apply_norm, norm_type, apply_dropout, drop_out):
        conv1 = ConvBlock(inChannel=2048, outChannel=128, kernel_size=3,
                          stride=1, padding=1, apply_norm=apply_norm,
                          norm_type=norm_type, apply_dropout=apply_dropout,
                          drop_out=drop_out)

        conv2 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                          stride=1, padding=1, apply_norm=apply_norm,
                          norm_type=norm_type, apply_dropout=apply_dropout,
                          drop_out=drop_out)

        conv3 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                          stride=1, padding=1, apply_norm=apply_norm,
                          norm_type=norm_type, apply_dropout=apply_dropout,
                          drop_out=drop_out)

        conv4 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                          stride=1, padding=1, apply_norm=apply_norm,
                          norm_type=norm_type, apply_dropout=apply_dropout,
                          drop_out=drop_out)

        conv5 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                          stride=1, padding=1, apply_norm=apply_norm,
                          norm_type=norm_type, apply_dropout=apply_dropout,
                          drop_out=drop_out)

        conv6 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                          stride=1, padding=1, apply_norm=apply_norm,
                          norm_type=norm_type, apply_dropout=apply_dropout,
                          drop_out=drop_out)

        conv7 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                          stride=1, padding=1, apply_norm=apply_norm,
                          norm_type=norm_type, apply_dropout=apply_dropout,
                          drop_out=drop_out)

        conv8 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                          stride=1, padding=1, apply_norm=apply_norm,
                          norm_type=norm_type, apply_dropout=apply_dropout,
                          drop_out=drop_out)

        conv9 = ConvBlock(inChannel=128, outChannel=128, kernel_size=3,
                          stride=1, padding=1, apply_norm=apply_norm,
                          norm_type=norm_type, apply_dropout=apply_dropout,
                          drop_out=drop_out)

        return nn.Sequential(
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
            conv7,
            conv8,
            conv9
        )

    def make_fc_block(self):
        return nn.Sequential(
            nn.Linear(6272, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet_bridge(x)
        # outs = []
        # for i in range(self.out_len):
        #     branch_output = self.conv_branchs[i](x)
        #     branch_output = torch.flatten(branch_output, 1)
        #     branch_output = self.fc_blocks[i](branch_output)
        #     outs.append(branch_output)
        x1 = self.conv1(x)
        x1 = torch.flatten(x1, 1)
        x1 = self.fc1(x1)

        x2 = self.conv2(x)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc2(x2)

        x3 = self.conv3(x)
        x3 = torch.flatten(x3, 1)
        x3 = self.fc3(x3)

        output = torch.cat((x1, x2, x3), dim=1)

        return output

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
    model = HomographyNet(out_len=3)
    model.eval()
    dummy_input = torch.ones([5, 3, 512, 512])
    dummy_input_2 = torch.ones([5, 3, 512, 512]) * .5
    output = model(dummy_input)
    output2 = model(dummy_input_2)
    print(output)
    print(output2)
    for param in model.parameters():
        print(param.data)
    torch.save(model, 'saved.pt')
