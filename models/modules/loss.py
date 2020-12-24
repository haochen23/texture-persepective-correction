import torch
import torch.nn as nn


class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output - target + 1e-16, p=2, dim=1).mean()
        return lossvalue


if __name__ == '__main__':
    lossObj1 = EPELoss()
    lossObj2 = nn.MSELoss()
    output = torch.randn([1, 3, 256, 256])
    target = torch.ones([1, 3, 256, 256])
    loss_1 = lossObj1(output, target)
    loss_2 = lossObj2(output, target)
    print(loss_1, loss_2)