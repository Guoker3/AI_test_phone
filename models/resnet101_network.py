import torch
import torch.nn as nn
import torch.nn.functional as F


class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=6):
        x = img

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()
        # att_size=6:输出的特征图将是6x6的尺寸
        att = F.adaptive_avg_pool2d(x, (att_size, att_size)).squeeze().permute(1, 2, 0)

        return fc, att