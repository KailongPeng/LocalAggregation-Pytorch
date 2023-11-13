'''
Pre-activation ResNet in PyTorch.
Adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_NORM_MOMENTUM = 0.003
Kailong = True


def batch_norm(num_features):
    return nn.BatchNorm2d(num_features, momentum=BATCH_NORM_MOMENTUM)


def layer_norm(num_features):
    return nn.LayerNorm(num_features)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, enable_shortcut=False, config=None):
        super(PreActBlock, self).__init__()
        if config.layer_norm:
            self.norm1 = layer_norm(in_planes)
        else:
            self.norm1 = batch_norm(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != self.expansion * planes or enable_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = batch_norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x

        out = self.conv1(out)
        out = self.conv2(self.relu2(self.norm2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, config=None):
        super(PreActBottleneck, self).__init__()
        if config.layer_norm:
            self.norm1 = layer_norm(in_planes)
        else:
            self.norm1 = batch_norm(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        if config.layer_norm:
            self.norm2 = layer_norm(planes)
        else:
            self.norm2 = batch_norm(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if config.layer_norm:
            self.norm3 = layer_norm(planes)
        else:
            self.norm3 = batch_norm(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = self.conv2(F.relu(self.norm2(out)))
        out = self.conv3(F.relu(self.norm3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=128, input_channels=3, config=None):
        super(PreActResNet, self).__init__()
        self.features_secondLastLayer = None
        self.features_lastLayer = None
        self.in_planes = 64

        # (input channel # = 3, output channel # = 64)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.padding = torch.nn.ConstantPad2d((0, 1, 0, 1), 0.)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.config = config
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # conv5_x

        if self.config.layer_norm:
            self.norm = layer_norm(512)
        else:
            self.norm = batch_norm(512)
        self.relu = nn.ReLU(inplace=True)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        first_flag = True
        for stride in strides:
            if first_flag:
                layers.append(block(self.in_planes, planes, stride, enable_shortcut=True, config=self.config))
                first_flag = False
            else:
                layers.append(block(self.in_planes, planes, stride, config=self.config))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.padding(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)


        out = self.norm(out)
        out = self.relu(out)

        # follows the TF implementation to replace avgpool with mean
        # Ref: https://github.com/neuroailab/LocalAggregation
        out = torch.mean(out, dim=(2, 3))
        if Kailong:
            self.features_secondLastLayer = out.detach().cpu().numpy()
        out = self.linear(out)  # out.shape = torch.Size([9, 128])
        if Kailong:
            self.features_lastLayer = out.detach().cpu().numpy()
        return out


# defining 5 types of PreActResNet, with different number of layers: 18, 34, 50, 101, 152
def PreActResNet18(num_classes=128, config=None):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes, config=config)


def PreActResNet34(num_classes=128, config=None):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes, config=config)


def PreActResNet50(num_classes=128, config=None):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes, config=config)


def PreActResNet101(num_classes=128, config=None):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes, config=config)


def PreActResNet152(num_classes=128, config=None):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes, config=config)
