import os
import torch
import torch.nn as nn

from model import common
import torchvision.models as models
def make_model(args, parent=False):
    return VGG(args[0])

# reference: torchvision
class VGG(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=None):
        super(VGG, self).__init__()
        norm = None
        self.norm = norm
        bias = not args.no_bias

        configs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            'ef': [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
            'wide': [128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 1024, 1024, 1024, 'M', 1024, 1024, 1024, 'M']
        }

        body_list = []
        in_channels = args.n_colors
        if args.data_train == 'ImageNet':
            for i, v in enumerate(configs[args.vgg_type]):
                if v == 'M':
                    body_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    conv2d = conv3x3(in_channels, v, kernel_size=3)
                    if norm is not None:
                        body_list += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        body_list += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
        else:
            for v in configs[args.vgg_type]:
                if v == 'M':
                    body_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    body_list.append(common.BasicBlock(
                        in_channels, v, args.kernel_size,
                        bias=bias, conv3x3=conv3x3, norm=norm
                    ))
                    in_channels = v
            
        self.features = nn.Sequential(*body_list)

        if args.data_train.find('CIFAR') >= 0:
            n_classes = int(args.data_train[5:])
            self.classifier = nn.Linear(in_channels, n_classes)
        elif args.data_train == 'ImageNet':
            n_classes = 1000
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, n_classes),
            )
        else:
            n_classes = 10
            self.classifier = nn.Linear(in_channels, n_classes)
            
        print(conv3x3, conv3x3 == common.default_conv or conv3x3 == nn.Conv2d)

        if conv3x3 == common.default_conv or conv3x3 == nn.Conv2d:
            self.load(args, strict=True)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        return x
