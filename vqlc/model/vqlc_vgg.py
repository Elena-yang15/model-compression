import os
import torch
import torch.nn as nn
from model import common
from model import compressed_layer
import torch.nn.functional as F


def make_model(args, parent=False):
    return Vqlc_VGG(args)


class CompressedBlock(nn.Module):
    def __init__(self, num_classes, dropout, embe_dim, num_clusters, cluster_size, channels, kernel_size):
        super(CompressedBlock, self).__init__()
        self.d = embe_dim  # the dimension of embedding vectors
        self.k = num_clusters  # the number of weight clusters
        self.n = kernel_size  # kernel_size
        self.V = torch.nn.Parameter(
            torch.randn(self.k, cluster_size, self.n, self.n)) 
        torch.nn.init.normal_(self.V, mean = 0, std = 0.5)
        self.register_buffer('K', torch.randn(self.d, self.k))
        torch.nn.init.normal_(self.K, mean=0, std=1.5)
        self.atten = compressed_layer.attention3DCNNLayer(embe_dim, cluster_size, channels, kernel_size, bias=True)

        self.bn = common.default_norm(channels[1])
        self.relu = common.default_act
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x, loss2 = self.atten(x, self.K, self.V)
        x = self.dropout(self.relu(self.bn(x)))
        return x, loss2
    


class BasicBlock(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super(BasicBlock, self).__init__()
        modules = [common.default_conv(channels[0], channels[1], kernel_size=1, stride=1, bias=True)]
        modules.append(common.default_norm(channels[1]))
        modules.append(common.default_act)
        modules.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


# reference: torchvision
class Vqlc_VGG(nn.Module):
    def __init__(self, args):
        super(Vqlc_VGG, self).__init__()

        args = args[0]
        norm = common.default_norm
        bias = not args.no_bias
        num_clusters = args.num_clusters
        cluster_size = args.cluster_size
        embe_dim = args.embe_dim
        dropout = args.dropout

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
        for i, v in enumerate(configs[args.vgg_type][:4]):
            if v == 'M':
                body_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                body_list.append(BasicBlock([in_channels, v], args.kernel_size, dropout))
                in_channels = v

        compressed_list = []
        for v in configs[args.vgg_type][4:]:
            if v == 'M':
                compressed_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                compressed_list.append(CompressedBlock(args.num_classes, dropout, embe_dim, num_clusters, cluster_size, [in_channels, v], args.kernel_size))
                in_channels = v

        # assert(args.data_train.find('CIFAR') >= 0)
        self.features = nn.Sequential(*body_list)
        self.compress = nn.Sequential(*compressed_list)
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

    def forward(self, x):
        x = self.features(x)
        x, loss2 = self.compress(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, loss2

