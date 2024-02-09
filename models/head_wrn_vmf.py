import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

#from wrn_vmf import *

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        #import pdb
        #pdb.set_trace()

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels) # 64 x 128
        features = out.view(-1, self.nChannels) # 64 x 128
        #out = self.fc(out)
        #return self.fc(out)
        #return out
        return self.fc(features), features

    def intermediate_forward(self, x, layer_index):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out
    
    def feature_list(self, x):
        out_list = [] 
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), out_list
         




class HeadWideResNet(nn.Module):
    """encoder + head"""
    #def __init__(self, args, num_classes):
    def __init__(self, args):
        super(HeadWideResNet, self).__init__()
        #model_fun, dim_in = model_dict[args.model]
        #if args.in_dataset == 'ImageNet-100':
        #    model = models.resnet34(pretrained=True)
        #    for name, p in model.named_parameters():
        #        if not name.startswith('layer4'):
        #            p.requires_grad = False

        num_classes = 10
        layers = 40
        widen_factor = 2
        droprate = 0.3

        #model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
        model = WideResNet(layers, num_classes, widen_factor, droprate)
        model_name = '/u/b/a/baihaoyue/private/ood_detection/woods_ood/CIFAR/snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt'
        if os.path.isfile(model_name):
            print('found pretrained model')
            #net.load_state_dict(torch.load(model_name)['params'])
            model.load_state_dict(torch.load(model_name))
            print('Model restored!')
        dim_in = model.fc.in_features
        #modules=list(model.children())[:-1] # remove last linear layer
        #modules=list(model.children())
        #self.encoder =nn.Sequential(*modules)
        self.encoder = model
        #else:
        #    self.encoder = model_fun()
        #self.fc = nn.Linear(dim_in, args.n_cls)
        self.fc = nn.Linear(dim_in, num_classes)
        #self.multiplier = multiplier
        if args.head == 'linear':
            self.head = nn.Linear(dim_in, args.feat_dim)
        elif args.head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, args.feat_dim)
            )
    
    def forward(self, x):
        #import pdb
        #pdb.set_trace()
        #feat = self.encoder(x).squeeze()
        x, feat = self.encoder(x)
        unnorm_features = self.head(feat)
        features= F.normalize(unnorm_features, dim=1)
        return features
    
    def intermediate_forward(self, x):
        feat = self.encoder(x).squeeze()
        return F.normalize(feat, dim=1)




