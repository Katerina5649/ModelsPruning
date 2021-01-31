import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Block(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsize=None):
        
        super(Block, self).__init__()
        # 32 -> 16, 16->8 need stride=2
        self.stride = stride
        
        # each block contains 2 3x3 conv
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        # second filter always use stride=1 to keep orginal size
        
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.downsize = downsize # a function used to make the size conform
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsize is not None:
            # resize the volume that goes through the skip connection
            residual = self.downsize(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block):
        super(ResNet, self).__init__()
        
        self.in_channel = 32
        self.downsize = None
        
        # very first layer, change [128 3 32 32] to [128 16 32 32]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        # input image: 32 x 32, then 3x3x16 conv with stride = 1 and pad = 1, so we have 32x32x16 in the first block
        self.layer1 = self.build_layer(block, 32, 3, stride=2)
        self.layer2 = self.build_layer(block, 64, 3, stride=2)
        self.layer3 = self.build_layer(block, 128, 3, stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        self.linear = nn.Linear(128, 10)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_() 
        
    def build_layer(self, block, out_channel, num_block=3, stride=1):
        if stride != 1 or self.in_channel != out_channel:
            self.downsize = nn.Sequential(
                # use 1x1 conv to do the size matching
                nn.Conv2d(self.in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.downsize = None
        # should have 3 layers
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, self.downsize))
        self.in_channel = out_channel
        # only the first block need to downsize
        for i in range(1, num_block):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 32x32x3 -> 32x32x16
        out = self.conv1(x)
        out = self.bn1(out)     
        out = self.relu(out)

        out = self.layer1(out)   # 32x32x16
        out = self.layer2(out)   # 16x16x32
        out = self.layer3(out)   # 8x8x64

        out = F.avg_pool2d(input=out, kernel_size=out.size(3))
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)

        return out