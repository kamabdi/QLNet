import torch.nn.functional as F
import torch.nn as nn
from quantizer_gpu import Quantizer

import numpy as np

class BS_Net(nn.Module):
    def __init__(self):
        super(BS_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.fc1 = nn.Linear(16*40, 50) # Fully Connected Layers
        self.fc2 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(40)
        self.bn3 = nn.BatchNorm1d(50)
        self.activation = nn.ReLU()
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x, layer=0, tree=None):
        if tree is None:
            prediction, activations = self.forward_baseline(x)
        else:
            # print("Noisy model")
            prediction, activations = self.forward_quantized(x, layer, tree)

        return prediction, activations

    def forward_baseline(self, x):
        x = self.conv1(x)
        activations=[x]
        x = self.activation(F.max_pool2d(self.bn1(x), 2))
        x = self.conv2(x)
        activations = np.concatenate((activations, [x]))
        x = self.activation(F.max_pool2d(self.bn2(x), 2))

        x = x.view(-1, 16*40) # flatten input to feed it to fully connected layer
        x = self.activation(self.bn3(self.fc1(x)))
        x = self.drop(x)
        x = self.fc2(x)
        # return F.log_softmax(x)
        return x, activations

    def forward_quantized(self, x, layer=0, tree=None):
        layer1 = self.conv1(x)
        if layer>=1:
            layer1 = self.quantize_activation(layer1, True, tree[0], 'lookup_table')
        layer1 = self.activation(F.max_pool2d(self.bn1(layer1), 2))

        layer2 = self.conv2(layer1)
        if layer>=2:
            layer2 = self.quantize_activation(layer2, True, tree[1], 'lookup_table')
        layer2 = self.activation(F.max_pool2d(self.bn2(layer2), 2))

        x = layer2.view(-1, 16*40) # flatten input to feed it to fully connected layer
        x = self.activation(self.bn3(self.fc1(x)))
        x = self.drop(x)
        x = self.fc2(x)
        return x, [layer1, layer2]

    def quantize_activation(self, input, ifTraining, tree, lookup_table):
        return Quantizer().apply(input, ifTraining, tree, lookup_table)



class BS_Net_German(nn.Module):
    def __init__(self, layer=0, input_size=1, num_classes=48):
        super(BS_Net_German, self).__init__()

        self.conv1 = nn.Conv2d(3, 150, kernel_size=7)
        self.bn1 = nn.BatchNorm2d(150)
        self.conv2 = nn.Conv2d(150, 200, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(200)
        self.conv3 = nn.Conv2d(200, 300, kernel_size=4)
        self.bn3 = nn.BatchNorm2d(300)
        self.fc1 = nn.Linear(300 * 3 * 3, 350)
        self.bn4 = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.conv_drop = nn.Dropout2d(p=0.2)
        self.fc_drop = nn.Dropout(p=0.5)


    def forward(self, x, layer=0, tree=None):
        if tree is None:
            prediction, activations = self.forward_baseline_german(x, layer)
        else:
            prediction, activations = self.forward_quantized_german(x, layer, tree)

        return prediction, activations


    def forward_baseline_german(self, x, layer):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        activations=[x]
        x = self.conv_drop(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        activations = np.concatenate((activations, [x]))
        x = self.conv_drop(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        activations = np.concatenate((activations, [x]))
        x = self.conv_drop(x)
        x = x.view(-1, 300 * 3 * 3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(self.fc_drop(x))
        return x, activations


    def forward_quantized_german(self, x, layer, tree):
        x = F.relu(self.bn1(self.conv1(x)))
        if layer>=1:
            x = self.quantize_activation(x, True, tree[0], 'lookup_table')
        x = self.conv_drop(self.pool(x))

        x = F.relu(self.bn2(self.conv2(x)))
        if layer>=2:
            x = self.quantize_activation(x, True, tree[1], 'lookup_table')
        x = self.conv_drop(self.pool(x))

        x = F.relu(self.bn3(self.conv3(x)))
        if layer>=3:
            x = self.quantize_activation(x, True, tree[2], 'lookup_table')
        x = self.conv_drop(self.pool(x))

        x = x.view(-1, 300 * 3 * 3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(self.fc_drop(x))

        return x, x

    def quantize_activation(self, input, ifTraining, tree, lookup_table):
        return Quantizer().apply(input, ifTraining, tree, lookup_table)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=0, tree=None):
        if tree is None:
            prediction, activations = self.forward_baseline(x, layer)
            #prediction, activations = self.forward_quantized(x, 0, tree=None)
        else:
            prediction, activations = self.forward_quantized(x, layer, tree)

        return prediction, activations

    def forward_baseline(self, x, n, tree=None):
        out = F.relu(self.bn1(self.conv1(x)))
        activations=[out]
        out = self.layer1(out)
        activations = np.concatenate((activations, [out]))
        out = self.layer2(out)
        activations = np.concatenate((activations, [out]))
        out = self.layer3(out)
        activations = np.concatenate((activations, [out]))
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, activations


    def forward_quantized(self, x, n, tree=None):
        out = F.relu(self.bn1(self.conv1(x)))
        l1 = out
        if n>=1:
            l1 = self.quantize_activation(l1, True, tree[0], 'lookup_table')
        # print(out.shape)
        out = self.layer1(l1)
        l2 = out
        if n>=2:
            l2 = self.quantize_activation(l2, True, tree[1], 'lookup_table')
        # print(out.shape)
        out = self.layer2(l2)
        l3 = out
        if n>=3:
            l3 = self.quantize_activation(l3, True, tree[2], 'lookup_table')
        # print(out.shape)
        out = self.layer3(l3)
        l4 = out
        if n>=4:
            l4 = self.quantize_activation(l4, True, tree[3], 'lookup_table')
        # print(out.shape)
        out = self.layer4(l4)
        # print(out.shape)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, [l1, l2, l3, l4]

    def quantize_activation(self, input, ifTraining, tree, lookup_table):
        # return Quantizer(ifQuantizing, ifTraining, tree, lookup_table).apply(input)
        return Quantizer().apply(input, ifTraining, tree, lookup_table)



def ResNet11(num_classes, channels):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes, channels)

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
