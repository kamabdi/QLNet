import torch.nn.functional as F
import torch.nn as nn
from quantizer_gpu import Quantizer

class BS_Net(nn.Module):
    def __init__(self):
        super(BS_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(16*20, 50) # Fully Connected Layers
        self.fc2 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(50)
        self.activation = nn.ReLU()

    def forward(self, x, n=0, tree=None):
        layer1 = self.activation(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        if n==1:
             layer1 = self.quantize_activation(layer1, True, tree[n-1], 'lookup_table')

        layer2 = self.activation(F.max_pool2d(self.bn2(self.conv2(layer1)), 2))

        x = layer2.view(-1, 16*20) # flatten input to feed it to fully connected layer
        x = self.activation(self.bn3(self.fc1(x)))
        x = F.dropout(x, p=0.25)
        x = self.fc2(x)
        return x

    def quantize_activation(self, input, ifTraining, tree, lookup_table):
        # return Quantizer(ifQuantizing, ifTraining, tree, lookup_table).apply(input)
        return Quantizer().apply(input, ifTraining, tree, lookup_table)
