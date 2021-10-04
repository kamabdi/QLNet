import torch.nn.functional as F
import torch.nn as nn

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

    def forward(self, x):
        x = self.activation(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = self.activation(F.max_pool2d(self.bn2(self.conv2(x)), 2))

        x = x.view(-1, 16*40) # flatten input to feed it to fully connected layer
        x = self.activation(self.bn3(self.fc1(x)))
        x = F.dropout(x, p=0.25)
        x = self.fc2(x)
        # return F.log_softmax(x)
        return x


class BS_Net_German(nn.Module):
    def __init__(self, layer=2, input_size=1, num_classes=48):
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


    def forward(self, x, layer=3, tree=None):
        if tree is None:
            prediction, activations = self.forward_baseline_german(x, layer)
        else:
            # print("Noisy model")
            prediction, activations = self.forward_noisy_german(x, mask, layer, noise)

        return prediction


    def forward_baseline_german(self, x, layer):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv_drop(self.pool(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv_drop(self.pool(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv_drop(self.pool(x))
        x = x.view(-1, 300 * 3 * 3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(self.fc_drop(x))
        return x, x


    def forward_noisy_german(self, x, mask, layer, noise):
        x = self.noisy_conv2d(x, self.conv1, torch.unsqueeze(mask[0], 1), noise)
        ab=[F.adaptive_avg_pool2d(x,  (1, 1))]
        x = self.activation(self.bn1(x))
        x = self.pool(x)

        x = self.noisy_conv2d(x, self.conv2, torch.unsqueeze(mask[1], 1), noise)
        ab = np.concatenate((ab, [F.adaptive_avg_pool2d(x,  (1, 1))]))
        x = self.activation(self.bn2(x))
        x = self.pool(x)

        if layer>2:
            x = self.noisy_conv2d(x, self.conv3, torch.unsqueeze(mask[2], 1), noise)
            ab = np.concatenate((ab, [F.adaptive_avg_pool2d(x,  (1, 1))]))
            x = self.activation(self.bn3(x))
            x = self.pool(x)
        if layer>3:
            x = self.noisy_conv2d(x, self.conv4, torch.unsqueeze(mask[3], 1), noise)
            ab = np.concatenate((ab, [F.adaptive_avg_pool2d(x,  (1, 1))]))
            x = self.activation(self.bn4(x))
            x = self.pool(x)

        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) # flatten input to feed it to fully connected layer
        x = self.activation(self.bn_fc(self.fc1(x)))
        x = self.drop(x)
        x = self.fc2(x)

        return x, ab


    def noisy_conv2d(self, input, filter, mask, scale):
        noise = scale*torch.rand_like(filter.weight)*mask
        x = F.conv2d(input, torch.nn.Parameter(filter.weight + noise.detach()), bias=filter.bias, stride=filter.stride, padding=filter.padding)
        return x
