from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from load_data import get_data
from qlnet_model_quantized import BS_Net
from train_utils_quantized import train, test
from  training_parameters import get_params


args = get_params()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader = get_data(args, dataset='mnist', ifTrain=True)
test_loader = get_data(args, dataset='mnist', ifTrain=False)

model = BS_Net()
model.load_state_dict(torch.load('mnist_baseline.pth'))
model.to(device).eval()

layer_id = 1
layer = 'layer' + str(layer_id)
tree = torch.load('tree_' + layer)
nodes = np.asarray([tree[i].centroid for i in range (0, np.shape(tree)[0])])

lookup_table = torch.FloatTensor(nodes).to(device).unsqueeze(0)
# lookup_table = torch.FloatTensor(nodes).to(device)
print(lookup_table.size())
train(model,train_loader, test_loader, args.epochs + 1, args.lr, device, layer_id, [lookup_table])


# test(model, test_loader, device)
