import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import load_activations as la
import hierarhical_tree_gpu as ht
from qlnet_model import BS_Net
from  training_parameters import get_params

from load_data import get_data
from train import train, test
from  training_parameters import get_params

def if_exist(path):
    if not os.path.exists(path) :
        os.makedirs(path)

## 1. Load model
args = get_params()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BS_Net()
model.load_state_dict(torch.load('mnist_baseline.pth'))
model.to(device).eval()
layer_id = 1
layer = 'layer' + str(layer_id)
activation_folder = os.path.join('./activations', layer)
if_exist(activation_folder)

### 2. Load train data
train_loader = get_data(args, dataset='mnist', ifTrain=True)
## 2. Extract activations for futher look-up dictionary construction
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    _, activations = model(data)
    activation = activations[0].cpu().data.numpy()
    torch.save(activation, activation_folder + layer + '_'+str(batch_idx)+'.npy')
    if batch_idx>6:break

# 3 Construct Look-up Dictionary
# parameters for look-up dictionary construction
n_cl = 10
density = 30
max_depth = 1

# Load activations
print 'Load activations'
data = la.load_data(activation_folder) # load patched data
print 'Constract tree'
tree = ht.construct(data, n_cl, density, max_depth)
torch.save(tree, 'tree_' + layer)
