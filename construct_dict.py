import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import load_activations as la
import hierarhical_tree_gpu as ht

from load_data import get_data
# from train_utils import train, test

from train_utils_quantized import train, test
from  training_parameters import get_params

def if_exist(path):
    if not os.path.exists(path) :
        os.makedirs(path)
# './models/mnist_baseline.pth'
## 1. Load model
args = get_params()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.model=="resnet":
    from model import ResNet11
    if args.dataset=="fashion" or args.dataset=="mnist": channels=1
    else: channels=3
    model = ResNet11(args.num_classes, channels)
elif args.dataset=="german":
    # from qlnet_model_quantized import BS_Net_German
    from model import BS_Net_German
    model = BS_Net_German(layer=args.layer-1, input_size=3, num_classes=48)
else:
    from model import BS_Net
    model = BS_Net()

model.load_state_dict(torch.load(args.pretrained_model))
model.to(device).eval()
layer_id = args.layer
layer = 'layer' + str(layer_id)
activation_folder = os.path.join('./activations', args.dataset, args.model, layer)
if_exist(activation_folder)

### 2. Load train data
# train_loader = get_data(args, dataset=args.dataset, ifTrain=True, ifShuffle=True)
train_loader = get_data(args, dataset=args.dataset, ifTrain=False, ifShuffle=True)
## 2. Extract activations for futher look-up dictionary construction

if args.save_activations:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        _, activations = model(data)
        activation = activations[layer_id-1].cpu().data.numpy()
        torch.save(activation, os.path.join(activation_folder, layer + '_'+str(batch_idx)+'.npy'))
        # if batch_idx>8:break

# 3 Construct Look-up Dictionary
# parameters for look-up dictionary construction
density = 30
n_cl = args.n_cl
max_depth = args.max_depth

# Load activations
print('Load activations of ' + layer)
data = la.load_data(activation_folder) # load patched data
print('Constract tree')
tree = ht.construct(data, n_cl, density, max_depth)
torch.save(tree, os.path.join('dictionary', args.model+'_'+args.dataset + '_tree_' + layer + '_depth_' + str(args.max_depth) + '_n_cl_' + str(args.n_cl)))
