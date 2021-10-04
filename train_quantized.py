from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os


from load_data import get_data
# from qlnet_model_quantized import BS_Net
from model import BS_Net, BS_Net_German
from train_utils_quantized import train, test
from  training_parameters import get_params


args = get_params()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader = get_data(args, dataset=args.dataset, ifTrain=True, ifShuffle=True)
test_loader = get_data(args, dataset=args.dataset, ifTrain=False, ifShuffle=False)

if args.model=="resnet":
	from model import ResNet11
	if args.dataset=="fashion" or args.dataset=="mnist": channels=1
	else: channels=3

	print("We use ResNet")
	model = ResNet11(args.num_classes, channels)
elif args.dataset=="german":
    model = BS_Net_German(layer=0, input_size=3, num_classes=48)
else:
	# from qlnet_model_quantized import BS_Net
	print("We use %s"%args.model)
	model = BS_Net()

model.load_state_dict(torch.load(args.pretrained_model))
model.to(device).eval()

layer_id = args.layer
lookup_table=[]
for l in range(1, layer_id+1):
    layer = 'layer' + str(l)
    # if (l==1):
    #     tree = torch.load(os.path.join('dictionary', args.model+'_'+args.dataset + '_tree_' + layer + '_depth_' + str(2) + '_n_cl_' + str(10)))
    # else:
    #     tree = torch.load(os.path.join('dictionary', args.model+'_'+args.dataset + '_tree_' + layer + '_depth_' + str(args.max_depth) + '_n_cl_' + str(args.n_cl)))
    tree = torch.load(os.path.join('dictionary', args.model+'_'+args.dataset + '_tree_' + layer + '_depth_' + str(args.max_depth) + '_n_cl_' + str(args.n_cl)))
    nodes = np.asarray([tree[i].centroid for i in range (0, np.shape(tree)[0])])
    table = torch.FloatTensor(nodes).to(device).unsqueeze(0)
    print(table.size())
    lookup_table.append(table)

# lookup_table = torch.FloatTensor(nodes).to(device)

args.out_name = args.model+'_'+args.dataset + '_ql_layer'+str(layer_id)+ '_depth_' + str(args.max_depth) + '_n_cl_' + str(args.n_cl)+'.pth'

test(model, test_loader, device, layer_id=layer_id, tree=lookup_table)

train(model,train_loader, test_loader, args, device, layer_id=layer_id, tree=lookup_table)

model.load_state_dict(torch.load('./models/' + args.out_name))
model.to(device).eval()
test(model, test_loader, device, layer_id=layer_id, tree=lookup_table)
