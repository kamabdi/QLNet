from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from load_data import get_data
# from baseline_model import BS_Net, BS_Net_German
from model import BS_Net, BS_Net_German, ResNet11
from train_utils import train, test
# from qlnet_model_quantized import BS_Net
# from qlnet_model_quantized_resnet import ResNet11
# from train_utils_quantized import train, test
from  training_parameters import get_params

# from pthflops import count_ops

# python train_baseline.py --dataset mnist --num_classes 10

args = get_params()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# train_loader = get_data(args, dataset='mnist', ifTrain=True)
# test_loader = get_data(args, dataset='mnist', ifTrain=False)

train_loader = get_data(args, dataset=args.dataset, ifTrain=True, ifShuffle=True)
test_loader = get_data(args, dataset=args.dataset, ifTrain=False, ifShuffle=False)

if args.dataset=="german":
    model = BS_Net_German(layer=0, input_size=3, num_classes=48)
else:
    model = BS_Net()
if args.model=='resnet':
    print("ResNet is selected")
    if args.dataset=="fashion" or args.dataset=="mnist": channels=1
    else: channels=3
    model = ResNet11(args.num_classes, channels)
model.to(device)

# args.out_name = 'mnist_baseline_resnet.pth'
# train(model, train_loader, test_loader, args, device)
args.out_name = args.model+'_'+args.dataset +'_baseline_1.pth'
train(model, train_loader, test_loader, args, device)


model.load_state_dict(torch.load('./models/' + args.out_name))
model.to(device).eval()
test(model, test_loader, device,  0, None)


# ##### Compute number of FLOAPS ################
# inp = torch.rand(1,3,32,32).to(device)
# model = ResNet11(args.num_classes)
# model.to(device).eval()
# count_ops(model, inp)
