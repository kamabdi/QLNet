to train the model run QLNet.ipynb in Colab

or 

python train_baseline.py
python construct_dict.py --layer 1 --dataset cifar100 --model resnet --pretrained_model mnist_baseline_resnet.pth --max_depth 1 --n_cl 10 --num_classes 10


python train_quantized.py --layer 1 --dataset cifar10 --pretrained_model ./models/mnist_baseline_resnet.pth --model resnet --n_cl 10 --max_depth 2 --num_classes 10



%%%%%%%%%%%%%%%%%%%%% RESULTS %%%%%%%%%%%%%%%
CIFAR-100
ResNet-11
Baseline: 73.42

			Accuracy        Look-up size (1, number, depth)
layer 1 - 	  depth 1  44.01    	1, 10, 64
		  depth 2  58.87	1, 100, 64
		  depth 3  60.67	1, 260, 64

layer 2 -	  depth 1  39.17    1, 10, 64
		  depth 2  53.06    1, 110, 64
		  depth 3  57.58/55.49    1, 320, 64 

shared dictionary
layer 2 -	  depth 1  40.44    1, 10, 64
		  depth 2  51.50    1, 100, 64
		  depth 3  56.06    1, 260, 64 


layer 3 -	  depth 1  11.70    1, 10, 128
		  depth 2  42.05    1, 110, 128
		  depth 3  47.38    1, 380, 128 

layer 4 -	  depth 1      1, 10, 256
		  depth 2      1, 110, 256
		  depth 3      1, 289, 256 


%%%%%%%%%%%%%%%%%%%%% RESULTS %%%%%%%%%%%%%%

MNIST-Fashion
LeNet
Baseline: 91
depth 1          87.28 75.28
depth 2          88.97 79.60
depth 3          89.60 80.82
depth 2(10), 4(8)      86.46

Baseline: 91.6
depth 4(8)       90.02 87.53


Baseline ResNet 93.54 
depth 4  (8)   92.35  93.13  91.66 


German Sign
Custom
Baseline: 93
depth 3 (10)                    51.64
depth 4 (10)                    59.20

depth 4  (8)      93.73  91.77  91.82 90.91
torch.Size([1, 4576, 64])
torch.Size([1, 4472, 64])
torch.Size([1, 4624, 128])
torch.Size([1, 4680, 256])




CIFAR-10 ResNet

Baseline: 86.99

depth 4 (8)        84.74 84.72
