to train the model run QLNet.ipynb in Colab

or 

```bash
python train_baseline.py
python construct_dict.py --layer 1 --dataset cifar100 --model resnet --pretrained_model mnist_baseline_resnet.pth --max_depth 1 --n_cl 10 --num_classes 10


python train_quantized.py --layer 1 --dataset cifar10 --pretrained_model ./models/mnist_baseline_resnet.pth --model resnet --n_cl 10 --max_depth 2 --num_classes 10
 ```


# RESULTS 

## MNIST-Fashion
```
LeNet
Baseline: 91.6
QLNet 4(8)90.02 

Baseline ResNet 93.54 
QLNet 4(8)   92.35

```


## CIFAR-10 ResNet

```
Baseline:  86.99
QLNet 4(8) 84.74 
```
