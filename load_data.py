'''
File to load data
ifTrain =True -> load train set
        = False -> load test set
'''
import torch
import torchvision
from torchvision import datasets, transforms

def get_data(args, dataset='mnist', ifTrain=True, ifShuffle=False):
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    if dataset=="mnist":
        dataSet = get_mnist(args, ifTrain)
    elif dataset=='cifar10':
        dataSet = get_cifar10(args, ifTrain)
    elif dataset=='cifar100':
        dataSet = get_cifar100(args, ifTrain)
    elif dataset=='fashion':
        dataSet = get_mnist_fashion(args, ifTrain, ifShuffle)
    elif dataset=='german':
        dataSet = get_german(args, ifTrain, ifShuffle)
    else:
        print("I don't know this dataset")
    return dataSet

def get_mnist(args, ifTrain):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=ifTrain, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=ifTrain, **kwargs)
    return loader


def get_cifar10(args, ifTrain):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    if ifTrain:
      transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    else:
      transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=ifTrain, download=True,
                       transform=transform),
        batch_size=args.batch_size, shuffle=ifTrain, **kwargs)

    return loader

def get_cifar100(args, ifTrain):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    if ifTrain:
      transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    else:
      transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=ifTrain, download=True,
                       transform=transform),
        batch_size=args.batch_size, shuffle=ifTrain, **kwargs)

    return loader


def get_mnist_fashion(args, ifTrain, ifShuffle):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=ifTrain, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=ifShuffle, **kwargs)
    return loader


def get_german(args, ifTrain, ifShuffle):
    from data import data_transforms, train_data_transforms, ImbalancedDatasetSampler
    if ifTrain:
        print("Train for training")
        split = "train"
        transform = train_data_transforms
        train_dataset = datasets.ImageFolder('/home/dragon/Desktop/Noisy_model/Full_precision/data/GTSRB/'+ split, transform=train_data_transforms)
        loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            # shuffle=ifShuffle,
            sampler=ImbalancedDatasetSampler(train_dataset),
            num_workers=args.num_workers, pin_memory=True)
    else:
        split = "test"
        transform = data_transforms
        val_dataset = datasets.ImageFolder('/home/dragon/Desktop/Noisy_model/Full_precision/data/GTSRB/'+ split, transform=data_transforms)
        loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

    return loader
