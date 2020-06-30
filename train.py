import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 3 and 6 epochs"""
    lr = lr * (0.1 ** (epoch // 6))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(net,trainloader, testloader, num_epoch, lr, device):
    criterion = nn.CrossEntropyLoss() #
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epoch): # loop over the dataset multiple times
        net.train()
        adjust_learning_rate(lr, optimizer, epoch)
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            prec1 = accuracy(outputs.data, labels, topk=(1,))[0]
            if i % 30 == 0: # print every 2 mini-batches
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}'.format(
                                epoch, i * len(inputs), len(trainloader.dataset),
                                100. * i / len(trainloader), loss.item(), prec1.item()))
        print("----- Validation ----------")
        test(net, testloader, device)
        print("---------------------------")
    print('Finished Training')
    return net

def test(net, testloader, device):
    net.eval()
    correct = 0.0
    total = 0.0
    i = 0.0
    for (images, labels) in testloader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        i=i+1.0
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100.0 * correct.item() / total))
