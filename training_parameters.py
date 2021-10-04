import argparse

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--layer', type=int, default=0, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-i', "--pretrained_model", default='./models/mnist_baseline.pth')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--model', default='lenet')
    parser.add_argument('--max_depth', type=int, default=1, metavar='D',
                        help='depth of dictionary tree')
    parser.add_argument('--n_cl', type=int, default=10, metavar='D',
                        help='depth of dictionary tree')
    parser.add_argument('--num_classes', type=int, default=10, metavar='D',
                        help='depth of dictionary tree')
    parser.add_argument('--num_workers', type=int, default=4, metavar='D',
                            help='number of thread for data processing')
    parser.add_argument('--save_activations', action='store_true', help="compute activations")
    args = parser.parse_args()
    return args
