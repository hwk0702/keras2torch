'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import random
import os
import argparse

from models import *
from utils import progress_bar
from timm import create_model
from timm.scheduler import create_scheduler

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--datadir', default='/datasets/', type=str, help='data directory')
parser.add_argument('--img_size', default=32, type=int, help='image resolution')
parser.add_argument('--modelname', default='resnet50d', type=str, help='model name')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--accumulation_steps', default=4, type=int, help='gradient accumulation')
parser.add_argument('--sched', default='cosine_annealing', type=str, help='learning rate scheduler')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--min_lr', default=0., type=float, help='minimum learning rate')
parser.add_argument('--decay-rate', '--dr', type=float, default=1., metavar='RATE',
                    help='LR decay rate (default: 1.)')
parser.add_argument('--t-in-epochs', action='store_false', default=True,
                    help='use step update instead of epoch')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--warmup_epochs', default=3, type=int, help='warmup_epochs')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


# Seed
torch_seed(args.seed)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.RandomCrop(args.img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=os.path.join(args.datadir, 'CIFAR10'), train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=os.path.join(args.datadir, 'CIFAR10'), train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if 'vit' in args.modelname:
    net = create_model(args.modelname, pretrained=args.pretrained, img_size=args.img_size)
else:
    net = create_model(args.modelname, pretrained=args.pretrained)

net = net.to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

if args.opt == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
elif args.opt == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=5e-4)
    
if args.sched == 'cosine_annealing':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.sched == 'cosine':
    scheduler, _  = create_scheduler(args, optimizer)

# Training
def train(epoch, accumulation_steps):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss /= accumulation_steps
        loss.backward()

        _, predicted = outputs.max(1)
        total += targets.size(0) 
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{args.modelname}_{args.opt}_lr-{args.lr}_seed-{args.seed}.pth')
        best_acc = acc

    return 

for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch, args.accumulation_steps)
    test(epoch)
    scheduler.step(epoch)
