from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import os
import time
import json
import random
import wandb
import logging
from collections import OrderedDict

import torch
from apex import amp
from contextlib import suppress
import argparse

from log import setup_default_logging
from models.resnet import ResNet50

_logger = logging.getLogger('train')

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

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, dataloader, criterion, optimizer, log_interval, accumulation_steps=1, device='cpu'):   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        # optimizer condition
        opt_cond = (idx + 1) % accumulation_steps == 0

        if opt_cond or idx == 0:
            data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # loss for accumulation steps
        loss /= accumulation_steps        
        loss.backward()

        if opt_cond:
            # loss update
            optimizer.step()
            optimizer.zero_grad()

            losses_m.update(loss.item()*accumulation_steps)

            # accuracy
            preds = outputs.argmax(dim=1) 
            acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))
            
            batch_time_m.update(time.time() - end)
        
            if (idx // accumulation_steps) % log_interval == 0 and idx != 0: 
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'Acc: {acc.avg:.3%} '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (idx+1)//accumulation_steps, len(dataloader)//accumulation_steps, 
                        loss       = losses_m, 
                        acc        = acc_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = inputs.size(0) / batch_time_m.val,
                        rate_avg   = inputs.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
   
        end = time.time()
    
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg)])
        
def test(model, dataloader, criterion, log_interval, device='cpu'):
    correct = 0
    total = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)
            
            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
                
    return OrderedDict([('acc',correct/total), ('loss',total_loss/len(dataloader))])
                
def fit(
    exp_name, model, epochs, trainloader, testloader, criterion, optimizer, scheduler, 
    savedir, log_interval, args, accumulation_steps=1, device='cpu'
):
    savedir = os.path.join(savedir,exp_name)
    os.makedirs(savedir, exist_ok=True)
    wandb.init(name=exp_name, project='Accumulation Steps', config=args)
    
    best_acc = 0

    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics = train(model, trainloader, criterion, optimizer, log_interval, accumulation_steps, device)
        eval_metrics = test(model, testloader, criterion, log_interval, device)

        scheduler.step()

        # wandb
        metrics = OrderedDict(epoch=epoch)
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        wandb.log(metrics)
    
        # checkpoint
        if best_acc < eval_metrics['acc']:
            state = {'best_epoch':epoch, 'best_acc':eval_metrics['acc']}
            json.dump(state, open(os.path.join(savedir, f'{exp_name}.json'),'w'), indent=4)

            weights = {'model':model.state_dict()}
            torch.save(weights, os.path.join(savedir, f'{exp_name}.pt'))
            
            _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

            best_acc = eval_metrics['acc']

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))


def run(args):
    setup_default_logging()
    torch_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # Load Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(os.path.join(args.datadir,'CIFAR100'), train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(os.path.join(args.datadir,'CIFAR100'), train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Build Model
    model = ResNet50(num_classes=100)
    model.to(device)
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # Set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Fitting model
    fit(exp_name           = args.exp_name,
        model              = model, 
        epochs             = args.epochs, 
        trainloader        = trainloader, 
        testloader         = testloader, 
        criterion          = criterion, 
        optimizer          = optimizer, 
        scheduler          = scheduler,
        savedir            = args.savedir,
        log_interval       = args.log_interval,
        accumulation_steps = args.accumulation_steps,
        device             = device,
        args               = args)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Tootouch's AMP Experiments")
    parser.add_argument('--exp-name',type=str,help='experiment name')
    parser.add_argument('--datadir',type=str,default='/datasets',help='data directory')
    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')
    parser.add_argument('--epochs',type=int,default=100,help='the number of epochs')
    parser.add_argument('--lr',type=float,default=0.1,help='learning_rate')
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--num-workers',type=int,default=8,help='the number of workers (threads)')
    parser.add_argument('--apex',action='store_true',default=False)
    parser.add_argument('--log-interval',type=int,default=10,help='log interval')
    parser.add_argument('--seed',type=int,default=223,help='223 is my birthday')
    parser.add_argument('--accumulation-steps',type=int,default=1,help='accumulation step size')

    args = parser.parse_args()

    run(args)