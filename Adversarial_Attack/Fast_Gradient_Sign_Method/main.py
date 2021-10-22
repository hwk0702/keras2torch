import timm
from timm.data import create_dataset
from timm.data.transforms import _pil_interp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from collections import OrderedDict
import pandas as pd
import math
import time
import logging
import os 

from tqdm import tqdm
import argparse

from fgsm import FGSM
from log import setup_default_logging

_logger = logging.getLogger('adv_attack')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def transforms_imagenet_eval(img_size, crop_pct, interpolation):

    # scaling image for cropping
    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    # transforms list
    tfl = [
        transforms.Resize(scale_size, _pil_interp(interpolation)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ]

    return transforms.Compose(tfl)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        return (input - mean) / std


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

def attack(args, eps, model, loader):
    # PGD
    atk = FGSM(model, eps=eps)

    _logger.info(atk)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Metrics
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_m = AverageMeter()

    end = time.time()
    # model.eval()
    for idx, (image, target) in enumerate(tqdm(loader)):

        adv_images = atk(image, target)
        target = target.cuda()
        output = model(adv_images)

        acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
        loss = criterion(output, target)
        top1.update(acc1.item(), image.size(0))
        top5.update(acc5.item(), image.size(0))
        loss_m.update(loss.item(), image.size(0))


    _logger.info('Total elapsed time (sec): %.2f' % (time.time() - end))
    _logger.info('Robust Loss:{loss_m.avg:7.4f} Acc@1: {top1.avg:7.3f} Acc@5: {top5.avg:7.3f}'.format(loss_m=loss_m, top1=top1, top5=top5))

    return OrderedDict([('loss',loss_m.avg), ('top1-acc',top1.avg), ('top5-acc',top5.avg)])

def save_results(savedir, exp_name, metrics):
    path = os.path.join(savedir, f'{exp_name}.csv')
    if not os.path.isfile(path):
        df = pd.DataFrame()
    else:
        df = pd.read_csv(path)

    df.append(metrics, ignore_index=True).to_csv(path, index=False)


def run(args):
    setup_default_logging(log_path=os.path.join(args.savedir,'output.log'))
    # Build model
    model = timm.create_model('resnet50', pretrained=True)

    # Load Data
    transform_eval = transforms_imagenet_eval(
        img_size      = args.img_size,
        crop_pct      = model.default_cfg['crop_pct'],
        interpolation = model.default_cfg['interpolation']
    )   

    dataset = create_dataset(root=args.data, name=args.dataset, split=args.split)
    dataset.transform = transform_eval
    
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Normalization
    model = nn.Sequential(
        Normalize(mean=model.default_cfg['mean'], std=model.default_cfg['std']),
        model
    ).cuda()
    
    eps_list = list(map(float,args.eps.split(',')))
    for eps in eps_list:
        # adv attack
        metrics = attack(args, eps, model, loader)
        metrics['epsilon'] = eps
    
        # save
        save_results(savedir=args.savedir, exp_name=args.exp_name, metrics=metrics)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Tootouch's PGD Experiments")
    parser.add_argument('--exp-name',type=str,default='fgsm_exp',help='experiment name')
    parser.add_argument('--savedir',type=str,default='./',help='save directory')
    parser.add_argument('--eps',type=str,default='0.001,0.003,0.005,0.01,0.03,0.05,0.1',help='epsilon')
    parser.add_argument('--img-size',type=str,default=224,help='image size')
    parser.add_argument('--batch-size',type=int,default=32,help='batch_size')
    parser.add_argument('--data',type=str,default='/datasets/Imagenet/ILSVRC/Data/CLS-LOC',help='imagenet data directory')
    parser.add_argument('--dataset',type=str,default='Imagenet',help='dataset name')
    parser.add_argument('--split',type=str,default='val',help='direectory name')
    parser.add_argument('--num_workers',type=int,default=8,help='the number of workers')
    args = parser.parse_args()

    run(args)
