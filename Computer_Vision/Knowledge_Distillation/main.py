import json
import os 
import torch

from utils import fit, torch_seed
from models import *
import torchvision.models as models
from loaddata import make_dataloader

import argparse
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--KD',action='store_true',help='Knowledge Distillation')
    parser.add_argument('--model',type=str,default='teacher',choices=['teacher','student'],help='select model')
    parser.add_argument('--datadir',type=str,default='./data',help='data directory')
    parser.add_argument('--logdir',type=str,default='./logs',help='logdir')
    parser.add_argument('--lr',type=float,default=0.1,help='learning rate')
    parser.add_argument('--batch_size',type=int,default=128,help='the number of batch size')
    parser.add_argument('--temperature',type=float,default=10,help='temperature value')
    parser.add_argument('--temp_scheduler',action='store_true')
    parser.add_argument('--alpha',type=float,default=0.1,help='alpha value')
    parser.add_argument('--loss_method',type=str,default='method1',help='loss method')
    parser.add_argument('--student_model',type=str,default='simplecnn',choices=['simplecnn','resnet18'],help='student model')

    parser.add_argument('--resume',action='store_true',help='restart training')
    parser.add_argument('--start_epoch',type=int,default=0,help='start epoch')
    parser.add_argument('--epochs',type=int,default=30,help='the number of epochs')
    args = parser.parse_args()
    
    print('[SETTING]')
    for k, v in vars(args).items():
        print(f'-{k}: {v}')

    torch_seed(223)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # make version
    if args.KD:
        if args.temp_scheduler:
            savefolder = f'{args.student_model}/{args.loss_method}/KD_{args.model}_alpha{args.alpha}_temp{args.temperature}_scheduler'
        else:
            savefolder = f'{args.student_model}/{args.loss_method}/KD_{args.model}_alpha{args.alpha}_temp{args.temperature}'
    else:
        savefolder = f'{args.model}'

    savedir = os.path.join(args.logdir, savefolder)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
        # save arguments
        json.dump(vars(args), open(os.path.join(savedir,'arguments.json'),'w'), indent=4)
        print('- savedir: ',savedir)

    # Labels in our dataset
    trainloader, validloader = make_dataloader(args.datadir, args.batch_size)

    # Load model
    if args.model=='teacher':
        model = ResNet18().to(device)
    elif args.model=='student': # checking for student model performance 
        if args.student_model=='simplecnn':
            model = Student().to(device)
        elif args.student_model=='resnet18':
            model = ResNet18().to(device)

    # resume
    if args.resume:
        saved_state = torch.load(os.path.join(savedir,f'{args.model}.pt'))
        model.load_state_dict(saved_state['model'])
        print("- I'm back")
        args.epochs += args.start_epoch
    
    # Knowledge Distillation
    if args.KD:
        teacher = ResNet18().to(device)
        
        # load saved teacher model
        saved_state = torch.load(os.path.join(args.logdir,'teacher/teacher.pt'))
        teacher.load_state_dict(saved_state['model'])
        print('- teacher is coming')

        # criterion
        distill_criterion = torch.nn.KLDivLoss(reduction='batchmean')
    else:
        teacher = None
        distill_criterion = None


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    writer = SummaryWriter(savedir)

    model = fit(model=model,
                start_epoch=args.start_epoch,
                epochs=args.epochs,
                trainloader=trainloader,
                testloader=validloader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                savedir=os.path.join(savedir, f'{args.model}.pt'),
                writer=writer,
                teacher=teacher,
                distill_criterion=distill_criterion,
                alpha=args.alpha,
                temperature=args.temperature,
                temperature_scheduler=args.temp_scheduler,
                loss_method=args.loss_method,
                device=device)
