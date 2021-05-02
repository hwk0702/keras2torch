import json
import os 
import torch

from utils import fit, torch_seed
from model import Teacher, Student
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
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--batch_size',type=int,default=128,help='the number of batch size')
    parser.add_argument('--temperature',type=int,default=10,help='temperature value')
    parser.add_argument('--alpha',type=float,default=0.1,help='alpha value')
    parser.add_argument('--epochs',type=int,default=30,help='the number of epochs')
    args = parser.parse_args()
    
    print('[SETTING]')
    for k, v in vars(args).items():
        print(f'-{k}: {v}')

    torch_seed(223)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # make version
    savefolder = f'KD_{args.model}_alpha{args.alpha}_temp{args.temperature}' if args.KD else f'{args.model}'
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
        model = Teacher().to(device)
    elif args.model=='student': # checking for student model performance 
        model = Student().to(device)
    
    # Knowledge Distillation
    if args.KD:
        teacher = Teacher().to(device)
        
        # load saved teacher model
        saved_state = torch.load(os.path.join(args.logdir,'teacher/teacher.pt'))
        teacher.load_state_dict(saved_state['model'])

        # criterion
        distill_criterion = torch.optim.KLDivLoss(reduction='batchmean')
    else:
        teacher = None
        distill_criterion = None


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(savedir)

    model = fit(model=model,
                epochs=args.epochs,
                trainloader=trainloader,
                testloader=validloader,
                criterion=criterion,
                optimizer=optimizer,
                savedir=os.path.join(savedir, f'{args.model}.pt'),
                writer=writer,
                teacher=teacher,
                distill_criterion=distill_criterion,
                alpha=args.alpha,
                temperature=args.temperature,
                device=device)
