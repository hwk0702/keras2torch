import os
import json
import numpy as np 

from tensorflow import keras
import torch

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

import argparse

from loaddata import create_squad_examples, create_inputs_targets, SquadDataset
from model import QABert 
from utils import fit, torch_seed

import warnings
warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',type=float,default=0.00005,help='learning rate')
    parser.add_argument('--batch_size',type=int,default=16,help='the number of batch size')
    parser.add_argument('--epochs',type=int,default=2,help='the number of epochs')
    parser.add_argument('--resume',action='store_true',help='resume training')
    args = parser.parse_args()
    
    print(f'[SETTING]: learning rate: {args.lr}, batch_size: {args.batch_size}, epochs: {args.epochs}')

    torch_seed(223)

    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')

    max_len = 384 # Maximun length of input sentence to the model

    # There are more than 550k samples in total; we will use 100k for this example.
    train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    train_path = keras.utils.get_file("train.json", train_data_url)
    eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    eval_path = keras.utils.get_file("eval.json", eval_data_url)

    with open(train_path) as f:
        raw_train_data = json.load(f)

    with open(eval_path) as f:
        raw_eval_data = json.load(f)

    # set tokenizer
    ## Save the slow pretrained tokenizer
    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    save_path = "bert_base_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

    ## Load the fast tokenizer from saved file
    tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)


    # preprocessing the data
    train_squad_examples = create_squad_examples(raw_train_data, tokenizer, max_len)
    x_train, y_train = create_inputs_targets(train_squad_examples)
    print(f"{len(train_squad_examples)} training points created.")

    eval_squad_examples = create_squad_examples(raw_eval_data, tokenizer, max_len)
    x_eval, y_eval = create_inputs_targets(eval_squad_examples)
    print(f"{len(eval_squad_examples)} evaluation points created.")

    # create dataloader
    trainset = SquadDataset(x_data=x_train, y_data=y_train)
    validset = SquadDataset(x_data=x_eval, y_data=y_eval)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True)
    validloader = torch.utils.data.DataLoader(validset,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    # build model
    model = QABert().to(device)

    # resume
    if args.resume:
        params = torch.load('QABert.pth')['params']
        model.load_state_dict(params)

    # criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # training
    model = fit(model=model,
                epochs=args.epochs,
                trainloader=trainloader,
                validloader=validloader,
                eval_squad_examples=eval_squad_examples,
                criterion=criterion,
                optimizer=optimizer,
                device=device)

    