import numpy as np
import pandas as pd
import torch
import transformers

from utils import fit, torch_seed
from model import BertSemanticModel
from loaddata import BertSemanticDataset

import argparse

import warnings
warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--no_grad',action='store_true',help='whether uses no_grad')
    parser.add_argument('--requires_grad',action='store_true',help='whtether uses requires_grad')
    parser.add_argument('--batch_size',type=int,default=32,help='the number of batch size')
    parser.add_argument('--epochs',type=int,default=2,help='the number of epochs')
    args = parser.parse_args()
    
    print(f'[SETTING]: learning: {args.lr}, no_grad: {args.no_grad}, requires_grad: {args.requires_grad}')

    torch_seed(223)

    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')

    max_length = 128 # Maximun length of input sentence to the model

    # Labels in our dataset
    labels = ["contradiction", "entailment", "neutral"]

    # There are more than 550k samples in total; we will use 100k for this example.
    train_df = pd.read_csv("../SNLI_Corpus/snli_1.0_train.csv", nrows=10000)
    valid_df = pd.read_csv("../SNLI_Corpus/snli_1.0_dev.csv")
    test_df = pd.read_csv("../SNLI_Corpus/snli_1.0_test.csv")

    train_df = (
        train_df[train_df.similarity != "-"]
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    valid_df = (
        valid_df[valid_df.similarity != "-"]
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    encoder = {
        'contradiction':0,
        'entailment':1,
        'neutral':2
    }

    train_df['similarity'] = train_df['similarity'].map(encoder)
    valid_df['similarity'] = valid_df['similarity'].map(encoder)
    test_df['similarity'] = test_df['similarity'].map(encoder)

    tokenizer = transformers.BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )

    trainset = BertSemanticDataset(
        sentence_pairs=train_df[['sentence1','sentence2']].values.astype('str'),
        targets=train_df['similarity'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    validset = BertSemanticDataset(
        sentence_pairs=valid_df[['sentence1','sentence2']].values.astype('str'),
        targets=valid_df['similarity'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )

    testset = BertSemanticDataset(
        sentence_pairs=test_df[['sentence1','sentence2']].values.astype('str'),
        targets=test_df['similarity'].values,
        tokenizer=tokenizer,
        max_length=max_length,
        include_targets=True
    )

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    validloader = torch.utils.data.DataLoader(
        dataset=validset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    model = BertSemanticModel().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    model.no_grad = args.no_grad 
    for param in model.bert.parameters():
        param.requires_grad = args.requires_grad

    model = fit(model=model,
                epochs=args.epochs,
                trainloader=trainloader,
                validloader=validloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device)
