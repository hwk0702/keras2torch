import sys
import os
import time
import torch
import numpy as np
import random

from model import ExactMatch

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)



def progress_bar(current, total, msg=None, term_width: int = None):
    if term_width is None:
        _, term_width = os.popen('stty size', 'r').read().split()
        term_width = int(term_width)
    else:
        term_width = term_width

    TOTAL_BAR_LENGTH = 65.
    last_time = time.time()
    begin_time = last_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def train(model, dataloader, criterion, optimizer, device):
    total_loss = 0
    
    model.train()
    for batch_idx, batch_i in enumerate(dataloader):
        # inputs and targets
        input_ids = batch_i[0]['input_ids'].to(device)
        attention_mask = batch_i[0]['attention_mask'].to(device)
        token_type_ids = batch_i[0]['token_type_ids'].to(device)
        start_targets = batch_i[1]['start_token_idx'].to(device)
        end_targets = batch_i[1]['end_token_idx'].to(device)
        
        # reset optimizer
        optimizer.zero_grad()
        
        # model output
        start_outputs, end_outputs = model(input_ids, attention_mask, token_type_ids)
        
        # loss
        start_loss = criterion(start_outputs, start_targets)
        end_loss = criterion(end_outputs, end_targets)
        loss = start_loss + end_loss
        loss.backward()
        
        # update optimizer
        optimizer.step()
        
        total_loss += loss.item()
    
        # massage
        progress_bar(current=batch_idx, 
                     total=len(dataloader),
                     msg='Loss: %.3f' % (total_loss/(batch_idx + 1)),
                     term_width=100)
        

# ExactMaching
def validation(model, dataloader, criterion, device, exactmatch):
    total_loss = 0
    
    total_start_preds, total_end_preds = [], []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_i in enumerate(dataloader):
            # inputs and targets
            input_ids = batch_i[0]['input_ids'].to(device)
            attention_mask = batch_i[0]['attention_mask'].to(device)
            token_type_ids = batch_i[0]['token_type_ids'].to(device)
            start_targets = batch_i[1]['start_token_idx'].to(device)
            end_targets = batch_i[1]['end_token_idx'].to(device)

            # model output
            start_outputs, end_outputs = model(input_ids, attention_mask, token_type_ids)
            
            _, start_preds = start_outputs.max(1)
            _, end_preds = end_outputs.max(1)
            total_start_preds.extend(start_preds.cpu().numpy())
            total_end_preds.extend(end_preds.cpu().numpy())
            
            # loss
            start_loss = criterion(start_outputs, start_targets)
            end_loss = criterion(end_outputs, end_targets)
            loss = start_loss + end_loss
            
            total_loss += loss.item()

            # massage
            progress_bar(current=batch_idx, 
                         total=len(dataloader),
                         msg='Loss: %.3f' % (total_loss/(batch_idx + 1)),
                         term_width=100)
            
        exactmatch.evaluate(start_preds=total_start_preds,
                            end_preds=total_end_preds)


            
def fit(model, epochs, trainloader, criterion, optimizer, device, validloader=None, eval_squad_examples=None):
    for epoch in range(epochs):
        print('Fit start')
        print(f'\nEpochs: {epoch+1}/{epochs}')
        train(model, trainloader, criterion, optimizer, device)
        if validloader is not None:
            eval_exactmatch = ExactMatch(squad_examples=eval_squad_examples)
            validation(model, validloader, criterion, device, eval_exactmatch)

        torch.save({'params':model.state_dict()}, 'QABert.pth')