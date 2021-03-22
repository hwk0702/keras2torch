import sys
import os
import time
import torch
import numpy as np
import random

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
    total = 0
    correct = 0 
    total_loss = 0
    
    model.train()
    for batch_idx, batch_i in enumerate(dataloader):
        # inputs and targets
        input_ids = batch_i['input_ids'].to(device)
        attention_mask = batch_i['attention_mask'].to(device)
        token_type_ids = batch_i['token_type_ids'].to(device)
        targets = batch_i['target'].to(device)
        
        # reset optimizer
        optimizer.zero_grad()
        
        # model output
        outputs = model(input_ids, attention_mask, token_type_ids)
        
        # accuracy
        _, predict = outputs.max(1)
        correct += predict.eq(targets.long()).cpu().float().sum().item()
        total += input_ids.size(0)
        
        # loss
        loss = criterion(outputs, targets)
        loss.backward()
        
        # update optimizer
        optimizer.step()
        
        total_loss += loss.item()
    
        
        # massage
        progress_bar(current=batch_idx, 
                     total=len(dataloader),
                     msg='Loss: %.3f | Acc: %.3f%%' % (total_loss/(batch_idx + 1), 
                                                               100.*(correct/total)),
                     term_width=100)
        
        
def validation(model, dataloader, criterion, device):
    total = 0
    correct = 0 
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_i in enumerate(dataloader):
            # inputs and targets
            input_ids = batch_i['input_ids'].to(device)
            attention_mask = batch_i['attention_mask'].to(device)
            token_type_ids = batch_i['token_type_ids'].to(device)
            targets = batch_i['target'].to(device)

            # model output
            outputs = model(input_ids, attention_mask, token_type_ids)

            # accuracy
            _, predict = outputs.max(1)
            correct += predict.eq(targets.long()).cpu().float().sum().item()
            total += input_ids.size(0)

            # loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # massage
            progress_bar(current=batch_idx, 
                         total=len(dataloader),
                         msg='Loss: %.3f | Acc: %.3f%%' % (total_loss/(batch_idx + 1), 
                                                                   100.*(correct/total)),
                         term_width=100)
            
            
def fit(model, epochs, trainloader, criterion, optimizer, device, validloader=None):
    for epoch in range(epochs):
        print('Fit start')
        print(f'\nEpochs: {epoch+1}/{epochs}')
        train(model, trainloader, criterion, optimizer, device)
        if validloader is not None:
            validation(model, validloader, criterion, device)

    return model