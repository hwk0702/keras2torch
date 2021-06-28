import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchsummary import summary
from matplotlib import pyplot as plt
import argparse
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=288):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

class anomaly_detecter(nn.Module):
    def __init__(self, input_channel, kernel_size=7, stride=2):
        super(anomaly_detecter, self).__init__()
        
        self.ad_layer = nn.Sequential(
            nn.Conv1d(input_channel, 32, kernel_size, stride=stride, padding=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 16, kernel_size, stride=stride, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 16, kernel_size, stride=stride, padding=3, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose1d(16, 32, kernel_size, stride=stride, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size, padding=3),
        )
        
    def forward(self, inputs):
        output = self.ad_layer(inputs)
        return output
    
# @profile
def train(model, train_data, optimizer, loss_fn, use_fp16=True, max_norm=None):
    
    epoch_loss = 0
    
    model.train() 

    for idx, batch in enumerate(train_data):
        
        optimizer.zero_grad(set_to_none=True)
        scaler = torch.cuda.amp.GradScaler()
                
        input = batch.to(device)
        
        with torch.cuda.amp.autocast(enabled=use_fp16):
            predictions = model.forward(input)
            train_loss = loss_fn(predictions, input)
        if use_fp16:
            scaler.scale(train_loss).backward()
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()
        
        epoch_loss += train_loss.item()
        
    return epoch_loss/len(train_data)

def validation(model, val_data, loss_fn):
    model.eval()
    
    with torch.no_grad():
        predictions = model.forward(val_data)
        val_loss = loss_fn(predictions, val_data)
        
    return val_loss.item()/len(val_data)

class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'\n Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False
    
def str2bool(v):
    if type(v) is not bool:
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    else:
        return v
    
@profile    
def main(use_summary: bool=False, use_collate: bool=False, 
         num_workers: int=0, use_early_stopping: bool=True, use_fp16: bool=True):
    
    master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

    df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
    df_small_noise_url = master_url_root + df_small_noise_url_suffix
    df_small_noise = pd.read_csv(
        df_small_noise_url, parse_dates=True, index_col="timestamp"
    )

    df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
    df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
    df_daily_jumpsup = pd.read_csv(
        df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
    )

    # Normalize and save the mean and std we get,
    # for normalizing test data.
    training_mean = df_small_noise.mean()
    training_std = df_small_noise.std()
    df_training_value = (df_small_noise - training_mean) / training_std

    TIME_STEPS = 288
    
    x = create_sequences(df_training_value.values)
    
    if use_collate:
        params = {'batch_size': 128,
                  'shuffle': True,
                  'num_workers': num_workers,
                 'collate_fn' : lambda x: default_collate(x).to(device)}
    
    else:
    
        params = {'batch_size': 128,
                  'shuffle': True,
                  'num_workers': num_workers,
                 'pin_memory' : True}

    val_rate = 0.1

    x_train = x[:int(len(x)*(1-val_rate))]
    x_val = x[int(len(x)*(1-val_rate)):]

    x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 2, 1)
    x_val = torch.tensor(x_val, dtype=torch.float32).permute(0, 2, 1).to(device)

    dataloader = DataLoader(x_train, **params)
    
        
    model = anomaly_detecter(x_train.shape[1]).to(device)
    if use_summary:
        summary(model, (1, 288))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    epochs = 1000
    history = dict()
    
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=20, verbose=1)
        
    for epoch in range(1, epochs+1):
        epoch_loss = train(model, dataloader, optimizer, loss_fn, use_fp16=use_fp16)
        val_loss = validation(model, x_val, loss_fn)

        history.setdefault('loss', []).append(epoch_loss) 
        history.setdefault('val_loss', []).append(val_loss) 

        sys.stdout.write(
            "\r" + f"[Train] Epoch : {epoch:^3}"\
            f"  Train Loss: {epoch_loss:.4}"\
            f"  Validation Loss: {val_loss:.4}"\
                        )
        
        if use_early_stopping:
            if early_stopping.validate(val_loss):
                break
        
        
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-S", "--use_summary", default=True, help="Use summary")
    ap.add_argument("-C", "--use_collate", default=False, help="Use collate function")
    ap.add_argument("-W", "--num_workers", type=int, default=8, help="number of workers")
    ap.add_argument("-E", "--use_early_stopping", default=False, help="Use early_stopping")
    ap.add_argument("-F", "--use_fp16", default=True, help="Use fp16")

    args = vars(ap.parse_args())
    
    use_summary = str2bool(args['use_summary'])
    use_collate = str2bool(args['use_collate'])
    num_workers = args['num_workers']
    use_early_stopping = str2bool(args['use_early_stopping'])
    use_fp16 = str2bool(args['use_fp16'])

    main(use_summary, use_collate, num_workers, use_early_stopping, use_fp16)
