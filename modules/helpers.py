
import ftplib

def upload(path, blocksize=2**32):
    session = ftplib.FTP('85.214.200.53','ftp-user','oqu7iyiJongae6Oon5foo5mau')
    session.cwd('models') 
    file = open(path,'rb') 
    name = path.split("/")[-1]
    session.storbinary(f'STOR {name}', file, blocksize=blocksize)
    file.close()
    session.quit()

import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?

def apply_mask(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?


class Activation(nn.Module):
    def __init__(self, activation=None):
        super().__init__()
        self.activation = activation
        
    def forward(self, x):
        return self.activation(x) if self.activation else x

    def get_str(self):
        return self.activation.__class__.__name__ if (self) else ''
    
    def __bool__(self):
        return (not self.activation == None)