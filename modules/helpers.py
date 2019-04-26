
import ftplib

def upload(path):
    session = ftplib.FTP('85.214.200.53','ftp-user','oqu7iyiJongae6Oon5foo5mau')
    session.cwd('models') 
    file = open(path,'rb') 
    name = path.split("/")[-1]
    session.storbinary(f'STOR {name}', file)
    file.close()
    session.quit()

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?

def apply_mask(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?