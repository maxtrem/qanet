
import ftplib

def upload(path):
    session = ftplib.FTP('85.214.200.53','ftp-user','oqu7iyiJongae6Oon5foo5mau')
    session.cwd('models') 
    file = open(path,'rb') 
    name = path.split("/")[-1]
    session.storbinary(f'STOR {name}', file)
    file.close()
    session.quit()


def apply_mask(target, mask, eps=-1e30):
    return target * mask + (1 - mask) * (eps)