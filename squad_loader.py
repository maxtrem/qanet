import torch
from torch.utils.data import Dataset

import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
class SQuADDataset(Dataset):
    def __init__(self, npz_file, device=device, no_answer_id = 1000, no_answer_y=0):
        """
            npz_file:   numpy file containing the data
            num_steps:  (int) number of batches to generate - number of training steps
            batch_size: (int) batch size for dataset
            device:     (torch.device) sets the device output tensors will located on
        """
        # sets device for location of outputs
        self.device = device
        self.na_id  = torch.tensor(no_answer_id)
        self.na_y   = torch.tensor(no_answer_y)
        data = np.load(npz_file)
        # ids for context tokens and chars
        self.context_idxs = torch.from_numpy(data["context_idxs"]).long()
        self.context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()
        # ids for question tokens and chars
        self.ques_idxs = torch.from_numpy(data["ques_idxs"]).long()
        self.ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long()
        # targets - answer spans y1s:start, y2s:end, ids for all examples
        self.y1s = torch.from_numpy(data["y1s"]).long()
        self.y2s = torch.from_numpy(data["y2s"]).long()
        self.ids = torch.from_numpy(data["ids"]).long()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        no_answer = (self.y1s[i] == self.no_answer_id)
        if bool(self.y1s[i] == self.no_answer_id) and bool(self.y2s[i] == self.no_answer_id):
            a_start, a_end = self.na_y, self.na_y
        else:
            a_start, a_end = self.y1s[i], self.y2s[i]
        res = (self.context_idxs[i],        # context tokens
               self.context_char_idxs[i],   # context chars
               self.ques_idxs[i],           # question tokens
               self.ques_char_idxs[i],      # question chars
               a_start, a_end,    # answer start, answer end
               no_answer, self.ids[i]) # question id
        
        # moving tensors to respective device
        res = tuple(map(lambda x: x.to(self.device), res))
        return res
    
#dataset = SQuADDataset('data/dev.npz', 600, 32)