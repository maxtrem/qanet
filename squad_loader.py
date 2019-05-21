import torch
from torch.utils.data import Dataset

import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SQuADDataset(Dataset):
    def __init__(self, npz_file, device=device, na_possible=True, no_answer_id=1000, no_answer_y=400):
        """
            npz_file:   numpy file containing the data
            num_steps:  (int) number of batches to generate - number of training steps
            batch_size: (int) batch size for dataset
            device:     (torch.device) sets the device output tensors will located on
        """
        assert isinstance(na_possible, bool)
        # sets device for location of outputs
        self.na_id  = no_answer_id
        self.na_y   = no_answer_y
        self.na_possible = na_possible
        
        data = np.load(npz_file)
        mask = (torch.from_numpy(data["y2s"]).long() != self.na_id)
        if na_possible:
            mask = torch.ones_like(mask)
            
        self.mask = mask
        # ids for context tokens and chars
        self.context_idxs = torch.from_numpy(data["context_idxs"]).long()[mask]
        self.context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()[mask]
        # ids for question tokens and chars
        self.ques_idxs = torch.from_numpy(data["ques_idxs"]).long()[mask]
        self.ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long()[mask]
        # targets - answer spans y1s:start, y2s:end, ids for all examples
        self.y1s = torch.from_numpy(data["y1s"]).long()[mask]
        self.y2s = torch.from_numpy(data["y2s"]).long()[mask]
        self.ids = torch.from_numpy(data["ids"]).long()[mask]
        self.nas = (self.y2s == self.na_id)
        self.y1s[self.nas] = self.na_y
        self.y2s[self.nas] = self.na_y
        
    def num_nas(self):
        return self.nas.sum().item()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        res = (self.context_idxs[i],        # context tokens
               self.context_char_idxs[i],   # context chars
               self.ques_idxs[i],           # question tokens
               self.ques_char_idxs[i],      # question chars
               self.y1s[i], self.y2s[i],    # answer start, answer end
               self.nas[i], self.ids[i]) # question id
        
        # moving tensors to respective device
        res = tuple(map(lambda x: x.to(device), res))
        return res
    
    
#dataset = SQuADDataset('data/dev.npz', na_possible=False)