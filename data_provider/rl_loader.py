import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")

class MSLRLLoader(Dataset):
    def __init__(self, root_path, setting, win_size, step=1, flag="train"):
        path_ds = os.path.join(root_path, setting)
        input_data = np.load(f'{path_ds}/input.npz', allow_pickle=True)
        if flag == "train":
            self.train_X = np.swapaxes(input_data['train_X'], 2, 1).astype(np.float16)
            self.train_preds = np.load(f'{path_ds}/bm_train_preds_new.npy', allow_pickle=True).astype(np.float16)
            self.train_y  = input_data['train_y'].astype(np.float16)
            self.train_error = input_data['train_error'].astype(np.float16) 
        elif flag == "valid":
            self.valid_X = np.swapaxes(input_data['valid_X'], 2, 1).astype(np.float16)
            self.valid_preds = np.load(f'{path_ds}/bm_valid_preds_new.npy', allow_pickle=True).astype(np.float16)
            self.valid_y  = input_data['valid_y' ].astype(np.float16)
            self.valid_error = input_data['valid_error'].astype(np.float16) 
        elif flag == "test":
            self.test_X  = np.swapaxes(input_data['test_X' ], 2, 1).astype(np.float16)
            self.test_preds = np.load(f'{path_ds}/bm_test_preds_new.npy', allow_pickle=True).astype(np.float16)
            self.test_labels = input_data["test_labels"]
        else:
            # self.
            pass
        self.step = step
        self.win_size = win_size
        self.mode = flag

    def __len__(self):
        if self.mode == "train":
            pass
        elif self.mode == "valid":
            pass
        elif self.mode == "test":
            pass
        else: #pretrain
            return
    def __getitem__(self, idx):
        index = idx * self.step
        if self.mode == "train":
            pass
        elif self.mode == "valid":
            pass
        elif self.mode == "test":
            pass
        else: #pretrain
            pass

class PSMRLLoader(Dataset):       
    pass

class SMAPRLLoader(object):
    pass

class SMDRLLoader(object):
    pass

class SWATRLLoader(object):
    pass