import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")

class MSLRLLoader(object):
    def __init__(self, root_path, setting, win_size, step=1):
        path_ds = os.path.join(root_path, setting)
        input_data = np.load(f'{path_ds}/input.npz', allow_pickle=True)
        self.train_X = input_data['train_X']
        self.valid_X = input_data['valid_X']
        self.test_X  = input_data['test_X' ]
        self.train_y  = input_data['train_y' ]
        self.valid_y  = input_data['valid_y' ]
        self.test_labels = input_data["test_labels"]
        self.train_error = input_data['train_error'] 
        self.valid_error = input_data['valid_error']  
        self.test_error  = input_data['test_error' ]  
        self.step = step
        self.win_size = win_size
        # train_data = 
    def __len__(self, flag):
        pass
    def __getitem__(self, index):
        index = index * self.step

