from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjustment, visual
from utils.metrics import NegativeCorr
from models.RLMantra import RLMantra
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from models.RLMantra import RLMantra
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import random

import copy

warnings.filterwarnings('ignore')

class OPT_RL_Mantra:
    def __init__(self,args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        model = RLMantra(self.args, self.device)
        return model
    
    def _get_data(self,flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer_actor(self):
        actor_optim = optim.Adam(self.model.actor.parameters(), lr=self.args.learning_rate)
        return actor_optim
    
    def _select_optimizer_critic(self):
        critic_optim = optim.Adam(self.URT.parameters(), lr=0.0001)
        return critic_optim
    
    def train_urt_reinforcment_learning(self, setting):
        pass

    def _select_criterion(self):
        if self.args.loss_type == "negative_corr":
            criterion = NegativeCorr(self.args.correlation_penalty)
        else:
            criterion = nn.MSELoss()
        return criterion
    
