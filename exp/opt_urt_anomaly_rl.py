from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjustment, visual
from utils.metrics import NegativeCorr
from models import MaelNet, KBJNet, DCDetector, ns_Transformer, FEDFormer, TimesNet, MaelNetB1, MaelNetS1, ns_TransformerB1, ns_TransformerS1
from models.URTRL import MultiHeadURTAnomaly
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
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

class OPT_URT_Anomaly_RL(Exp_Basic):
    def __init__(self,args):
        super(OPT_URT_Anomaly_RL, self).__init__(args)
    def _build_model(self):
        model_dict = {
            "MaelNet"    : MaelNet,
            "MaelNetB1":MaelNetB1,
            "MaelNetS1":MaelNetS1,
            "KBJNet"     : KBJNet,
            "DCDetector" : DCDetector,
            "NSTransformer": ns_Transformer,
            "NSTransformerB1": ns_TransformerB1,
            "NSTransformerS1": ns_TransformerS1,
            "FEDFormer" : FEDFormer,
            "TimesNet": TimesNet,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        self.URT = MultiHeadURTAnomaly(key_dim=self.args.win_size , query_dim=self.args.win_size*self.args.enc_in, hid_dim=4096, temp=1, att="cosine", n_head=self.args.urt_heads).float().to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model