import torch 
from torch import nn
import torch.nn.functional as F
from models.MaelNet import Model as MaelNet
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model,self).__init__()
        self.name = "MaelNetB1"
        self.num_models = configs.n_learner
        self.models = nn.ModuleList([MaelNet(configs).float() for _ in range(self.num_models)])
        for i in range(self.num_models):
            model = self.models[i]
            if (i%2)==0:
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        # stdev = np.random.uniform(0, 0.01)
                        stdev = np.random.uniform(0.001, 0.01)
                        m.weight.data.normal_(0, stdev)
                        print("stdev: " +str(stdev))
            else:
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        # stdev = random.uniform(0, 1)
                        # m.weight.data.uniform_(0, 0.01)
                        m.weight.data.uniform_(0, 0.001)
    def forward(self,x_enc):
        dec_out = []
        for i in range (self.num_models):
            do = self.models[i].forward(x_enc)
            dec_out.append(do)
        dec_out = torch.stack(dec_out)
        dec_out = torch.mean(dec_out,axis=0)
        return dec_out
    def forward_1learner(self, x_enc,idx=0):
        # Tanpa Stack dan mean
        i = idx
        do = self.models[i].forward(x_enc)
        return do