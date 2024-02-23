import torch.nn as nn
import torch.nn.functional as F
from models.causal_cnn import CausalCNNEncoder

class Actor(nn.Module):
    def __init__(self, configs, act_dim, obs_dim):
        super(Actor, self).__init__()
        #DPG
        self.cnn_encoder = CausalCNNEncoder(depth=3,
                                            kernel_size=3,
                                            in_channels=obs_dim,
                                            channels=40,
                                            out_channels=configs.hidden_dim_rl,
                                            reduced_size=configs.hidden_dim_rl)
        self.net = nn.Sequential(
            nn.Linear(configs.hidden_dim_rl, configs.hidden_dim_rl), 
            nn.ReLU(),
            nn.Linear(configs.hidden_dim_rl, configs.hidden_dim_rl), 
            nn.ReLU(),
            nn.Linear(configs.hidden_dim_rl, act_dim)
        )
            
    def forward(self, obs):
        x = F.relu(self.cnn_encoder(obs))
        x = self.net(x)
        return x
    

class Critic(nn.Module):
    def __init__(self, configs, act_dim, obs_dim):
        super(Critic, self).__init__()
        #DPG
        self.cnn_encoder = CausalCNNEncoder(depth=3,
                                            kernel_size=3,
                                            in_channels=obs_dim,
                                            channels=40,
                                            out_channels=configs.hidden_dim_rl,
                                            reduced_size=configs.hidden_dim_rl)
        self.act_layer = nn.Linear(act_dim, configs.hidden_dim_rl)
        self.net = nn.Sequential(
            nn.Linear(configs.hidden_dim_rl, configs.hidden_dim_rl), 
            nn.ReLU(),
            nn.Linear(configs.hidden_dim_rl, configs.hidden_dim_rl), 
            nn.ReLU(),
            nn.Linear(configs.hidden_dim_rl, act_dim)
        )
            
    def forward(self, obs, act):
        x = F.relu(self.cnn_encoder(obs) + self.act_layer(act))
        x = self.net(x)
        return x.squeeze()