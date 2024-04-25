import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.Attention import DAC_structure, AttentionLayerDCDetector
from layers.Embed import DataEmbedding,RevIN
from layers.Encoder import DCDetectorEncoder

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.name = "DCDetector"
        self.output_attention = configs.output_attention
        self.patch_size = configs.patch_size
        self.channel = configs.channel
        self.win_size = configs.n_windows
        
        # Patching List  
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(self.name, patchsize, configs.d_model, dropout=configs.dropout))
            self.embedding_patch_num.append(DataEmbedding(self.name, self.win_size//patchsize,configs. d_model, dropout=configs.dropout))

        self.embedding_window_size = DataEmbedding(self.name, configs.enc_in, configs.d_model, configs.dropout)
         
        # Dual Attention Encoder
        self.encoder = DCDetectorEncoder(
            [
                AttentionLayerDCDetector(
                    DAC_structure(self.win_size, configs.patch_size, configs.channel, False, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                    configs.d_model, configs.patch_size, configs.channel, configs.n_heads, self.win_size)for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)


    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel
        series_patch_mean = []
        prior_patch_mean = []
        revin_layer = RevIN(num_features=M)

        # Instance Normalization Operation
        x = revin_layer(x, 'norm')
        x_ori = self.embedding_window_size(x, None)
        
        # Mutil-scale Patching Operation 
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l') #Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l') #Batch channel win_size
            
            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p = patchsize) 
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size, None)
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p = patchsize) 
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num, None)
            
            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.append(series), prior_patch_mean.append(prior)

        series_patch_mean = list(series_patch_mean)
        prior_patch_mean = list(prior_patch_mean)
            
        if self.output_attention:
            return series_patch_mean, prior_patch_mean
        else:
            return None
        