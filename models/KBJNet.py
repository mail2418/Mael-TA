from layers.Embed import PositionalEncoding, Tcn_Global
import torch.nn as nn
from layers.Encoder import TransformerEncoderLayer
from torch.nn import TransformerEncoder
import math 

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.name = 'KBJNet'
        self.lr = 0.05
        self.batch = 128
        self.n_feats = configs.enc_in
        self.n_window = configs.n_windows
        #QKV
        self.tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=self.n_feats, kernel_size=3, dropout=0.2)
        #Embedding gk pake token embedding
        self.pos_encoder = PositionalEncoding(self.n_feats, 0.1, self.n_window)
        #Encoding hanya menggunakan fungsi linear, hanya 2 layer
        encoder_layers1 = TransformerEncoderLayer(d_model=self.n_feats, nhead=self.n_feats, dim_feedforward=1,
                                                  dropout=0.1)  # (seq_len, Batch, output_channel)
        encoder_layers2 = TransformerEncoderLayer(d_model=self.n_feats, nhead=self.n_feats, dim_feedforward=1,
                                                  dropout=0.1)
        self.transformer_encoder1 = nn.Sequential(nn.LayerNorm(self.n_feats), TransformerEncoder(encoder_layers1, num_layers=1))  # Add LayerNorm before TransformerEncoder
        self.transformer_encoder2 = nn.Sequential(nn.LayerNorm(self.n_feats), TransformerEncoder(encoder_layers2, num_layers=1))  # Add LayerNorm before TransformerEncoder
        self.fcn1 = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.LeakyReLU(), nn.Linear(self.n_feats, self.n_feats))
        self.fcn2 = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.LeakyReLU(), nn.Linear(self.n_feats, self.n_feats))
        #Decoding
        self.decoder1 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def callback(self, src, c):
        src2 = src + c
        g_atts = self.tcn(src2)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src2 = self.pos_encoder(src2)
        memory = self.transformer_encoder2(src2)
        return memory

    def forward(self, enc, dec):
        # Embedding Start
        g_atts = self.tcn(enc) #tidak mengubah dimensi # 128x25x5 32x25x100 32 38 100
        enc2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats) # 5x128x25 100x32x25
        enc2 = self.pos_encoder(enc2)
        z1 = self.transformer_encoder1(enc2)
        # Embedding End
        c1 = z1 + self.fcn1(z1)
        c1 = c1.permute(1, 2, 0) + enc
        x1 = self.decoder1(c1)

        z2 = self.fcn2(self.callback(enc, x1))
        c2 = z2 + self.fcn2(z2)
        c2 = c2.permute(1, 2, 0) + x1
        x2 = self.decoder2(c2)
        
        return x2.permute(0,2,1)