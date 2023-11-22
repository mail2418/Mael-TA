import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm

"""
an embedding…stores categorical data in a lower-dimensional vector than an indicator column.
"""

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model).float() * -(math.log(10000.0) / d_model)).exp()

        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        output = x + self.pe[:, :x.size(1)]
        return output #1 x 100x 512

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        In fact, this is a cropping module, cropping the extra rightmost padding (default is padding on both sides)

        tensor.contiguous() will return the same tensor with contiguous memory
        Some tensors do not occupy a whole block of memory, but are composed of different blocks of data
        The tensor's view() operation relies on the memory being a whole block, in which case it is only necessary
        to execute the contiguous() function, which turns the tensor into a continuous distribution in memory
        """
        x = torch.narrow(x, dim=2, start=0, length=x.size(2)-self.chomp_size).contiguous()
        return x

class TokenTCNEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size, dropout=0.2, n_windows=10):
        super(TokenTCNEmbedding, self).__init__()
        # TCN
        
        # self.leakyrelu = nn.LeakyReLU(True)
        self.dropout = nn.Dropout(dropout)
        layers = []
        self.num_levels = math.ceil(math.log2((n_windows - 1) * (2 - 1) / (kernel_size - 1) + 1))
        for i in range(self.num_levels):
            dilation_size = 2**i
            padding = (kernel_size - 1) * dilation_size
            self.chomp = Chomp1d(padding)
            self.leakyrelu = nn.LeakyReLU(True)
            self.tokenConv = weight_norm(nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, dilation=dilation_size, bias=False))
            self.net = nn.Sequential(self.tokenConv, self.chomp, self.leakyrelu, self.dropout)
            layers += [self.net]
            self.init_weights()
        self.network = nn.Sequential(*layers)

    def init_weights(self):
        self.tokenConv.weight.data.normal_(0, 0.001)

    def forward(self, x):
        #x.shape = 32x100x25 --> 25x512x3
        # src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        output = self.emb(x).detach()
        return output


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        output = self.embed(x)
        return output 


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenTCNEmbedding(c_in=c_in, d_model=d_model, kernel_size=kernel_size)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            ve = self.value_embedding(x)
            pe = self.position_embedding(x)
            x =  ve + pe 
        else:
            ve = self.value_embedding(x)
            pe = self.position_embedding(x)
            te = self.temporal_embedding(x_mark)
            x =  ve + pe + te
        return self.dropout(x)