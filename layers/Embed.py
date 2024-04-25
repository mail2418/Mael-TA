import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm

"""
an embedding…stores categorical data in a lower-dimensional vector than an indicator column.
"""

class PositionalEmbedding(nn.Module):
    def __init__(self, model_name, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.model_name = model_name
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        # if self.model_name in ["MaelNet", "MaelNetS1", "MaelNetB1"]:
        #     div_term = (torch.arange(0, d_model).float() * -(math.log(10000.0) / d_model)).exp()
        #     pe += torch.sin(position * div_term)
        #     pe += torch.cos(position * div_term)
        # else:
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # output = x + self.pe[:, :x.size(1)] if self.model_name == "MaelNet" else self.pe[:, :x.size(1)]
        output = self.pe[:, :x.size(1)]
        return output 
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
    def __init__(self, c_in, c_out, d_model, kernel_size, dropout=0.2, n_windows=5):
        super(TokenTCNEmbedding, self).__init__()
        # TCN
        self.dropout = nn.Dropout(dropout)
        layers = []
        self.num_levels = math.ceil(math.log2((n_windows - 1) * (2 - 1) /kernel_size))
        self.leakyrelu = nn.LeakyReLU(True)
        for i in range(self.num_levels-1):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.chomp = Chomp1d(padding)
            self.tokenConv = weight_norm(nn.Conv1d(in_channels=c_in, out_channels=c_out,
                                    kernel_size=3, padding=padding, dilation=dilation_size, bias=False))
            self.net = nn.Sequential(self.tokenConv, self.chomp, self.leakyrelu, self.dropout)
            layers += [self.net]
        # Last Layer of TCN Embedding
        dilation_size = 2 ** (self.num_levels - 1)
        padding = (kernel_size - 1) * dilation_size
        self.chomp = Chomp1d(padding)
        self.tokenConv = weight_norm(nn.Conv1d(in_channels=c_out, out_channels=d_model,
                                kernel_size=3, padding=padding, dilation=dilation_size, bias=False))
        self.net = nn.Sequential(self.tokenConv, self.chomp, self.leakyrelu, self.dropout)
        layers += [self.net]
        self.init_weights()
        self.network = nn.Sequential(*layers)
    def init_weights(self):
        self.tokenConv.weight.data.normal_(0, 0.001)

    def forward(self, x):
        x = x.permute(0, 2, 1) #permute, reshape, 
        x = self.network(x)
        x = x.transpose(1, 2)
        return x

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
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
        output = self.embed(x) # 32x100 4x25
        return output 

class DataEmbedding(nn.Module):
    def __init__(self, model_name, c_in, d_model, kernel_size=3, embed_type='fixed', freq='h', dropout=0.1, n_windows=5):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenTCNEmbedding(c_in=c_in, c_out= c_in, d_model=d_model, kernel_size=kernel_size, n_windows=n_windows) if model_name in ["MaelNet", "MaelNetS1", "MaelNetB1"] else TokenEmbedding(
            c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(model_name, d_model=d_model)
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
    
# Digunakan buat KBJNet
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)

class TemporalCnn(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalCnn, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding)
        self.leakyrelu = nn.LeakyReLU(True)
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv, self.chomp, self.leakyrelu, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.001)

    def forward(self, x):
        """
        :param x: size of (Batch, out_channel, seq_len)
        :return:size of (Batch, out_channel, seq_len)
        """
        out = self.net(x)
        return out

class Tcn_Global(nn.Module):

    def __init__(self, num_inputs, num_outputs, kernel_size=3, dropout=0.2):  # k>=d
        super(Tcn_Global, self).__init__()
        layers = []
        num_levels = math.ceil(math.log2((num_inputs - 1) * (2 - 1) / (kernel_size - 1) + 1))

        out_channels = num_outputs #out_channels = 25
        for i in range(num_levels):
            dilation_size = 2 ** i  # Expansion coefficient: 1，2，4，8……
            layers += [TemporalCnn(out_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size - 1) * dilation_size,
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x

#DCDetector
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        self.affine_weight=self.affine_weight.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.affine_bias=self.affine_bias.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
