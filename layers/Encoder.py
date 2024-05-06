import torch.nn as nn
import torch.nn.functional as F
from utils.tools import series_decomp

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, moving_avg = 24, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # self.decomp1 = series_decomp(moving_avg)
        # self.decomp2 = series_decomp(moving_avg)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None, asso_dispatch=False):
        # self.attention(queries, keys, values)

        # x, _ = self.decomp1(x)
        # y = x
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))

        # output, _= self.decomp2(x + y)
        # return output, attn
        if asso_dispatch:
            new_x, series, prior = self.attention(
                x, x, x,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
            x = x + self.dropout(new_x)
            y = x = self.norm1(x)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            return self.norm2(x + y), series, prior 
        
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn 


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, asso_dispatch=False):
        # x [B, L, D]
        attns = []
        series_list = []
        prior_list = []
        if self.conv_layers is not None:
            # The reason why we only import delta for the first attn_block of Encoder
            # is to integrate Informer into our framework, where row size of attention of Informer is changing each layer
            # and inconsistent to the sequence length of the initial input,
            # then no way to add delta to every row, so we make delta=0.0 (See our Appendix E.2)
            # 
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i==0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                if asso_dispatch:
                    x, series, prior = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, asso_dispatch=True)
                    series_list.append(series)
                    prior_list.append(prior)
                    continue
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if not asso_dispatch:
            return x, attns
        else:
            return x, series_list, prior_list


# Buat KBJNet

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.att = None
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        self.att = self.self_attn(src, src, src)[1]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

# encoder dcdetector
class DCDetectorEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(DCDetectorEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list