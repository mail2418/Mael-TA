import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Encoder import Encoder, EncoderLayer
from layers.Decoder import Decoder, DecoderLayer
from layers.Attention import AttentionLayer, DSAttention
from utils.tools import series_decomp

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, win_size, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=win_size, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers = layers + [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers = layers + [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E (Batch_Size X Sequence_Length X Number Elements )
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.name = "MaelNet"
        self.output_attention = configs.output_attention
        self.dec_in = configs.dec_in
        self.decomp = series_decomp(configs.moving_avg)
        # Embedding
        self.enc_embedding = DataEmbedding(self.name, configs.enc_in, configs.d_model, configs.kernel_size, configs.embed, configs.freq,
                                           configs.dropout, configs.n_windows)
        # Decoder Digunakan untuk mengaggregasi informasi dan memperbaiki prediksi dari simpel inisialisasi
        self.dec_embedding = DataEmbedding(self.name, configs.dec_in, configs.d_model, configs.kernel_size, configs.embed, configs.freq,
                                           configs.dropout, configs.n_windows, decode=True)
        # Encoder digunakan untuk mengekstrak informasi pada observasi sebelumnya
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,output_attention=configs.output_attention), 
                        configs.d_model,configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.moving_avg,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        
        # Projector digunakan untuk mempelajari faktor de-stationary
        self.tau_learner   = Projector(enc_in=configs.enc_in, win_size=configs.win_size, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, win_size=configs.win_size, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=configs.win_size)
            
    def forward(self, x_enc):
        x_raw = x_enc.clone().detach()

        # Normalization dari NS_Transformer
        means = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - means
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, means) # B x S x E, B x 1 x E -> B x S

        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, tau=tau, delta=delta)

        seasonal_init, trend_init = self.decomp(x_enc) #input dari decoder
        dec_out = self.dec_embedding(seasonal_init, None)
        seasonal_part, trend_part = self.decoder(x=dec_out, cross=enc_out, tau=tau, delta=None, trend=trend_init)

        dec_out = seasonal_part + trend_part
        #Denormalization dari NS_Transformer
        dec_out = dec_out * std_enc 
        dec_out = dec_out + means

        return dec_out  # [B, L, D]