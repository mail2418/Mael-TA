import torch
import torch.nn as nn
from layers.Encoder import Encoder, EncoderLayer
from layers.Decoder import Decoder, DecoderLayer
from layers.Attention import AttentionLayer, DSAttention
from layers.Embed import DataEmbedding

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
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
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
    """
    Non-stationary Transformer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.name = "ns_Transformer"
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(self.name, configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Decoder Digunakan untuk mengaggregasi informasi dan memperbaiki prediksi dari simpel inisialisasi
        self.dec_embedding = DataEmbedding(self.name, configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout) #dec_in = 7, d_model = 512 
        # Encoder digunakan untuk mengekstrak informasi pada observasi sebelumnya
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers) #e_layers == encoder layer = 2
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
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers) #d_layers = decode layer = 1
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        # Projector digunakan untuk mempelajari faktor de-stationary
        self.tau_learner   = Projector(enc_in=configs.enc_in, win_size=configs.win_size, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, win_size=configs.win_size, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=configs.win_size)

    def forward(self, x_enc, x_dec):

        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec.unsqueeze(2).repeat(1,1,x_enc.shape[2]), None)
        dec_out = self.decoder(x=dec_out, cross=enc_out, tau=tau, delta=delta)

        # Denormalization
        dec_out = dec_out[0] * std_enc 
        dec_out += mean_enc

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, L, D]
