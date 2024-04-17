# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.Embed import DataEmbedding
# from layers.Attention import AutoCorrelation, AttentionLayer
# from utils.tools import series_decomp, my_Layernorm
# from layers.Encoder import Encoder, EncoderLayer
# from layers.Decoder import Decoder, DecoderLayer

# class Model(nn.Module):
#     """
#     Autoformer is the first method to achieve the series-wise connection,
#     with inherent O(LlogL) complexity
#     Paper link: https://openreview.net/pdf?id=I55UqU-M11y
#     """

#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.name = "AutoFormer"
#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.label_len = configs.label_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention

#         # Decomp
#         kernel_size = configs.moving_avg
#         self.decomp = series_decomp(kernel_size)

#         # Embedding
#         self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                   configs.dropout)
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                         output_attention=configs.output_attention),
#                         configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     moving_avg=configs.moving_avg,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=my_Layernorm(configs.d_model)
#         )
#         # Decoder
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                       configs.dropout)
#             self.decoder = Decoder(
#                 [
#                     DecoderLayer(
#                         AttentionLayer(
#                             AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
#                                             output_attention=False),
#                             configs.d_model, configs.n_heads),
#                         AttentionLayer(
#                             AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                             output_attention=False),
#                             configs.d_model, configs.n_heads),
#                         configs.d_model,
#                         configs.c_out,
#                         configs.d_ff,
#                         moving_avg=configs.moving_avg,
#                         dropout=configs.dropout,
#                         activation=configs.activation,
#                     )
#                     for l in range(configs.d_layers)
#                 ],
#                 norm_layer=my_Layernorm(configs.d_model),
#                 projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
#             )
#         if self.task_name == 'imputation':
#             self.projection = nn.Linear(
#                 configs.d_model, configs.c_out, bias=True)
#         if self.task_name == 'anomaly_detection':
#             self.projection = nn.Linear(
#                 configs.d_model, configs.c_out, bias=True)
#         if self.task_name == 'classification':
#             self.act = F.gelu
#             self.dropout = nn.Dropout(configs.dropout)
#             self.projection = nn.Linear(
#                 configs.d_model * configs.seq_len, configs.num_class)

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         # decomp init
#         mean = torch.mean(x_enc, dim=1).unsqueeze(
#             1).repeat(1, self.pred_len, 1)
#         zeros = torch.zeros([x_dec.shape[0], self.pred_len,
#                              x_dec.shape[2]], device=x_enc.device)
#         seasonal_init, trend_init = self.decomp(x_enc)
#         # decoder input
#         trend_init = torch.cat(
#             [trend_init[:, -self.label_len:, :], mean], dim=1)
#         seasonal_init = torch.cat(
#             [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
#         # enc
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)
#         # dec
#         dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
#         seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
#                                                  trend=trend_init)
#         # final
#         dec_out = trend_part + seasonal_part
#         return dec_out

#     def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
#         # enc
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)
#         # final
#         dec_out = self.projection(enc_out)
#         return dec_out

#     def anomaly_detection(self, x_enc):
#         # enc
#         enc_out = self.enc_embedding(x_enc, None)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)
#         # final
#         dec_out = self.projection(enc_out)
#         return dec_out

#     def classification(self, x_enc, x_mark_enc):
#         # enc
#         enc_out = self.enc_embedding(x_enc, None)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         # Output
#         # the output transformer encoder/decoder embeddings don't include non-linearity
#         output = self.act(enc_out)
#         output = self.dropout(output)
#         # zero-out padding embeddings
#         output = output * x_mark_enc.unsqueeze(-1)
#         # (batch_size, seq_length * d_model)
#         output = output.reshape(output.shape[0], -1)
#         output = self.projection(output)  # (batch_size, num_classes)
#         return output

#     def forward(self, x_enc):
#         dec_out = self.anomaly_detection(x_enc)
#         return dec_out  # [B, L, D]