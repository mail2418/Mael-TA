import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
    
def FFT_for_Period(x, k=2):
    # xf shape [B, T, C], denoting the amplitude of frequency(T) given the datapiece at B,N
    xf = torch.fft.rfft(x, dim=1)

    # find period by amplitudes: here we assume that the peroidic features are basically constant
    # in different batch and channel, so we mean out these two dimensions, getting a list frequency_list with shape[T]
    # each element at pos t of frequency_list denotes the overall amplitude at frequency (t)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0

    #by torch.topk(),we can get the biggest k elements of frequency_list, and its positions(i.e. the k-main frequencies in top_list)
    _, top_list = torch.topk(frequency_list, k)

    #Returns a new Tensor 'top_list', detached from the current graph.
    #The result will never require gradient.Convert to a numpy instance
    top_list = top_list.detach().cpu().numpy()

    #peroid:a list of shape [top_k], recording the peroids of mean frequencies respectively
    period = x.shape[1] // top_list

    #Here,the 2nd item returned has a shape of [B, top_k],representing the biggest top_k amplitudes
    # for each piece of data, with N features being averaged.
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):    ##configs is the configuration defined for TimesBlock
      super(TimesBlock, self).__init__()
      self.k = configs.top_k    ##k denotes how many top frequencies are top_k = 5
      self.d_ff = configs.d_ff #d_ff == dimension fcn (fourier convolution network) = 2048
                                                              #taken into consideration
      # parameter-efficient design
      self.conv = nn.Sequential(
          Inception_Block_V1(configs.d_model, self.d_ff,
                            num_kernels=configs.num_kernels),
          nn.GELU(),
          Inception_Block_V1(self.d_ff, configs.d_model,
                            num_kernels=configs.num_kernels)
      )

    def forward(self, x):
        B, T, N = x.size() # 32 x 100 x 512
            #B: batch size  T: length of time series  N:number of features
        period_list, period_weight = FFT_for_Period(x, self.k)
            #FFT_for_Period() will be shown later. Here, period_list([top_k]) denotes
            #the top_k-significant period and peroid_weight([B, top_k]) denotes its weight(amplitude)

        res = []
        for i in range(self.k): #self.k = 5
            period = period_list[i]

            # padding : to form a 2D map, we need total length of the sequence, plus the part
            # to be predicted, to be devisible by the peroid, so padding is needed
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - T), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x

            # reshape: we need each channel of a single piece of data to be a 2D variable,
            # Also, in order to implement the 2D conv later on, we need to adjust the 2 dimensions
            # to be convolutioned to the last 2 dimensions, by calling the permute() func.
            # Whereafter, to make the tensor contiguous in memory, call contiguous()
            out = out[:,:period,:,None].repeat(1,1,1,length // period) #32x2x100x512 
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous() #32x104x512x2

            #2D convolution to grap the intra- and inter- peroid information
            out = self.conv(out)

            # reshape back, similar to reshape
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)

            #truncating down the padded part of the output and put it to result
            res.append(out[:, : T, :])
        res = torch.stack(res, dim=-1) #res: 4D [B, length , N, top_k]

        # adaptive aggregation
        #First, use softmax to get the normalized weight from amplitudes --> 2D [B,top_k]
        period_weight = F.softmax(period_weight, dim=1)

        #after two unsqueeze(1),shape -> [B,1,1,top_k],so repeat the weight to fit the shape of res
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1) # T=100 N=512

        #add by weight the top_k peroids' result, getting the result of this TimesBlock
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x
        return res
    

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        #params init
        self.configs = configs
        self.layer = configs.e_layers # num of encoder layers

        #stack TimesBlock for e_layers times to form the main part of TimesNet, named model
        self.model = nn.ModuleList([TimesBlock(configs)for _ in range(configs.e_layers)])

        #embedding & normalization
        # enc_in is the encoder input size, the number of features for a piece of data
        # d_model is the dimension of embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                        configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            
    def forward(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        # tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        # delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)
        # TimesNet
        for i in range(self.layer): #Encoder Layer
            enc_out = self.layer_norm(self.model[i](enc_out)) #Anggap sudah di Encoder
            # enc_out = self.layer_norm(self.model[i](enc_out, tau=tau, delta=delta))
        # porject back
        dec_out = self.projection(enc_out)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1,dec_out.shape[1], 1))
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1)
        return dec_out