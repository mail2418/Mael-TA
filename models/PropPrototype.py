import torch
from torch import nn
import torch.nn.functional as F
import random, math


class URTPropagation(nn.Module):

  def __init__(self, key_dim, query_dim, hid_dim, temp=1, att="cosine"):
    super(URTPropagation, self).__init__()
    #bentuk seperti attentionLayer
    self.linear_q = nn.Linear(query_dim, hid_dim, bias=True)
    self.linear_k = nn.Linear(key_dim, hid_dim, bias=True)
    #self.linear_v_w = nn.Parameter(torch.rand(8, key_dim, key_dim))
    # self.linear_v_w = nn.Parameter(torch.eye(key_dim).unsqueeze(0).repeat(8,1,1)) 
    # self.linear_v_w = nn.Parameter(torch.eye(key_dim).unsqueeze(0).repeat(16,1,1)) 
    # x = torch.eye(key_dim).unsqueeze(0).repeat(8,1,1)
    # print(x.shape)
    # print(x)
    self.temp     = temp
    self.att      = att
    # how different the init is
    for m in self.modules():
      if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)

  def forward_transform(self, samples):
    bs, n_extractors, fea_dim = samples.shape
    '''
    if self.training:
      w_trans = torch.nn.functional.gumbel_softmax(self.linear_v_w, tau=10, hard=True)
    else:
      # y_soft = torch.softmax(self.linear_v_w, -1)
      # index = y_soft.max(-1, keepdim=True)[1]
      index = self.linear_v_w.max(-1, keepdim=True)[1]
      y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
      w_trans = y_hard
      # w_trans = y_hard - y_soft.detach() + y_soft
    '''
    w_trans = self.linear_v_w 
    # compute regularization
    regularization = w_trans @ torch.transpose(w_trans, 1, 2)
    samples = samples.view(bs, n_extractors, fea_dim, 1)
    w_trans = w_trans.view(1, 8, fea_dim, fea_dim)
    return torch.matmul(w_trans, samples).view(bs, n_extractors, fea_dim), (regularization**2).sum()

  def forward(self, cat_proto):
    # cat_proto n_class*8*512 
    # return: n_class*8
    n_class, n_extractors, fea_dim = cat_proto.shape #B L H = x.shape
    q       = cat_proto.view(n_class, -1) # n_class * 8_512
    k       = cat_proto                   # n_class * 8 * 512
    q_emb   = self.linear_q(q)            # n_class * hid_dim
    k_emb   = self.linear_k(k)            # n_class * 8 * hid_dim  | 8 * hid_dim
    if self.att == "cosine":
      raw_score   = F.cosine_similarity(q_emb.view(n_class, 1, -1), k_emb.view(n_class, n_extractors, -1), dim=-1)
    elif self.att == "dotproduct":
      raw_score   = torch.sum( q_emb.view(n_class, 1, -1) * k_emb.view(n_class, n_extractors, -1), dim=-1 ) / (math.sqrt(fea_dim)) 
    else:
      raise ValueError('invalid att type : {:}'.format(self.att))
    # score   = F.softmax(self.temp * raw_score, dim=1)
    score   = F.softmax(self.temp * raw_score, dim=0)

    return score

class MultiHeadURT(nn.Module):
  def __init__(self, key_dim, query_dim, hid_dim, temp=1, att="cosine", n_head=1):
    super(MultiHeadURT, self).__init__()
    layers = []
    for _ in range(n_head):
      layer = URTPropagation(key_dim, query_dim, hid_dim, temp, att)
      layers.append(layer)
    self.layers = nn.ModuleList(layers)

  def forward(self, cat_proto):
    score_lst = []
    for i, layer in enumerate(self.layers):
      score = layer(cat_proto)
      score_lst.append(score)
    # n_class * n_extractor * n_head
    fin_score = torch.stack(score_lst, dim=-1)
    fin_score = torch.mean(fin_score,axis=-1)
    # return torch.stack(score_lst, dim=-1)
    # print("check urt out shape")
    # print(fin_score.shape)
    return fin_score