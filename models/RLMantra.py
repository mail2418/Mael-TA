import torch
from torch import nn
import torch.nn.functional as F
import random, math
from models import MaelNet, KBJNet, DCDetector, ns_Transformer, FEDFormer, TimesNet, MaelNetB1, MaelNetS1, ns_TransformerB1, ns_TransformerS1

# class RLMantra(nn.Module):

#   def __init__(self, key_dim, query_dim, hid_dim, temp=1, att="cosine"):
#     super(RLMantra, self).__init__()
#     #bentuk seperti attentionLayer
#     self.linear_q = nn.Linear(query_dim, hid_dim, bias=True)
#     self.linear_k = nn.Linear(key_dim, hid_dim, bias=True)
#     self.temp     = temp
#     self.att      = att
#     # how different the init is
#     for m in self.modules():
#       if isinstance(m, nn.Linear):
#         m.weight.data.normal_(0, 0.001)

#   def forward(self, cat_proto):
#     # cat_proto n_class*8*512 
#     # return: n_class*8
#     n_class, n_extractors, fea_dim = cat_proto.shape #B L H = x.shape
#     q       = cat_proto.view(n_class, -1) # n_class * 8_512
#     k       = cat_proto                   # n_class * 8 * 512
#     q_emb   = self.linear_q(q)            # n_class * hid_dim
#     k_emb   = self.linear_k(k)            # n_class * 8 * hid_dim  | 8 * hid_dim
#     if self.att == "cosine":
#       raw_score   = F.cosine_similarity(q_emb.view(n_class, 1, -1), k_emb.view(n_class, n_extractors, -1), dim=-1)
#     elif self.att == "dotproduct":
#       raw_score   = torch.sum( q_emb.view(n_class, 1, -1) * k_emb.view(n_class, n_extractors, -1), dim=-1 ) / (math.sqrt(fea_dim)) 
#     else:
#       raise ValueError('invalid att type : {:}'.format(self.att))
#     # score   = F.softmax(self.temp * raw_score, dim=1)
#     score   = F.softmax(self.temp * raw_score, dim=0)

#     return score

class RLMantra:
    def __init__(self,args, device):
      model_dict = {
          "MaelNet"    : MaelNet,
          "MaelNetB1":MaelNetB1,
          "MaelNetS1":MaelNetS1,
          "KBJNet"     : KBJNet,
          "DCDetector" : DCDetector,
          "NSTransformer": ns_Transformer,
          "NSTransformerB1": ns_TransformerB1,
          "NSTransformerS1": ns_TransformerS1,
          "FEDFormer" : FEDFormer,
          "TimesNet": TimesNet,
      }
      self.actor = model_dict[args.model_actor].Model(args).to(device)
      self.target_actor = model_dict[args.model_actor].Model(args).to(device)

      self.critic = model_dict[args.model_critic].Model(args).to(device)
      self.target_critic = model_dict[args.model_critic].Model(args).to(device)
    def forward():
       pass    