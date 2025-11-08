import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
import os

class multiTimeAttention(nn.Module):

    def __init__(self, input_dim, nhidden=16,
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim * num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        # attention query shape: torch.Size([1, 1, 128, 128]), key shape: torch.Size([128, 1, 190, 128]), value shape: torch.Size([128, 1, 190, 82]), mask shape: torch.Size([128, 1, 190, 82])
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(batch, -1, self.h * dim)
        return self.linears[-1](x)
    
class enc_mtan(nn.Module):

    def __init__(self, input_dim, query, nhidden=16,
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., num_ref_points=256, patch_len=16, stride=8, device='cuda'):
        super(enc_mtan, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = 0
        self.patch_num = (num_ref_points - self.patch_len) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.time_att = multiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)

        # LSTM
        self.local_lstm = nn.LSTM(input_size=self.nhidden, hidden_size=self.nhidden, batch_first=True)

        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=self.device)
        position = 48. * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device) *
                             -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
            query = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)

        out = self.time_att(query, key, x, mask)  # batch_size, num_ref_points, embed_dim

        batch_size, seq_len, dim = out.shape

        # patching
        out = rearrange(out, 'b l m -> b m l')
        out = self.padding_patch_layer(out)
        out = out.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        out = out.permute(0, 2, 3, 1).contiguous() # (batch, num_patches, patch_len, dim)

        # self.patch_num = out.shape[1]

        # lstm
        out = out.view(batch_size * self.patch_num, self.patch_len, self.nhidden)
        _, (out, _) = self.local_lstm(out)
        out = out.squeeze(0).view(batch_size, self.patch_num, self.nhidden)

        return out