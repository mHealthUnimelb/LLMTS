#pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
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


class enc_mtan_classif(nn.Module):
 
    def __init__(self, input_dim, query, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cpu', n_classes=2, patch_len=64, stride=16):
        super(enc_mtan_classif, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = 0
        self.nhidden = nhidden
        self.query = query
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, n_classes))
        self.enc = nn.GRU(nhidden, nhidden)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    # def forward(self, x, time_steps):
    #     time_steps = time_steps.cpu()
    #     mask = x[:, :, self.dim:]
    #     mask = torch.cat((mask, mask), 2)
    #     if self.learn_emb:
    #         key = self.learn_time_embedding(time_steps).to(self.device)
    #         query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
    #     else:
    #         key = self.time_embedding(time_steps, self.embed_time).to(self.device)
    #         query = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)
    #
    #     out = self.att(query, key, x, mask)
    #     out = out.permute(1, 0, 2)
    #     _, out = self.enc(out)
    #     return self.classifier(out.squeeze(0))

    # patching
    # def forward(self, x, time_steps):
    #     batch_size, seq_len, dim = x.shape
    #     # compute the required padding
    #     full_patches = math.ceil((seq_len - self.patch_len) / self.stride) + 1
    #     total_length = (full_patches - 1) * self.stride + self.patch_len
    #     padding_needed = total_length - seq_len
    #     if padding_needed > 0:
    #         # pad x and time_steps
    #         pad_layer = nn.ReplicationPad1d((0, padding_needed))
    #         x = pad_layer(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
    #         time_steps = pad_layer(time_steps.unsqueeze(1)).squeeze(1)
    #
    #     # patching x and time_steps
    #     x_patches = x.unfold(dimension=1, size=self.patch_len,
    #                          step=self.stride)  # Shape: [batch_size, num_patches, dim, patch_len]
    #     x_patches = x_patches.permute(0, 1, 3, 2)  # Shape: [batch_size, num_patches, patch_len, dim]
    #     self.num_patches = x_patches.shape[1]
    #     x_patches = x_patches.reshape(batch_size * self.num_patches, self.patch_len, dim)
    #     print("x_patches shape: ", x_patches.shape)
    #     time_steps_patches = time_steps.unfold(dimension=1, size=self.patch_len,
    #                                            step=self.stride)  # Shape: [batch_size, num_patches, patch_len]
    #     time_steps_patches = time_steps_patches.reshape(batch_size * self.num_patches, self.patch_len)
    #     print("time_steps_patches shape: ", time_steps_patches.shape)
    #     print("query shape: ", self.query.shape)  # 256
    #     mask = x_patches[:, :, self.dim:]
    #     mask = torch.cat((mask, mask), dim=-1)
    #
    #     if self.learn_emb:
    #         key = self.learn_time_embedding(time_steps_patches).to(self.device)
    #         query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
    #     else:
    #         key = self.time_embedding(time_steps_patches, self.embed_time).to(self.device)
    #         query = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)
    #
    #     out = self.att(query, key, x_patches, mask)
    #     out = out.view(batch_size, self.num_patches * query.shape[-2], self.nhidden)
    #     out = out.permute(1, 0, 2)
    #     _, out = self.enc(out)
    #     return self.classifier(out.squeeze(0))

    # first attention, then patching
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

        out = self.att(query, key, x, mask) # batch_size, num_ref_points, embed_dim
        # out = out.permute(1, 0, 2)
        # _, out = self.enc(out)
        # return self.classifier(out.squeeze(0))

        batch_size, seq_len, dim = out.shape
        # compute the required padding
        full_patches = math.ceil((seq_len - self.patch_len) / self.stride) + 1
        total_length = (full_patches - 1) * self.stride + self.patch_len
        padding_needed = total_length - seq_len
        if padding_needed > 0:
            # pad x and time_steps
            pad_layer = nn.ReplicationPad1d((0, padding_needed))
            out = pad_layer(out.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

        # patching out
        out_patches = out.unfold(dimension=1, size=self.patch_len,
                             step=self.stride)  # Shape: [batch_size, num_patches, dim, patch_len]
        out_patches = out_patches.permute(0, 1, 3, 2)  # Shape: [batch_size, num_patches, patch_len, dim]
        self.num_patches = out_patches.shape[1]
        out_patches = out_patches.reshape(batch_size * self.num_patches, self.patch_len, dim)
        print("out_patches shape: ", out_patches.shape)

        out_patches = out_patches.view(batch_size, self.num_patches * query.shape[-2], self.nhidden)
        out = out.permute(1, 0, 2)
        _, out = self.enc(out)
        return self.classifier(out.squeeze(0))


class enc_mtan_classif_activity(nn.Module):
 
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cpu'):
        super(enc_mtan_classif_activity, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru = nn.GRU(nhidden, nhidden, batch_first=True)
        self.classifier = nn.Linear(nhidden, 11)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, x, time_steps):
        batch = x.size(0)
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
        out = self.att(key, key, x, mask)
        out, _ = self.gru(out)
        out = self.classifier(out)
        return out

