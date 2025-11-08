#!pip install transformers

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer
from utils.tokenization import SerializerSettings, serialize_arr, serialize_arr
from .prompt import Prompt


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
        batch, seq_len, dim = value.size()
        if mask is not None:
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
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., num_ref_points=256, patch_size=16, stride=8, device='cuda'):
        super(enc_mtan, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.patch_size = patch_size
        self.stride = stride
        self.patch_num = 0
        self.patch_num = (num_ref_points - self.patch_size) // self.stride + 1
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
        out = out.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        out = out.permute(0, 2, 3, 1).contiguous() # (batch, num_patches, patch_size, dim)

        # lstm
        out = out.view(batch_size * self.patch_num, self.patch_size, self.nhidden)
        _, (out, _) = self.local_lstm(out)
        out = out.squeeze(0).view(batch_size, self.patch_num, self.nhidden)

        return out


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.num_ref_points
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.d_ff = 768
        self.patch_num = (configs.num_ref_points - self.patch_size) // self.stride + 1
        self.patch_num += 1

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if configs.pretrained == True:
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        else:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # False

        if self.task_name == 'long_term_forecast':
            self.in_layer = nn.Linear(configs.patch_size * 3, configs.d_model)
            self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num + configs.prompt_length)),
                                       configs.pred_len)

            self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform',
                                      prompt_pool=False,
                                      prompt_key=True, pool_size=self.configs.pool_size,
                                      top_k=self.configs.prompt_length, batchwise_prompt=False,
                                      prompt_key_init=self.configs.prompt_init, wte=self.gpt2.wte.weight)

            for layer in (self.gpt2, self.in_layer, self.out_layer):
                layer.cuda()
                layer.train()

        elif self.task_name == "ir_classification_mTAN":
            self.num_classes = configs.num_classes
            self.in_layer = enc_mtan(1, torch.linspace(0, 1., configs.num_ref_points), nhidden=configs.d_model,
                                embed_time=128, num_heads=configs.num_heads_mtan, learn_emb=configs.learn_emb, num_ref_points=configs.num_ref_points,
                                patch_size=configs.patch_size, stride=configs.stride)
            
            self.classifier = nn.Linear(int(configs.d_model * (self.patch_num + configs.prompt_length) * configs.feature_dim),
                                        self.num_classes)

            self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform',
                                      prompt_pool=False,
                                      prompt_key=True, pool_size=self.configs.pool_size,
                                      top_k=self.configs.prompt_length, batchwise_prompt=False,
                                      prompt_key_init=self.configs.prompt_init, wte=self.gpt2.wte.weight)

            for layer in (self.gpt2, self.in_layer, self.classifier):
                layer.cuda()
                layer.train()

        elif self.task_name == 'ir_forecast':
            self.in_layer = nn.Linear(configs.patch_size * 3, configs.d_model)
            self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num + configs.prompt_length)),
                                       configs.pred_len)

            self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform',
                                      prompt_pool=False,
                                      prompt_key=True, pool_size=self.configs.pool_size,
                                      top_k=self.configs.prompt_length, batchwise_prompt=False,
                                      prompt_key_init=self.configs.prompt_init, wte=self.gpt2.wte.weight)

            for layer in (self.gpt2, self.in_layer, self.out_layer):
                layer.cuda()
                layer.train()

    def forward(self, x_enc, time_steps, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):

        if self.task_name == 'long_term_forecast':
            dec_out, res = self.forecast(x_enc, time_steps, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], res  # [B, L, D]
        elif self.task_name == 'ir_classification':
            logits, res = self.classify(x_enc, time_steps, x_mark_enc, x_dec, x_mark_dec)
            return logits, res
        elif self.task_name == 'ir_classification_mTAN':
            logits, res = self.classify(x_enc, time_steps, x_mark_enc, x_dec, x_mark_dec)
            return logits, res
        elif self.task_name == 'ir_forecast':
            dec_out, res = self.forecast(x_enc, time_steps, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], res  # [B, L, D]

        return None

    def forecast(self, x_enc, time_steps, x_mark_enc, x_dec, x_mark_dec):

        B, L, M = x_enc.shape

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x = rearrange(x_enc, 'b l m -> (b m) l')

        def decompose(x):
            df = pd.DataFrame(x)
            trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(
                method='ffill')
            detrended = df - trend
            seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(
                method='bfill').fillna(method='ffill')
            residuals = df - trend - seasonal
            combined = np.stack([trend, seasonal, residuals], axis=1)
            return combined

        decomp_results = np.apply_along_axis(decompose, 1, x.cpu().numpy())
        x = torch.tensor(decomp_results).to(self.gpt2.device)
        x = rearrange(x, 'b l c d  -> b c (d l)', c=3)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b c n p -> b n (c p)', c=3)
        pre_prompted_embedding = self.in_layer(x.float())

        outs = self.prompt_pool(pre_prompted_embedding)
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']

        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
        outputs = self.out_layer(last_embedding.reshape(B * M * 3, -1))

        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B, m=M, c=3)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')

        res = dict()
        res['simlarity_loss'] = simlarity_loss
        res['similarity'] = sim

        outputs = outputs * stdev[:, :, :M]
        outputs = outputs + means[:, :, :M]

        return outputs, res

    def classify(self, x_enc, time_steps, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, L, M = x_enc.shape

        # treat each channel as separate sequence
        n_vars = M // 2
        values = x_enc[:, :, :n_vars]
        masks = x_enc[:, :, n_vars:2*n_vars]
        time_steps = time_steps.unsqueeze(1).expand(-1, n_vars, -1).reshape(B * n_vars, L)

        x = torch.stack((values, masks), dim=-1) # (B, L, n_vars, 2)
        x = x.permute(0, 2, 1, 3).reshape(B * n_vars, L, 2)
        
        pre_prompted_embedding = self.in_layer(x, time_steps)

        outs = self.prompt_pool(pre_prompted_embedding)  # add prompt embedding
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']
        total_prompt_len = outs['total_prompt_len']

        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
    
        outputs = last_embedding.reshape(B, -1)

        if self.configs.classify_pertp:
            outputs = outputs.unsqueeze(1)
            outputs = outputs.repeat(1, L, 1)

        logits = self.classifier(outputs)

        res = dict()
        res['simlarity_loss'] = simlarity_loss
        res['similarity'] = sim
        res['prompted_embedding'] = prompted_embedding
        res['total_prompt_len'] = total_prompt_len

        return logits, res
