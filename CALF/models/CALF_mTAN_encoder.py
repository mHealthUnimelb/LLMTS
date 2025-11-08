# Use encoder from mTAND

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from models.GPT2_arch import AccustumGPT2Model
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

        # lstm
        out = out.view(batch_size * self.patch_num, self.patch_len, self.nhidden)
        _, (out, _) = self.local_lstm(out)
        out = out.squeeze(0).view(batch_size, self.patch_num, self.nhidden)

        return out

class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=1, num_encoder_layers=1, device='cpu',
                 num_ref_points=128, learn_emb=True, num_ca_heads=1, configs=None):
        super(Encoder_PCA, self).__init__()
        
        self.encoder = enc_mtan(input_dim, torch.linspace(0, 1., num_ref_points), nhidden=hidden_dim,
                                    embed_time=128, num_heads=num_heads, learn_emb=learn_emb, num_ref_points=configs.num_ref_points, patch_len=configs.patch_len, stride=configs.stride)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_ca_heads)

        self.word_embedding = word_embedding.T

    def forward(self, x, time_steps):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)

        x = rearrange(x, 'b m l -> b l m')

        x = self.encoder(x, time_steps)

        x_time = x

        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, w_ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x_time, x


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )

        self.task_name = configs.task_name

        self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                      output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                           output_hidden_states=True)  # loads a pretrained GPT-2 base model

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
        self.gpt2 = get_peft_model(self.gpt2, peft_config)

        word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for i, (name, param) in enumerate(self.gpt2_text.named_parameters()):
            if 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.time_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)])

        self.text_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)])

        self.patch_num = (configs.num_ref_points - configs.patch_len) // configs.stride + 2

        self.in_layer = Encoder_PCA(input_dim=1, word_embedding=word_embedding, hidden_dim=configs.d_model,
                                    num_heads=configs.num_encoder_heads, device=device,
                                    num_ref_points=configs.num_ref_points,
                                    learn_emb=configs.learn_emb,
                                    num_ca_heads=configs.num_ca_heads, configs=configs)


        if configs.classify_pertp:
            self.projection_layer = nn.Linear(configs.d_model, configs.seq_len)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification_mTAN_encoder':
            self.out_layer = nn.Linear(configs.d_model * self.patch_num * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    def forecast(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time[:, -M:, :])
        outputs_text = self.out_layer(outputs_text[:, -M:, :])

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
        }

    def classification(self, x, time_steps):
        B, L, M = x.shape

        # treat each channel as separate sequence
        n_vars = M // 2
        values = x[:, :, :n_vars]
        masks = x[:, :, n_vars:2*n_vars]
        time_steps = time_steps.unsqueeze(1).expand(-1, n_vars, -1).reshape(B * n_vars, L)

        x = torch.stack((values, masks), dim=-1) # (B, T, n_vars, 2)

        x = x.permute(0, 2, 1, 3).reshape(B * n_vars, L, 2)

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x, time_steps)
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = outputs_time.reshape(B, -1)
        outputs_text = outputs_text.reshape(B, -1)

        if self.configs.classify_pertp:
            outputs_time = outputs_time.unsqueeze(1) # (batch, 1, hidden)
            outputs_text = outputs_text.unsqueeze(1)
            outputs_time = outputs_time.repeat(1, L, 1) # (batch, seq_len, hidden)
            outputs_text = outputs_text.repeat(1, L, 1)

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
        }

    def imputation(self, x, mask):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)

        stdev = torch.sqrt(torch.sum(x ** 2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
        }

    def anomaly_detection(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
        }

    def forward(self, x, time_steps, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        if self.task_name == 'classification_mTAN_encoder':
            output = self.classification(x, time_steps)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output