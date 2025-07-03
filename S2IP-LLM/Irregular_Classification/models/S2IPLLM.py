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


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.d_ff = 768
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
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

        elif self.task_name == "ir_classification":
            self.num_classes = configs.num_classes
            self.in_layer = nn.Linear(configs.patch_size * 3, configs.d_model)
            
            # if self.configs.classify_pertp:
            #     self.projection_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num + configs.prompt_length)), configs.seq_len)
            #     self.classifier = nn.Linear(configs.feature_dim, self.num_classes)
            # else:
            self.classifier = nn.Linear(int(configs.d_model / 3 * (self.patch_num + configs.prompt_length) * configs.feature_dim),
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

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):

        if self.task_name == 'long_term_forecast':
            dec_out, res = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], res  # [B, L, D]
        elif self.task_name == 'ir_classification':
            logits, res = self.classify(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return logits, res
        elif self.task_name == 'ir_forecast':
            dec_out, res = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], res  # [B, L, D]

        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

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

        outputs = outputs * stdev[:, :, :M]
        outputs = outputs + means[:, :, :M]

        return outputs, res

    def classify(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        print("x_enc shape: ", x_enc.shape)
        B, L, M = x_enc.shape
        print(f'B shape: {B}, L shape: {L}, M shape: {M}')  # B: 3  L: 2881  M: 83

        # normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # flatten (B, L, M) => (B*M, L) apply decomposition for each feature
        x = rearrange(x_enc, 'b l m -> (b m) l')
        print("x_enc shape: ", x_enc.shape)  # [3, 2881, 83]

        # decomposition function
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
        print("decomp_results shape: ", decomp_results.shape)  # [249 (batch_size*feature_dim), 2881, 3, 1]
        x = torch.tensor(decomp_results).to(self.gpt2.device)
        x = rearrange(x, 'b l c d  -> b c (d l)', c=3)
        print("x shape: ", x.shape)  # [249, 3, 2881]
        x = self.padding_patch_layer(x)
        print("x shape: ", x.shape)  # [249, 3, 2889]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # patching
        print("x shape: ", x.shape)  # [249, 3, 360, 16] [(batch_size*feature_dim), 3(decomp), num_patches, patch_size]
        x = rearrange(x, 'b c n p -> b n (c p)', c=3)
        print("x shape: ", x.shape)  # [249, 360, 48] [(batch_size*feature_dim), num_patches, 3(decomp) * patch_size]
        pre_prompted_embedding = self.in_layer(x.float())  # map 3(decomp) * patch_size to llm embedding_dim
        print("pre_prompted_embedding shape: ", pre_prompted_embedding.shape)  # [249, 360, 768]

        outs = self.prompt_pool(pre_prompted_embedding)  # add prompt embedding
        prompted_embedding = outs['prompted_embedding']
        print("prompted_embedding shape: ",
              prompted_embedding.shape)  # [249, 364, 768] [(batch_size*feature_dim), num_patches + prompt, embedding_dim]
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']

        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
        print("last_embedding shape: ", last_embedding.shape)  # [249 (batch_size*feature_dim), 364, 768]
        outputs = last_embedding.reshape(B * M * 3, -1)
        print("outputs shape: ", outputs.shape) # [747, 93184]

        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B, m=M, c=3)
        print("outputs shape: ", outputs.shape) # [3, 83, 3, 93184]
        outputs = outputs.sum(dim=2)
        print("outputs shape: ", outputs.shape) # [3, 83, 93184]
        
        # if self.configs.classify_pertp:
        #     outputs = self.projection_layer(outputs)
        #     outputs = outputs.permute(0, 2, 1)
        #     print("outputs shape: ", outputs.shape)
        #     logits = self.classifier(outputs)
        # else:
        outputs = outputs.reshape(B, -1)
        # nan_rows_mask = torch.isnan(outputs).all(dim=1)   # True where an entire row is NaN
        # num_nan_rows  = nan_rows_mask.sum().item()  # count them
        # print(f"logits before {num_nan_rows} rows are completely NaN")
        print("outputs shape: ", outputs.shape) # [3, 7734272]

        if self.configs.classify_pertp:
            outputs = outputs.unsqueeze(1)
            outputs = outputs.repeat(1, L, 1)

        logits = self.classifier(outputs)
        # nan_rows_mask = torch.isnan(logits).all(dim=1)   # True where an entire row is NaN
        # num_nan_rows  = nan_rows_mask.sum().item()  # count them
        # print(f"after classifer {num_nan_rows} rows are completely NaN")
        print("logits shape", logits.shape)

        res = dict()
        res['simlarity_loss'] = simlarity_loss

        return logits, res
