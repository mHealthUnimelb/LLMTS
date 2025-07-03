# first encode, then patching, add variable-wise attention
# with LSTM
# use linear mapping to reduce vocab_size
# treat each channel as a sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
# from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from models.GPT2_arch import AccustumGPT2Model
from transformers import GPT2Tokenizer
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
        print(f"attention query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}, mask shape: {mask.shape}")
        # attention query shape: torch.Size([1, 1, 128, 128]), key shape: torch.Size([128, 1, 190, 128]), value shape: torch.Size([128, 1, 190, 82]), mask shape: torch.Size([128, 1, 190, 82])
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        print("scores shape: ", scores.shape) # (128, 1, 128, 190, 82)
        print("mask shape: ", mask.shape) # (128, 1, 190, 82)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        print(
            f"forward query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}, mask shape: {mask.shape}")
        batch, seq_len, dim = value.size()
        print(f"batch: {batch}, seq_len: {seq_len}, dim: {dim}") # 128, 190, 768
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        print("mask shape: ", mask.shape)
        value = value.unsqueeze(1)
        print("value shape: ", value.shape)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(batch, -1, self.h * dim)
        return self.linears[-1](x)


# class enc_mtan_rnn(nn.Module):
#     def __init__(self, input_dim, query, latent_dim=2, nhidden=16,
#                  embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
#         super(enc_mtan_rnn, self).__init__()
#         self.embed_time = embed_time
#         self.dim = input_dim
#         print("self.dim: ", self.dim)
#         self.device = device
#         self.nhidden = nhidden
#         self.query = query
#         self.learn_emb = learn_emb
#         self.att = multiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
#         self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
#         self.hiddens_to_z0 = nn.Sequential(
#             nn.Linear(2 * nhidden, 50),
#             nn.ReLU(),
#             nn.Linear(50, latent_dim * 2))
#         print("latent_dim: ", latent_dim)
#         if learn_emb:
#             self.periodic = nn.Linear(1, embed_time - 1)
#             self.linear = nn.Linear(1, 1)
#
#     def learn_time_embedding(self, tt):
#         tt = tt.to(self.device)
#         tt = tt.unsqueeze(-1)
#         out2 = torch.sin(self.periodic(tt))
#         out1 = self.linear(tt)
#         return torch.cat([out1, out2], -1)
#
#     def fixed_time_embedding(self, pos):
#         d_model = self.embed_time
#         pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
#         position = 48. * pos.unsqueeze(2)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(np.log(10.0) / d_model))
#         pe[:, :, 0::2] = torch.sin(position * div_term)
#         pe[:, :, 1::2] = torch.cos(position * div_term)
#         return pe
#
#     def forward(self, x, time_steps):
#         print("x shape: ", x.shape) # (128, 190, 82)
#
#         time_steps = time_steps.cpu()
#         print("time_steps shape: ", time_steps.shape) # (128, 190)
#
#         mask = x[:, :, self.dim:]
#         print("mask shape: ", mask.shape) # (128, 190, 41)
#
#         mask = torch.cat((mask, mask), 2)
#         print("mask shape: ", mask.shape) # (128, 190, 82)
#
#         if self.learn_emb:
#             key = self.learn_time_embedding(time_steps).to(self.device)
#             query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
#         else:
#             key = self.fixed_time_embedding(time_steps).to(self.device)
#             query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
#         out = self.att(query, key, x, mask)
#         print("out shape: ", out.shape) # 128, 128, 768
#
#         out, _ = self.gru_rnn(out)
#         print("out shape: ", out.shape) # 128, 128, 1536
#
#         out = self.hiddens_to_z0(out)
#         print("out shape: ", out.shape) # 128, 128, 768
#         return out


class enc_mtan(nn.Module):

    def __init__(self, input_dim, query, nhidden=16,
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., patch_len=128, stride=64, device='cuda'):
        super(enc_mtan, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.patch_len = patch_len
        self.num_patches = 0
        self.stride = stride
        self.nhidden = nhidden
        self.query = query
        print("query shape: ", query.shape)
        self.time_att = multiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        # self.var_att = nn.MultiheadAttention(embed_dim=nhidden, num_heads=num_heads)
        # attention pooling layer
        # self.attention_pooling = nn.Sequential(
        #     nn.Linear(self.nhidden, 1),
        #     nn.Softmax(dim=2)
        # )

        # adaptive pooling layers
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(1024)

        # # GRU
        # self.local_gru = nn.GRU(input_size=self.nhidden, hidden_size=self.nhidden, batch_first=True)

        # LSTM
        self.local_lstm = nn.LSTM(input_size=self.nhidden, hidden_size=self.nhidden, batch_first=True)

        # global attention
        # self.global_att = nn.MultiheadAttention(embed_dim=self.nhidden, num_heads=4)

        # CNN
        # self.conv1d = nn.Conv1d(in_channels=self.nhidden, out_channels=self.nhidden, kernel_size=query.shape[-1])

        # self.classifier = nn.Sequential(
        #     nn.Linear(nhidden, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 2))
        # self.enc = nn.GRU(nhidden, nhidden)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        print("tt shape: ", tt.shape) # (128, 2, 100)   (1, 256)
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
        print("time_steps shape", time_steps.shape)
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
        print("out shape: ", out.shape)
        # out = out.permute(1, 0, 2)  # Shape: [num_ref_points, batch_size, nhidden]
        # out, _ = self.var_att(out, out, out)
        # out = out.permute(1, 0, 2)
        # print("out shape: ", out.shape)
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
        out_patches = out_patches.permute(0, 1, 3, 2).contiguous()  # Shape: [batch_size, num_patches, patch_len, dim]
        print("out_patches shape: ", out_patches.shape)
        self.num_patches = out_patches.shape[1]

        # # patch positional embeddings
        # patch_positions = torch.arange(self.num_patches, device=self.device).unsqueeze(0).repeat(batch_size, 1)  # Shape: [batch_size, num_patches]
        # patch_pos_emb = self.time_embedding(patch_positions, self.embed_time).to(
        #     self.device)  # Shape: [batch_size, num_patches, embed_time]
        #
        # # reshape patch_pos_emb to align with key and query
        # patch_pos_emb_key = patch_pos_emb.unsqueeze(2).expand(-1, -1, self.patch_len,
        #                                                   -1)  # Shape: [batch_size, num_patches, patch_len, embed_time]
        # patch_pos_emb_key = patch_pos_emb_key.reshape(batch_size * self.num_patches, self.patch_len, -1).contiguous() # [batch_size * num_patches, patch_len, embed_time]
        # print("patch_pos_emb shape: ", patch_pos_emb.shape)  # (batch_size, num_patches, patch_len, embed_time)
        # #
        # # Add patch positional embeddings to key and query
        # key += patch_pos_emb_key  # Broadcasting over batch_size, num_patches, patch_len, embed_time
        # print("key shape after adding patch_pos_emb: ", key.shape)
        # Expand query to match batch_num_patches and add patch positional embeddings
        # query = query.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, num_query_points, embed_time]
        # query = query.expand(batch_size * self.num_patches, -1,
        #                      -1)  # Shape: [batch_size, num_patches, num_query_points, embed_time]
        # print("query shape: ", query.shape)
        # patch_pos_emb_query = patch_pos_emb.view(batch_size * self.num_patches, 1, self.embed_time) # [batch_size * num_patches, 1, embed_time]
        # query = query.clone() + patch_pos_emb_query  # Using the fact that patch_pos_emb has shape [batch_size, num_patches, patch_len, embed_time]
        # print("query shape after adding patch_pos_emb: ", query.shape)

        # Reshape back to (batch_size, total_seq_len, embedding_dim)
        # out = out.view(batch_size, -1, self.nhidden)
        # out = out.view(batch_size, self.num_patches, query.shape[-2], self.nhidden)
        # out = out_patches.view(batch_size * self.num_patches, self.patch_len, self.nhidden)
        # print("out shape: ", out.shape)

        # adaptive pool
        # out = out.view(batch_size, self.num_patches * query.shape[-2], self.nhidden)
        # if (self.num_patches * query.shape[-2]) > 1024:
        #     out = out.permute(0, 2, 1)
        #     out = self.adaptive_pool(out)
        #     out = out.permute(0, 2, 1)
        # print("out shape: ", out.shape)

        # # attention layer
        # out = out.view(batch_size, self.num_patches, query.shape[-2], self.nhidden)
        # attention_weights = self.attention_pooling(out)
        # out = (out * attention_weights).sum(dim=2)
        # print("out shape: ", out.shape)

        # gru
        # out = out.view(batch_size * self.num_patches, query.shape[-2], self.nhidden)
        # _, out = self.local_gru(out)
        # out = out.squeeze(0).view(batch_size, self.num_patches, self.nhidden)
        # print("out shape: ", out.shape)
        # out = out.view(batch_size * self.num_patches, query.shape[-2], self.nhidden)

        # lstm
        out = out_patches.view(batch_size * self.num_patches, self.patch_len, self.nhidden)
        _, (out, _) = self.local_lstm(out)
        out = out.squeeze(0).view(batch_size, self.num_patches, self.nhidden)
        print("out shape: ", out.shape)

        # # Add patch positional embeddings to key and query
        # # patch positional embeddings
        # patch_positions = torch.arange(self.num_patches, device=self.device).unsqueeze(0).repeat(batch_size, 1)  # Shape: [batch_size, num_patches]
        # patch_pos_emb = self.time_embedding(patch_positions, out.shape[2]).to(
        #     self.device)  # Shape: [batch_size, num_patches, embed_time]
        # out += patch_pos_emb
        # print("out shape after adding patch_pos_emb: ", out.shape)

        # # global attention
        # out = out.transpose(0, 1)  # Shape: (num_patches, batch_size, nhidden)
        # global_out, _ = self.global_att(out, out, out)
        # out = global_out.transpose(0, 1)  # Shape: (batch_size, num_patches, nhidden)

        # CNN
        # out = out.view(batch_size * self.num_patches, self.nhidden, query.shape[-2])
        # out = self.conv1d(out)
        # print("out shape: ", out.shape) # (704, 768, 241)
        # out = out.squeeze(-1)
        # print("out shape: ", out.shape)
        # out = out.view(batch_size, self.num_patches, self.nhidden)
        # print("out shape: ", out.shape)

        # out = out.reshape(batch_size, out.shape[1] * self.num_patches, self.nhidden)
        # print("out reshape: ", out.shape)
        # out = out.permute(1, 0, 2)
        # _, out = self.enc(out)
        # return self.classifier(out.squeeze(0))
        return out

class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=1, num_encoder_layers=1, device='cpu',
                 num_ref_points=128, latent_dim=32, learn_emb=True, patch_len=16, stride=8, num_ca_heads=1, prompt_embeddings=None):
        super(Encoder_PCA, self).__init__()
        # self.linear = nn.Linear(input_dim, hidden_dim)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.encoder = enc_mtan_rnn(hidden_dim, torch.linspace(0, 1., num_ref_points), latent_dim,
        #                             nhidden=hidden_dim,
        #                             embed_time=128, learn_emb=learn_emb, num_heads=num_heads).to(device)
        print("input dim: ", input_dim)
        # self.encoder = enc_mtan_rnn(input_dim, torch.linspace(0, 1., num_ref_points), latent_dim,
        #                             nhidden=hidden_dim,
        #                             embed_time=128, learn_emb=learn_emb, num_heads=num_heads).to(device)
        self.encoder = enc_mtan(1, torch.linspace(0, 1., num_ref_points), nhidden=hidden_dim,
                                embed_time=128, num_heads=num_heads, learn_emb=learn_emb, patch_len=patch_len,
                                stride=stride).to(device)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_ca_heads)

        # self.word_embedding = word_embedding.T
        self.word_embedding = word_embedding
        self.vocab_size = word_embedding.shape[0]
        print("vocab size:", self.vocab_size)
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.prompt_embeddings = prompt_embeddings

        self.num_patches = math.ceil((num_ref_points - patch_len) / stride) + 1

    def forward(self, x, time_steps):
        B = x.shape[0]

        word_embedding = self.mapping_layer(self.word_embedding.permute(1, 0)).permute(1, 0)
        # add prompt for word token
        word_embedding = torch.cat([self.prompt_embeddings.squeeze(0), word_embedding], dim=0)
        print("word embedding shape: ", word_embedding.shape)

        if word_embedding.ndim == 2:
            word_embedding = word_embedding.repeat(B, 1, 1)
        elif word_embedding.shape[0] != B:
            word_embedding = word_embedding[0].repeat(B, 1, 1)
        print("word embedding shape: ", word_embedding.shape)

        x = rearrange(x, 'b m l -> b l m')
        print("x shape: ", x.shape) # (128, 82, 190)  (128, 190, 82)
        # x = self.linear(x)

        # x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        # x = self.encoder(x.transpose(0, 1), time_steps).transpose(0, 1)
        x = self.encoder(x, time_steps)
        print("x shape after encoder", x.shape)

        x_time = x

        q = x.transpose(0, 1)
        k = v = word_embedding.transpose(0, 1)
        x, w_ = self.cross_attention(q, k, v)
        print("weights shape: ", w_.shape)

        x = x.transpose(0, 1)

        return x_time, x


# class AttentionPooling(nn.Module):
#     def __init__(self, embedding_size):
#         super(AttentionPooling, self).__init__()
#         self.attention = nn.Linear(embedding_size, 1)
#
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, embedding)
#         weights = torch.softmax(self.attention(x), dim=1)  # Shape: (batch_size, seq_len, 1)
#         weighted_output = torch.sum(weights * x, dim=1)  # Shape: (batch_size, embedding)
#         return weighted_output


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )

        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     inference_mode=False,
        #     r=configs.r,
        #     lora_alpha=configs.lora_alpha,
        #     lora_dropout=configs.lora_dropout,
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        # )

        self.task_name = configs.task_name

        self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                      output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                           output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.word_embeddings = self.gpt2_text.wte.state_dict()['weight'].to(device)

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
        self.gpt2 = get_peft_model(self.gpt2, peft_config)

        # self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        # self.llama_config.num_hidden_layers = 6
        # self.llama_config.output_attentions = True
        # self.llama_config.output_hidden_states = True
        # try:
        #     self.llama = LlamaModel.from_pretrained(
        #         # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
        #         'huggyllama/llama-7b',
        #         trust_remote_code=True,
        #         local_files_only=True,
        #         config=self.llama_config,
        #         # load_in_4bit=True
        #     )
        #     self.llama_text = LlamaModel.from_pretrained(
        #         # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
        #         'huggyllama/llama-7b',
        #         trust_remote_code=True,
        #         local_files_only=True,
        #         config=self.llama_config,
        #         # load_in_4bit=True
        #     )
        # except EnvironmentError:  # downloads model from HF is not already done
        #     print("Local model files not found. Attempting to download...")
        #     self.llama = LlamaModel.from_pretrained(
        #         # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
        #         'huggyllama/llama-7b',
        #         trust_remote_code=True,
        #         local_files_only=False,
        #         config=self.llama_config,
        #         # load_in_4bit=True
        #     )
        #     self.llama_text = LlamaModel.from_pretrained(
        #         # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
        #         'huggyllama/llama-7b',
        #         trust_remote_code=True,
        #         local_files_only=False,
        #         config=self.llama_config,
        #         # load_in_4bit=True
        #     )
        # self.llama = get_peft_model(self.llama, peft_config)
        # self.llama_text = get_peft_model(self.llama_text, peft_config)

        # word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)
        # print("word_embedding_path: ", configs.word_embedding_path)

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

        # for name, param in self.llama.named_parameters():
        #     if 'norm' in name or 'embed_tokens' in name or 'lora' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        #
        # for name, param in self.llama_text.named_parameters():
        #     if 'embed_tokens' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # prompt
        # if configs.prompt:
        prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prompts', f'{configs.data}.txt')
        with open(prompt_file, 'r') as f:
            prompt_text = f.read()
        # tokenize the text to get input token IDs
        inputs = self.tokenizer(prompt_text, return_tensors='pt')
        input_ids = inputs["input_ids"]
        self.prompt_embeddings = self.gpt2_text.wte(input_ids).to(device)
        print("prompt_embeddings shape:", self.prompt_embeddings.shape)


        self.time_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)])

        self.text_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)])

        # self.time_proj = nn.ModuleList(
        #     [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in
        #      range(self.llama_config.num_hidden_layers + 1)])
        #
        # self.text_proj = nn.ModuleList(
        #     [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in
        #      range(self.llama_config.num_hidden_layers + 1)])

        print("config.dim: ", configs.dim) # 41
        self.in_layer = Encoder_PCA(configs.dim, self.word_embeddings, hidden_dim=configs.d_model,
                                    num_heads=configs.num_ca_heads, device=device,
                                    num_ref_points=configs.num_ref_points, latent_dim=configs.latent_dim,
                                    learn_emb=configs.learn_emb, patch_len=configs.patch_len, stride=configs.stride,
                                    num_ca_heads=configs.num_ca_heads, prompt_embeddings=self.prompt_embeddings)

        # self.attention_pooling = AttentionPooling(configs.d_model)
        # self.gru = nn.GRU(configs.d_model, configs.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification':
            # print("configs.d_model * configs.enc_in: ", configs.d_model * configs.enc_in)
            # self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
            self.num_patches = math.ceil((configs.num_ref_points - configs.patch_len) / configs.stride) + 1
            self.out_layer = nn.Linear(configs.d_model * self.num_patches * configs.dim, configs.num_class)
            # print("configs.d_model * configs.enc_in: ", configs.d_model * configs.enc_in) # 768 * 7 = 5376
            # encoder-only classification head
            # self.encoder_only_head = nn.Linear(configs.d_model * self.num_patches, configs.num_class)
            # self.alpha = 1.0 # weight for LLM classification loss
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()

        # for layer in (
        #         self.llama_text, self.llama, self.in_layer, self.out_layer, self.time_proj, self.text_proj
        # ):
        #     layer.to(device=device)
        #     layer.train()

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
        print("x shape", x.shape)
        print("time steps shape", time_steps.shape)

        # treat each channel as separate sequence
        n_vars = M // 2
        values = x[:, :, :n_vars]
        masks = x[:, :, n_vars:2*n_vars]
        time_steps = time_steps.unsqueeze(1).expand(-1, n_vars, -1).reshape(B * n_vars, L)
        print("time steps shape", time_steps.shape)

        x = torch.stack((values, masks), dim=-1) # (B, T, n_vars, 2)
        print("x shape", x.shape)

        x = x.permute(0, 2, 1, 3).reshape(B * n_vars, L, 2)
        print("x shape", x.shape)
        print("fisrt x:", x[:2, :, :1])
        print("second x:", x[:2, :, 1:])

        # print("x shape: ", x.shape) # (256, 2500, 2)

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x, time_steps)
        print("outputs_time1 shape: ", outputs_time1.shape) # (128, 128, 768)
        print("outputs_text1 shape: ", outputs_text1.shape) # (128, 128, 768)

        # encoder-only classifier head
        # encoder_output = outputs_time1.reshape(B, -1)
        # encoder_only_logits = self.encoder_only_head(encoder_output)  # shape [B, num_class]

        # add prompt
        # batch_prompt_embeddings = self.prompt_embeddings.repeat(B, 1, 1)
        # print("batch_prompt_embeddings shape", batch_prompt_embeddings.shape)
        # outputs_time1 = torch.cat([batch_prompt_embeddings, outputs_time1], dim=1)
        # outputs_text1 = torch.cat([batch_prompt_embeddings, outputs_text1], dim=1)
        # print("outputs_time1 shape: ", outputs_time1.shape)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        # llama_output_time = self.llama(inputs_embeds=outputs_time1, output_hidden_states=True)
        # outputs_time = llama_output_time.last_hidden_state
        # llama_output_text = self.llama_text(inputs_embeds=outputs_text1, output_hidden_states=True)
        # outputs_text = llama_output_text.last_hidden_state

        print("outputs_time shape: ", outputs_time.shape) # (128, 128, 768)
        print("outputs_text shape: ", outputs_text.shape) # (128, 128, 768)

        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        # intermidiate_feat_time = tuple(
        #     self.time_proj[idx](feat) for idx, feat in enumerate(llama_output_time.hidden_states)
        # )
        # intermidiate_feat_text = tuple(
        #     self.text_proj[idx](feat) for idx, feat in enumerate(llama_output_text.hidden_states)
        # )

        print("outputs_time shape: ", outputs_time.shape) # (128, 128, 768)
        print("outputs_text shape: ", outputs_text.shape) # (128, 128, 768)
        
        outputs_time = outputs_time.reshape(B, -1)
        outputs_text = outputs_text.reshape(B, -1)

        print("outputs_time shape: ", outputs_time.shape) # 128, 98304
        print("outputs_text shape: ", outputs_text.shape) # 128, 98304
        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
            # 'encoder_only': encoder_only_logits,
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
        if self.task_name == 'classification':
            output = self.classification(x, time_steps)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output
