import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from models.GPT2_arch import AccustumGPT2Model


class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        self.word_embedding = word_embedding.T

    def forward(self, x):
        print("x shape: ", x.shape) # (128, 2, 2500)
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)

        x = self.linear(x)
        print("x shape: ", x.shape) # (128, 2, 768)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        print("x shape: ", x.shape) # (128, 2, 768)

        x_time = x
        print("x_time shape: ", x_time.shape) # (128, 2, 768)

        q = x.transpose(0, 1)
        print("q shape: ", q.shape) # (2, 128, 768)
        k = v = self.word_embedding.transpose(0, 1)
        print("k shape: ", k.shape) # (18, 128, 768)
        x, w_ = self.cross_attention(q, k, v)
        print("weights shape: ", w_.shape) # (128, 2, 18)

        x = x.transpose(0, 1)
        print("x shape: ", x.shape) # (128, 2, 768)

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
        print("word_embedding_path: ", configs.word_embedding_path)

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

        self.in_layer = Encoder_PCA(configs.seq_len, word_embedding, hidden_dim=configs.d_model)

        if configs.classify_pertp:
            self.projection_layer = nn.Linear(configs.d_model, configs.seq_len)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification':
            if configs.classify_pertp:
                self.out_layer = nn.Linear(configs.enc_in, configs.num_class)
            else:
                self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
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

    def classification(self, x):
        B, L, M = x.shape

        # print("x shape: ", x.shape) # (256, 2500, 2)

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)
        print("outputs_time1 shape: ", outputs_time1.shape) # (128, 2, 768)
        print("outputs_text1 shape: ", outputs_text1.shape) # (128, 2, 768)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        print("outputs_time shape: ", outputs_time.shape) # (128, 2, 768)
        print("outputs_text shape: ", outputs_text.shape) # (128, 2, 768)

        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        print("outputs_time shape: ", outputs_time.shape) # (128, 2, 768)
        print("outputs_text shape: ", outputs_text.shape) # (128, 2, 768)

        # if self.configs.classify_pertp:
        #     outputs_time = self.projection_layer(outputs_time) # (batch, channel, seq_len)
        #     outputs_text = self.projection_layer(outputs_text)

        #     outputs_time = outputs_time.permute(0, 2, 1) # (batch, seq_len, channel)
        #     outputs_text = outputs_text.permute(0, 2, 1)
        # else:
        outputs_time = outputs_time.reshape(B, -1)
        outputs_text = outputs_text.reshape(B, -1)

        print("outputs_time shape: ", outputs_time.shape) # (128, 1536)
        print("outputs_text shape: ", outputs_text.shape) # (128, 1536)
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

    def forward(self, x, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        if self.task_name == 'classification':
            output = self.classification(x)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output