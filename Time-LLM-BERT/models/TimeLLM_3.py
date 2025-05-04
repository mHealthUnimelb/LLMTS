from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    # dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:]) #
    # print("dec_out shape: ", dec_out.shape) # (24, 1, 96)

    # self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
    #                                      head_dropout=configs.dropout)
    # head_nf = 8192

    def forward(self, x):
        print("x before flatten shape:", x.shape) # (24, 1, 128, 64)
        x = self.flatten(x)
        print("x after flatten shape:", x.shape) # (24, 1, 8192)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ClassificationHead(nn.Module):
    # def __init__(self, n_vars, input_dim, num_classes):
    #     super(ClassificationHead, self).__init__()
    #     self.n_vars = n_vars
    #     self.linear = nn.Linear(input_dim, num_classes)  # Maps to the number of classes

    def __init__(self, n_vars, nf, num_classes, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        # 48, 768
        print("input_dim shape: ", nf)
        print("num_classes: ", num_classes)
        # self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, num_classes)  # Mean: Maps to the number of classes
        # self.linear = nn.Linear(nf * n_vars, num_classes)  # Concatenation: Maps to the number of classes
        print("nf * n_vars: ", nf * n_vars)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # print("x before flatten shape:", x.shape) # (24, 2, 128, 1)  (24, 2, 128, 312)  (24, 128, 312)
        # x = self.flatten(x)
        print("x after flatten shape:", x.shape) # (24, 2, 128)  (24, 2, 39936)  (24, 39936)
        x = self.linear(x)  # Return logits for each class
        print("x after linear: ", x.shape)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff   # 128
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        # self.num_classes = configs.num_classes
        print("patch_len: ", self.patch_len) # 16
        print("stride: ", self.stride) # 8

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'This database includes 25 long-term ECG recordings of human subjects with atrial fibrillation (mostly paroxysmal). Of these, 23 records include the two ECG signals.'
        print("description", self.description)
        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        # self.project_patch_embedding_layer = nn.Linear(configs.d_model, configs.llm_dim)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        # self.patch_nums = configs.total_length // configs.seq_len
        self.head_nf = self.d_ff * self.patch_nums
        print("d_ff shape: ", self.d_ff) # 128
        print("patches_num: ", self.patch_nums) # 312   # focast: 64
        print("head_nf shape: ", self.head_nf) # 39936    # forecast: 8192

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.output_projection = ClassificationHead(configs.enc_in, (self.patch_nums + ) * self.self.d_llm, num_classes=configs.num_classes, head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        # self.conv2d_layer = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            print("x_enc shape: ", x_enc.shape)
            class_out = self.classification(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return class_out
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        print("x_enc shape: ", x_enc.shape) # (24, 512, 1)
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        print("x_enc shape: ", x_enc.shape) # (24, 512, 1)

        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        print("x_enc shape: ", x_enc.shape) # (24, 512, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        print("x_enc shape: ", x_enc.shape) # (24, 512, 1)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        print("x_enc shape: ", x_enc.shape) # (24, 512, 1)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        print("prompt_embeddings shape: ", prompt_embeddings.shape) # (24, 143, 768)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        print("x_enc shape: ", x_enc.shape) # (24, 1, 512)

        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        print("enc_out shape: ", enc_out.shape) # (24, 64, 32)
        print("n_vars shape: ", n_vars) # 1

        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        print("enc_out shape: ", enc_out.shape) # (24, 64, 768)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        print("llama_enc_out shape: ", llama_enc_out.shape) # (24, 207, 768)

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        print("dec_out shape: ", dec_out.shape) # (24, 207, 768)

        dec_out = dec_out[:, :, :self.d_ff]
        print("dec_out shape: ", dec_out.shape) # (24, 207, 128)

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        print("dec_out shape: ", dec_out.shape)

        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        print("dec_out shape: ", dec_out.shape) # (24, 1, 128, 207)

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:]) #
        print("dec_out shape: ", dec_out.shape) # (24, 1, 96)

        dec_out = dec_out.permute(0, 2, 1).contiguous()
        print("dec_out shape: ", dec_out.shape) # (24, 96, 1)

        dec_out = self.normalize_layers(dec_out, 'denorm')
        print("dec_out shape: ", dec_out.shape) # (24, 96, 1)

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


    def classification(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')
        print("x_enc shape: ", x_enc.shape)   # (24, 2500, 2)

        B, T, N = x_enc.size() # B: batch size, T: sequence length, N: num features
        print(B, T, N)
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        print("x_enc shape after permute: ", x_enc.shape)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: classify the sequence; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        print("x_enc shape: ", x_enc.shape) # (24, 2, 2500)

        time_enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        print("n_vars: ", n_vars) # 2
        print("enc_out shape: ", time_enc_out.shape) # (48, 312, 32)

        aligned_enc_out = self.reprogramming_layer(time_enc_out, source_embeddings, source_embeddings)
        print("enc_out shape: ", aligned_enc_out.shape) # (48, 312, 768)

        # time_enc_out = self.project_patch_embedding_layer(time_enc_out)
        # print("enc_out shape: ", time_enc_out.shape)

        aligned_llama_enc_out = torch.cat([prompt_embeddings, aligned_enc_out], dim=1)
        # time_llama_enc_out = torch.cat([prompt_embeddings, time_enc_out], dim=1)
        print("prompt_embeddings shape: ", prompt_embeddings.shape) # (48, 137, 768)    text embedding   (48, 136, 768)
        print("llama_enc_out shape: ", aligned_llama_enc_out.shape) # (48, 449, 768)    time series + text       (48, 448, 768)

        aligned_class_out = self.llm_model(inputs_embeds=aligned_llama_enc_out).last_hidden_state
        # time_class_out = self.llm_model(inputs_embeds=time_llama_enc_out).last_hidden_state
        print("class_out shape: ", aligned_class_out.shape) # (48, 448, 768)  (batch_size, seq_len, embedding)

        # aligned_class_out = aligned_class_out[:, :, :self.d_ff]
        # # time_class_out = time_class_out[:, :, :self.d_ff]
        # print("class_out shape: ", aligned_class_out.shape) # (48, 448, 128)
        #
        # aligned_class_out = torch.reshape(
        #     aligned_class_out, (-1, n_vars, aligned_class_out.shape[-2], aligned_class_out.shape[-1]))  # (batch_size, num_channels, seq_len, embedding)
        # # time_class_out = torch.reshape(
        # #     time_class_out, (-1, n_vars, time_class_out.shape[-2], time_class_out.shape[-1]))
        # aligned_class_out = aligned_class_out.permute(0, 1, 3, 2).contiguous()
        # # time_class_out = time_class_out.permute(0, 1, 3, 2).contiguous()
        # print("class_out shape: ", aligned_class_out.shape) # (24, 2, 128, 448)  (batch_size, num_channels, embedding, seq_len)
        #
        # aligned_class_out = aligned_class_out[:, :, :, -self.patch_nums:]
        # # time_class_out = time_class_out[:, :, :, -self.patch_nums:]
        # print("dec_out shape: ", aligned_class_out.shape) # (24, 2, 128, 312)   (batch_size, num_channels, embedding, seq_len)
        #
        # # option 1: mean
        # aligned_class_out = torch.mean(aligned_class_out, dim=1)
        # # time_class_out = torch.mean(time_class_out, dim=1)
        # print("dec_out shape: ", aligned_class_out.shape) # (24, 128, 312)
        #
        # # option 2: concatenation
        # # class_out = torch.cat([class_out[:, 0, :, :], class_out[:, 1, :, :]],
        # #                       dim=2)  # Concatenate along sequence length dimension
        # # print("class_out after concatenation: ", class_out.shape)  # (24, 128, 624)
        #
        # # # option 3: max
        # # class_out, _ = torch.max(class_out, dim=1)
        # # print("class_out after concatenation: ", class_out.shape)
        #
        # aligned_class_out = aligned_class_out.permute(0, 2, 1).contiguous()   # (24, 624, 128)
        # # time_class_out = time_class_out.permute(0, 2, 1).contiguous()
        # print("dec_out shape: ", aligned_class_out.shape)

        aligned_class_out = aligned_class_out.reshape(B, -1)
        aligned_logits = self.output_projection(aligned_class_out)
        # time_logits = self.output_projection(time_class_out)
        # logits = class_out[:, :, :, -1]
        print("class_out before projection: ", aligned_logits.shape) # (24, 2, 4)   (24, 4)

        # logits = self.output_projection(logits)
        # print("class_out after projection: ", logits.shape)

        # logits = logits.permute(0, 2, 1).contiguous()
        # print("class_out after projection: ", logits.shape) # (24, 4, 2)

        # logits = logits.mean(dim=-1)
        # print("logits shape: ", logits.shape)

        # logits = {
        #     "aligned_logits": aligned_logits,
        #     "time_logits": time_logits,
        # }
        return aligned_logits


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1) # time
        source_embedding = self.key_projection(source_embedding).view(S, H, -1) # text
        value_embedding = self.value_projection(value_embedding).view(S, H, -1) # text

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
