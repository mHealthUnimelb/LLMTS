import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = configs.gpu
        print(self.device)
        
        self.gpt2 = GPT2Model.from_pretrained(
            configs.llm_ckp_dir,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_ckp_dir)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.vocab_size = self.gpt2_tokenizer.vocab_size
        self.hidden_dim_of_llama = 768
        
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

    # def tokenizer(self, x):
    #     output = self.gpt2_tokenizer(x, return_tensors="pt")['input_ids'].to(self.device)
    #     result = self.gpt2.get_input_embeddings()(output)
    #     return result   
    
    def forecast(self, x_mark_enc):        
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        # print("x_mark_enc shape:", x_mark_enc.shape)
        # x_mark_enc = torch.cat([self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))], 0)
        batch = self.gpt2_tokenizer(x_mark_enc, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # print("batch shape", batch.shape)
        x_mark_enc = self.gpt2.get_input_embeddings()(batch["input_ids"])
        text_outputs = self.gpt2(inputs_embeds=x_mark_enc)[0]
        text_outputs = text_outputs[:, -1, :]
        print("text_outputs len:", len(text_outputs))
        return text_outputs
    
    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)