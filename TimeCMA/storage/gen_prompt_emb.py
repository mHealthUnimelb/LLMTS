import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path = 'FRED',
        model_name = "gpt2",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train'
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len =  input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len-1
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)

    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        # Time series value
        print("in_data", in_data[i, :, j])
        values = in_data[i, :, j].flatten().tolist() # Selects the entire sequence (:) of feature j in sample i
        values_str = ", ".join([str(int(value)) for value in values])

        # Last token
        # torch.diff(...) Computes differences between consecutive time steps.
        # torch.sum(...) Totals those differences to get an overall “trend.”
        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        trends_str = f"{trends.item():0f}" # Extracts the Python scalar from the tensor, formatted as an integer string
        
        # Date
        if self.data_path in ['FRED', 'ILI']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:00"
        elif self.data_path in ['ECG']:
            print("in_data_mark shape", in_data_mark.shape)
            print("in_data_mark", in_data_mark)
            start_date = f"{in_data_mark[0][0]}"
            end_date = f"{in_data_mark[0][-1]}"
        else: # ETTm1, ETTm2, Weather
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:{int(in_data_mark[i,0,5]):02d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:{int(in_data_mark[i,self.len,5]):02d}"

        # Prompt
        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)
        print("in_prompt: ", in_prompt)

        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def generate_embeddings(self, in_data, in_data_mark):
            input_templates = {
                'FRED': "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
                'ILI': "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
                'ETTh1': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTh2': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ECL': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTm1': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'ETTm2': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'Weather': "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends",
                'ECG': "From [t1] to [t2], the values were value1, ..., valuen every 4 milliseconds. The total trend value was Trends",
            }

            input_template = input_templates.get(self.data_path, input_templates['FRED'])
            
            tokenized_prompts = []
            max_token_count = 0
            for i in range(len(in_data)):
                # sample i
                for j in range(in_data.shape[2]):
                    # feature j
                    tokenized_prompt = self._prepare_prompt(input_template, in_data, in_data_mark, i, j).to(self.device)
                    max_token_count = max(max_token_count, tokenized_prompt.shape[1]) # Tracks the maximum sequence length
                    tokenized_prompts.append((i, tokenized_prompt.to(self.device), j)) # Stores (sample_index, token_ids, feature_index) in a list

            # Pre‐allocates a zero tensor to hold all embeddings, with shape [batch_size, max_seq_len, d_model, num_features]
            in_prompt_emb = torch.zeros((len(in_data), max_token_count, self.d_model, in_data.shape[2]), dtype=torch.float32, device=self.device)

            for i, tokenized_prompt, j in tokenized_prompts:
                prompt_embeddings = self.forward(tokenized_prompt) # [1, seq, d_model]
                padding_length = max_token_count - tokenized_prompt.shape[1]
                if padding_length > 0:
                    last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
                    # repeat the last token’s embedding to pad
                    padding = last_token_embedding.repeat(1, padding_length, 1)
                    prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1) # [1, max_seq, d_model]
                else:
                    prompt_embeddings_padded = prompt_embeddings
                        
                in_prompt_emb[i, :max_token_count, :, j] = prompt_embeddings_padded
                # Slicing out the final time‐step dimension (max_token_count-1) for all samples and features
                last_token_emb = in_prompt_emb[:, max_token_count-1:max_token_count, :, :]
                last_token_emb = last_token_emb.squeeze() # [batch_size, d_model, num_features]

            return last_token_emb