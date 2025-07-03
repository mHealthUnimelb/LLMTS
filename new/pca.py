# import torch
#
# from sklearn.decomposition import PCA
#
# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
#
# model = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
#
# wte = model.wte.state_dict()['weight'].cpu().numpy()
#
# pca = PCA(n_components=768)
#
# wte_pca = pca.fit_transform(wte.T)
#
# torch.save(wte_pca, "wte_pca_768.pt")


import torch
from sklearn.decomposition import PCA
from transformers import LlamaConfig, LlamaModel  # Import LlamaModel instead of GPT2Model

# Load the Llama3 model
llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
llama_config.output_attentions = True
llama_config.output_hidden_states = True

try:
    model = LlamaModel.from_pretrained('huggyllama/llama-7b', trust_remote_code=True, local_files_only=True,
                                       config=llama_config)
except EnvironmentError:
    model = LlamaModel.from_pretrained('huggyllama/llama-7b', trust_remote_code=True, local_files_only=False,
                                       config=llama_config)

# Access the embedding weights; for Llama models, use 'embed_tokens' instead of 'wte'
wte = model.embed_tokens.state_dict()['weight'].cpu().numpy()
print("wte shape: ", wte.shape)

# Perform PCA on the embedding weights
pca = PCA(n_components=1000)
wte_pca = pca.fit_transform(wte.T)

# Save the PCA-transformed embeddings
torch.save(wte_pca, "llama_wte_pca_1000.pt")
