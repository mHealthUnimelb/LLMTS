# import torch
# from transformers import GPT2Tokenizer, GPT2Model
#
# def save_word_embeddings(file_path='word_embeddings.pt'):
#     # Define the words for which we want to extract embeddings
#     words = ["Trend", "Season", "Cycling", "Growth", "Variability", "Autocorrelation", "Persistence",
#              "Pattern", "Shift", "Decomposition", "Happiness", "Harmony", "Echo", "Glimmer", "Key",
#              "Lemon", "Mountain", "Piano"]
#
#     # Load GPT-2 tokenizer and model
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     gpt2_model = GPT2Model.from_pretrained("gpt2")
#
#     # Tokenize the words and get their embeddings
#     input_ids = tokenizer(words, return_tensors="pt", padding=True, truncation=True).input_ids
#     with torch.no_grad():
#         word_embeddings = gpt2_model(input_ids).last_hidden_state[:, 0, :]  # Extract the embedding for each word
#
#     # Save the embeddings to a file
#     torch.save(word_embeddings, file_path)
#     print(f"Word embeddings saved to {file_path}")
#
# # Call the function to save the embeddings
# save_word_embeddings('word_embeddings.pt')


import torch
from transformers import GPT2Tokenizer, GPT2Model

def save_word_embeddings(file_path='word_embeddings.pt'):
    # Define the words for which we want to extract embeddings
    # words = ["Trend", "Seasonality", "Cycling", "Growth", "Variability", "Autocorrelation", "Persistence",
    #          "Pattern", "Shift", "Decomposition", "Happiness", "Harmony", "Echo", "Glimmer", "Key",
    #          "Lemon", "Mountain", "Piano"]

    words = ["Trend", "season", "down", "up", "cycle", "rise", "peak", "atility", "relation", "istence", "pattern",
             "shift", "position", "happy", "echo", "arm", "key", "mount", "regular", "missing", "heart", "ir"]

    # Load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2Model.from_pretrained("gpt2")

    # vocab = tokenizer.get_vocab()

    # Find the token IDs for the specified words
    token_ids = []
    for word in words:
        token_id = tokenizer(word)['input_ids'] # Token ID for the word

        tokens = tokenizer.convert_ids_to_tokens(token_id)
        print(f"Word: {word}, Tokens: {tokens}, Token IDs: {token_id}")

        # if word in vocab:
        #     token_id = vocab[word]
        #     token_ids.append(token_id)
        #     print(f"Word: {word}, Token ID: {token_id}")
        # else:
        #     print(f"Word: {word} not found in vocabulary")
        token_ids.append(token_id)
    # input_ids = tokenizer(words, add_special_tokens=False).input_ids  # Token IDs for each word
    # print(token_ids)
    #
    # token_id = tokenizer("Growth")['input_ids']
    # print(token_id)

    # # Extract the token embeddings from GPT-2's embedding layer
    token_embeddings = gpt2_model.get_input_embeddings()  # Access the embedding layer
    word_embeddings = token_embeddings(torch.tensor(token_ids))  # Get embeddings for the word IDs
    print(word_embeddings.shape)
    # convert word embeddings to shape (num_words, embedding_dim)
    word_embeddings = word_embeddings.squeeze(1)
    print(word_embeddings.shape)

    #
    # Save the embeddings to a file
    torch.save(word_embeddings.T, file_path)
    print(f"Word embeddings saved to {file_path}")

    # read .pt file
    word_embeddings = torch.load(file_path)
    print(word_embeddings.shape)

# Call the function to save the embeddings
save_word_embeddings('word_embeddings.pt')
