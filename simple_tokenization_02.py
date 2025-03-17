from transformers import AutoTokenizer
import os

# Load tokenizers for models similar to ChatGPT and Llama
tokenizer_chatgpt = AutoTokenizer.from_pretrained("gpt2")  # Using GPT-2 as a proxy for ChatGPT

# To use Llama 2, you need to:
# 1. Get your token from https://huggingface.co/settings/tokens
# 2. Set it as an environment variable:
#    export HUGGING_FACE_TOKEN=your_token_here
# 3. Accept the terms at: https://huggingface.co/meta-llama/Llama-2-7b

token = os.getenv('HUGGING_FACE_TOKEN')
tokenizer_llama = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b", 
    token=token
)  # Using official Llama 2 model

# Sample text to tokenize
text = "Tokenization can vary significantly between different models."

# Tokenize the text using both tokenizers
tokens_chatgpt = tokenizer_chatgpt.tokenize(text)
token_ids_chatgpt = tokenizer_chatgpt.convert_tokens_to_ids(tokens_chatgpt)

tokens_llama = tokenizer_llama.tokenize(text)
token_ids_llama = tokenizer_llama.convert_tokens_to_ids(tokens_llama)

# Print the tokens and their corresponding IDs for both tokenizers
print("ChatGPT-like Tokenizer:")
print("Tokens:", tokens_chatgpt)
print("Token IDs:", token_ids_chatgpt)

print("\nLlama-like Tokenizer:")
print("Tokens:", tokens_llama)
print("Token IDs:", token_ids_llama)