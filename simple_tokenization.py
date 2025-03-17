from transformers import AutoTokenizer


# Loading the Tokenizer: The script uses the AutoTokenizer class to load a pre-trained tokenizer. In this case,
# we're using the tokenizer for the bert-base-uncased model, which is a popular choice for many NLP tasks.
#
# Tokenizing Text: The tokenize method breaks down the input text into tokens. Tokens are the basic units that the
# model will process.
#
# Converting Tokens to IDs: The convert_tokens_to_ids method converts the tokens into their corresponding IDs,
# which are numerical representations that the model can understand.
#
# Output: The script prints the tokens and their IDs, demonstrating how text is converted into a format suitable for
# input into an LLM.

# To run this script, you need to have the transformers library installed. You can install it using pip:
# pip install transformers

# Load a pre-trained tokenizer from the Hugging Face model hub
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sample text to tokenize
text = "Tokenization is the first step in processing text for LLMs."

# Tokenize the text
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Print the tokens and their corresponding IDs
print("Tokens:", tokens)
print("Token IDs:", token_ids)