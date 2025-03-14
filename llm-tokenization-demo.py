import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import numpy as np

def demonstrate_tokenization():
    """
    A function to demonstrate how tokenization works in LLMs with
    visualizations and examples using both GPT tokenizers and BERT tokenizers.
    """
    print("Demonstrating Tokenization in Large Language Models")
    print("="*50)
    
    # Example text to tokenize
    example_texts = [
        "Hello, world!",
        "Neural networks are transforming AI.",
        "GPT models use byte-pair encoding for tokenization.",
        "こんにちは世界", # Hello world in Japanese
        "LLMs like Claude, GPT-4, and BERT all use different tokenization approaches."
    ]
    
    # ==================== GPT-style Tokenization ====================
    print("\n1. GPT-Style Tokenization (tiktoken)")
    print("-"*50)
    
    # Initialize GPT tokenizer (cl100k_base is used by many recent OpenAI models)
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Show tokenization results for our example texts
    for i, text in enumerate(example_texts):
        tokens = gpt_tokenizer.encode(text)
        decoded = [gpt_tokenizer.decode([token]) for token in tokens]
        
        print(f"\nExample {i+1}: '{text}'")
        print(f"Token IDs: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Decoded tokens: {decoded}")
    
    # ==================== BERT-style Tokenization ====================
    print("\n\n2. BERT-Style Tokenization (transformers)")
    print("-"*50)
    
    # Initialize BERT tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    for i, text in enumerate(example_texts):
        encoded = bert_tokenizer(text)
        tokens = bert_tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        
        print(f"\nExample {i+1}: '{text}'")
        print(f"Token IDs: {encoded['input_ids']}")
        print(f"Token count: {len(encoded['input_ids'])}")
        print(f"Decoded tokens: {tokens}")
    
    # ==================== Visual Comparison ====================
    print("\n\n3. Visual Comparison of Tokenizers")
    print("-"*50)
    
    # Prepare data for visualization
    token_counts = {
        'Text': [f"Example {i+1}" for i in range(len(example_texts))],
        'GPT': [len(gpt_tokenizer.encode(text)) for text in example_texts],
        'BERT': [len(bert_tokenizer(text)["input_ids"]) for text in example_texts]
    }
    
    token_df = pd.DataFrame(token_counts)
    
    # Create a bar chart comparing token counts
    plt.figure(figsize=(10, 6))
    x = np.arange(len(token_df['Text']))
    width = 0.35
    
    plt.bar(x - width/2, token_df['GPT'], width, label='GPT')
    plt.bar(x + width/2, token_df['BERT'], width, label='BERT')
    
    plt.xlabel('Examples')
    plt.ylabel('Token Count')
    plt.title('Token Count Comparison: GPT vs BERT')
    plt.xticks(x, token_df['Text'])
    plt.legend()
    
    plt.savefig('tokenization_comparison.png')
    print("Visualization saved as 'tokenization_comparison.png'")
    
    # ==================== Special Cases ====================
    print("\n\n4. Special Cases and Edge Examples")
    print("-"*50)
    
    special_cases = [
        "1234567890",                    # Numbers
        "https://www.example.com",       # URLs
        "She said, \"Hello!\"",          # Quotes
        "Python is easy-to-learn",       # Hyphens
        "🙂👍🔥"                         # Emojis
    ]
    
    print("\nGPT Tokenization of Special Cases:")
    for case in special_cases:
        gpt_tokens = gpt_tokenizer.encode(case)
        print(f"'{case}' → {len(gpt_tokens)} tokens: {[gpt_tokenizer.decode([t]) for t in gpt_tokens]}")
    
    print("\nBERT Tokenization of Special Cases:")
    for case in special_cases:
        bert_encoded = bert_tokenizer(case)
        bert_tokens = bert_tokenizer.convert_ids_to_tokens(bert_encoded["input_ids"])
        print(f"'{case}' → {len(bert_tokens)} tokens: {bert_tokens}")

if __name__ == "__main__":
    demonstrate_tokenization()
