# Tokenization Models Guide

This guide provides information about different tokenization models you can use with the tokenization demos in this repository.

## What is Tokenization?

Tokenization is the process of breaking down text into smaller units called tokens. For language models, tokenization is a critical preprocessing step that converts human-readable text into a format that models can understand. Different models use different tokenization approaches:

- **Word-based**: Splits text by spaces and punctuation (simple but creates large vocabularies)
- **Character-based**: Splits text into individual characters (small vocabulary but longer sequences)
- **Subword-based**: Splits text into subword units (balance between word and character tokenization)
  - **BPE (Byte-Pair Encoding)**: Used by GPT models
  - **WordPiece**: Used by BERT models
  - **SentencePiece**: Used by T5, XLM-R models
  - **Unigram**: Used by some SentencePiece implementations

## Available Demo Scripts

This repository includes several demo scripts for different tokenization approaches:

1. **simple_tokenization.py** - Basic tokenization using BERT tokenizer
2. **simple_tokenization_02.py** - Comparison between GPT-2 and OPT tokenizers
3. **llm-tokenization-demo.py** - Comprehensive demo comparing GPT-2 and BERT tokenizers with visualizations

## Tokenization Models to Test

### GPT Family Tokenizers (BPE-based)

- **gpt2**: The standard GPT-2 tokenizer (50,257 tokens)
- **openai-gpt**: Original GPT tokenizer (40,478 tokens)
- **gpt_neo**: EleutherAI's GPT-Neo tokenizer (50,257 tokens)
- **gpt-j-6b**: EleutherAI's GPT-J tokenizer (50,257 tokens)
- **codegen**: Salesforce's CodeGen tokenizer (50,257 tokens)
- **santacoder**: BigCode's SantaCoder tokenizer (49,152 tokens)
- **llama**: Meta's LLaMA tokenizer (32,000 tokens)
- **mistral**: Mistral AI's tokenizer (32,000 tokens)

### BERT Family Tokenizers (WordPiece-based)

- **bert-base-uncased**: Standard BERT tokenizer (30,522 tokens)
- **bert-base-cased**: Case-sensitive BERT tokenizer (28,996 tokens)
- **bert-large-uncased**: Larger BERT model tokenizer (30,522 tokens)
- **distilbert-base-uncased**: DistilBERT tokenizer (30,522 tokens)
- **roberta-base**: RoBERTa tokenizer (50,265 tokens)
- **albert-base-v2**: ALBERT tokenizer (30,000 tokens)
- **deberta-v3-base**: DeBERTa v3 tokenizer (128,100 tokens)

### T5 Family Tokenizers (SentencePiece-based)

- **t5-small**: T5 small model tokenizer (32,128 tokens)
- **t5-base**: T5 base model tokenizer (32,128 tokens)
- **t5-large**: T5 large model tokenizer (32,128 tokens)
- **flan-t5-base**: Flan-T5 tokenizer (32,128 tokens)
- **mt5-small**: Multilingual T5 tokenizer (250,112 tokens)

### Multilingual Tokenizers

- **xlm-roberta-base**: XLM-RoBERTa tokenizer (250,002 tokens)
- **mbart-large-50**: MBART tokenizer (250,054 tokens)
- **nllb-200-distilled-600M**: NLLB tokenizer (256,102 tokens)
- **m2m100_418M**: M2M100 tokenizer (128,112 tokens)
- **bert-base-multilingual-cased**: Multilingual BERT tokenizer (119,547 tokens)

### Code-Specific Tokenizers

- **codellama/CodeLlama-7b-hf**: Code Llama tokenizer (32,016 tokens)
- **Salesforce/codegen-350M-mono**: CodeGen tokenizer (50,257 tokens)
- **bigcode/santacoder**: SantaCoder tokenizer (49,152 tokens)
- **microsoft/codebert-base**: CodeBERT tokenizer (50,265 tokens)
- **microsoft/graphcodebert-base**: GraphCodeBERT tokenizer (50,265 tokens)

### Domain-Specific Tokenizers

- **dmis-lab/biobert-v1.1**: BioBERT for biomedical text (28,996 tokens)
- **allenai/scibert_scivocab_uncased**: SciBERT for scientific text (31,090 tokens)
- **nlpaueb/legal-bert-base-uncased**: Legal-BERT for legal text (30,522 tokens)
- **emilyalsentzer/Bio_ClinicalBERT**: Clinical BERT for medical text (28,996 tokens)
- **microsoft/biogpt**: BioGPT tokenizer (42,384 tokens)

## Usage Examples

### Basic Usage

```python
from transformers import AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
text = "Tokenization is the first step in processing text for LLMs."
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
```

### Comparing Multiple Tokenizers

```python
from transformers import AutoTokenizer

# Load different tokenizers
tokenizers = {
    "GPT-2": AutoTokenizer.from_pretrained("gpt2"),
    "BERT": AutoTokenizer.from_pretrained("bert-base-uncased"),
    "T5": AutoTokenizer.from_pretrained("t5-base"),
    "XLM-R": AutoTokenizer.from_pretrained("xlm-roberta-base")
}

# Sample text
text = "Tokenization varies significantly between different models."

# Compare tokenization results
for name, tokenizer in tokenizers.items():
    tokens = tokenizer.tokenize(text)
    print(f"\n{name} Tokenizer:")
    print(f"Token count: {len(tokens)}")
    print(f"Tokens: {tokens}")
```

## Creating a New Demo Script

Here's a template for creating a new tokenization demo script:

```python
from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Tokenization comparison demo")
    parser.add_argument("--models", nargs="+", default=["gpt2", "bert-base-uncased", "t5-base"],
                        help="List of models to compare")
    parser.add_argument("--text", type=str, default="This is a sample text to tokenize.",
                        help="Text to tokenize")
    args = parser.parse_args()
    
    # Load tokenizers
    tokenizers = {}
    for model in args.models:
        try:
            tokenizers[model] = AutoTokenizer.from_pretrained(model)
            print(f"Loaded tokenizer for {model}")
        except Exception as e:
            print(f"Failed to load tokenizer for {model}: {e}")
    
    # Tokenize text with each tokenizer
    results = {}
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(args.text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        results[name] = {
            "tokens": tokens,
            "token_ids": token_ids,
            "count": len(tokens)
        }
        
        print(f"\n{name} Tokenization:")
        print(f"Token count: {len(tokens)}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
    
    # Create visualization
    if len(results) > 1:
        counts = [data["count"] for data in results.values()]
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), counts)
        plt.title("Token Count Comparison")
        plt.xlabel("Tokenizer")
        plt.ylabel("Number of Tokens")
        plt.savefig("tokenization_comparison.png")
        print("\nVisualization saved as 'tokenization_comparison.png'")

if __name__ == "__main__":
    main()
```

## Tokenization Algorithms Explained

### Byte-Pair Encoding (BPE)

BPE is a subword tokenization algorithm used by GPT models:

1. Start with a vocabulary of individual characters
2. Identify the most frequent adjacent byte pairs
3. Merge these pairs to form new tokens
4. Repeat until desired vocabulary size is reached

BPE handles out-of-vocabulary words well by breaking them into subword units.

### WordPiece

WordPiece is used by BERT and similar models:

1. Start with a vocabulary of individual characters
2. Find pair of tokens that maximizes likelihood of training data when merged
3. Add this merged token to vocabulary
4. Repeat until desired vocabulary size is reached

WordPiece uses a different scoring mechanism than BPE and marks subwords with ## prefix.

### SentencePiece

SentencePiece treats the input as a raw character sequence:

1. No pre-tokenization (treats spaces as characters)
2. Language-agnostic (works well for non-space-separated languages)
3. Can use either BPE or Unigram algorithm underneath
4. Reversible tokenization (can perfectly reconstruct original text)

### Unigram Language Model

Used by some SentencePiece implementations:

1. Start with a large vocabulary
2. Calculate loss when removing each token
3. Remove tokens with smallest loss
4. Repeat until desired vocabulary size is reached

## Comparing Tokenization Approaches

Different tokenization approaches have different strengths:

- **Vocabulary size**: Larger vocabularies can represent more words directly but require more parameters
- **Handling of rare words**: Subword tokenization helps with rare or unseen words
- **Language support**: Some tokenizers are better for specific languages or multilingual use
- **Special tokens**: Different models use different special tokens (e.g., [CLS], etc.)
- **Case sensitivity**: Some tokenizers are case-sensitive, others lowercase all text

## Choosing the Right Tokenizer

Consider these factors when choosing a tokenizer:

1. **Model compatibility**: Use the tokenizer designed for your model
2. **Task requirements**: Some tokenizers work better for specific tasks
3. **Language requirements**: Consider multilingual needs
4. **Domain specificity**: Some tokenizers are optimized for specific domains (code, biomedical, etc.)
5. **Vocabulary size**: Larger vocabularies may represent text more efficiently but increase model size

## Additional Resources

- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
- [SentencePiece Documentation](https://github.com/google/sentencepiece)
- [Understanding Tokenization in NLP](https://huggingface.co/learn/nlp-course/chapter2/4)
- [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)