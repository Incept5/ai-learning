# NLP and LLM Learning Repository

This repository contains a collection of demos and examples for learning about Natural Language Processing (NLP) and Large Language Models (LLMs). It includes examples of tokenization, embeddings, sentiment analysis, and more.

## Repository Structure

The repository is organized into the following main sections:

### Tokenization

The `tokenization` directory contains demos and examples related to tokenization, which is the process of breaking text into tokens that can be processed by language models.

- [TOKENIZATION_MODELS.md](tokenization/TOKENIZATION_MODELS.md) - Guide to different tokenization models
- [llm-tokenization-demo.py](tokenization/llm-tokenization-demo.py) - Comprehensive demo comparing GPT-2 and BERT tokenizers
- [simple_tokenization.py](tokenization/simple_tokenization.py) - Basic tokenization using BERT tokenizer
- [simple_tokenization_02.py](tokenization/simple_tokenization_02.py) - Comparison between GPT-2 and OPT tokenizers
- [tiktoken_demo.py](tokenization/tiktoken_demo.py) - Demo of OpenAI's tiktoken library
- [sentencepiece_demo.py](tokenization/sentencepiece_demo.py) - Demo of Google's SentencePiece tokenizer
- [tokenization_comparison.py](tokenization/tokenization_comparison.py) - Comparison of different tokenization approaches

### Embeddings

The `embeddings` directory contains demos and examples related to text embeddings, which are vector representations of text that capture semantic meaning.

- [EMBEDDING_MODELS.md](embeddings/EMBEDDING_MODELS.md) - Guide to different embedding models
- [embeddings_demo.py](embeddings/embeddings_demo.py) - Demo using SentenceTransformers models
- [huggingface_embeddings_demo.py](embeddings/huggingface_embeddings_demo.py) - Demo using Hugging Face Transformers models
- [openai_embeddings_demo.py](embeddings/openai_embeddings_demo.py) - Demo using OpenAI's embedding models
- [ollama_embedding.py](embeddings/ollama_embedding.py) - Demo using Ollama's local embedding models

### Other Demos

- [gp2_demo.py](gp2_demo.py) - Demo showcasing GPT-2 limitations and hallucinations
- [sentiment_01.py](sentiment_01.py) - Simple sentiment analysis using a pre-trained model

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com:Incept5/ai-learning.git
   cd ai-learning
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (for API-based demos):
   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit the .env file with your API keys
   # Get your OpenAI API key from: https://platform.openai.com/account/api-keys
   # Get your Hugging Face token from: https://huggingface.co/settings/tokens
   ```

## Running the Demos

### Tokenization Demos

```bash
# Basic tokenization demo
python tokenization/simple_tokenization.py

# Comprehensive tokenization comparison
python tokenization/llm-tokenization-demo.py

# OpenAI's tiktoken demo
python tokenization/tiktoken_demo.py

# SentencePiece demo
python tokenization/sentencepiece_demo.py
```

### Embedding Demos

```bash
# List available SentenceTransformers models
python embeddings/embeddings_demo.py --list-models

# Run embedding demo with a specific model
python embeddings/embeddings_demo.py --model all-mpnet-base-v2

# OpenAI embeddings demo (requires API key)
python embeddings/openai_embeddings_demo.py

# Hugging Face embeddings demo
python embeddings/huggingface_embeddings_demo.py

# Ollama embeddings demo (requires Ollama installation)
python embeddings/ollama_embedding.py
```

### Other Demos

```bash
# GPT-2 limitations demo
python gp2_demo.py

# Sentiment analysis demo
python sentiment_01.py
```

## Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Ollama Documentation](https://ollama.com/docs)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.