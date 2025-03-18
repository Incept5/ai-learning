# Embedding Models Guide

This guide provides information about different embedding models you can use with the embedding demos in this repository.

## What are Embeddings?

Embeddings are vector representations of text (or other data) that capture semantic meaning. They allow us to:
- Measure similarity between texts
- Perform semantic search
- Cluster similar documents
- Build recommendation systems
- And much more!

## Available Demo Scripts

This repository includes several demo scripts for different embedding models:

1. **embeddings_demo.py** - Uses SentenceTransformers models
2. **huggingface_embeddings_demo.py** - Uses Hugging Face Transformers models directly
3. **openai_embeddings_demo.py** - Uses OpenAI's embedding models
4. **ollama_embedding.py** - Uses Ollama's local embedding models

## SentenceTransformers Models (embeddings_demo.py)

SentenceTransformers is a Python library that provides easy access to state-of-the-art pretrained models for generating sentence embeddings.

### Usage

```bash
# List available models
python embeddings_demo.py --list-models

# Use a specific model
python embeddings_demo.py --model all-mpnet-base-v2

# Use a custom model not in the recommended list
python embeddings_demo.py --custom-model paraphrase-multilingual-mpnet-base-v2
```

### Recommended Models

- **all-MiniLM-L6-v2**: Fast and efficient general-purpose model (384 dimensions)
- **all-mpnet-base-v2**: High quality general-purpose model (768 dimensions)
- **all-distilroberta-v1**: Distilled RoBERTa model with good performance (768 dimensions)
- **paraphrase-multilingual-MiniLM-L12-v2**: Multilingual model supporting 50+ languages (384 dimensions)
- **multi-qa-mpnet-base-dot-v1**: Optimized for semantic search and question answering (768 dimensions)
- **all-MiniLM-L12-v2**: Larger version of MiniLM with better performance (384 dimensions)
- **msmarco-distilbert-base-v4**: Optimized for information retrieval tasks (768 dimensions)
- **paraphrase-albert-small-v2**: Lightweight model with good performance (768 dimensions)
- **stsb-roberta-large**: High-quality model optimized for semantic textual similarity (1024 dimensions)
- **gtr-t5-large**: T5-based model with strong performance (768 dimensions)

## Hugging Face Transformers Models (huggingface_embeddings_demo.py)

This demo uses Hugging Face's Transformers library directly to generate embeddings from various models.

### Usage

```bash
# List available models
python huggingface_embeddings_demo.py --list-models

# Use a specific model
python huggingface_embeddings_demo.py --model roberta-base

# Use a custom model from Hugging Face Hub
python huggingface_embeddings_demo.py --custom-model intfloat/multilingual-e5-large
```

### Recommended Models

- **bert-base-uncased**: Original BERT base model (768 dimensions)
- **roberta-base**: RoBERTa base model with improved training (768 dimensions)
- **distilbert-base-uncased**: Distilled version of BERT, smaller and faster (768 dimensions)
- **albert-base-v2**: A Lite BERT with parameter reduction techniques (768 dimensions)
- **xlm-roberta-base**: Multilingual RoBERTa model supporting 100 languages (768 dimensions)
- **microsoft/mpnet-base**: MPNet with better performance than BERT/RoBERTa (768 dimensions)
- **google/electra-small-discriminator**: Smaller, efficient ELECTRA model (256 dimensions)
- **sentence-transformers/all-MiniLM-L6-v2**: Optimized for sentence embeddings (384 dimensions)
- **intfloat/e5-small-v2**: E5 model optimized for text embeddings (384 dimensions)
- **facebook/contriever-msmarco**: Contriever model fine-tuned on MS MARCO (768 dimensions)

## OpenAI Embedding Models (openai_embeddings_demo.py)

This demo uses OpenAI's API to generate embeddings. You'll need an OpenAI API key.

### Setup

1. Create a `.env` file in the project root
2. Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

### Usage

```bash
# List available models
python openai_embeddings_demo.py --list-models

# Use a specific model
python openai_embeddings_demo.py --model text-embedding-3-large
```

### Available Models

- **text-embedding-3-small**: Smallest and most cost-effective model (1536 dimensions)
- **text-embedding-3-large**: Most powerful model for high accuracy (3072 dimensions)
- **text-embedding-ada-002**: Legacy model, good balance of performance and cost (1536 dimensions)

## Ollama Embedding Models (ollama_embedding.py)

This demo uses Ollama to run embedding models locally on your machine.

### Setup

1. Install Ollama from [ollama.com](https://ollama.com/)
2. Start the Ollama service
3. Pull the models you want to use: `ollama pull all-minilm`

### Usage

```bash
# List available models
python ollama_embedding.py --list-models

# Use a specific model
python ollama_embedding.py --model nomic-embed-text
```

### Recommended Models

- **all-minilm**: Default embedding model in Ollama (384 dimensions)
- **nomic-embed-text**: Nomic AI's text embedding model (768 dimensions)
- **mxbai-embed-large**: MxbAI's large embedding model (1024 dimensions)
- **ember**: Ember embedding model (1024 dimensions)
- **e5**: E5 embedding model (1024 dimensions)
- **bge**: BGE embedding model (768 dimensions)
- **gte**: GTE embedding model (768 dimensions)

## Comparing Models

Different embedding models have different strengths:

- **Dimension size**: Higher dimensions can capture more information but require more memory
- **Speed**: Smaller models are faster but may be less accurate
- **Multilingual support**: Some models support multiple languages
- **Domain specificity**: Some models are optimized for specific domains or tasks
- **Hosting**: Local models (Ollama) vs. API-based models (OpenAI)

## Choosing the Right Model

Consider these factors when choosing an embedding model:

1. **Task requirements**: What level of accuracy do you need?
2. **Speed requirements**: How fast do you need to generate embeddings?
3. **Resource constraints**: What are your memory and compute limitations?
4. **Language support**: Do you need multilingual capabilities?
5. **Privacy concerns**: Do you need to keep data local?
6. **Cost considerations**: API-based models have usage costs

## Additional Resources

- [SentenceTransformers Documentation](https://www.sbert.net/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings)
- [Ollama Documentation](https://ollama.com/docs)