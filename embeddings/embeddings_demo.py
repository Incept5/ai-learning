import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate


# Dictionary of recommended embedding models with their descriptions
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "Fast and efficient general-purpose model (384 dimensions)",
    "all-mpnet-base-v2": "High quality general-purpose model (768 dimensions)",
    "all-distilroberta-v1": "Distilled RoBERTa model with good performance (768 dimensions)",
    "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual model supporting 50+ languages (384 dimensions)",
    "multi-qa-mpnet-base-dot-v1": "Optimized for semantic search and question answering (768 dimensions)",
    "all-MiniLM-L12-v2": "Larger version of MiniLM with better performance (384 dimensions)",
    "msmarco-distilbert-base-v4": "Optimized for information retrieval tasks (768 dimensions)",
    "paraphrase-albert-small-v2": "Lightweight model with good performance (768 dimensions)",
    "stsb-roberta-large": "High-quality model optimized for semantic textual similarity (1024 dimensions)",
    "gtr-t5-large": "T5-based model with strong performance (768 dimensions)"
}


def get_embeddings(sentences, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings for a list of sentences using a pre-trained model.

    Args:
        sentences (list): List of sentences to generate embeddings for
        model_name (str): Name of the Sentence Transformer model to use

    Returns:
        np.ndarray: Array of embeddings, one per sentence
    """
    # Check if model exists in our recommended list
    if model_name not in EMBEDDING_MODELS and model_name != "custom":
        print(f"Warning: Using model '{model_name}' which is not in the recommended list.")
        print("Available recommended models:")
        for model, desc in EMBEDDING_MODELS.items():
            print(f"  - {model}: {desc}")
    
    # Load the model
    model = SentenceTransformer(model_name)
    
    # Print model info
    print(f"Using model: {model_name}")
    print(f"Embedding dimensions: {model.get_sentence_embedding_dimension()}")

    # Generate embeddings
    embeddings = model.encode(sentences)

    return embeddings


def calculate_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1 (np.ndarray): First embedding vector
        embedding2 (np.ndarray): Second embedding vector

    Returns:
        float: Cosine similarity score (0-1)
    """
    # Reshape embeddings for sklearn's cosine_similarity
    e1 = embedding1.reshape(1, -1)
    e2 = embedding2.reshape(1, -1)

    # Calculate and return similarity
    return cosine_similarity(e1, e2)[0][0]


def list_models():
    """Print all available recommended embedding models with descriptions"""
    print("\nAvailable embedding models:")
    print("-" * 80)
    for model, desc in EMBEDDING_MODELS.items():
        print(f"{model}")
        print(f"    {desc}")
    print("-" * 80)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Sentence embedding similarity demo")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Embedding model to use")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available recommended models and exit")
    parser.add_argument("--custom-model", type=str, 
                        help="Use a custom model not in the recommended list")
    args = parser.parse_args()
    
    # If --list-models flag is provided, list models and exit
    if args.list_models:
        list_models()
        return
    
    # Determine which model to use
    model_name = args.custom_model if args.custom_model else args.model
    
    # Define example sentence pairs
    similar_pair = [
        "The cat sat on the mat.",
        "A feline rested on the rug."
    ]

    dissimilar_pair = [
        "The sky is blue.",
        "Bananas are yellow."
    ]

    mixed_examples = [
        "I enjoy reading science fiction novels.",
        "Reading sci-fi books is my favorite hobby.",
        "The restaurant serves delicious pasta dishes.",
        "My computer needs a hardware upgrade."
    ]

    # Get embeddings for all sentences
    print("Loading model and generating embeddings...")
    all_sentences = similar_pair + dissimilar_pair + mixed_examples
    all_embeddings = get_embeddings(all_sentences, model_name=model_name)

    # Calculate and display similarity for the provided examples
    print("\n--- Example Pairs ---")

    # Similar pair
    sim_score = calculate_similarity(all_embeddings[0], all_embeddings[1])
    print(f"Similar pair similarity score: {sim_score:.4f}")
    print(f"  - \"{similar_pair[0]}\"")
    print(f"  - \"{similar_pair[1]}\"")

    # Dissimilar pair
    dissim_score = calculate_similarity(all_embeddings[2], all_embeddings[3])
    print(f"\nDissimilar pair similarity score: {dissim_score:.4f}")
    print(f"  - \"{dissimilar_pair[0]}\"")
    print(f"  - \"{dissimilar_pair[1]}\"")

    # Compare all mixed examples with each other
    print("\n--- Mixed Examples Similarity Matrix ---")
    start_idx = 4  # Index where mixed examples start in all_embeddings
    n_mixed = len(mixed_examples)

    # Create similarity table
    header = [""] + mixed_examples
    
    # Calculate similarity scores for the table
    table = []
    for i in range(n_mixed):
        row = [mixed_examples[i]]
        for j in range(n_mixed):
            emb_i = all_embeddings[start_idx + i]
            emb_j = all_embeddings[start_idx + j]
            sim = calculate_similarity(emb_i, emb_j)
            row.append(f"{sim:.3f}")
        table.append(row)

    print(tabulate(table, headers=header, tablefmt="grid"))

    # Calculate and print similarity matrix
    for i in range(n_mixed):
        print(f"Sent {i + 1} ", end="")
        for j in range(n_mixed):
            emb_i = all_embeddings[start_idx + i]
            emb_j = all_embeddings[start_idx + j]
            sim = calculate_similarity(emb_i, emb_j)
            print(f"{sim:.4f}    ", end="")
        print()

    print("\nSentence reference:")
    for i, sent in enumerate(mixed_examples):
        print(f"Sentence {i + 1}: {sent}")


if __name__ == "__main__":
    main()
    print("\nTip: Run with --list-models to see all available embedding models")
    print("Example: python embeddings_demo.py --model all-mpnet-base-v2")
    print("Example with custom model: python embeddings_demo.py --custom-model paraphrase-multilingual-mpnet-base-v2")