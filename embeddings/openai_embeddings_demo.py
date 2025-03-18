import numpy as np
import os
import argparse
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import requests

# Load environment variables from .env file
load_dotenv()

# Dictionary of OpenAI embedding models with their descriptions
OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small": "Smallest and most cost-effective model (1536 dimensions)",
    "text-embedding-3-large": "Most powerful model for high accuracy (3072 dimensions)",
    "text-embedding-ada-002": "Legacy model, good balance of performance and cost (1536 dimensions)"
}

def get_openai_embeddings(sentences, model_name="text-embedding-3-small"):
    """
    Generate embeddings for a list of sentences using OpenAI's embedding models.
    
    Args:
        sentences (list): List of sentences to generate embeddings for
        model_name (str): Name of the OpenAI embedding model to use
        
    Returns:
        np.ndarray: Array of embeddings, one per sentence
    """
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    
    # Check if model exists in our recommended list
    if model_name not in OPENAI_EMBEDDING_MODELS:
        print(f"Warning: Using model '{model_name}' which is not in the recommended list.")
        print("Available OpenAI embedding models:")
        for model, desc in OPENAI_EMBEDDING_MODELS.items():
            print(f"  - {model}: {desc}")
    
    # Print model info
    print(f"Using OpenAI model: {model_name}")
    
    # API endpoint
    url = "https://api.openai.com/v1/embeddings"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Store all embeddings
    all_embeddings = []
    
    # Process each sentence
    for sentence in sentences:
        # Request body
        data = {
            "input": sentence,
            "model": model_name
        }
        
        # Make the API call
        response = requests.post(url, headers=headers, json=data)
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Error from OpenAI API: {response.text}")
        
        # Extract embedding
        embedding = response.json()["data"][0]["embedding"]
        all_embeddings.append(embedding)
    
    # Convert to numpy array
    return np.array(all_embeddings)

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
    """Print all available OpenAI embedding models with descriptions"""
    print("\nAvailable OpenAI embedding models:")
    print("-" * 80)
    for model, desc in OPENAI_EMBEDDING_MODELS.items():
        print(f"{model}")
        print(f"    {desc}")
    print("-" * 80)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="OpenAI Embedding Similarity Demo")
    parser.add_argument("--model", type=str, default="text-embedding-3-small",
                        help="OpenAI embedding model to use")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available OpenAI models and exit")
    args = parser.parse_args()
    
    # If --list-models flag is provided, list models and exit
    if args.list_models:
        list_models()
        return
    
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
    print("Generating OpenAI embeddings...")
    all_sentences = similar_pair + dissimilar_pair + mixed_examples
    
    try:
        all_embeddings = get_openai_embeddings(all_sentences, model_name=args.model)
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use OpenAI embeddings, you need to:")
        print("1. Create a .env file in the project root")
        print("2. Add your OpenAI API key: OPENAI_API_KEY=your_api_key_here")
        return

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
    print("\nTip: Run with --list-models to see all available OpenAI embedding models")
    print("Example: python openai_embeddings_demo.py --model text-embedding-3-large")