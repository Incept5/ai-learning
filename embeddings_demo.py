import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate


def get_embeddings(sentences, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings for a list of sentences using a pre-trained model.

    Args:
        sentences (list): List of sentences to generate embeddings for
        model_name (str): Name of the Sentence Transformer model to use

    Returns:
        np.ndarray: Array of embeddings, one per sentence
    """
    # Load the model
    model = SentenceTransformer(model_name)

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


def main():
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
    all_embeddings = get_embeddings(all_sentences)

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

    # Print header row
    print("     ", end="")
    for i in range(n_mixed):
        print(f"Sent {i + 1}   ", end="")
    print()

    # Create similarity table
    header = [""] + mixed_examples
    table = [[mixed_examples[i]] + [f"{n_mixed[i][j]:.3f}"
                           for j in range(len(mixed_examples))] for i in range(len(mixed_examples))]

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