import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def compute_word_embeddings(words):
    """
    Compute simple embeddings for words using character-level features
    and calculate pairwise similarity.
    """
    # Create a simple character-level embedding
    # We'll use character n-grams (1, 2, and 3) as features
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))

    # Fit and transform the words to get embeddings
    embeddings = vectorizer.fit_transform(words)

    # Compute cosine similarity between all pairs
    similarity_matrix = cosine_similarity(embeddings)

    # Create a DataFrame for better visualization
    similarity_df = pd.DataFrame(similarity_matrix,
                                 index=words,
                                 columns=words)

    return similarity_df


if __name__ == "__main__":
    # List of words to analyze
    words = ["tea", "coffee", "mud", "dirt"]

    # Compute and display similarity matrix
    similarity_table = compute_word_embeddings(words)

    print("Word Similarity Matrix (based on character-level features):")
    print(similarity_table.round(3))

    # Optional: Display more detailed explanation
    print("\nExplanation:")
    print("- Values range from 0 to 1, where 1 means identical")
    print("- Higher values indicate more similar character patterns")
    print("- This simple model uses character n-grams as features")

    # Optional: Find the most similar pair
    if len(words) > 1:
        # Create a copy of the similarity matrix
        sim_matrix = similarity_table.values.copy()
        # Set diagonal to 0 to ignore self-similarity
        np.fill_diagonal(sim_matrix, 0)
        # Find the indices of the maximum similarity
        max_i, max_j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
        print(f"\nMost similar pair: '{words[max_i]}' and '{words[max_j]}' "
              f"with similarity score: {sim_matrix[max_i, max_j]:.3f}")