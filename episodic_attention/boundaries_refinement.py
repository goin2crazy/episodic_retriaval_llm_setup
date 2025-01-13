import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

def similarity_fn(text, tokenizer, model):
    """
    Computes a similarity matrix for tokens based on KV cache keys from an LLM.

    Args:
        text (str): Input text.
        tokenizer: Corresponding tokenizer for the model.
        model: Pretrained language model (e.g., LLaMA-2 or similar).

    Returns:
        np.ndarray: Similarity matrix of shape (n, n) where n is the number of tokens.
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Max token ID: {input_ids.max().item()}, Model vocab size: {model.config.vocab_size}")

    # Forward pass through the model with KV caching enabled
    with torch.no_grad():
        output = model(input_ids, use_cache=True)

    # Access the KV cache from the model's output
    past_key_values = output.past_key_values  # List of tuples (keys, values) for each layer

    # Extract keys for the last layer
    keys = past_key_values[-1][0].clone()  # Shape: (batch_size, num_heads, seq_length, head_dim)
    batch_size, num_heads, seq_length, head_dim = keys.shape

    # Merge heads into a single key vector per token
    keys_merged = keys.permute(0, 2, 1, 3).reshape(batch_size, seq_length, num_heads * head_dim)

    # Replace NaN, inf values with 0
    keys_merged = torch.nan_to_num(keys_merged, nan=0, posinf=0, neginf=0)

    # Normalize the keys to prevent large values
    keys_merged = torch.nn.functional.normalize(keys_merged, p=2, dim=-1)
    print(f"Keys merged max: {keys_merged.max().item()}, min: {keys_merged.min().item()}")

    # Compute the similarity matrix
    similarity_matrix = torch.matmul(keys_merged, keys_merged.transpose(-1, -2))

    # Convert to numpy for further processing
    return similarity_matrix.squeeze(0).cpu().numpy()

def get_initial_boundaries(surprise, threshold):
    """
    Get initial boundaries based on surprise values.
    """
    boundaries = [0]  # Start with the first token
    for i, s in enumerate(surprise):
        if s > threshold:  # If surprise exceeds the threshold, mark as a boundary
            boundaries.append(i)
    boundaries.append(len(surprise))  # End with the last token
    return boundaries

def boundary_refinement(similarity_matrix, initial_boundaries):
    """
    Refines boundaries based on modularity and conductance metrics.

    Args:
        similarity_matrix (np.ndarray): A similarity matrix (Adjacency matrix A^h).
        initial_boundaries (list): List of initial boundaries as indices.

    Returns:
        list: Refined boundaries.
    """
    def modularity(A, communities):
        m = np.sum(A) / 2
        modularity_score = 0
        for community in communities:
            for i in community:
                for j in community:
                    modularity_score += (A[i, j] - np.sum(A[i, :]) * np.sum(A[:, j]) / (2 * m))
        return modularity_score / (2 * m)

    def conductance(A, communities):
        conductance_scores = []
        for community in communities:
            S = set(community)
            volume = np.sum(A[community, :])
            cut = np.sum(A[np.ix_(community, list(set(range(A.shape[0])) - S))])
            conductance_scores.append(cut / min(volume, np.sum(A) - volume))
        return np.mean(conductance_scores)

    def get_communities(boundaries):
        return [list(range(boundaries[i], boundaries[i+1])) for i in range(len(boundaries) - 1)]

    boundaries = initial_boundaries[:]
    best_boundaries = boundaries[:]
    best_score = float('-inf')

    for _ in range(10):  # Max iterations for refinement
        communities = get_communities(boundaries)
        modularity_score = modularity(similarity_matrix, communities)
        conductance_score = conductance(similarity_matrix, communities)

        # Score optimization: maximize modularity and minimize conductance
        score = modularity_score - conductance_score

        if score > best_score:
            best_score = score
            best_boundaries = boundaries[:]

        # Adjust boundaries
        for i in range(1, len(boundaries) - 1):
            boundaries[i] += np.random.choice([-1, 1])  # Move boundary slightly

            # Ensure valid boundaries
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = boundaries[i - 1] + 1
            if boundaries[i] >= boundaries[i + 1]:
                boundaries[i] = boundaries[i + 1] - 1

    return best_boundaries
