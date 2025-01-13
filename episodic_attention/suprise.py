import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_surprise_scores(tokens, surprise_scores, title='Surprise Scores for Each Token'):
    """
    Visualizes the surprise scores for each token using a bar plot.

    Args:
        tokens (list): List of tokens (strings).
        surprise_scores (list): List of surprise scores corresponding to each token.
    """
    # Create a bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(tokens, surprise_scores, color='skyblue')

    # Add labels and title
    plt.xlabel('Tokens')
    plt.ylabel('Surprise Score')
    plt.title(title)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    plt.show()

    
def compute_surprise_scores(text, model, processor, batch_size=4):
    # Tokenize the input text
    tokens = processor.tokenizer(text, return_tensors="pt")
    token_ids = tokens.input_ids[0]  # Get the token IDs (remove batch dimension)

    # Initialize list to store surprise scores
    surprise_scores = []

    # Iterate over the sequence in batches
    for start_idx in range(1, len(token_ids), batch_size):
        # Determine the end index for the batch
        end_idx = min(start_idx + batch_size, len(token_ids) + 1)

        # Create a batch of progressively longer subsequences
        batch_subsequences = [token_ids[:i] for i in range(start_idx, end_idx)]

        # Pad subsequences to the same length
        max_len = max(len(seq) for seq in batch_subsequences)
        padded_batch = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in batch_subsequences]
        padded_batch = torch.stack(padded_batch)  # Shape: (batch_size, max_len)

        # Get the actual token IDs for the next positions
        actual_token_ids = token_ids[start_idx:end_idx]

        # Prepare the input dictionary for the model
        context_tokens = {
            "input_ids": padded_batch,
            "attention_mask": (padded_batch != 0).int()  # Create attention mask for padding
        }


        # Perform batched inference
        with torch.no_grad():
            outputs = model(**context_tokens, use_cache=True)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

        # Get the probabilities for the last token in each subsequence
        probs = torch.softmax(logits[:, -1, :], dim=-1)  # Shape: (batch_size, vocab_size)

        # Gather the probabilities of the actual next tokens
        print(actual_token_ids)

        next_token_probs = probs[torch.arange(len(actual_token_ids)), actual_token_ids].cpu().numpy()

        # Compute surprise scores for the batch
        batch_surprise_scores = -np.log(next_token_probs + 1e-10)  # Avoid log(0)

        # Append the surprise scores to the list
        surprise_scores.extend(batch_surprise_scores)

    return processor.tokenizer.batch_decode(token_ids[1:].tolist()), surprise_scores