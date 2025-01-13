import os
import json
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

class EventMemory:
    def __init__(self, emb_model, emb_tokenizer, memory=None):
        """
        Initialize the EventMemory class.
        
        Args:
            emb_model: The embedding model for encoding events.
            emb_tokenizer: The tokenizer for the embedding model.
            memory (list): A list to store encoded events, each as (embedding, timestamp, text).
        """
        self.emb_model = emb_model
        self.emb_tokenizer = emb_tokenizer
        self.memory = memory if memory is not None else []

    def encode_events(self, messages):
        """
        Encode events/messages into embeddings.
        
        Args:
            messages (list of str): List of messages/events to encode.
        
        Returns:
            numpy.ndarray: Encoded embeddings.
        """
        tokens = self.emb_tokenizer(messages, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.emb_model(**tokens).last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings.numpy()

    def store_event(self, event_text, timestamp):
        """
        Store an event in memory with its embedding, timestamp, and text.
        
        Args:
            event_text (str): The text of the event.
            timestamp (str): The timestamp of the event.
        """
        embedding = self.encode_events([event_text])[0]
        self.memory.append((embedding, timestamp, event_text))

    def retrieve_similar_events(self, query, k=3):
        """
        Retrieve similar events to a given query using k-Nearest Neighbors.
        
        Args:
            query (str): The query text.
            k (int): Number of similar events to retrieve.
        
        Returns:
            list: List of tuples containing (embedding, timestamp, text) for similar events.
        """
        if not self.memory:
            raise ValueError("Memory is empty. Store some events before retrieving.")
        
        # Prepare memory embeddings
        embeddings = np.array([entry[0] for entry in self.memory])
        timestamps = [entry[1] for entry in self.memory]
        texts = [entry[2] for entry in self.memory]

        # Encode query
        query_embedding = self.encode_events([query])[0]

        # Perform k-NN search
        knn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(embeddings)
        distances, indices = knn.kneighbors([query_embedding])

        # Retrieve events and ensure temporal contiguity
        similar_events = []
        for idx in indices[0]:
            similar_events.append((embeddings[idx], timestamps[idx], texts[idx]))

        return similar_events

# Example usage:
# emb_model = YourEmbeddingModel()
# emb_tokenizer = YourTokenizer()
# event_memory = EventMemory(emb_model, emb_tokenizer)

# event_memory.store_event("Some event text", "2025-01-12")
# results = event_memory.retrieve_similar_events("Query text", k=3)
