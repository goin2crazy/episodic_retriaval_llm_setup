import json 
import os 
import numpy as np 


# Initialize memory database synced with local files
class MemoryDatabase:
    def __init__(self, directory="./memory_data"):
        self.directory = directory
        self.file_path = os.path.join(directory, "mdatabase.json")
        os.makedirs(directory, exist_ok=True)
        # Load existing memory or initialize empty list
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.memory = [tuple(entry) for entry in json.load(f)]
        else:
            self.memory = []

    def save(self):
        """Save the memory to a local file."""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=4)

    def append(self, entry):
        """Add a new entry to memory and sync with file."""
        embedding, timestamp, event_text = entry
        self.memory.append((embedding.tolist(), timestamp, event_text))
        self.save()

    def __getitem__(self, idx):
        """Retrieve an item by index."""
        embedding, timestamp, event_text = self.memory[idx]
        return np.array(embedding), timestamp, event_text

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        for entry in self.memory:
            yield np.array(entry[0]), entry[1], entry[2]