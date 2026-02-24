import torch
import torch.nn as nn

class HashEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, bucket_size, num_hash_functions):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.bucket_size = bucket_size
        self.num_hash_functions = num_hash_functions

        # Shared bucket embeddings
        self.weight = nn.Parameter(
            torch.randn(bucket_size, hidden_dim) * 0.02
        )

        # Fixed random hash parameters
        self.register_buffer(
            "hash_a",
            torch.randint(1, bucket_size, (num_hash_functions,))
        )
        self.register_buffer(
            "hash_b",
            torch.randint(0, bucket_size, (num_hash_functions,))
        )
        self.register_buffer(
            "sign_a",
            torch.randint(1, bucket_size, (num_hash_functions,))
        )
        self.register_buffer(
            "sign_b",
            torch.randint(0, bucket_size, (num_hash_functions,))
        )

    def forward(self, input_ids):
        device = input_ids.device

        # input_ids: (B, T)
        B, T = input_ids.shape
        input_ids = input_ids.long()

        # Expand for multiple hashes
        # shape: (B, T, H)
        ids = input_ids.unsqueeze(-1)

        hash_a = self.hash_a.to(device)
        hash_b = self.hash_b.to(device)
        sign_a = self.sign_a.to(device)
        sign_b = self.sign_b.to(device)


        # Compute hash bucket indices
        buckets = (ids * hash_a + hash_b) % self.bucket_size
        # shape: (B, T, num_hash_functions)

        # Compute sign hash (+1 / -1)
        signs = ((ids * sign_a + sign_b) % 2) * 2 - 1
        signs = signs.float()

        # Lookup embeddings
        # (B, T, num_hash_functions, hidden_dim)
        embedded = self.weight[buckets]

        # Apply sign
        embedded = embedded * signs.unsqueeze(-1)

        # Average across hash functions
        output = embedded.mean(dim=2)

        return output