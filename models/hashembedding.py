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

        # Fixed random hash parameters (buffers)
        self.register_buffer(
            "hash_a", torch.randint(1, bucket_size, (num_hash_functions,), dtype=torch.long)
        )
        self.register_buffer(
            "hash_b", torch.randint(0, bucket_size, (num_hash_functions,), dtype=torch.long)
        )
        self.register_buffer(
            "sign_a", torch.randint(1, bucket_size, (num_hash_functions,), dtype=torch.long)
        )
        self.register_buffer(
            "sign_b", torch.randint(0, bucket_size, (num_hash_functions,), dtype=torch.long)
        )

    def forward(self, input_ids):
        # Ensure input is long and get device
        input_ids = input_ids.to(self.weight.device).long()

        # Expand for multiple hashes
        ids = input_ids.unsqueeze(-1)  # (B, T, 1)


        # Compute bucket indices
        buckets = (ids * self.hash_a + self.hash_b) % self.bucket_size  # (B, T, num_hash_functions)

        # Compute sign (+1/-1)
        signs = ((ids * self.sign_a + self.sign_b) % 2) * 2 - 1
        signs = signs.float()  # ensure same device

        # Lookup embeddings (weight must be on same device)
        embedded = self.weight[buckets]  # (B, T, num_hash_functions, hidden_dim)

        # Apply sign
        print(embedded.device, signs.device)
        embedded = embedded * signs.unsqueeze(-1)

        # Average across hash functions
        return embedded.mean(dim=2)