import torch
from torch.utils.data import Dataset
import numpy as np

class MemmapTokenDataset(Dataset):
    def __init__(self, tokens_path, seq_len):
        """Loads contiguous token blocks from a uint16 memmap."""
        self.data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = torch.from_numpy( 
                    self.data[start : start + self.seq_len + 1].astype(np.int64) 
                )
        return chunk[:-1], chunk[1:]  # inputs, labels


class DataCollator:
    def __init__(self, seq_len: int):
        self.causal_mask = torch.tril( torch.ones(seq_len, seq_len, dtype=torch.bool) ).unsqueeze(0)

    def __call__(self, batch):
        inputs, labels = zip(*batch)
        # Attention Mask is a dead code as no mask is really required in current models. 
        # It is for future formats of data.
        return {
            "input_ids": torch.stack(inputs),
            "attention_mask": self.causal_mask,
            "labels": torch.stack(labels),
        }