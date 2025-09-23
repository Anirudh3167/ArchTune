import time, os, torch

from .load_data import load_data
from .preprocess import preprocess_wiki_text_generation, preprocess_openwebtext
from .tokenizer import get_tokenizer

from typing import List
from tqdm.notebook import tqdm

# Mapping dataset name → preprocessing function
PREPROCESS_FUNCS = {
    "wiki-text-103": preprocess_wiki_text_generation,
    "openwebtext": preprocess_openwebtext,
}


class Data:
    stats = {"total_tokens": 0, "total_chunks": 0, "seq_len":0}
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        tokenizer=None,
        seq_len: int = 512,
        num_samples: int = None,
        sample_start_idx: int = 0,
        streaming: bool = False,
        get_stats: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer or get_tokenizer()
        self.seq_len = seq_len
        self.stats["seq_len"] = seq_len

        # Time logging
        t0 = time.time()

        # Step 1: Load
        ds = load_data(dataset_name, split=split, num_samples=num_samples, 
                       sample_start_idx=sample_start_idx, streaming=streaming, **kwargs)
        t1 = time.time()
        self.stats["load_time"] = t1 - t0

        # Step 2: Preprocess
        preprocess_fn = PREPROCESS_FUNCS.get(dataset_name)
        if preprocess_fn is not None: # Open compatibility with other datasets
            ds = preprocess_fn(ds)
        t2 = time.time()
        self.stats["preprocess_time"] = t2 - t1

        # Step 3: Tokenize + Chunk
        token_stream = self.generate_token_stream(ds)
        total_tokens = len(token_stream)
        t3 = time.time()
        self.stats["tokenize_time"] = t3 - t2

        # Chunk into fixed-size sequences
        tokenized_chunks = self.chunk_tokens(token_stream)
        t4 = time.time()
        self.stats["chunk_time"] = t4 - t3

        # Save
        self.data = tokenized_chunks
        self.total_tokens = total_tokens
        self.stats["total_tokens"] = total_tokens
        self.stats["total_chunks"] = len(tokenized_chunks)
        self.stats["total_time"] = t4 - t0
        if get_stats:
            print(self.stats)

    def generate_token_stream(self, ds):
        bos_token, eos_token = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        def tokenize_example(example):
            tokens = self.tokenizer(
                example["text"],
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=False,
                add_special_tokens=False
            )["input_ids"]
            return {"input_ids": [bos_token] + tokens + [eos_token]}

        tokenized_dataset = ds.map(
                tokenize_example,
                batched=False,
                num_proc=max(1, os.cpu_count()-1),
                desc="Tokenizing",
                remove_columns=ds.column_names,
        )
    
        # Flatten into a single token stream
        token_stream = [token for example in tokenized_dataset for token in example["input_ids"]]
        return token_stream
        
    def chunk_tokens(self, token_stream: List[int]) -> List[List[int]]:
        """
        Splits the token stream into fixed-length chunks for training.
        """
        range_start = 0
        range_end = len(token_stream) - self.seq_len - 1
        step = self.seq_len + 1
        
        chunks = [
            torch.tensor(token_stream[i:i + step], dtype=torch.long)
            for i in tqdm(range(range_start, range_end, step),
                          total=(range_end - range_start) // step + 1,
                          desc="Chunking")
        ]
        return chunks
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],  # language modeling objective
        }

    def __add__(self, other):
        assert isinstance(other, Data), "Only `Data` instances can be added"
        merged = Data.__new__(Data)  # create new instance without calling __init__
        merged.tokenizer = self.tokenizer
        merged.seq_len = self.seq_len
        merged.data = self.data + other.data
        merged.total_tokens = self.total_tokens + other.total_tokens
        merged.stats["total_tokens"] = self.stats["total_tokens"] + other.stats["total_tokens"]
        merged.stats["total_chunks"] = self.stats["total_chunks"] + other.stats["total_chunks"]
        merged.stats["load_time"] = self.stats["load_time"] + other.stats["load_time"]
        merged.stats["preprocess_time"] = self.stats["preprocess_time"] + other.stats["preprocess_time"]
        merged.stats["tokenize_time"] = self.stats["tokenize_time"] + other.stats["tokenize_time"]
        merged.stats["chunk_time"] = self.stats["chunk_time"] + other.stats["chunk_time"]
        merged.stats["total_time"] = self.stats["total_time"] + other.stats["total_time"]
        return merged
