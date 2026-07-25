from dataclasses import dataclass, asdict

@dataclass(slots=True)
class Hyperparameters:
    vocab_size: int = 50304 # Updated once the tokenizer is built
    seq_len: int = 512
    d_model: int = 256
    n_head: int = 4
    n_layers: int = 4
    hidden_dim: int = 1024
    rope_local_base: float = 10_000.0
    bias: bool = False
    rope_base: float = 1_000_000.0
    
    dropout: float = 0 
    lr: float = 1e-3
    hidden_lr: float = 1e-3  # Applicable for Muon
    momentum: float = 0.95
    batch_size: int = 28     # micro-batch size per gradient step
    grad_accum: int = 1

    device: str = 'cpu'
    epochs: int = 1
    weight_decay: float = 0
    warmup_ratio: float = 0.02

    seed: int = 1337
    eval_steps=1_000
    logging_steps=50
    save_steps=2_000
    ckpt_dir_path: str = './ckpt'

    
    def to_dict(self):
        return asdict(self)