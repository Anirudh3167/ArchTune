from dataclasses import dataclass
@dataclass
class Hyperparameters:
    batch_size: int = 12 #32 #32 #64 #16                 # micro-batch size per gradient step
    seq_len: int = 1024 #384 #384
    vocab_size: int = 50_257
    n_layer: int = 12 #6 #16 #8 #4
    n_head: int = 4 #8 #4
    n_embed: int =  256 #384 #128 #384 #512 #256
    dropout: float = 0 #0.01
    lr: float = 5e-4
    ckpt_path: str = './ckpt.pt'
    device: str = 'cpu'
    bias: bool = False

    out_dir: str = "./output_dir"
    epochs: int = 1 #2 #4 #3 #2 #1
    grad_accum: int = 1 #4 #1
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    seed: int = 1337
    eval_steps=50#500
    logging_steps=50#500
    save_steps=1_000
    loss_reduction = "mean"

    logits_softcapping: bool = False
    
config = Hyperparameters()