from dataclasses import dataclass, asdict
@dataclass
class GPTHyperparameters:
    batch_size: int = 12 #32 #32 #64 #16                 # micro-batch size per gradient step
    seq_len: int = 1024 #384 #384
    vocab_size: int = 50_257
    n_layer: int = 4 #6 #16 #8 #4
    n_head: int = 4 #8 #4
    n_embed: int =  512 #384 #128 #384 #512 #256
    dropout: float = 0 #0.01
    lr: float = 1e-3#5e-4
    hidden_lr: float = 1e-3  # Applicable for Muon
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

    
    n_experts: int = 8
    k_expts: int = 2
    ff_layers = ["ffn","moe","moe","ffn"]

    logits_softcapping: bool = False

    # Additional Customization Config
    use_embed_scaling: bool = False

    def to_dict(self):
        return asdict(self)

@dataclass
class GemmaHyperparameters:
    vocab_size: int = 50304
    seq_len: int = 512 #32_768
    n_embed: int = 256 #512
    n_head: int = 4
    n_layer: int = 4 #6 #18
    hidden_dim: int = 1024 #2048
    head_dim: int = 256
    qk_norm: bool = True
    n_kv_groups: int = 1
    rope_local_base: float = 10_000.0
    rope_base: float = 1_000_000.0
    sliding_window: int = 256
    layer_types = [  # 5:1 ratio of sliding : full attention
        # "sliding_attention",
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
    ]
    query_pre_attn_scalar: int = 256
    use_embed_scaling: bool = False

    n_experts: int = 8
    k_expts: int = 2
    ff_layers = ["ffn","moe","moe","ffn"]
    
    dropout: float = 0 #0.01
    lr: float = 1e-3#5e-4
    hidden_lr: float = 1e-3  # Applicable for Muon

    ckpt_path: str = './ckpt.pt'
    device: str = 'cpu'
    bias: bool = False
    out_dir: str = "./output_dir"
    epochs: int = 1 #2 #4 #3 #2 #1
    grad_accum: int = 1 #4 #1
    weight_decay: float = 0
    warmup_ratio: float = 0.02
    seed: int = 1337
    eval_steps=50#500
    logging_steps=50#500
    save_steps=1_000
    batch_size: int = 28 #32 #32 #64 #16                 # micro-batch size per gradient step
    
    def to_dict(self):
        return asdict(self)