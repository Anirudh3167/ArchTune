import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.notebook import tqdm
from .utils import top_k_top_p_filtering, RMSNorm, compute_rope_params
from .feedforward import FeedForward
from .attention_layer import GroupedQueryAttention

class TransformerBlock(nn.Module):
    def __init__(self, config, attn_type: str):
        super().__init__()
        self.attn_type = attn_type

        self.att = GroupedQueryAttention(
            d_in=config.n_embed,
            num_heads=config.n_head,
            num_kv_groups=config.n_kv_groups,
            head_dim=config.head_dim,
            qk_norm=config.qk_norm,
            query_pre_attn_scalar=config.query_pre_attn_scalar
        )
        self.ff = FeedForward(config)
        self.input_layernorm = RMSNorm(config.n_embed, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(config.n_embed, eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(config.n_embed, eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(config.n_embed, eps=1e-6)

    def forward(
        self,
        x,
        mask_global,
        mask_local,
        cos_global,
        sin_global,
        cos_local,
        sin_local,
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        # Shortcut connection for feed forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x

class Gemma3Model(nn.Module):
    def __init__(self, config,tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        assert config.layer_types is not None and len(config.layer_types) == config.n_layer

        # Main model parameters
        self.tok_embedding = nn.Embedding(config.vocab_size, config.n_embed)

        self.blocks = nn.ModuleList([
            TransformerBlock(config, attn_type)for attn_type in config.layer_types
        ])

        self.final_norm = RMSNorm(config.n_embed, eps=1e-6)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.config = config

        # Reusuable utilities
        cos_local, sin_local = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=config.rope_local_base,
            context_length=config.seq_len
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=config.head_dim,
            theta_base=config.rope_base,
            context_length=config.seq_len
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len, device):
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # mask_global (future is masked: j > i)
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 1 1 1 1 1 1 1
        #     1:  0 0 1 1 1 1 1 1
        #     2:  0 0 0 1 1 1 1 1
        #     3:  0 0 0 0 1 1 1 1
        #     4:  0 0 0 0 0 1 1 1
        #     5:  0 0 0 0 0 0 1 1
        #     6:  0 0 0 0 0 0 0 1
        #     7:  0 0 0 0 0 0 0 0
        mask_global = torch.triu(ones, diagonal=1)

        # far_past (too far back is masked: i - j >= sliding_window)
        # where sliding_window = 4
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 0 0 0 0 0 0 0
        #     1:  0 0 0 0 0 0 0 0
        #     2:  0 0 0 0 0 0 0 0
        #     3:  0 0 0 0 0 0 0 0
        #     4:  1 0 0 0 0 0 0 0
        #     5:  1 1 0 0 0 0 0 0
        #     6:  1 1 1 0 0 0 0 0
        #     7:  1 1 1 1 0 0 0 0
        far_past = torch.triu(ones, diagonal=self.config.sliding_window).T

        # Local (sliding_window) = future OR far-past
        # mask_local
        #     j:  0 1 2 3 4 5 6 7
        # i
        # 0:      0 1 1 1 1 1 1 1
        # 1:      0 0 1 1 1 1 1 1
        # 2:      0 0 0 1 1 1 1 1
        # 3:      0 0 0 0 1 1 1 1
        # 4:      1 0 0 0 0 1 1 1
        # 5:      1 1 0 0 0 0 1 1
        # 6:      1 1 1 0 0 0 0 1
        # 7:      1 1 1 1 0 0 0 0
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def forward(self, input_ids, labels=None): #, **kwargs):
        b, seq_len = input_ids.shape
        x = self.tok_embedding(input_ids) * (self.config.n_embed ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)

        for block in self.blocks:
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return {"logits":logits, "loss":loss}

    @property
    def device(self):   return next(self.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        x: str,
        num_tokens: int = 100,
        test_seq_len: int = 384,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        strategy: str = "sampling"  # "sampling" or "beam"
    ):
        """
        Robust generation:
          - sampling with top-k/top-p + repetition penalty (recommended)
          - or a very small beam search (beam=2)
        """
        self.eval()
        # encode -> 1D tensor (seq_len,)
        tokens = self.tokenizer.encode(x, add_special_tokens=False, return_tensors="pt").to(self.device)[0].long()
    
        if strategy == "beam":
            # Very simple beam=2 implementation (deterministic-ish, often repetitive)
            beams = [(tokens, 0.0)]  # (tokens_tensor, log_prob_score)
            for _ in tqdm(range(num_tokens), desc="Generating (beam):", leave=False):
                new_beams = []
                for toks, score in beams:
                    out = self(toks[-test_seq_len:].unsqueeze(0))["logits"][0, -1]  # logits (vocab,)
                    log_probs = F.log_softmax(out / max(temperature, 1e-8), dim=-1)
                    top_logvals, top_indices = torch.topk(log_probs, 2)  # beam=2 expansions per beam
                    for lv, idx in zip(top_logvals, top_indices):
                        new_tok = torch.cat((toks, idx.view(1).to(toks.device)))
                        new_beams.append((new_tok, score + float(lv)))
                # keep top 2 beams
                beams = sorted(new_beams, key=lambda b: b[1], reverse=True)[:2]
            out_tokens = beams[0][0]
            return self.tokenizer.decode(out_tokens.tolist())
    
        # --- sampling branch ---
        for _ in tqdm(range(num_tokens), desc="Generating (sampling):", leave=False):
            out = self(tokens[-self.config.seq_len:].unsqueeze(0))["logits"][0, -1]  # logits (vocab,)
    
            # --- repetition penalty (HuggingFace style) ---
            if repetition_penalty != 1.0:
                # operate in-place on a cloned tensor to be safe
                logits = out.clone()
                prev_tokens = set(tokens.tolist())
                for tid in prev_tokens:
                    # If score < 0: multiply by penalty, else divide by penalty
                    if logits[tid] < 0:
                        logits[tid] *= repetition_penalty
                    else:
                        logits[tid] /= repetition_penalty
            else:
                logits = out.clone()
    
            # --- apply temperature and filtering on logits ---
            # Protect against extreme zero temperature
            temp = max(temperature, 1e-8)
            logits = logits / temp
    
            # Filter logits using top-k / top-p (works on logits)
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
            # Numerical safety: if all values are -inf (can happen in pathological cases), fall back
            if not torch.isfinite(filtered_logits).any():
                filtered_logits = logits.clone()
    
            probs = F.softmax(filtered_logits, dim=-1)
    
            # Safety guard: if probs is invalid or sums to zero, fallback to stable softmax
            if (not torch.isfinite(probs).all()) or probs.sum() <= 0:
                probs = F.softmax(logits, dim=-1)
    
            # final draw
            try:
                tkn = torch.multinomial(probs, num_samples=1).to(tokens.device)
            except RuntimeError:
                # Last-ditch fallback: pick argmax (greedy)
                tkn = torch.tensor([int(torch.argmax(probs))], device=tokens.device)
    
            tokens = torch.cat((tokens, tkn.view(-1).long()))
    
        return self.tokenizer.decode(tokens.tolist())