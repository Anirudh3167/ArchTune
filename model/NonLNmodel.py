import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from tqdm.notebook import tqdm

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    """
    From HuggingFace's generation_utils (adapted).
    logits: 1D tensor of shape (vocab_size,)
    """
    top_k = min(max(int(top_k), 0), logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the top-k-th token
        values, _ = torch.topk(logits, top_k)
        min_value = values[-1]
        logits = torch.where(logits < min_value, torch.full_like(logits, filter_value), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative prob above top_p (shifted right to keep at least one token)
        sorted_indices_to_remove = cumulative_probs > top_p
        if sorted_indices_to_remove.numel() > 1:
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        if indices_to_remove.numel() > 0:
            logits[indices_to_remove] = filter_value

    return logits

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

        # self.cos = nn.Buffer(theta.cos(), persistent=False)
        # self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_attn.weight.data[: config.n_embed].zero_()  # init query proj weights to zero
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        # self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        nn.init.zeros_(self.c_proj.weight)
        self.rotary = Rotary(config.n_embed//config.n_head, config.seq_len)
        # regularization
        self.attn_dropout, self.resid_dropout = nn.Dropout(config.dropout), nn.Dropout(config.dropout)
        self.n_head, self.n_embed, self.dropout = config.n_head, config.n_embed, config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # self.attn_scale = 0.12
        # self.attn_scale = (1.0 / (config.n_embed // config.n_head) ** 0.5) * 0.12
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embed, dim=2)
        q,k,v = [i.view(B,T, self.n_head, C // self.n_head) for i in [q,k,v]]
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        q,k,v = [i.transpose(1, 2) for i in [q,k,v]] # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = F.scaled_dot_product_attention(q, k, v, 
                                           attn_mask=None, 
                                           dropout_p=self.dropout if self.training else 0, 
                                           is_causal=True,
                                           # scale=self.attn_scale
                                          )
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return self.resid_dropout(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        # self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        nn.init.zeros_(self.c_proj.weight)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.dropout(self.c_proj(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.csa = CausalSelfAttention(config)
        self.mlp = MLP(config)
        
    def forward(self, x):
        y = norm(x)
        x = x + self.csa(y)
        y = norm(x)
        return x + self.mlp(y)

class GPTLNremoved(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.token_embedding_table = nn.Embedding(self.vocab_size, config.n_embed)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embed,self.vocab_size, bias = False)
        # self.token_embedding_table.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
        # self.lm_head.weight.detach().zero_() # @Grad62304977
        nn.init.zeros_(self.lm_head.weight)
        
        self.tokenizer = tokenizer
        self.config = config
        
        # report number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        print("Internal vocab size is set to: ", self.vocab_size)
        print("number of parameters: %.2fM" % (num_params/1e6,))
    
    def forward(self, input_ids, labels = None, **kwargs):
        # x : (Batch_Size, Seq_len)    # targets : (Batch_Size, Seq_len)
        B, S = input_ids.shape
        x = norm(self.token_embedding_table(input_ids))
        for module in self.blocks:   x = module(x)
        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        if self.config.logits_softcapping:
            logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        output = {"logits" : logits}
        if labels != None:
          B, S, V = logits.shape
          output["loss"] = F.cross_entropy(logits.view(B*S, V), labels.view(-1), 
                                           reduction=self.config.loss_reduction,
                                           ignore_index = -1)
        return output
          
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