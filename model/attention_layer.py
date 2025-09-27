import torch, torch.nn as nn, torch.nn.functional as F
from .utils import GemmaRotary, Rotary, norm, RMSNorm
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_attn.weight.data[: config.n_embed].zero_()  # init query proj weights to zero
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        nn.init.zeros_(self.c_proj.weight)  # @Grad62304977
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


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_head % config.n_kv_groups == 0, "config.n_head must be divisible by config.n_kv_groups"

        self.num_heads = config.n_head
        self.num_kv_groups = config.n_kv_groups
        self.group_size = config.n_head // config.n_kv_groups

        if config.head_dim is None:
            assert config.n_embed % config.n_head == 0, "`config.n_embed` must be divisible by `config.n_head` if `head_dim` is not set"
            head_dim = config.n_embed // config.n_head
        else:
            head_dim = config.head_dim

        self.head_dim = head_dim
        self.d_out = config.n_head * head_dim

        self.W_query = nn.Linear(config.n_embed, self.d_out, bias=False)
        self.W_key = nn.Linear(config.n_embed, config.n_kv_groups * head_dim, bias=False)
        self.W_value = nn.Linear(config.n_embed, config.n_kv_groups * head_dim, bias=False)

        self.out_proj = nn.Linear(self.d_out, config.n_embed, bias=False)

        if config.qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None
        
        self.rotary = GemmaRotary(config.n_embed//config.n_head, config.seq_len)

        if config.query_pre_attn_scalar is not None:
            self.scaling = (config.query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = (head_dim) ** -0.5

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')


    def forward(self, x):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = self.rotary(queries)
        keys = self.rotary(keys)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Scale queries
        queries = queries * self.scaling

        # Attention using PyTorch SDPA with FlashAttention backend
        context = F.scaled_dot_product_attention(
            queries,   # [b, h, t, d]
            keys,      # [b, h, t, d]
            values,    # [b, h, t, d]
            attn_mask=None,      # No explicit mask needed
            dropout_p=0.0,       # Or your dropout value
            is_causal=True       # Enables causal masking
        )

        # Reshape context and project
        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)