import torch, torch.nn as nn, torch.nn.functional as F
from .utils import Rotary, norm, RMSNorm, apply_rope, check_precision
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


# class GroupedQueryAttention(nn.Module):
#     def __init__(
#         self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False,
#         query_pre_attn_scalar=None, dtype=None,
#     ):
#         super().__init__()
#         assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

#         self.num_heads = num_heads
#         self.num_kv_groups = num_kv_groups
#         self.group_size = num_heads // num_kv_groups

#         if head_dim is None:
#             assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
#             head_dim = d_in // num_heads

#         self.head_dim = head_dim
#         self.d_out = num_heads * head_dim

#         self.W_query = nn.Linear(d_in, self.d_out, bias=False)
#         self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)
#         self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)

#         self.out_proj = nn.Linear(self.d_out, d_in, bias=False)

#         if qk_norm:
#             # self.q_norm = RMSNorm(head_dim, eps=1e-6)
#             # self.k_norm = RMSNorm(head_dim, eps=1e-6)
#             # Reducing the eps for prevent nan in the loss
#             self.q_norm = RMSNorm(head_dim, eps=1e-4)
#             self.k_norm = RMSNorm(head_dim, eps=1e-4)
#         else:
#             self.q_norm = self.k_norm = None

#         if query_pre_attn_scalar is not None:
#             self.scaling = (query_pre_attn_scalar) ** -0.5
#         else:
#             self.scaling = (head_dim) ** -0.5


#     def forward(self, x, mask, cos, sin):
#         b, num_tokens, _ = x.shape

#         check_precision(x, "Input to Attention")

#         # Apply projections
#         queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
#         keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
#         values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

#         # Reshape
#         queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
#         keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
#         values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

#         check_precision(queries, "Queries (Post-Projections)")
#         check_precision(keys, "Keys (Post-Projections)")
#         check_precision(values, "Values (Post-Projections)")

#         # # Optional normalization
#         # if self.q_norm:
#         #     queries = self.q_norm(queries)
#         # if self.k_norm:
#         #     keys = self.k_norm(keys)
#         if self.q_norm:
#             queries = self.q_norm(queries.float()).to(x.dtype)

#         if self.k_norm:
#             keys = self.k_norm(keys.float()).to(x.dtype)

#         # Apply RoPE
#         queries = apply_rope(queries.float(), cos, sin)
#         keys = apply_rope(keys.float(), cos, sin)

#         # Expand K and V to match number of heads
#         keys = keys.repeat_interleave(self.group_size, dim=1)
#         values = values.repeat_interleave(self.group_size, dim=1)
#         check_precision(queries, "Queries (Pre-Attention)")
        
#         check_precision(keys, "Keys (Pre-Attention)")
#         check_precision(values, "Values (Pre-Attention)")

#         # Scale queries
#         queries = queries * self.scaling

#         # Attention
#         attn_scores = torch.matmul(queries.float(),keys.transpose(2, 3).float())
#         # print("Attn Scores Shape: ", attn_scores.shape)
#         # print("Attn Mask Shape: ", mask.shape)
#         check_precision(attn_scores, "Attention Scores (Pre-Softmax)")
#         attn_scores = attn_scores.masked_fill(mask, -1e4) # -torch.inf) causes nan issues in mixed precision
#         attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(queries.dtype)
#         check_precision(attn_weights, "Attention Scores (Post-Softmax)")

#         # print("Context Shape: ", (attn_weights @ values).shape)
#         # print("Context Transpose Shape: ", (attn_weights @ values).transpose(1, 2).shape)
#         # context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
#         context = torch.matmul(attn_weights,values.float()).to(values.dtype).transpose(1, 2).reshape(b, num_tokens, self.d_out)
#         check_precision(context, "Attention Scores (Post-Context)")
#         return self.out_proj(context)

class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False,
        query_pre_attn_scalar=None, dtype=None,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-4)
            self.k_norm = RMSNorm(head_dim, eps=1e-4)
        else:
            self.q_norm = self.k_norm = None
        
        if query_pre_attn_scalar is not None:
            self.scaling = (query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = (head_dim) ** -0.5


    def forward(self, x, mask, cos, sin):

        b, t, _ = x.shape

        # projections
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        # reshape
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # optional qk norm
        if self.q_norm:
            q = self.q_norm(q.float()).to(x.dtype)

        if self.k_norm:
            k = self.k_norm(k.float()).to(x.dtype)

        # RoPE
        q = apply_rope(q.float(), cos, sin).to(x.dtype)
        k = apply_rope(k.float(), cos, sin).to(x.dtype)

        # expand KV heads (GQA)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Scale queries
        queries = queries * self.scaling
        
        # FlashAttention via SDPA
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )

        context = context.transpose(1, 2).reshape(b, t, self.d_out)

        return self.out_proj(context)