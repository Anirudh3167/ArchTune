import torch, torch.nn as nn, torch.nn.functional as F
from .utils import Rotary, norm, RMSNorm

class CausalSelfAttention(nn.Module):
    def __init__(self, config, is_gemma_model = False):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_attn.weight.data[: config.n_embed].zero_()  # init query proj weights to zero
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        nn.init.zeros_(self.c_proj.weight)  # @Grad62304977
        self.is_gemma_model = is_gemma_model
        if self.is_gemma_model:
            self.q_norm = RMSNorm(config.n_embed // config.n_head, eps=1e-6, bias=False)
            self.k_norm = RMSNorm(config.n_embed // config.n_head, eps=1e-6, bias=False)
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
        if self.is_gemma_model:
            q, k = self.q_norm(q), self.k_norm(k)
        else:
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
