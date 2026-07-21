import torch, math, torch.nn as nn, torch.nn.functional as F

def precompute_rope_freqs(head_dim, seq_len, theta=10000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)          # (seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    # x: (B, nh, T, head_dim)
    T = x.size(2)
    hd = x.size(-1)
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)   # (1,1,T,hd/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, seq_len: int, 
                 bias: bool = False, dropout: float = 0.0, rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout, self.resid_dropout = nn.Dropout(dropout), nn.Dropout(dropout)
        self.num_heads, self.d_model, self.dropout = num_heads, d_model, dropout
        self.head_dim = d_model // num_heads

        cos, sin = precompute_rope_freqs(self.head_dim, seq_len, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(seq_len, seq_len))
                                        .view(1, 1, seq_len, seq_len))

    def forward(self, x, attention_mask=None):
        """If attention_mask is not providede default to causal mask by SDPA attention."""
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q, k, v = [i.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for i in (q, k, v)]

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=(attention_mask is None)
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
        
class SwiGLU(nn.Module):
    def __init__(self, d_model:int):
        super().__init__()
        hidden_dim = (int(8 * d_model / 3)//64) * 64 # Round of multiple to 64
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
    
class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.csa = CausalSelfAttention(config.d_model, config.n_head, config.seq_len, config.bias, config.dropout, config.rope_theta)
    self.ff = SwiGLU(config.d_model)
    self.ln1, self.ln2 = [RMSNorm(config.d_model) for _ in range(2)]
  def forward(self, x, attention_mask):
    x = x + self.csa(self.ln1(x), attention_mask)
    return x + self.ff(self.ln2(x))

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.token_embedding_table = nn.Embedding(config.vocab_size, config.d_model)
    self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

    # in GPT.__init__, after self.blocks is built:
    self.apply(self._init_weights)
    for pn, p in self.named_parameters():
        if pn.endswith('c_proj.weight') or pn.endswith('w3.weight'):
            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))
    self.ln = RMSNorm(config.d_model)
    self.out_proj = nn.Linear(config.d_model,config.vocab_size, bias = config.bias)
    # self.drop = nn.Dropout(config.dropout)  ## Not sure about the idea of using dropout.
    self.seq_len = config.seq_len
    # weight tying
    self.token_embedding_table.weight = self.out_proj.weight

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
  def forward(self, input_ids, attention_mask = None, labels = None):
    # input_ids : (Batch_Size, Seq_len)    # labels : (Batch_Size, Seq_len)
    input_ids = self.token_embedding_table(input_ids)

    attention_mask = None # Attention Mask is currently not required in Trail runs. Later it needs to be updated

    for module in self.blocks:   input_ids = module(input_ids, attention_mask)
    logits, loss = self.out_proj(self.ln(input_ids)), None
    if labels != None:
      B, S, V = logits.shape
      loss = F.cross_entropy(logits.view(B*S, V), labels.view(B*S).long(), ignore_index = 0)
    return {"logits":logits, "loss": loss}

  @property
  def device(self):   return next(self.parameters()).device

  @torch.no_grad()
  def generate(self, x: str, tokenizer, num_tokens: int = 100, with_argmax: bool = False):
    """ x: textinput , tokenizer: Huggingface Tokenizer, num_tokens: number of tokens to generate , 
        with_argmax: whether to use argmax or sampling """
    x = torch.tensor( tokenizer.encode(x)[:-1], device = self.device ) # Discard the [SEP] token here.
    for _ in range(num_tokens):
        out = self(x[-self.seq_len:].unsqueeze(0))["logits"]     # Input : (1,seq_len)  Output : (1,seq_len,vocab_size)
        if with_argmax:
            tkn = out[0,-1].argmax()
        else: 
            tkn = torch.multinomial( F.softmax(out[:,-1], dim = -1), num_samples = 1 ).to(device = x.device)[0]
        x = torch.cat( (x, tkn), dim=-1 )
    return tokenizer.decode( x.tolist() )