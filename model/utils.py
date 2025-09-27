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

def norm(x: Tensor,eps:float=None):
    return F.rms_norm(x, (x.size(-1),),eps=eps)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        # Gemma3 stores zero-centered weights and uses (1 + weight) during forward
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        # Match HF Gemma3: compute norm in float32, then scale by (1 + w)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale)

        if self.shift is not None:
            out = out + self.shift

        return out


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / max_seq_len) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
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
    


class GemmaRotary(nn.Module):
    def __init__(self, head_dim, theta_base=1_000_000, context_length=4096):
        """
        theta_base = 1_000_000 due to global only Rotary
        """
        super().__init__()
        assert head_dim % 2 == 0, "Embedding dimension must be even"

        # Compute the inverse frequencies
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

        # Generate position indices
        positions = torch.arange(context_length)

        # Compute the angles
        angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

        # Expand angles to match the head_dim
        angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

        # Precompute sine and cosine
        self.cos = torch.cos(angles)
        self.sin = torch.sin(angles)


    def forward(self,x):
        # x: (batch_size, num_heads, seq_len, head_dim)
        device = x.device
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"

        # Split x into first half and second half
        x1 = x[..., : head_dim // 2]  # First half
        x2 = x[..., head_dim // 2 :]  # Second half

        # Adjust sin and cos shapes
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, seq_len, head_dim)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0).to(device)

        # Apply the rotary transformation
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * self.sin)

        # It's ok to use lower-precision after applying cos and sin rotation
        return x_rotated


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
