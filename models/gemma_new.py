import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.notebook import tqdm
from .utils import top_k_top_p_filtering, RMSNorm, compute_rope_params, check_precision
from .feedforward import FeedForward, SparseMoE
from .attention_layer import GroupedQueryAttention
from .hashembedding import HashEmbeddingLayer

class TransformerBlock(nn.Module):
    def __init__(self, config, attn_type: str, ff_layer_type: str):
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
        if ff_layer_type == 'ffn':
            self.ff = FeedForward(config)
        else:
            self.ff = SparseMoE(config)
        # self.input_layernorm = RMSNorm(config.n_embed, eps=1e-6)
        # self.post_attention_layernorm = RMSNorm(config.n_embed, eps=1e-6)
        # self.pre_feedforward_layernorm = RMSNorm(config.n_embed, eps=1e-6)
        # self.post_feedforward_layernorm = RMSNorm(config.n_embed, eps=1e-6)
        # Reducing the eps for prevent nan in the loss
        self.input_layernorm = RMSNorm(config.n_embed, eps=1e-4)
        self.post_attention_layernorm = RMSNorm(config.n_embed, eps=1e-4)
        self.pre_feedforward_layernorm = RMSNorm(config.n_embed, eps=1e-4)
        self.post_feedforward_layernorm = RMSNorm(config.n_embed, eps=1e-4)

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
        
        check_precision(x, f"Input layer")
        x = self.input_layernorm(x)
        
        check_precision(x, f"Input Layernorm layer")

        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn = self.att(x, attn_mask, cos, sin)
        check_precision(x_attn, f"Post Attn layer")
        x_attn = self.post_attention_layernorm(x_attn)
        check_precision(x_attn, f"Post Attn layernorm layer")
        x = shortcut + x_attn
        check_precision(x, f"Residual Stream after Layer {self.attn_type}")

        # Shortcut connection for feed forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x

class Gemma3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.layer_types is not None and len(config.layer_types) == config.n_layer

        # Main model parameters
        self.tok_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        if config.embedding_type == 'hash':
            self.tok_embedding = HashEmbeddingLayer(config.vocab_size, config.n_embed, config.bucket_size, config.num_hash_functions)

        self.blocks = nn.ModuleList([
            TransformerBlock(config, attn_type, ff_layer_type)for attn_type,ff_layer_type in zip(config.layer_types, config.ff_layers)
        ])

        # self.final_norm = RMSNorm(config.n_embed, eps=1e-6)
        self.final_norm = RMSNorm(config.n_embed, eps=1e-4)
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
        self.register_buffer("cos_local", cos_local.float(), persistent=False)
        self.register_buffer("sin_local", sin_local.float(), persistent=False)
        self.register_buffer("cos_global", cos_global.float(), persistent=False)
        self.register_buffer("sin_global", sin_global.float(), persistent=False)

        self.mask_global, self.mask_local = self._create_masks(config.seq_len, self.device)

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

    def forward(self, input_ids, attention_mask=None, labels=None): #, **kwargs):
        b, t = input_ids.shape  # 't' is the current sequence length
        x = self.tok_embedding(input_ids) * (self.config.n_embed ** 0.5)
        check_precision(x, "Post-Embedding")
        # Slice RoPE params and masks to current sequence length 't'
        # Buffers are (Seq_Len, Head_Dim) or (Seq_Len, Seq_Len)
        cos_global = self.cos_global[:t]
        sin_global = self.sin_global[:t]
        cos_local = self.cos_local[:t]
        sin_local = self.sin_local[:t]
        
        mask_global = self.mask_global[:t, :t].to(x.device)
        mask_local = self.mask_local[:t, :t].to(x.device)

        if attention_mask is not None:
            # Convert 2D -> 4D (1, 1, t, t) for broadcasting
            mask_global = mask_global.unsqueeze(0).unsqueeze(0)
            mask_local = mask_local.unsqueeze(0).unsqueeze(0)
            
            # attention_mask is likely (B, t)
            dataset_mask_hidden = ~attention_mask.to(torch.bool)
            
            # Merge: (1, 1, t, t) | (B, 1, t, t) -> (B, 1, t, t)
            mask_global = mask_global | dataset_mask_hidden.unsqueeze(1)
            mask_local = mask_local | dataset_mask_hidden.unsqueeze(1)
        mask_local = ~mask_local
        mask_global = ~mask_global
        # b, seq_len = input_ids.shape
        # x = self.tok_embedding(input_ids) * (self.config.n_embed ** 0.5)
        # # mask_global, mask_local = self._create_masks(seq_len, x.device)
        # # Get the 2D geometric masks (Seq, Seq)
        # mask_global, mask_local = self.mask_global.to(x.device), self.mask_local.to(x.device)   

        # # 2. Combine with dataset mask if it exists
        # # Dataset mask should be (Batch, Seq_Len) or (Batch, 1, 1, Seq_Len)
        # if attention_mask is not None:
        #     # --- THE FIX: Convert 2D -> 4D ---
        #     # Shape: (1, 1, Seq, Seq)
        #     mask_global = mask_global.unsqueeze(0).unsqueeze(0)
        #     mask_local = mask_local.unsqueeze(0).unsqueeze(0)
        #     # We invert the dataset mask if 1=visible (standard) to 1=hidden
        #     # If your dataset already gives 1=hidden, skip the '~'
        #     dataset_mask_hidden = ~attention_mask.to(torch.bool)
            
        #     # Broadcast and merge using logical OR
        #     mask_global = mask_global | dataset_mask_hidden.unsqueeze(1)
        #     mask_local = mask_local | dataset_mask_hidden.unsqueeze(1)

        for idx,block in enumerate(self.blocks):
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=cos_global,
                sin_global=sin_global,
                cos_local=cos_local,
                sin_local=sin_local,
                # cos_global=self.cos_global,
                # sin_global=self.sin_global,
                # cos_local=self.cos_local,
                # sin_local=self.sin_local,
            )
            check_precision(x, f"Post-Block {idx}")

        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.to(self.device).reshape(-1))
        return {"logits":logits, "loss":loss}

    @property
    def device(self):   return next(self.parameters()).device