import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.notebook import tqdm
from .utils import top_k_top_p_filtering, norm, RMSNorm
from .attention_layer import CausalSelfAttention
from .mlp_layer import FeedForward

class GemmaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = CausalSelfAttention(config, is_gemma_model = True)
        self.ff = FeedForward(config)

        self.pre_attention_norm = RMSNorm(config.n_embed, eps=1e-6, bias=False)
        self.post_attention_norm = RMSNorm(config.n_embed, eps=1e-6, bias=False)
        self.pre_mlp_norm = RMSNorm(config.n_embed, eps=1e-6, bias=False)
        self.post_mlp_norm = RMSNorm(config.n_embed, eps=1e-6, bias=False)

    def forward(self,x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.pre_attention_norm(x)
        x_attn = self.att(x)
        x_attn = self.post_attention_norm(x)
        x = shortcut + x_attn

        # Shortcut connection for feed forward block
        shortcut = x
        x_ffn = self.pre_mlp_norm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_mlp_norm(x_ffn)
        x = shortcut + x_ffn
        return x


class Gemma(nn.Module):
    def __init__(self, config, tokenizer):
        """
        Args:
            config: Config
            tokenizer: Tokenizer
        """
        super().__init__()
        self.vocab_size = config.vocab_size
        self.token_embedding_table = nn.Embedding(self.vocab_size, config.n_embed)
        self.blocks = nn.ModuleList([GemmaBlock(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embed,self.vocab_size, bias = False)
        self.final_norm = RMSNorm(config.n_embed, eps=1e-6, bias=False)
        # self.token_embedding_table.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
        nn.init.zeros_(self.lm_head.weight)  # @Grad62304977
        
        self.tokenizer = tokenizer
        self.config = config
        
        # report number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        print("Internal vocab size is set to: ", self.vocab_size)
        print("number of parameters: %.2fM" % (num_params/1e6,))
    
    def forward(self, input_ids, labels = None, **kwargs):
        # x : (Batch_Size, Seq_len)    # targets : (Batch_Size, Seq_len)
        B, S = input_ids.shape
        x = self.token_embedding_table(input_ids) * (self.config.n_embed**0.5)
        # x = norm(x,eps=1e-6)
        for module in self.blocks:   x = module(x)
        x = self.final_norm(x)
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