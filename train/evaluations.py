import torch
import torch.nn.functional as F

def evaluate_generations(model, 
                         max_new_tokens: int = 30,
                         prompts = [
                             "Once upon a time",
                             "The war is going towards the Germany"
                         ]):
    """Generate QA + NTP outputs and return as dict metric strings."""
    res = {}      
    for idx,prompt in enumerate(prompts):
        res[f"Sample_{idx+1}"] = model.generate(prompt, num_tokens = max_new_tokens)
    return res

def preprocess_logits_for_metrics(logits, labels):
    """
    Reduce memory by only keeping predictions we care about.
    Instead of returning full logits [B, T, V], we return argmax ids [B, T].
    """
    if isinstance(logits, tuple):  # some models return (logits, past_key_values)
        logits = logits[0]
    preds = torch.argmax(logits, dim=-1)
    return preds

def calculate_perplexity(logits, targets, ignore_index=-100):
    # logits: (batch_size, seq_len, vocab_size), targets: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=-1)  # Convert to log probabilities
    log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # Select correct token log probs
    
    # Mask out ignored indices (e.g., padding)
    if ignore_index is not None:
        mask = (targets != ignore_index)
        log_probs = log_probs * mask
        n_tokens = mask.sum()
    else:
        n_tokens = log_probs.numel()
    
    # Compute mean negative log-likelihood and perplexity
    nll = -log_probs.sum() / n_tokens
    perplexity = torch.exp(nll)
    return perplexity

# Example usage
# logits = torch.randn(2, 10, 1000)  # Batch of 2, sequence length 10, vocab size 1000
# targets = torch.randint(0, 1000, (2, 10))
# ppl = calculate_perplexity(logits, targets)
# print(f"Perplexity: {ppl.item()}")   