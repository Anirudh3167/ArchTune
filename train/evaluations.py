import torch

prompts = ["Once upon a time",
              "The war is going towards the Germany"]
    
def evaluate_generations(model, max_new_tokens: int = 30):
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
