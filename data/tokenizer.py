# from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast

def next_multiple(curr_val: int, multiple: int = 128) -> int:
    return curr_val + (multiple - (curr_val%multiple))


def get_tokenizer(model_name: str = None, add_special_tokens: bool = True):
    """
    Load a tokenizer with common configs.
    
    Args:
        model_type (str): One of 'gpt2' | 'bert' (currently only 'gpt' supported).
        model_name (str): Optional specific tokenizer name or path.
        add_special_tokens (bool): Whether to ensure <pad>, <eos>, etc. are set.

    Returns:
        tokenizer: HuggingFace tokenizer
    """
    # Default tokenizer for GPT models (can override via model_name)
    # tokenizer_name = model_name or "gpt2"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer = GPT2TokenizerFast()
    if add_special_tokens:
        # Define missing special tokens
        special_tokens = {}
        if tokenizer.pad_token is None:
            special_tokens["pad_token"] = "<|pad|>"
        if tokenizer.eos_token is None:
            special_tokens["eos_token"] = "<|endoftext|>"
        if tokenizer.bos_token is None:
            special_tokens["bos_token"] = "<|startoftext|>"
        if tokenizer.unk_token is None:
            special_tokens["unk_token"] = "<|unk|>"

        # Only add if at least one is missing
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)

    # Optional: round max length to nearest multiple of 128
    target_vocab_size = next_multiple(len(tokenizer), 128)
    extra_tokens = target_vocab_size - len(tokenizer)

    # Step 2: Add [unused] tokens to match the target size
    if extra_tokens > 0:
        unused_tokens = [f"[unused{i}]" for i in range(extra_tokens)]
        tokenizer.add_tokens(unused_tokens)

    # Step 3: (Optional) Confirm updated vocab size
    print(f"Updated vocab size: {len(tokenizer)}")

    return tokenizer
