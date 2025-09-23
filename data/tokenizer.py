from transformers import AutoTokenizer

def next_multiple(curr_val: int, multiple: int = 128) -> int:
    return curr_val + (multiple - (curr_val%multiple))


def get_tokenizer(model_type: str = "gpt2", model_name: str = None, add_special_tokens: bool = True):
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
    tokenizer_name = model_name or "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if add_special_tokens:
        # Handle missing pad token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

        # Set default special tokens if missing
        special_tokens = {
            "eos_token": tokenizer.eos_token or "<|endoftext|>",
            "bos_token": tokenizer.bos_token or "<|startoftext|>",
            "unk_token": tokenizer.unk_token or "<|unk|>",
            "pad_token": tokenizer.pad_token
        }

        tokenizer.add_special_tokens(special_tokens)

    # Optional: round max length to nearest multiple of 128
    if tokenizer.vocab_size and tokenizer.vocab_size < 100_000:  # not "longformer" style
        tokenizer.vocab_size = next_multiple(tokenizer.vocab_size, 128)

    return tokenizer
