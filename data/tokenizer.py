from transformers import AutoTokenizer

def next_multiple(curr_val: int, multiple: int = 128) -> int:
    return curr_val + (multiple - (curr_val%multiple))


def get_tokenizer(model_name: str = None, 
                  special_tokens: dict = {"pad_token": "<|pad|>",
                          "bos_token": "<|startoftext|>",
                          "unk_token": "<|unk|>"}
                ):
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

    if special_tokens != {}:
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
