from transformers import AutoTokenizer

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<startoftext>",
    "eos_token": "<endoftext>",
    "unk_token": "<unk>",
}

def get_tokenizer(model_name: str = None, special_tokens: dict = SPECIAL_TOKENS):
    """
    Load a tokenizer with common configs.
    
    Args:
        model_name (str): Optional specific tokenizer name or path.
        special_tokens (dict): Adds <pad>, <eos>, etc.

    Returns:
        tokenizer: HuggingFace tokenizer
    """
    # Default tokenizer for GPT models (can override via model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name or "gpt2", use_fast=True)
    tokenizer.add_special_tokens(special_tokens)

    # Step 2: Add [unused] tokens to round vocab size to nearest multiple of 128
    if extra_vocab_needed := (-len(tokenizer)) % 128:
        tokenizer.add_tokens( [f"[unused{i}]" for i in range(extra_vocab_needed)] )

    # Step 3: (Optional) Confirm updated vocab size
    print(f"Updated vocab size: {len(tokenizer)}")

    return tokenizer
