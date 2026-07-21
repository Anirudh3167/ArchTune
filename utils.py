from transformers import AutoTokenizer

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<startoftext>",
    "eos_token": "<endoftext>",
    # Bot and Sep token can be used later when the model is able to understand chat format.
    # "bot_token": "<bot>",  # Marks the start of bot message
    # "sep_token": "<sep>",  # For chatsequence seperation
    "unk_token": "<unk>",
}
VOCAB_SIZE_MULTIPLE = 128

def load_hf_tokenizer(model_name: str | None = None, special_tokens: dict = SPECIAL_TOKENS):
    """ GPT2 tokenizer is default, can override via model_name """
    tokenizer = AutoTokenizer.from_pretrained(model_name or "gpt2", use_fast=True)
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def add_vocab_to_nearest_multiple(tokenizer, multiple = VOCAB_SIZE_MULTIPLE):
    """ In-place update for Tokenizer object. Add tokens to round vocab size to next multiple of 128"""
    if extra_vocab_needed := (-len(tokenizer)) % multiple:
        tokenizer.add_tokens( [f"[unused{i}]" for i in range(extra_vocab_needed)] )
    print(f"Updated vocab size to next multiple of 128: {len(tokenizer):,}")

def get_tokenizer_with_increased_vocab(model_name: str | None = None, special_tokens: dict = SPECIAL_TOKENS):
    """
    Load a tokenizer with common configs.
    
    Args:
        model_name (str): Optional specific tokenizer name or path. Defaults to "gpt2".
        special_tokens (dict): Adds <pad>, <eos>, etc.

    Returns:
        tokenizer: HuggingFace tokenizer
    """
    tokenizer = load_hf_tokenizer(model_name, special_tokens)
    add_vocab_to_nearest_multiple(tokenizer)
    return tokenizer
