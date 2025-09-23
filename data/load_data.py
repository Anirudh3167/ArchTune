from datasets import Dataset, load_dataset
from .registry import SUPPORTED_DATASETS


def load_data(
    dataset_name: str,
    split: str = "train",
    num_samples: int = None,
    streaming: bool = False,
    **kwargs
) -> Dataset:
    """
    Load a dataset by name from registry or Hugging Face.
    Supports:
    - Hugging Face datasets (via registry or directly)
    - Optional streaming mode (requires num_samples)
    - Optional subset selection
    """

    if streaming and num_samples is None:
        raise ValueError("num_samples must be specified when streaming=True")

    # Check if dataset is supported via registry
    if dataset_name in SUPPORTED_DATASETS:
        config = SUPPORTED_DATASETS[dataset_name]

        load_args = {
            "path": config["url"],
            "split": split,
            "streaming": streaming,
            **kwargs
        }

        # Include subset if provided
        if "subset" in config:
            load_args["name"] = config["subset"]

        ds = load_dataset(**load_args)

    else:
        # Fallback: load from Hugging Face hub directly
        print(f"[Info] Dataset '{dataset_name}' not in registry. Loading from Hugging Face...")
        ds = load_dataset(dataset_name, split=split, streaming=streaming, **kwargs)

    # Apply sampling if needed
    if num_samples is not None and streaming:
        # Manual iteration (streaming mode returns iterator)
        ds = Dataset.from_list([x for _, x in zip(range(num_samples), ds)])

    elif num_samples is not None:
        # Non-streaming: use select
        ds = ds.select(range(min(num_samples, len(ds))))

    return ds
