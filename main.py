from .utils import get_tokenizer_with_increased_vocab
from .load_data import MemmapTokenDataset, DataCollator
from .HyperParamsConfig import Hyperparameters
from .gpt_model_with_rope import GPT
from .Muon import create_muon_optimizer_and_scheduler
from .train_wrapper import run_training
from torch.utils.data import DataLoader
import torch

def run_pipeline(train_path: str, val_path: str, custom_config: Hyperparameters | None = None):
    # 1. Prepare tokenizer and config
    tokenizer = get_tokenizer_with_increased_vocab()
    config = Hyperparameters() if custom_config is None else custom_config
    
    # 2. Prepare datasets
    train_dataset = MemmapTokenDataset(train_path,seq_len=config.seq_len)
    val_dataset = MemmapTokenDataset(val_path,seq_len=config.seq_len)
    collator = DataCollator(seq_len=config.seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=42,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=42,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )

    # 3. Load model
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(config.device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 4. Create optimizer and scheduler
    optimizer, lr_scheduler = create_muon_optimizer_and_scheduler(model, config, len(train_loader) // config.batch_size)

    # 5. Run training
    run_training(config, model, optimizer, lr_scheduler, train_loader, val_loader)