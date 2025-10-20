from .utils import init_accelerator, save_model, train_step
from .build_optimizer import build_muon_optimizer, create_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import perf_counter

def train(
        model,
        train_config,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        logger: bool | None = None,
):
    if train_config.num_epochs is None:
        train_config.num_epochs = 1
    if train_config.gradient_accumulation_steps is None:
        train_config.gradient_accumulation_steps = 1

    steps_per_epoch = len(train_dataset) // (train_config.train_batch_size * train_config.gradient_accumulation_steps)
    train_config.num_train_steps = steps_per_epoch * train_config.num_epochs

    accelerator = init_accelerator(train_config, logger)

    optimizer = build_muon_optimizer(model, train_config)
    lr_scheduler = create_scheduler(train_config.num_train_steps, optimizer)

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    train_loader = accelerator.prepare(
        DataLoader(train_dataset, collate_fn=data_collator, batch_size=train_config.train_batch_size, shuffle=True)
    )
    eval_loader = None
    if eval_dataset is not None:
        train_config.num_eval_steps = len(eval_dataset) // train_config.eval_batch_size
        eval_loader = accelerator.prepare(
            DataLoader(eval_dataset, collate_fn=data_collator, batch_size=train_config.eval_batch_size)
        )

    overall_train_start_time = perf_counter()
    global_step = 0

    model.train()
    # Main training loop
    for epoch in range(train_config.num_epochs):
        
        # Initialize tqdm for global steps
        loop = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}", position=0, leave=True)
        logger_start_time = perf_counter()