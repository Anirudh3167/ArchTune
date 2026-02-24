# from accelerate import Accelerator
# from .build_optimizer import build_muon_optimizer, create_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evaluations import calculate_perplexity
# from dataclasses import asdict
from time import perf_counter
from .evalute import eval_loop
from .utils import save_model, init_accelerator, accelerate_dataset_wrapper
from .build_optimizer import build_muon_optimizer, create_scheduler

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.norm(2)  # use p.grad directly, not .data
            total_norm += param_norm.pow(2)
    total_norm = total_norm**0.5
    return total_norm

def init_train_config_vars(train_config, train_dataset, eval_dataset = None):
    if train_config.num_epochs is None:
        train_config.num_epochs = 1
    if train_config.gradient_accumulation_steps is None:
        train_config.gradient_accumulation_steps = 1

    steps_per_epoch = len(train_dataset) // (train_config.train_batch_size * train_config.gradient_accumulation_steps)
    train_config.num_train_steps = steps_per_epoch * train_config.num_epochs
    train_config.steps_per_epoch = steps_per_epoch 

    if eval_dataset:
        train_config.num_eval_steps = len(eval_dataset) // train_config.eval_batch_size


def prepare_datasets(train_dataset, eval_dataset = None, collate_fn = None, 
                     train_config = None):
    train_loader = accelerate_dataset_wrapper(train_dataset, collate_fn=collate_fn, 
                                              batch_size=train_config.train_batch_size)

    eval_loader = None
    if eval_dataset:
        eval_loader = accelerate_dataset_wrapper(eval_dataset, collate_fn=collate_fn,
                                                 batch_size=train_config.eval_batch_size)
    return train_loader, eval_loader    

def train(
        model,
        train_config,
        optimizer = None,
        lr_scheduler = None,
        accelerator = None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        logger: bool | None = None,
):
    init_train_config_vars(train_config, train_dataset, eval_dataset)

    if not accelerator:
        accelerator = init_accelerator(model, train_config, logger)

    if not optimizer:
        optimizer = build_muon_optimizer(model, train_config)
    
    if not lr_scheduler:
        lr_scheduler = create_scheduler(train_config.num_train_steps, optimizer)

    # model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    train_loader, eval_loader = prepare_datasets(train_dataset, eval_dataset, collate_fn=data_collator,
                     train_config = train_config)
    overall_train_start_time = perf_counter()
    global_step = 0

    model.train()
    # Main training loop
    for epoch in range(train_config.num_epochs):
        
        # Initialize tqdm for global steps
        loop = tqdm(total=train_config.steps_per_epoch, desc=f"Epoch {epoch + 1}", position=0, leave=True)
        # Initialize metrics to track
        logger_start_time = perf_counter()
        train_loss = 0
        train_acc = 0
        for batch in train_loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    logits = outputs["logits"]
                    labels = batch["labels"]
                    accuracy = logits.argmax(dim=-1).view(-1).eq(labels.view(-1)).float().mean()

                train_loss += loss.item()
                train_acc += accuracy.item()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Update tqdm only when optimizer updates (i.e., global step)
                    overall_loss = train_loss / train_config.gradient_accumulation_steps
                    overall_acc = train_acc / train_config.gradient_accumulation_steps
                    loop.set_postfix({
                    "loss": overall_loss,
                    "acc": overall_acc,
                    "lr": lr_scheduler.get_last_lr()[0]
                    })

                    train_loss = 0
                    train_acc = 0
                    loop.update(1)

                    # Logging
                    if global_step % train_config.logging_steps == 0:
                        metrics = {
                            "loss": overall_loss,
                            "accuracy": overall_acc,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "grad_norm": get_grad_norm(model),
                            "time": perf_counter() - logger_start_time,
                            "perplexity": calculate_perplexity(logits, labels).item(),
                        }
                        accelerator.log(metrics, step=global_step)
                        accelerator.print(f"Epoch {epoch+1}, Step {global_step}: {metrics}")
                        logger_start_time = perf_counter()

                    # Save model
                    if global_step % train_config.save_steps == 0:
                        save_model(model, accelerator, train_config.output_dir + f"/checkpoint-{global_step}")
                        model.train()

                    # Evaluation
                    if eval_loader is not None and global_step % train_config.eval_steps == 0:
                        metrics = eval_loop(model, eval_loader, train_config)
                        accelerator.log(metrics, step=global_step)
                        accelerator.print(f"Epoch {epoch+1}, Step {global_step}: {metrics}")
                        model.train()

        loop.close()

    # Save at end of training
    save_model(model, accelerator, train_config.output_dir + f"/checkpoint-{global_step}")
    accelerator.end_training()
    print(f"Total training time: {perf_counter() - overall_train_start_time}")
