from accelerate import Accelerator
from dataclasses import asdict
from time import perf_counter
from .evaluations import calculate_perplexity
from .evalute import eval_loop
import torch, wandb
from torch.utils.data import DataLoader


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def init_accelerator(model, train_config, logger=None):
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision="fp16" if train_config.mixed_precision else None,
        log_with="wandb" if logger else None
    )
    if logger and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=train_config.project_name,
            config=asdict(train_config),
            init_kwargs= {"wandb": {"mode": "online",
                                    "dir": train_config.output_dir,}} # W&B online mode can be changed later
            )
        wandb.watch(model, log="all",log_freq=max(50, train_config.logging_steps))
    return accelerator

def accelerate_dataset_wrapper(dataset, collate_fn=None, batch_size = None,
                               shuffle = True, accelerator=None):
    dataloader = dataset
    if type(dataset) is not DataLoader:
        dataloader = DataLoader(dataset, 
                                collate_fn=collate_fn, 
                                batch_size=batch_size, 
                                shuffle=shuffle)
    if accelerator:
        dataloader = accelerator.prepare(dataloader)
        return dataloader

def save_model(model, accelerator = None, out_dir = "./final_model/"):
    model.eval()
    if accelerator:
        accelerator.wait_for_everyone()
        accelerator.save_model(model, out_dir)
    else:
        torch.save(model.state_dict(), out_dir)

def train_step(model, optimizer, lr_scheduler, accelerator, 
               loop, global_step, train_config, train_loader,
                eval_loader=None):
    model.train()

    # Initialize metrics to track
    train_loss = 0
    train_acc = 0

    for batch in train_loader:
        with accelerator.accumulate(model):
            with accelerator.autocast():
                outputs = model(**batch)
                loss = outputs["loss"]
                logits = outputs["logits"]
                labels = batch["labels"]
                accuracy = logits.argmax(dim=-1).view(-1).eq(labels.view(-1)).float().sum()

            train_loss += loss.item()
            train_acc += accuracy.item()

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Update tqdm only when optimizer updates (i.e., global step)
                loop.set_postfix({
                    "loss": train_loss / train_config.gradient_accumulation_steps,
                    "acc": train_acc / (train_config.seq_len * train_config.gradient_accumulation_steps),
                    "lr": lr_scheduler.get_last_lr()[0]
                })
                loop.update(1)

                # Logging
                if global_step % train_config.logging_steps == 0:
                    metrics = {
                        "loss": loss.item(),
                        "accuracy": accuracy.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "grad_norm": get_grad_norm(model),
                        "time": perf_counter() - logger_start_time,
                        "perplexity": calculate_perplexity(logits, labels).item(),
                    }
                    accelerator.log(metrics, step=global_step)
                    accelerator.print(f"Epoch {train_config.current_epoch}, Step {global_step}: {metrics}")
                    logger_start_time = perf_counter()

                # Save model
                if global_step % train_config.save_steps == 0:
                    save_model(model, accelerator, train_config.output_dir + f"/checkpoint-{global_step}")
                    model.train()

                # Evaluation
                if eval_loader is not None and global_step % train_config.eval_steps == 0:
                    metrics = eval_loop(model, eval_loader, train_config)
                    accelerator.log(metrics, step=global_step)
                    accelerator.print(f"Epoch {train_config.current_epoch}, Step {global_step}: {metrics}")
                    model.train()
