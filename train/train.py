from accelerate import Accelerator
from .build_optimizer import build_muon_optimizer, create_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evaluations import evaluate_generations, calculate_perplexity
import torch


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train(
        model,
        train_config,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
):
    if train_config.num_epochs is None:
        train_config.num_epochs = 1

    steps_per_epoch = len(train_dataset) // train_config.train_batch_size
    train_config.num_train_steps = steps_per_epoch * train_config.num_epochs

    accelerator = Accelerator(gradient_accumulation_steps=train_config.gradient_accumulation_steps)

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

    global_step = 0
    optimizer.zero_grad()

    # Main training loop
    for epoch in range(train_config.num_epochs):
        model.train()
        
        # Initialize tqdm for global steps
        loop = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}", position=0, leave=True)

        for batch in train_loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    logits = outputs["logits"]
                    labels = batch["labels"]
                    accuracy = logits.argmax(dim=-1).view(-1).eq(labels.view(-1)).float().mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Update tqdm only when optimizer updates (i.e., global step)
                    loop.set_postfix({
                        "loss": loss.item(),
                        "acc": accuracy.item(),
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
                            "perplexity": calculate_perplexity(logits, labels).item(),
                        }
                        accelerator.log(metrics, step=global_step)
                        accelerator.print(f"Epoch {epoch+1}, Step {global_step}: {metrics}")

                    # Save model
                    if global_step % train_config.save_steps == 0:
                        accelerator.wait_for_everyone()
                        model.eval()
                        accelerator.save_model(model, train_config.output_dir + f"/checkpoint-{global_step}")
                        model.train()

                    # Evaluation
                    if eval_loader is not None and global_step % train_config.eval_steps == 0:
                        model.eval()
                        eval_loss = 0.0
                        correct_predictions = 0
                        total_predictions = 0
                        total_perplexity = 0.0

                        for eval_batch in tqdm(eval_loader, total=train_config.num_eval_steps, desc="Evaluating"):
                            with torch.no_grad():
                                outputs = model(**eval_batch)
                                loss = outputs["loss"]
                                logits = outputs["logits"]
                                labels = eval_batch["labels"]

                                eval_loss += loss.item()
                                preds = logits.argmax(dim=-1)
                                correct_predictions += preds.eq(labels).sum().item()
                                total_predictions += labels.numel()
                                total_perplexity += calculate_perplexity(logits, labels).item()

                        avg_loss = eval_loss / train_config.num_eval_steps
                        eval_accuracy = correct_predictions / total_predictions
                        eval_perplexity = total_perplexity / train_config.num_eval_steps

                        metrics = {
                            "eval_loss": avg_loss,
                            "eval_accuracy": eval_accuracy,
                            "eval_perplexity": eval_perplexity,
                        }
                        metrics.update(evaluate_generations(model))

                        accelerator.log(metrics, step=global_step)
                        accelerator.print(f"Epoch {epoch+1}, Step {global_step}: {metrics}")
                        model.train()

        loop.close()

        # Save at end of epoch
        accelerator.wait_for_everyone()
        model.eval()
        accelerator.save_model(model, train_config.output_dir + f"/checkpoint-{global_step}")
        model.train()
