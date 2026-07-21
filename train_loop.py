import torch, wandb
from tqdm.auto import tqdm

def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, config, eval_loader = None):
    """" This code is designed to run in parallel with the gradscalar at fp16. This will run with DataParallel.
     The commented lines indicate the code without the gradscalar """
    device = config.device
    model.train()
    total_loss = 0

    pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}")
    avg_loss = torch.zeros(50,device = device)
    avg_acc = torch.zeros(50, device = device)
    curr_loss = 0
    acc = 0

    scaler = torch.amp.GradScaler("cuda")
    
    for idx, batch in pbar:
        optimizer.zero_grad(set_to_none=True) 

        # --- Forward pass with AMP ---
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch.get("attention_mask"),
                labels         = batch.get("labels"),    # ensure pad positions are -100, not pad_token_id
            )
            loss   = outputs["loss"].mean()  # .mean is used because each GPU provides it's own loss
            logits = outputs["logits"]
        
        # --- Accuracy (FP32) ---
        labels      = batch["labels"].to(torch.long)
        mask        = labels != torch.zeros(1)  # Masking doesn't make any sense in pre-training with chunking
        predictions = torch.argmax(logits, dim=-1)
        correct     = (predictions == labels) & mask
        acc = (correct.sum().float() / mask.sum().float()) * 100

        # --- Backward ---
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 1.0 is standard for LMs
        # optimizer.step()
        # --- Backward ---
        scaler.scale(loss).backward()
    
        # Unscale before gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()

        slot = idx % 50
        avg_loss[slot] = loss.detach()
        avg_acc[slot] = acc.detach()

        # window grows from 1 to 50, then stays at 50 (rolling)
        window         = min(idx + 1, 50)
        reporting_loss = avg_loss.sum().item() / window
        reporting_acc  = avg_acc.sum().item()  / window
        curr_loss = reporting_loss
        acc = reporting_acc

        if idx % 10 == 0 and idx != 0:
            train_step = idx + epoch * dataloader.total_batches
            wandb.log({
                "train/loss"    : reporting_loss,
                "train/lr"      : optimizer.param_groups[0]["lr"],
                "train/acc"     : reporting_acc,
                "step"          : train_step,
                "tokens_trained": train_step * config.batch_size * config.seq_len,
            })
            if idx % 500 == 0 and idx != 0: # Log generation & evaluate every 500 steps
                model.eval()
                    
                with torch.no_grad():
                    eval_loss = 0
                    for batch in eval_loader:
                        out = model(
                            input_ids  =  batch["input_ids"],
                            attention_mask = batch.get("attention_mask"),
                            labels = batch.get("labels")
                        )
                        eval_loss += out["loss"].mean().item() # .mean() is due to each GPU generates it's own loss.
    
                    wandb.log({
                        "eval_loss": eval_loss / len(eval_loader),
                        "step": idx + epoch * len(dataloader),
                        "generation 1": model.module.generate("The government of France is"),
                        "generation 2": model.module.generate("The war in Europe"),
                        "generation 3": model.module.generate("The construction of"),
                    }
                    )

                if idx % 2000 == 0:
                    # Save checkpoint
                    torch.save(model.module.state_dict(), f"{config.total_params}M_params__checkpoint_{idx}.pt")
                model.train()

        pbar.set_postfix({"loss": f"{curr_loss:.4f}", "acc": f"{acc:.2f}%"})

    return total_loss / len(dataloader)