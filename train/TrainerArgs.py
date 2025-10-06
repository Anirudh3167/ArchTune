from transformers import TrainingArguments

def get_trainer_args(config):
    return TrainingArguments(
                output_dir=config.out_dir,
                overwrite_output_dir=True,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                gradient_accumulation_steps=config.grad_accum,
                num_train_epochs=config.epochs,
                logging_dir="./logs",
                save_strategy="steps",
                eval_strategy="steps",
                eval_steps=config.eval_steps,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                save_total_limit=3,
                learning_rate=config.lr,
                # weight_decay=config.weight_decay,
                warmup_ratio=config.warmup_ratio,
                # lr_scheduler_type="cosine",
                dataloader_num_workers=3,
                save_safetensors=False,
                report_to=[],
            )
        