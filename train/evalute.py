from time import perf_counter
from tqdm import tqdm
import torch

from .evaluations import calculate_perplexity, evaluate_generations 

def eval_loop(model, test_datalodaer, train_config):
    eval_start_time = perf_counter()
    model.eval()
    eval_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_perplexity = 0.0
    eval_iterator = test_datalodaer
    if not train_config.disable_eval_tqdm:
        eval_iterator = tqdm(test_datalodaer, total=train_config.num_eval_steps, desc="Evaluating")
    for eval_batch in eval_iterator:
        with torch.no_grad():
            outputs = model(**eval_batch)
            loss = outputs["loss"]
            logits = outputs["logits"]
            labels = eval_batch["labels"].to(model.device)

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
        "eval_time": perf_counter() - eval_start_time
    }
    metrics.update(evaluate_generations(model))

    return metrics