from torch.utils.data import DataLoader
from .HyperParamsConfig import Hyperparameters
from .train_loop import train_one_epoch
import wandb, torch

def run_training(config: Hyperparameters, model, optimizer, scheduler, avg_val_loss, train_loader: DataLoader, val_loader: DataLoader):
    """ Requires Kaggle secrets to define wandb api key  as 'wandb_api_key' """   
    # Initialize WandB
    from kaggle_secrets import UserSecretsClient
    def setup_wandb(project_name="Llama_tokenizer_training"):
        user_secrets = UserSecretsClient()
        wandb_key = user_secrets.get_secret("wandb_api_key")
        wandb.login(key=wandb_key)
        return wandb.init(project=project_name)

    run = setup_wandb()
    wandb.config.update(config)

    for epoch in range(config.epochs):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, config, val_loader)
        model.eval()
        wandb.log({
            "epoch/train_loss": avg_train_loss,
            "epoch/val_loss": avg_val_loss,
            "epoch": epoch
        })
        
        # Save checkpoint
        torch.save(model.module.state_dict(), f"{config.total_params}M_params__Final.pt")

    run.finish()