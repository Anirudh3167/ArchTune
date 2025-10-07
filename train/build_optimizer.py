from .Muon import MuonWithAuxAdam
from .lr_scheduler import MuonScheduler

def build_muon_optimizer(model,cfg):
    """
    cfg must contain (hidden_lr, hidden_momentum, lr, adam_betas, adam_eps, weight_decay)
    """
    hidden_matrix_params = [
        p for n, p in model.blocks.named_parameters()
        # p for n, p in self.model.named_parameters()
        if p.ndim >= 2 and "embed" not in n
    ]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    optimizer = MuonWithAuxAdam([
        # Muon: hidden matrices
        {"params": hidden_matrix_params,
            "use_muon": True,
        "lr": getattr(cfg, "hidden_lr", 0.05),
        "initial_lr": getattr(cfg, "hidden_lr", 0.05),
        "momentum": getattr(cfg, "hidden_momentum", 0.95),
        "weight_decay": getattr(cfg, "weight_decay", 0.0),
            },

        # AdamW: everything else
        {"params": embed_params + scalar_params + head_params,
            "use_muon": False,
        "lr": getattr(cfg, "lr", 0.008),
        "initial_lr": getattr(cfg, "lr", 0.008),
        "betas": getattr(cfg, "adam_betas", (0.8, 0.95)),
        "eps": getattr(cfg, "adam_eps", 1e-10),
        "weight_decay": getattr(cfg, "weight_decay", 0.0),
        }
    ])
    return optimizer

def create_scheduler(num_training_steps: int, optimizer = None):
    """
    Override to use custom Muon scheduler with momentum warmup.
    """
    lr_scheduler = MuonScheduler(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        cooldown_frac=0.45  # 45% training spent on cooldown of lr
        # cooldown_frac=getattr(cfg, "cooldown_frac", 0.1)
    )
    return lr_scheduler