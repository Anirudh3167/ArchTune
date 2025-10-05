from transformers import Trainer
from .Muon import MuonWithAuxAdam

class CustomTrainer(Trainer):
    def create_optimizer(self):
        cfg = self.args
        if self.optimizer is None:
            hidden_matrix_params = [
                p for n, p in self.model.blocks.named_parameters()
                if p.ndim >= 2 and "embed" not in n
            ]
            embed_params = [p for n, p in self.model.named_parameters() if "embed" in n]
            scalar_params = [p for p in self.model.parameters() if p.ndim < 2]
            head_params = [self.model.lm_head.weight]

            self.optimizer = MuonWithAuxAdam([
                # Muon: hidden matrices
                {"params": hidden_matrix_params,
                 "use_muon": True,
                "lr": getattr(cfg, "hidden_lr", 0.05),
                "momentum": getattr(cfg, "hidden_momentum", 0.95),
                "weight_decay": getattr(cfg, "weight_decay", 0.0),
                 },

                # AdamW: everything else
                {"params": embed_params + scalar_params + head_params,
                 "use_muon": False,
                "lr": getattr(cfg, "lr", 0.008),
                "betas": getattr(cfg, "adam_betas", (0.8, 0.95)),
                "eps": getattr(cfg, "adam_eps", 1e-10),
                "weight_decay": getattr(cfg, "weight_decay", 0.0),
                }
            ])
        return self.optimizer