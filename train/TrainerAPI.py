from transformers import Trainer
from .Muon import MuonWithAuxAdam

class CustomTrainer(Trainer):
    def create_optimizer(self):
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
                 "lr": 0.05,        # was 0.005 → now tuned down
                 "momentum": 0.95,
                 "weight_decay": 0},

                # AdamW: embeddings
                {"params": embed_params,
                 "use_muon": False,
                 "lr": 0.008,        # scaled down from 0.0015
                 "betas": (0.8, 0.95),
                 "eps": 1e-10,
                 "weight_decay": 0},

                # AdamW: scalars (bias, norm, etc.)
                {"params": scalar_params,
                 "use_muon": False,
                 "lr": 0.008,
                 "betas": (0.8, 0.95),
                 "eps": 1e-10,
                 "weight_decay": 0},

                # AdamW: output head
                {"params": head_params,
                 "use_muon": False,
                 "lr": 0.008,
                 "betas": (0.8, 0.95),
                 "eps": 1e-10,
                 "weight_decay": 0},
            ])
        return self.optimizer