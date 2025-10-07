class MuonScheduler:
    def __init__(self, optimizer, num_training_steps, cooldown_frac=0.1):
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps
        self.cooldown_frac = cooldown_frac
        self.last_step = -1
    
    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


    def get_lr_scale(self, step):
        x = step / self.num_training_steps
        if x < 1 - self.cooldown_frac:
            return 1.0
        else:
            w = (1 - x) / self.cooldown_frac
            return w * 1.0 + (1 - w) * 0.1  # decay to 10% of base LR

    def step(self, step = None):
        if step is not None:
            self.last_step = step
        else:
            self.last_step += 1

        warmup_steps = max(300, int(0.01 * self.num_training_steps))
        frac = min(self.last_step / warmup_steps, 1.0)
        lr_scale = self.get_lr_scale(self.last_step)

        for group in self.optimizer.param_groups:
            if group.get("use_muon", False):
                # Update LR for Muon groups
                group["lr"] = group["initial_lr"] * lr_scale

                # Momentum warmup: 0.85 → 0.95 over first self.num_training_steps steps
                # frac = min(self.last_step / 300.0, 1.0)
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
            else:
                # For Adam groups, you can also scale LR if desired
                group["lr"] = group["initial_lr"] * lr_scale