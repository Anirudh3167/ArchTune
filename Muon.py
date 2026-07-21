import torch

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum_new = momentum * beta + grad * (1 - beta)
    update = grad * (1 - beta) + momentum_new * beta if nesterov else momentum_new
    # update = grad + momentum_new * beta if nesterov else momentum_new
    if update.ndim == 4:  # conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update, momentum_new

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam. Uses AdamW internally
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["initial_lr"] = group.get("initial_lr", group["lr"])
                assert set(group.keys()) == set(["params", "lr", "initial_lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["initial_lr"] = group.get("initial_lr", group["lr"])
                assert set(group.keys()) == set(["params", "lr", "initial_lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None: continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update, state["momentum_buffer"] = muon_update(p.grad.detach(), state["momentum_buffer"], beta=group["momentum"])
                    if group["weight_decay"] != 0:
                        # p.mul_(1 - group["initial_lr"] * group["weight_decay"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                        # p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] != 0:
                        # p.mul_(1 - group["initial_lr"] * group["weight_decay"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
    
class MuonScheduler:
    def __init__(self, optimizer, num_training_steps, cooldown_frac=0.1, warmup_frac=0.05):
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps
        self.cooldown_frac = cooldown_frac
        self.warmup_steps = min(300, int(warmup_frac * num_training_steps))  # actually uses warmup_frac now
        self.last_step = -1

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_lr_scale(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps          # linear warmup 0 -> 1
        x = step / self.num_training_steps
        if x < 1 - self.cooldown_frac:
            return 1.0
        w = (1 - x) / self.cooldown_frac
        return w + (1 - w) * 0.1

    def step(self, step=None):
        self.last_step = step if step is not None else self.last_step + 1
        frac = min(self.last_step / self.warmup_steps, 1.0)
        lr_scale = self.get_lr_scale(self.last_step)
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale
            if group.get("use_muon", False):
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

def build_muon_optimizer(model, cfg):
    # Bug 3 fix: track embed param ids to avoid overlap
    embed_param_ids = {id(p) for n, p in model.named_parameters() if "embed" in n}

    hidden_matrix_params = [
        p for n, p in model.named_parameters()
        if p.ndim >= 2
        and "embed" not in n
        and "out_proj" not in n # Due to weight tying -- not working on small models (in improvement)
        and id(p) not in embed_param_ids
    ]

    embed_params = [
        p for n, p in model.named_parameters() if "embed" in n
    ]

    # Bug 3 fix: exclude embed params from scalar collection
    scalar_params = [
        p for n, p in model.named_parameters()
        if p.ndim < 2 and id(p) not in embed_param_ids
    ]

    optimizer = MuonWithAuxAdam([
        {
            "params": hidden_matrix_params,
            "use_muon": True,
            "lr": getattr(cfg, "hidden_lr", 0.002),
            "initial_lr": getattr(cfg, "hidden_lr", 0.002),
            "momentum": getattr(cfg, "hidden_momentum", 0.95),
            "weight_decay": getattr(cfg, "weight_decay", 0.1),
        },
        {
            "params": embed_params + scalar_params,
            "use_muon": False,
            "lr": getattr(cfg, "lr", 8e-3),
            "initial_lr": getattr(cfg, "lr", 8e-3),
            "betas": getattr(cfg, "adam_betas", (0.9, 0.95)),
            "eps": getattr(cfg, "adam_eps", 1e-8),
            "weight_decay": getattr(cfg, "weight_decay", 0.1),
        }
    ])
    return optimizer

def create_scheduler(num_training_steps: int, optimizer: MuonWithAuxAdam):
    lr_scheduler = MuonScheduler(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        cooldown_frac=0.15,      # bug 4 fix: 0.1 not 0.45
        warmup_frac=0.05,       # scheduler bug 1 fix: explicit warmup
    )
    return lr_scheduler

def create_muon_optimizer_and_scheduler(model, cfg, num_training_steps: int):
    """"A wrapper that creates Muon optimizer and scheduler"""
    optimizer = build_muon_optimizer(model, cfg)
    lr_scheduler = create_scheduler(num_training_steps, optimizer)
    return optimizer, lr_scheduler