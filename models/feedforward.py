import torch.nn as nn, torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        nn.init.zeros_(self.c_proj.weight)  # @Grad62304977
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.dropout(self.c_proj(x))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, config.hidden_dim, bias=False)
        self.fc2 = nn.Linear(config.n_embed, config.hidden_dim, bias=False)
        self.fc3 = nn.Linear(config.hidden_dim, config.n_embed, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)


class Expert(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, config.hidden_dim, bias=False)
        self.fc2 = nn.Linear(config.hidden_dim, config.n_embed, bias=False)
    
    def forward(self, x):
        x = self.fc1(x)
        # x = nn.functional.gelu(x, approximate="tanh")
        x = F.relu(x).square()
        return self.fc2(x)
    
class TopKNoisyRouter(nn.Module):
    def __init__(self,n_embed, n_experts, top_k):
        super(TopKNoisyRouter, self).__init__()
        self.topk = top_k
        self.route_linear = nn.Linear(n_embed, n_experts, bias=False)
        self.noise_linear = nn.Linear(n_embed, n_experts, bias=False)
        
    def forward(self, x):
        logits = self.route_linear(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise
        # Pick only the TOP K experts.
        top_k_logits, indices = noisy_logits.topk(self.topk, dim=1)
        # Mask the other values with -inf
        zeroes = torch.full_like(noisy_logits, -float("inf"))
        sparse_logits = zeroes.scatter(-1,indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices
    
class SparseMoE(nn.Module):
    def __init__(self, config):
        super(SparseMoE, self).__init__()
        self.topk = config.k_expts
        self.router = TopKNoisyRouter(config.n_embed, config.n_experts, config.k_expts)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
    
    def forward(self, x):
        gating_output, indices = self.router(x)  # indices: [B, T, top_k]
        final_output = torch.zeros_like(x)

        B, T, D = x.shape
        E = len(self.experts)
        # Sanity check: indices must be in valid range
        assert (indices >= 0).all() and (indices < E).all(), \
            f"Invalid expert index detected. Got max={indices.max()}, expected < {E}"

        flat_x = x.view(-1, D)  # [B*T, D]
        flat_indices = indices.view(-1, self.topk)  # [B*T, top_k]
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))  # [B*T, n_experts]

        for i, expert in enumerate(self.experts):
            # Mask where expert i is in the top_k for each token
            expert_mask = (flat_indices == i).any(dim=-1)  # [B*T]

            if expert_mask.any():
                expert_input = flat_x[expert_mask]
                expert_output = expert(expert_input)  # [num_selected, D]
                gating_scores = flat_gating_output[expert_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output.view(-1, D)[expert_mask] += weighted_output

        return final_output

    # def forward(self, x):
    #     gating_output, indices = self.router(x)
    #     final_output = torch.zeros_like(x)

    #     # Reshape inputs for batch processing
    #     flat_x = x.view(-1, x.size(-1))
    #     flat_gating_output = gating_output.view(-1, gating_output.size(-1))

    #     # Process each expert in parallel
    #     for i, expert in enumerate(self.experts):
    #         # Create a mask for the inputs where the current expert is in top-k
    #         expert_mask = (indices == i).any(dim=-1)
    #         flat_mask = expert_mask.view(-1)

    #         if flat_mask.any():
    #             expert_input = flat_x[flat_mask]
    #             expert_output = expert(expert_input)

    #             # Extract and apply gating scores
    #             gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
    #             weighted_output = expert_output * gating_scores
    #             final_output[flat_mask] += weighted_output.squeeze(1)
        
    #     return final_output