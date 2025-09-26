import torch.nn as nn, torch.nn.functional as F

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