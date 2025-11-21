import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from transunet.configs import ModelConfig

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        ed = config.hidden_size
        self.fc1 = nn.Linear(ed, config.mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.mlp_dim, ed)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        ed = config.hidden_size
        nh = config.num_heads
        assert ed % nh == 0, "hidden_size must be divisible by num_heads"

        self.ed, self.nh = ed, nh
        self.hd = ed // nh

        self.q = nn.Linear(ed, ed)
        self.k = nn.Linear(ed, ed)
        self.v = nn.Linear(ed, ed)
        self.o = nn.Linear(ed, ed)

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.q(x).view(B, N, self.nh, self.hd).transpose(1, 2)
        K = self.k(x).view(B, N, self.nh, self.hd).transpose(1, 2)
        V = self.v(x).view(B, N, self.nh, self.hd).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hd)
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, V)

        ctx = ctx.transpose(1, 2).reshape(B, N, self.ed)
        return self.o(ctx)


# ---------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------
class TransBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        ed = config.hidden_size
        self.attn = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(ed)
        self.mlp = MLP(config)
        self.norm2 = nn.LayerNorm(ed)

    def forward(self, x):
        x = x + self.attn(x)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


# ---------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransBlock(config) for _ in range(config.num_transformer_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x