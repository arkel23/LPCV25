import torch
from torch import nn
from einops import reduce
from einops.layers.torch import Reduce, Rearrange


class Head(nn.Module):
    def __init__(self, classifier, hidden_size, num_classes, bsd=True, model_name=None):
        super().__init__()

        if 'tresnet_m_mod' in model_name:
            self.clf_pool = True
            self.head = nn.Linear(int(hidden_size * 1.5), num_classes)

        elif classifier == 'cls':
            self.head = nn.Linear(hidden_size, num_classes)
        else:
            self.head_pool = Reduce('b s d -> b d', 'mean')
            self.head = nn.Linear(hidden_size, num_classes)

        if not bsd:
            self.rearrange = Rearrange('b d h w -> b (h w) d')

    def forward(self, x):
        # x shape: B (batch size), S (sequence length), D (hidden dim size) or B, D, H, W
        if hasattr(self, 'clf_pool'):
            x1, x2 = x
            x1 = reduce(x1, 'b d h w -> b d', 'mean')
            x2 = reduce(x2, 'b d h w -> b d', 'mean')
            x = torch.cat([x1, x2], dim=-1)
            x = self.head(x)
            return x

        if hasattr(self, 'rearrange'):
            x = self.rearrange(x)

        if hasattr(self, 'head_pool'):
            x = self.head_pool(x)
            x = self.head(x)
        elif hasattr(self, 'head'):
            x = self.head(x[:, 0, :])

        return x
