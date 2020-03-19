import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(F.relu(out))
        return out          #F.sigmoid(out)

def binary_cross_entropy_with_logits(input, target, pos_weight=1, size_average=True, reduce=True):
    """ calc binary cross entropy with logits
    Parameters
    ----------
    input : 1-D FloatTensor
        logits
    target : 1-D LongTensor
        0 or 1 indicator for binary class
    pos_weight : int, optional
        Unbalanced data handling by using weighted cross entropy loss
    size_average : bool
        If it is false, this func returns 1-D vector
    Returns
    -------
    loss : 0-D or 1-D FloatTensor
        binary cross entropy (averaged if size_average==True
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    l = 1 + (pos_weight - 1) * target
    loss = input - input * target + l * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()
