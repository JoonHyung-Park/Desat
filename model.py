import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        #self.fc3 = nn.Linear(8,1)
        self.dropout = nn.Dropout(0.6)
        #self.dropout2 = nn.Dropout(0.7)
    def forward(self, x):
        out = self.dropout(F.relu(self.fc1(x)))
        #out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc2(out)
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

class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        target =target.long()
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
