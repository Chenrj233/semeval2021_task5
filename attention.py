from torch import nn
import torch
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.):
        super(SelfAttention, self).__init__()
        self.getscore = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input):
        batch_size, max_len, layer, embedding = input.size(0), input.size(1), input.size(2), input.size(3)
        input = self.dropout(input)
        scores = self.getscore(input.contiguous().view(-1, embedding)).view(batch_size, max_len, layer)
        scores = F.softmax(scores, dim=-1)
        output = scores.unsqueeze(3).expand_as(input).mul(input).sum(dim=2)
        return output