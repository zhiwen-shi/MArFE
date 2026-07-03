import torch
import torch.nn as nn
import numpy as np
import math


class WL1Loss(nn.Module):
    def __init__(self, args):
        super(WL1Loss, self).__init__()
    def forward(self, sr, hr, s):
        error = sr - hr
        loss = torch.einsum('ij,jdhw->idhw', s, error)
        loss = torch.mean(torch.abs(loss))      
        return loss
