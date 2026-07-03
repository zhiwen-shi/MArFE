import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
from argparse import Namespace
import random 
import math
from model.rdn import make_rdn
from model.resblock import ResBlock
import numpy as np
from utils import make_coord
from model.mlp import MLP


def make_model(args, parent=False):
    return LIIF(args)


class LIIFDecoder(nn.Module):
    def __init__(self, args, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        mlp_in_dim = 64 * 9 + 4
        mlp_out_dim =2  
        mlp_hidden_list = [256, 256, 256]
        self.imnet = MLP(mlp_in_dim, mlp_out_dim, mlp_hidden_list)
    def query_rgb(self, feat, coord, cell=None):

        feat = feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode='nearest',
                                align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, (3, 3), padding=(1, 1)).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble: 
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2  
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst: # 默认step = 1
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift   # shape: [N, C*9, 1, Q]
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]  
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)  

                if self.cell_decode: 
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:  
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas): 
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret


    def forward(self, x, coord, cell, size): #feat, coord, cell [-1,1] --> [-1,1]
        b, c, h ,w = x.shape
        pred = self.query_rgb(x, coord, cell)
        pred = pred.permute(0, 2, 1).view(b, -1, size[0], size[1])

        return pred


class LIIF(nn.Module):  # change for mri liif
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = make_rdn()
        self.decoder = LIIFDecoder(self.args)
    def set_scale(self, scale, scale2):
        self.scale = scale
        self.scale2 = scale2
    def forward(self, inp, bsize = None):  # input: lr, ref_hr, ref_lr, self.args.ref_type, epoch

        inp = inp[0]
        B,C,H,W = inp.shape
        H_hr = round(H*self.scale)
        W_hr = round(W*self.scale2)
        feat = self.encoder((inp-0.5)/0.5) # reset input range from [0,1] to [-1,1]  #input lr bchw 6 2 32 32
        size = [H_hr, W_hr]  # 128*128

        hr_coord = make_coord(size).to("cuda")
        hr_coord = hr_coord.unsqueeze(0).expand(B, -1, -1)  # [B, Q, 2]
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / H_hr
        cell[:, 1] *= 2 / W_hr
        pred = self.decoder(feat, hr_coord, cell, size)

        return pred*0.5+0.5, pred*0.5+0.5 # reset input range from [-1,1] to [0,1]

