# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# import time
# from thop import profile

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import common
from model.rdn import make_rdn
from model.resblock import ResBlock

# import common
# from rdn import make_rdn
# from resblock import ResBlock


def make_model(args, parent=False):
    return LIIF(args)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = 64
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }['B']

        self.out_dim = G0

        self.sub_mean = nn.Identity()
        self.add_mean = nn.Identity()

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(2, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])


    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        return x

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class LIIF(nn.Module):

    def __init__(self, args,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        # self.encoder = make_rdn()
        self.encoder = RDN(args)

        imnet_in_dim = self.encoder.out_dim
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2 # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(in_dim = imnet_in_dim, out_dim=2, hidden_list=[256, 256, 256, 256]) 

        self.mixer = nn.Conv2d(64*2, 64, 1, padding=0, stride=1)

    def set_scale(self, scale, scale2):
        self.scale = scale
        self.scale2 = scale2

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            # print(F.grid_sample(feat, coord.flip(-1).unsqueeze(1),mode='nearest', align_corners=False).shape,ret.shape)  None
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]) #16,576,48,48


        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2 #1/48
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        # print(make_coord(feat.shape[-2:], flatten=False).shape,feat_coord.shape) #48,48,2   2,48,48   1,2,48,48   16,2,48,48

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone() #16,2304,2
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1) #16,2304,576   16,576,1,2304
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1) #16,2304,2
                rel_coord = coord - q_coord #16,2304,2
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1] #rel_coord 16,2304,2
                inp = torch.cat([q_feat, rel_coord], dim=-1) #16,2304,578

                if self.cell_decode:
                    rel_cell = cell.clone()
                    # print('1',rel_cell.shape) #16,2304,2
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1] #16,2304,2
                    inp = torch.cat([inp, rel_cell], dim=-1) #16,2304,580

                bs, q = coord.shape[:2] #16,2304
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1) #36864,3   16,2304,3
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1]) #16,2304
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0) #16,2304
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1) #16,2304,3
        return ret

    def forward(self, inp, bsize = None): ##16,3,48,48  16,2304,2  16,2304,2
        ref = inp[2]
        inp = inp[0]
        feat = self.encoder((inp-0.5)/0.5)
        # self.feat = self.encoder((inp-0.5)/0.5)
        with torch.no_grad():
            ref = self.encoder((ref-0.5)/0.5)
        feat = torch.cat([feat,ref],1)
        self.feat = self.mixer(feat)  # multi_constract fusion


        B,C,H,W = inp.shape
        H_hr = round(H*self.scale)
        W_hr = round(W*self.scale2)
        coord = make_coord((H_hr,W_hr)).repeat(B,1,1).to(inp.device)
        cell = torch.ones_like(coord).to(inp.device)
        cell[:, :, 0] *= 2 / H_hr
        cell[:, :, 1] *= 2 / W_hr
        if bsize is not None:
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = self.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        else:
            pred = self.query_rgb(coord, cell)
        pred = pred.view(B,H_hr,W_hr,2).permute(0, 3, 1, 2).contiguous()

        return pred*0.5+0.5, pred*0.5+0.5


# # cal para
# if __name__ == '__main__':

#     class Args:
#         def __init__(self):
#             self.scale = [2]  
#             # self.scale2 = 2
#             self.rgb_range = 1

#     args = Args()
#     model = LIIF(args)  # 2x or 4x or 6x
#     model.set_scale(scale=6, scale2=6)

#     model.eval()
#     model.cuda()

#     inp = torch.randn(1,2,64,64).cuda()
#     refhr = torch.randn(1,2,128,128).cuda()  # 1,2,128,128  or 1,2,256,256
#     reflr = torch.randn(1,2,64,64).cuda()
#     input = (inp, refhr, reflr)
    
#     bufer_infer = 0
#     for i in range(100):
#         # print("inference time:", i)
#         time_start = time.time()
#         out = model(input)
#         # print(out.shape)
#         time_end = time.time()
#         infer_time_ms = (time_end - time_start) * 1000  # 秒转毫秒
#         bufer_infer += infer_time_ms

#     avg_infer_time = bufer_infer / 100
#     print(f'Avg Inference time: {avg_infer_time:.2f} ms')

#     flops, params = profile(model, inputs=(input,))
#     print('Model:{:.2f} GFLOPs and {:.2f}M parameters'.format(flops*2 / 1e9, params / 1e6))
