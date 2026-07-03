# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# import time
# from thop import profile

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


# import common
# from rdn import make_rdn
# from resblock import ResBlock

def make_model(args, parent=False):
    return DUALRef(args)
 
class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)
    
# def patch_norm_2d(x, kernel_size=3):
#     mean = F.avg_pool2d(x, kernel_size=kernel_size, padding=kernel_size//2)
#     mean_sq = F.avg_pool2d(x**2, kernel_size=kernel_size, padding=kernel_size//2)
#     var = mean_sq - mean**2
#     return (x-mean)/(var + 1e-6)
    
class Fourier_reparam_linear(nn.Module):
    def __init__(self,in_features,out_features,high_freq_num,low_freq_num,phi_num,alpha):
        super(Fourier_reparam_linear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num =high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha=alpha
        self.bases=self.init_bases()
        self.lamb=self.init_lamb()
        self.bias=nn.Parameter(torch.Tensor(self.out_features,1),requires_grad=True)
        self.init_bias()

    def init_bases(self):
        phi_set=np.array([2*math.pi*i/self.phi_num for i in range(self.phi_num)])
        high_freq=np.array([i+1 for i in range(self.high_freq_num)])
        low_freq=np.array([(i+1)/self.low_freq_num for i in range(self.low_freq_num)])
        if len(low_freq)!=0:
            T_max=2*math.pi/low_freq[0]
        else:
            T_max=2*math.pi/min(high_freq) # 取最大周期作为取点区间  Tmax = 2*pi/min_freq
        points=np.linspace(-T_max/2,T_max/2,self.in_features)
        bases=torch.Tensor((self.high_freq_num+self.low_freq_num)*self.phi_num,self.in_features)
        i=0
        for freq in low_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        for freq in high_freq:
            for phi in phi_set:
                base=torch.tensor([math.cos(freq*x+phi) for x in points])
                bases[i,:]=base
                i+=1
        bases=self.alpha*bases
        bases=nn.Parameter(bases,requires_grad=False)
        return bases

    
    def init_lamb(self):
        self.lamb=torch.Tensor(self.out_features,(self.high_freq_num+self.low_freq_num)*self.phi_num)
        with torch.no_grad():
            m=(self.low_freq_num+self.high_freq_num)*self.phi_num
            for i in range(m):
                dominator=torch.norm(self.bases[i,:],p=2)
                self.lamb[:,i]=nn.init.uniform_(self.lamb[:,i],-np.sqrt(6/m)/dominator,np.sqrt(6/m)/dominator)
        self.lamb=nn.Parameter(self.lamb,requires_grad=True)
        return self.lamb

    def init_bias(self):
        with torch.no_grad():
            nn.init.zeros_(self.bias)
        
    def forward(self,x): # x: B, C, H, W
        B, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)
        weight=torch.matmul(self.lamb,self.bases)
        output=torch.matmul(x_reshaped,weight.transpose(0,1))
        output=output+self.bias.T
        output=output.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return output

# class qkv_attn(nn.Module):  # simple multi-head self-attention code form hiif
#     def __init__(self, midc, heads):
#         super().__init__()

#         self.headc = midc // heads
#         self.heads = heads
#         self.midc = midc

#         self.qkv_proj = nn.Linear(midc, midc * 3, bias=True)

#         self.kln = nn.LayerNorm(self.headc)
#         self.vln = nn.LayerNorm(self.headc)
#         self.sm = nn.Softmax(dim=-1)

#         self.proj1 = nn.Linear(midc, midc)
#         self.proj2 = nn.Linear(midc, midc)

#         # self.proj_drop = nn.Dropout(0.)

#         self.act = nn.GELU()

#     def forward(self, x_inp):
#         b,c,h,w = x_inp.shape
#         x = x_inp.permute(0,2,3,1).view(b, -1, c)  # bchw --> b hw c
#         B, HW, C = x.shape
#         bias = x

#         qkv = self.qkv_proj(x).reshape(B, HW, self.heads, 3 * self.headc)
#         qkv = qkv.permute(0, 2, 1, 3)
#         q, k, v = qkv.chunk(3, dim=-1) # B, heads, HW, headc

#         k = self.kln(k)
#         v = self.vln(v)

#         v = torch.matmul(k.transpose(-2, -1), v) / (HW)
#         # v = self.sm(v)
#         v = torch.matmul(q, v)
#         v = v.permute(0, 2, 1, 3).reshape(B, HW, C)

#         ret = v + bias
#         bias = self.proj2(self.act(self.proj1(ret))) + bias
#         out  = bias.view(b, h, w, c).permute(0,3,1,2) # b hw c -->  bchw

#         return out

# for Galerkin attention code from HiNoTE
    
class LayerNorm(nn.Module):
    """
    Custom Layer Normalization layer.

    Args:
        d_model (int): The dimensionality of the input features.
        eps (float, optional): Small constant to prevent division by zero. Default is 1e-5.
    """
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        Forward pass for layer normalization.

        Args:
            x (torch.Tensor): Input tensor with shape (..., d_model).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

class Galerkin_attn(nn.Module):
    """
    Galerkin multi-head attention layer.

    Args:
        midc (int): Number of intermediate channels.
        heads (int): Number of attention heads.
        act (nn.Module): Activation function to use.
    """
    def __init__(self, midc, heads):
        super(Galerkin_attn, self).__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1, bias=False)
        self.o_proj1 = nn.Conv2d(midc, midc, 1, bias=False)
        self.o_proj2 = nn.Conv2d(midc, midc, 1, bias=False)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()

    def forward(self, x):
        """
        Forward pass for the attention layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with the same shape as input after applying attention.
        """
        B, C, H, W = x.shape
        bias = x
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 3 * self.headc) # [B, 3C, H, W] --> [ B, heads, HW, 3*headc]
        qkv = qkv.permute(0, 2, 1, 3) #[ B, heads, HW, 3*headc]
        q, k, v = qkv.chunk(3, dim=-1) #[ B, heads, HW, headc]

        k = self.kln(k) #layer norm
        v = self.vln(v)
        v = torch.matmul(k.transpose(-2, -1), v) / (H * W) # [B, heads, headc, headc]
        v = torch.matmul(q, v) ## [B, heads, HW, headc]

        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)
        ret = v.permute(0, 3, 1, 2) + bias  # → [B, C, H, W] + [B, C, H, W]
        bias = self.o_proj2(self.act1(self.o_proj1(ret))) + bias
        return self.act2(bias)
    

class Galerkin_cross_attn_single(nn.Module):  # 保留单向输入 输出融合特征
    """
    Galerkin multi-head attention layer with cross-attention.

    Args:
        midc (int): Number of intermediate channels.
        heads (int): Number of attention heads.
        act (nn.Module): Activation function to use.
    """
    def __init__(self, midc, heads):
        super(Galerkin_cross_attn, self).__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1, bias=False)
        self.o_proj1 = nn.Conv2d(midc, midc, 1, bias=False)
        self.o_proj2 = nn.Conv2d(midc, midc, 1, bias=False)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()

    def forward(self, x, x_ref):
        """
        Forward pass for the attention layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, height, width).
            x_ref (torch.Tensor): Reference input tensor with shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with the same shape as input after applying attention.
        """
        B, C, H, W = x.shape
        B_ref, C_ref, H_ref, W_ref = x_ref.shape
        bias = x
        bias_ref = x_ref

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 3 * self.headc) # [B, 3C, H, W] --> [ B, heads, HW, 3*headc]
        qkv = qkv.permute(0, 2, 1, 3) #[ B, heads, HW, 3*headc]
        q, k, v = qkv.chunk(3, dim=-1) #[ B, heads, HW, headc]

        qkv_ref = self.qkv_proj(x_ref).permute(0, 2, 3, 1).reshape(B_ref, H_ref * W_ref, self.heads, 3 * self.headc)
        qkv_ref = qkv_ref.permute(0, 2, 1, 3)
        q_ref, k_ref, v_ref = qkv_ref.chunk(3, dim=-1)


        k = self.kln(k) #layer norm
        v = self.vln(v)
        k_ref = self.kln(k_ref) #layer norm
        v_ref = self.vln(v_ref)

        v = torch.matmul(k_ref.transpose(-2, -1), v) / (H * W) # [B, heads, headc, headc]
        v = torch.matmul(q_ref, v) ## [B, heads, HW, headc]
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)
        ret = v.permute(0, 3, 1, 2) + bias  # → [B, C, H, W] + [B, C, H, W]
        bias = self.o_proj2(self.act1(self.o_proj1(ret))) + bias

        return self.act2(bias)

class Galerkin_cross_attn(nn.Module):
    """
    Galerkin multi-head attention layer with cross-attention.

    Args:
        midc (int): Number of intermediate channels.
        heads (int): Number of attention heads.
        act (nn.Module): Activation function to use.
    """
    def __init__(self, midc, heads):
        super(Galerkin_cross_attn, self).__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1, bias=False)
        self.o_proj1 = nn.Conv2d(midc, midc, 1, bias=False)
        self.o_proj2 = nn.Conv2d(midc, midc, 1, bias=False)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()

    def forward(self, x, x_ref):
        """
        Forward pass for the attention layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, height, width).
            x_ref (torch.Tensor): Reference input tensor with shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with the same shape as input after applying attention.
        """
        B, C, H, W = x.shape
        B_ref, C_ref, H_ref, W_ref = x_ref.shape
        bias = x
        bias_ref = x_ref

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 3 * self.headc) # [B, 3C, H, W] --> [ B, heads, HW, 3*headc]
        qkv = qkv.permute(0, 2, 1, 3) #[ B, heads, HW, 3*headc]
        q, k, v = qkv.chunk(3, dim=-1) #[ B, heads, HW, headc]

        qkv_ref = self.qkv_proj(x_ref).permute(0, 2, 3, 1).reshape(B_ref, H_ref * W_ref, self.heads, 3 * self.headc)
        qkv_ref = qkv_ref.permute(0, 2, 1, 3)
        q_ref, k_ref, v_ref = qkv_ref.chunk(3, dim=-1)


        k = self.kln(k) #layer norm
        v = self.vln(v)
        k_ref = self.kln(k_ref) #layer norm
        v_ref = self.vln(v_ref)

        v = torch.matmul(k_ref.transpose(-2, -1), v) / (H * W) # [B, heads, headc, headc]
        v = torch.matmul(q_ref, v) ## [B, heads, HW, headc]
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)
        ret = v.permute(0, 3, 1, 2) + bias  # → [B, C, H, W] + [B, C, H, W]
        bias = self.o_proj2(self.act1(self.o_proj1(ret))) + bias

   
        v_ref = torch.matmul(k.transpose(-2, -1), v_ref) / (H_ref * W_ref) # [B, heads, headc, headc]
        v_ref = torch.matmul(q, v_ref) ## [B, heads, HW, headc]
        v_ref = v_ref.permute(0, 2, 1, 3).reshape(B_ref, H_ref, W_ref, C_ref)
        ret_ref = v_ref.permute(0, 3, 1, 2) + bias_ref  # → [B, C, H, W] + [B, C, H, W]
        bias_ref = self.o_proj2(self.act1(self.o_proj1(ret_ref))) + bias_ref

        return self.act2(bias), self.act2(bias_ref)

# class Galerkin_attn_r(nn.Module):
#     """
#     Galerkin multi-head attention layer.

#     Args:
#         midc (int): Number of intermediate channels.
#         heads (int): Number of attention heads.
#         act (nn.Module): Activation function to use.
#     """
#     def __init__(self, midc, heads):
#         super(Galerkin_attn_r, self).__init__()

#         self.headc = midc // heads
#         self.heads = heads
#         self.midc = midc

#         self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1, bias=False)
#         self.o_proj1 = nn.Conv2d(midc, midc, 1, bias=False)
#         self.o_proj2 = nn.Conv2d(midc, midc, 1, bias=False)

#         self.kln = LayerNorm((self.heads, 1, self.headc))
#         self.vln = LayerNorm((self.heads, 1, self.headc))

#         self.act1 = nn.GELU()
#         self.act2 = nn.GELU()

#         self.scale_proj = nn.Conv2d(1, self.heads * self.headc, kernel_size=1)

#     def forward(self, x, ratio):
#         """
#         Forward pass for the attention layer.

#         Args:
#             x (torch.Tensor): Input tensor with shape (batch, channels, height, width).

#         Returns:
#             torch.Tensor: Output tensor with the same shape as input after applying attention.
#         """
#         B, C, H, W = x.shape
#         bias = x
#         scale_bias = self.scale_proj(ratio.unsqueeze(1)) # # [B, heads * headc, H, W]
#         scale_bias = scale_bias.view(B, self.heads, self.headc, H, W).permute(0, 1, 3, 4, 2)  # [B, heads, H, W, headc]
#         scale_bias = scale_bias.reshape(B, self.heads, H * W, self.headc) ## [B, heads, HW, headc]


#         qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 3 * self.headc) # [B, 3C, H, W] --> [ B, heads, HW, 3*headc]
#         qkv = qkv.permute(0, 2, 1, 3) #[ B, heads, HW, 3*headc]
#         q, k, v = qkv.chunk(3, dim=-1) #[ B, heads, HW, headc]
#         q = q + scale_bias


#         k = self.kln(k) #layer norm
#         v = self.vln(v)
#         v = torch.matmul(k.transpose(-2, -1), v) / (H * W) # [B, heads, headc, headc]
#         v = torch.matmul(q, v) ## [B, heads, HW, headc]

#         v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)
#         ret = v.permute(0, 3, 1, 2) + bias  # → [B, C, H, W] + [B, C, H, W]
#         bias = self.o_proj2(self.act1(self.o_proj1(ret))) + bias
#         return self.act2(bias)

class ImplicitDecoder(nn.Module):
    def __init__(self, args, in_channels=64, hidden_dims=[64, 64, 64, 64, 64], heads=4):
        super().__init__()

        last_dim_K = in_channels * 9 + in_channels * 9
        
        last_dim_Q = 4

        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        self.mode =  args.mode

        # if self.mode == 'pe_liif+relu':
        #     for hidden_dim in hidden_dims:
        #         self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
        #                                     nn.ReLU(),
        #                                     ResBlock(channels = hidden_dim*2, nConvLayers = 4)
        #                                     ))    
        #         self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
        #                                     nn.ReLU()))
        #         last_dim_K = hidden_dim*2
        #         last_dim_Q = hidden_dim
        
        # if self.mode == 'pe+relu+bn':
        #     for hidden_dim in hidden_dims:
        #         self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
        #                                     nn.ReLU(),
        #                                     ResBlock(channels = hidden_dim*2, nConvLayers = 4)
        #                                     ))    
        #         self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
        #                                     nn.BatchNorm2d(hidden_dim),
        #                                     nn.ReLU()))
        #         last_dim_K = hidden_dim*2
        #         last_dim_Q = hidden_dim
        
        if self.mode == 'pe+relu+sin': # ori model
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4)
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim*2
                last_dim_Q = hidden_dim

        if self.mode == 'pe+relu+sin2': # ori model + [q,z]
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4)
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
                last_dim_Q = hidden_dim

        # if self.mode == 'pe+relu+sin2+matt': # ori model + [q,z] + multi-head self-attention
        #      # add multi-head self-attention
        #     self.matt1 = qkv_attn(hidden_dims[0], heads)
        #     self.matt2 = qkv_attn(hidden_dims[0], heads)
        #     for hidden_dim in hidden_dims:
        #         self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
        #                                     nn.ReLU(),
        #                                     ResBlock(channels = hidden_dim*2, nConvLayers = 4)
        #                                     ))    
        #         self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
        #                                     SineAct()))
        #         last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
        #         last_dim_Q = hidden_dim
        
        if self.mode == 'pe+relu+sin2+gatt': # ori model + [q,z] + galerkin attention
             # add galerkin attention
            self.gatt1 = Galerkin_attn(hidden_dims[0], heads)
            self.gatt2 = Galerkin_attn(hidden_dims[0], heads)
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4)
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
                last_dim_Q = hidden_dim
        
        if self.mode == 'pe+relu+sin2+gatt_cross': # ori model + [q,z] + galerkin_based cross attention
             # add galerkin attention
            self.gatt1 = Galerkin_cross_attn(hidden_dims[0], heads)
            self.gatt2 = Galerkin_cross_attn(hidden_dims[0], heads)
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4)
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
                last_dim_Q = hidden_dim
        
        if self.mode == 'pe+relu+sin2+gatt_cross_1': # ori model + [q,z] + galerkin_based cross attention
             # add galerkin attention
            self.gatt1 = Galerkin_cross_attn(hidden_dims[0], heads)
            self.gatt2 = Galerkin_cross_attn(hidden_dims[0], heads)
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4)
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
                last_dim_Q = hidden_dim
        
        if self.mode == 'pe+relu+sin2+gatt_add': # ori model + [q,z] + galerkin attention
             # add galerkin attention
            self.gatt1 = Galerkin_attn(hidden_dims[0], heads)
            self.gatt2 = Galerkin_attn(hidden_dims[0], heads)
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4)
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
                last_dim_Q = hidden_dim
        
        if self.mode == 'pe+relu+sin2+gatt_add+fr': # ori model + [q,z] + galerkin attention + fourier reparametrization
             # add galerkin attention
            self.gatt1 = Galerkin_attn(hidden_dims[0], heads)
            self.gatt2 = Galerkin_attn(hidden_dims[0], heads)
            high_freq_num=128
            low_freq_num=128
            phi_num=32
            alpha=0.05
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4),
                                            Fourier_reparam_linear(hidden_dim*2,hidden_dim*2,high_freq_num,low_freq_num,phi_num,alpha)
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct(),
                                            Fourier_reparam_linear(hidden_dim,hidden_dim,high_freq_num,low_freq_num,phi_num,alpha)))
                last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
                last_dim_Q = hidden_dim
        
        if self.mode == 'pe+relu+sin2+gatt_add+fr1': # ori model + [q,z] + galerkin attention + fourier reparametrization
             # add galerkin attention
            self.gatt1 = Galerkin_attn(hidden_dims[0], heads)
            self.gatt2 = Galerkin_attn(hidden_dims[0], heads)
            high_freq_num=128
            low_freq_num=128
            phi_num=32
            alpha=0.05
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4),
                                            Fourier_reparam_linear(hidden_dim*2,hidden_dim*2,high_freq_num,low_freq_num,phi_num,alpha)
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
                last_dim_Q = hidden_dim
        
        if self.mode == 'pe+relu+sin2+gatt_add+fr2': # ori model + [q,z] + galerkin attention + fourier reparametrization
             # add galerkin attention
            self.gatt1 = Galerkin_attn(hidden_dims[0], heads)
            self.gatt2 = Galerkin_attn(hidden_dims[0], heads)
            high_freq_num=128
            low_freq_num=128
            phi_num=32
            alpha=0.05
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4)
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct(),
                                            Fourier_reparam_linear(hidden_dim,hidden_dim,high_freq_num,low_freq_num,phi_num,alpha)))
                last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
                last_dim_Q = hidden_dim
        
        if self.mode == 'fpe+relu+sin2+gatt_add': # ori model + [q,z] + galerkin attention + fourier positional encoding
            # add galerkin attention
            self.gatt1 = Galerkin_attn(hidden_dims[0], heads)
            self.gatt2 = Galerkin_attn(hidden_dims[0], heads)
            last_dim_Q = 20  # fourier  补充原本rel 20=2+18=4*4+2(freq=4)  44=2+40+2  (freq=10 abs/rel)  80+2 (freq=10 abs+rel)

            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4),
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
                last_dim_Q = hidden_dim
        
        if self.mode == 'fpe+relu+sin2+gatt_add+fr2': # ori model + [q,z] + galerkin attention + fourier positional encoding
            # add galerkin attention
            self.gatt1 = Galerkin_attn(hidden_dims[0], heads)
            self.gatt2 = Galerkin_attn(hidden_dims[0], heads)
            high_freq_num=128
            low_freq_num=128
            phi_num=32
            alpha=0.05
            last_dim_Q = 42  # fourier 40 +2  (abs/rel)  80+2 (abs+rel)

            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                            nn.ReLU(),
                                            ResBlock(channels = hidden_dim*2, nConvLayers = 4),
                                            ))    
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct(),
                                            Fourier_reparam_linear(hidden_dim,hidden_dim,high_freq_num,low_freq_num,phi_num,alpha)))
                last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
                last_dim_Q = hidden_dim

        # if self.mode == 'pe+relu+sin2+gatt_before': # ori model + [q,z] + galerkin attention
        #      # add galerkin attention
        #     self.gatt1 = Galerkin_attn(4, heads=2)
        #     self.gatt2 = Galerkin_attn(4, heads=2)
        #     for hidden_dim in hidden_dims:
        #         self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
        #                                     nn.ReLU(),
        #                                     ResBlock(channels = hidden_dim*2, nConvLayers = 4)
        #                                     ))    
        #         self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
        #                                     SineAct()))
        #         last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
        #         last_dim_Q = hidden_dim

        # if self.mode == 'pe+relu+sin2+gatt+r': # ori model + [q,z] + galerkin attention
        #      # add galerkin attention
        #     self.gatt1 = Galerkin_attn_r(hidden_dims[0], heads)
        #     self.gatt2 = Galerkin_attn_r(hidden_dims[0], heads)
        #     for hidden_dim in hidden_dims:
        #         self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
        #                                     nn.ReLU(),
        #                                     ResBlock(channels = hidden_dim*2, nConvLayers = 4)
        #                                     ))    
        #         self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
        #                                     SineAct()))
        #         last_dim_K = hidden_dim*2 + in_channels * 9 + in_channels * 9
        #         last_dim_Q = hidden_dim
        
        # if self.mode == 'pe+relu+sin+bn':
        #     for hidden_dim in hidden_dims:
        #         self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
        #                                     nn.ReLU(),
        #                                     ResBlock(channels = hidden_dim*2, nConvLayers = 4)
        #                                     ))    
        #         self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
        #                                     nn.BatchNorm2d(hidden_dim),
        #                                     SineAct()))
        #         last_dim_K = hidden_dim*2
        #         last_dim_Q = hidden_dim
        
        self.last_layer = nn.Conv2d(hidden_dims[-1], 2, 1)
        self.ref_branch = nn.Sequential(nn.Conv2d(in_channels * 9, hidden_dims[-2], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-2],hidden_dims[-1], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1],2, 1),
                            nn.ReLU())
        self.in_branch = nn.Sequential(nn.Conv2d(in_channels * 9, hidden_dims[-2], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-2],hidden_dims[-1], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1],2, 1),
                            nn.ReLU())
        
    def _make_pos_encoding(self, x, size): 
        B, C, H, W = x.shape
        H_up, W_up = size
       
        h_idx = -1 + 1/H + 2/H * torch.arange(H, device=x.device).float()  #归一化坐标，范围从 -1 到 1，并偏移以确保边缘坐标正确映射 #32
        w_idx = -1 + 1/W + 2/W * torch.arange(W, device=x.device).float()  # 32
        in_grid = torch.stack(torch.meshgrid(h_idx, w_idx), dim=0) # （2, 32, 32）

        h_idx_up = -1 + 1/H_up + 2/H_up * torch.arange(H_up, device=x.device).float()
        w_idx_up = -1 + 1/W_up + 2/W_up * torch.arange(W_up, device=x.device).float()
        up_grid = torch.stack(torch.meshgrid(h_idx_up, w_idx_up), dim=0)
        
        rel_grid = (up_grid - F.interpolate(in_grid.unsqueeze(0), size=(H_up, W_up), mode='nearest-exact'))
        rel_grid[:,0,:,:] *= H
        rel_grid[:,1,:,:] *= W

        return rel_grid.contiguous().detach()
    
    def _make_fourier_pos_encoding(self, x, size, abs=False, rel=False):  # add fourier position encoding   absolute + relative
        """
        Args:
        x: input feature map (B, C, H, W)
        size: target upsample sizw (H_up, W_up)
        num_freqs: num_freqs used to encode the position
        Returns:
        fourier_encoded_grid: [1, 2*num_freqs*2, H_up, W_up]  # abs+rel C_pe = 40+40  4*4+2=18(save ori rel coord)
        """
        B, C, H, W = x.shape
        H_up, W_up = size

        # para for positional encoding
        num_freqs = 4  # change for 4 or 10
        freq_bands = 2 ** torch.arange(num_freqs).float() * math.pi  # [num_freqs]
        pos_encoding = []

       
        h_idx = -1 + 1/H + 2/H * torch.arange(H, device=x.device).float()  #归一化坐标，范围从 -1 到 1，并偏移以确保边缘坐标正确映射 #32
        w_idx = -1 + 1/W + 2/W * torch.arange(W, device=x.device).float()  # 32
        in_grid = torch.stack(torch.meshgrid(h_idx, w_idx), dim=0) # （2, 32, 32）

        h_idx_up = -1 + 1/H_up + 2/H_up * torch.arange(H_up, device=x.device).float()
        w_idx_up = -1 + 1/W_up + 2/W_up * torch.arange(W_up, device=x.device).float()
        up_grid = torch.stack(torch.meshgrid(h_idx_up, w_idx_up), dim=0)

        if abs == True:
            for i in range(2):  # x, y
                for freq in freq_bands:
                    pos_encoding.append(torch.sin(freq * up_grid[i]))
                    pos_encoding.append(torch.cos(freq * up_grid[i]))
        
        
        if rel == True:
            rel_grid = (up_grid - F.interpolate(in_grid.unsqueeze(0), size=(H_up, W_up), mode='nearest-exact'))
            rel_grid[:,0,:,:] *= H
            rel_grid[:,1,:,:] *= W
            rel_grid = rel_grid.squeeze(0) # 12hw --> 2hw

            for i in range(2):  # x, y
                for freq in freq_bands:
                    pos_encoding.append(torch.sin(freq * rel_grid[i]))
                    pos_encoding.append(torch.cos(freq * rel_grid[i]))
        
        # pos_enc = torch.stack(pos_encoding, dim=0).unsqueeze(0)
        pos_enc = torch.cat([rel_grid, torch.stack(pos_encoding, dim=0)], dim=0).unsqueeze(0)  # 补充上原本的rel坐标
        return pos_enc.contiguous().detach()

    def step(self, x, ref, syn_inp):
        q = syn_inp
        q_ref =syn_inp
        k = x
        k_ref = ref
        kk = torch.cat([k,k_ref],dim=1)

        if self.mode == 'pe+relu+sin2': # ori model + [q,z]
            kk = self.K[0](kk)
            dim = kk.shape[1]//2
            q = kk[:,:dim]*self.Q[0](q)
            q_ref  = kk[:,dim:]*self.Q[0](q_ref)
            for i in range(1, len(self.K)):
                kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
                dim = kk.shape[1]//2
                q = kk[:,:dim]*self.Q[i](q)
                q_ref = kk[:,dim:]*self.Q[i](q_ref)
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        # if self.mode == 'pe+relu+sin2+matt': # ori model + [q,z] + multi-head slef attention
        #     kk = self.K[0](kk)
        #     dim = kk.shape[1]//2
        #     q = kk[:,:dim]*self.Q[0](q)  # b, hidden_dim, h, w
        #     q = self.matt2(self.matt1(q))

        #     q_ref  = kk[:,dim:]*self.Q[0](q_ref) # b, hidden_dim, h, w
        #     q_ref = self.matt2(self.matt1(q_ref))

        #     for i in range(1, len(self.K)):
        #         kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
        #         dim = kk.shape[1]//2
        #         q = kk[:,:dim]*self.Q[i](q)
        #         q_ref = kk[:,dim:]*self.Q[i](q_ref)
        #     q = self.last_layer(q)
        #     q_ref = self.last_layer(q_ref)
        #     return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        if self.mode == 'pe+relu+sin2+gatt': # ori model + [q,z] + multi-head slef attention
            kk = self.K[0](kk)
            dim = kk.shape[1]//2
            q = kk[:,:dim]*self.Q[0](q)  # b, hidden_dim, h, w
            q = self.gatt2(self.gatt1(q))

            q_ref  = kk[:,dim:]*self.Q[0](q_ref) # b, hidden_dim, h, w
            q_ref = self.gatt2(self.gatt1(q_ref))

            for i in range(1, len(self.K)):
                kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
                dim = kk.shape[1]//2
                q = kk[:,:dim]*self.Q[i](q)
                q_ref = kk[:,dim:]*self.Q[i](q_ref)
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        if self.mode == 'pe+relu+sin2+gatt_cross': # ori model + [q,z] + multi-head slef attention
            kk = self.K[0](kk)
            dim = kk.shape[1]//2 
            q = kk[:,:dim]*self.Q[0](q)  # b, hidden_dim, h, w
            q_ref  = kk[:,dim:]*self.Q[0](q_ref) # b, hidden_dim, h, w
            q, q_ref = self.gatt1(q, q_ref)
            q, q_ref = self.gatt2(q, q_ref) # galerkin_based cross attention

            for i in range(1, len(self.K)):
                kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
                dim = kk.shape[1]//2
                q = kk[:,:dim]*self.Q[i](q)
                q_ref = kk[:,dim:]*self.Q[i](q_ref)
                q, q_ref = self.gatt1(q, q_ref)
                q, q_ref = self.gatt2(q, q_ref) # galerkin_based cross attention
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        if self.mode == 'pe+relu+sin2+gatt_cross_1': # ori model + [q,z] + multi-head slef attention
            kk = self.K[0](kk)
            dim = kk.shape[1]//2 
            q = kk[:,:dim]*self.Q[0](q)  # b, hidden_dim, h, w
            q_ref  = kk[:,dim:]*self.Q[0](q_ref) # b, hidden_dim, h, w
            q, q_ref = self.gatt1(q, q_ref)
            q, q_ref = self.gatt2(q, q_ref) # galerkin_based cross attention

            for i in range(1, len(self.K)):
                kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
                dim = kk.shape[1]//2
                q = kk[:,:dim]*self.Q[i](q)
                q_ref = kk[:,dim:]*self.Q[i](q_ref)
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        if self.mode == 'pe+relu+sin2+gatt_add': # ori model + [q,z] + multi-head slef attention
            kk = self.K[0](kk)
            dim = kk.shape[1]//2
            q = kk[:,:dim]+self.Q[0](q)  # b, hidden_dim, h, w
            q = self.gatt2(self.gatt1(q))

            q_ref  = kk[:,dim:]+self.Q[0](q_ref) # b, hidden_dim, h, w
            q_ref = self.gatt2(self.gatt1(q_ref))

            for i in range(1, len(self.K)):
                kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
                dim = kk.shape[1]//2
                q = kk[:,:dim]+self.Q[i](q)
                q_ref = kk[:,dim:]+self.Q[i](q_ref)
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        if self.mode == 'pe+relu+sin2+gatt_add+fr': # ori model + [q,z] + multi-head slef attention
            kk = self.K[0](kk)
            dim = kk.shape[1]//2
            q = kk[:,:dim]+self.Q[0](q)  # b, hidden_dim, h, w
            q = self.gatt2(self.gatt1(q))

            q_ref  = kk[:,dim:]+self.Q[0](q_ref) # b, hidden_dim, h, w
            q_ref = self.gatt2(self.gatt1(q_ref))

            for i in range(1, len(self.K)):
                kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
                dim = kk.shape[1]//2
                q = kk[:,:dim]+self.Q[i](q)
                q_ref = kk[:,dim:]+self.Q[i](q_ref)
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        if self.mode == 'pe+relu+sin2+gatt_add+fr1': # ori model + [q,z] + multi-head slef attention
            kk = self.K[0](kk)
            dim = kk.shape[1]//2
            q = kk[:,:dim]+self.Q[0](q)  # b, hidden_dim, h, w
            q = self.gatt2(self.gatt1(q))

            q_ref  = kk[:,dim:]+self.Q[0](q_ref) # b, hidden_dim, h, w
            q_ref = self.gatt2(self.gatt1(q_ref))

            for i in range(1, len(self.K)):
                kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
                dim = kk.shape[1]//2
                q = kk[:,:dim]+self.Q[i](q)
                q_ref = kk[:,dim:]+self.Q[i](q_ref)
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        if self.mode == 'pe+relu+sin2+gatt_add+fr2': # ori model + [q,z] + multi-head slef attention
            kk = self.K[0](kk)
            dim = kk.shape[1]//2
            q = kk[:,:dim]+self.Q[0](q)  # b, hidden_dim, h, w
            q = self.gatt2(self.gatt1(q))

            q_ref  = kk[:,dim:]+self.Q[0](q_ref) # b, hidden_dim, h, w
            q_ref = self.gatt2(self.gatt1(q_ref))

            for i in range(1, len(self.K)):
                kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
                dim = kk.shape[1]//2
                q = kk[:,:dim]+self.Q[i](q)
                q_ref = kk[:,dim:]+self.Q[i](q_ref)
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)

        if self.mode == 'fpe+relu+sin2+gatt_add':
            kk = self.K[0](kk)
            dim = kk.shape[1]//2
            q = kk[:,:dim]+self.Q[0](q)  # b, hidden_dim, h, w
            q = self.gatt2(self.gatt1(q))

            q_ref  = kk[:,dim:]+self.Q[0](q_ref) # b, hidden_dim, h, w
            q_ref = self.gatt2(self.gatt1(q_ref))

            for i in range(1, len(self.K)):
                kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
                dim = kk.shape[1]//2
                q = kk[:,:dim]+self.Q[i](q)
                q_ref = kk[:,dim:]+self.Q[i](q_ref)
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        if self.mode == 'fpe+relu+sin2+gatt_add+fr2':
            kk = self.K[0](kk)
            dim = kk.shape[1]//2
            q = kk[:,:dim]+self.Q[0](q)  # b, hidden_dim, h, w
            q = self.gatt2(self.gatt1(q))

            q_ref  = kk[:,dim:]+self.Q[0](q_ref) # b, hidden_dim, h, w
            q_ref = self.gatt2(self.gatt1(q_ref))

            for i in range(1, len(self.K)):
                kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
                dim = kk.shape[1]//2
                q = kk[:,:dim]+self.Q[i](q)
                q_ref = kk[:,dim:]+self.Q[i](q_ref)
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        # if self.mode == 'pe+relu+sin2+gatt_before': # ori model + [q,z] + multi-head slef attention
        #     kk = self.K[0](kk)
        #     dim = kk.shape[1]//2
        #     q = self.gatt2(self.gatt1(q))
        #     q_ref = self.gatt2(self.gatt1(q_ref))

        #     q = kk[:,:dim]*self.Q[0](q)  # b, hidden_dim, h, w
        #     q_ref  = kk[:,dim:]*self.Q[0](q_ref) # b, hidden_dim, h, w

        #     for i in range(1, len(self.K)):
        #         kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
        #         dim = kk.shape[1]//2
        #         q = kk[:,:dim]*self.Q[i](q)
        #         q_ref = kk[:,dim:]*self.Q[i](q_ref)
        #     q = self.last_layer(q)
        #     q_ref = self.last_layer(q_ref)
        #     return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        
        # if self.mode == 'pe+relu+sin2+gatt+r': # ori model + [q,z] + multi-head slef attention
        #     kk = self.K[0](kk)
        #     dim = kk.shape[1]//2
        #     ratio = q[:,2,:,:]   #syinp = torch.cat([rel_coord, ratio, ratio_ref], dim=1) # 6 4 96 96
        #     ratio_ref =  q[:,3,:,:]

        #     q = kk[:,:dim]*self.Q[0](q)  # b, hidden_dim, h, w
        #     q = self.gatt2(self.gatt1(q, ratio), ratio)

        #     q_ref  = kk[:,dim:]*self.Q[0](q_ref) # b, hidden_dim, h, w
        #     q_ref = self.gatt2(self.gatt1(q_ref, ratio_ref), ratio_ref)

        #     for i in range(1, len(self.K)):
        #         kk = self.K[i](torch.cat([q,q_ref,x,ref], dim=1))
        #         dim = kk.shape[1]//2
        #         q = kk[:,:dim]*self.Q[i](q)
        #         q_ref = kk[:,dim:]*self.Q[i](q_ref)
        #     q = self.last_layer(q)
        #     q_ref = self.last_layer(q_ref)
        #     return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
         
        else:  # ori model pe+relu+sin
            for i in range(len(self.K)):
                kk = self.K[i](kk)
                dim = kk.shape[1]//2
                q = kk[:,:dim]*self.Q[i](q)
                q_ref = kk[:,dim:]*self.Q[i](q_ref)
            q = self.last_layer(q)
            q_ref = self.last_layer(q_ref)
            return q + self.in_branch(x) ,q_ref + self.ref_branch(ref)
        

    def batched_step(self, x, syn_inp, bsize):
        with torch.no_grad():
            h, w = syn_inp.shape[-2:]
            ql = 0
            preds = []
            while ql < w:
                qr = min(ql + bsize//h, w)
                pred = self.step(x[:, :, :, ql: qr], syn_inp[:, :, :, ql: qr])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=-1)
        return pred


    def forward(self, x, ref, size, bsize=None): #feat, ref, size[H_hr, W_hr], bsize  [-1,1] --> [-1,1]
        B, C, H_in, W_in = x.shape  # bchw 6 64 32 32
        Bref, Cref, H_in_ref, W_in_ref = ref.shape # bchw 6 64 128 128

        
        # change for IGA-INR
        # if H_in == W_in ==1:
        #     rel_coord = (self._make_pos_encoding(x, size).expand(B, -1, *size)) 
        #     ratio_scale = math.sqrt((H_in * W_in) / (size[0] * size[1]))
        #     ratio = x.new_ones(B, 1, *size) * ratio_scale

        #     ratio_ref_scale = math.sqrt((H_in_ref * W_in_ref) / (size[0] * size[1]))
        #     ratio_ref = ref.new_ones(Bref, 1, *size) * ratio_ref_scale
        # else: 

        if self.mode == 'fpe+relu+sin2+gatt_add' or self.mode == 'fpe+relu+sin2+gatt_add+fr2':
            rel_coord = (self._make_fourier_pos_encoding(x, size, abs=False, rel=True).expand(B, -1, *size)) #-1代表自动推断并保持不变 # （1，2，96, 96) --> (6, 2, 96, 96)
            ratio = (x.new_tensor([math.sqrt((H_in*W_in)/(size[0]*size[1]))]).view(1, -1, 1, 1).expand(B, -1, *size)) #保持维度一致
            ratio_ref = (ref.new_tensor([math.sqrt((H_in_ref*W_in_ref)/(size[0]*size[1]))]).view(1, -1, 1, 1).expand(Bref, -1, *size))

        else:
            rel_coord = (self._make_pos_encoding(x, size).expand(B, -1, *size)) #-1代表自动推断并保持不变 # （1，2，96, 96) --> (6, 2, 96, 96)
            ratio = (x.new_tensor([math.sqrt((H_in*W_in)/(size[0]*size[1]))]).view(1, -1, 1, 1).expand(B, -1, *size)) #保持维度一致
            ratio_ref = (ref.new_tensor([math.sqrt((H_in_ref*W_in_ref)/(size[0]*size[1]))]).view(1, -1, 1, 1).expand(Bref, -1, *size))


        syn_inp = torch.cat([rel_coord, ratio, ratio_ref], dim=1) # 6 4 96 96
        x = F.interpolate(F.unfold(x, 3, padding=1).view(B, C*9, H_in, W_in), size=syn_inp.shape[-2:], mode='bilinear') # upsample
        ref = F.interpolate(F.unfold(ref, 3, padding=1).view(B, C*9, H_in_ref, W_in_ref), size=syn_inp.shape[-2:], mode='bilinear') #upsample
        if bsize is None: 
            pred = self.step(x, ref, syn_inp)
        else:
            pred = self.batched_step(x, syn_inp, bsize)
        return pred


class DUALRef(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = make_rdn()
        self.decoder = ImplicitDecoder(self.args)
        # self.mixer = nn.Conv2d(64*2, 64, 1, padding=0, stride=1)  # 未使用
    
    def set_scale(self, scale, scale2):
        self.scale = scale
        self.scale2 = scale2

    def forward(self, inp, bsize = None):  # input: lr, ref_hr, ref_lr, self.args.ref_type, epoch
        if len(inp)==5:
            epoch = inp[4]
        else:
            epoch = None
        ref_type = inp[3]
        if ref_type == None: # training default == none
            ref_type = random.randint(1,2) 
            if epoch is not None and epoch < 10:
                ref_type = 1  # epoch 0-9: ref_type = 1 lr（32*32） + ref_hr(128*128)
        ref = inp[ref_type] 
        inp = inp[0]

        B,C,H,W = inp.shape
        B,C,H_ref,W_ref = ref.shape
        H_hr = round(H*self.scale)
        W_hr = round(W*self.scale2)
        feat = self.encoder((inp-0.5)/0.5) # reset input range from [0,1] to [-1,1]  #input lr bchw 6 2 32 32
        with torch.no_grad():  # shared encoder weights with lr and ref input
            ref = self.encoder((ref-0.5)/0.5)  # reset input range from [0,1] to [-1,1]  #input ref_hr bchw 6 2 128 128
        ref.requires_grad = True
        size = [H_hr, W_hr]  # 128*128
        pred,pred_ref = self.decoder(feat, ref, size, bsize)

        return pred*0.5+0.5, pred_ref*0.5+0.5   # reset input range from [-1,1] to [0,1]


# # cal para
# if __name__ == '__main__':

#     class Args:
#         def __init__(self):
#             self.scale = [2]  
#             # self.scale2 = 2
#             self.rgb_range = 1
#             self.mode = "pe+relu+sin2+gatt_add+fr2"  # pe+relu+sin2+gatt_add+fr2  ori_model: pe+relu+sin

#     args = Args()
#     model = DUALRef(args)  # 2x or 4x or 6x
#     model.set_scale(scale=4, scale2=4)

#     model.eval()
#     model.cuda()

#     inp = torch.randn(1,2,64,64).cuda()
#     refhr = torch.randn(1,2,64,64).cuda()  # 1,2,128,128  or 1,2,256,256 or 1,2,384,384
#     reflr = torch.randn(1,2,64,64).cuda()
#     ref_type = 1 

#     input = (inp, refhr, reflr, ref_type)
    
#     bufer_infer = 0

#     with torch.no_grad():  # 关闭梯度追踪，释放显存
#         for _ in range(5):  # warm up
#             _ = model(input)
#         for i in range(100):
#             # print("inference time:", i)
#             time_start = time.time()
#             out = model(input)
#             # print(out.shape)
#             time_end = time.time()
#             infer_time_ms = (time_end - time_start) * 1000  # 秒转毫秒
#             bufer_infer += infer_time_ms

#         avg_infer_time = bufer_infer / 100
#         print(f'Avg Inference time: {avg_infer_time:.2f} ms')

#     flops, params = profile(model, inputs=(input,))
#     print('Model:{:.2f} GFLOPs and {:.2f}M parameters'.format(flops*2 / 1e9, params / 1e6))

