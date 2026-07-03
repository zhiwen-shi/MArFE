import torch
import torch.nn as nn
import numpy as np
import math


class GKLoss(nn.Module):
    def __init__(self, args):
        super(GKLoss, self).__init__()

    '''Compute frequency domain L1 loss with weighted by both amplitude and distance.

       Parameters:
      - FSsr: Frequency spectrum of super-resolution image.
      - FShr: Frequency spectrum of reference image.
      - alpha: Weight for amplitude in the exponent.
      - beta: Weight for distance from center in the exponent.

      Returns:
      - loss: Weighted L1 loss in frequency domain.
     '''

    def forward(self, sr, FSsr, hr, alpha=2.0, beta=1.0):
        b,c,h,w = hr.shape
        hr_comp = hr[:,0:1,:,:]+1j*hr[:,1:2,:,:]


        if FSsr is None:
            sr_comp = sr[:,0:1,:,:]+1j*sr[:,1:2,:,:]
            FSsr = 1 / math.sqrt(h*w) * torch.fft.fftn(sr_comp, dim=[2,3])
            FSsr = torch.fft.fftshift(FSsr, dim=[2,3])
        FShr = 1 / math.sqrt(h*w) * torch.fft.fftn(hr_comp, dim=[2,3])
        FShr = torch.fft.fftshift(FShr, dim=[2,3])

        # Calculate distance from center for each frequency bin 生成归一化距离mask
        center_i, center_j = h // 2, w // 2
        distance_from_center = torch.ones_like(FSsr)
        for i in range(h):
            for j in range(w):
                distance_from_center[:,:,i,j] = torch.sqrt(torch.square(torch.tensor(i) - center_i) + torch.square(torch.tensor(j) - center_j))

        distance_from_center = torch.abs(distance_from_center)/torch.max(torch.abs(distance_from_center))
        mask_dis = torch.exp(alpha * distance_from_center)

        # Calculate distance from center for each frequency bin 生成归一化幅值mask
        mask_amp = torch.exp(beta * torch.abs(FSsr) / torch.max(torch.abs(FSsr)))
 
        # Calculate weighted L1 loss

        loss = torch.mean(torch.abs((FSsr-FShr) * mask_dis * mask_amp))
        return loss 

if __name__ == "__main__":
    model_test = GKLoss(0)  
    a = torch.randn([5,2,20,20]) 
    b = torch.randn([5,2,20,20]) 
    loss = model_test(a,None,b,5,5) 
    print(loss)
