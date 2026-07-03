import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy.ndimage import gaussian_filter

def gaussian_filter(image,channels,sigma):
    kernel,kernel_size = create_gaussian_kernel(channels,sigma)
    kernel = kernel.cuda()
    # print(kernel.shape)
    # print(kernel_size)
    image = F.conv2d(image, kernel, padding= kernel_size // 2)
    return image
def create_gaussian_kernel(channels,sigma):
    size = int(2 * 3 * sigma + 1)
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-3*sigma)**2+(y-3*sigma)**2)/(2*sigma**2)), (size, size))
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.view(1, 1, size, size).repeat(1, channels, 1, 1)
    return kernel, size

class HFLoss(nn.Module):
    def __init__(self, args):
        super(HFLoss, self).__init__()

    '''Compute space domain L1 loss with highfreq.

       Parameters:
      - sr: super-resolution image.
      - hr: reference image.
      - sigma: guassian filter sigma.

      Returns:
      - loss: L1 loss with highfreq in space domain.
     '''
    def forward(self, sr, hr, sigma=5.0):
       
        # Calculate guassian filter for lowpass image
        b,c,h,w = sr.shape
        lowpass_sr = gaussian_filter(sr, c, sigma)
        highfreq_sr = sr - lowpass_sr
        lowpass_hr = gaussian_filter(hr, c, sigma)
        highfreq_hr = hr - lowpass_hr
 
        # Calculate weighted L1 loss
        loss = torch.mean(torch.abs((highfreq_sr-highfreq_hr)))
        return loss 

if __name__ == "__main__":
    model_test = HFLoss(0)  
    a = torch.randn([5,2,20,20]) 
    b = torch.randn([5,2,20,20]) 
    loss = model_test(a,b) 
    print(loss)
