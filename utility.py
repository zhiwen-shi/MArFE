import os
import math
import time
import datetime
from functools import reduce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from skimage.metrics import structural_similarity as ssimcalcu
from math import log10

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = './experiment/' + args.save
        else:
            self.dir = './experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale) #根据参数设置绘制多条折线（每条折线代表不同放大倍数下的指标）
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test)) #将图形保存为pdf文件
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR') # 复原结果保存为三张图片，分别对应SR、LR和HR
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy() #对图片进行归一化处理，并将其转换为numpy数组
            misc.imsave('{}{}.png'.format(filename, p), ndarr)

def quantize(img, rgb_range):  # RGB重量化
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def save_fig(x, y, pred, fig_name, srresult): #绘制结果，并显示超分指标
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray)
        ax[0].set_title('LR', fontsize=30)
       
        ax[1].imshow(pred, cmap=plt.cm.gray)
        ax[1].set_title('SR', fontsize=30)
        ax[1].set_xlabel("PSNR:{:.4f}\nSSIM:{:.4f}\nMSE:{:.4f}".format(srresult[0],srresult[1],srresult[2]),fontsize=20)

        ax[2].imshow(y, cmap=plt.cm.gray)
        ax[2].set_title('HR', fontsize=30)
        f.savefig(fig_name)
        plt.close()

# def save_single_fig(x, y, pred, lr_path, hr_path, sr_path): # for save single sr and hr fig
#         cv2.imwrite(lr_path, x)
#         cv2.imwrite(hr_path, y)
#         cv2.imwrite(sr_path, pred)

def save_single_fig(x, y, pred, ref_lr, ref_hr, lr_path, hr_path, sr_path, ref_lr_path, ref_hr_path): # for save single sr and hr fig change for ref_lr ref_hr
        cv2.imwrite(lr_path, x) 
        cv2.imwrite(hr_path, y)
        cv2.imwrite(sr_path, pred)
        cv2.imwrite(ref_lr_path, ref_lr)
        cv2.imwrite(ref_hr_path, ref_hr)
# change for save ref_hr ref_lr
def calc_psnr(lr, sr,  hr, ref_hr, ref_lr, FSsr=None, img_name=None, save=False, save_single=False, scale=1, savefile=None, ref=None):
    
    if FSsr is not None: #如果给定FSsr参数，则将其通过ifftshift和ifftn转换为时域，并计算其幅度
        FSsr = torch.fft.ifftshift((FSsr), dim=[2,3])
        sr = math.sqrt(FSsr.shape[2]*FSsr.shape[3]) * (torch.fft.ifftn(FSsr, dim=[2,3]))
        srmagnitude = torch.abs(sr) 
    else:
        srmagnitude = (sr[:, 0:1, :, :] ** 2 + sr[:, 1:2, :, :] ** 2).sqrt()

    lrmagnitude = (lr[:, 0:1, :, :] ** 2 + lr[:, 1:2, :, :] ** 2).sqrt()
    hrmagnitude = (hr[:, 0:1, :, :] ** 2 + hr[:, 1:2, :, :] ** 2).sqrt() #分别计算lr、sr和hr的幅度
    
    ref_lrmagnitude = (ref_lr[:, 0:1, :, :] ** 2 + ref_lr[:, 1:2, :, :] ** 2).sqrt()
    ref_hrmagnitude = (ref_hr[:, 0:1, :, :] ** 2 + ref_hr[:, 1:2, :, :] ** 2).sqrt() #分别计算ref_lr和ref_hr的幅度

    lrcpu = lrmagnitude[0,0,:,:].cpu().numpy()
    hrcpu = hrmagnitude[0,0,:,:].cpu().numpy()
    srcpu = srmagnitude[0,0,:,:].cpu().numpy() #从GPU转移到CPU，并转换为numpy数组
    
    ref_lrcpu = ref_lrmagnitude[0,0,:,:].cpu().numpy()
    ref_hrcpu = ref_hrmagnitude[0,0,:,:].cpu().numpy()

    if ref is not None:
        refmagnitude = (ref[:, 0:1, :, :] ** 2 + ref[:, 1:2, :, :] ** 2).sqrt()
        refcpu = refmagnitude[0,0,:,:].cpu().numpy()

    peak_signal = (hrmagnitude.max()-hrmagnitude.min()).item()
    mse = (srmagnitude - hrmagnitude).pow(2).mean().item()
    errormap = torch.abs(srmagnitude - hrmagnitude).cpu().numpy()
    errormap = errormap[0,0,:,:]
    psnr = 10*log10(peak_signal**2/mse)
    ssim = ssimcalcu(srcpu,hrcpu,data_range=srcpu.max() - srcpu.min())
    if save:    
        pthroot = os.path.join('./savefigresult','{:s}'.format(savefile), 'x{:.1f}_{:.1f}'.format(scale[0],scale[1]))
        if not os.path.exists(pthroot):
            os.makedirs(pthroot)
        img_path = os.path.join(pthroot, 'results_{:s}.png'.format(img_name))
        srresult = [psnr,ssim,mse]
        save_fig(lrcpu*255, hrcpu*255, srcpu*255, img_path, srresult)  #是否保存的是频域图？
        if save_single == True:
            fig_clip = os.path.join(pthroot,('x{:.1f}_{:.1f}'.format(scale[0],scale[1]) + "_single_fig"))
            os.makedirs(fig_clip, exist_ok=True)
            lr_path = os.path.join(fig_clip, "{:s}_lr.png".format(img_name))
            hr_path = os.path.join(fig_clip, "{:s}_hr.png".format(img_name))
            sr_path = os.path.join(fig_clip, "{:s}_sr.png".format(img_name))
            ref_lr_path = os.path.join(fig_clip, "{:s}_ref_lr.png".format(img_name))
            ref_hr_path = os.path.join(fig_clip, "{:s}_ref_hr.png".format(img_name))
            save_single_fig(lrcpu*255, hrcpu*255, srcpu*255, ref_lrcpu*255, ref_hrcpu*255, lr_path, hr_path, sr_path, ref_lr_path, ref_hr_path)
    return psnr,ssim,mse


# def calc_psnr(lr, sr,  hr, FSsr=None, img_name=None, save=False, save_single=False, scale=1, savefile=None, ref=None):
    
#     if FSsr is not None: #如果给定FSsr参数，则将其通过ifftshift和ifftn转换为时域，并计算其幅度
#         FSsr = torch.fft.ifftshift((FSsr), dim=[2,3])
#         sr = math.sqrt(FSsr.shape[2]*FSsr.shape[3]) * (torch.fft.ifftn(FSsr, dim=[2,3]))
#         srmagnitude = torch.abs(sr) 
#     else:
#         srmagnitude = (sr[:, 0:1, :, :] ** 2 + sr[:, 1:2, :, :] ** 2).sqrt()

#     lrmagnitude = (lr[:, 0:1, :, :] ** 2 + lr[:, 1:2, :, :] ** 2).sqrt()
#     hrmagnitude = (hr[:, 0:1, :, :] ** 2 + hr[:, 1:2, :, :] ** 2).sqrt() #分别计算lr、sr和hr的幅度
#     lrcpu = lrmagnitude[0,0,:,:].cpu().numpy()
#     hrcpu = hrmagnitude[0,0,:,:].cpu().numpy()
#     srcpu = srmagnitude[0,0,:,:].cpu().numpy() #从GPU转移到CPU，并转换为numpy数组
#     if ref is not None:
#         refmagnitude = (ref[:, 0:1, :, :] ** 2 + ref[:, 1:2, :, :] ** 2).sqrt()
#         refcpu = refmagnitude[0,0,:,:].cpu().numpy()

#     peak_signal = (hrmagnitude.max()-hrmagnitude.min()).item()
#     mse = (srmagnitude - hrmagnitude).pow(2).mean().item()
#     errormap = torch.abs(srmagnitude - hrmagnitude).cpu().numpy()
#     errormap = errormap[0,0,:,:]
#     psnr = 10*log10(peak_signal**2/mse)
#     ssim = ssimcalcu(srcpu,hrcpu,data_range=srcpu.max() - srcpu.min())
#     if save:    
#         pthroot = os.path.join('./savefigresult','{:s}'.format(savefile), 'x{:.1f}_{:.1f}'.format(scale[0],scale[1]))
#         if not os.path.exists(pthroot):
#             os.makedirs(pthroot)
#         img_path = os.path.join(pthroot, 'results_{:s}.png'.format(img_name))
#         srresult = [psnr,ssim,mse]
#         save_fig(lrcpu*255, hrcpu*255, srcpu*255, img_path, srresult)  #是否保存的是频域图？
#         if save_single == True:
#             fig_clip = os.path.join(pthroot,('x{:.1f}_{:.1f}'.format(scale[0],scale[1]) + "_single_fig"))
#             os.makedirs(fig_clip, exist_ok=True)
#             lr_path = os.path.join(fig_clip, "{:s}_lr.png".format(img_name))
#             hr_path = os.path.join(fig_clip, "{:s}_hr.png".format(img_name))
#             sr_path = os.path.join(fig_clip, "{:s}_sr.png".format(img_name))
#             save_single_fig(lrcpu*255, hrcpu*255, srcpu*255, lr_path, hr_path, sr_path)
#     return psnr,ssim,mse



def make_optimizer(args, my_model):  #创建优化器
    trainable = filter(lambda x: x.requires_grad, my_model.parameters()) #过滤器，筛选出模型中需要梯度更新的参数

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):  #创建学习率调度器
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay, #设置步长和衰减因子
            gamma=args.gamma,
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    scheduler.step(args.start_epoch - 1)

    return scheduler

