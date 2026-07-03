import os
import math
import matplotlib
matplotlib.use('Agg')
import utility
import torch
import numpy as np
from decimal import Decimal
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import random
import functorch
from functorch import make_functional, vmap, vjp, jvp, jacrev,make_functional_with_buffers,grad

class DataConsistency(nn.Module):
    def __init__(self):
        super(DataConsistency, self).__init__()

    def forward(self, FSsr, hr, shape1, shape2):
        b,c,h,w = hr.shape
        hr_comp = hr[:,0:1,:,:]+1j*hr[:,1:2,:,:] # 获取hr图像的尺寸，并将hr图像转换为复数形式；
        k_hr = 1 / math.sqrt(hr.shape[2]*hr.shape[3]) * torch.fft.fftshift(torch.fft.fftn(hr_comp, dim=[2,3])) #FFT 频域平移

        mask1 = torch.ones_like(k_hr)
        mask1[:,:,h//2-math.floor(shape1/2):h//2+math.ceil(shape1/2),w//2-math.floor(shape2/2):w//2+math.ceil(shape2/2)] = 0
        k_out = FSsr*mask1 + k_hr*(1-mask1) # 结合掩码进行滤波
        
        k_out = torch.fft.ifftshift((k_out), dim=[2,3]) # 逆频域平移
        x_res = math.sqrt(h*w) * torch.fft.ifftn(k_out, dim=[2,3]) # 傅里叶逆变换
        x_res = torch.cat([x_res.real,x_res.imag],dim=1) #复数转实数
        return x_res

# # add for INR-IGA change for dual inr
# def compute_NTK(lr, lr_ref, sr, hr, model, ref_type, epoch):

#     #  lr, lr_ref, sr, hr, hr_ref shape = [b,2,h,w]
#     b_lr, c_lr, h_lr, w_lr = lr.shape
#     b_ref, c_ref, h_ref, w_ref = lr_ref.shape
#     b_sr, c_sr, h_sr, w_sr = sr.shape
    
#     # set scale to downsample hr & sr
#     scale_h =  h_lr/h_sr
#     scale_w =  w_lr/w_sr

#     lr = lr.permute(0,2,3,1).reshape(b_lr,-1,c_lr)  # [b,h,w,c] to [b,h*w,c]
#     lr_ref = lr_ref.permute(0,2,3,1).reshape(b_ref,-1,c_ref)   # [b,h,w,c] to [b,h*w,c]  #6 1024  2

#     sr = F.interpolate(sr, scale_factor=(scale_h, scale_w), mode='bilinear') # downsample
#     hr = F.interpolate(hr, scale_factor=(scale_h, scale_w), mode='bilinear')
#     sr = sr.permute(0,2,3,1).reshape(b_sr,-1,c_sr) # 6 1024  2
#     hr = hr.permute(0,2,3,1).reshape(b_sr,-1,c_sr) # 6 1024  2

#     tensor_tar = torch.sum(abs(sr-hr),dim=2,keepdim=True) #[b,h*w,1] 6 1024 1
#     max_indices_tar = torch.argmax(tensor_tar.squeeze(2), dim=1, keepdim=True).squeeze(1) # [b]--> per batch max error

#     lr = lr[:,max_indices_tar,:]  # b,b,2
#     lr_ref = lr_ref[:,max_indices_tar,:] #b,b,2

#     # new_model = model + flatten + linear
#     class NTKWrapper(nn.Module):
#         def __init__(self, base_model):
#             super().__init__()
#             self.base_model = base_model
#             self.reduce = nn.Linear(2, 1, bias=False)
#             nn.init.constant_(self.reduce.weight, torch.tensor(1.0/torch.sqrt(torch.tensor(2))))
#             for param in self.reduce.parameters():
#                 param.requires_grad = False  # fix linear layer
#         def forward(self, lr, hr_ref, lr_ref, ref_type, epoch):
#             inputs = (lr, hr_ref, lr_ref, ref_type, epoch)
#             out, _ = self.base_model(inputs)  # only sr for lr_target is used
#             B, C, H, W = out.shape
#             out = out.permute(0, 2, 3, 1).reshape(-1, C)  # [BHW, C]
#             return self.reduce(out)  # [BHW, 1]
    
#     new_model= NTKWrapper(model).cuda()
#     fmodel,params=make_functional(new_model)
#     def compute_loss(params, lr, lr_ref, ref_type, epoch): # for adapt model input   # shape: [1, 1, 6, in_dim]
#         lr = lr.unsqueeze(0).unsqueeze(0).permute(2,3,0,1)
#         lr_ref = lr_ref.unsqueeze(0).unsqueeze(0).permute(2,3,0,1)
#         hr_ref = F.interpolate(lr_ref, scale_factor=(1/scale_h, 1/scale_w), mode='bilinear')

#         predictions = fmodel(params, lr, hr_ref, lr_ref, ref_type, epoch).sum() 
#         return predictions
    
#     # inputs=inputs.squeeze(0)
#     ft_compute_grad=vmap(grad(compute_loss, argnums=0),(None, 0, 0, None, None))

#     # cal pNTK with current epoch
#     jac1 = ft_compute_grad(params, lr, lr_ref, ref_type, epoch) # model input: lr, ref_hr, ref_lr, self.args.ref_type, epoch

#     jac1=list(jac1)
#     jac1.pop() # Remove the gradients of the additional layer in "new_model" of parameters.
#     jac1 = torch.cat([tensor.flatten(1) if len(tensor.shape) > 2 else tensor for tensor in jac1],dim=1)
#     pNTK =torch.matmul(jac1,jac1.T)  # shape 6*6
#     return pNTK

# # Construction Strategy
# def calculate_precondition_matrix(Gram_matrix,xuhao,replace_start,replace_end):
#     eigenvalues, eigenvectors = torch.linalg.eigh(Gram_matrix) 
#     sorted_indices = torch.argsort(eigenvalues, descending=True)
#     eigenvalues = eigenvalues[sorted_indices]
#     eigenvectors = eigenvectors[:, sorted_indices]
#     # adjusted eigenvalues
#     adjusted_eigenvalues=torch.zeros(len(eigenvalues))
#     S = torch.eye(Gram_matrix.size(0)).cuda()
#     for i in range(replace_start,replace_end):
#         if eigenvalues[i]<=0:
#             break
#         else:
#             adjusted_eigenvalues[i]=eigenvalues[xuhao]
#             S-=(1-adjusted_eigenvalues[i]/eigenvalues[i])*(eigenvectors[:, i].view(-1, 1) @ eigenvectors[:, i].view(-1, 1).T)
#     S = S / S.abs().max()  #归一化 放大数值
#     return S.detach()

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        self.psnr_max = None

        self.DataConsistency = DataConsistency()
    
    # writer = SummaryWriter('./experiment/IXI-gkloss-4/logs')  #recording training process

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # train on integer scale factors (x2, x3, x4) for 1 epoch to maintain stability
        if epoch == 1 and self.args.load == '.':
            self.loader_train.dataset.first_epoch = True
            # adjust learning rate
            lr = 5e-5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # train on all scale factors for remaining epochs
        else:
            self.loader_train.dataset.first_epoch = False
            # adjust learning rate
            lr = self.args.lr * (2 ** -(epoch // 30))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))

        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            if isinstance(lr,list):
                lr, ref_hr, ref_lr, hr = self.prepare(lr[0], lr[1], lr[2], hr)
            else:
                lr, hr = self.prepare(lr, hr)
                ref_hr = None
            scale = hr.size(2) / lr.size(2)
            scale2 = hr.size(3) / lr.size(3)
            timer_data.hold()
            self.optimizer.zero_grad()

            # inference
            self.model.get_model().set_scale(scale, scale2)  #设置模型的缩放比例，根据是否提供参考高分辨率图像ref_hr，分别调用不同的模型进行处理
            if ref_hr is None:
                sr = self.model(lr)
            else:
                sr = self.model((lr, ref_hr, ref_lr, self.args.ref_type, epoch))
            if isinstance(sr,tuple): #检查sr的类型，如果是元组，则将其拆分为sr和Refsr两个变量
                sr, Refsr = sr   
            else:
                Refsr = None
            
            # change for IGA_method
            # pNTK = compute_NTK(lr, ref_lr, sr, hr, self.model, self.args.ref_type, epoch)
            # S = calculate_precondition_matrix(pNTK,2, 0, 3)
            # loss = self.loss(sr, Refsr, None, hr, ref_hr, lr.shape[2], lr.shape[3], S)

            # loss function_ori
            loss = self.loss(sr, Refsr, None, hr, ref_hr, lr.shape[2], lr.shape[3])
            
            # backward
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
                # writer.add_scalar('loss', loss.item(), batch)

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        target = self.model
        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir, 'model', 'model_latest.pt')
        )
        if epoch % self.args.save_every == 0:
            torch.save(
                target.state_dict(),
                os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
            )
            self.ckp.write_log('save ckpt epoch{:.4f}'.format(epoch))

    def test(self):
        self.model.eval()
        # epoch = self.scheduler.last_epoch

        with torch.no_grad():
            if self.args.test_only:
                scale_list = range(len(self.args.scale))
                logger = print
            else:
                scale_list = [9,19,29]
                logger = self.ckp.write_log

            eval_psnr_avg = []
            for idx_scale in scale_list:
                self.loader_test.dataset.set_scale(idx_scale)
                scale = self.args.scale[idx_scale]
                scale2 = self.args.scale2[idx_scale]

                eval_psnr = 0
                eval_ssim = 0
                idx = 0
                for idx_img, (lr, hr, filename, _) in tqdm(enumerate(self.loader_test),total=len(self.loader_test)):
                    idx += 1
                    filename = filename[0]
                    # prepare LR & HR images
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        if isinstance(lr,list):
                            lr, ref_hr, ref_lr, hr = self.prepare(lr[0], lr[1], lr[2], hr)
                        else:
                            lr, hr = self.prepare(lr, hr)
                            ref_hr = None
                            ref_lr = None
                    else:
                        if isinstance(lr,list):
                            lr, ref_hr, ref_lr = self.prepare(lr[0], lr[1], lr[2])
                        else:
                            lr, = self.prepare(lr)
                            ref_hr = None
                            ref_lr = None
                    lr, hr, ref_hr, ref_lr = self.crop_border(lr, hr, ref_hr, ref_lr, scale, scale2)

                    # inference
                    if self.args.test_bicubic:
                        # print('Using bicubic interpolation')
                        sr = F.interpolate(lr, scale_factor=scale, mode='bicubic')
                    
                    else:
                        self.model.get_model().set_scale(scale, scale2)
                        if ref_hr is None:
                            sr = self.model(lr)
                        else:
                            sr = self.model((lr, ref_hr, ref_lr, self.args.ref_type_test))
                        if isinstance(sr,tuple):
                            sr,Refsr = sr                    

                    # if not no_eval:
                    #     save_single = False
                    #     if idx < 10:
                    #         save_single = True
                    #     psnr, ssim, mse = utility.calc_psnr(
                    #         lr, sr,  hr, img_name=filename, scale=[scale, scale2], 
                    #         save = self.args.save_results, save_single = save_single, savefile = self.args.savefigfilename,ref = ref_hr
                    #     )
                    #     eval_psnr += psnr
                    #     eval_ssim += ssim 

                    if not no_eval:  # change for save ref_hr ref_lr
                        save_single = False
                        if idx < 10:
                            save_single = True
                        psnr, ssim, mse = utility.calc_psnr(
                            lr, sr,  hr, ref_hr, ref_lr, img_name=filename, scale=[scale, scale2], 
                            save = self.args.save_results, save_single = save_single, savefile = self.args.savefigfilename,ref = ref_hr
                        )
                        eval_psnr += psnr
                        eval_ssim += ssim


                if scale == scale2:
                    logger('[{} x{}]\tPSNR: {:.4f} SSIM: {:.4f}'.format(
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))
                else:
                    logger('[{} x{}/x{}]\tPSNR: {:.4f} SSIM: {:.4f}'.format(
                        self.args.data_test,
                        scale,
                        scale2,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))
                eval_psnr_avg.append(eval_psnr / len(self.loader_test))
            eval_psnr_avg = np.mean(eval_psnr_avg)
            # writer.add_scalar('PSNR on training data', eval_psnr_avg, epoch) #recording psnr

        if not self.args.test_only: #training mode and save the best model
            if self.psnr_max is None or self.psnr_max < eval_psnr_avg:
                self.psnr_max = eval_psnr_avg
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.ckp.dir, 'model', 'model_best.pt')
                )
                logger('save ckpt PSNR:{:.4f}'.format(eval_psnr_avg))


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def crop_border(self, img_lr, img_hr, img_ref_hr, img_ref_lr, scale, scale2):  #使得lr和hr图像取整，进行图像裁剪
        N, C, H_lr, W_lr = img_lr.size()
        N, C, H_hr, W_hr = img_hr.size()
        H = H_lr if round(H_lr * scale) <= H_hr else math.floor(H_hr / scale)     #scale scale2 约束图像长宽
        W = W_lr if round(W_lr * scale2) <= W_hr else math.floor(W_hr / scale2)

        step = []
        for s in [scale, scale2]:
            if s == int(s):
                step.append(1)
            elif s * 2 == int(s * 2):
                step.append(2)
            elif s * 5 == int(s * 5):
                step.append(5)
            elif s * 10 == int(s * 10):
                step.append(10)
            elif s * 20 == int(s * 20):
                step.append(20)
            elif s * 50 == int(s * 50):
                step.append(50)

        H_new = H // step[0] * step[0]
        if H_new % 2 == 1:
            H_new = H // (step[0] * 2) * step[0] * 2

        W_new = W // step[1] * step[1]
        if W_new % 2 == 1:
            W_new = W // (step[1] * 2) * step[1] * 2

        img_lr = img_lr[:, :, :H_new, :W_new]
        img_hr = img_hr[:, :, :round(scale * H_new), :round(scale2 * W_new)]
        if img_ref_hr is not None:
            img_ref_hr = img_ref_hr[:, :, :round(scale * H_new), :round(scale2 * W_new)]
            img_ref_lr = img_ref_lr[:, :, :H_new, :W_new]
        return img_lr, img_hr, img_ref_hr, img_ref_lr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
