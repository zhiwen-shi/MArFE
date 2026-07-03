import os
import cv2
import numpy as np

# def myfun(img,x,y,w,h):
#     img1=img[y:y+h,x:x+w]
#     img1 = cv2.resize(img1,(w*2,h*2),interpolation=cv2.INTER_CUBIC)
#     return img1


img = cv2.imread('/home/shizw/Data/Dual-ArbNet/savefigresult/IXI_sin2_gatt_cross/x2.0_2.0/x2.0_2.0_single_fig/IXI012-HH-1211-T2_60_sr.png')
save_dir = '/home/shizw/Data/Dual-ArbNet/2d_sr_results/zoom_in_2d_img_abla_0310'
os.makedirs(save_dir, exist_ok=True)

# for x2
# IXI012-HH-1211-T2_60_sr
# IXI017-Guys-0698-T2_55_sr

# McMRSR  /home/shizw/DJI_Data/zhangjm/Data/MICCAI23/McM/savefigresult/mcm_ixi_szw/x2.0_2.0/sr_IXI012-HH-1211-T2_60.png
# WavTrans /home/shizw/DJI_Data/zhangjm/Data/MICCAI23/WavTrans/savefigresult/wav_ixi_szw/x2.0_2.0/sr_IXI012-HH-1211-T2_60.png
# Meta-SR  /home/shizw/DJI_Data/zhangjm/Data/MICCAI23/Metaloader2/savefigresult/meta_IXI_szw/x2.0_2.0/sr_IXI012-HH-1211-T2_60.png

# LIIF	Data/Dual-ArbNet/savefigresult/IXI_liif
# Diinn  /home/shizw/DJI_Data/zhangjm/Data/MICCAI23/WACV/savefigresult/wacv_IXI_szw/x2.0_2.0/sr_IXI012-HH-1211-T2_60.png 
# MArFE /home/shizw/Data/Dual-ArbNet/savefigresult/IXI_sin2_gatt_add_fr2_totalloss0.05hf/x2.0_2.0/x2.0_2.0_single_fig/IXI012-HH-1211-T2_60_sr.png


# for x6
# IXI013-HH-1212-T2_69_hr
# IXI017-Guys-0698-T2_50_hr

# McMRSR  /home/shizw/DJI_Data/zhangjm/Data/MICCAI23/McM/savefigresult/mcm_ixi_szw_2/x6.0_6.0/sr_IXI012-HH-1211-T2_60.png
# WavTrans /home/shizw/DJI_Data/zhangjm/Data/MICCAI23/WavTrans/savefigresult/wav_ixi_szw_2/x6.0_6.0/sr_IXI012-HH-1211-T2_60.png
# Meta-SR  /home/shizw/DJI_Data/zhangjm/Data/MICCAI23/Metaloader2/savefigresult/meta_IXI_szw/x6.0_6.0/sr_IXI012-HH-1211-T2_60.png

# LIIF	Data/Dual-ArbNet/savefigresult/IXI_liif
# Diinn  /home/shizw/DJI_Data/zhangjm/Data/MICCAI23/WACV/savefigresult/wacv_IXI_szw/x6.0_6.0/sr_IXI012-HH-1211-T2_60.png 
# MArFE /home/shizw/Data/Dual-ArbNet/savefigresult/IXI_sin2_gatt_add_fr2_totalloss0.05hf/x6.0_6.0/x6.0_6.0_single_fig/IXI012-HH-1211-T2_60_sr.png


# for x2 x4 x6 INR Ablation
# IXI012-HH-1211-T2_60_sr
# IXI017-Guys-0698-T2_50_hr

# Vanilla INR  /home/shizw/Data/Dual-ArbNet/savefigresult/IXI_test
# decoder+ /home/shizw/Data/Dual-ArbNet/savefigresult/IXI_sin2/x6.0_6.0/x6.0_6.0_single_fig/IXI017-Guys-0698-T2_50_sr.png
# rpe /home/shizw/Data/Dual-ArbNet/savefigresult/IXI_sin2_gatt_cross/x6.0_6.0/x6.0_6.0_single_fig/IXI017-Guys-0698-T2_50_sr.png


sp = img.shape  # hwc
print(sp)

img_h = sp[0] 
img_w = sp[1] 


# for sr 1 x2
x = 120
y = 100
w = 70
h = 70

# # for lr 1 x2
# x = 120 // 2
# y = 100 // 2
# w = 70 // 2
# h = 70 // 2

# # for sr 2 x2
# x = 80
# y = 60
# w = 70
# h = 70

# # for lr 2 x2
# x = 80 // 2
# y = 60 // 2
# w = 70 // 2
# h = 70 // 2


# # for sr 1 x6
# x = 80
# y = 110
# w = 70
# h = 70

# # for lr 1 x6
# x = 80 // 6
# y = 110 // 6
# w = 70 // 6
# h = 70 // 6

# # for sr 2 x6
# x = 80
# y = 60
# w = 70
# h = 70

# for lr 2 x6
# x = 80 // 6
# y = 60 // 6
# w = 70 // 6
# h = 70 // 6



# save gt with rect
# gt_zoom_in = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
# gt_zoom_in_save_path = os.path.join(save_dir, "gt_2d_seg_with_rect_6x_2.png")
# cv2.imwrite(gt_zoom_in_save_path, gt_zoom_in)

# save sr with zoom_in
sr_zoom_in = img[y:y+h,x:x+w]
sr_zoom_in = cv2.resize(sr_zoom_in,(img_w,img_h))
sr_zoom_in_save_path = os.path.join(save_dir, "rpe_2d_sr_zoom_in_2x_1.png")
cv2.imwrite(sr_zoom_in_save_path, sr_zoom_in)

# save sr with ori+zoom_in 
# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
# img_big = myfun(img,x,y,w,h)
# img_big = cv2.resize(img_big,(img_w,img_h))
# img2 = cv2.resize(img,(img_w*2,img_h))
# img2[0:img_h,0:img_w] = img
# img2[0:img_h,img_w:img_w*2] = img_big
# cv2.rectangle(img2, (img_w, 0), (img_w*2, img_h), (0, 0, 255), 1)
# sr_zoom_in_save_path = os.path.join(save_dir, "gt_2d_sr_zoom_in_2x_box.png")
# cv2.imwrite(sr_zoom_in_save_path, img2)

# # change size for svg
# img = cv2.imread('/home/shizw/Data/Dual-ArbNet/ants_2d_seg_results/zoom_in_2d_img/sr_2d_seg_zoom_in_bicubic_2x.png')
# img = cv2.resize(img,(256,256))
# cv2.imwrite('/home/shizw/Data/Dual-ArbNet/ants_2d_seg_results/zoom_in_2d_img/size_change/sr_2d_seg_zoom_in_bicubic_2x_256.png',img)



# 各算法位置
# MArFE 13	Data/Dual-ArbNet/savefigresult/IXI_sin2_gatt_add_fr2_totalloss0.05hf
# HIIF	13	Data/Dual-ArbNet/savefigresult/IXI_hiif
# LIIF	13	Data/Dual-ArbNet/savefigresult/IXI_liif
# Bicubic	13	Data/Dual-ArbNet/savefigresult/IXI_bicubic
# Diinn	13	DJI_Data/zhangjm/Data/MICCAI23/WACV
# SRNO	13	Data/Dual-ArbNet/savefigresult/IXI_srno
# DualArb	13	Data/Dual-ArbNet/savefigresult/IXI_test
# Meta-SR	13	DJI_Data/zhangjm/Data/MICCAI23/Metaloader2
# WavTrans	13	DJI_Data/zhangjm/Data/MICCAI23/WavTrans           2x→1.5x   4x→6x+8x
# McMRSR	13	DJI_Data/zhangjm/Data/MICCAI23/McM
