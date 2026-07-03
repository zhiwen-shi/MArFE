# # ref_file = '~/Data/MICCAIData/multiname.txt'
# ref_file = '/media/data3/shizw/MICCAIData/multiname.txt'
# with open(ref_file, 'r') as f:
#             for line in f.readlines():
#                 line = line.strip('\n')
#                 lr, ref = line.split(' ')
#                 print('success')


#test for ants
import ants
# import antspynet
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v3 as iio

# sr_path = "/home/shizw/Data/Dual-ArbNet/savefigresult/IXI_bicubic/x2.0_2.0/x2.0_2.0_single_fig/IXI017-Guys-0698-T2_55_sr.png"  
# gt_path = "/home/shizw/Data/Dual-ArbNet/savefigresult/IXI_bicubic/x2.0_2.0/x2.0_2.0_single_fig/IXI017-Guys-0698-T2_55_hr.png"

# for zhanjm

#2x
# sr_path = "/home/shizw/DJI_Data/zhangjm/Data/MICCAI23/McM/savefigresult/mcm_ixi_szw/x2.0_2.0/sr_IXI017-Guys-0698-T2_55.png"  
# gt_path = "/home/shizw/DJI_Data/zhangjm/Data/MICCAI23/McM/savefigresult/mcm_ixi_szw/x2.0_2.0/hr_IXI017-Guys-0698-T2_55.png"

#4x 6x
# sr_path = "/home/shizw/DJI_Data/zhangjm/Data/MICCAI23/McM/savefigresult/mcm_ixi_szw_2/x4.0_4.0/sr_IXI017-Guys-0698-T2_55.png"  
# gt_path = "/home/shizw/DJI_Data/zhangjm/Data/MICCAI23/McM/savefigresult/mcm_ixi_szw_2/x4.0_4.0/hr_IXI017-Guys-0698-T2_55.png"

# sr_path = "/home/shizw/DJI_Data/zhangjm/Data/MICCAI23/McM/savefigresult/mcm_ixi_szw_2/x6.0_6.0/sr_IXI017-Guys-0698-T2_55.png"  
# gt_path = "/home/shizw/DJI_Data/zhangjm/Data/MICCAI23/McM/savefigresult/mcm_ixi_szw_2/x6.0_6.0/hr_IXI017-Guys-0698-T2_55.png"

# for zhangjm arsr
sr_path = "/home/shizw/DJI_Data/zhangjm/Data/MICCAI23/WACV/savefigresult/wacv_IXI_szw/x2.0_2.0/sr_IXI017-Guys-0698-T2_55.png"  
gt_path = "/home/shizw/DJI_Data/zhangjm/Data/MICCAI23/WACV/savefigresult/wacv_IXI_szw/x2.0_2.0/hr_IXI017-Guys-0698-T2_55.png"


output_dir = "/home/shizw/Data/Dual-ArbNet/ants_2d_seg_results"
os.makedirs(output_dir, exist_ok=True)

COLOR_MAP = {
    0: (0, 0, 0),       # 背景 - 黑色
    1: (0, 154, 222),     # 脑脊液(CSF) - 蓝色  绿色 22, 146, 111
    2: (248, 230, 32),     # 灰质(GM) - 黄色
    3: (227, 26, 28)      # 白质(WM) - 红色
    }

# 2d read
voxel_size_2d = (0.94, 0.94) # for IXI
def read_png_to_ants_2d(png_path, voxel_size):
    png_data = iio.imread(png_path, mode="L")  # mode="L" 表示8位灰度图
    png_data = png_data.astype(np.float32)
    png_data = (png_data - png_data.min()) / (png_data.max() - png_data.min())
    ants_img = ants.from_numpy(png_data, spacing=voxel_size)
    return ants_img


sr_2d_img = read_png_to_ants_2d(sr_path, voxel_size_2d)
gt_2d_img = read_png_to_ants_2d(gt_path, voxel_size_2d)



# 2D seg
def ants_2d_segmentation(ants_2d_img):

    # ANTs 2D GM/WM/CSF
    mask = ants.get_mask(ants_2d_img)  # 2d mask  仅保留脑组织，去除背景
    seg_result = ants.atropos(
        a=ants_2d_img,          # 2D 输入图像
        m='[0.2,1x1]',          # 2D 空间平滑（1x1 对应 2D 核）
        c='[5,0]',              # 迭代次数
        i='kmeans[3]',          # 3 类分割（GM/WM/CSF）
        x=mask,                 # 2D 掩码
    )
    
    seg_image_2d = seg_result['segmentation']  # 2D 分割结果
    prob_images_2d = seg_result['probabilityimages']  # 2D 概率图
    return seg_image_2d, prob_images_2d


sr_seg_2d_img, sr_prob_2d_imgs = ants_2d_segmentation(sr_2d_img)
gt_seg_2d_img, gt_prob_2d_imgs = ants_2d_segmentation(gt_2d_img)


# save colour results
def seg_2d_to_colored_png(seg_2d_img, color_map):

    seg_np = seg_2d_img.numpy()
    h, w = seg_np.shape
    colored_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_val, rgb in color_map.items():
        mask = seg_np == class_val
        colored_img[mask] = rgb

    return colored_img
    
colored_output_path = os.path.join(output_dir, "sr_2d_seg_result_colored.png")
colored_img = seg_2d_to_colored_png(sr_seg_2d_img, COLOR_MAP)
iio.imwrite(colored_output_path, colored_img)

# colored_output_path = os.path.join(output_dir, "gt_2d_seg_result_colored.png")
# seg_2d_to_colored_png(gt_seg_2d_img, colored_output_path, COLOR_MAP)


# cal dice 
def calculate_2d_seg_dice(seg_pred_img, seg_gt_img):

    def dice_coefficient(mask_pred, mask_gt):
        mask_pred = (mask_pred > 0).astype(np.float32)
        mask_gt = (mask_gt > 0).astype(np.float32)
        intersection = np.sum(mask_pred * mask_gt)
        union = np.sum(mask_pred) + np.sum(mask_gt)
        if union == 0:
            return 1.0  
        return 2 * intersection / union


    seg_pred_np = seg_pred_img.numpy()
    seg_gt_np = seg_gt_img.numpy()

    # CSF: 1, GM: 2, WM: 3

    csf_pred = (seg_pred_np == 1).astype(np.uint8)
    csf_gt = (seg_gt_np == 1).astype(np.uint8)

    gm_pred = (seg_pred_np == 2).astype(np.uint8)
    gm_gt = (seg_gt_np == 2).astype(np.uint8)
    
    wm_pred = (seg_pred_np == 3).astype(np.uint8)
    wm_gt = (seg_gt_np == 3).astype(np.uint8)

    csf_dice = dice_coefficient(csf_pred, csf_gt)
    gm_dice = dice_coefficient(gm_pred, gm_gt)
    wm_dice = dice_coefficient(wm_pred, wm_gt)
    avg_dice = (gm_dice + wm_dice + csf_dice) / 3
    
    # 保留4位小数
    return {
        "CSF_dice": round(csf_dice, 4),
        "GM_dice": round(gm_dice, 4),
        "WM_dice": round(wm_dice, 4),
        "Average_dice": round(avg_dice, 4)
    }

dice_result = calculate_2d_seg_dice(sr_seg_2d_img, gt_seg_2d_img)


print("2D分割Dice系数结果")
for key, value in dice_result.items():
    print(f"  {key}: {value}")


# save results
# seg_2d_np = seg_2d_img.numpy()
# seg_2d_png = (seg_2d_np / seg_2d_np.max() * 255).astype(np.uint8)
# iio.imwrite(os.path.join(output_dir, "2d_seg_result.png"), seg_2d_png)



# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.imshow(ants_2d_img.numpy(), cmap="gray")
# ax1.set_title("Original 2D MRI Slice (PNG)")
# ax1.axis("off")
# ax2.imshow(seg_2d_img.numpy(), cmap="viridis")
# ax2.set_title("2D Segmentation (GM/WM/CSF)")
# ax2.axis("off")
# plt.savefig(os.path.join(output_dir, "2d_seg_vis.png"), dpi=300, bbox_inches="tight")
# plt.close()


# # 计算 2D 分割的各组织面积
# def analyze_2d_seg(seg_2d_img, voxel_size):

#     gm_mask = ants.threshold_image(seg_2d_img, 1, 1)
#     wm_mask = ants.threshold_image(seg_2d_img, 2, 2)
#     csf_mask = ants.threshold_image(seg_2d_img, 3, 3)
    
#     # 计算 2D 面积（体素大小×像素数，单位：mm²）
#     pixel_area = voxel_size[0] * voxel_size[1]
#     gm_area = gm_mask.sum() * pixel_area
#     wm_area = wm_mask.sum() * pixel_area
#     csf_area = csf_mask.sum() * pixel_area
    
#     return {
#         "GM_area(mm²)": round(gm_area, 2),
#         "WM_area(mm²)": round(wm_area, 2),
#         "CSF_area(mm²)": round(csf_area, 2)
#     }

# seg_2d_analysis = analyze_2d_seg(seg_2d_img, voxel_size_2d)
# print("2D PNG 分割结果（面积）")
# for k, v in seg_2d_analysis.items():
#     print(f"{k}: {v}")
