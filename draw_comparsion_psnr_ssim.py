from matplotlib import pyplot as plt



x = [91.88,357.12,148.99,57.18,76.18,46.75]  # infer time
y = [37.768,33.304,34.940,31.414,32.101,32.569]  # psnr
y_2 = [0.967,0.914,0.958,0.916,0.934,0.936]  # ssim

colors = ['red', 'orange','green','c','blue','purple']  # 建立颜色列表
labels = ['Ours','McMRSR', 'WavTrans','Meta-SR','LIIF','Diinn']  # 建立标签类别列表
markers=['o','^','p','v','D','*'] #建立形状列表

save_path = '/home/shizw/Data/Dual-ArbNet/2d_sr_results'

# # for psnr
# for i in range(5):
#     plt.scatter(x[i], y[i], c=colors[i], label=labels[i],marker=markers[i], s=250)

# plt.title('PSNR Comparison on IXI',fontsize=20)
# plt.xlabel('Time (ms)',fontsize=14)
# plt.grid(linestyle=':')

# plt.legend(loc=1,labelspacing=1) #是否保存画图标签,4为图例位置在右下

# plt.tick_params(labelsize=14)
# plt.savefig(save_path+'/psnr_time_plot.png', dpi=300, bbox_inches='tight')
# plt.close()


# for ssim
for i in range(5):
    plt.scatter(x[i], y_2[i], c=colors[i], label=labels[i],marker=markers[i], s=250)

plt.title('SSIM Comparison on IXI',fontsize=20)
plt.xlabel('Time (ms)',fontsize=14)
plt.grid(linestyle=':')

plt.legend(loc=1,labelspacing=1) #是否保存画图标签,4为图例位置在右下 1为右上
plt.tick_params(labelsize=14)
plt.savefig(save_path+'/ssim_time_plot.png', dpi=300, bbox_inches='tight')
plt.close()



# colors = ['red', 'orange','deeppink','green','c','blue','purple']  # 建立颜色列表
# labels = ['PRN', 'MSR','MSR_m','MSRCR','SSR','ACE','HE']  # 建立标签类别列表
# markers=['o','^','p','v','s','D','*'] #建立形状列表