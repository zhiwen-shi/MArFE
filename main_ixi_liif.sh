#训练IXI数据集 使用其他对比模型
python main.py --n_GPUs=1  \
               --num_GPUs='0'  \
               --data_train='RefMRI' \
               --name_train='T2mattrain' \
               --data_test='RefMRI' \
               --name_test='T2matval' \
               --dir_data='/media/data1/shizw/MICCAIData'  \
               --loss="1*L1"  \
               --model="hiif"  \
               --save="hiif_IXI_unfold"  \
               --batch_size=6 \
               --patch_size=32 \
               --resume=0  \
               --n_color=2 \
               --rgb_range=1 \
               --ref_mat='IXIPDrefmat' \
               --ref_list='multinameIXI.txt' \
               --pre_train=None \
               --mode=None


# 其他对比模型
#  --model="dualref" 
#  --model="liif" 
#  --model="hiif" 
#  --model="srno" 

# mode 
# pe_liif+relu baseline
# pe+relu+bn
# pe+relu+sin
# pe+relu+sin+bn

# pe+relu+sin2 cat[q,z]  choose this
# pe+relu+sin2+matt cat[q,z]  with multi-heads attention
# pe+relu+sin2+gatt cat[q,z]  with Galerkin-type attention
# pe+relu+sin2+gatt_add cat[q,z]  with Galerkin-type attention + add type
# pe+relu+sin2+gatt_add_fr | fr1 | fr2  with fourier reparameterization
# fpe+relu+sin2+gatt_add  fourier relative positinal encoding
# fpe+relu+sin2+gatt_add+fr2 
# pe+relu+sin2+gatt_add_fr2

#loss
# 1*L1+0.05*KLoss baseline
# 1*L1+0.05*GKLoss+0.1*HFLoss
# 1*L1+0.05*GKLoss+0.05*HFLoss choose this better
# 1*WL1+0.05*WKLoss IGA test

