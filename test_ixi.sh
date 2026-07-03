# 测试 IXI 数据集
python main.py --n_GPUs=1  \
               --num_GPUs='6'  \
               --data_train='RefMRI' \
               --name_train='T2mattrain' \
               --data_test='RefMRI' \
               --name_test='T2mattest' \
               --dir_data='/media/data1/shizw/MICCAIData'  \
               --resume=0  \
               --n_color=2 \
               --rgb_range=1 \
               --pre_train="./experiment/IXI_pe_liif+relu+sin2+gatt_add+fr2_loss_0.1hf/model/model_best.pt" \
               --test_only \
               --test_bicubic \
               --scale="1.5+2+3+4+6+8" \
               --model="dualref"  \
               --ref_type_test=1 \
               --ref_mat='IXIPDrefmat' \
               --ref_list='multinameIXI.txt' \
               --save_results \
               --savefigfilename="IXI_bicubic" \
               --mode='pe+relu+sin2+gatt_add+fr2'

#   --test_bicubic \