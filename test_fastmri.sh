# 测试 FastMRI 数据集
python main.py --n_GPUs=1  \
               --num_GPUs='6'  \
               --data_train='RefMRI' \
               --name_train='FSPDmattrain' \
               --data_test='RefMRI' \
               --name_test='FSPDmattest' \
               --dir_data='/media/data1/shizw/MICCAIData'  \
               --resume=0  \
               --n_color=2 \
               --rgb_range=1 \
               --pre_train="./experiment/FastMRI_pe_liif+relu+sin2+gatt_add+fr2_loss/model/model_best.pt" \
               --test_only \
               --scale="2+3+4" \
               --model="dualref"  \
               --ref_type_test=1 \
               --ref_mat='fastMRIref_mat' \
               --ref_list='multiname.txt' \
               --save_results \
               --savefigfilename="FastMRI_0802" \
               --mode='pe+relu+sin2+gatt_add+fr2'

#   --test_bicubic \
