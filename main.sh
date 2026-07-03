#训练FastMRI数据集
python main.py --n_GPUs=1  \
               --data_train='RefMRI' \
               --name_train='FSPDmattrain' \
               --data_test='RefMRI' \
               --name_test='FSPDmatval' \
               --dir_data='/media/data3/shizw/MICCAIData'  \
               --loss="1*L1+0.05*KLoss"  \
               --model="dualref"  \
               --save=""  \
               --batch_size=6 \
               --patch_size=32 \
               --resume=0  \
               --n_color=2 \
               --rgb_range=1 \
               --ref_mat='fastMRIref_mat' \
               --ref_list='multiname.txt' \
               --pre_train=None

#训练IXI数据集
python main.py --n_GPUs=1  \
               --data_train='RefMRI' \
               --name_train='T2mattrain' \
               --data_test='RefMRI' \
               --name_test='T2matval' \
               --dir_data='/media/data3/shizw/MICCAIData'  \
               --loss="1*L1+0.05*KLoss"  \
               --model="dualref"  \
               --save=""  \
               --batch_size=6 \
               --patch_size=32 \
               --resume=0  \
               --n_color=2 \
               --rgb_range=1 \
               --ref_mat='IXIPDrefmat' \
               --ref_list='multinameIXI.txt' \
               --pre_train=None