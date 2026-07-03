# 测试 FastMRI 数据集
python main.py --n_GPUs=1  \
               --data_train='RefMRI' \
               --name_train='FSPDmattrain' \
               --data_test='RefMRI' \
               --name_test='FSPDmattest' \
               --dir_data='/media/data3/shizw/MICCAIData'  \
               --resume=0  \
               --n_color=2 \
               --rgb_range=1 \
               --pre_train="./experiment/fastMRI/model/model_best.pt" \
               --test_only \
               --scale="1.5+2+3+4+6+8" \
               --model="dualref"  \
               --ref_type_test=1 \
               --ref_mat='fastMRIref_mat' \
               --ref_list='multiname.txt' \
               --save_results  \
               --savefigfilename="fastMRI"

# 测试 IXI 数据集
python main.py --n_GPUs=1  \
               --data_train='RefMRI' \
               --name_train='T2mattrain' \
               --data_test='RefMRI' \
               --name_test='T2mattest' \
               --dir_data='/media/data3/shizw/MICCAIData'  \
               --resume=0  \
               --n_color=2 \
               --rgb_range=1 \
               --pre_train="./experiment/IXI/model/model_best.pt" \
               --test_only \
               --scale="1.5+2+3+4+6+8" \
               --model="dualref"  \
               --ref_type_test=1 \
               --ref_mat='IXIPDrefmat' \
               --ref_list='multinameIXI.txt' \
               --save_results  \
               --savefigfilename="IXI"