# ref_file = '~/Data/MICCAIData/multiname.txt'
ref_file = '/media/data3/shizw/MICCAIData/multiname.txt'
with open(ref_file, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                lr, ref = line.split(' ')
                print('success')