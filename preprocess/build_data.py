import random
import shutil

f=open('./lists/lists_Synapse/train_lenths.txt')
slices=[]
cnt = 0
perc = 0.01
original_path = '/data2/billycheng/datasets/train_npz_new_224/'
new_path = '/data2/billycheng/datasets/train_npz_new_224_10th/'
for line in f:
    sli = line.split('\n')[0]
    shutil.copyfile(original_path+str(sli)+'.npz',new_path+str(sli)+'npz')
