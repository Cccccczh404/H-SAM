
import random

f=open('./lists/lists_Synapse/train_full.txt')
slices=[]
cnt = 0
perc = 0.1
for line in f:
    slices.append(line.strip())
    cnt += 1

sample_num = int(cnt*perc)
slices_sel = random.sample(slices, sample_num) 
with open('./lists/lists_Synapse/train.txt', 'w') as f:
    for sli in slices_sel:
        f.writelines(sli)
        f.writelines('\n')
