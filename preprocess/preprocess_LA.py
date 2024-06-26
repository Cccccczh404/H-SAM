import numpy as np
import h5py


f = open('./LA/data/test.list',"r")
test_split = f.readlines()
f.close()

for i in test_split:
    path = './LA/data/2018LA_Seg_Training Set/' + i[:-1] + '/mri_norm2.h5'
    data = h5py.File(path)
    image = data['image'][:]
    label = data['label'][:]
    np.save('./LA/test_vol_h5/' + i[:-1] + '.npy', {'img': np.transpose(image, (2, 0, 1)), 'label': np.transpose(label, (2, 0, 1))})

import os
f = open('./LA/data/train.list',"r")
train_split = f.readlines()
f.close()
train_split = list(np.array(train_split)[[0, 1, 2, 3]])
os.makedirs(r'./LA/train_npz/', exist_ok=True)
train_list = []
for i in train_split:
    path = './LA/data/2018LA_Seg_Training Set/' + i[:-1] + '/mri_norm2.h5'
    data = h5py.File(path)
    imageall = data['image'][:]
    labelall = data['label'][:]
    gt = labelall.sum(0).sum(0)
    pick = np.where(gt > 0)[0]
    for s in pick:
        img = imageall[:, :, s]
        label = labelall[:, :, s]
        if len(str(s)) == 1:
            name = '00' + str(s)
        elif len(str(s)) == 2:
            name = '0' + str(s)
        else:
            name = str(s)
        np.save(r'./LA/train_npz/' + i[:-1] + '_slice' + name + '.npy', {'image': img, 'label':label})
        train_list.append(i[:-1] + '_slice' + name)

savep = r'./lists/lists_LA_4/'
with open(savep + 'train.txt',"w") as f:
    for i in train_list:
        f.writelines(i + '\n')
f.close()