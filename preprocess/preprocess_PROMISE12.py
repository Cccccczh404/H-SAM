import numpy as np

test_split = ['Case16',
    'Case35',
    'Case34',
    'Case49',
    'Case23',
    'Case12',
    'Case28',
    'Case43',
    'Case20',
    'Case13'
    ]

train_split = ['Case29',
    'Case14',
    'Case05'
    ]

train_list = []
for i in train_split:
    data = np.load('./PROMISE/all_data_prepro/' + i + '.npy', allow_pickle=True).item()
    labelall = data['label']
    imageall = data['img']
    gt = labelall.sum(-1).sum(-1)
    pick = np.where(gt > 0)[0]
    for s in pick:
        img = imageall[s, :, :]
        label = labelall[s, :, :]
        if len(str(s)) == 1:
            name = '00' + str(s)
        elif len(str(s)) == 2:
            name = '0' + str(s)
        else:
            name = str(s)
        np.save(r'./PROMISE/train_npz/' + i + '_slice' + name + '.npy', {'image': img, 'label':label})
        train_list.append(i + '_slice' + name)

savep = r'./lists/lists_PROMISE_3/'
with open(savep + 'train.txt',"w") as f:
    for i in train_list:
        f.writelines(i + '\n')
f.close()