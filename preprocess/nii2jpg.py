# -*- coding: utf-8 -*-
# @Time    : 2020/12/30 15:22
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Visualization.py
# 读取mha文件，并将其转化成Image
import SimpleITK as sitk
import os, cv2
 
 
def mha2jpg(save_dir, mhaPath, wc=40, ws=300, name='pred', case='0001'):
    # 使用SimpleITK读取数据，并使用GetArrayFromImage()函数获得图像信息
    image = sitk.ReadImage(mhaPath)
    img_data = sitk.GetArrayFromImage(image)
 
    channel = img_data.shape[0]
    low = wc - ws / 2
    high = wc + ws / 2
 
    all_img = []
    # 将医疗图像中的取值范围设置在（wc - ws / 2， wc + ws / 2）之间
    # 然后归一化0-255之间并保存
    for s in range(channel//2, channel-channel//4):
        slicer = img_data[s, :, :]
        slicer[slicer < low] = low
        slicer[slicer > high] = high
        img = cv2.normalize(slicer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(save_dir, case+'_'+str(s) +name +'_' + '.jpg'), img)
 
    return all_img
 
 
save_dir = './erasing_results/jpg/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
case_list = ['0001','0002','0003','0004','0008','0022','0025','0029','0032','0035','0036','0038']
for case in case_list:
    gz_path_pred = './soft2_results/predictions/case'+case+'_pred.nii.gz'
    new_img_nc = mha2jpg(save_dir, gz_path_pred, name='pred', case = case)
    gz_path_gt = './soft2_results/predictions/case'+case+'_gt.nii.gz'
    new_img_nc = mha2jpg(save_dir, gz_path_gt, name='gt', case = case)
    gz_path_img = './soft2_results/predictions/case'+case+'_img.nii.gz'
    new_img_nc = mha2jpg(save_dir, gz_path_img, name='img', case = case)
 
 
 