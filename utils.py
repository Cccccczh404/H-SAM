import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic
import csv


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1, stage=2, mode='test'):
    lab = label.squeeze(0)
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            l = lab[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != input_size[0] or y != input_size[1]:
                slice = zoom(slice, (input_size[0] / x, input_size[1] / y), order=3)  # previous using 0
            new_x, new_y = slice.shape[0], slice.shape[1]  # [input_size[0], input_size[1]]
            if new_x != patch_size[0] or new_y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / new_x, patch_size[1] / new_y), order=3)  # previous using 0, patch_size[0], patch_size[1]
            inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            net.eval()
            with torch.no_grad():
                outputs1,outputs2,_,_ = net(inputs, multimask_output, patch_size[0], gt=None, mode='test')
                if stage == 3:
                    output_masks = (outputs1['masks'] + outputs2['masks'])/2
                elif stage == 2:
                    output_masks = outputs2['masks']

                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                out_h, out_w = out.shape
                if x != out_h or y != out_w:
                    pred = zoom(out, (x / out_h, y / out_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred
        # only for debug
        # if not os.path.exists('/output/images/pred'):
        #     os.makedirs('/output/images/pred')
        # if not os.path.exists('/output/images/label'):
        #     os.makedirs('/output/images/label')
        # assert prediction.shape[0] == label.shape[0]
        # for i in range(label.shape[0]):
        #     imageio.imwrite(f'/output/images/pred/pred_{i}.png', prediction[i])
        #     imageio.imwrite(f'/output/images/label/label_{i}.png', label[i])
        # temp = input('kkpsa')
    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    metric_list_dice = []
    metric_list_hd = []
    for i in range(1, classes + 1):
        metric_list_dice.append(calculate_metric_percase(prediction == i, label == i)[0])
        metric_list_hd.append(calculate_metric_percase(prediction == i, label == i)[1])
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        with open(test_save_path + '/' + 'dice' + ".csv",'a+',newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(metric_list_dice)
    return metric_list

def mask_latent_code_spatial_wise(latent_code, loss, percentile=1 / 3.0, random=False, loss_type='corr', if_detach=True, if_soft=False):
    '''
    given a latent code return a perturbed code where top % areas are masked 
    '''
    use_gpu = True if latent_code.device != torch.device('cpu') else False
    code = latent_code
    num_images = code.size(0)
    spatial_size = code.size(2) * code.size(3)
    H, W = code.size(2), code.size(3)

    gradient = torch.autograd.grad(loss, [code])[0]
    # mask gradient with largest response:
    spatial_mean = torch.mean(gradient, dim=1, keepdim=True)
    spatial_mean = spatial_mean.squeeze().view(num_images, spatial_size)

    # select the threshold at top XX percentile
    if random:
        percentile = np.random.rand() * percentile

    vector_thresh_percent = int(spatial_size * percentile)
    vector_thresh_value = torch.sort(spatial_mean, dim=1, descending=True)[
        0][:, vector_thresh_percent]

    vector_thresh_value = vector_thresh_value.view(
        num_images, 1).expand(num_images, spatial_size)

    if if_soft:
        vector = torch.where(spatial_mean > vector_thresh_value,
                             0.5 * torch.rand_like(spatial_mean),
                             torch.ones_like(spatial_mean))
    else:
        vector = torch.where(spatial_mean > vector_thresh_value,
                             torch.zeros_like(spatial_mean),
                             torch.ones_like(spatial_mean))

    mask_all = vector.view(num_images, 1, H, W)
    if not if_detach:
        masked_latent_code = latent_code * mask_all
    else:
        masked_latent_code = code * mask_all

    try:
        decoder_function.zero_grad()
    except:
        pass
    return masked_latent_code, mask_all

def set_grad(module, requires_grad=False):
    for p in module.parameters():  # reset requires_grad
        p.requires_grad = requires_grad
