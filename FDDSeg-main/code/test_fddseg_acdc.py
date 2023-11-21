import argparse
import os
import re
import shutil
import time
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

import numpy as np
# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDCC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDCC/pCE_SPS', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_cct1', help='model_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='fold')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')


def get_fold_idsacdcc(fold):
    all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 41)]
    fold1_training_set = [
        "patient{:0>3}".format(i) for i in range(1, 95)]
    fold1_testing_set = [
        "patient{:0>3}".format(i) for i in range(101, 131)]
    if fold == "fold1":
        return [fold1_training_set, fold1_testing_set]
    else:
        return "ERROR KEY"

def get_fold_ids(fold):
    all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 41)]
    fold1_training_set = [
        "patient{:0>3}".format(i) for i in range(1, 26)]
    fold1_testing_set = [
        "patient{:0>3}".format(i) for i in range(26, 41)]
    if fold == "fold1":
        return [fold1_training_set, fold1_testing_set]
    else:
        return "ERROR KEY"

def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    return dice, hd95, asd

def test_single_volume_keep(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDCC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_aux1,_,_,_ = net(input)
            out_aux1_soft = torch.softmax(out_aux1, dim=1)
            out = torch.argmax(out_aux1_soft, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    org_img_path = "../data/ACDCC_training/{}.nii.gz".format(case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDCC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    arr=(15,4,128,128)
    t=0.0
    prediction_fea = np.zeros(arr)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_aux1,_,_,_ = net(input)
            out_aux1_soft = torch.softmax(out_aux1, dim=1)
            out = torch.argmax(out_aux1_soft, dim=1).squeeze(0)
            # out_fea = x_128_soft.squeeze(0)
            out = out.cpu().detach().numpy()
            out_fea = out_fea.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            pred_fea = out_fea
            prediction[ind] = pred

    case = case.replace(".h5", "")
    org_img_path = "../data/ACDCC_training/{}.nii.gz".format(case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    fea_itk = sitk.GetImageFromArray(prediction_fea.astype(np.float32))
    # fea_itk.CopyInformation(org_img_itk)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

def Inference(FLAGS):
    train_ids, test_ids = get_fold_idsacdcc(FLAGS.fold)
    all_volumes = os.listdir(
        FLAGS.root_path + "/ACDCC_training_volumes")
    image_list = []
    for ids in test_ids:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, all_volumes))
        image_list.extend(new_data_list)
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type)
    test_save_path = "../model/{}_{}/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, 'iter_23600_dice_0.8962.pth')

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        print(case)
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        print(first_metric, second_metric, third_metric)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    print(avg_metric)
    print((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)
    return ((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)[0]


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    total = 0.0
    for i in [1]:
        # for i in [5]:
        FLAGS.fold = "fold{}".format(i)
        print("Inference fold{}".format(i))
        mean_dice = Inference(FLAGS)
        total += mean_dice
    print(total/5.0)

