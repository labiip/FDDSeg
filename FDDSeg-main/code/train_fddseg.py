import argparse
import logging
import os
import random
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from val_2D import test_single_volume_cct, test_single_volume_ds, test_single_volume_cct2
from label_propagation import label_propagation
from active_boundary_loss import active_boundary_loss

def str2bool(s):
    if s not in {'F', 'T'}:
        raise ValueError('Not a valid boolean string')
    return s == 'T'

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDCC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDCC/pCE_SPS', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022,
                    help='random seed')
parser.add_argument('--warmup', default=30000, type=int)
parser.add_argument('--anneal', default='T', type=str2bool)
parser.add_argument('--temp', default=2, type=float)
parser.add_argument('--alp', default=1.0, type=float)
parser.add_argument('--bet', default=0.0, type=float)
# parser.add_argument('num_classes', type=int, default=4, help='num_class')
args = parser.parse_args()

def decompose(input_x, input_y):
    def _one_hot_encoder(num_classes, input_tensor=input_y):
        tensor_list = []
        for i in range(4):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float() + 0.0

    y = _one_hot_encoder(input_y)

    x00 = input_x[:, 0, :, :]
    x11 = input_x[:, 1, :, :]
    x22 = input_x[:, 2, :, :]
    x33 = input_x[:, 3, :, :]

    x = torch.softmax(input_x, dim=1)

    x0 = x[:, 0, :, :]
    x1 = x[:, 1, :, :]
    x2 = x[:, 2, :, :]
    x3 = x[:, 3, :, :]

    y0 = y[:, 0, :, :]
    y1 = y[:, 1, :, :]
    y2 = y[:, 2, :, :]
    y3 = y[:, 3, :, :]

    x0_w = x0 * y0
    x1_w = x1 * y1
    x2_w = x2 * y2
    x3_w = x3 * y3

    y0__ = y0 - 1.0
    y0_ = y0__ * y0__
    x0_w0 = x0 * y0_
    x0_w0 = x0_w0 + (x1 + x2 + x3)

    y1__ = y1 - 1.0
    y1_ = y1__ * y1__
    x1_w0 = x1 * y1_
    x1_w0 = x1_w0 + (x0 + x2 + x3)

    y2__ = y2 - 1.0
    y2_ = y2__ * y2__
    x2_w0 = x2 * y2_
    x2_w0 = x2_w0 + (x0 + x1 + x3)

    y3__ = y3 - 1.0
    y3_ = y3__ * y3__
    x3_w0 = x3 * y3_
    x3_w0 = x3_w0 + (x1 + x2 + x0)

    x0_none = torch.cat([x00 * y0_, x11, x22, x33], 1)
    x1_none = torch.cat([x00, x11 * y1_, x22, x33], 1)
    x2_none = torch.cat([x00, x11, x22 * y2_, x33], 1)
    x3_none = torch.cat([x00, x11, x22, x33 * y3_], 1)

    x0_none = torch.softmax(x0_none, dim=1)
    x1_none = torch.softmax(x1_none, dim=1)
    x2_none = torch.softmax(x2_none, dim=1)
    x3_none = torch.softmax(x3_none, dim=1)

    return x0_w, x1_w, x2_w, x3_w, x0_w0, x1_w0, x2_w0, x3_w0, x0_none, x1_none, x2_none, x3_none

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
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
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type='unet_cct1', in_chns=1, class_num=num_classes)
    model2 = net_factory(net_type='unet_cct2', in_chns=1, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    model.train()
    model2.train()

    optimizer = optim.SGD([{'params': model.parameters()}, {'params':model2.parameters(), 'lr': base_lr}], lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)
    mse_loss = torch.nn.MSELoss(reduction='mean')

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 1.0
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            pseudo_labels, su_mask = label_propagation(volume_batch.cpu(), label_batch.cpu(), 'ACDC')
            pseudo_labels = pseudo_labels.cuda()

            outputs, x_32, x_64, x_128 = model(volume_batch)
            outputs_aux1, aux_32, aux_64, aux_128 = model2(volume_batch)

            outputs_soft1 = torch.softmax(outputs, dim=1)
            outputs_soft2 = torch.softmax(outputs_aux1, dim=1)
            x_64 = torch.softmax(x_64, dim=1)

            t_resize64 = Resize([64, 64])
            t_resize32 = Resize([32, 32])
            t_resize128 = Resize([128, 128])

            if iter_num < args.warmup:
                alp = 0
            else:
                if args.anneal:
                    alp = args.alp * sigmoid_rampup(iter_num+1 - args.warmup, max_iterations - args.warmup)
                else:
                    alp = args.alp

            loss_ce1 = ce_loss(outputs, label_batch[:].long())

            loss_ce2 = ce_loss(outputs_aux1, label_batch[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)

            beta = random.random() + 1e-10
            gama = random.random() + 1e-10

            pseudo_supervision = torch.argmax(
                (beta * outputs_soft1.detach() + (1.0-beta) * outputs_soft2.detach()), dim=1, keepdim=False)

            pseudo_labels_32 = t_resize32(pseudo_supervision.unsqueeze(1)) # pseudo_labels
            pseudo_labels_64 = t_resize64(pseudo_supervision.unsqueeze(1))
            pseudo_labels_128 = t_resize128(pseudo_supervision.unsqueeze(1))

            ### decoupled
            x0_w, x1_w, x2_w, x3_w, x0_w0, x1_w0, x2_w0, x3_w0, x0_none, x1_none, x2_none, x3_none = decompose(x_64, pseudo_labels_64)
            aux_x0_w, aux_x1_w, aux_x2_w, aux_x3_w, aux_x0_w0, aux_x1_w0, aux_x2_w0, aux_x3_w0, aux_x0_none, aux_x1_none, aux_x2_none, aux_x3_none = decompose(aux_64, pseudo_labels_64)
            tckd_0_w = mse_loss(x0_w.unsqueeze(1),aux_x0_w.unsqueeze(1))
            tckd_1_w = mse_loss(x1_w.unsqueeze(1), aux_x1_w.unsqueeze(1))
            tckd_2_w = mse_loss(x2_w.unsqueeze(1), aux_x2_w.unsqueeze(1))
            tckd_3_w = mse_loss(x3_w.unsqueeze(1), aux_x3_w.unsqueeze(1))
            tckd_0_w0 = mse_loss(x0_w0.unsqueeze(1), aux_x0_w0.unsqueeze(1))
            tckd_1_w0 = mse_loss(x1_w0.unsqueeze(1), aux_x1_w0.unsqueeze(1))
            tckd_2_w0 = mse_loss(x2_w0.unsqueeze(1), aux_x2_w0.unsqueeze(1))
            tckd_3_w0 = mse_loss(x3_w0.unsqueeze(1), aux_x3_w0.unsqueeze(1))
            tckd = tckd_0_w+tckd_1_w+tckd_2_w+tckd_3_w+tckd_0_w0+tckd_1_w0+tckd_2_w0+tckd_3_w0
            nckd_0_none = mse_loss(x0_none.unsqueeze(1), aux_x0_none.unsqueeze(1))
            nckd_1_none = mse_loss(x1_none.unsqueeze(1), aux_x1_none.unsqueeze(1))
            nckd_2_none = mse_loss(x2_none.unsqueeze(1), aux_x2_none.unsqueeze(1))
            nckd_3_none = mse_loss(x3_none.unsqueeze(1), aux_x3_none.unsqueeze(1))
            nckd = nckd_0_none+nckd_1_none+nckd_2_none+nckd_3_none
            loss_dkd_64 = 0.4 * tckd + 0.6 * nckd

            x0_w_32, x1_w_32, x2_w_32, x3_w_32, x0_w0_32, x1_w0_32, x2_w0_32, x3_w0_32, x0_none_32, x1_none_32, x2_none_32, x3_none_32 = decompose(x_32, pseudo_labels_32)
            aux_x0_w_32, aux_x1_w_32, aux_x2_w_32, aux_x3_w_32, aux_x0_w0_32, aux_x1_w0_32, aux_x2_w0_32, aux_x3_w0_32, aux_x0_none_32, aux_x1_none_32, aux_x2_none_32, aux_x3_none_32 = decompose(aux_32, pseudo_labels_32)
            tckd_0_w_32 = mse_loss(x0_w_32.unsqueeze(1),aux_x0_w_32.unsqueeze(1))
            tckd_1_w_32 = mse_loss(x1_w_32.unsqueeze(1), aux_x1_w_32.unsqueeze(1))
            tckd_2_w_32 = mse_loss(x2_w_32.unsqueeze(1), aux_x2_w_32.unsqueeze(1))
            tckd_3_w_32 = mse_loss(x3_w_32.unsqueeze(1), aux_x3_w_32.unsqueeze(1))
            tckd_0_w0_32 = mse_loss(x0_w0_32.unsqueeze(1), aux_x0_w0_32.unsqueeze(1))
            tckd_1_w0_32 = mse_loss(x1_w0_32.unsqueeze(1), aux_x1_w0_32.unsqueeze(1))
            tckd_2_w0_32 = mse_loss(x2_w0_32.unsqueeze(1), aux_x2_w0_32.unsqueeze(1))
            tckd_3_w0_32 = mse_loss(x3_w0_32.unsqueeze(1), aux_x3_w0_32.unsqueeze(1))
            tckd_32 = tckd_0_w_32 + tckd_1_w_32 + tckd_2_w_32 + tckd_3_w_32 + tckd_0_w0_32 + tckd_1_w0_32 + tckd_2_w0_32 + tckd_3_w0_32
            nckd_0_none_32 = mse_loss(x0_none_32.unsqueeze(1), aux_x0_none_32.unsqueeze(1))
            nckd_1_none_32 = mse_loss(x1_none_32.unsqueeze(1), aux_x1_none_32.unsqueeze(1))
            nckd_2_none_32 = mse_loss(x2_none_32.unsqueeze(1), aux_x2_none_32.unsqueeze(1))
            nckd_3_none_32 = mse_loss(x3_none_32.unsqueeze(1), aux_x3_none_32.unsqueeze(1))
            nckd_32 = nckd_0_none_32 + nckd_1_none_32 + nckd_2_none_32 + nckd_3_none_32
            loss_dkd_32 = 0.4 * tckd_32 + 0.6 * nckd_32

            x0_w_128, x1_w_128, x2_w_128, x3_w_128, x0_w0_128, x1_w0_128, x2_w0_128, x3_w0_128, x0_none_128, x1_none_128, x2_none_128, x3_none_128 = decompose(x_128, pseudo_labels_128)
            aux_x0_w_128, aux_x1_w_128, aux_x2_w_128, aux_x3_w_128, aux_x0_w0_128, aux_x1_w0_128, aux_x2_w0_128, aux_x3_w0_128, aux_x0_none_128, aux_x1_none_128, aux_x2_none_128, aux_x3_none_128 = decompose(aux_128, pseudo_labels_128)
            tckd_0_w_128 = mse_loss(x0_w_128.unsqueeze(1),aux_x0_w_128.unsqueeze(1))
            tckd_1_w_128 = mse_loss(x1_w_128.unsqueeze(1), aux_x1_w_128.unsqueeze(1))
            tckd_2_w_128 = mse_loss(x2_w_128.unsqueeze(1), aux_x2_w_128.unsqueeze(1))
            tckd_3_w_128 = mse_loss(x3_w_128.unsqueeze(1), aux_x3_w_128.unsqueeze(1))
            tckd_0_w0_128 = mse_loss(x0_w0_128.unsqueeze(1), aux_x0_w0_128.unsqueeze(1))
            tckd_1_w0_128 = mse_loss(x1_w0_128.unsqueeze(1), aux_x1_w0_128.unsqueeze(1))
            tckd_2_w0_128 = mse_loss(x2_w0_128.unsqueeze(1), aux_x2_w0_128.unsqueeze(1))
            tckd_3_w0_128 = mse_loss(x3_w0_128.unsqueeze(1), aux_x3_w0_128.unsqueeze(1))
            tckd_128 = tckd_0_w_128 + tckd_1_w_128 + tckd_2_w_128 + tckd_3_w_128 + tckd_0_w0_128 + tckd_1_w0_128 + tckd_2_w0_128 + tckd_3_w0_128
            nckd_0_none_128 = mse_loss(x0_none_128.unsqueeze(1), aux_x0_none_128.unsqueeze(1))
            nckd_1_none_128 = mse_loss(x1_none_128.unsqueeze(1), aux_x1_none_128.unsqueeze(1))
            nckd_2_none_128 = mse_loss(x2_none_128.unsqueeze(1), aux_x2_none_128.unsqueeze(1))
            nckd_3_none_128 = mse_loss(x3_none_128.unsqueeze(1), aux_x3_none_128.unsqueeze(1))
            nckd_128 = nckd_0_none_128 + nckd_1_none_128 + nckd_2_none_128 + nckd_3_none_128
            loss_dkd_128 = 0.4 * tckd_128 + 0.6 * nckd_128

            loss_dkd = loss_dkd_32 + loss_dkd_64 + loss_dkd_128
            ###

            pseudo_supervision = gama * pseudo_supervision + (1.0 - gama) * pseudo_labels.squeeze(1).long()

            loss_32 = dice_loss(x_32, pseudo_labels_32)
            loss_64 = dice_loss(x_64, pseudo_labels_64)
            loss_128 = dice_loss(x_128, pseudo_labels_128)

            loss_arg_total = 0.0

            loss_pse_sup = 0.5 * (dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(
                1)) + dice_loss(outputs_soft2, pseudo_supervision.unsqueeze(1)))
            loss_back = loss_32 + loss_64 + loss_128

            loss = loss_ce + 0.5 * loss_pse_sup + 0.35 * (loss_32 + loss_64 + loss_128) + 0.15 * loss_dkd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_back', loss_back, iter_num)
            # writer.add_scalar('info/loss_seg1', loss_seg1, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup: %f, loss_arg_total: %f, loss_back: %f, loss_dkd: %f, alpha: %f'%
                (iter_num, loss.item(), loss_ce.item(), loss_pse_sup.item(), loss_arg_total, loss_back, loss_dkd, alpha))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                print('image:', image.shape)
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                metric_list_aux = 0.0
                for i_batch, sampled_batch in enumerate(valloader):

                    metric_i, metric_i_aux = test_single_volume_cct2(
                        sampled_batch["image"], sampled_batch["label"], model, model2, classes=num_classes)

                    metric_list += np.array(metric_i)
                    metric_list_aux += np.array(metric_i_aux)

                metric_list = metric_list / len(db_val)
                metric_list_aux = metric_list_aux / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                performance_aux = np.mean(metric_list_aux, axis=0)[0]
                mean_hd95_aux = np.mean(metric_list_aux, axis=0)[1]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice_man : %f mean_hd95_man : %f' % (iter_num, performance, mean_hd95))
                logging.info(
                    'iteration %d : mean_dice_aux : %f mean_hd95_aux : %f' % (iter_num, performance_aux, mean_hd95_aux))
                model.train()

            if iter_num > 0 and iter_num % 500 == 0:
                if alpha > 0.01:
                    alpha = alpha - 0.01
                else:
                    alpha = 0.01

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
