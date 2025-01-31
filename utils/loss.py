import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch


def get_loss(output, sample):
    y = sample['y']
    mask = sample['mask_label']
    l1_loss = l1_loss_func(output, y, mask)


    loss = l1_loss * 10
    return loss


def l1_loss_func(pred, gt, mask):
    return F.l1_loss(pred[mask == 1.], gt[mask == 1.])
    #return F.l1_loss(pred, gt)