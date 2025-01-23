import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch


def get_loss(output, sample):
    y = sample['y']
    if output.shape[-1] == 511:
        y = y[..., :511, :511]
    #output = output[..., :3, :511, :511]
    l1_loss = l1_loss_func(output, y)


    loss = l1_loss
    return loss


def l1_loss_func(pred, gt):
    # return F.l1_loss(pred[mask == 1.], gt[mask == 1.])
    return F.l1_loss(pred, gt)