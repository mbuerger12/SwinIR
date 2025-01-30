import os
import csv
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import torch

def to_cuda(sample):
    sampleout = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            sampleout[key] = val.cuda()
        elif isinstance(val, list):
            new_val = []
            for e in val:
                if isinstance(e, torch.Tensor):
                    new_val.append(e.cuda())
                else:
                    new_val.append(val)
            sampleout[key] = new_val
        else:
            sampleout[key] = val
    return sampleout


def seed_all(seed):
    # Fix all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def new_log(folder_path, args=None):
    os.makedirs(folder_path, exist_ok=True)
    n_exp = len(os.listdir(folder_path))
    randn  = round((time.time()*1000000) % 1000)
    experiment_folder = os.path.join(folder_path, f'experiment_{n_exp}_{randn}')
    os.mkdir(experiment_folder)

    if args is not None:
        args_dict = args.__dict__
        write_params(args_dict, os.path.join(experiment_folder, 'args' + '.csv'))

    return experiment_folder, n_exp, randn


def write_params(params, path):
    with open(path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['key', 'value'])
        for data in params.items():
            writer.writerow([el for el in data])

def store_images(image_folder, experiment_folder, pred, label, epoch):
    """
    Stores the label and multiple predictions in an epoch-specific subfolder:
    <image_folder>/<experiment_folder>/epoch_{epoch}/

    Args:
        image_folder (str): Base folder where images are stored.
        experiment_folder (str): Name of the specific experiment folder.
        preds (list or tuple of torch.Tensor): Multiple prediction tensors.
        label (torch.Tensor): Label tensor.
        epoch (int): Current epoch number.
    """
    # 1. Create the base experiment folder
    experiment_path = os.path.join(image_folder, experiment_folder)
    os.makedirs(experiment_path, exist_ok=True)

    # 2. Create the epoch-specific directory: e.g., "epoch_0", "epoch_1", etc.
    epoch_dir = os.path.join(experiment_path, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # 3. Store the label
    index = str(len(os.listdir(epoch_dir)))
    plot_tensor_image(label, epoch_dir, title=f"label_{index}")

    # 4. Loop over predictions and store each one
    plot_tensor_image(pred, epoch_dir, title=f"prediction_{index}")


def plot_tensor_image(img_tensor, path, title="Image", cmap="viridis", slice_idx=0):
    """
    Plots the given image tensor and saves it as a PNG to the given path.
    """
    # Handle batch dimension
    if len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1:
        # (1, C, H, W) => (C, H, W)
        img_tensor = img_tensor[0]

    if len(img_tensor.shape) == 4:
        # For (N, C, H, W) with N>1, pick a slice_idx if needed
        if slice_idx < 0 or slice_idx >= img_tensor.shape[0]:
            raise ValueError(f"Invalid slice_idx {slice_idx} for shape {img_tensor.shape}")
        img_tensor = img_tensor[slice_idx]

    # Convert tensor to numpy on CPU
    img = img_tensor.detach().cpu().numpy()

    # If the shape is (C, H, W) => convert to (H, W, C)
    if len(img.shape) == 3 and img.shape[0] in [1, 3]:
        img = img.transpose(1, 2, 0)  # (C, H, W) => (H, W, C)

        # If single channel, squeeze out
        if img.shape[-1] == 1:
            img = img[..., 0]

    plt.figure(figsize=(6, 6))
    if len(img.shape) == 2:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")

    # Save
    save_path = os.path.join(path, f"{title}.png")
    plt.savefig(save_path)
    plt.close()  # close the figure to free memory
