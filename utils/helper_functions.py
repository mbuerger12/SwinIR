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

def store_images(image_folder, experiment_folder, pred, label):
    experiment_folder = os.path.join(image_folder, experiment_folder)
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    dir_name = os.path.join(experiment_folder, f"epoch_{str(len(os.listdir(experiment_folder)))}")
    os.mkdir(dir_name)
    plot_tensor_image(pred, dir_name, title="pred")
    plot_tensor_image(label, dir_name, title="y")


def plot_tensor_image(img_tensor, path, title="Image", cmap="viridis", slice_idx=0, ):
    """
    Plots the given image tensor.

    Parameters:
        img_tensor (torch.Tensor): The tensor to plot. Shape can be
            (N, C, H, W), (C, H, W), (H, W), or (1, C, H, W).
        title (str): Title of the plot.
        cmap (str): Colormap for grayscale images (default: 'viridis').
        slice_idx (int): The index of the slice to plot if the input has multiple slices (default: 0).
    """
    # Handle batch dimension (N, C, H, W) or (1, C, H, W)
    if len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1:  # Single batch
        img_tensor = img_tensor[0]  # Remove batch dimension

    if len(img_tensor.shape) == 4:  # Batch of channels (C, H, W)
        # Select the specified slice along the channel dimension
        if slice_idx < 0 or slice_idx >= img_tensor.shape[0]:
            raise ValueError(f"Invalid slice_idx {slice_idx} for tensor with shape {img_tensor.shape}")
        img_tensor = img_tensor[slice_idx]  # Select the desired channel

    # Move tensor to CPU and convert to NumPy
    img = img_tensor.detach().cpu().numpy()

    # Handle different shapes
    if len(img.shape) == 3:  # Multi-channel image (C, H, W)
        img = img.transpose(1, 2, 0)  # Convert to (H, W, C)
        if img.shape[2] == 1:  # Single channel, convert to 2D
            img = img.squeeze(-1)

    elif len(img.shape) != 2:  # If not (H, W) or (H, W, C), raise error
        raise ValueError(f"Unsupported tensor shape after processing: {img_tensor.shape}")
    """
    # Normalize image for display if needed
    if img.max() > 1 or img.min() < 0:
        img = (img - img.min()) / (img.max() - img.min())
    """
    # Plot the image
    plt.figure(figsize=(6, 6))
    if len(img.shape) == 2:  # Grayscale image
        plt.imshow(img, cmap=cmap)
    else:  # RGB image
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")

    save_path = os.path.join(path, title) + ".png"
    plt.savefig(save_path)
    plt.show()