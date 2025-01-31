import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# from your local code

import os
import tifffile
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from torchvision.transforms.functional import crop
from torchvision.transforms import RandomCrop

from .geo_tifffile import read_geotiff3D  # or wherever your read_geotiff3D is
from osgeo import gdal
import matplotlib.pyplot as plt


def apply_random_crop_lr_hr(
    lr_tensor: torch.Tensor,
    hr_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    lowres_crop_size: int,
    upscale_factor: int
):
    """
    Takes an LR image (lr_tensor) and the corresponding HR image (hr_tensor),
    crops them randomly so they match in spatial location, then returns the cropped versions.

    Args:
        lr_tensor: Low-resolution tensor of shape [C, H, W]
        hr_tensor: High-resolution tensor of shape [C, H, W] (or bigger)
        mask_tensor: Corresponding mask of shape [C, H, W]
        lowres_crop_size: how many pixels to crop from the LR image
        upscale_factor: ratio of HR resolution to LR resolution
    Returns:
        (lr_cropped, hr_cropped, mask_cropped)
    """
    # LR random crop
    i, j, h, w = RandomCrop.get_params(lr_tensor, output_size=(lowres_crop_size, lowres_crop_size))
    lr_cropped = crop(lr_tensor, i, j, h, w)

    # For HR, crop the corresponding region
    # We simply multiply the same offsets by the upscale_factor
    i_hr, j_hr = i * upscale_factor, j * upscale_factor
    h_hr, w_hr = h * upscale_factor, w * upscale_factor
    hr_cropped = crop(hr_tensor, i_hr, j_hr, h_hr, w_hr)
    mask_cropped = crop(mask_tensor, i_hr, j_hr, h_hr, w_hr)

    return lr_cropped, hr_cropped, mask_cropped


def depth_to_colormap(depth_image: torch.Tensor, colormap='viridis') -> torch.Tensor:
    """
    Convert a single-channel depth image to a 3-channel RGB representation using a Matplotlib colormap.
    Expects depth_image in shape [H, W] or [1, H, W]. Returns [3, H, W].
    """
    depth_image_2d = depth_image.squeeze()  # shape => [H, W]
    colormap_func = plt.get_cmap(colormap)
    # Map [H, W] -> [H, W, 4], then slice :3 for RGB
    rgb_image = colormap_func(depth_image_2d.cpu().numpy())[..., :3]
    # Convert to Torch [3, H, W]
    rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()
    return rgb_tensor


class MagicBathyDataset(Dataset):
    """
    A dataset that loads satellite imagery, aerial imagery, (optionally) bathymetry,
    applies normalization, random cropping, etc., and can yield multiple patches per image.
    """

    def __init__(
        self,
        images: list,
        labels: list,
        bathymetry_images: list,
        norm_params: dict,
        bathymetry: bool = False,
        transform=None,
        target_transform=None,
        num_patches_per_image: int = 1,
        lowres_crop_size: int = 32,
        upscale_factor: int = 4
    ):
        """
        Args:
            images: List of paths to LR images (S2).
            labels: List of paths to HR images (aerial).
            bathymetry_images: List of paths to bathymetry images, if bathymetry=True.
            norm_params: Dictionary with normalization parameters (e.g. "s2_an", "aerial_an", etc.).
            bathymetry: Whether to use bathymetry as an extra input channel.
            transform: Optional transform for LR images.
            target_transform: Optional transform for HR images.
            num_patches_per_image: If >1, each image is repeated multiple times, each yielding a random crop.
            lowres_crop_size: Crop size in LR space.
            upscale_factor: Factor between LR and HR resolution.
        """
        self.images = images
        self.labels = labels
        self.bathymetry_images = bathymetry_images
        self.crop = False

        self.norm_params = norm_params
        self.bathymetry = bathymetry
        self.transform = transform
        self.target_transform = target_transform

        self.num_patches_per_image = num_patches_per_image
        self.lowres_crop_size = lowres_crop_size
        self.upscale_factor = upscale_factor

    def __len__(self):
        # Each image is repeated for the specified number of patches
        return len(self.images) * self.num_patches_per_image

    def __getitem__(self, idx):
        """
        For a global index `idx`, figure out which base image this corresponds to,
        load that image, do a random crop (or other transforms), and return the result.
        """
        # Which real image index are we using?
        image_idx = idx // self.num_patches_per_image

        # Load the actual file paths
        img_path = self.images[image_idx]
        label_path = self.labels[image_idx]
        bath_path = self.bathymetry_images[image_idx] if self.bathymetry_images else None

        # Read & convert to float32
        img = tifffile.imread(img_path).astype(np.float32)
        label = tifffile.imread(label_path).astype(np.float32)
        bath = tifffile.imread(bath_path).astype(np.float32) if bath_path else None

        # -------------------------------------------------
        # 1) Normalization logic
        # -------------------------------------------------
        if "agia_napa" in img_path:
            # Example usage of your stored norm_params
            # e.g. norm_params['s2_an'] is [min_val, max_val] for S2
            s2_min, s2_max = self.norm_params["s2_an"]
            aer_min, aer_max = self.norm_params["aerial_an"]
            depth_min_val = -30.443  # you had a hard-coded number
        elif "puck_lagoon" in img_path:
            s2_min, s2_max = self.norm_params["s2_pl"]
            aer_min, aer_max = self.norm_params["aerial_pl"]
            depth_min_val = -11.0
        else:
            # fallback or raise an error
            raise ValueError(f"Unknown location in path: {img_path}")

        # Normalize S2
        img = (img - s2_min) / (s2_max - s2_min)
        # For label (aerial)
        label = (label - aer_min) / (aer_max - aer_min)
        # For bath
        if bath is not None:
            # scale from [some negative ... 0] to [0..1], e.g. dividing by negative depth
            bath /= depth_min_val  # e.g. dividing by -30.443

        # -------------------------------------------------
        # 2) Channel Reordering or Clamping
        # -------------------------------------------------
        # If "agia_napa" => maybe reorder channels. Example:
        # S2 might be BGR => pick [2,1,0] => [R,G,B].
        # label might be BGR => etc. Adjust as needed.
        # Below is from your original code:
        if "agia_napa" in img_path:
            # for demonstration, reorder S2 from BGR to [2,1,0]
            img = img[..., [2, 1, 0]]

        # clamp to [0, 1]
        img = np.clip(img, 0, 1)
        label = np.clip(label, 0, 1)

        # -------------------------------------------------
        # 3) Convert to Torch
        # -------------------------------------------------
        # Reorder shape from [H, W, C] -> [C, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(label).permute(2, 0, 1).float()

        bath_tensor = None
        if bath is not None:
            bath = np.clip(bath, 0, 1)
            bath_tensor = torch.from_numpy(bath).unsqueeze(0).float()

        # Optionally resize to some standard shape if you want (like 64×64 for LR, 256×256 for HR)
        # e.g.:
        lr_size = (128, 128)
        hr_size = (128, 128)

        # S2 LR
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0), size=lr_size, mode="bicubic", align_corners=False
        ).squeeze(0)

        # Aerial HR

        label_tensor = F.interpolate(
            label_tensor.unsqueeze(0), size=hr_size, mode="bicubic", align_corners=False
        ).squeeze(0)

        # Bath LR
        if bath_tensor is not None:
            bath_tensor = F.interpolate(
                bath_tensor.unsqueeze(0), size=lr_size, mode="bicubic", align_corners=False
            ).squeeze(0)

        # -------------------------------------------------
        # 4) Combine channels for the "source"
        # -------------------------------------------------
        # source = [S2 LR (3ch), Bath LR (1ch)] => 4 channels total
        # Or if your S2 has more than 3 channels, adjust accordingly.
        if bath_tensor is not None:
            source_tensor = torch.cat([img_tensor, bath_tensor], dim=0)  # shape => [4, H, W]
        else:
            source_tensor = img_tensor  # shape => [3, H, W]

        # Make mask for label
        # E.g. your logic was something like (y != 0).all(dim=0)
        mask_label = (label_tensor != 0).all(dim=0, keepdim=True).float()
        mask_label = mask_label.repeat(label_tensor.shape[0], 1, 1)

        # -------------------------------------------------
        # 5) Random crop for data augmentation
        # -------------------------------------------------
        # If you want a random patch from LR  => e.g. 32×32
        # and from HR => 32*4 => 128×128
        if self.crop == True:
            source_tensor, label_tensor, mask_label = apply_random_crop_lr_hr(
                lr_tensor=source_tensor,
                hr_tensor=label_tensor,
                mask_tensor=mask_label,
                lowres_crop_size=self.lowres_crop_size,
                upscale_factor=self.upscale_factor
            )

        # Apply optional transforms
        if self.transform:
            lr_cropped = self.transform(source_tensor)
        if self.target_transform:
            hr_cropped = self.target_transform(label_tensor)

        return {
            "img_path": img_path,
            "source": source_tensor,   # shape => [C, H, W], either 3 or 4 channels
            "y": label_tensor,        # shape => [C, H*scale, W*scale]
            "mask_label": mask_label
        }


class MagicBathyNetDataLoader:
    """
    Higher-level class that sets up train, val, and test DataLoaders for the MagicBathyDataset.
    It handles things like splitting IDs, test sets, etc.
    """

    def __init__(
        self,
        root_dir: str,
        locations: list,
        norm_params: dict,
        bathymetry: bool = False,
        batch_size: int = 16,
        num_workers: int = 0,
        test_size: float = 0.15,
        val_size: float = 0.15,
        num_patches_per_image: int = 1
    ):
        """
        Args:
            root_dir: Root directory of your dataset (contains subfolders for each location).
            locations: E.g. ["agia_napa", "puck_lagoon"].
            norm_params: Dict with normalization info, e.g. {"s2_an": [0,10000], "aerial_an": [0,65535]} etc.
            bathymetry: Whether to include bathymetry data (and paths).
            batch_size: DataLoader batch size.
            num_workers: Dataloader number of workers.
            test_size: Fraction of dataset used for test. (Currently you do manual ID splitting.)
            val_size: Fraction of dataset used for val. (Again, might be partly manual.)
            num_patches_per_image: How many times to sample each image in the dataset.
        """
        self.root_dir = root_dir
        self.locations = locations
        self.norm_params = norm_params
        self.bathymetry = bathymetry

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size

        self.num_patches_per_image = num_patches_per_image

        # Build the three Dataloaders
        self.datasets = self._prepare_datasets()

        self.dataloaders = {
            phase: DataLoader(
                self.datasets[phase],
                batch_size=self.batch_size,
                shuffle=True if phase == 'train' else False,
                num_workers=self.num_workers,
                drop_last=False
            )
            for phase in ['train', 'val', 'test']
        }

    def _prepare_datasets(self):
        """
        This method collects all the paths for S2, aerial, (optionally) bathymetry.
        Splits them into train/val/test sets by ID or location.

        Returns:
            dict with "train", "val", "test" => MagicBathyDataset
        """

        # 1) Gather all file paths
        #    Example directory structure:
        #    root_dir/location/img/s2/*.tif
        #    root_dir/location/img/aerial/*.tif
        #    root_dir/location/depth/aerial/*.tif
        s2_paths = []
        aerial_paths = []
        bath_paths = []

        for loc in self.locations:
            s2_dir = os.path.join(self.root_dir, loc, "img", "s2_explicit")
            aer_dir = os.path.join(self.root_dir, loc, "img", "aerial_explicit")
            depth_dir = os.path.join(self.root_dir, loc, "depth", "aerial")

            s2_paths_loc = [
                os.path.join(s2_dir, f)
                for f in os.listdir(s2_dir)
                if f.endswith(".tif")
            ]
            aer_paths_loc = [
                os.path.join(aer_dir, f)
                for f in os.listdir(aer_dir)
                if f.endswith(".tif")
            ]
            # Only gather bath if bathymetry=True
            if self.bathymetry:
                bath_paths_loc = [
                    os.path.join(depth_dir, f)
                    for f in os.listdir(depth_dir)
                    if f.endswith(".tif")
                ]
            else:
                bath_paths_loc = [None] * len(s2_paths_loc)  # placeholder

            s2_paths.extend(sorted(s2_paths_loc))
            aerial_paths.extend(sorted(aer_paths_loc))
            bath_paths.extend(sorted(bath_paths_loc))

        # 2) You already do a bunch of custom ID-based splitting.
        #    For example, you might have sets of test IDs for "agia_napa" or "puck_lagoon."
        #    Let's demonstrate the logic with a dictionary approach:
        #    We'll keep it simpler here. If you want manual ID checks, do that logic here.
        #    For now, let's just do a standard train/val/test split on these path lists
        #    (assuming they're in matching order).
        #    If they're not guaranteed to be the same length or match 1-1, you'll need a matching approach.

        # Quick check
        assert len(s2_paths) == len(aerial_paths) == len(bath_paths), (
            "Number of S2, Aerial, and Bath files must match. "
            f"Found: S2={len(s2_paths)}, Aerial={len(aerial_paths)}, Bath={len(bath_paths)}"
        )

        # Make a list of all triplets
        all_triplets = list(zip(s2_paths, aerial_paths, bath_paths))

        # Shuffle if needed (not strictly necessary)
        # from random import shuffle
        # shuffle(all_triplets)

        # test_size fraction
        n_total = len(all_triplets)
        n_test = int(n_total * self.test_size)
        n_val = int(n_total * self.val_size)
        # The rest is train

        # Let's do a simple slice-based split
        test_triplets = all_triplets[:n_test]
        val_triplets = all_triplets[n_test:n_test + n_val]
        train_triplets = all_triplets[n_test + n_val:]

        # 3) Construct each dataset
        train_ds = MagicBathyDataset(
            images=[t[0] for t in train_triplets],
            labels=[t[1] for t in train_triplets],
            bathymetry_images=[t[2] for t in train_triplets],
            bathymetry=self.bathymetry,
            norm_params=self.norm_params,
            num_patches_per_image=self.num_patches_per_image,
            lowres_crop_size=32,
            upscale_factor=4
        )
        val_ds = MagicBathyDataset(
            images=[t[0] for t in val_triplets],
            labels=[t[1] for t in val_triplets],
            bathymetry_images=[t[2] for t in val_triplets],
            bathymetry=self.bathymetry,
            norm_params=self.norm_params,
            num_patches_per_image=self.num_patches_per_image,
            lowres_crop_size=32,
            upscale_factor=4
        )
        test_ds = MagicBathyDataset(
            images=[t[0] for t in test_triplets],
            labels=[t[1] for t in test_triplets],
            bathymetry_images=[t[2] for t in test_triplets],
            bathymetry=self.bathymetry,
            norm_params=self.norm_params,
            num_patches_per_image=self.num_patches_per_image,
            lowres_crop_size=32,
            upscale_factor=4
        )

        return {"train": train_ds, "val": val_ds, "test": test_ds}