"""Function that calcualte the statistics of a given image (directory)"""
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class NPZLoader:
    """Loader for .npz files"""

    def __init__(self, path, transform=None):
        """Initialization function"""
        self.path = path
        self.files = list(Path(path).glob("*.npz"))
        self.transform = transform

    def __len__(self):
        """Return the length of the loader"""
        return len(self.files)

    def __getitem__(self, item):
        """Return a given image"""
        numpy_array = np.load(str(self.files[item]))["images"]  # Shape: (C, H, W)
        return numpy_array


def get_mean_std(loader, outfile):
    """Compute statistics of images for normalization."""
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, total_pixels = 0, 0, 0

    for batch in tqdm(loader):
        # Convert to torch tensor
        images = torch.tensor(batch, dtype=torch.float32)  # Shape: (C, H, W)

        # Accumulate sum and squared sum for each channel
        channels_sum += images.sum(dim=[1, 2])  # Sum over height and width
        channels_sqrd_sum += (images ** 2).sum(dim=[1, 2])  # Squared sum
        total_pixels += images.shape[1] * images.shape[2]  # Total pixels per channel

    # Compute mean and standard deviation
    mean = channels_sum / total_pixels
    std = ((channels_sqrd_sum / total_pixels) - mean ** 2) ** 0.5

    # Save results
    with open(outfile, "w") as f:
        f.write(f"Mean: {mean.tolist()}\n")
        f.write(f"Std: {std.tolist()}\n")

    return mean, std


path = "/gscratch/aims/datasets/cellpainting/rxrx3-core/raw"

# Create a custom dataloader
dataloader = NPZLoader(path)

# Compute mean and standard deviation
get_mean_std(dataloader, "/gscratch/aims/mingyulu/cell_painting/stats.jsonl")
