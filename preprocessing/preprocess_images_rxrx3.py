"""Preprocessing RxRx3-core images."""
import os

import numpy as np
import torch
from tqdm import tqdm

from datasets import load_dataset


def parse_key(entry_key):
    """Helper function to parse the key and extract base key and subkey"""

    parts = entry_key.split("/")
    base_key = "/".join(parts[:2]) + "/" + parts[-1].split("_")[0]
    subkey = int(parts[-1].split("_s1_")[-1]) - 1  # Extract numeric suffix
    return base_key, subkey


def my_collate_fn(batch):
    """Customized collate function for DataLoader"""
    return batch


if __name__ == "__main__":
    """Save rxrx3-core to npz"""

    rxrx3_core = load_dataset("recursionpharma/rxrx3-core")
    sorted_data = rxrx3_core["train"].sort("__key__")

    output_dir = "/gscratch/aims/datasets/cellpainting/rxrx3-core/raw"
    os.makedirs(output_dir, exist_ok=True)

    dataloader = torch.utils.data.DataLoader(
        sorted_data, batch_size=6, shuffle=False, drop_last=False, collate_fn=my_collate_fn
    )

    grouped = {}

    for batch in tqdm(dataloader):
        img_list = []
        well_id = None  # To track the current well ID

        for image in batch:
            base_key, subkey = parse_key(image["__key__"])

            if base_key not in grouped:
                grouped[base_key] = [None] * 6

            # Check if current base_key matches the well_id in this batch
            if well_id is None:
                well_id = base_key
            elif well_id != base_key:
                raise ValueError("Mismatched well_id within batch.")

            grouped[base_key][subkey] = np.array(image["jp2"])

        # Save grouped images for this batch
        for base_key, images in grouped.items():

            if any(img is None for img in images):
                raise ValueError(f"Incomplete group for {base_key}. Missing images.")

            stacked_images = np.stack(images, axis=0)

            filename = base_key.replace("/", "_") + ".npz"
            file_path = os.path.join(output_dir, filename)
            np.savez(file_path, images=stacked_images)

        grouped.clear()

    print("Grouped images saved as individual .npz files with shape [6, H, W]")
