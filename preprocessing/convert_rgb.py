import csv
import os
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

DEFAULT_CHANNELS = (1, 2, 3, 4, 5)

# Hoechst
# ConA
# Phalloidin
# Syto14
# MitoTracker
# WGA

RGB_MAP = {
    1: {"rgb": np.array([42, 255, 31]), "range": [0, 107]},
    2: {"rgb": np.array([45, 255, 252]), "range": [0, 191]},
    3: {"rgb": np.array([250, 0, 253]), "range": [0, 89]},
    4: {"rgb": np.array([19, 0, 249]), "range": [0, 51]},
    5: {"rgb": np.array([255, 0, 25]), "range": [0, 64]},
    # 6: {
    #     'rgb': np.array([254, 255, 40]),
    #     'range': [0, 191]
    #
}


def convert_tensor_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image

    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : list of int
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.

    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """

    colored_channels = []
    h, w, _ = t.shape

    for i, channel in enumerate(channels):
        x = (t[:, :, i] / vmax) / (
            (rgb_map[channel]["range"][1] - rgb_map[channel]["range"][0]) / 255
        ) + rgb_map[channel]["range"][0] / 255
        x = np.where(x > 1.0, 1.0, x)
        x_rgb = np.array(np.outer(x, rgb_map[channel]["rgb"]).reshape(h, w, 3), dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)

    return im


def convert_all_to_rgb_for_hf(data_folder, target_folder, split_name="train"):
    # Make directory structure: target_folder/train/
    split_folder = os.path.join(target_folder, split_name)
    os.makedirs(split_folder, exist_ok=True)

    for file_name in tqdm(os.listdir(data_folder)):
        if file_name.endswith(".npz"):
            file_path = os.path.join(data_folder, file_name)
            tensor = np.load(file_path)
            tensor = tensor["images"]
            # if self.dataset == "rxrx3-core":
            # Reorder to match Bray 2017
            channel_order = [1, 3, 4, 0, 2]
            tensor = tensor[channel_order, :, :]
            tensor = np.transpose(tensor, (1, 2, 0))

            if tensor.ndim == 3:
                rgb_image = convert_tensor_to_rgb(tensor)

                # Save as .png in the split folder
                out_name = os.path.splitext(file_name)[0].split(".")[0] + ".png"
                out_path = os.path.join(split_folder, out_name)
                Image.fromarray(rgb_image.astype(np.uint8)).save(out_path)


# convert_all_to_rgb_for_hf(
#     "/gscratch/aims/datasets/cellpainting/rxrx3-core/raw",
#     "/gscratch/aims/datasets/cellpainting/rxrx3-core/raw_rgb"
#     )


def generate_metadata_csv(image_dir, output_csv_path):
    """
    Generates a CSV file with filename and custom label from images in the directory.

    Args:
        image_dir (str): Path to the folder containing images (e.g. train/)
        output_csv_path (str): Path to save the generated metadata CSV
    """
    rows = []

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            name_without_ext = os.path.splitext(filename)[0]  # e.g. gene-022_Plate6_N05
            parts = name_without_ext.split("_")
            if len(parts) >= 3 and "Plate" in parts[1]:
                gene = parts[0]  # gene-022
                plate_number = re.sub(r"\D", "", parts[1])  # extract "6" from Plate6
                well = parts[2]  # N05
                label = f"{gene}_{plate_number}_{well}"  # gene-022_6_N05

                rows.append([filename, label])

    # Write to CSV
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "label"])  # header
        writer.writerows(rows)

    print(f"Metadata CSV written to {output_csv_path} with {len(rows)} rows.")


generate_metadata_csv(
    "/gscratch/aims/datasets/cellpainting/rxrx3-core/raw_rgb",
    "/gscratch/aims/datasets/cellpainting/rxrx3-core/raw_rgb/test.csv",
)
