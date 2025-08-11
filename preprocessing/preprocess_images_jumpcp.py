"""Preprocessing cell painting images."""
import argparse
import glob
import itertools
import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.helper import parallelize

# from training.datasets import CellPainting


# def numpy_to_img(arr, outfile, outdir="."):
#     img = Image.fromarray(arr)
#     img.save(outfile)
#     return


def illumination_threshold(arr, perc=0.0028):
    """Return threshold value to not display a percentage of highest pixels"""

    perc = perc / 100

    h = arr.shape[0]
    w = arr.shape[1]

    # find n pixels to delete
    total_pixels = h * w
    n_pixels = total_pixels * perc
    n_pixels = int(np.around(n_pixels))

    # find indexes of highest pixels
    flat_inds = np.argpartition(arr, -n_pixels, axis=None)[-n_pixels:]
    inds = np.array(np.unravel_index(flat_inds, arr.shape)).T

    max_values = [arr[i, j] for i, j in inds]

    threshold = min(max_values)

    return threshold


def sixteen_to_eight_bit(arr, display_max, display_min=0):
    """Convert unit16 to unit8"""
    threshold_image = (arr.astype(float) - display_min) * (arr > display_min)

    scaled_image = threshold_image * (256.0 / (display_max - display_min))
    scaled_image[scaled_image > 255] = 255

    scaled_image = scaled_image.astype(np.uint8)

    return scaled_image


def group_samples(indir):
    """Group images in different sites to a single sample and organize by channels"""
    dirlist = glob.glob(os.path.join(indir, "*"))

    basenames = [os.path.basename(d) for d in dirlist]

    # Group images by their sample name (e.g., r09c01f05p01)
    # Group by base name excluding channel
    plate_groups = [
        list(g) for _, g in itertools.groupby(sorted(basenames), lambda x: x[0:6])
    ]
    fullpath_groups = []
    basenames_groups = []

    # order = [1, 2, 4, 0, 3] -> [w4, w1, w2, w5, w3]
    # https://academic.oup.com/gigascience/article/6/12/giw014/2865213#120204609

    # Channel Mapping:
    # ch02 - Alexa 568 -> w4 (F-actin, Golgi, plasma membrane)
    # ch05 - Hoechst 33342 -> w1 (Nucleus/DNA)
    # ch04 - Alexa 488 -> w2 (Endoplasmic Reticulum)
    # ch01 - Alexa 647 -> w5 (Mitochondria)
    # ch03 - Alexa 488 long -> w3 (Nucleoli/RNA)

    # channel_mapping = {
    #     "ch1": 3,
    #     "ch2": 0,
    #     "ch3": 4,
    #     "ch4": 2,
    #     "ch5": 1
    # }
    # order = [3, 0, 4,2,1]

    order = [3, 2, 0, 4, 1]
    sample_list = []

    for g in plate_groups:
        fullpath_group = []
        basenames_group = []
        for f in g:
            if f.endswith(".tiff"):
                fullpath_group.append(os.path.join(indir, f))
                basenames_group.append(f)
        if len(fullpath_group) != 0:
            fullpath_groups.append(fullpath_group)
            basenames_groups.append(basenames_group)

    for i, plate in enumerate(fullpath_groups):
        plate_files = []

        for f in plate:
            if f.endswith(".tiff"):
                plate_files.append(f)

        # Group by site fXX, where fXX ranges from f01 to f16
        sample_groups = [
            list(g)
            for _, g in itertools.groupby(
                sorted(
                    plate_files, key=lambda x: x.split("/")[-1][7:10]
                ),  # Extract the site (e.g., fXX from r09c01f05p01)
                lambda x: x.split("/")[-1][7:10],  # Group by the site (fXX)
            )
        ]

        for g in sample_groups:
            # Apply the correct channel order
            ordered_group = [x for _, x in sorted(zip(order, g))]
            sample_list.append(ordered_group)
    return sample_list


def process_sample(imglst, metadata_path, outdir="."):
    """Aggregate well level sample"""

    plate_info = pd.read_csv(metadata_path)

    sample = np.zeros((1080, 1080, 5), dtype=np.uint8)
    refimg = imglst[0]

    plate_full = refimg.split("/")[-1][:12]
    plate = plate_full[:6]
    sampleid = plate_full[7:9]

    matching_rows = plate_info[plate_info["FileName_OrigRNA"].str[:12] == plate_full]

    if matching_rows.empty:
        print(f"No matching well found for plate {plate_full}")
        return

    well = matching_rows["Metadata_Well"].values[0]

    for i, imgfile in enumerate(imglst):
        arr = np.array(Image.open(imgfile))

        threshold = illumination_threshold(arr)
        scaled_arr = sixteen_to_eight_bit(arr, threshold)
        sample[:, :, i] = scaled_arr

    outfile = f"{plate}-{well}-{sampleid}.npz"
    outpath = os.path.join(outdir, outfile)
    np.savez(outpath, sample=sample)

    return


def get_mean_std(loader, outfile):
    """Compute statistiscs of images."""
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for batch in tqdm(loader):
        images = batch
        images = images["input"]
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    with open(outfile, "w") as f:
        f.write(f"Mean:{mean}\n")
        f.write(f"Std:{std}")

    return mean, std


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")

    parser.add_argument("--batch_name", type=str, help="batch name of images")
    return parser.parse_args()


def main(args):
    """Convert tiff to npz"""
    n_cpus = 16
    base_dir = "/data/nobackup/mingyulu/datasets/2020_11_04_CPJUMP1/images/"
    file_names = [f for f in os.listdir(base_dir)]
    output_base_dir = "/data/nobackup/mingyulu/datasets/cell_painting_full/CPJUMP1"

    for file_name in file_names:

        batch_num_match = re.match(r"(BR\d{8,})", file_name)

        if not batch_num_match:
            print(f"Skipping {file_name} as it doesn't contain a valid batch number.")
            continue

        batch_num = batch_num_match.group()
        # batch_name = file_name.split("-Measurement1")[
        #     0
        # ]   Extract full batch name up to '-Measurement1'

        indir = os.path.join(base_dir, f"{file_name}/Images")
        metadata_path = (
            "/homes/gws/mingyulu/contrastive-learning-with-treatment/image_label/"
            f"{batch_num}/load_data.csv"
        )
        outdir = os.path.join(output_base_dir, batch_num)

        # Create output directory if it doesn't exist
        os.makedirs(outdir, exist_ok=True)

        # Group samples and process them in parallel
        sample_groups = group_samples(indir)
        _ = parallelize(
            process_sample,
            sample_groups,
            n_cpus,
            metadata_path=metadata_path,
            outdir=outdir,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
