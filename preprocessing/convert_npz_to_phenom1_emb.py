"""Functions to transform cell painting image to clip embeddings"""
import argparse
import os
import random
import re

import h5py
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import RandomCrop

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPImageProcessor,
    CLIPVisionModel,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from src import constants
from src.open_phenom.hugginface_mae import MAEModel

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
    # }
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


class CellPaintingDataset(Dataset):
    """Customized dataset for loading cell painting images"""

    def __init__(
        self,
        data_directory,
        dataset,
        file_ids,
        preprocessor=None,
        n_crops=5,
        crop_size=336,
        convert_to_rgb=False,
    ):
        self.data_directory = data_directory
        self.dataset = dataset

        if dataset == "rxrx3-core":
            file_path_metadata = hf_hub_download(
                "recursionpharma/rxrx3-core",
                filename="metadata_rxrx3_core.csv",
                repo_type="dataset",
            )
            self.metadata = pd.read_csv(file_path_metadata)
            self.file_ids = file_ids
        else:
            self.file_ids = file_ids

        self.n_crops = n_crops
        self.crop_size = crop_size
        self.preprocessor = preprocessor
        self.convert_to_rgb = convert_to_rgb

    def __len__(self):
        """Return length of the dataset"""
        return len(self.file_ids)

    def __getitem__(self, idx):
        """Return item from the dataloader"""
        name = self.file_ids[idx]
        if self.dataset == "rxrx3-core":
            file_ids = self.metadata[self.metadata.treatment == name].well_id.tolist()
            files = [self.rename_file(f) + ".npz" for f in file_ids]
            if len(files) >= 70:
                files = random.sample(files, 70)
        else:
            files = [f for f in os.listdir(self.data_directory) if f.startswith(name)]

        imgs = []

        for f in files:
            filepath = os.path.join(self.data_directory, f)
            image = self.load_view(filepath)
            if image is not None:
                img_crops = self.transform(
                    image, n_crops=self.n_crops, crop_size=self.crop_size
                )
                imgs.extend(img_crops)

        return (name, imgs)

    def load_view(self, filepath):
        """Load cell painting images"""
        npz = np.load(filepath, allow_pickle=True)

        if self.dataset == "rxrx3-core":
            image = npz["images"]
            return image
        else:
            if "sample" in npz:
                image = npz["sample"].astype(np.float32)
                return image

        return None

    def rename_file(self, filename):
        return re.sub(r"_(\d+)_", r"_Plate\1_", filename)

    def transform(self, X, n_crops=5, crop_size=336):
        """Transform cell painting images"""
        transformed_images = []

        if self.dataset == "rxrx3-core":
            # Reorder to match Bray 2017
            channel_order = [1, 3, 4, 0, 2]
            X = X[channel_order, :, :]
            X = np.transpose(X, (1, 2, 0))

        if self.convert_to_rgb:
            X = convert_tensor_to_rgb(X)

        random_crop = RandomCrop(crop_size)

        # For each crop (n_crops times)
        for _ in range(n_crops):
            cropped_img = random_crop(torch.tensor(X).permute(2, 0, 1))  # Shape: (C, H, W)

            cropped_img = cropped_img.permute(1, 2, 0).numpy()  # Shape: (H, W, C)

            if self.preprocessor:
                cropped_img = self.preprocessor(
                    cropped_img, return_tensors="pt"
                ).pixel_values[0]

            # Ensure correct shape (C, H, W)
            if cropped_img.shape[0] not in {3, 5}:
                cropped_img = torch.tensor(cropped_img).permute(2, 0, 1)

            transformed_images.append(cropped_img)

        transformed_images = torch.stack(transformed_images)
        return transformed_images


def my_collate_fn(batch):
    """Customized collate function, return the list as it is"""
    return batch


def generate_embeddings(pretrained_clip, batch, output_file, model_card="clip"):
    """Process a batch of images, generate embeddings, and save them"""

    batch_size = len(batch)

    for i in range(batch_size):
        name, image_slides = batch[i][0], batch[i][1]

        final_embeddings = []

        with torch.no_grad():

            for image in image_slides:
                image = image[None, :, :, :]

                if model_card == "openphenom":
                    embeddings = pretrained_clip.predict(image)
                else:
                    embeddings = pretrained_clip(image).pooler_output

                final_embeddings.append(embeddings)

            final_embeddings = torch.stack(final_embeddings)
            final_embeddings = torch.mean(embeddings, dim=0).cpu().numpy()

            save(output_file, final_embeddings, name)


def save(output_file, final_embeddings, name):
    """Save the embeddings in the same HDF5 file, appending to the dataset"""
    with h5py.File(output_file, "a") as hf:

        if "embeddings" not in hf:
            dataset = hf.create_dataset(
                "embeddings",
                data=final_embeddings[None, :],
                maxshape=(
                    None,
                    final_embeddings.shape[0],
                ),  # Unlimited rows, fixed columns
                chunks=True,  # Enables more efficient resizing
            )

            # Create a dataset to store the names/IDs associated with each row
            name_dataset = hf.create_dataset(
                "names",
                (1,),  # Start with one entry, the current name
                maxshape=(None,),  # Unlimited entries
                dtype=h5py.string_dtype(encoding="utf-8"),  # String type for storing IDs
            )
            name_dataset[0] = name  # Store the first name/ID

        else:
            # If the dataset exists, resize and append the new row
            dataset = hf["embeddings"]
            name_dataset = hf["names"]

            # Resize the embeddings dataset
            dataset.resize((dataset.shape[0] + 1, final_embeddings.shape[0]))
            dataset[-1, ::] = final_embeddings  # Append the new row of embeddings

            # Resize the name dataset and append the new name (ID)
            name_dataset.resize((name_dataset.shape[0] + 1,))
            name_dataset[-1] = name  # Append the name/ID for the row

    del final_embeddings  # Free memory after saving


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")
    parser.add_argument("--input_dir", type=str, help="input file directory")

    parser.add_argument(
        "--model_card", type=str, help="pretrained model card for embeddings"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="path to directory of datasets.",
        default=constants.DATASET_DIR,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to be processed",
        default="jumpcp",
    )
    parser.add_argument(
        "--convert_to_rgb",
        action="store_true",
        help="Whether to convert data to RGB format.",
        default=False,
    )
    parser.add_argument("--n_crop", type=int, help="number of crops", default=5)
    parser.add_argument("--output_file", type=str, help="output file name")

    return parser.parse_args()


def main(args):
    """Convert images into embeddings"""
    accelerator = Accelerator()

    # Load pre-trained embedding model, e.g. DINOv2 or CLIP.

    if "siglip" in args.model_card:
        pretrained_model = SiglipVisionModel.from_pretrained(args.model_card)
        processor = SiglipImageProcessor.from_pretrained(args.model_card)
        crop_size = processor.size["height"]
    elif "dino" in args.model_card:
        pretrained_model = AutoModel.from_pretrained(args.model_card)
        processor = AutoImageProcessor.from_pretrained(args.model_card)

        if args.model_card == "facebook/dino-vitb8":
            crop_size = processor.size["height"]
        else:
            crop_size = processor.crop_size["height"]

    elif "clip" in args.model_card:
        pretrained_model = CLIPVisionModel.from_pretrained(args.model_card)
        processor = CLIPImageProcessor.from_pretrained(args.model_card)
        crop_size = processor.crop_size["height"]
    elif args.model_card == "openphenom":
        pretrained_model = MAEModel.from_pretrained("recursionpharma/OpenPhenom")
        crop_size = 256
        processor = None

    pretrained_model.eval()
    if processor:
        processor.do_resize = False

    output_file = (
        f"/gscratch/aims/datasets/cellpainting/{args.dataset}/img/{args.output_file}"
    )

    org_files = os.listdir(args.input_dir)

    if args.dataset == "rxrx3-core":
        file_path_metadata = hf_hub_download(
            "recursionpharma/rxrx3-core",
            filename="metadata_rxrx3_core.csv",
            repo_type="dataset",
        )
        metadata = pd.read_csv(file_path_metadata)
        file_ids = metadata.treatment.unique().tolist()
    else:
        file_ids = list(
            set(
                [
                    "-".join(file.split("-")[:-1])
                    for file in org_files
                    if file.endswith(".npz")
                ]
            )
        )

    if os.path.isfile(output_file):
        with h5py.File(output_file, "r") as hf:
            output_ids = [i.decode("utf-8") for i in hf["names"][:]]
        file_ids = [id for id in file_ids if id not in output_ids]

    dataset = CellPaintingDataset(
        args.input_dir,
        args.dataset,
        file_ids,
        processor,
        n_crops=args.n_crop,
        crop_size=crop_size,
        convert_to_rgb=args.convert_to_rgb,
    )
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=my_collate_fn
    )

    pretrained_model, dataloader = accelerator.prepare(pretrained_model, dataloader)
    # Iterate over batches of data and generate embeddings
    progress_bar = tqdm(
        range(len(dataloader)),
        initial=0,
        desc="batch",
        disable=not accelerator.is_main_process,
    )

    for batch in dataloader:
        generate_embeddings(pretrained_model, batch, output_file, args.model_card)
        if accelerator.is_main_process:
            progress_bar.update(1)

    print(f"Parallelized {args.model_card} embeddings generation done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
