"""Functions to transform cell painting image to clip embeddings"""
import argparse
import os
import re

import h5py
import numpy as np
import torch
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
    parser.add_argument("--n_crop", type=int, help="number of crops", default=5)
    parser.add_argument("--output_file", type=str, help="output file name")

    return parser.parse_args()


class CellPaintingDataset(Dataset):
    """Customized dataset for loading cell painting images"""

    def __init__(
        self, data_directory, dataset, file_ids, preprocessor=None, n_crops=5, crop_size=336
    ):
        self.data_directory = data_directory
        self.dataset = dataset
        self.file_ids = file_ids
        self.n_crops = n_crops
        self.crop_size = crop_size
        self.preprocessor = preprocessor

    def __len__(self):
        """Return length of the dataset"""
        return len(self.file_ids)

    def __getitem__(self, idx):
        """Return item from the dataloader"""
        name = self.file_ids[idx]
        imgs = []

        filepath = os.path.join(self.data_directory, name)
        image = self.load_view(filepath)
        if image is not None:
            img_crops = self.transform(
                image, n_crops=self.n_crops, crop_size=self.crop_size
            )
            imgs.extend(img_crops)

        name = name.split(".")[0]

        # parts = name.split('_')
        # compound, plate, well = parts
        # plate_num = ''.join(filter(str.isdigit, plate))
        # name = f"{compound}_{plate_num}_{well}"

        return (name, imgs)

    def load_view(self, filepath):
        """Load cell painting images"""
        npz = np.load(filepath, allow_pickle=True)

        image = npz["images"]
        return image

    def rename_file(self, filename):
        return re.sub(r"_(\d+)_", r"_Plate\1_", filename)

    def transform(self, X, n_crops=5, crop_size=336):
        """Transform cell painting images"""
        transformed_images = []

        if self.dataset == "rxrx3-core":
            channel_order = [1, 3, 4, 0, 2]
            X = X[channel_order, :, :]
            X = np.transpose(X, (1, 2, 0))

        random_crop = RandomCrop(crop_size)

        H, W, num_channels = X.shape

        # For each crop (n_crops times)
        for _ in range(n_crops):
            cropped_img = random_crop(
                torch.tensor(X).permute(2, 0, 1)
            )  # Shape: (num_channels, H, W)

            cropped_img = cropped_img.permute(
                1, 2, 0
            ).numpy()  # Shape: (H, W, num_channels)
            channel_crops = []
            for i in range(num_channels):
                channel_img = cropped_img[:, :, i]

                # Convert grayscale channel to RGB
                channel_rgb = np.repeat(
                    channel_img[:, :, np.newaxis], 3, axis=2
                )  # Shape: (H, W, 3)

                channel_rgb = torch.tensor(channel_rgb, dtype=torch.float32).permute(
                    2, 0, 1
                )  # Shape: (3, H, W)
                if self.preprocessor:
                    channel_crops.append(
                        self.preprocessor(channel_rgb, return_tensors="pt").pixel_values[0]
                    )
                else:
                    channel_crops.append(channel_rgb)

            transformed_images.append(torch.stack(channel_crops))

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

        with torch.no_grad():
            crops_tensor = []

            for crop_imgs in image_slides:
                channel_embeddings = [channel.unsqueeze(0) for channel in crop_imgs]
                crops_tensor.append(torch.cat(channel_embeddings, dim=0))

            crops_tensor = torch.stack(crops_tensor)
            flattened_tensor = crops_tensor.view(
                -1, *crops_tensor.shape[2:]
            )  # Shape: [num_crops * 5, H, W]

            if model_card == "openphenom":
                embeddings = pretrained_clip.predict(flattened_tensor)
            else:
                embeddings = pretrained_clip(
                    flattened_tensor
                ).pooler_output  # Shape: [num_crops * 5, clip_emb_dim]

            embeddings = embeddings.view(len(image_slides), 5, -1)

            final_embeddings = torch.mean(embeddings, dim=0).cpu().numpy()

            save(output_file, final_embeddings, name)


def save(output_file, final_embeddings, name):
    """Save the embeddings in the same HDF5 file, appending to the dataset"""
    with h5py.File(output_file, "a") as hf:

        if "embeddings" not in hf:
            dataset = hf.create_dataset(
                "embeddings",
                data=final_embeddings[None, :, :],
                maxshape=(
                    None,
                    5,
                    final_embeddings.shape[1],
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
            dataset.resize((dataset.shape[0] + 1, 5, final_embeddings.shape[1]))
            dataset[-1, ::] = final_embeddings  # Append the new row of embeddings

            # Resize the name dataset and append the new name (ID)
            name_dataset.resize((name_dataset.shape[0] + 1,))
            name_dataset[-1] = name  # Append the name/ID for the row

    del final_embeddings  # Free memory after saving


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

    output_file = f"/gscratch/aims/datasets/cellpainting/rxrx3-core/img/{args.output_file}"

    org_files = os.listdir(args.input_dir)

    file_ids = list(set([file for file in org_files if file.endswith(".npz")]))

    if os.path.isfile(output_file):
        with h5py.File(output_file, "r") as hf:
            output_ids = [i.decode("utf-8") + ".npz" for i in hf["names"][:]]
        file_ids = [id for id in file_ids if id not in output_ids]

    dataset = CellPaintingDataset(
        args.input_dir,
        args.dataset,
        file_ids,
        processor,
        n_crops=args.n_crop,
        crop_size=crop_size,
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
