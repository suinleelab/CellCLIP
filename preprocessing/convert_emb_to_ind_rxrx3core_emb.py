"""Evaluation pipeline for silbiling perturbation in 2024 Chandrasekaran et al."""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop
from tqdm import tqdm
from transformers import (  # CLIPImageProcessor,; CLIPVisionModel,
    AutoImageProcessor,
    AutoModel,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from src import constants
from src.helpler import load
from src.open_phenom.hugginface_mae import MAEModel
from src.open_phenom.vit_encoder import build_imagenet_baselines
from src.transformations.cloome import _transform

DEFAULT_CHANNELS = (1, 2, 3, 4, 5)

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
    """Customized dataset for loading embedding images"""

    def __init__(
        self,
        img_path,
        existed_ids,
        transform=None,
        crop_size=336,
        convert_to_rgb=False,
    ):

        self.img_path = img_path
        self.is_embeddings = False
        self.transform = transform
        self.crop_size = crop_size
        self.convert_to_rgb = convert_to_rgb

        if img_path.endswith(".parquet"):
            embeddings = pd.read_parquet(img_path)
            feature_columns = [c for c in embeddings.columns if c.startswith("feature_")]
            embeddings["embeddings"] = np.split(
                embeddings[feature_columns].values,
                len(embeddings),
                axis=0,
            )
            self.img_file = embeddings
            self.img_ids = embeddings["well_id"].tolist()
            self.is_embeddings = True

        elif img_path.endswith(".h5"):
            self.img_file = h5py.File(img_path, "r")
            self.img_ids = [sample.decode("utf-8") for sample in self.img_file["names"][:]]
            self.is_embeddings = True
        else:
            self.img_ids = [
                file.rsplit(".", 1)[0]
                for file in os.listdir(img_path)
                if file.endswith(".npz")
            ]
        self.img_ids = [idx for idx in self.img_ids if idx not in existed_ids]

    def __len__(self):
        """Return length of the dataset"""
        return len(self.img_ids)

    def __getitem__(self, idx):
        """Return item from the dataloader"""
        sample_id = self.img_ids[idx]

        if self.is_embeddings:
            X = self.img_file["embeddings"][idx]
        else:
            filepath = os.path.join(self.img_path, "{}.npz".format(sample_id))
            X = self.load_view(filepath=filepath)

            channel_order = [1, 3, 4, 0, 2]
            X = X[channel_order, :, :]
            X = np.transpose(X, (1, 2, 0))

            if self.convert_to_rgb:
                X = convert_tensor_to_rgb(X)

            if self.crop_size:
                random_crop = RandomCrop(self.crop_size)
                X = random_crop(torch.tensor(X).permute(2, 0, 1))  # Shape: (C, H, W)
                X = X.permute(1, 2, 0).numpy()  # Shape: (H, W, C)

            if self.transform:
                # X = self.transform(X)
                X = self.transform(X, return_tensors="pt").pixel_values[0]

        return (sample_id, X)

    def load_view(self, filepath):
        """Load all channels for one sample"""
        npz = np.load(filepath, allow_pickle=True)
        image = npz["images"].astype(np.float32)
        return image


def my_collate_fn(batch):
    """Customized collate function, return the list as it is"""
    return batch


def save(output_file, final_embeddings, well_ids):
    """Save the embeddings in the same HDF5 file, appending to the dataset"""
    batch_size, dim = final_embeddings.shape

    with h5py.File(output_file, "a") as hf:
        if "embeddings" not in hf:
            hf.create_dataset(
                "embeddings",
                data=final_embeddings[:, None, :],
                maxshape=(None, 1, dim),
                chunks=True,
            )
            hf.create_dataset(
                "well_id",
                data=np.array(well_ids, dtype=h5py.string_dtype(encoding="utf-8")),
                maxshape=(None,),
            )
        else:
            emb_ds = hf["embeddings"]
            id_ds = hf["well_id"]

            emb_ds.resize((emb_ds.shape[0] + batch_size, 1, dim))
            emb_ds[-batch_size:, 0, :] = final_embeddings

            id_ds.resize((id_ds.shape[0] + batch_size,))
            id_ds[-batch_size:] = well_ids

    del final_embeddings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="path to model check point",
        default=None,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cell_clip",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cell_clip",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=512,
        help=("Input dimension of pretrained embeddings"),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=("Opeput file name"),
    )
    parser.add_argument(
        "--convert_to_rgb",
        action="store_true",
        help="Whether to convert data to RGB format.",
        default=False,
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="",
        help=("precomputed embeddings dir"),
    )
    return parser.parse_args()


def main(args):
    """Perform inference on jumpcp datasets."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess_fn = None

    if "dino" in args.model_type:
        model = AutoModel.from_pretrained(args.model_type)
        preprocess_fn = AutoImageProcessor.from_pretrained(args.model_type)

        if args.model_type == "facebook/dino-vitb8":
            crop_size = preprocess_fn.size["height"]
        else:
            crop_size = preprocess_fn.crop_size["height"]
    elif "vit" in args.model_type:
        # ViT encoder pretraind with ImageNet
        model_dicts = build_imagenet_baselines()
        model = model_dicts[args.model_type]
        preprocess_fn = _transform(
            256,
            256,
            False,
            None,
            "crop",
        )
    elif "siglip" in args.model_type:
        model = SiglipVisionModel.from_pretrained(args.model_type)
        preprocess_fn = SiglipImageProcessor.from_pretrained(args.model_type)
        crop_size = preprocess_fn.size["height"]
    # elif "clip" in args.model_type:
    #     model = CLIPVisionModel.from_pretrained(args.model_type)
    #     preprocess_fn = CLIPImageProcessor.from_pretrained(args.model_type)
    #     crop_size = preprocess_fn.crop_size["height"]
    elif args.model_type == "openphenom":
        model = MAEModel.from_pretrained("recursionpharma/OpenPhenom")
        crop_size = 256
    else:
        model = load(
            args.ckpt_path, device, args.model_type, args.input_dim, args.loss_type
        )
        crop_size = None
        if args.model_type == "cloome":
            preprocess_fn = _transform(
                520,
                520,
                False,
                "img",
                "crop",
            )

    model.to(device)
    model.eval()

    if preprocess_fn:
        preprocess_fn.do_resize = False

    output_dir = os.path.join(
        constants.DATASET_DIR, "rxrx3-core/output_emb", args.model_type
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"{args.output_file}")

    existed_ids = []
    if os.path.isfile(output_file):
        with h5py.File(output_file, "r") as hf:
            existed_ids = [i.decode("utf-8") for i in hf["well_id"][:]]

    dataset = CellPaintingDataset(
        args.img_dir,
        existed_ids,
        preprocess_fn,
        crop_size,
        convert_to_rgb=args.convert_to_rgb,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,  # collate_fn=my_collate_fn
    )

    # Iterate over batches of data and generate embeddings
    progress_bar = tqdm(
        range(len(dataloader)),
        initial=0,
        desc="batch",
    )

    for batch in dataloader:

        well_id, img = batch[0], batch[1]
        with torch.no_grad():
            img = torch.tensor(img).to(device)

            if "dino" in args.model_type:
                output = model(img).pooler_output.detach().cpu().numpy()
            elif "vit" in args.model_type:
                output = model(img).detach().cpu().numpy()
            else:
                output = model.encode_image(img).detach().cpu().numpy()
            save(output_file, output, well_id)

        progress_bar.update(1)

    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
