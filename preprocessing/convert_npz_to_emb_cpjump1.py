"""Functions to transform JUMPCP image to open pheom (MAE) embeddings"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop
from tqdm import tqdm

from configs.model_config import ModelConfig
from src import constants
from src.clip.model import Cloome
from src.datasets import CellPainting
from src.open_phenom.hugginface_mae import MAEModel
from src.open_phenom.mae import load_mae
from src.open_phenom.vit_encoder import ViTClassifier, build_imagenet_baselines
from src.transformations.cloome import _transform


class CellPaintingDataset(Dataset):
    """Customized dataset for loading cell painting images"""

    def __init__(
        self,
        data_directory,
        file_ids,
        plate,
        preprocessor,
        n_crops=5,
        crop_size=336,
        model_type="openphenom",
    ):
        self.data_directory = data_directory

        # if args.dataset == "rxrx3-core":
        #     file_path_metadata = hf_hub_download(
        #         "recursionpharma/rxrx3-core",
        #         filename="metadata_rxrx3_core.csv",
        #         repo_type="dataset",
        #     )
        #     rxrx3_core_metadata = pd.read_csv(file_path_metadata)

        self.file_ids = file_ids
        self.n_crops = n_crops
        self.crop_size = crop_size
        self.preprocessor = preprocessor
        self.model_type = model_type

        experiment_df = pd.read_csv(
            "/gscratch/aims/mingyulu/cell_painting/"
            "label_data/jumpcp/experiment-metadata.tsv",
            sep="\t",
        ).query("Batch=='2020_11_04_CPJUMP1'")
        perturbation = experiment_df[
            experiment_df.Assay_Plate_Barcode == plate
        ].Perturbation.values[0]

        if perturbation == "compound":
            self.meta_data = pd.read_csv(
                (
                    "/gscratch/aims/mingyulu/cell_painting/"
                    "label_data/jumpcp/plate_map/JUMP-Target-1_compound_platemap.txt"
                ),
                sep="\t",
            )
        elif perturbation == "crispr":
            self.meta_data = pd.read_csv(
                (
                    "/gscratch/aims/mingyulu/cell_painting/"
                    "label_data/jumpcp/plate_map/JUMP-Target-1_crispr_platemap.txt"
                ),
                sep="\t",
            )
        elif perturbation == "orf":
            self.meta_data = pd.read_csv(
                (
                    "/gscratch/aims/mingyulu/cell_painting/"
                    "label_data/jumpcp/plate_map/JUMP-Target-1_orf_platemap.txt"
                ),
                sep="\t",
            )

    def __len__(self):
        """Return length of the dataset"""
        return len(self.file_ids)

    def __getitem__(self, idx):
        """Return item from the dataloader"""
        fine_name = self.file_ids[idx]
        files = [f for f in os.listdir(self.data_directory) if f.startswith(fine_name)]
        well_position = fine_name.split("-")[-1]
        imgs = []

        for f in files:
            filepath = os.path.join(self.data_directory, f)
            image = self.load_view(filepath)
            if image is not None:
                img_crops = self.transform(
                    image, n_crops=self.n_crops, crop_size=self.crop_size
                )
                imgs.extend(img_crops)
        imgs = torch.stack(imgs)

        sample_id = self.meta_data.loc[
            self.meta_data.well_position == well_position, "broad_sample"
        ].values

        if len(sample_id) == 0 or pd.isna(sample_id[0]):
            sample_id = "NaN"
        else:
            sample_id = sample_id[0]

        return (well_position, sample_id, imgs)

    def load_view(self, filepath):
        """Load cell painting images"""
        npz = np.load(filepath, allow_pickle=True)
        if "sample" in npz:
            image = npz["sample"].astype(np.float32)
            return image
        return None

    def transform(self, X, n_crops=5, crop_size=256):
        """Transform cell painting images"""
        transformed_images = []
        random_crop = RandomCrop(crop_size)

        H, W, num_channels = X.shape

        # For each crop (n_crops times)
        for _ in range(n_crops):
            cropped_img = random_crop(
                torch.tensor(X).permute(2, 0, 1)
            )  # Shape: (num_channels, H, W)

            if self.model_type == "openphenom":
                # Permuted order when converting tiff to npz.
                # inverse_order = [1, 4, 3, 0, 2]
                inverse_order = [0, 1, 2, 3, 4]
                cropped_img = cropped_img[inverse_order, :, :]

            elif self.model_type == "cloome":
                cropped_img = cropped_img.detach().cpu().numpy()
                cropped_img = np.transpose(cropped_img, (1, 2, 0))

            if self.preprocessor:
                transformed_images.append(self.preprocessor(cropped_img))
            else:
                transformed_images.append(cropped_img)

        transformed_images = torch.stack(transformed_images)
        return transformed_images


def my_collate_fn(batch):
    """Customized collate function, return the list as it is"""
    return batch


def generate_embeddings(model, model_type, batch, output_file, device):
    """Process a batch of images, generate embeddings, and save them"""

    batch_size = len(batch)

    for i in range(batch_size):
        well_position, broad_sample, crop_imgs = batch[i][0], batch[i][1], batch[i][2]

        with torch.no_grad():
            crop_imgs = crop_imgs.to(device)
            if model_type == "mae":
                output = model.predict(crop_imgs)
            elif model_type == "cloome":
                output = model.encode_image(crop_imgs)
            elif model_type == "wsl":
                output = model.get_embeddings(crop_imgs)
            else:
                output = model(crop_imgs)

            output = torch.mean(output, dim=0, keepdim=True).cpu().numpy()
            save(output_file, output, well_position, broad_sample)


def save(output_file, final_embeddings, well_position, broad_sample):
    """Save the embeddings in the same HDF5 file, appending to the dataset"""
    with h5py.File(output_file, "a") as hf:

        if "embeddings" not in hf:
            dataset = hf.create_dataset(
                "embeddings",
                data=final_embeddings[None, :, :],
                maxshape=(
                    None,
                    1,
                    final_embeddings.shape[1],
                ),  # Unlimited rows, fixed columns
                chunks=True,  # Enables more efficient resizing
            )

            # Create a dataset to store the names/IDs associated with each row
            well_dataset = hf.create_dataset(
                "well_position",
                (1,),  # Start with one entry, the current name
                maxshape=(None,),  # Unlimited entries
                dtype=h5py.string_dtype(encoding="utf-8"),  # String type for storing IDs
            )
            well_dataset[0] = well_position  # Store the first name/ID

            broad_sample_dataset = hf.create_dataset(
                "broad_sample",
                (1,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            broad_sample_dataset[0] = broad_sample

        else:
            # If the dataset exists, resize and append the new row
            dataset = hf["embeddings"]
            well_dataset = hf["well_position"]
            broad_sample_dataset = hf["broad_sample"]

            # Resize the embeddings dataset
            dataset.resize((dataset.shape[0] + 1, 1, final_embeddings.shape[1]))
            dataset[-1, ::] = final_embeddings

            well_dataset.resize((well_dataset.shape[0] + 1,))
            well_dataset[-1] = well_position

            broad_sample_dataset.resize((broad_sample_dataset.shape[0] + 1,))
            broad_sample_dataset[-1] = broad_sample

    del final_embeddings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")
    parser.add_argument(
        "--model_type",
        type=str,
        help="model type to compute embeddings, e.g. cloom",
        default=None,
    )
    parser.add_argument(
        "--model_info",
        type=str,
        help="Information specific to model type, e.g. vit_large_patch16_384",
        default=None,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="path to model check point",
        default=None,
    )
    parser.add_argument(
        "--plate", type=str, help="Plate number of jumpcp, e.g. BR00117016", default=None
    )
    parser.add_argument(
        "--data_dir", type=str, help="path to data folder", default=constants.DATASET_DIR
    )
    parser.add_argument(
        "--n_crop",
        default=1,
        type=int,
        help=("number of crops"),
    )
    parser.add_argument(
        "--crop_size",
        default=256,
        type=int,
        help=("resolution of cropped images."),
    )

    return parser.parse_args()


def main(args):
    """Convert npz files to embeddings"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = None

    if args.model_type == "mae":
        if args.ckpt_path:
            model = load_mae()
            state_dict = torch.load(args.ckpt_path)
            model.load_state_dict(state_dict["model"])
        else:
            model = MAEModel.from_pretrained("recursionpharma/OpenPhenom")
    elif args.model_type == "wsl":
        image_path = "/gscratch/aims/datasets/cellpainting/jumpcp/jumpcp_full"
        molpath = "/gscratch/aims/datasets/cellpainting/jumpcp/mol/jumpcp_cell_caption.csv"
        dataset = CellPainting(
            "/gscratch/aims/datasets/cellpainting/jumpcp/mol/jumpcp_training_label2.csv",
            "text",
            transforms=None,
            image_directory_path=image_path,
            molecule_path=molpath,
            dataset="jumpcp",
        )

        label_map = dataset.label_map

        model = ViTClassifier(args.model_info, len(label_map))
        state_dict = torch.load(args.ckpt_path)
        model.load_state_dict(state_dict["model"])

    elif args.model_type == "vit":
        # ViT encoder pretraind with ImageNet
        model_dicts = build_imagenet_baselines()
        model = model_dicts[args.model_type]
    elif args.model_type == "cloome":
        model = Cloome(**ModelConfig.cloome_config)
        state_dict = torch.load(args.ckpt_path)
        model.load_state_dict(state_dict["model"])
        processor = _transform(520, 520, True, "dataset", "crop")

    model.to(device)
    model.eval()

    print(f"Loading pre-trained model {args.model_type} from {args.ckpt_path}.")

    image_directory_path = os.path.join(
        args.data_dir, f"jumpcp/2020_11_04_CPJUMP1/{args.plate}"
    )

    file_ids = list(
        set(
            [
                "-".join(file.split("-")[:-1])
                for file in os.listdir(image_directory_path)
                if file.endswith(".npz")
            ]
        )
    )
    output_dir = os.path.join(constants.DATASET_DIR, f"jumpcp/output_emb/{args.model_type}")

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{args.plate}.h5")

    plate_info = pd.read_csv(
        os.path.join(constants.OUT_DIR, f"label_data/jumpcp/{args.plate}/load_data.csv")
    )
    if os.path.isfile(output_file):
        with h5py.File(output_file, "r") as hf:
            saved_ids = [i.decode("utf-8") for i in hf["well_position"][:]]
        file_ids = [id for id in file_ids if id.split("-")[1] not in saved_ids]
        n_total_ids = len(file_ids) + len(saved_ids)
    else:
        n_total_ids = len(file_ids)

    if n_total_ids != len(plate_info["Metadata_Well"].unique().tolist()):
        raise IndexError(f"{image_directory_path} does not contains images from all wells.")

    dataset = CellPaintingDataset(
        image_directory_path,
        file_ids,
        args.plate,
        processor,
        n_crops=args.n_crop,
        crop_size=args.crop_size,
        model_type=args.model_type,
    )
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=my_collate_fn
    )

    # Iterate over batches of data and generate embeddings
    progress_bar = tqdm(
        range(len(dataloader)),
        initial=0,
        desc="batch",
    )
    print(f"preprocessing plate: {args.plate}")
    for batch in dataloader:
        generate_embeddings(model, args.model_type, batch, output_file, device)
        progress_bar.update(1)

    print(f"{args.model_type} embeddings generation done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
