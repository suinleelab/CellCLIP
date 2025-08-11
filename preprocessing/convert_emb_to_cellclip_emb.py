"""Evaluation pipeline for silbiling perturbation in 2024 Chandrasekaran et al."""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import constants
from src.helper import load


class CellPaintingDataset(Dataset):
    """Customized dataset for loading embedding images"""

    def __init__(self, data_directory, plate):
        self.img_file = h5py.File(data_directory, "r")  # Lazy load HDF5 file
        self.img_ids = [name.decode("utf-8") for name in self.img_file["names"][:]]

        self.plate_info = pd.read_csv(
            f"/gscratch/aims/mingyulu/cell_painting/label_data/jumpcp/{plate}/load_data.csv"
        )

        if len(self.img_ids) != len(self.plate_info["Metadata_Well"].unique().tolist()):
            raise IndexError(
                f"{data_directory} does not contains embeddings from all wells."
            )

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
        return len(self.img_ids)

    def __getitem__(self, idx):
        """Return item from the dataloader"""
        idx = self.img_ids[idx]
        well_position = idx.split("-")[-1]
        embeddings_idx = self.img_ids.index(idx)
        X = self.img_file["embeddings"][embeddings_idx]
        X = np.mean(X, axis=0)

        sample_id = self.meta_data[self.meta_data.well_position == well_position][
            "broad_sample"
        ].values[0]

        if pd.isna(sample_id):
            sample_id = "control"

        return (well_position, sample_id, X)


def my_collate_fn(batch):
    """Customized collate function, return the list as it is"""
    return batch


def generate_embeddings(pretrained_clip, batch, output_file, device):
    """Process a batch of images, generate embeddings, and save them"""

    batch_size = len(batch)

    for i in range(batch_size):
        well_position, broad_sample, img = batch[i][0], batch[i][1], batch[i][2]

        with torch.no_grad():
            img = torch.tensor(img).to(device).unsqueeze(0)
            output = pretrained_clip.encode_image(img).detach().cpu().numpy()
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


# def load(model_path, device, model_type, vision_width):
#     """Load pretrained model"""

#     checkpoint = torch.load(model_path)
#     cell_clip_config = ModelConfig.newcell_clip_config
#     cell_clip_config["vision_width"] = vision_width
#     model = New_CellClip(**cell_clip_config)

#     state_dict = checkpoint["model"]

#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()

#     print(f"Loading pre-trained model {model_type} from {model_path}.")

#     return model


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
        "--input_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cell_clip",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--pretrained_emb",
        type=str,
        default="clip@B16_224_512",
        help=("Embeddings for pretrained model, e.g. clip@B16_224_512"),
    )

    return parser.parse_args()


def main(args):
    """Perform inference on jumpcp datasets."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load(args.ckpt_path, device, args.model_type, args.input_dim)

    plates = [
        plate.split("_")[0]
        for plate in os.listdir(
            os.path.join(constants.DATASET_DIR, "jumpcp/img", args.pretrained_emb)
        )
        if plate.startswith("BR")
    ]
    output_dir = os.path.join(
        constants.DATASET_DIR, "jumpcp/output_emb", args.model_type, args.pretrained_emb
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for plate in plates:
        # Load the dataset
        image_directory_path = os.path.join(
            constants.DATASET_DIR,
            "jumpcp/img",
            args.pretrained_emb,
            f"{plate}_{args.pretrained_emb}.h5",
        )
        output_file = os.path.join(output_dir, f"{plate}.h5")

        dataset = CellPaintingDataset(image_directory_path, plate)
        dataloader = DataLoader(
            dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=my_collate_fn
        )

        # Iterate over batches of data and generate embeddings
        progress_bar = tqdm(
            range(len(dataloader)),
            initial=0,
            desc="batch",
        )
        print(f"preprocessing plate: {plate}")
        for batch in dataloader:
            generate_embeddings(model, batch, output_file, device)
            progress_bar.update(1)

    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
