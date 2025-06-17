"""Scripts to preprocess molphenix feature with a pretrained MPNN++"""
import os
from collections import OrderedDict

import h5py
import pandas as pd
import torch
from graphium.config._loader import (
    load_accelerator,
    load_architecture,
    load_datamodule,
    load_yaml_config,
)
from graphium.data.collate import graphium_collate_fn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from src import constants
from src.mpnn.data.datamodule import MultitaskFromSmilesDataModuleBray
from src.mpnn.model import FullGraphMultiTaskNetworkNew


def convert_features_dtype(feats):
    """Convert features to dtype"""
    if isinstance(feats, torch.Tensor):
        feats = feats.to(torch.float32)
    elif isinstance(feats, (Data, Batch, dict)):
        for key, val in feats.items():
            if isinstance(val, torch.Tensor) and (val.is_floating_point()):
                feats[key] = val.to(dtype=torch.float32)
    return feats


def get_existing_keys(outfile_hdf):
    """Retrieve existing sample keys from the HDF5 file to avoid duplicates."""
    if not os.path.exists(outfile_hdf):
        return set()

    with h5py.File(outfile_hdf, "r") as f:
        if "df" in f:
            return set(f["df"]["index"][:])  # Read existing sample keys
    return set()


def save_mol_features(dataloader, model, outfile_hdf, dataset_name):
    """Extract molecular features and save only new entries to HDF5."""
    # existing_keys = get_existing_keys(outfile_hdf)

    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Processing {dataset_name}"
        ):
            chem_features = convert_features_dtype(batch["features"])
            sample_keys = [
                str(key) for key in batch["sample_keys"]
            ]  # Ensure keys are strings
            # Filter out already saved keys
            # new_keys = [key for key in sample_keys if key not in existing_keys]
            # if not new_keys:
            #     continue

            output_features = model(chem_features).detach().cpu().numpy()

            # Create DataFrame with only new entries
            feature_columns = [f"dim_{i}" for i in range(output_features.shape[1])]
            feature_df = pd.DataFrame(
                output_features, index=sample_keys, columns=feature_columns
            )

            try:
                feature_df.to_hdf(
                    outfile_hdf, key="df", mode="a", format="table", append=True
                )
            except Exception as e:
                print(f"Error saving {dataset_name} batch {batch_idx}: {e}")

            # Free memory
            del output_features
            torch.cuda.empty_cache()


def main():
    """Main function to load the model and extract features"""

    # Load config and model
    cfg = load_yaml_config(
        os.path.join(
            constants.OUT_DIR, "configs/graphium_configs/config_gps_10M_pcqm4m.yaml"
        )
    )
    cfg, accelerator_type = load_accelerator(cfg)
    datamodule = load_datamodule(cfg, accelerator_type)

    # Initialize the network
    _, model_kwargs = load_architecture(
        cfg,
        in_dims=datamodule.in_dims,
    )
    state_dict = torch.load(
        os.path.join(constants.OUT_DIR, "results/mpnn/models/last-v1.ckpt"),
        map_location="cpu",
    )

    # Rename keys to match the expected model format
    new_state_dict = OrderedDict()
    for key, value in state_dict["state_dict"].items():
        new_key = key.replace("model.encoder_manager.", "encoder_manager.")
        new_key = new_key.replace("model.", "")
        new_key = new_key.replace("node_fully_connected", "node_model.fully_connected")
        new_key = new_key.replace("edge_fully_connected", "edge_model.fully_connected")
        new_state_dict[new_key] = value

    # Load model
    model = FullGraphMultiTaskNetworkNew(**model_kwargs)
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    outfile_hdf = os.path.join(
        constants.DATASET_DIR, "jumpcp/mol/molphenix_all_384_20_epochs.h5"
    )
    # Define dataset configurations
    dataset_configs = [
        # ("bray2017_train", "fit", ["train_ds", "val_ds"]),
        # ("bray2017_train", "test", ["test_ds"]),
        # ("bray2017_eval", "fit", ["train_ds", "val_ds"]),
        # ("bray2017_eval", "test", ["test_ds"]),
        # ("bray2017_test", "fit", ["train_ds", "val_ds"]),
        # ("bray2017_test", "test", ["test_ds"]),
        ("jumpcp", "fit", ["train_ds", "val_ds"]),
        ("jumpcp", "test", ["test_ds"]),
    ]

    print("Extracting and saving features batch-wise...")

    for config_name, setup_mode, dataset_splits in dataset_configs:
        cfg_path = os.path.join(
            constants.OUT_DIR, f"configs/graphium_configs/{config_name}.yaml"
        )
        dataset_cfg = load_yaml_config(cfg_path)

        datamodule = MultitaskFromSmilesDataModuleBray(**dataset_cfg["datamodule"]["args"])
        datamodule.prepare_data()
        datamodule.setup(setup_mode)  # Setup either "fit" (train+val) or "test"

        print(f"Saving module model:{setup_mode};{dataset_splits}..")

        for split in dataset_splits:
            dataloader = DataLoader(
                getattr(datamodule, split),
                batch_size=32,
                shuffle=False,
                collate_fn=graphium_collate_fn,
            )
            save_mol_features(
                dataloader, model, outfile_hdf, dataset_name=f"{config_name}_{split}"
            )

    print(f"All features saved to {outfile_hdf}")


if __name__ == "__main__":
    """Main function to perform feature preprocessing"""
    main()
