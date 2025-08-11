"""Evluation for retrieval between image and chemical pairs."""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from open_clip import get_tokenizer  # works on open-clip-torch>=2.23.0, timm>=0.9.8
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from configs.data_config import DataAugmentationConfig
from src import constants
from src.clip.clip import tokenize

# from src.clip.clip import load_long_clip, load_model
from src.helper import compute_model_stats, load
from src.transformations.cloome import CloomeAugmentation


class CellPaintingDataset(Dataset):
    """Customized dataset for loading all cell painting images per treatment"""

    def __init__(self, data_directory, file_ids, mole_struc, context_length, transforms):
        self.data_directory = data_directory

        if os.path.isdir(data_directory):
            self.is_hdf5 = False
            assert os.path.exists(
                data_directory
            ), f"Image directory {data_directory} does not exist."
        elif os.path.isfile(data_directory) and data_directory.endswith(".h5"):
            self.is_hdf5 = True
            self.h5_path = data_directory
            self.img_file = h5py.File(data_directory, "r")

            try:
                self.img_ids = [name.decode("utf-8") for name in self.img_file["names"][:]]
            except KeyError:
                self.img_ids = [
                    name.decode("utf-8").replace(".npz", "")
                    for name in self.img_file["well_id"][:]
                ]

        else:
            raise ValueError(
                "image_directory_path must be either a valid directory or HDF5 file."
            )

        self.file_ids = file_ids
        self.transforms = transforms
        self.mole_struc = mole_struc
        self.context_length = context_length

        if mole_struc == "morgan":
            # molecule_filename = "morgan_chiral_fps_all_1024.hdf5"
            molecule_filename = "molphenix_all_384_20_epochs.h5"

        elif mole_struc == "text":
            molecule_filename = "cell_long_captions_all.csv"
        elif mole_struc == "plate":
            molecule_filename = "cell_long_captions_all.csv"
        elif mole_struc == "embedding":
            molecule_filename = "caption_emb_small_all.csv"

        molecule_file = os.path.join(
            constants.DATASET_DIR, "bray2017/mol", molecule_filename
        )
        assert os.path.isfile(
            molecule_file
        ), f"Molecule file {molecule_file} does not exist."

        if mole_struc in ["text", "plate", "embedding"]:
            molecule_df = pd.read_csv(molecule_file, index_col=["ID"])
        elif mole_struc == "morgan":
            molecule_df = pd.read_hdf(molecule_file, key="df")

        molecule_df["new_index"] = molecule_df.index.str.rsplit("-", n=1).str[0]
        molecule_df.set_index("new_index", inplace=True)
        molecule_df = molecule_df[~molecule_df.index.duplicated(keep="first")]

        if self.context_length == 256:
            self.tokenizer = get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        self.molecule_df = molecule_df

    def __len__(self):
        """Return length of the dataset"""
        return len(self.file_ids)

    def __getitem__(self, idx):
        """Return item from the dataloader"""
        key = self.file_ids[idx]
        imgs = []

        if self.is_hdf5:
            mask = np.array([well_id.startswith(key) for well_id in self.img_ids])
            imgs = [torch.from_numpy(img) for img in self.img_file["embeddings"][mask]]

        else:
            files = [f for f in os.listdir(self.data_directory) if f.startswith(key)]
            for f in files:
                filepath = os.path.join(self.data_directory, f)
                image = self.load_view(filepath)
                if self.transforms is not None:
                    imgs.append(self.transforms(image))
                else:
                    image = torch.tensor(image)
                    if len(image.shape) == 3:
                        image = image.squeeze()
                    imgs.append(image)

        mol_fp = self.molecule_df.loc[key].values

        if self.mole_struc == "morgan":
            mol = torch.tensor(mol_fp).unsqueeze(0)
        elif self.mole_struc == "text":
            if self.context_length == 256:
                mol = self.tokenizer(mol_fp, context_length=self.context_length)
            elif self.context_length == 512:
                output = self.tokenizer(
                    mol_fp[0],
                    padding="max_length",
                    max_length=self.context_length,
                    return_tensors="pt",
                )
                mol = {
                    "input_ids": output["input_ids"].squeeze(0),
                    "attention_mask": output["attention_mask"].squeeze(0),
                }
            else:
                mol = tokenize(mol_fp, self.context_length, truncate=True).flatten()

        return (
            (
                imgs,
                {
                    "channels": np.asarray([c for c in range(5)]),
                },
            ),
            mol,
        )

    def load_view(self, filepath):
        """Load cell painting images"""
        npz = np.load(filepath, allow_pickle=True)
        if "sample" in npz:
            image = npz["sample"].astype(np.float32)
            return image
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUT_DIR
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="path to model check point",
        default=None,
    )
    parser.add_argument(
        "--split_label_dir",
        type=str,
        help="output parent directory",
        default=constants.SPLIT_LABEL_DIR,
    )
    parser.add_argument(
        "--is_train",
        help="whether to use training index",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_params",
        help="whether to use training index",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cell_clip",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--input_dim", default=768, type=int, help="dimension for embeddings "
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help=("Eval batch size."),
        default=32,
    )
    parser.add_argument(
        "--image_resolution_train",
        default=520,
        nargs="+",
        type=int,
        help="resolution for training set ",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="clip",
        help=("Loss type , e.g. cloob, clip."),
    )
    parser.add_argument(
        "--image_resolution_val",
        default=[520, 696],
        nargs="+",
        type=int,
        help="resolution for validation set ",
    )
    return parser.parse_args()


def get_features(test_loader, model, device, model_type):
    """Get CLIP embedding"""
    all_image_features = []
    all_text_features = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            for i in range(len(batch)):
                (images, extra_tokens), treatments = batch[i][0], batch[i][1]
                images = torch.stack(images)
                extra_tokens = {
                    k: torch.from_numpy(v)
                    .to(device)
                    .unsqueeze(0)
                    .expand(len(images), -1)
                    .contiguous()
                    for k, v in extra_tokens.items()
                }
                if model_type == "clip_channelvit":
                    img_features = model.encode_image(images.to(device), extra_tokens)
                else:
                    img_features = model.encode_image(images.to(device))

                # obtain perturbation mean
                img_features = torch.mean(img_features, axis=0).unsqueeze(0)

                if model_type in ["molphenix", "cloome_mpnn"]:
                    text_features = model.encode_mols(treatments.to(device))
                elif model_type in [
                    "bert_clip",
                    "clip_channelvit",
                    "cell_clip_mae",
                    "cell_clip",
                ]:
                    treatments = {k: v.to(device) for k, v in treatments.items()}
                    text_features = model.encode_text(treatments)
                else:
                    text_features = model.encode_text(treatments.to(device))

                all_image_features.append(img_features.detach().cpu())
                all_text_features.append(text_features.detach().cpu())

    return torch.cat(all_image_features), torch.cat(all_text_features)


def my_collate_fn(batch):
    """Customized collate function, return the list as it is"""
    return batch


def get_metrics(image_features, text_features):
    """Compute retrieval accuracy."""
    metrics = {}
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)
    rankings = {}
    all_top_samples = {}
    all_preds = {}
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        rankings[name] = ranking
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        all_preds[name] = preds
        top_samples = np.where(preds < 10)[0]
        all_top_samples[name] = top_samples
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return rankings, all_top_samples, all_preds, metrics, logits


def main(args):
    """Evaluating model performance in retrieval."""
    if not args.ckpt_path:
        checkpoint_path = hf_hub_download(
            "anasanchezf/cloome", "cloome-retrieval-zero-shot.pt"
        )
    else:
        checkpoint_path = args.ckpt_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(
        checkpoint_path,
        device,
        args.model_type,
        args.input_dim,
        args.loss_type,
    )

    if args.compute_params:

        dummy_img = torch.randn(
            1, 5, args.image_resolution_train, args.image_resolution_train
        ).to(device)

        if args.model_type == "cloome":
            dummy_treatment = torch.rand(1, 1024).to(device)
        elif args.model_type == "bert_clip":
            dummy_treatment = torch.rand(1, 512).long().to(device)
        else:
            dummy_treatment = torch.rand(1, 256).long().to(device)

        if args.model_type == "clip_channelvit":
            dummy_channels = torch.arange(5).unsqueeze(0).to(device)

            dummy_input = (dummy_img, dummy_channels, dummy_treatment)

        else:
            dummy_input = (dummy_img, dummy_treatment)

        params, flops = compute_model_stats(model, dummy_input)
        similarity_flops = 768 ** 2 * 512 / 1e9  # in GFLOPs

        print(f"Total Parameters: {params:.2f}M")
        print(f"Total FLOPs: {flops+similarity_flops:.2f} GFLOPs")

    sample_index_file = os.path.join(
        args.outdir, args.split_label_dir, "cellpainting-split-test-imgpermol.csv"
    )
    # preprocess_val = _transform(320, [224, 224], False, "dataset", "crop")
    configs = DataAugmentationConfig.cloome_old_config
    preprocess_val = CloomeAugmentation(
        n_px_tr=args.image_resolution_train,
        n_px_val=args.image_resolution_val,
        is_train=False,
        normalization_mean=configs["normalization"]["mean"],
        normalization_std=configs["normalization"]["std"],
        normalize="dataset",
        preprocess="crop",
    )
    # Load the dataset
    image_directory_path = os.path.join(constants.DATASET_DIR, "bray2017/cellpainting_full")
    if args.model_type in ["clip_resnet", "long_clip"]:
        mole_struc = "text"
        context_length = 248
    elif args.model_type in [
        "pubmed_clip",
        "pubmed_clip_phenom1",
        "mil_cell_clip",
        "cell_clip",
        "cell_sigclip",
        "cell_clip_mae",
        "clip_channelvit",
        "bert_clip",
    ]:
        mole_struc = "text"
        if args.model_type in [
            "bert_clip",
            "clip_channelvit",
            "cell_clip_mae",
            "cell_clip",
        ]:
            context_length = 512
        else:
            context_length = 256

        if args.model_type in ["cell_clip", "mil_cell_clip"]:
            image_directory_path = os.path.join(
                constants.DATASET_DIR, "bray2017/img/dinov2-giant_ind.h5"
            )
    elif args.model_type == "cell_clip_mae":
        mole_struc = "text"
        context_length = 256
        image_directory_path = os.path.join(
            constants.DATASET_DIR, "bray2017/cellpainting_full"
        )
    elif args.model_type == "pubmed_emb_clip":
        mole_struc = "embedding"
        context_length = 1532
    elif args.model_type == "cloome_mpnn":
        mole_struc = "morgan"
        context_length = 77
    else:
        mole_struc = "morgan"
        context_length = 77
    sample_index_file = pd.read_csv(
        os.path.join(
            args.outdir, args.split_label_dir, "cellpainting-split-test-imgpermol.csv"
        )
    )
    test_ids = [
        "-".join(idx.split("-")[:-1]) for idx in sample_index_file["SAMPLE_KEY"].tolist()
    ]
    test_dataset = CellPaintingDataset(
        data_directory=image_directory_path,
        file_ids=test_ids,
        context_length=context_length,
        mole_struc=mole_struc,
        transforms=preprocess_val,
    )
    test_loader = DataLoader(
        test_dataset, num_workers=4, batch_size=args.batch_size, collate_fn=my_collate_fn
    )
    # Calculate the image features
    val_img_features, val_text_features = get_features(
        test_loader, model, device, args.model_type
    )
    rankings, all_top_samples, all_preds, metrics, logits = get_metrics(
        val_img_features, val_text_features
    )
    print(metrics)


if __name__ == "__main__":
    args = parse_args()
    main(args)
