"""Evluation for retrieval between image and chemical pairs."""
import argparse
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm import tqdm

# from configs.data_config import DataAugmentationConfig
from src import constants
from src.datasets import CellPainting
from src.helpler import load
from src.transformations.cloome import _transform


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
        "--embedding_type", type=str, help="embedding for images", default=None
    )
    parser.add_argument(
        "--input_dim", default=768, type=int, help="dimension for embeddings "
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
        "--unique",
        help="whether to use average embedding for evaluation",
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
        "--loss_type",
        type=str,
        default="clip",
        help=("Loss types, e.g. clip, cloob."),
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
        help=(
            "resolution for training set. "
            "Need to specify when not using embedding models. "
        ),
    )
    parser.add_argument(
        "--image_resolution_val",
        default=[520, 696],
        nargs="+",
        type=int,
        help=(
            "resolution for validation set. "
            "Need to specify when not using embedding models. "
        ),
    )

    return parser.parse_args()


def get_features(test_loader, model, device, model_type):
    """Get CLIP embedding"""
    all_image_features = []
    all_text_features = []

    with torch.no_grad():
        for batch in tqdm(test_loader):

            (images, extra_tokens), treatments = batch
            extra_tokens = {k: v.to(device) for k, v in extra_tokens.items()}

            if args.model_type == "cell_clip":
                images = model.encode_mil(images.to(device))

            if model_type == "clip_channelvit":
                img_features = model.encode_image(images.to(device), extra_tokens)
            else:
                img_features = model.encode_image(images.to(device))

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
    if args.unique and args.embedding_type is None:
        raise ValueError("Embedding type cannot be None")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt_path is None:
        ckpt_path = hf_hub_download(
            "suinleelab/CellCLIP",
            "model.safetensors"
        )
    else:
        ckpt_path = args.ckpt_path

    model = load(ckpt_path, device, args.model_type, args.input_dim, args.loss_type)

    sample_index_file = os.path.join(
        args.outdir, args.split_label_dir, "cellpainting-split-test-imgpermol.csv"
    )

    preprocess_val = _transform(
        args.image_resolution_train, args.image_resolution_val, False, "dataset", "crop"
    )

    # Load the dataset
    image_directory_path = os.path.join(constants.DATASET_DIR, "bray2017/cellpainting_full")

    if args.model_type in ["clip_resnet", "long_clip"]:
        mole_struc = "text"
        context_length = 248
    elif args.model_type in [
        "pubmed_clip",
        "pubmed_clip_phenom1",
        "cell_clip",
        "cell_sigclip",
        "cell_clip_mae",
        "clip_channelvit",
    ]:
        mole_struc = "text"
        molecule_path = os.path.join(
            constants.DATASET_DIR, "bray2017/mol/cell_long_captions_all.csv"  #
        )

        if args.model_type in [
            "cell_clip",
            "bert_clip",
            "clip_channelvit",
            "cell_clip_mae",
            "cell_clip"
        ]:
            context_length = 512
        else:
            context_length = 256

    elif args.model_type == "molphenix":
        mole_struc = "morgan"
        context_length = 77
        molecule_path = os.path.join(
            constants.DATASET_DIR, "bray2017/mol/molphenix_all_384_20_epochs.h5"  #
        )
    else:
        molecule_path = os.path.join(
            constants.DATASET_DIR, "bray2017/mol/morgan_chiral_fps_all_1024.hdf5"  #
        )
        mole_struc = "morgan"
        context_length = 77

    if args.unique:
        image_directory_path = os.path.join(
            constants.DATASET_DIR, "bray2017/img", args.embedding_type
        )
        preprocess_val = None

    test_dataset = CellPainting(
        sample_index_file,
        mole_struc=mole_struc,
        context_length=context_length,
        transforms=preprocess_val,
        image_directory_path=image_directory_path,
        molecule_path=molecule_path,
        unique=args.unique,
    )
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=args.batch_size)

    # Calculate the image features
    val_img_features, val_text_features = get_features(
        test_loader, model, device, args.model_type
    )
    # val_text_features = val_text_features[torch.randperm(val_text_features.size(0))]

    _, _, _, metrics, _ = get_metrics(
        val_img_features, val_text_features
    )
    print(metrics)


if __name__ == "__main__":
    args = parse_args()
    main(args)
