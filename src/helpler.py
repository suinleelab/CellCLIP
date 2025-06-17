"""
Utility function and class
[1] https://github.com/openai/CLIP/issues/111
"""
import glob
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.distributed as dist
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from configs.model_config import ModelConfig
from src.clip.model import (
    BERT_CLIP,
    CellClip,
    CellCLIP_MAE,
    CLIP_ChannelViT,
    CLIP_ResNet,
    Cloome,
    Cloome_MPNN,
    Cloome_old,
    Cloome_phenom1,
    MilCellClip,
    Molphenix,
    Pubmed_CLIP,
    Pubmed_CLIP_phenom1,
)


def compute_model_stats(model, input_size=(3, 224, 224)):
    """
    Compute number of parameters and FLOPs for a given model.

    Args:
    ----
        model (torch.nn.Module): Pretrained model (e.g., DINOv2 ViT).
        input_size (tuple): Input tensor size, default (3, 224, 224).

    Return:
    ------
        params (float): Total parameters (in millions).
        flops (float): Total FLOPs (in billions).
    """
    model.eval()  # set to eval mode
    dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)

    # Compute FLOPs
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total() / 1e9  # convert to GFLOPs (GigaFLOPs)

    # Compute Parameters
    params_table = parameter_count_table(model)
    params = sum(p.numel() for p in model.parameters()) / 1e6  # convert to millions

    print(params_table)  # optional: shows nice table

    return params, total_flops


def parallelize(func, iterable, n_workers, **kwargs):
    """Helper function for parallelization"""
    f = partial(func, **kwargs)
    if n_workers > 1:
        with Pool(n_workers) as p:
            results = p.map(f, iterable)
    else:
        results = list(map(f, iterable))
    return results


def compute_grad_norm(accelerator, model):
    """Compute gradient norm. To be run under the accelerator main process."""
    model = accelerator.unwrap_model(model)
    grads = [
        param.grad.detach().cpu().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    return torch.cat(grads).norm()


def compute_param_norm(accelerator, model):
    """Compute the parameter norm. To be run under the accelerator main process."""
    model = accelerator.unwrap_model(model)
    params = [
        param.data.detach().cpu().flatten()
        for param in model.parameters()
        if param.data is not None
    ]
    return torch.cat(params).norm()


def get_max_steps(folder_path):
    """Get maximum number of training steps for results in a folder."""

    path_pattern = os.path.join(folder_path, "ckpt_steps_*.pt")
    files = glob.glob(path_pattern)

    if not files:
        return None

    max_steps = max(
        files, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
    )
    return int(os.path.basename(max_steps).split("_")[-1].split(".")[0])


def print_args(args):
    """Print script name and args."""
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")


class AllGatherFunction(torch.autograd.Function):
    """
    Custom autograd function for distributed training that performs an all-gather
    on input tensors across all nodes during the forward pass and sums
    then scatters gradients during the backward pass.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)


def all_gather(tensor):
    """Wrapper function for all-gather."""
    return AllGatherFunction.apply(tensor)


def get_metrics(image_features, text_features):
    """Evaluate retrieval."""

    metrics = {}
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        logit = logit.detach().cpu()
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def load(model_path, device, model_type, input_dim=768, loss_type="clip"):
    """Load pretrained model from checkpoint."""
    MODEL_CONFIGS = {
        "old_cloome": (Cloome_old, ModelConfig.old_cloome_config),
        "cloome": (Cloome, ModelConfig.cloome_config),
        "cloome_mpnn": (Cloome_MPNN, ModelConfig.cloome_mpnn_config),
        "bert_clip": (BERT_CLIP, ModelConfig.bert_clip_config),
        "clip_resnet": (CLIP_ResNet, ModelConfig.clip_resnet_config),
        "clip_channelvit": (CLIP_ChannelViT, ModelConfig.clip_channelvit_config),
        "pubmed_clip": (Pubmed_CLIP, ModelConfig.pubmed_clip_config),
        "cell_clip_mae": (CellCLIP_MAE, ModelConfig.cell_clip_mae_config),
        "cloome_phenom1": (Cloome_phenom1, ModelConfig.cloome_phenom1_config),
        "molphenix": (Molphenix, ModelConfig.molphenix_config),
    }

    # Special handling for new_cell_clip due to dynamic config
    if model_type == "cell_clip":

        model_config = ModelConfig.cell_clip_config.copy()
        model_config["vision_width"] = input_dim
        model_config["use_bias"] = True if loss_type in ["sigclip", "s2l"] else False

        model = CellClip(**model_config)

    elif model_type == "mil_cell_clip":

        model_config = ModelConfig.mil_cell_clip_config.copy()
        model_config["vision_width"] = input_dim
        model_config["use_bias"] = True if loss_type in ["sigclip", "s2l"] else False

        model = MilCellClip(**model_config)

    elif model_type == "pubmed_clip_phenom1":

        model_config = ModelConfig.pubmed_clip_phenom1_config.copy()
        model_config["vision_width"] = input_dim
        model = Pubmed_CLIP_phenom1(**model_config)
    else:
        try:
            ModelClass, config = MODEL_CONFIGS[model_type]
            model = ModelClass(**config)
        except KeyError:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types are: {list(MODEL_CONFIGS.keys()) + ['new_cell_clip']}"
            )

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {model_path}: {str(e)}")

    # Handle state dict format
    if model_type == "old_cloome":
        state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
        }
    else:
        state_dict = checkpoint["model"]

    # Convert to float32 if on CPU
    if str(device) == "cpu":
        model.float()

    # Load state dict and move to device
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict into model: {str(e)}")

    model.to(device)
    model.eval()

    print(f"Successfully loaded {model_type} model from {model_path}")

    return model
