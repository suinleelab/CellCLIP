# © Recursion Pharmaceuticals 2024
from typing import Dict

import timm.models.vision_transformer as vit
import torch
import torch.nn as nn

def build_imagenet_baselines() -> Dict[str, torch.jit.ScriptModule]:
    """This returns the prepped imagenet encoders from timm, not bad for microscopy data."""
    vit_backbones = [
        _make_vit_bray2017(vit.vit_small_patch16_384),
        _make_vit_bray2017(vit.vit_base_patch16_384),
        _make_vit_bray2017(vit.vit_base_patch8_224),
        _make_vit_bray2017(vit.vit_large_patch16_384),
    ]
    model_names = [
        "vit_small_patch16_384",
        "vit_base_patch16_384",
        "vit_base_patch8_224",
        "vit_large_patch16_384",
    ]
    imagenet_encoders = list(map(_make_torchscripted_encoder_bray2017, vit_backbones))
    return {name: model for name, model in zip(model_names, imagenet_encoders)}


def _make_torchscripted_encoder(vit_backbone) -> torch.jit.ScriptModule:
    dummy_input = torch.testing.make_tensor(
        (2, 6, 256, 256),
        low=0,
        high=255,
        dtype=torch.uint8,
        device=torch.device("cpu"),
    )
    encoder = torch.nn.Sequential(
        Normalizer(),
        torch.nn.LazyInstanceNorm2d(
            affine=False, track_running_stats=False
        ),  # this module performs self-standardization, very important
        vit_backbone,
    ).to(device="cpu")
    _ = encoder(dummy_input)  # get those lazy modules built
    return torch.jit.freeze(torch.jit.script(encoder.eval()))

def _make_torchscripted_encoder_bray2017(vit_backbone) -> torch.jit.ScriptModule:
    dummy_input = torch.testing.make_tensor(
        (2, 5, 256, 256),
        low=0,
        high=255,
        dtype=torch.uint8,
        device=torch.device("cpu"),
    )
    encoder = torch.nn.Sequential(
        Normalizer(),
        torch.nn.LazyInstanceNorm2d(
            affine=False, track_running_stats=False
        ),  # this module performs self-standardization, very important
        vit_backbone,
    ).to(device="cpu")
    _ = encoder(dummy_input)  # get those lazy modules built
    return encoder

def _make_vit(constructor):
    return constructor(
        pretrained=True,  # download imagenet weights
        img_size=256,  # 256x256 crops
        in_chans=6,  # we expect 6-channel microscopy images
        num_classes=0,
        fc_norm=None,
        class_token=True,
        global_pool="avg",  # minimal perf diff btwn "cls" and "avg"
    )

def _make_vit_bray2017(constructor):
    return constructor(
        pretrained=True,  # download imagenet weights
        img_size=256,  # 256x256 crops
        in_chans=5,  # we expect 6-channel microscopy images
        num_classes=0,
        fc_norm=None,
        class_token=True,
        global_pool="avg",  # minimal perf diff btwn "cls" and "avg"
    )

class ViTClassifier(nn.Module):
    """Vision Transformer (ViT)-based classifier for weakly supervised learning."""

    def __init__(self, vit_type: str, num_classes: int) -> None:
        """
        Initialize the ViTClassifier.

        Args:
            vit_type (str): Type of ViT backbone (e.g., 'vit_base_patch16_384').
            num_classes (int): Number of output classes for classification.
        """
        super().__init__()

        # Load the pre-trained ViT backbones
        model_dicts = build_imagenet_baselines()
        if vit_type not in model_dicts:
            raise ValueError(f"Invalid ViT type: {vit_type}. Must be one of {list(model_dicts.keys())}.")

        self.vit_type = vit_type
        self.vit_backbone = model_dicts[vit_type]

        if vit_type == "vit_base_patch16_384":
            vit_output_dim = 768
        elif vit_type == "vit_large_patch16_384":
            vit_output_dim = 1024
        else:
            raise ValueError(f"Unsupported ViT type: {vit_type}. Please add its output dimension.")

        self.classifier_head = nn.Linear(vit_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier."""
        embeddings = self.vit_backbone(x)
        output = self.classifier_head(embeddings)
        return output

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from the ViT backbone."""
        return self.vit_backbone(x)



class Normalizer(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float()
        pixels /= 255.0
        return pixels
