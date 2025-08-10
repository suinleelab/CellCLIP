"""
Utility functions for CLIP and long CLIP testing.

[1] https://github.com/beichenzbc/Long-CLIP/tree/main
"""
import hashlib
import os
import urllib
import warnings
from typing import List, Union

import torch
from packaging import version
from PIL import Image
from torch import nn
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from configs.model_config import ModelConfig
from src.clip.model import (
    CellCLIP,
    CellCLIP_MAE,
    CLIP_ChannelViT,
    CLIP_ResNet,
    Cloome,
    Cloome_MPNN,
    Cloome_old,
    Cloome_phenom1,
    Molphenix,
    build_model,
)
from src.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from src.constants import _MODELS

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, ",
                "but the SHA256 checksum does not match; re-downloading the file",
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
            ),
        ]
    )


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: str = None,
):
    """
    Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`,
        or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that
        the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(
            _MODELS[name],
            download_root or os.path.expanduser("/gscratch/aims/mingyulu/.cache/clip"),
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with open(model_path, "rb") as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(
                opened_file, map_location=device if jit else "cpu"
            ).eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(
                    f"File {model_path} is not a JIT archive. ",
                    "Loading as a state dict instead",
                )
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict(), long_clip=False).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)
    ][-1]

    def _node_get(node: torch._C.Node, key: str):
        """
        Gets attributes of a node which is polymorphic over return type.
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(
                    _node_get(node, "value")
                ).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def load_long_clip(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: str = None,
):
    """
    Load a long CLIP (248 context length)[1] with pretrained CLIP (77 context length)

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`,
        or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that
        the returned model can take as its input
    """

    def available_models() -> List[str]:
        """Returns the names of available CLIP models"""
        return list(_MODELS.keys())

    def _download(url: str, root: str):
        os.makedirs(root, exist_ok=True)
        filename = os.path.basename(url)

        expected_sha256 = url.split("/")[-2]
        download_target = os.path.join(root, filename)

        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")

        if os.path.isfile(download_target):
            if (
                hashlib.sha256(open(download_target, "rb").read()).hexdigest()
                == expected_sha256
            ):
                return download_target
            else:
                warnings.warn(
                    f"{download_target} exists, ",
                    "but the SHA256 checksum does not match; re-downloading the file",
                )

        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            != expected_sha256
        ):
            raise RuntimeError(
                "Model has been downloaded but the SHA256 checksum does not not match"
            )

        return download_target

    if name in _MODELS:
        model_path = _download(
            _MODELS[name], download_root or os.path.expanduser("~/.cache/clip")
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with open(model_path, "rb") as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(
                opened_file, map_location=device if jit else "cpu"
            ).eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(
                    f"File {model_path} is not a JIT archive. ",
                    "Loading as a state dict instead",
                )
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), long_clip=False).to(device)

    positional_embedding_pre = model.positional_embedding.type(model.dtype)

    length, dim = positional_embedding_pre.shape
    keep_len = 20
    posisitonal_embedding_new = torch.zeros(
        [4 * length - 3 * keep_len, dim], dtype=model.dtype
    )

    for i in range(keep_len):
        posisitonal_embedding_new[i] = positional_embedding_pre[i]
    for i in range(length - 1 - keep_len):
        posisitonal_embedding_new[4 * i + keep_len] = positional_embedding_pre[i + keep_len]
        posisitonal_embedding_new[4 * i + 1 + keep_len] = (
            3 * positional_embedding_pre[i + keep_len] / 4
            + 1 * positional_embedding_pre[i + 1 + keep_len] / 4
        )
        posisitonal_embedding_new[4 * i + 2 + keep_len] = (
            2 * positional_embedding_pre[i + keep_len] / 4
            + 2 * positional_embedding_pre[i + 1 + keep_len] / 4
        )
        posisitonal_embedding_new[4 * i + 3 + keep_len] = (
            1 * positional_embedding_pre[i + keep_len] / 4
            + 3 * positional_embedding_pre[i + 1 + keep_len] / 4
        )

    posisitonal_embedding_new[4 * length - 3 * keep_len - 4] = (
        positional_embedding_pre[length - 1]
        + 0
        * (positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2])
        / 4
    )
    posisitonal_embedding_new[4 * length - 3 * keep_len - 3] = (
        positional_embedding_pre[length - 1]
        + 1
        * (positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2])
        / 4
    )
    posisitonal_embedding_new[4 * length - 3 * keep_len - 2] = (
        positional_embedding_pre[length - 1]
        + 2
        * (positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2])
        / 4
    )
    posisitonal_embedding_new[4 * length - 3 * keep_len - 1] = (
        positional_embedding_pre[length - 1]
        + 3
        * (positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2])
        / 4
    )

    positional_embedding_res = posisitonal_embedding_new.clone()

    model.positional_embedding = nn.Parameter(posisitonal_embedding_new, requires_grad=True)
    model.positional_embedding_res = nn.Parameter(
        positional_embedding_res, requires_grad=True
    )

    if str(device) == "cpu":
        model.float()

    return model, _transform(model.visual.input_resolution)


def load_model(
    model_type,
    pretrained: bool = False,
    name: str = "ViT-B/16",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: str = None,
    image_resolution_train: int = 224,
    vision_width: int = 768,
    loss_type: str = "clip",
):
    """Helpler function that initialize different types of model."""

    if model_type == "cloome":
        model = Cloome(**ModelConfig.cloome_config)
    elif model_type == "cloome_old":
        model = Cloome_old(**ModelConfig.old_cloome_config)
    elif model_type == "clip":
        model, _ = load(pretrained, name, device, jit, download_root)

    elif model_type == "clip_channelvit":
        model = CLIP_ChannelViT(**ModelConfig.clip_channelvit_config)

    elif model_type == "clip_resnet":

        model = CLIP_ResNet(**ModelConfig.clip_resnet_config)

        if pretrained:
            pre_trained_clip, _ = load(name, device, jit, download_root)

            # Load pre-trained CLIP text encoder.

            model.transformer = pre_trained_clip.transformer
            model.text_projection = pre_trained_clip.text_projection

            model.token_embedding = pre_trained_clip.token_embedding
            model.positional_embedding = pre_trained_clip.positional_embedding

    elif model_type == "cell_clip":
        config = ModelConfig.cell_clip_config
        config["vision_width"] = vision_width
        config["use_bias"] = True if loss_type in ["s2l", "sigclip"] else False
        model = CellCLIP(**config)
    elif model_type == "cell_clip_mae":
        model = CellCLIP_MAE(**ModelConfig.cell_clip_mae_config)
    elif model_type == "molphenix":
        model = Molphenix(**ModelConfig.molphenix_config)
    elif model_type == "cloome_phenom1":
        model = Cloome_phenom1(**ModelConfig.cloome_phenom1_config)
    elif model_type == "cloome_mpnn":
        model = Cloome_MPNN(**ModelConfig.cloome_mpnn_config)

    return model


def tokenize(
    texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False
) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
        A two-dimensional tensor containing the resulting tokens,
        shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0,
        since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result
