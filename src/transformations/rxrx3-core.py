"""
Transformation function inherited from [1]
[1]https://github.com/ml-jku/cloome/blob/main/src/clip/clip.py
"""
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomCrop,
    RandomResizedCrop,
    RandomRotation,
    Resize,
    ToTensor,
)

from src.clip.clip import load

pretrained_clip, _ = load("ViT-B/16", device="cpu")
pretrained_clip.eval()


def _transform(
    n_px_tr: int,
    n_px_val: int,
    is_train: bool,
    normalize: str = "dataset",
    preprocess: str = "downsize",
):
    """Transformation function from [1]."""
    if normalize == "img":
        normalize = NormalizeByImage()
    elif normalize == "dataset":
        normalize = Normalize(
            (47.1314, 40.8138, 53.7692, 46.2656, 28.7243),
            (24.1384, 23.6309, 28.1681, 23.4018, 28.7255),
        )  # normalize for CellPainting
    elif normalize == "None":
        normalize = None

    if is_train:
        if preprocess == "crop":
            resize = RandomCrop(n_px_tr)
        elif preprocess == "downsize":
            resize = RandomResizedCrop(
                n_px_tr, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC
            )
        elif preprocess == "rotate":
            resize = Compose([RandomRotation((0, 360)), CenterCrop(n_px_tr)])

    else:
        if preprocess == "crop" or "rotate":
            resize = Compose(
                [
                    CenterCrop(n_px_val),
                ]
            )
        elif preprocess == "downsize":
            resize = Compose(
                [
                    Resize(n_px_val, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(n_px_val),
                ]
            )
    if normalize:
        return Compose(
            [
                ToTensor(),
                resize,
                normalize,
            ]
        )
    else:
        return Compose(
            [
                ToTensor(),
                resize,
            ]
        )


class NormalizeByImage(object):
    """
    Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Return:
        ------
            Tensor: Normalized Tensor image.
        """
        for t in tensor:
            t.sub_(t.mean()).div_(t.std() + 1e-7)
        return tensor
