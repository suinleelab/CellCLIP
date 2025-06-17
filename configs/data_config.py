"""Configurations for the data transformations."""


class DataAugmentationConfig:
    """Configuration for cell augmentation settings."""

    cloome_config = {
        "name": "CellAugmentation",
        "channel_mask": [],
        "normalization": {
            "mean": [43.4913, 36.0689, 46.5934, 43.5823, 24.7100],
            "std": [23.1659, 23.4488, 27.0771, 22.9874, 28.9110],
        },
    }
    cloome_old_config = {
        "name": "CellAugmentation",
        "channel_mask": [],
        "normalization": {
            "mean": [47.1314, 40.8138, 53.7692, 46.2656, 28.7243],
            "std": [24.1384, 23.6309, 28.1681, 23.4018, 28.7255],
        },
    }
    jumpcp_config = {
        "name": "CellAugmentation",
        "channel_mask": [],
        "normalization": {
            "mean": [
                4.031743599139058,
                1.565935237087539,
                3.77367898215863,
                3.4605251427133257,
                4.1723172504050225,
                6.780529773318951,
                6.787385700135139,
                6.778120829362721,
            ],
            "std": [
                17.318438884455695,
                12.015918256263747,
                16.966058078452495,
                15.064776266287147,
                17.964118200870608,
                21.638766346725316,
                21.670565699654457,
                21.639488585095584,
            ],
        },
    }
