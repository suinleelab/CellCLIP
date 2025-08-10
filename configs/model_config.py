"""Configurations for models."""


class ModelConfig:
    """Configuration for CLIP model."""

    clip_resnet_config = {
        "embed_dim": 512,
        # vision
        "vision_layers": [3, 4, 6, 3],
        "input_channels": 5,
        "embed_dim": 512,
        "learnable_logit_scale": True,
        "logit_scale": 14.3,
        # text
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
    }

    pubmed_clip_config = {
        "embed_dim": 512,
        # vision
        "image_resolution": 520,
        "vision_layers": [3, 4, 6, 3],
        "input_channels": 5,
        "vision_width": 64,
        # text
        "context_length": 256,
    }
    bert_clip_config = {
        "embed_dim": 512,
        # vision
        "image_resolution": 520,
        "vision_layers": [3, 4, 6, 3],
        "input_channels": 5,
        "vision_width": 64,
        # text
        "context_length": 512,
    }
    pubmedbert_clip_config = {
        "embed_dim": 512,
        # vision
        "image_resolution": 520,
        "vision_layers": [3, 4, 6, 3],
        "input_channels": 5,
        "vision_width": 64,
        # text
        "context_length": 512,
    }
    molphenix_config = {
        "embed_dim": 256,  # latent emb dims
        # vision
        "vision_width": 384,  # input (vision) embedding dims
        "vision_heads": 8,
    }
    cell_clip_config = {
        "embed_dim": 512,  # latent emb dims
        # vision
        "vision_layers": 12,
        "input_channels": 5,
        "vision_width": 768,  # vision embedding dims
        "vision_heads": 8,
        # text
        "context_length": 256,
        "pooling": "attention",
    }
    cell_sigclip_config = {
        "embed_dim": 512,  # latent emb dims
        # vision
        "vision_layers": 12,
        "input_channels": 5,
        "vision_width": 768,  # vision embedding dims
        "vision_heads": 8,
        # text
        "context_length": 256,
    }
    cell_clip_mae_config = {
        "embed_dim": 384,
        # text
        "context_length": 256,
        "pretrained": False,
    }
    clip_channelvit_config = {
        "embed_dim": 512,
        # vision
        "image_resolution": 224,
        "channels": 5,
        "depth": 12,
        "mlp_ratio": 4,
        "vision_heads": 4,
        "vision_patch_size": 16,
        "hcs": True,
        # text
        "context_length": 256,
    }
    clip_small_channelvit_config = {
        "embed_dim": 384,
        # vision
        "image_resolution": 224,
        "channels": 5,
        "depth": 12,
        "mlp_ratio": 4,
        "vision_heads": 6,
        "vision_patch_size": 8,
        "hcs": True,
        # text
        "input_size": 1024,
        "molecule_layers": 4,
        "hidden_dim": 1024,
    }
    cloome_phenom1_config = {
        # vision
        "vision_width": 384,  # vision embedding dims
        "vision_heads": 8,
        "vision_layers": 6,
        "embed_dim": 512,
        "learnable_logit_scale": True,
        "logit_scale": 14.3,
        # moleculer encoder
        "input_size": 1024,
        "molecule_layers": 4,
        "hidden_dim": 1024,
    }
    cloome_config = {
        # vision
        "vision_layers": [3, 4, 6, 3],
        "input_channels": 5,
        "embed_dim": 512,
        "learnable_logit_scale": True,
        "logit_scale": 14.3,
        # moleculer encoder
        "input_size": 1024,
        "molecule_layers": 4,
        "hidden_dim": 1024,
    }
    cloome_mpnn_config = {
        # vision
        "vision_layers": [3, 4, 6, 3],
        "input_channels": 5,
        "embed_dim": 512,
        # "learnable_logit_scale": True,
        # "logit_scale": 14.3,
        "vision_width": 384,  # input (vision) embedding dims
        "vision_heads": 8,
    }
    old_cloome_config = {
        # vision
        "vision_layers": [3, 4, 6, 3],
        "input_channels": 5,
        "embed_dim": 512,
        "learnable_inv_tau": True,
        # moleculer encoder
        "input_size": 1024,
        "molecule_layers": 4,
        "hidden_dim": 1024,
    }
    mae_config = {
        "_attn_implementation_autoset": True,
        "apply_loss_unmasked": False,
        "architectures": ["MAEModel"],
        "crop_size": -1,
        "decoder": {
            "_target_": "mae_modules.CAMAEDecoder",
            "depth": 8,
            "embed_dim": 512,
            "mlp_ratio": 4,
            "norm_layer": {
                "_partial_": True,
                "_target_": "torch.nn.LayerNorm",
                "eps": 1e-06,
            },
            "num_heads": 16,
            "num_modalities": 5,
            "qkv_bias": True,
            "tokens_per_modality": 256,
        },
        "encoder": {
            "_target_": "mae_modules.MAEEncoder",
            "channel_agnostic": True,
            "max_in_chans": 5,
            "vit_backbone": {
                "_target_": "vit.sincos_positional_encoding_vit",
                "vit_backbone": {
                    "_target_": "vit.vit_small_patch16_256",
                    "global_pool": "avg",
                },
            },
        },
        "fourier_loss": {"_target_": "loss.FourierLoss", "num_multimodal_modalities": 5},
        "fourier_loss_weight": 0.01,
        "input_norm": {
            "_args_": [
                {"_target_": "normalizer.Normalizer"},
                {
                    "_target_": "torch.nn.InstanceNorm2d",
                    "affine": False,
                    "num_features": None,
                    "track_running_stats": False,
                },
            ],
            "_target_": "torch.nn.Sequential",
        },
        "layernorm_unfreeze": True,
        "loss": {"_target_": "torch.nn.MSELoss", "reduction": "none"},
        "lr_scheduler": {
            "_partial_": True,
            "_target_": "torch.optim.lr_scheduler.OneCycleLR",
            "anneal_strategy": "cos",
            "max_lr": 0.0001,
            "pct_start": 0.1,
        },
        "mask_fourier_loss": True,
        "mask_ratio": 0.25,
        "model_type": "MAE",
        "norm_pix_loss": False,
        "num_blocks_to_freeze": 0,
        "optimizer": {
            "_partial_": True,
            "_target_": "timm.optim.lion.Lion",
            "betas": [0.9, 0.95],
            "lr": 0.0001,
            "weight_decay": 0.05,
        },
        "torch_dtype": "float32",
        "transformers_version": "4.46.1",
        "trim_encoder_blocks": None,
        "use_MAE_weight_init": True,
    }
