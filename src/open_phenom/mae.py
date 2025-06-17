from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from transformers.utils import cached_file
from configs.model_config import ModelConfig
from src.open_phenom.loss import FourierLoss
from src.open_phenom.mae_modules import CAMAEDecoder, MAEDecoder, MAEEncoder
from src.open_phenom.mae_utils import flatten_images
from src.open_phenom.normalizer import Normalizer
from src.open_phenom.vit import (
    generate_2d_sincos_pos_embeddings,
    sincos_positional_encoding_vit,
    vit_small_patch16_256,
)

TensorDict = Dict[str, torch.Tensor]

class MAEModel(nn.Module):
    # Loss metrics
    TOTAL_LOSS = "loss"
    RECON_LOSS = "reconstruction_loss"
    FOURIER_LOSS = "fourier_loss"

    def __init__(   
            self, 
            # decoder attributes
            mask_ratio = 0.0,
            max_in_chans = 11,
            channel_agnostic=True,
            global_pool="avg",
            # encoder attributes
            depth = 8,
            embed_dim = 512,
            mlp_ratio = 4,
            num_heads = 16,
            num_modalities=6,
            qkv_bias=True,
            tokens_per_modality=256,
            # MAE loss
            fourier_loss_weight = 0.0,
            mask_fourier_loss=True,
            return_channelwise_embeddings=False,
            use_MAE_weight_init=False,
        ):
        super().__init__()

        self.mask_ratio = mask_ratio

        # Could use Hydra to instantiate instead
        self.encoder = MAEEncoder(
            vit_backbone=sincos_positional_encoding_vit(
                vit_backbone=vit_small_patch16_256(global_pool=global_pool)
            ),
            max_in_chans=max_in_chans,  # upper limit on number of input channels
            channel_agnostic=channel_agnostic,
        )
        self.decoder = CAMAEDecoder(
            depth=depth,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            num_heads=num_heads,
            num_modalities=num_modalities,
            qkv_bias=qkv_bias,
            tokens_per_modality=tokens_per_modality,
        )
        self.input_norm = torch.nn.Sequential(
            Normalizer(),
            nn.InstanceNorm2d(None, affine=False, track_running_stats=False),
        )

        self.fourier_loss_weight = fourier_loss_weight
        self.mask_fourier_loss = mask_fourier_loss
        self.return_channelwise_embeddings = return_channelwise_embeddings
        self.tokens_per_channel = (
            256  # hardcode the number of tokens per channel since we are patch16 crop 256
        )

        # loss stuff
        self.loss = torch.nn.MSELoss(reduction="none")

        self.fourier_loss = FourierLoss(num_multimodal_modalities=num_modalities)
        if self.fourier_loss_weight > 0 and self.fourier_loss is None:
            raise ValueError(
                "FourierLoss weight is activated but no fourier_loss was defined in constructor"
            )
        elif self.fourier_loss_weight >= 1:
            raise ValueError(
                "FourierLoss weight is too large to do mixing factor, weight should be < 1"
            )

        self.patch_size = int(self.encoder.vit_backbone.patch_embed.patch_size[0])

        # projection layer between the encoder and decoder
        self.encoder_decoder_proj = nn.Linear(
            self.encoder.embed_dim, self.decoder.embed_dim, bias=True
        )

        self.decoder_pred = nn.Linear(
            self.decoder.embed_dim,
            self.patch_size ** 2 * (1 if self.encoder.channel_agnostic else self.in_chans),
            bias=True,
        )  # linear layer from decoder embedding to input dims

        # overwrite decoder pos embeddings based on encoder params
        self.decoder.pos_embeddings = generate_2d_sincos_pos_embeddings(  # type: ignore[assignment]
            self.decoder.embed_dim,
            length=self.encoder.vit_backbone.patch_embed.grid_size[0],
            use_class_token=self.encoder.vit_backbone.cls_token is not None,
            num_modality=(
                self.decoder.num_modalities if self.encoder.channel_agnostic else 1
            ),
        )

        if use_MAE_weight_init:
            w = self.encoder.vit_backbone.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            torch.nn.init.normal_(self.encoder.vit_backbone.cls_token, std=0.02)
            torch.nn.init.normal_(self.decoder.mask_token, std=0.02)

            self.apply(self._MAE_init_weights)

    def setup(self, stage: str) -> None:
        super().setup(stage)

    def _MAE_init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def decode_to_reconstruction(
        encoder_latent: torch.Tensor,
        ind_restore: torch.Tensor,
        proj: torch.nn.Module,
        decoder: torch.nn.Module,
        pred: torch.nn.Module,
    ) -> torch.Tensor:
        """Feed forward the encoder latent through the decoders necessary projections and transformations."""
        decoder_latent_projection = proj(
            encoder_latent
        )  # projection from encoder.embed_dim to decoder.embed_dim
        decoder_tokens = decoder.forward_masked(
            decoder_latent_projection, ind_restore
        )  # decoder.embed_dim output
        predicted_reconstruction = pred(decoder_tokens)  # linear projection to input dim
        return predicted_reconstruction[:, 1:, :]  # drop class token

    def forward(
        self, imgs: torch.Tensor, constant_noise: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = self.input_norm(imgs)
        latent, mask, ind_restore = self.encoder.forward_masked(
            imgs, self.mask_ratio, constant_noise
        )  # encoder blocks
        reconstruction = self.decode_to_reconstruction(
            latent,
            ind_restore,
            self.encoder_decoder_proj,
            self.decoder,
            self.decoder_pred,
        )
        return latent, reconstruction, mask

    def compute_MAE_loss(
        self,
        reconstruction: torch.Tensor,
        img: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes final loss and returns specific values of component losses for metric reporting."""
        loss_dict = {}
        img = self.input_norm(img)
        target_flattened = flatten_images(
            img,
            patch_size=self.patch_size,
            channel_agnostic=self.encoder.channel_agnostic,
        )

        loss: torch.Tensor = self.loss(
            reconstruction, target_flattened
        )  # should be with MSE or MAE (L1) with reduction='none'
        loss = loss.mean(dim=-1)  # average over embedding dim -> mean loss per patch (N,L)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches only
        loss_dict[self.RECON_LOSS] = loss.item()
        # compute fourier loss
        if self.fourier_loss_weight > 0:
            floss: torch.Tensor = self.fourier_loss(reconstruction, target_flattened)
            if not self.mask_fourier_loss:
                floss = floss.mean()
            else:
                floss = floss.mean(dim=-1)
                floss = (floss * mask).sum() / mask.sum()

            loss_dict[self.FOURIER_LOSS] = floss.item()

        # here we use a mixing factor to keep the loss magnitude appropriate with fourier
        if self.fourier_loss_weight > 0:
            loss = (1 - self.fourier_loss_weight) * loss + (
                self.fourier_loss_weight * floss
            )
        return loss, loss_dict
    
    def predict(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = self.input_norm(imgs)
        X = self.encoder.vit_backbone.forward_features(
            imgs
        )  # 3d tensor N x num_tokens x dim
        if self.return_channelwise_embeddings:
            N, _, d = X.shape
            num_channels = imgs.shape[1]
            X_reshaped = X[:, 1:, :].view(N, num_channels, self.tokens_per_channel, d)
            pooled_segments = X_reshaped.mean(
                dim=2
            )  # Resulting shape: (N, num_channels, d)
            latent = pooled_segments.view(N, num_channels * d).contiguous()
        else:
            latent = X[:, 1:, :].mean(dim=1)  # 1 + 256 * C tokens
        return latent
    
def load_mae():
    """Load MAE model."""
    configs = ModelConfig.mae_config
    encoder_configs = configs["encoder"]
    decoder_configs = configs["decoder"]

    model = MAEModel(
        # encoder attributes
        mask_ratio = configs["mask_ratio"],
        max_in_chans =  encoder_configs["max_in_chans"],
        channel_agnostic= encoder_configs["channel_agnostic"],
        global_pool= encoder_configs["vit_backbone"]["vit_backbone"]["global_pool"],
        # decoder attributes
        depth = decoder_configs["depth"],
        embed_dim = decoder_configs["embed_dim"],
        mlp_ratio = decoder_configs["mlp_ratio"],
        num_heads = decoder_configs["num_heads"],
        num_modalities= decoder_configs["num_modalities"],
        qkv_bias= decoder_configs["qkv_bias"],
        tokens_per_modality= decoder_configs["tokens_per_modality"],
        # MAE loss
        fourier_loss_weight = configs["fourier_loss_weight"],
        mask_fourier_loss= configs["mask_fourier_loss"],
        return_channelwise_embeddings=False,
        use_MAE_weight_init=configs["use_MAE_weight_init"]
    )
    return model