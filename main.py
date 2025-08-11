"""
Main script for training treatment-images pair for cell painting

Mainly adopted from open_clip[1]

[1] https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/main.py
[2] https://amsword.medium.com/gradient-backpropagation-with-torch-distributed
"""

import argparse
import glob
import os
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from hflayers import Hopfield
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import wandb
from src import constants
from src.clip.clip import load_model
from src.clip.methods import (
    bi_cwcl_loss,
    clip,
    cloob,
    cwcl_loss,
    cwcl_ma_loss,
    s2l_loss,
    sigmoid_loss,
)
from src.datasets import get_cellpainting_dataset
from src.helper import (
    all_gather,
    compute_grad_norm,
    compute_param_norm,
    get_max_steps,
    get_metrics,
    print_args,
)
from src.scheduler import (
    const_lr,
    const_lr_cooldown,
    cosine_lr,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")

    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUT_DIR
    )
    parser.add_argument(
        "--split_label_dir",
        type=str,
        help="output parent directory",
        default=constants.SPLIT_LABEL_DIR,
    )
    parser.add_argument(
        "--split",
        type=int,
        help="index of dataset split file",
    )
    parser.add_argument(
        "--is_train",
        help="whether to use training index",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training",
        default=42,
    )
    parser.add_argument(
        "--wandb",
        help="whether to monitor model training with wandb",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--wandb_id",
        help="id for monitor training if laod from checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cell_clip",
        help=("Model types, e.g. cloome, cell_clip."),
    )
    parser.add_argument(
        "--embedding_name",
        type=str,
        default=None,
        help=("image embeddings, e.g. siglip@224, clip@224."),
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help=("Path to training input directory."),
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        help="Dimension of input emebddings.",
        default=768,
    )
    parser.add_argument(
        "--molecule_path",
        type=str,
        default=None,
        help=("Path to molecule (text) data."),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bray2017",
        help=("dataset name, e.g. bray2017 or jumpcp"),
    )
    parser.add_argument(
        "--unique",
        help="whether to use unique perturbation.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="clip",
        help=("Loss types, e.g. cloob, clip."),
    )
    parser.add_argument(
        "--pretrained",
        help="whether to use pretrained text encoder from CLIP.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--resume",
        help="whether to use resume from previous training.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--fine_tune_ckpt",
        help="path to ckpt for fine tuning.",
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="training epochs",
        default=50,
    )
    parser.add_argument(
        "--image_resolution_train",
        default=224,
        nargs="+",
        type=int,
        help="resolution for training set ",
    )
    parser.add_argument(
        "--image_resolution_val",
        default=224,
        nargs="+",
        type=int,
        help="resolution for validation set ",
    )
    parser.add_argument(
        "--val_subset_ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help=(
            "Training batch size. When training with Accelerate, "
            "the batch size passed to the dataloader is the batch size per GPU"
        ),
        default=32,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5.0e-4,
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            "LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown'"
            " (constant w/ cooldown). Default: cosine"
        ),
    )
    parser.add_argument("--eps", type=float, default=1.0e-8, help="Adam epsilon.")
    parser.add_argument(
        "--epochs-cooldown",
        type=int,
        default=None,
        help=(
            "When scheduler w/ cooldown used, "
            "perform cooldown from total_epochs - cooldown_epochs onwards."
        ),
    )
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=1000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--num_cycles", type=int, default=5, help="Number of cosine cycle during training."
    )

    # CLIP temperature
    parser.add_argument(
        "--init-inv-tau", type=float, default=14.3, help="Initial inverse tau."
    )
    parser.add_argument(
        "--learnable-inv-tau",
        default=False,
        action="store_true",
        help="Use a trainable logit scale for the nce loss.",
    )
    # Cloome hopfield params
    parser.add_argument(
        "--scale-hopfield", type=float, default=14.3, help="Scale for Hopfield retrieval."
    )
    parser.add_argument(
        "--learnable-scale-hopfield",
        default=False,
        action="store_true",
        help="Use a trainable logit scale for the Hopfield retrieval.",
    )
    parser.add_argument(
        "--ckpt_freq", type=int, default=1000, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--log_freq", type=int, default=20, help="How often to check model training."
    )
    parser.add_argument(
        "--eval_freq", type=int, default=500, help="How often to evaluate model training."
    )
    parser.add_argument(
        "--keep_all_ckpts",
        help="whether to keep all the checkpoints",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main(args):
    """Training scripts for contrastive learning."""
    set_seed(args.opt_seed)  # Seed for model optimization.

    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        # kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    accelerator.print("number of GPU available", torch.cuda.device_count())

    if accelerator.is_main_process:
        print_args(args)

    if args.wandb and accelerator.is_main_process:
        assert wandb is not None, "Please install wandb."
        accelerator.print("Starting wandb.")
        wandb.init(
            project=(
                f"Cell Painting {args.dataset}-{args.model_type}-{args.loss_type}-"
                f"{args.epochs}-{args.batch_size}-{args.embedding_name}-{args.lr}"
                f"unique=f{args.unique}"
            ),
            dir="/gscratch/aims/mingyulu/cell_painting/wandb",
            name=f"{args.model_type}-split_{args.split}-train_{args.is_train}",
            config=vars(args),
            id=args.wandb_id,
            resume="allow",
        )

    # Obtain training & evaluation data

    train_dataloader = get_cellpainting_dataset(
        args,
        accelerator.num_processes,
        is_train=args.is_train,
    )
    eval_dataloader = get_cellpainting_dataset(
        args,
        accelerator.num_processes,
        is_train=False,
        subset=args.val_subset_ratio,
    )
    accelerator.print(
        "Initialize training and eval loader. Number of samples,",
        f"train:{train_dataloader.num_samples}",
        f"eval:{eval_dataloader.num_samples}.",
    )

    # Initlialize model.

    model = load_model(
        args.model_type,
        args.pretrained,
        args.image_resolution_train,
        vision_width=args.input_dim,
        loss_type=args.loss_type,
    )

    model_outdir = os.path.join(
        constants.OUT_DIR,
        "results",
        args.dataset,
        "models",
        args.model_type,
        (
            f"epochs_{args.epochs}_{args.img_dir.split('/')[-1]}_"
            f"{args.loss_type}_batch_size={args.batch_size}_"
            f"lr={args.lr}_pretrained={args.pretrained}_"
            f"cycle={args.num_cycles}_warmup={args.warmup}_"
            f"unique={args.unique}_seed={args.opt_seed}"
        ),
    )

    if accelerator.is_main_process:
        os.makedirs(model_outdir, exist_ok=True)

    exclude = (
        lambda n, p: p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "bias" in n
        or "logit_scale" in n
    )

    def include(n, p):
        return not exclude(n, p)

    # include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # Adjust steps for DDP.
    steps_per_epoch = len(train_dataloader) // accelerator.num_processes
    total_steps = steps_per_epoch * args.epochs

    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "const":
        scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "const-cooldown":
        assert (
            args.epochs_cooldown is not None
        ), "Please specify the number of cooldown epochs for this lr schedule."
        cooldown_steps = steps_per_epoch * args.epochs_cooldown
        scheduler = const_lr_cooldown(
            optimizer,
            args.lr,
            args.warmup,
            total_steps,
            cooldown_steps,
            args.lr_cooldown_power,
            args.lr_cooldown_end,
        )
    elif args.lr_scheduler == "cosine-restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            warmup=args.warmup,
            num_cycles=args.num_cycles,
            num_training_steps=total_steps,
        )
    elif args.lr_scheduler == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.1,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )

    else:
        raise ValueError(
            f"Unknown scheduler, {args.lr_scheduler}. "
            f"Available options are: cosine, const, const-cooldown."
        )

    epoch = 0
    steps = 0
    total_steps_time = 0

    if args.fine_tune_ckpt:

        try:
            ckpt = torch.load(args.fine_tune_ckpt, map_location="cpu")
            accelerator.print(f"Loading pretrained checkpoint at {args.fine_tune_ckpt} ")

        except RuntimeError:
            accelerator.print(
                f"Pretrained check point at {args.fine_tune_ckpt} does not exist."
            )

    elif args.resume:
        # Check if there is an existing checkpoint to resume from. This occurs when
        # model runs are interrupted (e.g., exceeding job time limit).

        existing_steps = get_max_steps(model_outdir)

        if existing_steps is not None:
            ckpt_path = os.path.join(model_outdir, f"ckpt_steps_{existing_steps:0>8}.pt")

            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                steps = ckpt["steps"]
                epoch = ckpt["epoch"]
                total_steps_time = ckpt["total_steps_time"]
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])

                accelerator.print(
                    f"Resuming checkpoint at epoch {epoch}; steps {steps} from {ckpt_path}"
                )

            except RuntimeError:
                accelerator.print(f"Check point at {ckpt_path} does not exist.")

    if args.loss_type == "cloob":
        hopfield_layer = Hopfield(
            input_size=512,
            scaling=args.scale_hopfield,
            normalize_hopfield_space=False,
            normalize_hopfield_space_affine=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,
            normalize_state_pattern=False,
            normalize_state_pattern_affine=False,
            normalize_stored_pattern=False,
            normalize_stored_pattern_affine=False,
            state_pattern_as_static=True,
            pattern_projection_as_static=True,
            stored_pattern_as_static=True,
            disable_out_projection=True,
            num_heads=1,
            dropout=False,
        )
        model, hopfield_layer = accelerator.prepare(model, hopfield_layer)
    else:
        model = accelerator.prepare(model)

    loss_fct_img = nn.CrossEntropyLoss()
    loss_fct_tx = nn.CrossEntropyLoss()

    optimizer, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        optimizer, scheduler, train_dataloader, eval_dataloader
    )

    progress_bar = tqdm(
        range(total_steps),
        initial=steps,
        desc="Steps",
        disable=not accelerator.is_main_process,
    )
    steps_start_time = time.time()
    while steps < total_steps:
        for _, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            model.train()

            (images, extra_tokens), treatments = batch
            m = model.module if accelerator.use_distributed else model

            if args.model_type == "cell_clip":
                images = m.encode_mil(images)

            if args.model_type == "clip_channelvit":
                img_features, text_features, logit_scale = model(
                    images, extra_tokens, treatments
                )
            else:
                if args.loss_type in ["s2l", "sigclip"]:
                    img_features, text_features, logit_scale, bias = model(
                        images, treatments
                    )
                else:
                    img_features, text_features, logit_scale = model(images, treatments)

            if accelerator.use_distributed:
                # Gather all image and text features from all GPUs
                all_image_features = all_gather(img_features)
                all_text_features = all_gather(text_features)
                all_images = all_gather(images)
            else:
                all_image_features = img_features
                all_text_features = text_features
                all_images = images

            if args.loss_type == "clip":
                loss = clip(
                    all_image_features,
                    all_text_features,
                    logit_scale,
                    loss_fct_img,
                    loss_fct_tx,
                )
            elif args.loss_type == "cwcl":
                loss = cwcl_loss(
                    all_images,
                    all_image_features,
                    all_text_features,
                    logit_scale,
                    loss_fct_tx,
                )
            elif args.loss_type == "bi_cwcl":
                loss = bi_cwcl_loss(
                    all_images,
                    all_image_features,
                    all_text_features,
                    logit_scale,
                )
            elif args.loss_type == "cwcl_ma":
                loss = cwcl_ma_loss(
                    all_images,
                    all_image_features,
                    all_text_features,
                    logit_scale,
                    loss_fct_tx,
                )
            elif args.loss_type == "cloob":
                loss = cloob(
                    all_image_features,
                    all_text_features,
                    logit_scale.exp(),
                    hopfield_layer,
                )
            elif args.loss_type == "sigclip":
                loss = sigmoid_loss(
                    all_image_features,
                    all_text_features,
                    logit_scale,
                    bias,
                )
            elif args.loss_type == "s2l":
                loss = s2l_loss(
                    all_image_features,
                    all_text_features,
                    logit_scale,
                    bias,
                )
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 20.0)

            optimizer.step()
            scheduler.step()

            # m.logit_inv_tau.data = torch.clamp(m.logit_inv_tau.data, 0, 4.6052)

            with torch.no_grad():
                m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

            if accelerator.sync_gradients:
                steps += 1
                progress_bar.update(1)

                if steps % steps_per_epoch == 0:
                    epoch += 1

                if steps % args.log_freq == 0 and accelerator.is_main_process:

                    steps_time = time.time() - steps_start_time
                    total_steps_time += steps_time

                    # Check gradient norm and parameter norm.
                    grad_norm = compute_grad_norm(accelerator, model)
                    param_norm = compute_param_norm(accelerator, model)

                    info = f"Step[{steps}/{total_steps}]"
                    info += f", steps_time: {steps_time:.3f}"
                    info += f", loss: {loss.detach().cpu().item():.5f}"
                    info += f", temperature: {m.logit_scale.data.exp():.6f}"
                    info += f", gradient norms: {grad_norm:.5f}"
                    info += f", parameters norms: {param_norm:.5f}"
                    info += f", lr: { scheduler.get_last_lr()[0]:.6f}"
                    accelerator.print(info, flush=True)

                    if args.wandb:
                        wandb.log(
                            {
                                "step": steps,
                                "loss": loss.detach().cpu().item(),
                                "temperature": m.logit_scale.data.exp(),
                                "steps_time": steps_time,
                                "gradient norm": grad_norm,
                                "parameter norm": param_norm,
                                "lr": scheduler.get_last_lr()[0],
                            }
                        )

                    steps_start_time = time.time()

                if (
                    steps % args.ckpt_freq == 0 or steps == total_steps
                ) and accelerator.is_main_process:
                    if not args.keep_all_ckpts:
                        pattern = os.path.join(model_outdir, "ckpt_steps_*.pt")
                        for filename in glob.glob(pattern):
                            os.remove(filename)

                    torch.save(
                        {
                            "model": accelerator.get_state_dict(model),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                            "total_steps_time": total_steps_time,
                        },
                        os.path.join(model_outdir, f"ckpt_steps_{steps:0>8}.pt"),
                    )
                    accelerator.print(f"Checkpoint saved at step: {steps} epoch:{epoch}")
                    steps_start_time = time.time()

                if steps % args.eval_freq == 0 or steps == 1 or steps == total_steps:

                    model.eval()
                    accelerator.print("Evaluation with retrieval task..")

                    with torch.no_grad():
                        all_eval_image_features = []
                        all_eval_text_features = []
                        all_images = []

                        for index, batch in enumerate(eval_dataloader):

                            (images, extra_tokens), treatments = batch
                            with accelerator.autocast():
                                if args.model_type == "cell_clip":
                                    images = m.encode_mil(images)

                                if args.model_type in [
                                    "clip_channelvit",
                                ]:
                                    img_features, text_features, logit_scale = model(
                                        images, extra_tokens, treatments
                                    )
                                else:
                                    if args.loss_type in ["s2l", "sigclip"]:
                                        (
                                            img_features,
                                            text_features,
                                            logit_scale,
                                            bias,
                                        ) = model(images, treatments)
                                    else:
                                        img_features, text_features, logit_scale = model(
                                            images, treatments
                                        )

                            all_eval_image_features.append(img_features)
                            all_eval_text_features.append(text_features)
                            all_images.append(images)

                        all_eval_image_features = torch.cat(all_eval_image_features)
                        all_eval_text_features = torch.cat(all_eval_text_features)
                        all_images = torch.cat(all_images)

                        if accelerator.use_distributed:
                            all_eval_image_features = accelerator.gather_for_metrics(
                                all_eval_image_features
                            )
                            all_eval_text_features = accelerator.gather_for_metrics(
                                all_eval_text_features
                            )
                            all_images = accelerator.gather_for_metrics(all_images)

                        if args.loss_type == "clip":
                            val_loss = clip(
                                all_eval_image_features,
                                all_eval_text_features,
                                logit_scale,
                                loss_fct_img,
                                loss_fct_tx,
                            )
                        elif args.loss_type == "cwcl":
                            val_loss = cwcl_loss(
                                all_images,
                                all_eval_image_features,
                                all_eval_text_features,
                                logit_scale,
                                loss_fct_tx,
                            )
                        elif args.loss_type == "bi_cwcl":
                            val_loss = bi_cwcl_loss(
                                all_images,
                                all_eval_image_features,
                                all_eval_text_features,
                                logit_scale,
                            )
                        elif args.loss_type == "cwcl_ma":
                            val_loss = cwcl_ma_loss(
                                all_images,
                                all_eval_image_features,
                                all_eval_text_features,
                                logit_scale,
                                loss_fct_tx,
                            )
                        elif args.loss_type == "sigclip":
                            val_loss = sigmoid_loss(
                                all_eval_image_features,
                                all_eval_text_features,
                                logit_scale,
                                bias,
                            )
                        elif args.loss_type == "s2l":
                            val_loss = s2l_loss(
                                all_eval_image_features,
                                all_eval_text_features,
                                logit_scale,
                                bias,
                            )
                        elif args.loss_type == "cloob":
                            val_loss = cloob(
                                all_eval_image_features,
                                all_eval_text_features,
                                logit_scale.exp(),
                                hopfield_layer,
                            )

                        if accelerator.is_main_process:
                            # Evaluation only in the main process
                            metrics = get_metrics(
                                all_eval_image_features, all_eval_text_features
                            )
                            modalities = {"image_to_text", "text_to_image"}
                            for name in modalities:
                                info = (
                                    f"Evaluation: {name} retrieval at epoch:{epoch}, "
                                    f"Steps: {steps}/{total_steps}, "
                                    f"Val loss: {val_loss:.4f}, "
                                    f"mean_rank: {metrics[f'{name}_mean_rank']:.1f}, "
                                    f"median_rank: {metrics[f'{name}_median_rank']:.1f}, "
                                    f"R@1: {metrics[f'{name}_R@1']:.4f}, "
                                    f"R@5: {metrics[f'{name}_R@5']:.4f}, "
                                    f"R@10: {metrics[f'{name}_R@10']:.4f}."
                                )
                                accelerator.print(info, flush=True)

                                if args.wandb:
                                    wandb.log(
                                        {
                                            "step": steps,
                                            "val loss": val_loss,
                                            f"{name}_mean_rank": metrics[
                                                f"{name}_mean_rank"
                                            ],
                                            f"{name}_median_rank": metrics[
                                                f"{name}_median_rank"
                                            ],
                                            f"{name}_R@{1}": metrics[f"{name}_R@1"],
                                            f"{name}_R@{5}": metrics[f"{name}_R@5"],
                                            f"{name}_R@{10}": metrics[f"{name}_R@10"],
                                        }
                                    )
                    steps_start_time = time.time()

                if steps == total_steps:
                    break


if __name__ == "__main__":
    args = parse_args()
    main(args)
