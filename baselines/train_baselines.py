"""Training scrips and functions for weakly supervised learning for cell painting"""
import argparse
import glob
import os
import time

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from timm.optim.lion import Lion
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

import wandb
from src import constants
from src.datasets import get_cellpainting_dataset
from src.helper import compute_param_norm, get_max_steps, print_args
from src.open_phenom.mae import load_mae
from src.open_phenom.vit_encoder import ViTClassifier
from src.scheduler import (
    const_lr,
    const_lr_cooldown,
    cosine_lr,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")
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
        required=True,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUT_DIR
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help=("Path to training input directory."),
        required=True,
    )
    parser.add_argument(
        "--molecule_path",
        type=str,
        default=None,
        required=True,
        help=("Path to molecule (text) data."),
    )
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training",
        default=42,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bray2017",
        help=("dataset name, e.g. bray2017 or jumpcp"),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="densenet",
        help="Model type for weakly supervised or self-supervised.",
    )
    parser.add_argument(
        "--model_info",
        type=str,
        default="vit_base_patch16_384",
        help="Type of ViT, e.g. vit_base_patch16_384",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="training epochs",
        default=50,
    )
    parser.add_argument(
        "--image_resolution_train",
        default=256,
        nargs="+",
        type=int,
        help="resolution for training set ",
    )
    parser.add_argument(
        "--image_resolution_val",
        default=256,
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
        default=0.95,
    )
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")

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
        "--warmup", type=int, default=1000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--ckpt_freq", type=int, default=1000, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--eval_freq", type=int, default=3000, help="How often to evaluate model training."
    )
    parser.add_argument(
        "--log_freq", type=int, default=20, help="How often to check model training."
    )
    parser.add_argument(
        "--resume",
        help="whether to use resume from previous training.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--keep_all_ckpts",
        help="whether to keep all the checkpoints",
        action="store_true",
        default=False,
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
        "--unique",
        help="whether to use unique perturbation.",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main(args):
    """Train classifiers (WSL) based on ViT or DenseNet"""
    set_seed(args.opt_seed)  # Seed for model optimization.
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    accelerator.print("number of GPU available", torch.cuda.device_count())

    if args.wandb and accelerator.is_main_process:
        assert wandb is not None, "Please install wandb."
        accelerator.print("Starting wandb.")
        wandb.init(
            project=(
                f"Cell Painting WSL {args.dataset}-{args.model_type}-{args.model_info}-"
                f"{args.epochs}-{args.batch_size}-{args.lr}"
            ),
            dir="/gscratch/aims/mingyulu/cell_painting/wandb",
            name=f"{args.model_type}-split_{args.split}",
            config=vars(args),
            id=args.wandb_id,
            resume="allow",
        )
    if accelerator.is_main_process:
        print_args(args)

    train_dataloader = get_cellpainting_dataset(
        args,
        accelerator.num_processes,
        is_train=True,
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
    label_map = train_dataloader.label_map

    if args.model_type == "vit":
        model = ViTClassifier(args.model_info, len(label_map))
    elif args.model_type == "mae":
        model = load_mae()

    model_outdir = os.path.join(
        constants.OUT_DIR,
        "results",
        args.dataset,
        "models",
        f"{args.model_type}_{args.model_info}",
        f"epochs_{args.epochs}_batch_size={args.batch_size}_lr={args.lr}",
    )

    if accelerator.is_main_process:
        os.makedirs(model_outdir, exist_ok=True)

    epoch = 0
    steps = 0
    total_steps_time = 0

    if args.model_type == "mae":
        optimizer = Lion(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=(args.beta1, args.beta2),
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

    steps_per_epoch = len(train_dataloader) // accelerator.num_processes
    total_steps = steps_per_epoch * args.epochs

    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "cosine-anneling":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
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
            max_lr=args.lr,
            total_steps=total_steps * accelerator.num_processes,
            pct_start=0.3,
            anneal_strategy="cos",
        )
    else:
        raise ValueError(
            f"Unknown scheduler, {args.lr_scheduler}. "
            f"Available options are: cosine, const, const-cooldown."
        )

    optimizer, scheduler, train_dataloader, eval_dataloader, model = accelerator.prepare(
        optimizer, scheduler, train_dataloader, eval_dataloader, model
    )

    if args.resume:
        existing_steps = get_max_steps(model_outdir)
        # Check if checkpoint exists.

        if existing_steps is not None:
            ckpt_path = os.path.join(model_outdir, f"ckpt_steps_{existing_steps:0>8}.pt")

            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                steps = ckpt["steps"]
                epoch = ckpt["epoch"]
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                total_steps_time = ckpt["total_steps_time"]
                model.load_state_dict(ckpt["model"])

                accelerator.print(
                    f"Resuming checkpoint at epoch {epoch}; steps {steps} from {ckpt_path}"
                )
            except RuntimeError:
                accelerator.print(f"Check point at {ckpt_path} does not exist.")

    progress_bar = tqdm(
        range(total_steps),
        initial=steps,
        desc="Steps",
        disable=not accelerator.is_main_process,
    )
    steps_start_time = time.time()

    if args.model_type == "vit":
        loss_fn = nn.CrossEntropyLoss()

    model.train()

    while steps < total_steps:
        for _, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            (images, _), treatments = batch
            images = images.to(accelerator.device)

            if args.model_type == "vit":
                y_true = torch.tensor(
                    [label_map.get(t, 0) for t in treatments], dtype=torch.long
                ).to(accelerator.device)
                y_pred_logits = model(images)

                loss = loss_fn(y_pred_logits, y_true)

            elif args.model_type == "mae":
                _, reconstruction, masking = model(images)
                m = model.module if accelerator.use_distributed else model

                loss, _ = m.compute_MAE_loss(reconstruction, images, masking)

            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()

            if accelerator.sync_gradients:
                steps += 1
                progress_bar.update(1)

                if steps % steps_per_epoch == 0:
                    epoch += 1

                if steps % args.log_freq == 0 and accelerator.is_main_process:

                    steps_time = time.time() - steps_start_time
                    total_steps_time += steps_time

                    # Check gradient norm and parameter norm.
                    param_norm = compute_param_norm(accelerator, model)
                    if args.model_type == "vit":
                        pred_class = y_pred_logits.argmax(1)
                        acc = (pred_class == y_true).float().mean().item()
                        metric = acc
                    elif args.model_type == "mae":
                        metric = loss

                    info = f"Step[{steps}/{total_steps}]"
                    info += f", steps_time: {steps_time:.3f}"
                    info += f", loss: {loss.detach().cpu().item():.5f}"
                    info += f", acc: {metric:.5f}"
                    info += f", parameters norms: {param_norm:.5f}"
                    info += f", lr: { scheduler.get_last_lr()[0]:.6f}"
                    accelerator.print(info, flush=True)

                    if args.wandb:
                        wandb.log(
                            {
                                "step": steps,
                                "loss": loss.detach().cpu().item(),
                                "metric": metric,
                                "steps_time": steps_time,
                                "parameter norm": param_norm,
                                "lr": scheduler.get_last_lr()[0],
                            }
                        )

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
                    accelerator.print("Evaluation..")

                    with torch.no_grad():
                        all_preds = []
                        all_labels = []
                        total_val_loss = 0

                        for _, batch in enumerate(eval_dataloader):
                            (val_images, _), treatments = batch
                            val_images = val_images.to(accelerator.device)

                            if args.model_type == "vit":
                                y_true = torch.tensor(
                                    [label_map.get(t, 0) for t in treatments],
                                    dtype=torch.long,
                                ).to(accelerator.device)
                                y_pred_logits = model(val_images)

                                loss = loss_fn(y_pred_logits, y_true)
                            elif args.model_type == "mae":
                                _, reconstruction, masking = model(val_images)
                                m = model.module if accelerator.use_distributed else model

                                loss, _ = m.compute_MAE_loss(
                                    reconstruction, val_images, masking
                                )

                            total_val_loss += loss.item()

                            if args.model_type == "vit":
                                all_preds.append(y_pred_logits.argmax(dim=1))
                                all_labels.append(y_true)

                        if args.model_type == "vit":
                            all_preds = torch.cat(all_preds)
                            all_labels = torch.cat(all_labels)

                            val_acc = (all_preds == all_labels).float().mean().item()
                            info = (
                                f"Evaluation at epoch:{epoch}, "
                                f"Steps: {steps}/{total_steps}, "
                                f"Val loss: {total_val_loss/len(eval_dataloader):.4f}, "
                                f"Acc: {val_acc}"
                            )
                            metric = val_acc
                        elif args.model_type == "mae":
                            info = (
                                f"Evaluation at epoch:{epoch}, "
                                f"Steps: {steps}/{total_steps}, "
                                f"Val loss: {total_val_loss/len(eval_dataloader):.4f}, "
                            )
                            metric = total_val_loss / len(eval_dataloader)

                        param_norm = compute_param_norm(accelerator, model)
                        accelerator.print(info, flush=True)

                        if args.wandb and accelerator.is_main_process:
                            wandb.log(
                                {
                                    "step": steps,
                                    "loss": loss.detach().cpu().item(),
                                    "val_metric": metric,
                                    "parameter norm": param_norm,
                                    "lr": scheduler.get_last_lr()[0],
                                }
                            )

                    model.train()

                steps_start_time = time.time()

            if steps == total_steps:
                break


if __name__ == "__main__":
    args = parse_args()
    main(args)
