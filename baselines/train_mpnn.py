"""Training script for MPNN++ as pretrained molecule encoder for Molphenix."""
import os
from datetime import datetime

from graphium.config._loader import (
    load_accelerator,
    load_architecture,
    load_metrics,
    load_predictor,
    load_trainer,
    load_yaml_config,
)
from graphium.data.datamodule import MultitaskFromSmilesDataModule
from graphium.utils.safe_run import SafeRun
from lightning.pytorch.utilities.model_summary import ModelSummary
from loguru import logger

from src import constants

# And for compatibility with the Paperspace environment variables we will do the following:


def main():
    """Multi-Task pretraining with largemix dataset"""
    filename = "config_gps_10M_pcqm4m.yaml"

    cfg = load_yaml_config(
        os.path.join(constants.OUT_DIR, "configs/graphium_configs", filename)
    )
    cfg, accelerator_type = load_accelerator(cfg)

    datamodule = MultitaskFromSmilesDataModule(
        **cfg["datamodule"]["args"],
    )
    datamodule.prepare_data(True)
    datamodule.setup("fit")

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dims=datamodule.in_dims,
    )
    metrics = load_metrics(cfg)
    logger.info(metrics)

    predictor = load_predictor(
        cfg,
        model_class,
        model_kwargs,
        metrics,
        datamodule.get_task_levels(),
        accelerator_type,
        datamodule.featurization,
        datamodule.task_norms,
    )

    logger.info(predictor.model)
    logger.info(ModelSummary(predictor, max_depth=4))
    date_time_suffix = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    trainer = load_trainer(cfg, accelerator_type, date_time_suffix)

    logger.info("About to set the max nodes etc.")
    predictor.set_max_nodes_edges_per_graph(datamodule, stages=["train", "val"])
    ckpt_path = "/gscratch/aims/mingyulu/cell_painting/results/mpnn/models/last-v1.ckpt"
    with SafeRun(
        name="TRAINING", raise_error=cfg["constants"]["raise_train_error"], verbose=True
    ):
        trainer.fit(
            model=predictor,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

    predictor.set_max_nodes_edges_per_graph(datamodule, stages=["test"])

    # Run the model validation
    with SafeRun(
        name="TESTING", raise_error=cfg["constants"]["raise_train_error"], verbose=True
    ):
        trainer.test(
            model=predictor,
            datamodule=datamodule,
        )


if __name__ == "__main__":
    main()
