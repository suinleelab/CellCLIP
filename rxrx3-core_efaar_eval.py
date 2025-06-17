"""
Benchmarking scripts from [1]

[1]https://github.com/recursionpharma/EFAAR_benchmarking
"""
import argparse
import os
import re

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from efaar_benchmarking.benchmarking import (
    BenchmarkConfig,
    compound_gene_benchmark,
    known_relationship_benchmark,
)
from efaar_benchmarking.constants import COMPOUND_CONCENTRATIONS
from efaar_benchmarking.efaar import pca_centerscale_on_controls
from sklearn.utils import Bunch
from tqdm import tqdm

from src import constants


def load_emb_data(file_path):
    """Load embeddings and well_id from an HDF5 file into a structured DataFrame."""

    with h5py.File(file_path, "r") as img_file:
        # Extract datasets
        try:
            embeddings = img_file["embeddings"][:]
            names = img_file["well_id"][:]  # names for imgs
        except KeyError as e:
            raise KeyError(f"Missing dataset in {file_path} file: {e}")

        # Flatten embeddings if needed

        if len(embeddings.shape) == 3:
            embeddings = np.mean(embeddings, axis=1)
            # embeddings = embeddings[:, 1, :]
        embed_dim = embeddings.shape[-1]
        embeddings_flat = embeddings.reshape(-1, embed_dim)

        # Convert well_id to strings (ensure proper decoding)
        names = np.array(
            [
                re.sub(
                    r"(?i)Plate", "", n.decode("utf-8") if isinstance(n, bytes) else str(n)
                ).strip()
                for n in names
            ]
        )
        # Create a DataFrame with feature columns
        df = pd.DataFrame(
            embeddings_flat, columns=[f"feature_{i}" for i in range(embed_dim)]
        )
        df.insert(0, "well_id", names)

        return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluating perturbation matching for Rxrx3-core."
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="Name of the precompued embeddings.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="method name",
    )
    return parser.parse_args()


def main(args):
    """Pipieline to evaluate biological relationship"""
    # Run EFAAR pipelines

    rxrx3_metadata = pd.read_csv(
        os.path.join(constants.OUT_DIR, "label_data/rxrx3-core/metadata_rxrx3_core.csv")
    )

    if not args.filepath:
        embeddings = pd.read_parquet(
            os.path.join(
                constants.DATASET_DIR, "rxrx3-core/OpenPhenom_rxrx3_core_embeddings.parquet"
            )
        )
    else:
        embeddings = load_emb_data(args.filepath)

    rxrx3_metadata["perturbation"] = rxrx3_metadata["treatment"].apply(
        lambda x: x.split("_")[0] if "_control" not in x else x
    )

    embeddings_mrged = rxrx3_metadata.merge(
        embeddings.rename(columns={"well_id": "external_well_id"})
        .groupby("external_well_id")
        .mean(),
        left_on="well_id",
        right_index=True,
    )

    feature_columns = [c for c in embeddings_mrged.columns if c.startswith("feature_")]
    metadata_columns = [c for c in embeddings_mrged.columns if not c.startswith("feature_")]

    pert_colname = "perturbation"
    experiment_colname = "experiment_name"
    control_key = "EMPTY_control"

    print("fitting aligner...")

    X = embeddings_mrged[feature_columns].astype(float).values
    embeddings_pcacs = pca_centerscale_on_controls(
        X,
        embeddings_mrged[metadata_columns],
        pert_col=pert_colname,
        batch_col=experiment_colname,
        control_key=control_key,
    )

    assert embeddings_mrged[metadata_columns].shape[0] == embeddings_pcacs.shape[0]

    new_metadata = embeddings_mrged[metadata_columns].copy().reset_index()
    new_features = pd.DataFrame(
        embeddings_pcacs, columns=[f"feature_{i}" for i in range(embeddings_pcacs.shape[1])]
    )
    aligned_embeddings = pd.concat([new_metadata, new_features], axis=1)

    assert aligned_embeddings.feature_0.isna().sum() == 0

    # remove controls from henceforth analysis
    merged = aligned_embeddings[
        ~(
            (aligned_embeddings["perturbation_type"] == "COMPOUND")
            & (aligned_embeddings[pert_colname].str.contains("_control"))
        )
    ]

    # aggregate to perturbation-level
    agg_func = {col: "mean" for col in merged.columns if col.startswith("feature_")}
    map_data = (
        merged.groupby(["perturbation_type", pert_colname, "concentration"], dropna=False)
        .agg(agg_func)
        .reset_index()
    )
    map_data = map_data[
        map_data.concentration.isin(COMPOUND_CONCENTRATIONS) | map_data.concentration.isna()
    ]
    features_cols = [col for col in map_data.columns if col.startswith("feature_")]
    metadata_cols = [col for col in map_data.columns if col not in features_cols]

    assert map_data.feature_0.isna().sum() == 0

    recall_thr_pairs = [
        # [(0.01, 0.99)],
        # [(0.02, 0.98)],
        # [(0.04, 0.96)],
        [(0.05, 0.95)],
        # [(0.06, 0.94)],
        # [(0.08, 0.92)],
        # [(0.10, 0.90)]
    ]

    print("Computing recall...")

    for i, recalls in enumerate(recall_thr_pairs):
        bmdb_metrics = known_relationship_benchmark(
            Bunch(metadata=map_data[metadata_cols], features=map_data[features_cols]),
            recall_thr_pairs=recall_thr_pairs[i],
            pert_col=pert_colname,
            log_stats=True,
        )
        print("Recall Results", bmdb_metrics[list(bmdb_metrics.columns)[::-1]])

    all_compound_results = []

    print("Computing compound_gene_benchmark...")

    for seed in tqdm(range(10)):
        compound_results = compound_gene_benchmark(
            Bunch(metadata=map_data[metadata_cols], features=map_data[features_cols]),
            check_random=False,
            config=BenchmarkConfig(random_seed=seed, quantiles=[0.5, 0.75, 0.95]),
        )
        compound_results["seed"] = seed
        all_compound_results.append(compound_results)
    compound_results = pd.concat(all_compound_results)

    median_results = (
        compound_results.groupby("concentration")
        .agg(
            avg_precision_mean=("ap_quantile_0.5", "mean"),
            avg_precision_std=("ap_quantile_0.5", "std"),
            baseline_mean=("ap_quantile_0.5_baseline", "mean"),
            baseline_std=("ap_quantile_0.5_baseline", "std"),
        )
        .reset_index()
    )

    median_results["concentration"] = median_results["concentration"].astype(str)

    upper_quantile_results = (
        compound_results.groupby("concentration")
        .agg(
            avg_precision_mean=("ap_quantile_0.75", "mean"),
            avg_precision_std=("ap_quantile_0.75", "std"),
            baseline_mean=("ap_quantile_0.75_baseline", "mean"),
            baseline_std=("ap_quantile_0.75_baseline", "std"),
        )
        .reset_index()
    )

    upper_quantile_results["concentration"] = upper_quantile_results[
        "concentration"
    ].astype(str)

    mean_results = (
        compound_results.groupby("concentration")
        .agg(
            avg_precision_mean=("average_precision", "mean"),
            avg_precision_std=("average_precision", "std"),
            baseline_mean=("average_precision_baseline", "mean"),
            baseline_std=("average_precision_baseline", "std"),
        )
        .reset_index()
    )

    mean_results["concentration"] = mean_results["concentration"].astype(str)

    mean_results_auc = (
        compound_results.groupby("concentration")
        .agg(
            auc_roc_mean=("auc_roc", "mean"),
            auc_roc_std=("auc_roc", "std"),
            baseline_auc_mean=("auc_roc_baseline", "mean"),
            baseline_auc_std=("auc_roc_baseline", "std"),
        )
        .reset_index()
    )

    mean_results_auc["concentration"] = mean_results_auc["concentration"].astype(str)

    # Define the metrics and their corresponding labels
    metrics = [
        {
            "mean_results": median_results,
            "y_mean": "avg_precision_mean",
            "y_std": "avg_precision_std",
            "baseline_mean": "baseline_mean",
            "baseline_std": "baseline_std",
            "title": "Median Average Precision vs Baseline Precision",
            "yaxis_title": "Median Avg. Precision",
        },
        {
            "mean_results": upper_quantile_results,
            "y_mean": "avg_precision_mean",
            "y_std": "avg_precision_std",
            "baseline_mean": "baseline_mean",
            "baseline_std": "baseline_std",
            "title": "Upper quantile of Average Precision vs Baseline Precision",
            "yaxis_title": "Upper quantile Avg. Precision",
        },
        {
            "mean_results": mean_results,
            "y_mean": "avg_precision_mean",
            "y_std": "avg_precision_std",
            "baseline_mean": "baseline_mean",
            "baseline_std": "baseline_std",
            "title": "Mean Average Precision vs Baseline",
            "yaxis_title": "Mean Avg. Precision",
        },
        {
            "mean_results": mean_results_auc,
            "y_mean": "auc_roc_mean",
            "y_std": "auc_roc_std",
            "baseline_mean": "baseline_auc_mean",
            "baseline_std": "baseline_auc_std",
            "title": "Mean AUC ROC vs Baseline AUC ROC",
            "yaxis_title": "Mean AUC ROC",
        },
    ]

    # Loop through the metrics and create the figures

    print("generating figures...")

    for i, metric in enumerate(metrics):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=metric["mean_results"]["concentration"],
                y=metric["mean_results"][metric["y_mean"]],
                error_y=dict(type="data", array=metric["mean_results"][metric["y_std"]]),
                mode="lines+markers",
                name=args.model_name,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=metric["mean_results"]["concentration"],
                y=metric["mean_results"][metric["baseline_mean"]],
                error_y=dict(
                    type="data", array=metric["mean_results"][metric["baseline_std"]]
                ),
                mode="lines+markers",
                name="Random Baseline",
            )
        )

        fig.update_layout(
            title=metric["title"],
            xaxis_title="Concentration",
            yaxis_title=metric["yaxis_title"],
            template="plotly_white",
        )

        fig.write_image(f"results/rxrx3-core/imgs/{args.model_name}_{i}.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
