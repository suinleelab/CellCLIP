"""
Evaluation pipelines for cross pertrubation [1]

[1] https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

from src import constants
from src.benchmark import utils


def load_emb_data(plate, pca_kernel=None, feature_type="profile"):
    """Load all data from a single experiment into a single dataframe"""
    path = os.path.join(
        constants.DATASET_DIR, "jumpcp/output_emb/wsl", f"{plate}.h5"
    )
    experiment_df = (
        pd.read_csv(
            "/gscratch/aims/mingyulu/cell_painting/label_data/jumpcp/experiment-metadata.tsv",
            sep="\t",
        )
        .query("Batch=='2020_11_04_CPJUMP1'")
        .query("Density==100")
        .query('Antibiotics=="absent"')
    )

    perturbation = experiment_df[experiment_df.Assay_Plate_Barcode == plate].Perturbation.values[0]

    with h5py.File(path, "r") as img_file:
        # Extract the datasets
        embeddings = img_file["embeddings"][:]
        embed_dim = embeddings.shape[-1]
        embeddings_flat = embeddings.reshape(-1, embed_dim)

        well_position = img_file["well_position"][:]
        broad_sample = [sample.decode("utf-8") for sample in img_file["broad_sample"][:]]

        perturbation = "crispr"
        if perturbation == "compound":
            meta_data = pd.read_csv(
                (
                    "/gscratch/aims/mingyulu/cell_painting/"
                    "label_data/jumpcp/plate_map/JUMP-Target-1_compound_platemap.txt"
                ),
                sep="\t",
            )
        elif perturbation == "crispr":
            meta_data = pd.read_csv(
                (
                    "/gscratch/aims/mingyulu/cell_painting/"
                    "label_data/jumpcp/plate_map/JUMP-Target-1_crispr_metadata.tsv"
                ),
                sep="\t",
            )
        elif perturbation == "orf":
            meta_data = pd.read_csv(
                (
                    "/gscratch/aims/mingyulu/cell_painting/"
                    "label_data/jumpcp/plate_map/JUMP-Target-1_orf_metadata.tsv"
                ),
                sep="\t",
            )

        if pca_kernel is not None and feature_type == "emb":
            pcaed_values = pca_kernel.transform(embeddings_flat)

            if perturbation == "compound":
                control_idx =  np.where(np.array(broad_sample) == "NaN")[0]
            else:
                control_idx = meta_data[meta_data.control_type == "negcon"].index

            scaler = StandardScaler()
            scaler.fit(pcaed_values)
            # control = pcaed_values[control_idx]
            # scaler.fit(control)

            embeddings_flat = scaler.transform(pcaed_values)

        # Create a DataFrame and store embeddings as a single column with numpy arrays
        df = pd.DataFrame(
            {
                "embeddings": [np.array(embedding) for embedding in embeddings_flat],
                "Metadata_Well": well_position.astype(str),
                # "Metadata_broad_sample": broad_sample.astype(str)
            }
        )
        df["Metadata_Plate"] = plate

    return df


def process_modality(
    modality_experiments_df,
    modality_perturbation,
    modality_2,
    feature_type,
    pca_kernel=None,
):
    """Load and preprocess plate data"""

    # TODO batch correction based on plate

    modality_df = pd.DataFrame()
    modality_emb_df = pd.DataFrame()

    for plate in modality_experiments_df.Assay_Plate_Barcode.unique():
        data_df = utils.load_data(
            plate, "normalized_feature_select_negcon_batch.csv.gz"
        ).assign(Metadata_modality=modality_perturbation)

        if modality_2:
            data_df = data_df.assign(Metadata_matching_target=lambda x: x.Metadata_gene)

        modality_df = utils.concat_profiles(modality_df, data_df)
        emb_data = load_emb_data(plate, pca_kernel, feature_type)
        modality_emb_df = utils.concat_profiles(modality_emb_df, emb_data)
        modality_emb_df = pd.merge(
            modality_df,
            modality_emb_df,
            left_on=["Metadata_Plate", "Metadata_Well"],
            right_on=["Metadata_Plate", "Metadata_Well"],
            how="left",
        )
        testing_label = pd.read_csv("/gscratch/aims/datasets/cellpainting/jumpcp/mol/jumpcp_testing_label2.csv")
        testing_label["Metadata_Plate"] = testing_label["SAMPLE_KEY"].str.split("-").str[0]
        testing_label["Metadata_Well"] = testing_label["SAMPLE_KEY"].str.split("-").str[1]
        testing_label = testing_label[testing_label["Metadata_Plate"] == plate]

        modality_emb_df = modality_emb_df[modality_emb_df.Metadata_Well.isin(testing_label.Metadata_Well.tolist())]
        modality_df = modality_df[modality_df.Metadata_Well.isin(testing_label.Metadata_Well.tolist())]

    if modality_perturbation == "compound":
        # Set Metadata_broad_sample value to "DMSO" for DMSO wells only for compounds.
        modality_df["Metadata_broad_sample"].fillna("DMSO", inplace=True)
        modality_emb_df["Metadata_broad_sample"].fillna("DMSO", inplace=True)

    # Remove empty wells
    modality_df = utils.remove_empty_wells(modality_df)
    modality_emb_df = utils.remove_empty_wells(modality_emb_df)

    return modality_emb_df if feature_type == "emb" else modality_df


def train_control_pca(experiment_df, kernel, n_components=500, feature_type="profile"):
    """Functions that train a PCA with controls sampels for batch correction."""

    perturbations = {
        plate: experiment_df[
            experiment_df.Assay_Plate_Barcode == plate
        ].Perturbation.values[0]
        for plate in experiment_df.Assay_Plate_Barcode
    }

    controls = pd.DataFrame()
    controls_emb_df = pd.DataFrame()

    for plate in perturbations.keys():
        data_df = utils.load_data(
            plate, "normalized_feature_select_negcon_batch.csv.gz"
        ).assign(Metadata_modality=perturbations[plate])

        controls = utils.concat_profiles(controls, data_df)
        if feature_type == "emb":
            emb_data = load_emb_data(plate)
            controls_emb_df = utils.concat_profiles(controls_emb_df, emb_data)
            controls_emb_df = pd.merge(
                controls,
                controls_emb_df,
                left_on=["Metadata_Plate", "Metadata_Well"],
                right_on=["Metadata_Plate", "Metadata_Well"],
                how="left",
            )

    # Load controls in all plates.
    if feature_type == "emb":
        controls = controls_emb_df

    controls = controls[controls["Metadata_control_type"] == "negcon"]

    feature_df = utils.get_featuredata(controls)
    feature_values = (
        feature_df.values
        if feature_type != "emb"
        else np.stack(controls["embeddings"].values)
    )

    # Fit a PCA kernel on all the control images

    kernel_pca = KernelPCA(n_components=n_components, kernel=kernel)
    kernel_pca.fit(feature_values)

    return kernel_pca


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluating perturbation matching for JUMP-CP1."
    )
    parser.add_argument("--kernel", type=str, help="Kernel of PCA", default="linear")
    parser.add_argument(
        "--pca_n_components", type=int, help="Reduced dimension of kernelPCA", default=500
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        help="feature types of cell painting images.",
        default="profile",
    )
    parser.add_argument(
        "--emb_type",
        type=str,
        help="model types of cell painting embeddings.",
        default="openphenom",
    )
    parser.add_argument(
        "--batch_correction",
        default=False,
        action="store_true",
        help="Whether to perform batch correction.",
    )
    return parser.parse_args()


def main(args):
    """Main function to run experiments on all modalities and cell types."""

    replicability_map_df = pd.DataFrame()
    replicability_fr_df = pd.DataFrame()
    matching_map_df = pd.DataFrame()
    matching_fr_df = pd.DataFrame()
    gene_compound_matching_map_df = pd.DataFrame()
    gene_compound_matching_fr_df = pd.DataFrame()

    replicate_feature = "Metadata_broad_sample"
    batch_size = 100000
    null_size = 100000

    experiment_df = (
        pd.read_csv(
            "path_to_cpjump1_metadata/experiment-metadata.tsv",
            sep="\t",
        )
        .query("Batch=='2020_11_04_CPJUMP1'")
        .query("Density==100")
        .query('Antibiotics=="absent"')
    )
    experiment_df.drop(
        experiment_df[
            (experiment_df.Perturbation == "compound") & (experiment_df.Cell_line == "Cas9")
        ].index,
        inplace=True,
    )
    # experiment_df.drop(
    #     experiment_df[
    #         (experiment_df.Perturbation != "compound")
    #     ].index,
    #     inplace=True,
    # )
    target1_metadata = pd.read_csv(
        "path_to_jumpcp_plate_metadata/plate_map/JUMP-Target-1_compound_metadata_additional_annotations.tsv",
        sep="\t",
        usecols=["broad_sample", "target_list"],
    ).rename(
        columns={
            "broad_sample": "Metadata_broad_sample",
            "target_list": "Metadata_target_list",
        }
    )
    if args.batch_correction:
        control_pca = train_control_pca(
            experiment_df, args.kernel, args.pca_n_components, args.feature_type
        )
    else:
        control_pca = None

    # Iterate through cell types, e.g. A524, USO2

    for cell in experiment_df.Cell_type.unique():
        cell_df = experiment_df.query("Cell_type==@cell")
        modality_1_perturbation = "compound"
        modality_1_experiments_df = cell_df.query("Perturbation==@modality_1_perturbation")

        for modality_1_timepoint in modality_1_experiments_df.Time.unique():
            modality_1_timepoint_df = modality_1_experiments_df.query(
                "Time==@modality_1_timepoint"
            )
            modality_1_df = process_modality(
                modality_1_timepoint_df,
                modality_1_perturbation,
                modality_2=False,
                feature_type=args.feature_type,
                pca_kernel=control_pca,
            )
            description = f"{modality_1_perturbation}_{cell}_{utils.time_point(modality_1_perturbation, modality_1_timepoint)}"
            # Calculate replicability mAP
            print(f"Computing {description} replicability")

            modality_1_df["Metadata_negcon"] = np.where(
                modality_1_df["Metadata_control_type"] == "negcon", 1, 0
            )  # Create dummy column

            pos_sameby = ["Metadata_broad_sample"]
            pos_diffby = []
            neg_sameby = ["Metadata_Plate"]
            neg_diffby = ["Metadata_negcon"]

            metadata_df = utils.get_metadata(modality_1_df)
            feature_df = utils.get_featuredata(modality_1_df)
            feature_values = (
                control_pca.transform(feature_df.values)
                if args.feature_type != "emb"
                else np.stack(modality_1_df["embeddings"].values)
            )

            result = utils.run_pipeline(
                metadata_df,
                feature_values,
                pos_sameby,
                pos_diffby,
                neg_sameby,
                neg_diffby,
                anti_match=False,
                batch_size=batch_size,
                null_size=null_size,
            )

            result = result.query("Metadata_negcon==0").reset_index(drop=True)

            replicability_map_df, replicability_fr_df = utils.create_replicability_df(
                replicability_map_df,
                replicability_fr_df,
                result,
                pos_sameby,
                0.05,
                modality_1_perturbation,
                cell,
                modality_1_timepoint,
            )

            # Remove DMSO wells
            modality_1_df = utils.remove_negcon_and_empty_wells(modality_1_df)

            # Create consensus profiles
            modality_1_consensus_df = utils.consensus(
                modality_1_df, replicate_feature, args.feature_type
            )

            # Filter out non-replicable compounds
            replicable_compounds = list(
                replicability_map_df[
                    (replicability_map_df.Description == description)
                    & (replicability_map_df.above_q_threshold)
                ][replicate_feature]
            )
            modality_1_consensus_df = modality_1_consensus_df.query(
                "Metadata_broad_sample==@replicable_compounds"
            ).reset_index(drop=True)

            # Adding additional gene annotation metadata
            modality_1_consensus_df = (
                modality_1_consensus_df.merge(
                    target1_metadata, on="Metadata_broad_sample", how="left"
                )
                .assign(
                    Metadata_matching_target=lambda x: x.Metadata_target_list.str.split("|")
                )
                .drop(["Metadata_target_list"], axis=1)
            )

            # Calculate compound-compound matching
            print(f"Computing {description} matching")

            pos_sameby = ["Metadata_matching_target"]
            pos_diffby = []
            neg_sameby = []
            neg_diffby = ["Metadata_matching_target"]

            metadata_df = utils.get_metadata(modality_1_consensus_df)
            feature_df = utils.get_featuredata(modality_1_consensus_df)

            feature_values = (
                control_pca.transform(feature_df.values)
                if args.feature_type != "emb"
                else np.stack(modality_1_consensus_df["embeddings"].values)
            )

            result = utils.run_pipeline(
                metadata_df,
                feature_values,
                pos_sameby,
                pos_diffby,
                neg_sameby,
                neg_diffby,
                anti_match=True,
                batch_size=batch_size,
                null_size=null_size,
                multilabel_col="Metadata_matching_target",
            )

            matching_map_df, matching_fr_df = utils.create_matching_df(
                matching_map_df,
                matching_fr_df,
                result,
                pos_sameby,
                0.05,
                modality_1_perturbation,
                cell,
                modality_1_timepoint,
            )

            all_modality_2_experiments_df = cell_df.query(
                "Perturbation!=@modality_1_perturbation"
            )

            for (
                modality_2_perturbation
            ) in all_modality_2_experiments_df.Perturbation.unique():
                modality_2_experiments_df = all_modality_2_experiments_df.query(
                    "Perturbation==@modality_2_perturbation"
                )
                for modality_2_timepoint in modality_2_experiments_df.Time.unique():
                    modality_2_timepoint_df = modality_2_experiments_df.query(
                        "Time==@modality_2_timepoint"
                    )
                    modality_2_df = process_modality(
                        modality_2_timepoint_df,
                        modality_2_perturbation,
                        modality_2=True,
                        feature_type=args.feature_type,
                        pca_kernel=control_pca,
                    )

                    # Description
                    description = f"{modality_2_perturbation}_{cell}_{utils.time_point(modality_2_perturbation, modality_2_timepoint)}"

                    # Calculate replicability mAP

                    if not replicability_map_df.Description.str.contains(description).any():
                        print(f"Computing {description} replicability")

                        modality_2_df["Metadata_negcon"] = np.where(
                            modality_2_df["Metadata_control_type"] == "negcon", 1, 0
                        )  # Create dummy column

                        pos_sameby = ["Metadata_broad_sample"]
                        pos_diffby = []
                        neg_sameby = ["Metadata_Plate"]
                        neg_diffby = ["Metadata_negcon"]

                        metadata_df = utils.get_metadata(modality_2_df)
                        feature_df = utils.get_featuredata(modality_2_df)
                        feature_values = (
                            control_pca.transform(feature_df.values)
                            if args.feature_type != "emb"
                            else np.stack(modality_2_df["embeddings"].values)
                        )

                        result = utils.run_pipeline(
                            metadata_df,
                            feature_values,
                            pos_sameby,
                            pos_diffby,
                            neg_sameby,
                            neg_diffby,
                            anti_match=False,
                            batch_size=batch_size,
                            null_size=null_size,
                        )

                        result = result.query("Metadata_negcon==0").reset_index(drop=True)

                        (
                            replicability_map_df,
                            replicability_fr_df,
                        ) = utils.create_replicability_df(
                            replicability_map_df,
                            replicability_fr_df,
                            result,
                            pos_sameby,
                            0.05,
                            modality_2_perturbation,
                            cell,
                            modality_2_timepoint,
                        )

                    # Remove negcon wells
                    modality_2_df = utils.remove_negcon_and_empty_wells(modality_2_df)

                    # Create consensus profiles
                    modality_2_consensus_df = utils.consensus(
                        modality_2_df, "Metadata_broad_sample", args.feature_type
                    )
                    # Filter out non-replicable genes
                    replicable_genes = list(
                        replicability_map_df[
                            (
                                replicability_map_df.Description
                                == f"{modality_2_perturbation}_{cell}_{utils.time_point(modality_2_perturbation, modality_2_timepoint)}"
                            )
                            # & (replicability_map_df.above_q_threshold)
                        ][replicate_feature]
                    )
                    modality_2_consensus_df = modality_2_consensus_df.query(
                        "Metadata_broad_sample==@replicable_genes"
                    ).reset_index(drop=True)

                    # Filter out reagents without a sister guide
                    genes_without_sister = (
                        modality_2_consensus_df.Metadata_gene.value_counts()
                        .reset_index()
                        .query("count==1")
                        .Metadata_gene.to_list()
                    )

                    modality_2_consensus_for_matching_df = modality_2_consensus_df.query(
                        "Metadata_gene!=@genes_without_sister"
                    ).reset_index(drop=True)

                    # Calculate cripsr-crispr matching
                    if modality_2_perturbation == "crispr":
                        if not matching_map_df.Description.str.contains(description).any():
                            print(f"Computing {description} matching")

                            pos_sameby = ["Metadata_matching_target"]
                            pos_diffby = []
                            neg_sameby = []
                            neg_diffby = ["Metadata_matching_target"]

                            metadata_df = utils.get_metadata(
                                modality_2_consensus_for_matching_df
                            )
                            feature_df = utils.get_featuredata(
                                modality_2_consensus_for_matching_df
                            )
                            feature_values = (
                                control_pca.transform(feature_df.values)
                                if args.feature_type != "emb"
                                else np.stack(
                                    modality_2_consensus_for_matching_df[
                                        "embeddings"
                                    ].values
                                )
                            )

                            result = utils.run_pipeline(
                                metadata_df,
                                feature_values,
                                pos_sameby,
                                pos_diffby,
                                neg_sameby,
                                neg_diffby,
                                anti_match=False,
                                batch_size=batch_size,
                                null_size=null_size,
                            )

                            matching_map_df, matching_fr_df = utils.create_matching_df(
                                matching_map_df,
                                matching_fr_df,
                                result,
                                pos_sameby,
                                0.05,
                                modality_2_perturbation,
                                cell,
                                modality_2_timepoint,
                            )

                    # Filter out genes that are not perturbed by ORFs or CRISPRs
                    perturbed_genes = list(
                        set(modality_2_consensus_df.Metadata_matching_target)
                    )

                    modality_1_filtered_genes_df = (
                        modality_1_consensus_df[
                            ["Metadata_broad_sample", "Metadata_matching_target"]
                        ]
                        .copy()
                        .explode("Metadata_matching_target")
                        .query("Metadata_matching_target==@perturbed_genes")
                        .reset_index(drop=True)
                        .groupby(["Metadata_broad_sample"])
                        .Metadata_matching_target.apply(list)
                        .reset_index()
                    )

                    modality_1_consensus_filtered_df = modality_1_consensus_df.drop(
                        columns=["Metadata_matching_target"]
                    ).merge(
                        modality_1_filtered_genes_df,
                        on="Metadata_broad_sample",
                        how="inner",
                    )

                    # Calculate gene-compound matching mAP
                    description = (
                        f"{modality_1_perturbation}_{cell}_{utils.time_point(modality_1_perturbation, modality_1_timepoint)}"
                        f"-{modality_2_perturbation}_{cell}_{utils.time_point(modality_2_perturbation, modality_2_timepoint)}"
                    )
                    print(f"Computing {description} matching")

                    modality_1_modality_2_df = utils.concat_profiles(
                        modality_1_consensus_filtered_df, modality_2_consensus_df
                    )

                    pos_sameby = ["Metadata_matching_target"]
                    pos_diffby = ["Metadata_modality"]
                    neg_sameby = []
                    neg_diffby = ["Metadata_matching_target", "Metadata_modality"]

                    metadata_df = utils.get_metadata(modality_1_modality_2_df)
                    feature_df = utils.get_featuredata(modality_1_modality_2_df)
                    feature_values = (
                        control_pca.transform(feature_df.values)
                        if args.feature_type != "emb"
                        else np.stack(modality_1_modality_2_df["embeddings"].values)
                    )

                    result = utils.run_pipeline(
                        metadata_df,
                        feature_values,
                        pos_sameby,
                        pos_diffby,
                        neg_sameby,
                        neg_diffby,
                        anti_match=True,
                        batch_size=batch_size,
                        null_size=null_size,
                        multilabel_col="Metadata_matching_target",
                    )

                    (
                        gene_compound_matching_map_df,
                        gene_compound_matching_fr_df,
                    ) = utils.create_gene_compound_matching_df(
                        gene_compound_matching_map_df,
                        gene_compound_matching_fr_df,
                        result,
                        pos_sameby,
                        0.05,
                        modality_1_perturbation,
                        modality_2_perturbation,
                        cell,
                        modality_1_timepoint,
                        modality_2_timepoint,
                    )

    print("==========replicability_fr_df==========")

    print(replicability_fr_df[["Description", "timepoint", "fr"]].to_markdown(index=False))
    print(replicability_fr_df["fr"].mean(), replicability_fr_df["fr"].std())
    print("==========matching_fr_df=========")

    print(matching_fr_df[["Description", "timepoint", "fr"]].to_markdown(index=False))
    print(matching_fr_df["fr"].mean(), matching_fr_df["fr"].std())

    print("=========gene_compound_matching_fr_df========")
    print(
        gene_compound_matching_fr_df[["Description", "Cell", "fr"]].to_markdown(index=False)
    )
    print(gene_compound_matching_fr_df["fr"].mean(),gene_compound_matching_fr_df["fr"].std() )

    return True


if __name__ == "__main__":
    """Main function"""
    args = parse_args()
    main(args)
