"""Helper function and scripts for preprocessing molecules."""

import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from src import constants


def morgan_from_smiles(smiles, radius=3, nbits=1024, chiral=True):
    """Convert SMILE string into Morgan fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=3, nBits=nbits, useChirality=chiral
    )
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def generate_cell_caption(
    cell_type, perturbation_type, target, target_info=None, control_info=None
):
    """Generating captions for cell painting images."""

    if perturbation_type == "compound":

        prompt = (
            f"{cell_type} cells treated with {perturbation_type}:"
            f" {target}, SMILES: {target_info}"
        )

        if "DMSO" in str(target):
            prompt = (
                f"{cell_type} cells treated with {perturbation_type}:"
                f" control, {target}, SMILES: {target_info}"
            )
        elif "control" in str(target):
            prompt = f"{cell_type} cells treated with {perturbation_type}:" f" control"

        if len(prompt) > 512:
            prompt = f"{cell_type} cells treated with SMILES: {target_info}"

            if len(prompt) > 512:
                prompt = f"{cell_type} cells treated with {perturbation_type}, {target}"

    elif perturbation_type == "crispr":

        if "control" in str(control_info):
            prompt = (
                f"{cell_type} cells treated with {perturbation_type} "
                f"sequence: {target_info}, {control_info}"
            )
        elif (
            "nan" in str(target_info)
            and "nan" in str(target_info)
            or "control" in str(target)
        ):
            prompt = f"{cell_type} cells treated with control, no treatment."
        elif "guide" in str(target_info):
            prompt = (
                f"{cell_type} cells treated with {perturbation_type} "
                f",targeting genes: {target}"
            )
        else:
            prompt = (
                f"{cell_type} cells treated with {perturbation_type} "
                f"sequence: {target_info}, targeting genes: {target}"
            )

    elif perturbation_type == "orf":

        if target_info and "control" in str(target_info):
            prompt = (
                f"{cell_type} cells treated with {perturbation_type}"
                f" {target_info} {control_info}, targeting genes: {target}."
            )
        elif "nan" in str(target):
            prompt = f"{cell_type} cells treated with control, no treatment."
        else:
            prompt = (
                f"{cell_type} cells treated with {perturbation_type}"
                f", targeting genes: {target}"
            )

    return prompt


# def generate_cell_caption_from_gpt(
#     smile, drug_name, cell_type, model="text-embedding-3-small"
# ):
#     # client = OpenAI(api_key=constants.OPENAI_API_KEY)

#     """Generating emebdding from openAI API."""
#     prompt = (
#         f"A cell painting image of {cell_type} cells "
#         f"  treated with the drug '{drug_name}',"
#         f"represented by the SMILES notation '{smile}'.. "
#         "The image features five fluorescence channels, "
#         "Alexa 568, F-actin, Golgi, plasma membrane"
#         "Hoechst 33342, Nucleus/DNA"
#         "Alexa 488, Endoplasmic Reticulum"
#         "Alexa 647, Mitochondria"
#         "Alexa 488 long, Nucleoli/RNA"
#     )

#     prompt = prompt.replace("\n", " ")
#     response = client.embeddings.create(input=[prompt], model=model)
#     return response.data[0].embedding


def parallelize_data(func, data, n_workers):
    """Helper function for parallelization with improved data handling."""
    with Pool(n_workers) as pool:
        results = pool.starmap(func, data)
    return results


def parallelize(func, iterable, n_workers, **kwargs):
    """Helper function for parallelization with improved data handling."""

    f = partial(func, **kwargs)
    if n_workers > 1:
        with Pool(n_workers) as p:
            results = p.map(f, iterable)
    else:
        results = list(map(f, iterable))
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Contrastive Learning.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bray2017", "jumpcp", "rxrx3-core"],
        help="Dataset source of cellpainting images.",
        required=True,
    )
    parser.add_argument("--caption_type", type=str, help="Types of contrastive pair.")
    parser.add_argument("--output_file", type=str, help="Output file name.")
    parser.add_argument("--img_dir", type=str, help="Image dir")

    return parser.parse_args()


def main(args):
    """Main function for preprocessing contrastive pairs."""

    n_cpus = 8

    if args.dataset == "jumpcp":
        split_dir = os.path.join(constants.OUT_DIR, "label_data/jumpcp/")
        plate_lists = [
            plate for plate in os.listdir(args.img_dir) if plate.startswith("BR")
        ]

        plate_lists.sort()

        if args.caption_type == "caption":
            final_df = pd.DataFrame(columns=["ID", "prompt"])
            final_df.set_index("ID", inplace=True)
        else:
            final_df = pd.DataFrame(columns=["ID"])
            final_df.set_index("ID", inplace=True)

        for plate in plate_lists:

            exp_meta_data = pd.read_csv(
                os.path.join(split_dir, "experiment-metadata.tsv"), sep="\t"
            )

            plate_exp_metadata = exp_meta_data[exp_meta_data.Assay_Plate_Barcode == plate]
            perturbation = plate_exp_metadata["Perturbation"].values[0]
            cell_type = plate_exp_metadata["Cell_type"].values[0]

            if perturbation == "compound":
                target_metadata = pd.read_csv(
                    os.path.join(
                        split_dir,
                        "plate_map",
                        "JUMP-Target-1_compound_metadata_additional_annotations.tsv",
                    ),
                    sep="\t",
                )
                plate_map = pd.read_csv(
                    os.path.join(
                        split_dir, "plate_map/JUMP-Target-1_compound_platemap.txt"
                    ),
                    sep="\t",
                )

            elif perturbation == "crispr":
                target_metadata = pd.read_csv(
                    os.path.join(split_dir, "plate_map/JUMP-Target-1_crispr_metadata.tsv"),
                    sep="\t",
                )
                plate_map = pd.read_csv(
                    os.path.join(split_dir, "plate_map/JUMP-Target-1_crispr_platemap.txt"),
                    sep="\t",
                )
            elif perturbation == "orf":
                target_metadata = pd.read_csv(
                    os.path.join(split_dir, "plate_map/JUMP-Target-1_orf_metadata.tsv"),
                    sep="\t",
                )
                plate_map = pd.read_csv(
                    os.path.join(split_dir, "plate_map/JUMP-Target-1_orf_platemap.txt"),
                    sep="\t",
                )
            else:
                raise ValueError(f"Unknown perturbation type : {perturbation}")

            target_plate_metadata = pd.merge(
                target_metadata,
                plate_map,
                left_on="broad_sample",
                right_on="broad_sample",
                how="right",
            )
            plate_img_dir = os.path.join(args.img_dir, plate)

            img_files = os.listdir(plate_img_dir)

            file_names = [f.replace(".npz", "") for f in img_files]
            file_ids = pd.DataFrame(
                {
                    "ID": [
                        plate
                        + "-"
                        + f.split("-")[1]
                        + "-"
                        + f.split("-")[0]
                        + "-"
                        + f.split("-")[2]
                        for f in file_names
                    ],
                    "well_position": [
                        file.split("-")[-2] for file in img_files if file.endswith(".npz")
                    ],
                }
            )
            csv = pd.merge(
                file_ids,
                target_plate_metadata,
                left_on="well_position",
                right_on="well_position",
                how="left",
            )

            csv["cell_type"] = cell_type
            csv["perturbation"] = perturbation

            print("Generating cell painting prompt...")

            if perturbation == "compound":
                data_pairs = list(
                    zip(
                        csv["cell_type"],
                        csv["perturbation"],
                        csv["pert_iname"],
                        csv["smiles"],
                    )
                )
            elif perturbation == "crispr":
                csv["control_info"] = csv["pert_type"] + " " + csv["control_type"]

                data_pairs = list(
                    zip(
                        csv["cell_type"],
                        csv["perturbation"],
                        csv["gene"],
                        csv["target_sequence"],
                        csv["control_info"],
                    )
                )
            else:
                data_pairs = list(
                    zip(
                        csv["cell_type"],
                        csv["perturbation"],
                        csv["gene"],
                        csv["pert_type"],
                        csv["control_type"],
                    )
                )

            ids = file_ids["ID"]

            if args.caption_type == "morgan" and perturbation == "compound":
                print(f"Generating morgan finger print for plate {plate}")

                smiles = [data[3] for data in data_pairs]
                fps = parallelize(morgan_from_smiles, smiles, n_cpus)
                columns = [str(i) for i in range(fps[0].shape[0])]

                df = pd.DataFrame(fps, index=ids, columns=columns)
                final_df = pd.concat([final_df, df])

            elif args.caption_type == "caption":
                fps = parallelize_data(generate_cell_caption, data_pairs, n_cpus)

                df = pd.DataFrame(fps, index=ids, columns=["prompt"])
                max_length = df["prompt"].str.len().max()

                print("Maximum length of 'prompt' column:", max_length)

                final_df = pd.concat([final_df, df])

        # final_df["SAMPLE_KEY"] = final_df.index

        if args.caption_type == "morgan":
            final_df.to_hdf(
                "/gscratch/aims/datasets/cellpainting/jumpcp/mol/jumpcp_morgan_1024.hd5",
                key="df",
                mode="w",
            )
        else:
            final_df.to_csv(
                "/gscratch/aims/datasets/cellpainting/jumpcp/mol/jumpcp_cell_caption.csv"
            )

    elif args.dataset == "rxrx3-core":

        final_df = pd.DataFrame(columns=["ID", "prompt"])
        final_df.set_index("ID", inplace=True)

        file_path_metadata = hf_hub_download(
            "recursionpharma/rxrx3-core",
            filename="metadata_rxrx3_core.csv",
            repo_type="dataset",
        )
        csv = pd.read_csv(file_path_metadata)

        pert_types = ["COMPOUND", "CRISPR"]

        for perturbation in pert_types:
            metadata = csv[csv.perturbation_type == perturbation]

            if perturbation == "CRISPR":
                data_pairs = list(
                    zip(
                        metadata["cell_type"],
                        metadata["perturbation_type"].str.lower(),
                        metadata["gene"],
                        metadata["treatment"],
                    )
                )
            else:
                metadata["SMILES"] = metadata["SMILES"].str.split(" ").str[0]

                data_pairs = list(
                    zip(
                        metadata["cell_type"],
                        metadata["perturbation_type"].str.lower(),
                        metadata["treatment"] + metadata["concentration"],
                        metadata["SMILES"],
                    )
                )

            fps = parallelize_data(generate_cell_caption, data_pairs, n_cpus)
            ids = metadata["well_id"]
            df = pd.DataFrame(fps, index=ids, columns=["prompt"])
            final_df = pd.concat([final_df, df])

        max_length = final_df["prompt"].str.len().max()

        print("Maximum length of 'prompt' column:", max_length)

        outfile_path = os.path.join(
            "/gscratch/aims/datasets/cellpainting/"
            "rxrx3-core/mol/rxrx3-core_cell_caption.csv"
        )
        final_df.to_csv(outfile_path)

    else:

        split_index_name = "datasplit1"
        files = ["train", "val", "test"]
        split_dir = os.path.join(constants.OUT_DIR, "label_data", "bray_2017", "split_data")

        dfs = []

        for file in files:
            split_path = os.path.join(split_dir, f"{split_index_name}-{file}.csv")
            df = pd.read_csv(split_path)
            dfs.append(df)

        # Concatenate the DataFrames together
        csv = pd.concat(dfs, ignore_index=False)

        csv["ID"] = csv.apply(
            lambda row: "-".join(
                [str(row["PLATE_ID"]), str(row["WELL_POSITION"]), str(row["SITE"])]
            ),
            axis=1,
        )

        ids = csv["ID"]
        smiles = csv["SMILES"]
        csv["cell_type"] = "U2OS"

        if args.caption_type == "morgan":
            print("Generating morgan finger print...")
            outfile_hdf = os.path.join(split_dir, args.output_file)

            fps = parallelize(morgan_from_smiles, smiles, n_cpus)
            columns = [str(i) for i in range(fps[0].shape[0])]

            df = pd.DataFrame(fps, index=ids, columns=columns)

            df.to_hdf(outfile_hdf, key="df", mode="w")

        elif args.caption_type == "cell_captions":
            print("Generating cell painting prompt...")

            outfile_path = os.path.join(split_dir, args.output_file)

            csv["perturbation"] = "compound"
            data_pairs = list(
                zip(csv["cell_type"], csv["perturbation"], csv["CPD_NAME"], csv["SMILES"])
            )
            fps = parallelize_data(generate_cell_caption, data_pairs, n_cpus)

            df = pd.DataFrame(fps, index=ids, columns=["prompt"])
            max_length = df["prompt"].str.len().max()

            print("Maximum length of 'prompt' column:", max_length)
            df.to_csv(outfile_path)

        # elif args.caption_type == "cell_caption_gpt":
        #     print("Generating GPT embeddings...")
        #     outfile_path = os.path.join(split_dir, args.output_file)
        #     data_pairs = list(zip(csv["SMILES"], csv["CPD_NAME"], csv["cell_type"]))

        #     fps = parallelize_data(generate_cell_caption_from_gpt, data_pairs, n_cpus)
        #     df = pd.DataFrame({"embedding": fps}, index=ids)
        #     df.to_csv(outfile_path)

    print("Done!!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
