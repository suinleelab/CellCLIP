"""Generate molecules SMILES for CP-JUMP1"""
import os

import pandas as pd

from src import constants

# Output path

output_path = (
    "/gscratch/aims/mingyulu/cell_painting/label_data/jumpcp/jumpcp_id_to_smiles.csv"
)

split_dir = os.path.join(constants.OUT_DIR, "label_data/jumpcp/")

plate_lists = [
    plate
    for plate in os.listdir(
        "/gscratch/aims/datasets/cellpainting/jumpcp/2020_11_04_CPJUMP1"
    )
    if plate.startswith("BR")
]
plate_lists.sort()

all_records = []

for plate in plate_lists:
    # Load metadata
    exp_meta_data = pd.read_csv(
        os.path.join(split_dir, "experiment-metadata.tsv"), sep="\t"
    )
    plate_exp_metadata = exp_meta_data[exp_meta_data.Assay_Plate_Barcode == plate]

    if (
        plate_exp_metadata.empty
        or plate_exp_metadata["Perturbation"].values[0] != "compound"
    ):
        continue

    # Load compound metadata and platemap
    target_metadata = pd.read_csv(
        os.path.join(
            split_dir,
            "plate_map",
            "JUMP-Target-1_compound_metadata_additional_annotations.tsv",
        ),
        sep="\t",
    )
    plate_map = pd.read_csv(
        os.path.join(split_dir, "plate_map", "JUMP-Target-1_compound_platemap.txt"),
        sep="\t",
    )

    # Merge metadata with platemap
    target_plate_metadata = pd.merge(
        target_metadata, plate_map, on="broad_sample", how="right"
    )

    # Get image filenames and build full ID
    plate_img_dir = os.path.join(
        "/gscratch/aims/datasets/cellpainting/jumpcp/2020_11_04_CPJUMP1", plate
    )
    img_files = [f for f in os.listdir(plate_img_dir) if f.endswith(".npz")]

    # Reconstruct ID from filenames (format: PLATE-WELL-ROWCOL-FIELD)
    ids = [f.replace(".npz", "") for f in img_files]
    ids = [
        plate + "-" + f.split("-")[1] + "-" + f.split("-")[0] + "-" + f.split("-")[2]
        for f in ids
    ]
    well_positions = [f.split("-")[1] for f in ids]

    id_df = pd.DataFrame({"ID": ids, "well_position": well_positions})

    # Merge with SMILES
    merged = pd.merge(
        id_df,
        target_plate_metadata[["well_position", "smiles"]],
        on="well_position",
        how="left",
    )

    all_records.append(merged[["ID", "smiles", "well_position"]])
# Combine and save
final_df = pd.concat(all_records).drop_duplicates().reset_index(drop=True)
final_df.columns = ["ID", "SMILES", "Well"]
final_df.to_csv(output_path, index=False)

print(f"Saved {len(final_df)} ID-to-SMILES mappings to {output_path}")
