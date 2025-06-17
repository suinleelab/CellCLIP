"""Split JUMPCP into train and test sets"""
import os

import pandas as pd

# Paths
metadata_path = "/gscratch/aims/mingyulu/cell_painting/label_data/jumpcp/"
label_path = "/gscratch/aims/datasets/cellpainting/jumpcp/mol"

# Load metadata and filter
experiment_df = (
    pd.read_csv(
        os.path.join(metadata_path, "experiment-metadata.tsv"),
        sep="\t",
    )
    .query("Batch=='2020_11_04_CPJUMP1'")
    .query("Density==100")
    .query('Antibiotics=="absent"')
)

# Define grouping columns
group_columns = ["Cell_type", "Time", "Perturbation"]

# Load labels
label = pd.read_csv(os.path.join(label_path, "jumpcp_label.csv"), sep=",", header=0)

# Extract SAMPLE_KEY and deduplicate
label["batch"] = label["SAMPLE_KEY"].str.rsplit("-").str[0]
label["UNIQUE_SAMPLE_KEY"] = label["SAMPLE_KEY"].str.rsplit("-", n=1).str[0]

label = label.drop_duplicates(subset="UNIQUE_SAMPLE_KEY").reset_index(drop=True)

label["treatment"] = label["SAMPLE_KEY"].apply(lambda x: "-".join(x.split("-")[1:3]))


# Merge metadata with labels to align information
merged = pd.merge(
    label, experiment_df, left_on="batch", right_on="Assay_Plate_Barcode", how="left"
)


def consistent_sample(group):
    """Split data based on perturnation"""
    ids = group["treatment"].unique().tolist()
    group = group.sort_values(by="treatment")
    train_size = int(0.75 * len(ids))
    train_idx = ids[:train_size]
    test_idx = ids[train_size:]

    train_group = group.loc[group.treatment.isin(train_idx)]
    test_group = group.loc[group.treatment.isin(test_idx)]

    return train_group, test_group


# Initialize lists for train and test keys
train_keys = []
test_keys = []

# Group by and sample consistently
for _, group in merged.groupby(group_columns):
    train_group, test_group = consistent_sample(group)
    train_keys.extend(train_group["UNIQUE_SAMPLE_KEY"].tolist())
    test_keys.extend(test_group["UNIQUE_SAMPLE_KEY"].tolist())

new_label = pd.read_csv(os.path.join(label_path, "jumpcp_label.csv"), sep=",", header=0)
new_label["UNIQUE_SAMPLE_KEY"] = new_label["SAMPLE_KEY"].str.rsplit("-", n=1).str[0]
new_label["batch"] = new_label["SAMPLE_KEY"].str.rsplit("-").str[0]

# Filter the original labels for train and test
train_label = new_label[new_label.UNIQUE_SAMPLE_KEY.isin(train_keys)]
test_label = new_label[new_label.UNIQUE_SAMPLE_KEY.isin(test_keys)]

# Validate results
assert set(train_keys).isdisjoint(set(test_keys)), "Train and test keys overlap!"
# Print lengths of train and test labels
print(f"Train labels: {len(train_label)}, Test labels: {len(test_label)}")

# Save training and testing labels
train_label.to_csv(os.path.join(label_path, "jumpcp_training_label2.csv"), index=False)
test_label.to_csv(os.path.join(label_path, "jumpcp_testing_label2.csv"), index=False)
